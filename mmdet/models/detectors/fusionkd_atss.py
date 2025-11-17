# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


from mmdet.registry import MODELS
from mmdet.utils import (InstanceList, OptInstanceList, OptConfigType, reduce_mean)
from .Fusionkd_single_stage import FusionKDSingleStageDetector

from ..utils import permute_and_flatten
from ..utils import images_to_levels, multi_apply, unpack_gt_instances
import re
from typing import Optional, Tuple, Union
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.transform import resize
import torch

def visualize_channel_avg(x):
    """
    可视化特征金字塔中各层的平均通道特征

    参数:
    x (tuple): 包含多个特征层的元组, 每个层形状为 [1, 256, H, W]
    """
    # 检查输入是否为元组
    # 使用方法:
    # x = self.extract_feat(batch_inputs)  # 获取特征元组
    # visualize_channel_avg(x)
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    #
    # if not isinstance(x, tuple):
    #     print("Error: Input should be a tuple of feature maps")
    #     return

    num_levels = len(x)

    # 创建子图
    fig, axes = plt.subplots(1, num_levels, figsize=(18, 4))
    if num_levels == 1:
        axes = [axes]  # 确保单层时axes仍是列表

    # 确定最大原始尺寸用于归一化
    max_size = max([feat.shape[2] for feat in x])

    for i, feat in enumerate(x):
        # 计算通道平均值 [1, H, W]
        channel_avg = torch.mean(feat.detach(), dim=1)[0]  # [0] 移除批次维度 => [H, W]

        # 归一化到 [0, 1]
        min_val = torch.min(channel_avg)
        max_val = torch.max(channel_avg)
        normalized = (channel_avg - min_val) / (max_val - min_val + 1e-8)

        # 上采样到统一尺寸便于比较
        upsampled = F.interpolate(
            normalized[None, None, ...],  # 添加批次和通道维度
            size=(max_size, max_size),
            mode='bilinear',
            align_corners=False
        )[0, 0]  # 移除添加的维度


        # 可视化
        ax = axes[i]
        im = ax.imshow(upsampled.cpu().numpy(), cmap='viridis')
        fig.colorbar(im, ax=ax, shrink=0.7)  # 添加颜色条

        # 添加原始分辨率信息
        orig_shape = feat.shape
        title = f'Level {i}\n({orig_shape[2]}×{orig_shape[3]})'
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.suptitle('Average Channel Features Across Levels', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()


@MODELS.register_module()
class FusionKDATSS(FusionKDSingleStageDetector):

    def __init__(self,
                 kd_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(kd_cfg=kd_cfg,**kwargs)
        # 添加动态蒸馏权重控制
        self.kd_weight_scheduler = {
            'cls': 1.0, 'reg': 1.0, 'center': 1.0, 'feat': 0.5
        }
        self.current_epoch = 0
        self.total_epochs = kwargs.get('train_cfg', {}).get('max_epochs', 30)

        # 梯度协调参数
        self.gradient_stats = {
            'det_loss': 0.0,
            'kd_loss': 0.0,
            'ratio_threshold': 0.6  # KD损失不应超过总损失的60%
        }

        # 性能监控
        self.best_map = 0.0
        self.performance_history = []

        self.loss_center_kd = None
        if kd_cfg.get('loss_center_kd', None):
            self.loss_center_kd = MODELS.build(kd_cfg['loss_center_kd'])
        #add by jojo
        # 文本适配模块
        self.text_dim=768
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True)
        )
        #
        # 跨模态注意力
        self.cross_attn = CrossModalAttention(
            v_dim=256,  # 学生特征维度
            l_dim=256,  # 投影后文本维度 512-256
            num_heads=8)

        # 文本增强的FPN
        self.text_aware_fpn = nn.ModuleList([
            TextAwareFPNLayer(256)
            for _ in range(5)
        ])

        #visual enhanced text
        self.text_enhancer = VisualEnhancedText(text_dim=768)
        #add end

    def update_kd_weights(self, epoch, strategy='adaptive'):
        """根据训练进度动态调整蒸馏权重"""
        self.current_epoch = epoch
        progress = epoch / self.total_epochs

        if strategy == 'adaptive':
            # 自适应衰减策略
            if epoch < 10:
                # 前期：强蒸馏，快速学习教师知识
                decay_factor = 1.0 - progress * 0.2
            elif epoch < 20:
                # 中期：平衡蒸馏，防止过拟合
                decay_factor = 0.8 - (progress - 0.33) * 0.4
            else:
                # 后期：弱蒸馏，主要优化检测任务
                decay_factor = max(0.2, 0.4 - (progress - 0.66) * 0.5)

        elif strategy == 'cosine':
            # 余弦衰减
            import math
            decay_factor = 0.5 * (1 + math.cos(math.pi * progress))

        elif strategy == 'performance_aware':
            # 性能感知衰减
            if len(self.performance_history) > 3:
                recent_improvement = self.performance_history[-1] - self.performance_history[-3]
                if recent_improvement < 0.01:  # 性能提升缓慢
                    decay_factor = max(0.2, decay_factor * 0.8)
                else:
                    decay_factor = min(1.0, decay_factor * 1.1)

        self.kd_weight_scheduler = {
            'cls': decay_factor,
            'reg': decay_factor * 0.8,  # 回归任务权重稍低
            'center': decay_factor,
            'feat': 0.3 * decay_factor  # 特征蒸馏权重最低
        }

        return self.kd_weight_scheduler

    def check_gradient_balance(self, logger):
        """检查梯度平衡，防止蒸馏损失主导"""
        det_loss = self.gradient_stats['det_loss']
        kd_loss = self.gradient_stats['kd_loss']
        total_loss = det_loss + kd_loss + 1e-8

        kd_ratio = kd_loss / total_loss

        if kd_ratio > self.gradient_stats['ratio_threshold']:
            # KD损失过大，需要调整
            adjustment = self.gradient_stats['ratio_threshold'] / kd_ratio
            for key in self.kd_weight_scheduler:
                self.kd_weight_scheduler[key] *= adjustment

            if logger and self.current_epoch % 10 == 0:
                logger.info(f'Gradient balance adjusted: KD ratio {kd_ratio:.3f}')

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        #student branch
        stu_x = self.extract_feat(batch_inputs)
        # visualize_channel_avg(stu_x)
        # for i in range(len(stu_x)):
            # print("stu:",stu_x[i].shape)
            # stu: torch.Size([2, 256, 100, 100])
            # stu: torch.Size([2, 256, 50, 50])
            # stu: torch.Size([2, 256, 25, 25])
            # stu: torch.Size([2, 256, 13, 13])
            # stu: torch.Size([2, 256, 7, 7])
        # stu_cls_scores, stu_bbox_preds, stu_centernesses, stu_cls_hold, stu_reg_hold = \
        #     multi_apply(self.forward_hkd_single,
        #                 stu_x,
        #                 self.bbox_head.scales,
        #                 module=self)
        #---------------------------------------------------


        #teacher branch--------------------------------------
        tea_x = self.teacher.extract_feat(batch_inputs)
        # visualize_channel_avg( tea_x)
        # for i in range(len( tea_x )):
        #     print( "tea:",tea_x [i].shape)
        #     tea: torch.Size([2, 256, 100, 100])
        #     tea: torch.Size([2, 256, 50, 50])
        #     tea: torch.Size([2, 256, 25, 25])
        #     tea: torch.Size([2, 256, 13, 13])
        #     tea: torch.Size([2, 256, 7, 7])

        tea_cls_scores, tea_bbox_preds, tea_centernesses, tea_cls_hold, tea_reg_hold = \
            self.forward_crosskd_single_tea(
                        tea_x,
                        batch_inputs,
                        batch_data_samples,
                        module=self.teacher)
        # for i in range(len(tea_cls_scores)):
        #     print("tea_cls_scores:", tea_cls_scores[i].shape)#[1,10000,256]
        #     print("tea_bbox_preds:", tea_bbox_preds[i].shape)#[1,4,100/50/25/13/7,100/50/25/13/7]
        #     print("tea_centernesses:", tea_centernesses[i].shape)#[1,1,100/50/25/13/7,100/50/25/13/7]


        #--------------------------------------------------------
        # 执行可视化
        # 3. 执行可视化
        # 执行可视化
        # 执行可视化
        # visualize_features(
        #     tea_cls_scores,
        #     tea_bbox_preds,
        #     tea_centernesses,
        #     mode='max',  # 可选 'pca' 或 'max'
        #     target_size=300,  # 统一尺寸
        #     save_path='./feature_visualization_tea'  # 保存路径
        # )
        # #----------------------------------------------



        text_embeds=tea_cls_hold## [2,256,768]
        # 文本投影
        proj_text = self.text_proj(text_embeds)# [2,256,512]
        # 文本增强的特征金字塔
        enhanced_feats = []
        for i, feat in enumerate( stu_x ):
            enhanced = self.text_aware_fpn[i](feat, proj_text)
            enhanced_feats.append(enhanced)
            # print("enhanced:",enhanced.shape) #same
        # visualize_channel_avg(enhanced_feats)
        # print("visualize_channel_avg(enhanced_feats) done !")

        # 跨模态注意力融合
        fused_feats = [self.cross_attn(f, proj_text) for f in enhanced_feats]
        # visualize_channel_avg(fused_feats)
        # print("visualize_channel_avg(fused_feats) done !")


        # for i in range(len(fused_feats)):
        #     print("fused_feats",fused_feats[i].shape)#same
        #     # torch.Size([1, 256, 100, 100])
        #     # torch.Size([1, 256, 50, 50])
        #     # torch.Size([1, 256, 25, 25])
        #     # torch.Size([1, 256, 13, 13])
        #     # torch.Size([1, 256, 7, 7])


        # stu_cls_scores, stu_bbox_preds, stu_centernesses, stu_cls_hold, stu_reg_hold = \
        #     multi_apply(self.forward_hkd_single,
        #                 stu_x,
        #                 self.bbox_head.scales,#ori
        #                 module=self)#result-74.10

        # 分析特征空间


        stu_cls_scores, stu_bbox_preds, stu_centernesses, _, _ = \
            multi_apply(self.forward_hkd_single,
                        stu_x,
                        self.bbox_head.scales,  # ori
                        module=self)  # result-74.10

        # 执行可视化（无边界框）
        # 执行可视化
        # 3. 执行可视化
        # 执行可视化
        # 执行可视化
        # visualize_features(
        #     stu_cls_scores,
        #     stu_bbox_preds,
        #     stu_centernesses,
        #     mode='max',  # 可选 'pca' 或 'max'
        #     target_size=300,  # 统一尺寸
        #     save_path='./feature_visualization_stu'  # 保存路径
        # )

        # for i in range(len(stu_cls_scores)):
        #     print("fused_feats",stu_cls_scores[i].shape) # torch.Size([1, 80, 100, 100])
        #     print("fused_feats",  stu_bbox_preds[i].shape)  # torch.Size([1, 4, 100, 100])
        #     print("stu_centernesses:",stu_centernesses[i].shape) #stu_centernesses: torch.Size([1, 1, 100, 100])

        # for i in range(len(tea_cls_scores)):
        #     print("tea_cls_scores:", tea_cls_scores[i].shape)#[1,10000,256]
        #     print("tea_bbox_preds:", tea_bbox_preds[i].shape)#[1,4,100/50/25/13/7,100/50/25/13/7]
        #     print("tea_centernesses:", tea_centernesses[i].shape)#[1,1,100/50/25/13/7,100/50/25/13/7]


        #-------------------------------------------------------
        enhanced_text = []
        for i, feat in enumerate(stu_x):
            enhanced = self.text_enhancer(text_embeds,feat )
            enhanced_text.append(enhanced)

        reused_cls_feat=enhanced_text
        reused_reg_feat=fused_feats
        reused_cls_scores, reused_bbox_preds, reused_centernesses = \
            self.reuse_teacher_head1(reused_reg_feat,
                                     reused_cls_feat
                                     )
        # for i in range(len(reused_cls_scores)):
        #     print("reused_cls_scores:", reused_cls_scores[i].shape)  # [1,10000,256]
        #     print("reused_bbox_preds:", reused_bbox_preds[i].shape)  # [1,4,100/50/25/13/7,100/50/25/13/7]
        #     print("reused_centernesses:", reused_centernesses[i].shape)  # [1,1,100/50/25/13/7,100/50/25/13/7]
        #     # 执行可视化
        # visualize_features(
        #     reused_cls_scores,
        #     reused_bbox_preds,
        #     reused_centernesses,
        #     mode='max',  # 可选 'pca' 或 'max'
        #     target_size=300,  # 统一尺寸
        #     save_path='./feature_visualization_reused'  # 保存路径
        # )
        # #----------------------
        # analyze_feature_spaces_pca_all_levels(stu_x, tea_x, fused_feats, levels=None)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        losses = self.loss_by_feat(
                                   tea_cls_scores,
                                   tea_bbox_preds,
                                   tea_centernesses,
                                   tea_x,
                                   tea_reg_hold, #add
                                   stu_cls_scores,
                                   stu_bbox_preds,
                                   stu_centernesses,
                                   stu_x,
                                   fused_feats,
                                   text_embeds, #add
                                    proj_text,
                                    reused_cls_scores,
                                    reused_bbox_preds,
                                    reused_centernesses,
                                   batch_gt_instances,
                                   batch_img_metas,
                                   batch_gt_instances_ignore)
        # print("losses:",losses)
        return losses
    #
    def forward_hkd_single(self, x, scale, module):
        cls_feat, reg_feat = x, x
        cls_feat_hold, reg_feat_hold = x, x
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                cls_feat_hold = cls_feat

            cls_feat = cls_conv.activate(cls_feat)

        for i, reg_conv in enumerate(module.bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                reg_feat_hold = reg_feat
            reg_feat = reg_conv.activate(reg_feat)
        cls_score = module.bbox_head.atss_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.atss_reg(reg_feat)).float()#atss head
        centerness = module.bbox_head.atss_centerness(reg_feat)
      #
        for i in range(len(cls_score)):
            assert torch.isfinite(cls_score[i]).all(), "s-logits-"
            assert torch.isfinite(bbox_pred[i]).all(), "s-reg-"
            assert torch.isfinite(centerness[i]).all(), "s-center-"
        return cls_score, bbox_pred, centerness, cls_feat_hold, reg_feat_hold


    def forward_crosskd_single_tea(self, visual_feats, batch_inputs,batch_data_samples,module):
        # add by jojo
        text_prompts = ['crazing. inclusion. patches. pitted surface. rolled-in scale. scratches']
        # text_prompts = [
        #     data_samples.text for data_samples in batch_data_samples
        # ]
        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]
        # print(" gt_labels:", gt_labels)

        new_text_prompts = []
        positive_maps = []
        if len(set(text_prompts)) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            tokenized, caption_string, tokens_positive, _ = \
                module.get_tokens_and_prompts(
                    text_prompts[0], True)
            new_text_prompts = [caption_string] * len(batch_inputs)
            # print("new_text_prompts", new_text_prompts)
            for gt_label in gt_labels:
                new_tokens_positive = [
                    tokens_positive[label] for label in gt_label
                ]
                _, positive_map = module.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
        else:
            for text_prompt, gt_label in zip(text_prompts, gt_labels):
                tokenized, caption_string, tokens_positive, _ = \
                    module.get_tokens_and_prompts(
                        text_prompt, True)
                new_tokens_positive = [
                    tokens_positive[label] for label in gt_label
                ]
                _, positive_map =module.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
                new_text_prompts.append(caption_string)

        for i, data_samples in enumerate(batch_data_samples):
            # .bool().float() is very important
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            data_samples.gt_instances.positive_maps = positive_map

        language_feats  = module.language_model(new_text_prompts)

        feat_inputs = {"visual":  visual_feats ,
                       "lang": language_feats}
        self.text_masks = language_feats['masks']
        #--------此处无问题
        bbox_pred, centerness, cls_score = module.bbox_head.head(visual_feats,
                                                       language_feats)

        for i in range(len(cls_score)):
            assert torch.isfinite(cls_score[i]).all(), "t-cls-logits"
            assert torch.isfinite(bbox_pred[i]).all(), "t-reg_"
            assert torch.isfinite(centerness[i]).all(), "t-center-"


        #-------done!
        if module.bbox_head.head.early_fuse:
            embedding = module.bbox_head.head.dyhead_tower(feat_inputs)['lang']['hidden']  # text-visual pairs
            # print("early fuse!") #enter this!
        else:
            embedding = language_feats['embedded']  # text-visual pairs

        embedding = F.normalize(embedding, p=2, dim=-1)# [2,256,768]

        reg_feat_hold=[]
        for i, feature in enumerate(visual_feats):
            reg_feat0=module.bbox_head.head.dyhead_tower(feat_inputs)['visual'][i]#负责bbox_pred&cls_score_
            reg_feat_hold.append(reg_feat0)
        cls_feat_hold=embedding
                # print("cla_feat_hold:", cls_feat_hold.shape)  # [2,256,768]
            # reg_feat: torch.Size([2, 256, 100, 100])
            # reg_feat: torch.Size([2, 256, 50, 50])
            # torch.Size([2, 256, 25, 25])
            # torch.Size([2, 256, 13, 13])
            # torch.Size([2, 256, 7, 7])

        return cls_score, bbox_pred, centerness,cls_feat_hold, reg_feat_hold

    def reuse_teacher_head1(self, reused_reg_feat,reused_cls_feat
                            ):

        bbox_preds = []
        centerness = []
        cls_logits = []
        MAX_CLAMP_VALUE = 50000
        for i in range(len(reused_reg_feat)):
            visual = reused_reg_feat[i]
            embedding = reused_cls_feat[i]

            dot_product_proj_tokens = self.teacher.bbox_head.head.dot_product_projection_text(embedding /
                                                                                              2.0)
            dot_product_proj_tokens_bias = torch.matmul(
                embedding, self.teacher.bbox_head.head.bias_lang) + self.teacher.bbox_head.head.bias0

            B, C, H, W = visual.shape

            bbox_pred = self.teacher.bbox_head.head.scales[i](self.teacher.bbox_head.head.bbox_pred(visual))
            bbox_preds.append(bbox_pred)
            centerness.append(self.teacher.bbox_head.head.centerness(visual))

            dot_product_proj_queries = permute_and_flatten(
                visual, B, self.teacher.bbox_head.head.num_base_priors, C, H, W)

            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(
                1, self.teacher.bbox_head.head.num_base_priors, 1)
            dot_product_logit = (
                                        torch.matmul(dot_product_proj_queries,
                                                     dot_product_proj_tokens.transpose(-1, -2)) /
                                        self.teacher.bbox_head.head.log_scale.exp()) + bias
            dot_product_logit = torch.clamp(
                dot_product_logit, max=MAX_CLAMP_VALUE)
            dot_product_logit = torch.clamp(
                dot_product_logit, min=-MAX_CLAMP_VALUE)
            cls_logits.append(dot_product_logit)

        reused_cls_score = cls_logits
        reused_bbox_pred = bbox_preds
        reused_centerness = centerness
        return reused_cls_score, reused_bbox_pred, reused_centerness

    #-----add by jojo 20250606--------
    def dist2(self,tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
        diff = (tensor_a - tensor_b) ** 2
        #   print(diff.size())      batchsize x 1 x W x H,
        #   print(attention_mask.size()) batchsize x 1 x W x H
        diff = diff * attention_mask
        diff = diff * channel_attention_mask
        diff = torch.sum(diff) ** 0.5
        return diff
    #-------add end-------------------

    def loss_by_feat(
            self,
            tea_cls_scores: List[Tensor],
            tea_bbox_preds: List[Tensor],
            tea_centernesses: List[Tensor],
            tea_feats: List[Tensor],
            tea_reg_hold,  # add
            cls_scores: List[Tensor],#stu
            bbox_preds: List[Tensor],
            centernesses: List[Tensor],
            feats: List[Tensor],
            fused_feats,
            t_text, #add by jojo
            s_text,#add by jojo
            reused_cls_scores,
            reused_bbox_preds,
            reused_centernesses,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # 获取当前KD权重
        kd_weights = self.kd_weight_scheduler
       # # # ---------this is for atss head---------
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.bbox_head.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        # self.bbox_head.get_targets主要用于 ​​动态分配正负样本​​，解决目标检测中因锚点（Anchor）或锚点自由（Anchor - Free）方法导致的样本不均衡问题。在
        # ATSSHead中，该方法会根据自适应策略筛选高质量的正样本，而非依赖固定 IoU阈值。

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        #srudent branch
        losses_cls, losses_bbox, loss_centerness, \
            bbox_avg_factor = multi_apply(
                self.bbox_head.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                avg_factor=avg_factor)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox,loss_centerness=loss_centerness)
       #  # --------------end

        featmap_sizes1 = [featmap.size()[-2:] for featmap in reused_bbox_preds]
        assert len(featmap_sizes1) == self.teacher.bbox_head.prior_generator.num_levels

        device1 = reused_cls_scores[0].device
        anchor_list1, valid_flag_list1 = self.teacher.bbox_head.get_anchors(
            featmap_sizes1, batch_img_metas, device=device1)

        cls_reg_targets1 = self.teacher.bbox_head.get_targets(
            anchor_list1,
            valid_flag_list1,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list1, labels_list1, label_weights_list1, bbox_targets_list1,
         bbox_weights_list1, avg_factor1) = cls_reg_targets1
        avg_factor1 = reduce_mean(
            torch.tensor(avg_factor1, dtype=torch.float, device=device1)).item()

        anchors1 = torch.cat(anchor_list1, dim=1)
        labels1 = torch.cat(labels_list1, dim=1)
        label_weights1 = torch.cat(label_weights_list1, dim=1)
        bbox_targets1 = torch.cat(bbox_targets_list1, dim=1)

        tea_cls_scores = torch.cat(tea_cls_scores, dim=1)  ##[2,13343,256]
        # for i in range(len(reused_cls_scores)):
        #     print("resued_cls_scores:", reused_cls_scores[i].shape) [2,10000/2500/625/169,256]
        reused_cls_scores = torch.cat(reused_cls_scores, dim=1)  # [2,13343,256]


        # for i in range(len(anchors1)):
        #     print("anchors1:", anchors1[i].shape)#[13343,4]
        #     print("labels1:", labels1[i].shape)#[13343,256]
        #     print("label_weights1", label_weights1[i].shape)#[13343]
        #     print("bbox_targets1", bbox_targets1[i].shape)#[13343,4]

        tea_centernesses_ = []
        tea_bbox_preds_ = []
        reused_centernesses_ = []
        reused_bbox_preds_ = []
        for tea_bbox_pred, tea_centerness, reused_bbox_pred, reused_centerness in (
                zip(tea_bbox_preds, tea_centernesses, reused_bbox_preds, reused_centernesses)):
            tea_centernesses_.append(
                tea_centerness.permute(0, 2, 3,
                                       1).reshape(tea_cls_scores.size(0), -1, 1))
            tea_bbox_preds_.append(
                tea_bbox_pred.permute(0, 2, 3,
                                      1).reshape(tea_cls_scores.size(0), -1, 4))
            reused_centernesses_.append(
                reused_centerness.permute(0, 2, 3,
                                          1).reshape(reused_cls_scores.size(0), -1, 1))
            reused_bbox_preds_.append(
                reused_bbox_pred.permute(0, 2, 3,
                                         1).reshape(reused_cls_scores.size(0), -1, 4))
        tea_bbox_preds = torch.cat(tea_bbox_preds_, dim=1)
        tea_centernesses = torch.cat(tea_centernesses_, dim=1)
        reused_bbox_preds = torch.cat(reused_bbox_preds_, dim=1)  # torch.Size([2, 13343, 4])
        reused_centernesses = torch.cat(reused_centernesses_, dim=1)  # torch.Size([2, 13343, 1])

        # 2. 计算改进的KD损失
        kd_losses = self.calculate_enhanced_kd_loss(
            tea_cls_scores, tea_bbox_preds, tea_centernesses,
            reused_cls_scores, reused_bbox_preds, reused_centernesses,
            kd_weights, device
        )
        losses.update(kd_losses)

        # 3. 修复梯度统计计算
        try:
            # 正确提取损失值
            det_loss_value = 0.0
            for k in ['loss_cls', 'loss_bbox', 'loss_centerness']:
                if k in losses:
                    # 确保是张量然后获取数值
                    if isinstance(losses[k], torch.Tensor):
                        det_loss_value += losses[k].item()
                    elif isinstance(losses[k], (int, float)):
                        det_loss_value += losses[k]

            kd_loss_value = 0.0
            for k in losses:
                if 'kd' in k:
                    if isinstance(losses[k], torch.Tensor):
                        kd_loss_value += losses[k].item()
                    elif isinstance(losses[k], (int, float)):
                        kd_loss_value += losses[k]

            # 更新梯度统计（使用指数移动平均）
            self.gradient_stats['det_loss'] = (
                    0.9 * self.gradient_stats['det_loss'] + 0.1 * det_loss_value
            )
            self.gradient_stats['kd_loss'] = (
                    0.9 * self.gradient_stats['kd_loss'] + 0.1 * kd_loss_value
            )

        except Exception as e:
            print(f"Gradient statistics error: {e}")
            # 设置默认值保证训练继续
            self.gradient_stats['det_loss'] = 1.0
            self.gradient_stats['kd_loss'] = 0.5

        return losses

    def calculate_enhanced_kd_loss(self, tea_cls, tea_bbox, tea_center,
                                   reused_cls, reused_bbox, reused_center,
                                   kd_weights, device):
        """改进的KD损失计算"""
        losses = {}

        try:
            # 1. 自适应温度缩放的分类KD损失
            temperature = max(0.5, 4.0 * (1 - self.current_epoch / self.total_epochs))

            cls_loss = 0
            for t, s in zip(tea_cls, reused_cls):
                # 确保维度匹配
                if t.shape == s.shape:
                    # 使用温度缩放的KL散度
                    t_soft = F.softmax(t / temperature, dim=-1)
                    s_soft = F.log_softmax(s / temperature, dim=-1)
                    cls_loss += F.kl_div(s_soft, t_soft, reduction='batchmean') * (temperature ** 2)

            if cls_loss > 0:
                losses['loss_cls_kd'] = cls_loss / len(tea_cls) * kd_weights['cls']

            # 2. 焦点回归KD损失
            reg_loss = 0
            for t, s in zip(tea_bbox, reused_bbox):
                if t.shape == s.shape:
                    # 使用smooth L1损失，对困难样本加权
                    diff = torch.abs(t - s)
                    weight = torch.exp(diff)  # 困难样本权重更高
                    reg_loss += (F.smooth_l1_loss(s, t, reduction='none') * weight).mean()

            if reg_loss > 0:
                losses['loss_reg_kd'] = reg_loss / len(tea_bbox) * kd_weights['reg']

            # 3. 中心度KD损失
            center_loss = 0
            for t, s in zip(tea_center, reused_center):
                if t.shape == s.shape:
                    center_loss += F.binary_cross_entropy_with_logits(
                        s, t.sigmoid(), reduction='mean')

            if center_loss > 0:
                losses['loss_center_kd'] = center_loss / len(tea_center) * kd_weights['center']

        except Exception as e:
            print(f"Enhanced KD loss error: {e}")
            # 保证返回有效的损失张量
            for key in ['loss_cls_kd', 'loss_reg_kd', 'loss_center_kd']:
                losses[key] = torch.tensor(0.0, device=device)

        return losses
    def pred_imitation_loss_single(self,
                                   labels,
                                   anchors,
                                   tea_cls_score,
                                   tea_bbox_pred,
                                   tea_centernesses,
                                   reused_cls_score,
                                   reused_bbox_pred,
                                   reused_centernesses,
                                   label_weights,
                                   bbox_targets,#add by jojo 20250425
                                   avg_factor):

        # classification branch distillation

        # print("tea_bbox_pred",tea_cls_score .shape)#[2,10000,256]

        # tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
        # reused_cls_score = reused_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
        # label_weights = label_weights.reshape(-1)

        #Add by jojo 20250423
        # tea_cls_score0 = tea_cls_score.reshape(-1, 256)  # [2,100000,256]
        # reused_cls_score0 = reused_cls_score.reshape(-1, 256)
        # label_weights0 = label_weights.reshape(-1)
        # Loss is not computed for the padded regions of the text.
        # ===== this change =====
        anchors = anchors.reshape(-1, 4)

        pos_inds = (labels.sum(-1) > 0).reshape(-1)
        assert (self.text_masks.dim() == 2)
        text_mask = (self.text_masks > 0).unsqueeze(1)#([2, 1, 256])
        text_mask = text_mask.repeat(1, tea_cls_score.size(1), 1)# torch.Size([2, 10000, 256])
        tea_cls_score= torch.masked_select(tea_cls_score, text_mask).contiguous()#focus on the valid area
        reused_cls_score=torch.masked_select(reused_cls_score, text_mask).contiguous()
        # print("tea_cls_score:",tea_cls_score)#logits value
        # labels1=labels.repeat(1, tea_cls_score.size(1), 1)
        # labels1 = torch.masked_select(labels1, text_mask)#error
        label_weights = label_weights[...,
        None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        # print("reused_cls_score:",reused_cls_score.shape)#torch.Size([380000])
        # print("tea_cls_score",tea_cls_score.shape)#torch.Size([380000])
        # print("label_weights",label_weights.shape)#torch.Size([380000])

        tea_bbox_pred = tea_bbox_pred.reshape(-1, 4)
        reused_bbox_pred = reused_bbox_pred.reshape(-1, 4)

        tea_centernesses = tea_centernesses.reshape(-1)
        reused_centernesses = reused_centernesses.reshape(-1)

        label_weights = label_weights.reshape(-1)

        #------transformed into 2d
        tea_cls_score = tea_cls_score.repeat(1, 1).permute(1,0)
        reused_cls_score=reused_cls_score.repeat(1, 1).permute(1,0)
        #------end

        # -----add end-------
        # print("reused_cls_score:",reused_cls_score.shape)#torch.Size([507034])


        loss_cls_kd = self.loss_cls_kd(
            reused_cls_score,
            tea_cls_score,
            label_weights,
            avg_factor=avg_factor)

        # loss_cls_kd = self.loss_cls_kd(
        #     reused_cls_score,
        #     tea_cls_score)

        # print("loss_cls_kd:", loss_cls_kd)#yes!

        # regression branch distillation
        #add by jojo 20250423
        if pos_inds.sum() > 0:
            #targets are teacher pos is student
            tea_bbox_pred= tea_bbox_pred[pos_inds]
            reused_bbox_pred = reused_bbox_pred[pos_inds]

            pos_anchors = anchors[pos_inds]
            pos_centerness = reused_centernesses[pos_inds]#centress reused

            centerness_targets = self.teacher.bbox_head.centerness_target(
                pos_anchors, tea_bbox_pred)#tea_centerness
            # print("tea_centernesses", tea_centernesses.shape)
            # print("centerness_targets",centerness_targets.shape)

            if torch.isnan(centerness_targets).any():
                print('=====Centerness includes NaN=====')

                mask = ~torch.isnan(centerness_targets)
                # tea_centernesses = centerness_targets[mask]
                pos_centerness = pos_centerness[mask]
                pos_anchors = pos_anchors[mask]
                tea_bbox_pred = tea_bbox_pred[mask]
                reused_bbox_pred = reused_bbox_pred[mask]

                if tea_bbox_pred.shape[0] == 0:
                    loss_bbox = reused_bbox_pred.sum() * 0
                    loss_centerness = pos_centerness.sum() * 0
                    # tea_centernesses= tea_bbox_pred .new_tensor(0.)
                    return loss_cls_kd, loss_bbox, loss_centerness

                    # The decoding process takes the offset into consideration.
            pos_anchors[:, 2:] += 1
            reused_bbox_pred = self.teacher.bbox_head.bbox_coder.decode(
                        pos_anchors, reused_bbox_pred)
            tea_bbox_pred=self.teacher.bbox_head.bbox_coder.decode(
                        pos_anchors, tea_bbox_pred)

            # print("reused_bbox_pred:",reused_bbox_pred.shape)
            # print(reused_bbox_pred)
            # print("tea_bbox_pred:",tea_bbox_pred.shape)
            # print(tea_bbox_pred)

            loss_reg_kd = self.loss_reg_kd(
                reused_bbox_pred,
                tea_bbox_pred,
                weight=centerness_targets,
                avg_factor=1.0)
            # centerness loss
            # loss_center_kd  = self.loss_center_kd(
            #     pos_centerness, centerness_targets, avg_factor=avg_factor)
            loss_center_kd = self.loss_center_kd(
                pos_centerness, tea_centernesses[pos_inds].sigmoid(), avg_factor=avg_factor)
            # print("loss_reg_kd_0", loss_reg_kd)
            # print("loss_center_kd_0", loss_center_kd)
        else:
            loss_reg_kd = reused_bbox_pred.sum() * 0
            loss_center_kd = reused_centernesses.sum() * 0
            # print("loss_reg_kd_1",loss_reg_kd)
            # print("loss_center_kd_1",loss_center_kd)


        return loss_cls_kd, loss_reg_kd, loss_center_kd





class CrossModalAttention(nn.Module):
    def __init__(self, v_dim, l_dim, num_heads=8):
        super().__init__()
        # 视觉特征投影
        self.vision_proj = nn.Sequential(
            nn.Conv2d(v_dim, l_dim, 1),
            nn.GroupNorm(8, l_dim)
        )

        # 文本特征压缩与投影
        self.text_compress = nn.Sequential(
            nn.Conv1d(256, 64, 3, padding=1),  # 压缩位置维度256→64
            nn.ReLU(inplace=True)
        )
        self.text_proj1 = nn.Linear(64, l_dim)

        # 多头注意力机制
        self.attn = nn.MultiheadAttention(
            embed_dim=l_dim,
            num_heads=num_heads,
            batch_first=True  # 使用batch_first格式
        )

    def forward(self, visual_feat, text_feat):
        """
        visual_feat: [B, C, H, W]
        text_feat: [B, 256, 512] (D=256位置数，每个位置512维)
        """
        # 视觉特征处理
        B, C, H, W = visual_feat.size()
        v_proj = self.vision_proj(visual_feat)  # [B, l_dim, H, W]
        v_flat = v_proj.view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, l_dim]

        # 文本特征处理
        # 压缩位置维度 256→64
        text_compressed = self.text_compress(text_feat)  # [B, 64, 512] [2,64,512]
        # 投影到与视觉相同的维度空间
        l_proj = self.text_proj1(text_compressed.permute(0, 2, 1))  # [B, 512, 64] → [B, 64, l_dim]

        # 交叉注意力计算
        attn_out, _ = self.attn(
            query=v_flat,  # [B, HW, l_dim]
            key=l_proj,  # [B, 64, l_dim]
            value=l_proj  # [B, 64, l_dim]
        )

        # 恢复空间维度
        output = attn_out.permute(0, 2, 1).view(B, -1, H, W)
        return output


class TextAwareFPNLayer(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # 增加文本特征压缩层
        # 文本特征压缩层（处理3D文本特征）
        self.text_compress = nn.Sequential(
            nn.Conv1d(256, 64, 3, padding=1),  # 处理位置维度
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 压缩位置维度 [B, 64, 1]
        )

        # 门控生成器
        self.text_gate = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feat_dim),
            nn.Sigmoid()
        )

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, 3, padding=1),
            nn.GroupNorm(8, feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, visual_feat, text_feat):
        """
        visual_feat: [B, C, H, W] (C=feat_dim)
        text_feat: [B, 256, 512] (D=256位置数, 每个位置512维)
        """
        # 步骤1：文本特征压缩
        # [B, 256, 512] -> [B, 64, 512] -> [B, 64, 1]
        compressed_text = self.text_compress(text_feat)  # [B, 64, 1]

        # 步骤2：通道维度转换
        # [B, 64, 1] -> [B, 1, 512]
        text_feat = compressed_text.permute(0, 2, 1)  # [B, 1, 64]
        text_feat = F.interpolate(text_feat, size=256)  # [B, 1, 512]

        # 步骤3：生成门控信号
        # [B, 1, 512] -> [B, 512] -> [B, C]
        gate = self.text_gate(text_feat.squeeze(1))  # [B, C]
        gate = gate.view(-1, gate.size(1), 1, 1)  # [B, C, 1, 1]

        # 步骤4：门控增强
        gated_feat = visual_feat * gate  # [B, C, H, W]

        # 步骤5：残差融合
        fused = self.fusion(torch.cat([visual_feat, gated_feat], dim=1))
        return fused + visual_feat


class VisualEnhancedText(nn.Module):
    def __init__(self, text_dim=768, vis_dim=256, num_heads=8):  # 修改1：调整头数与维度
        super().__init__()
        # 视觉上下文提取（适配768维度）
        self.vis_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(vis_dim, text_dim, 1),  # 输出通道调整为768
            nn.LayerNorm([text_dim, 1, 1])
        )

        # 跨模态注意力（适配768维度）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim,  # 768
            num_heads=num_heads,  # 头数12（768/64=12）
            batch_first=True,
            kdim=text_dim,  # 新增键值维度指定
            vdim=text_dim
        )

        # 门控融合（适配768维度）
        self.gate = nn.Sequential(
            nn.Linear(text_dim * 2, text_dim),  # 输入1536→768
            nn.Sigmoid()
        )

    def forward(self, text_feat, visual_feat):
        """
        输入维度说明：
        text_feat:  [B, 256, 768] （L=256位置数，D=768维度）
        visual_feat: [B, 256, H, W]
        """
        # 视觉上下文提取
        vis_ctx = self.vis_context(visual_feat)  # [B,768,1,1]
        vis_ctx = vis_ctx.view(vis_ctx.size(0), -1).unsqueeze(1)  # [B,1,768]


        # print("text_feat:",text_feat.shape)#text_feat: torch.Size([2, 256, 768])
        # print("vis_ctx:",vis_ctx.shape)#vis_ctx: torch.Size([2, 1, 768])

        # 交叉注意力（Query=Text, Key/Value=Visual）
        attn_output, _ = self.cross_attn(
            query=text_feat,
            key=vis_ctx,
            value=vis_ctx
        )  # [B,768,256]
        # print("attn_output",attn_output.shape)

        # 残差连接
        # attn_output = attn_output.permute(0, 2, 1)  # [B,256,768]
        temp=torch.cat([text_feat, attn_output],dim=-1)
        # print("temp:", temp.shape) #temp: torch.Size([2, 256, 1536])
        gate = self.gate( temp)

        enhanced_text = gate * text_feat + (1 - gate) * attn_output

        return enhanced_text  # 保持[B,256,768]

