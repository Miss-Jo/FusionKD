# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import mmcv
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import images_to_levels, multi_apply, unmap
from .anchor_head import AnchorHead

global counter
counter=0

def visualize_class_comparison(cls_feat, atss_cls, class_indices, original_image=None, figsize=(15, 10)):
    """
    改进的类别比较可视化，处理尺寸不匹配问题

    参数:
    cls_feat: 分类分支特征图 [1, C, H, W]
    atss_cls: 分类卷积层but
    class_indices: 要比较的类别索引列表
    original_image: 原始图像 [H, W, C]
    figsize: 图像大小
    """
    # 确保不计算梯度
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    global counter
    counter=counter+1
    with torch.no_grad():
        # 获取特征图尺寸
        _, _, feat_H, feat_W = cls_feat.shape

        # 准备多子图
        n = len(class_indices)
        rows = int(np.sqrt(n))
        cols = int(np.ceil(n / rows))

        # 创建图
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows * cols > n:
            # 移除多余的空图
            for i in range(n, rows * cols):
                fig.delaxes(axes.flatten()[i])

        # 获取分类权重
        weights = atss_cls.weight.data

        for idx, class_idx in enumerate(class_indices):
            # 计算当前类别的CAM
            class_weights = weights[class_idx]  # [in_channels, kH, kW]

            # 自动处理权重尺寸匹配
            c_in = class_weights.size(0)
            kH, kW = class_weights.size(1), class_weights.size(2)

            if kH != feat_H or kW != feat_W:
                # 重新调整权重大小以匹配特征图
                class_weights = torch.nn.functional.interpolate(
                    class_weights.unsqueeze(0),  # [1, in_channels, kH, kW]
                    size=(feat_H, feat_W),
                    mode='bilinear',
                    align_corners=False
                )[0]  # 移除batch维度

            # 添加必要的维度
            class_weights = class_weights.unsqueeze(0)  # [1, in_channels, H, W]

            # 计算CAM
            cam = torch.sum(cls_feat * class_weights, dim=1, keepdim=True)
            cam = F.relu(cam)

            if original_image is not None:
                cam = F.interpolate(cam, size=original_image.shape[:2], mode='bilinear', align_corners=False)

            cam = cam.squeeze().cpu().numpy()
            cam -= cam.min()
            cam /= cam.max() + 1e-8

            # 获取子图坐标
            ax = axes.flatten()[idx]

            if original_image is not None:
                ax.imshow(original_image)
                ax.imshow(cam, cmap='jet', alpha=0.5)
            else:
                ax.imshow(cam, cmap='jet')

            ax.set_title(f"Class {class_idx}")
            ax.axis('off')

        plt.tight_layout()
        plt.suptitle("Class Attention Comparison", fontsize=16, y=0.95)
        save_path="GLIP_atss_head"
        if save_path:
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(f"{save_path}/level_{counter}.png", dpi=500, bbox_inches='tight')
            plt.close(save_path)
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.transform import resize
def visualize_features(tea_cls_scores, tea_bbox_preds, tea_centernesses,
                       mode='pca', target_size=100, save_path=None):
    """
    多尺度特征可视化工具

    参数:
        tea_cls_scores: list of tensors, 分类特征 [1, N, 256]
        tea_bbox_preds: list of tensors, 边界框预测 [1, 4, H, W]
        tea_centernesses: list of tensors, 中心度特征 [1, 1, H, W]
        mode: 'pca' 或 'max'，分类特征可视化模式
        target_size: 统一缩放的目标尺寸
        save_path: 图片保存路径（如不提供则显示）
    """
    num_levels = len(tea_cls_scores)

    for i in range(num_levels):
        # 1. 特征提取与转换
        ## 分类特征处理 [2](@ref)
        print(len(tea_cls_scores[i][0].shape))
        if len(tea_cls_scores[i][0].shape)==3:
            cls_data = tea_cls_scores[i][0].permute(1,2,0).reshape(-1,80).detach().cpu().numpy()  # [N, 256]
        else:
            cls_data = tea_cls_scores[i][0].detach().cpu().numpy()  # [N, 256]

        H_bbox, W_bbox = tea_bbox_preds[i].shape[2], tea_bbox_preds[i].shape[3]
        cls_map = cls_data.reshape(H_bbox, W_bbox, -1)  # [H, W, 256]

        ## 边界框特征处理 [8](@ref)
        bbox_data = tea_bbox_preds[i][0].detach().cpu().numpy()  # [4, H, W]

        ## 中心度特征处理 [4](@ref)
        center_data = tea_centernesses[i][0, 0].detach().cpu().numpy()  # [H, W]

        # 2. 特征归一化与上采样
        def normalize_and_resize(data, order=0):
            """ 归一化并上采样到目标尺寸 """
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-7)
            return resize(data_norm, (target_size, target_size), order=order,
                          preserve_range=True, anti_aliasing=False)

        # 3. 分类特征可视化模式选择
        if mode == 'pca':
            # PCA降维到RGB三通道 [5](@ref)
            pca = PCA(n_components=3)
            cls_flat = cls_data.reshape(-1, cls_data.shape[-1])
            cls_rgb = pca.fit_transform(cls_flat).reshape(H_bbox, W_bbox, 3)
            cls_vis = normalize_and_resize(cls_rgb)
        else:  # 'max'模式
            cls_max = np.max(cls_data, axis=1).reshape(H_bbox, W_bbox)
            cls_vis = normalize_and_resize(cls_max)

        # 4. 特征统一上采样
        bbox_vis = [normalize_and_resize(bbox_data[j]) for j in range(4)]
        center_vis = normalize_and_resize(center_data)

        # 5. 创建可视化画布 [2,4](@ref)
        plt.figure(figsize=(16, 4))
        plt.suptitle(f'Level {i} Feature Maps | Size: {H_bbox}x{W_bbox}', fontsize=16)

        # 6. 绘制分类特征
        plt.subplot(1, 6, 1)
        plt.title(f'Cls Logits ({mode.upper()})')
        if mode == 'pca':
            plt.imshow(cls_vis)
        else:
            plt.imshow(cls_vis, cmap='viridis') #viridis
        plt.colorbar( shrink=0.8)

        # 7. 绘制中心度特征
        plt.subplot(1, 6, 2)
        plt.title('Centerness')
        plt.imshow(center_vis, cmap='hot')
        plt.colorbar( shrink=0.8)

        # 8. 绘制边界框特征（四通道）
        channels = ['dx1', 'dy1', 'dx2', 'dy2']
        for j in range(4):
            plt.subplot(1, 6, 3 + j)  # 第二行位置
            plt.title(f'BBox: {channels[j]}')
            plt.imshow(bbox_vis[j], cmap='plasma')
            plt.colorbar( shrink=0.8)

        # 9. 保存或显示结果
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(f"{save_path}/level_{i}.png", dpi=500, bbox_inches='tight')
            plt.close(save_path)
        else:
            plt.show()



@MODELS.register_module()
class ATSSHead(AnchorHead):
    """Detection Head of `ATSS <https://arxiv.org/abs/1912.02424>`_.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        pred_kernel_size (int): Kernel size of ``nn.Conv2d``
        stacked_convs (int): Number of stacking convs of the head.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to ``dict(type='GN', num_groups=32,
            requires_grad=True)``.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Defaults to False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_centerness (:obj:`ConfigDict` or dict): Config of centerness loss.
            Defaults to ``dict(type='CrossEntropyLoss', use_sigmoid=True,
            loss_weight=1.0)``.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 pred_kernel_size: int = 3,
                 stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 reg_decoded_bbox: bool = True,
                 loss_centerness: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='atss_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            reg_decoded_bbox=reg_decoded_bbox,
            init_cfg=init_cfg,
            **kwargs)

        self.sampling = False
        self.loss_centerness = MODELS.build(loss_centerness)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pred_pad_size = self.pred_kernel_size // 2
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.atss_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 1,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        # cls_score,bbox_pred, centerness=multi_apply(self.forward_single, x, self.scales)
        # visualize_features(
        #     cls_score,
        #     bbox_pred,
        #     centerness,
        #     mode='max',  # 可选 'pca' 或 'max'
        #     target_size=300,  # 统一尺寸
        #     save_path='./feature_visualization_infer_101'  # 保存路径
        # )

        return multi_apply(self.forward_single, x, self.scales)

    def forward_single(self, x: Tensor, scale: Scale) -> Sequence[Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            # print("this is!")
            cls_feat = cls_conv(cls_feat)
            # print("cls_feat:",cls_feat.shape) #[1,80,7,7]


        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
            # print("reg_feat:", reg_feat.shape)#[1,256,13,13]
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        # print("cls_score:", cls_score.shape)#[1,80,7,7]
        # print(" centerness:", centerness.shape)#[1,1,7,7]
        # print(" bbox_pred", bbox_pred.shape)#[1,4,7,7]
        # 使用示例：
        # 比较多个类别的注意力分布
        # original_image=mmcv.imread("examples_jojo/scratches_6.jpg")
        # visualize_class_comparison(
        #     cls_feat,
        #     self.atss_cls,
        #     class_indices=[0,5],  # person, bicycle, car, motorcycle
        #     original_image=original_image
        # )


        return cls_score, bbox_pred, centerness

    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, centerness: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_targets: Tensor, avg_factor: float) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            avg_factor (float): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness, centerness_targets, avg_factor=avg_factor)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            centernesses: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
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
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets
        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, loss_centerness, \
            bbox_avg_factor = multi_apply(
                self.loss_by_feat_single,
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
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness)

    def centerness_target(self, anchors: Tensor, gts: Tensor) -> Tensor:
        """Calculate the centerness between anchors and gts.

        Only calculate pos centerness targets, otherwise there may be nan.

        Args:
            anchors (Tensor): Anchors with shape (N, 4), "xyxy" format.
            gts (Tensor): Ground truth bboxes with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Centerness between anchors and gts.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def get_targets(self,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True) -> tuple:
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = multi_apply(
             self._get_targets_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             unmap_outputs=unmap_outputs)
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, avg_factor)

    def _get_targets_single(self,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            num_level_anchors: List[int],
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors (List[int]): Number of anchors of each scale
                level.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
                sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances,
                                             num_level_anchors_inside,
                                             gt_instances, gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, sampling_result)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        """Get the number of valid anchors in every level."""

        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
