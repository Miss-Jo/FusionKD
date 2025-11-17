import torch
# from mmcv.runner import _load_checkpoint
from mmengine.runner import load_checkpoint
# from mmdet.models import (SingleStageDetector, DETECTORS,
#                           build_backbone, build_neck, build_head)
# from mmdet.models.detectors import SingleStageDetector  # 更新基类路径
from .single_stage import SingleStageDetector
from mmdet.registry import MODELS
# from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
#                          OptInstanceList, reduce_mean)
from mmdet.structures import DetDataSample #added by jojo 20250415


# MMDetection 3.x 对代码结构进行了重构：
#
# ​​注册器机制​​：使用 mmdet.registry 统一管理模型组件，弃用直接导入 DETECTORS。
# ​​构建函数迁移​​：build_backbone、build_neck、build_head 函数被整合进注册器 MODELS。
# ​​模型基类路径变更​​：SingleStageDetector 移动到 mmdet.models.detectors
@MODELS.register_module()
class LAD(SingleStageDetector):
    """Label Assignment Distillation https://arxiv.org/abs/2108.10520"""

    def __init__(self,
                 # student
                 backbone,#在config文件中已做修改,由于V2.0与V3.0差异
                 neck,
                 bbox_head,
                 # teacher
                 teacher_backbone,
                 teacher_neck,
                 teacher_bbox_head,
                 teacher_pretrained,
                 # cfgs
                 train_cfg=None,
                 test_cfg=None,
                 # pretrained=None,
                 data_preprocessor= None,):#add by jojo 20250415
        super().__init__(backbone=backbone, neck=neck,  bbox_head=bbox_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, data_preprocessor=data_preprocessor)#add by jojo 20250415 data_preprocessor pretrained=pretrained,
        # self.teacher_backbone = build_backbone(teacher_backbone)

        self.teacher_backbone = MODELS.build(teacher_backbone)#modified by jojo 20250415
        if teacher_neck is not None:
            # self.teacher_neck = build_neck(teacher_neck)
            self.teacher_neck = MODELS.build(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        # self.teacher_bbox_head = build_head(teacher_bbox_head)##modified by jojo 20250415
        self.teacher_bbox_head = MODELS.build(teacher_bbox_head)
        self.init_teacher_weights(teacher_pretrained=teacher_pretrained)

    def init_teacher_weights(self, teacher_pretrained):
        # ckpt = _load_checkpoint(teacher_pretrained, map_location='cpu')
        # load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')
        ckpt = torch.load(teacher_pretrained, map_location='cpu')#modified by jojo 20250415

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        teacher_ckpt = dict()
        for key in ckpt:
            teacher_ckpt['teacher_' + key] = ckpt[key]
        self.load_state_dict(teacher_ckpt, strict=False)
        print("Init teacher weights done")

    def teacher_eval(self):
        self.teacher_backbone.eval()
        self.teacher_neck.eval()
        self.teacher_bbox_head.eval()
        # add by jojo 20250415
        for param in self.teacher_backbone.parameters():
            param.requires_grad_(False)
        for param in self.teacher_neck.parameters():
            param.requires_grad_(False)
        for param in self.teacher_bbox_head.parameters():
            param.requires_grad_(False)
        # add end


    def student_eval(self):
        self.backbone.eval()
        self.neck.eval()
        self.bbox_head.eval()

    def student_train(self):
        self.backbone.train()
        self.neck.train()
        self.bbox_head.train()

    # def extract_teacher_feat(self, img):
    #     x = self.teacher_backbone(img)
    #     if self.with_neck:
    #         x = self.teacher_neck(x)
    #     return x
    from torch import Tensor
    from typing import List, Tuple, Optional
    def extract_teacher_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """教师模型特征提取（适配3.x参数）"""
        # 假设教师模型与学生模型结构相同
        x = self.teacher_backbone(batch_inputs)
        if self.with_neck:
            x = self.teacher_neck(x)
        return x
    # #
    # # # 在mmdet3.0版本中,需要更新版本为3.0版本loss
    # def forward_train(self,
    #          img,
    #          img_metas,
    #          gt_bboxes,
    #          gt_labels,
    #          gt_bboxes_ignore=None):
    #     super(SingleStageDetector, self).forward_train(img, img_metas)
    #     # forward student
    #
    #     x = self.extract_feat(img)
    #     # print("x:",x.shape)
    #
    #     outs = self.bbox_head(x)
    #
    #     # teacher infers Label Assignment results
    #     with torch.no_grad():
    #         # MUST force teacher to `eval` every training step, b/c at the
    #         # beginning of epoch, the runner calls all nn.Module elements
    #         # to be `train`
    #         self.teacher_eval()
    #
    #         # assignment result is obtained based on only teacher
    #         x_teacher = self.extract_teacher_feat(img)
    #         la_results = self.teacher_bbox_head.forward_la(
    #             x_teacher, img_metas, gt_bboxes, gt_labels,
    #             gt_bboxes_ignore=None)
    #
    #     # student receives the assignment results to learn
    #     losses = self.bbox_head.forward_train_wo_la(
    #         outs, img_metas, gt_bboxes, gt_labels,
    #         gt_bboxes_ignore=gt_bboxes_ignore, la_results=la_results)
    #
    #     return losses


    #modified by jojo 20250415
    from typing import List, Tuple, Optional
    import torch
    from torch import Tensor
    def loss(self,
             batch_inputs: Tensor,
             batch_data_samples: List[DetDataSample],  # 替换原有分散参数
             **kwargs) -> dict:
        """3.x 版本的核心训练方法"""

        # ===================== 参数转换 =====================
        # 将 DetDataSample 转换为旧版格式（兼容原有逻辑）
        img_metas = [s.metainfo for s in batch_data_samples]
        gt_bboxes = [s.gt_instances.bboxes for s in batch_data_samples]
        gt_labels = [s.gt_instances.labels for s in batch_data_samples]
        # gt_bboxes_ignore = [s.gt_instances.get('bboxes_ignore', None) for s in batch_data_samples]

        # ===================== 核心逻辑 =====================
        # 特征提取（保持原有实现）
        x = self.extract_feat(batch_inputs)  # 注意输入参数改为 batch_inputs

        # 学生模型前向
        outs = self.bbox_head(x)


        # ===================== 教师模型部分 =====================
        with torch.no_grad():
            # 确保教师模型在评估模式
            self.teacher_eval()  # 需要实现该方法

            # 教师模型特征提取
            x_teacher = self.extract_teacher_feat(batch_inputs)  # 适配新参数
            # outs_teacher = self.teacher_bbox_head(x_teacher)#addded by jojo 20250415

            # 生成标签分配结果（需适配新数据格式）
            la_results = self.teacher_bbox_head.forward_la(
                x_teacher,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=None,
                # return_outputs=True
            )


        # ============== 核心修复：构建伪梯度路径 ==============
        ## 强制引用所有需要参与梯度的参数
        unused_params = []
        pseudo_loss = 0.0
        for name, param in self.named_parameters():
            if param.grad is None and param.requires_grad:
                unused_params.append(name)
                pseudo_loss += param.sum() * 0.0  # 不影响实际损失
        print("len(unused_params) ", len(unused_params))
        if len(unused_params) > 0:
            print(f"123 - {unused_params}")



        # ===================== 损失计算 =====================
        losses = self.bbox_head.forward_train_wo_la(
            outs,
            img_metas=img_metas,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            gt_bboxes_ignore=None,
            la_results=la_results,
        )#modified by jojo 20250415




        losses['pseudo'] = pseudo_loss#add by jojo 20150415
        print("losses:", losses)
        # losses: {'loss_cls': tensor(1.2678, device='cuda:0', grad_fn= < MulBackward0 >), 'loss_bbox': tensor(0.8785,
        #                                                                                                      device='cuda:0',
        #                                                                                                      grad_fn= < MulBackward0 >), 'loss_iou': tensor(
        #     0.3507, device='cuda:0', grad_fn= < MulBackward0 >), 'pseudo': tensor(0., device='cuda:0',
        #                                                                           grad_fn= < AddBackward0 >)}


        print("losses done!")

        return losses
