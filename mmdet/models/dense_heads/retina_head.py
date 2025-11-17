# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from .anchor_head import AnchorHead


global counter
counter=0

def visualize_class_comparison(cls_feat, atss_cls, class_indices, original_image=None, figsize=(15, 10)):
    """
    改进的类别比较可视化，处理尺寸不匹配问题

    参数:
    cls_feat: 分类分支特征图 [1, C, H, W]
    atss_cls: 分类卷积层
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
        _,_,feat_H, feat_W = cls_feat.shape
        print(cls_feat.shape)

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

        weights = atss_cls.weight.data
        print("weights:",weights.shape) #$[80,256,3,3] #[7,1024]

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
        save_path="MGD_frcnn_head"
        if save_path:
            import os
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(f"{save_path}/level_{counter}.png", dpi=500, bbox_inches='tight')
            plt.close(save_path)
        plt.show()

@MODELS.register_module()
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        assert stacked_convs >= 0, \
            '`stacked_convs` must be non-negative integers, ' \
            f'but got {stacked_convs} instead.'
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        in_channels = self.in_channels
        for i in range(self.stacked_convs):
            self.cls_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            in_channels = self.feat_channels
        self.retina_cls = nn.Conv2d(
            in_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        reg_dim = self.bbox_coder.encode_size
        self.retina_reg = nn.Conv2d(
            in_channels, self.num_base_priors * reg_dim, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        # add
        import mmcv
        original_image = mmcv.imread("examples_jojo/scratches_6.jpg")
        visualize_class_comparison(
            cls_feat,
            self.retina_cls,
            class_indices=[0, 5],  # person, bicycle, car, motorcycle
            original_image=original_image
        )
        # add end
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
