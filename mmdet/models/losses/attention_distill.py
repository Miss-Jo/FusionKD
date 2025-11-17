import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from .utils import weighted_loss

def norm(feat: torch.Tensor) -> torch.Tensor:
    """Normalize the feature maps to have zero mean and unit variances."""
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()[:2]
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    @staticmethod
    def get_attention(x):
        return F.adaptive_avg_pool2d(x, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(y))

    @staticmethod
    def get_attention(x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.cat([avg_out, max_out], dim=1)



@weighted_loss
def AttentionDistill_loss(pred, target):
    pred = norm(pred)
    target = norm(target)
    s_ca = ChannelAttention.get_attention(pred)
    t_ca = ChannelAttention.get_attention(target)

    # 空间注意力蒸馏
    s_sa = SpatialAttention.get_attention(pred)
    t_sa = SpatialAttention.get_attention(target)
    return (F.mse_loss(s_ca, t_ca.detach(),reduction='none')/2+
            F.mse_loss(s_sa, t_sa.detach(), reduction='none') / 2)

@MODELS.register_module()
class AttentionDistillLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(AttentionDistillLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None) -> torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * AttentionDistill_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss