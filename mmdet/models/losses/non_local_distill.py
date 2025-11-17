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
class NonLocalModule(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        self.conv_theta = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_phi = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_g = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_out = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, x):
        batch_size = x.size(0)
        g = self.conv_g(x).view(batch_size, -1, x.size(2) * x.size(3))
        theta = self.conv_theta(x).view(batch_size, -1, x.size(2) * x.size(3))
        phi = self.conv_phi(x).view(batch_size, -1, x.size(2) * x.size(3))

        f = torch.matmul(theta.transpose(1, 2), phi)
        f = F.softmax(f, dim=-1)

        y = torch.matmul(g, f.transpose(1, 2))
        y = y.view(batch_size, -1, x.size(2), x.size(3))
        return x + self.conv_out(y)

    @staticmethod
    def get_relation(x):
        theta = x.view(x.size(0), x.size(1), -1)
        phi = x.view(x.size(0), x.size(1), -1)
        return torch.matmul(theta.transpose(1, 2), phi)

@weighted_loss
def NonLocalDistill_loss(pred, target):
    pred = norm(pred)
    target = norm(target)
    s_rel = NonLocalModule.get_relation(pred )
    t_rel = NonLocalModule.get_relation(target )
    loss=F.kl_div(
                F.log_softmax(s_rel / 3.0, dim=-1),
                F.softmax(t_rel.detach() / 3.0, dim=-1),
                reduction='none')
    return loss
@MODELS.register_module()
class NonLocalDistillLoss(nn.Module):
    def __init__(self,  reduction='mean',loss_weight=1.0):
        super(NonLocalDistillLoss,self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target,weight=None,          # 新增标准参数
                avg_factor=None,      # 新增标准参数
                reduction_override=None) -> torch.Tensor: # 新增标准参数):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * NonLocalDistill_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)

        return loss

