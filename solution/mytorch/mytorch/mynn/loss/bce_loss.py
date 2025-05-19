import torch
from torch import nn

class CustomBCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomBCELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, probs, targets):
        eps = 1e-7  # 避免log(0)的情况
        
        # 确保概率值在[0,1]范围内
        probs = torch.clamp(probs, eps, 1-eps)
        
        # 计算二元交叉熵损失
        # BCE = -(y * log(p) + (1-y) * log(1-p))
        loss = -(targets * torch.log(probs) + 
                (1 - targets) * torch.log(1 - probs))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss