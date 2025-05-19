import torch
from torch import nn

class CustomMarginRankingLoss(nn.Module):
    def __init__(self, margin=0.0, reduction='mean'):
        super(CustomMarginRankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, x1, x2, target):
        # 计算损失
        # target = 1 表示 x1 应该大于 x2
        # target = -1 表示 x1 应该小于 x2
        loss = torch.clamp(-target * (x1 - x2) + self.margin, min=0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 