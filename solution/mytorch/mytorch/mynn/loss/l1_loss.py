import torch
from torch import nn

class CustomL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomL1Loss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        # 计算绝对误差
        loss = torch.abs(pred - target)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 