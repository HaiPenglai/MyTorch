import torch
from torch import nn

class CustomMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomMSELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        # 计算平方误差
        loss = (pred - target).pow(2)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 