import torch
from torch import nn

class CustomSoftMarginLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomSoftMarginLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        # pred: 预测值
        # target: 目标值，应为-1或1
        loss = torch.log(1 + torch.exp(-target * pred))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 