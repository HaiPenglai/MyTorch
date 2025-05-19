import torch
from torch import nn

class CustomSmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean'):
        super(CustomSmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        
    def forward(self, pred, target):
        # pred: 预测值
        # target: 目标值
        diff = torch.abs(pred - target)
        beta = self.beta
        
        # 当|pred - target| < beta时使用MSE，否则使用L1
        loss = torch.where(diff < beta,
                          0.5 * diff ** 2 / beta,
                          diff - 0.5 * beta)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 