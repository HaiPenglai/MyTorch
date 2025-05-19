import torch
from torch import nn

class CustomHuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean'):
        super(CustomHuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, pred, target):
        # pred: 预测值
        # target: 目标值
        diff = torch.abs(pred - target)
        delta = self.delta
        
        # 当|pred - target| <= delta时使用MSE，否则使用L1
        loss = torch.where(diff <= delta,
                          0.5 * diff ** 2,
                          delta * diff - 0.5 * delta ** 2)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 