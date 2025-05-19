import torch
from torch import nn
import math

class CustomGaussianNLLLoss(nn.Module):
    def __init__(self, reduction='mean', full=False):
        super(CustomGaussianNLLLoss, self).__init__()
        self.reduction = reduction
        self.full = full
        
    def forward(self, pred_mean, pred_var, target):
        # pred_mean: 预测的均值
        # pred_var: 预测的方差
        # target: 目标值
        
        # 确保方差为正
        pred_var = torch.clamp(pred_var, min=1e-7)
        
        # 计算损失
        loss = 0.5 * (torch.log(pred_var) + 
                      (target - pred_mean).pow(2) / pred_var)
        
        if self.full:
            loss += 0.5 * math.log(2 * math.pi)
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 