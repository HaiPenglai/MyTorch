import torch
from torch import nn

class CustomHingeEmbeddingLoss(nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        super(CustomHingeEmbeddingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, pred, target):
        # pred: 预测值
        # target: 目标值，应为1或-1
        # 当y=1时，损失为x；当y=-1时，损失为max(0, margin - x)
        loss = torch.where(target == 1,
                          pred,
                          torch.clamp(self.margin - pred, min=0.0))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 