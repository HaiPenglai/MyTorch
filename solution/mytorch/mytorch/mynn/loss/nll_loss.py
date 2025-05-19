import torch
from torch import nn

class CustomNLLLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomNLLLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        # pred应该是log_softmax的输出，也就是对数概率
        # target是类别索引
        batch_size = pred.size(0)
        loss = -pred[torch.arange(batch_size), target]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 