import torch
from torch import nn

class CustomKLDivLoss(nn.Module):
    def __init__(self, reduction='mean', log_target=False):
        super(CustomKLDivLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target
        
    def forward(self, pred, target):
        # 计算公式：KL(p||q) = p * (log(p) - log(q))
        
        if self.log_target:
            # target=log(p), pred=log(q)
            loss = torch.exp(target) * (target - pred)
        else:
            # target=p, pred=log(q)
            loss = target * (torch.log(target + 1e-7) - pred)
        
        if self.reduction == 'batchmean':
            # 特殊处理batchmean，只除以batch size
            return loss.sum() / pred.size(0)
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 