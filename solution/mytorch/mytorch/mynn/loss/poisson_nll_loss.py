import torch
from torch import nn

class CustomPoissonNLLLoss(nn.Module):
    def __init__(self, reduction='mean', log_input=True, full=False):
        super(CustomPoissonNLLLoss, self).__init__()
        self.reduction = reduction
        self.log_input = log_input
        self.full = full
        
    def forward(self, pred, target):
        if self.log_input:
            loss = torch.exp(pred) - target * pred
        else:
            loss = pred - target * torch.log(pred + 1e-7)
            
        if self.full:
            # 添加斯特林近似项
            loss += target * torch.log(target + 1e-7) - target + 0.5 * torch.log(2 * torch.pi * target + 1e-7)
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 