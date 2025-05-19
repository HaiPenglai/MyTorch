import torch
from torch import nn

class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, logits, targets):
        eps = 1e-7
        
        # 使用sigmoid将logits转换为概率
        probs = torch.sigmoid(logits)
        
        # 计算二元交叉熵损失
        # BCE = -(y * log(p) + (1-y) * log(1-p))
        loss = -(targets * torch.log(probs + eps) + 
                (1 - targets) * torch.log(1 - probs + eps))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 