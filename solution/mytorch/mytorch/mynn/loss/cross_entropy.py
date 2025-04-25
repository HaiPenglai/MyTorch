import torch
from torch import nn

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, logits, targets):
        # 数值稳定的LogSoftmax计算
        log_softmax = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        
        # 计算NLL损失
        loss = -log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 根据reduction参数返回结果
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
    