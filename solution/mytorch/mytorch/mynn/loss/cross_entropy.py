import torch
from torch import nn

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, logits, targets):
        log_softmax = logits.log_softmax(dim=-1)
        
        batch_size = logits.size(0)
        # TODO，填写一句话计算出每个样本的交叉熵损失
        '''《pass》'''
        #《
        loss = -log_softmax[torch.arange(batch_size), targets]
        #》
        
        if self.reduction == 'mean':
            # TODO，用取均值的方法处理多个样本的交叉熵损失
            '''《pass》'''
            #《
            return loss.mean()
            #》
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss