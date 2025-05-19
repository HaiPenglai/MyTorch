import torch
from torch import nn

class CustomTripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2.0, eps=1e-6, swap=False, reduction='mean'):
        super(CustomTripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction
        
    def forward(self, anchor, positive, negative):
        # 计算正样本距离
        d_pos = torch.pairwise_distance(anchor, positive, p=self.p, eps=self.eps)
        
        # 计算负样本距离
        d_neg = torch.pairwise_distance(anchor, negative, p=self.p, eps=self.eps)
        
        if self.swap:
            # 如果启用swap，计算positive和negative之间的距离
            d_swap = torch.pairwise_distance(positive, negative, p=self.p, eps=self.eps)
            d_neg = torch.min(d_neg, d_swap)
            
        # 计算三元组损失
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 