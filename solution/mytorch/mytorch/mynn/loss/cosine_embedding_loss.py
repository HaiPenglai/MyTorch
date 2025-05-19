import torch
from torch import nn

class CustomCosineEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.0, reduction='mean'):
        super(CustomCosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, x1, x2, target):
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(x1, x2, dim=1)
        
        # 根据target计算损失
        # target = 1 表示相似，target = -1 表示不相似
        loss = torch.where(target == 1,
                          1 - cos_sim,  # 相似样本的损失
                          torch.clamp(cos_sim - self.margin, min=0))  # 不相似样本的损失
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 