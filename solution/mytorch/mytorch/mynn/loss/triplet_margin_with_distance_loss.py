import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTripletMarginWithDistanceLoss(nn.Module):
    def __init__(self, distance_function=None, margin=1.0, swap=False, reduction='mean'):
        super(CustomTripletMarginWithDistanceLoss, self).__init__()
        self.distance_function = distance_function if distance_function is not None else lambda x, y: F.pairwise_distance(x, y, p=2)
        self.margin = margin
        self.swap = swap
        self.reduction = reduction
        
    def forward(self, anchor, positive, negative):
        # 计算锚点-正样本距离
        distance_pos = self.distance_function(anchor, positive)
        
        # 计算锚点-负样本距离
        distance_neg = self.distance_function(anchor, negative)
        
        # 如果启用swap，计算正样本-负样本距离并取最小值
        if self.swap:
            distance_swap = self.distance_function(positive, negative)
            distance_neg = torch.min(distance_neg, distance_swap)
        
        # 计算triplet loss
        losses = F.relu(distance_pos - distance_neg + self.margin)
        
        # 根据reduction返回结果
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses