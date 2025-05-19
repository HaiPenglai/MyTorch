import torch
from torch import nn

class CustomMultiMarginLoss(nn.Module):
    def __init__(self, p=1, margin=1.0, weight=None, reduction='mean'):
        super(CustomMultiMarginLoss, self).__init__()
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        # 确保输入是2D张量
        if input.dim() != 2:
            raise ValueError('Expected 2D input tensor')

        # 获取每个样本的正确类别分数
        correct_scores = input[torch.arange(input.size(0)), target]

        # 计算损失（逐样本计算）
        loss = torch.zeros(input.size(0), dtype=input.dtype, device=input.device)
        for i in range(input.size(1)):
            # 使用掩码（mask）过滤掉"i"等于 target 的样本，从而避免"if i != target"的比较歧义
            mask = (i != target)
            if mask.any():
                # 计算每个错误类别的损失（仅对 mask 为 True 的样本计算）
                loss[mask] += torch.clamp(self.margin - (correct_scores[mask] - input[mask, i]), min=0) ** self.p

        # 应用权重（如果提供）
        if self.weight is not None:
            loss *= self.weight[target]

        # 应用reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
