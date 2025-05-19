import torch
import torch.nn as nn

class MultiLabelSoftMarginLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(MultiLabelSoftMarginLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, input, target):
        # 手动实现步骤：
        # 1. 计算sigmoid概率
        prob = torch.sigmoid(input)
        
        # 2. 计算二元交叉熵损失
        loss = - (target * torch.log(prob + 1e-8) + 
                 (1 - target) * torch.log(1 - prob + 1e-8))
        
        # 3. 应用类别权重
        if self.weight is not None:
            loss = loss * self.weight
            
        # 4. 对每个样本的所有类别求平均（与PyTorch原生实现一致）
        loss = loss.mean(dim=1)
        
        # 5. 根据reduction处理最终输出
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss