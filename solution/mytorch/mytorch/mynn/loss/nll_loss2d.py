import torch
from torch import nn

class CustomNLLLoss2d(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomNLLLoss2d, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        # pred应该是log_softmax的输出，形状为(N, C, H, W)
        # target是类别索引，形状为(N, H, W)
        N, C, H, W = pred.size()
        
        # 重塑输入以匹配PyTorch的实现
        pred = pred.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
        pred = pred.view(-1, C)  # (N*H*W, C)
        target = target.view(-1)  # (N*H*W)
        
        # 计算每个像素的损失
        loss = -pred[torch.arange(pred.size(0)), target]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss.view(N, H, W) 