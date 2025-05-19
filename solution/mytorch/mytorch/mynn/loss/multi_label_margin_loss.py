import torch
from torch import nn

class CustomMultiLabelMarginLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomMultiLabelMarginLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        # 确保pred和target形状相同
        assert pred.size() == target.size()
        
        batch_size = pred.size(0)
        losses = []
        
        for i in range(batch_size):
            # 获取当前样本的预测和标签
            x = pred[i]
            y = target[i]
            
            # 找到第一个-1的位置，只考虑之前的标签
            first_neg = (y == -1).nonzero()
            if first_neg.numel() > 0:
                first_neg = first_neg[0].item()
                pos_indices = y[:first_neg]
            else:
                pos_indices = y
            
            # 获取所有可能的负样本索引(不在正样本中的)
            all_indices = torch.arange(len(x))
            neg_indices = all_indices[~torch.isin(all_indices, pos_indices)]
            
            # 计算损失
            sample_loss = 0.0
            for pos in pos_indices:
                for neg in neg_indices:
                    margin = 1 - (x[pos] - x[neg])
                    sample_loss += torch.clamp(margin, min=0)
            
            # 平均损失
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                sample_loss /= (len(pos_indices) * len(neg_indices))
            
            losses.append(sample_loss)
        
        # 将所有样本的损失组合成张量
        losses = torch.stack(losses) if batch_size > 0 else torch.tensor(0.0)
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses