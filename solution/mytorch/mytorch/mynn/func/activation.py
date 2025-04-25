import torch
import torch.nn as nn

class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()
        
    def forward(self, x):
        # 大于0的元素保持不变，小于0的元素置为0
        return x * (x > 0).float()
    
    
class CustomSigmoid(nn.Module):
    def __init__(self):
        super(CustomSigmoid, self).__init__()
        
    def forward(self, x):
        # Sigmoid函数实现: 1 / (1 + exp(-x))
        return 1 / (1 + torch.exp(-x))
    
    
class CustomSoftmax(nn.Module):
    def __init__(self, dim=None):
        super(CustomSoftmax, self).__init__()
        self.dim = dim if dim is not None else -1  # 默认最后一个维度
        
    def forward(self, x):
        # 数值稳定性的改进：减去最大值
        max_vals = torch.max(x, dim=self.dim, keepdim=True).values
        exp_input = torch.exp(x - max_vals)  # 减去最大值防止数值溢出
        sum_exp = torch.sum(exp_input, dim=self.dim, keepdim=True)
        return exp_input / sum_exp


class CustomTanh(nn.Module):
    def __init__(self):
        super(CustomTanh, self).__init__()
        
    def forward(self, x):
        exp_pos = torch.exp(x)
        exp_neg = torch.exp(-x)
        return (exp_pos - exp_neg) / (exp_pos + exp_neg)
    

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
        


class CustomNorm(nn.Module):
    def __init__(self, p=2, dim=None, keepdim=False):
        super(CustomNorm, self).__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim
        
    def forward(self, x):
        if self.p == float('inf'):
            # 无穷范数 (最大绝对值)
            return torch.max(torch.abs(x), dim=self.dim, keepdim=self.keepdim).values
        elif self.p == -float('inf'):
            # 负无穷范数 (最小绝对值)
            return torch.min(torch.abs(x), dim=self.dim, keepdim=self.keepdim).values
        elif self.p == 0:
            # 0范数 (非零元素个数)，转换为与输入相同的类型
            if self.dim is not None:
                raise ValueError("0-norm does not support dim argument")
            return torch.sum(x != 0).to(x.dtype)  # 转换为输入的类型
        elif self.p == 1:
            # 1范数 (绝对值之和)
            return torch.sum(torch.abs(x), dim=self.dim, keepdim=self.keepdim)
        elif self.p == 2:
            # 2范数 (欧几里得范数)
            return torch.sqrt(torch.sum(torch.pow(x, 2), dim=self.dim, keepdim=self.keepdim))
        else:
            # 一般p范数
            return torch.pow(torch.sum(torch.pow(torch.abs(x), self.p), 
                                     dim=self.dim, keepdim=self.keepdim), 1/self.p)