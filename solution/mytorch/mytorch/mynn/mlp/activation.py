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



