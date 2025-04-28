import torch
import torch.nn as nn

class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()
        
    def forward(self, x):
        # TODO，前向传播，返回relu(x), 使用大于算子要简单一点
        '''《pass》'''
        #《
        return x * (x > 0)
        #》

class CustomSigmoid(nn.Module):
    def __init__(self):
        super(CustomSigmoid, self).__init__()
        
    def forward(self, x):
        # TODO, sigmoid前向传播, 提示：torch当中自带exp
        '''《pass》'''
        #《
        return 1 / (1 + torch.exp(-x))
        #》
    
class CustomSoftmax(nn.Module):
    def __init__(self, dim=None):
        super(CustomSoftmax, self).__init__()
        self.dim = dim if dim is not None else -1  # 默认最后一个维度 TODO 思考为什么softmax有维度参数，但tanh\relu\sigmoid没有【因为softmax通常是某一个维度内部做】
        
    def forward(self, x):
        # TODO，把下面的不稳定sigmoid改成稳定版本
        '''《
        exp_input = torch.exp(x)
        sum_exp = torch.sum(exp_input, dim=self.dim, keepdim=True) # TODO问题:为什么要keepdim?【因为要指定广播方向】
        return exp_input / sum_exp
        》'''
        #《
        max_vals = torch.max(x, dim=self.dim, keepdim=True).values
        exp_input = torch.exp(x - max_vals)
        sum_exp = torch.sum(exp_input, dim=self.dim, keepdim=True)
        return exp_input / sum_exp
        #》

class CustomTanh(nn.Module):
    def __init__(self):
        super(CustomTanh, self).__init__()
        
    def forward(self, x):
        # TODO，tanh前向传播，提示：指数先算出来摆着
        '''《pass》'''
        #《
        exp_pos = torch.exp(x)
        exp_neg = torch.exp(-x)
        return (exp_pos - exp_neg) / (exp_pos + exp_neg)
        #》



