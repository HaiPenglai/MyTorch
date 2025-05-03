import torch
from torch import nn


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


class MySoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        max_vals = torch.max(input, dim=dim, keepdim=True).values
        exp_input = torch.exp(input - max_vals)
        sum_exp = torch.sum(exp_input, dim=dim, keepdim=True)
        output = exp_input / sum_exp
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        dim = ctx.dim
        grad_input = output * (grad_output - (output * grad_output).sum(dim=dim, keepdim=True))
        return grad_input, None

class MySoftmax(nn.Module):
    def __init__(self, dim=None):
        super(MySoftmax, self).__init__()
        self.dim = dim if dim is not None else -1
    
    def forward(self, x):
        return MySoftmaxFunction.apply(x, self.dim)