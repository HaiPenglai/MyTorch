import torch
from torch import nn

class CustomSoftsign(nn.Module):
    def __init__(self):
        super(CustomSoftsign, self).__init__()
        
    def forward(self, x):
        # TODO，前向传播，返回softsign(x), x / (1 + |x|)
        '''《pass》'''
        #《
        return torch.nn.functional.softsign(x)
        #》
        
        
class MySoftsignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Softsign: x / (1 + |x|)
        return input / (1 + torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Derivative: d/dx[x/(1+|x|)] = 1/(1+|x|)^2
        denominator = (1 + torch.abs(input)).pow(2)
        grad_input = grad_output / denominator
        return grad_input

class MySoftsign(nn.Module):
    def __init__(self):
        super(MySoftsign, self).__init__()
    
    def forward(self, x):
        return MySoftsignFunction.apply(x) 