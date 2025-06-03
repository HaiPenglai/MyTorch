import torch
from torch import nn

class CustomRReLU(nn.Module):
    def __init__(self, lower=1.0/8, upper=1.0/3):
        super(CustomRReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        
    def forward(self, x):
        # TODO，前向传播，返回rrelu(x), x if x >= 0 else a*x (a sampled from [lower, upper])
        '''《pass》'''
        #《
        if self.training:
            # During training: sample random slope from uniform distribution
            neg_slope = torch.empty_like(x).uniform_(self.lower, self.upper)
            return torch.where(x >= 0, x, neg_slope * x)
        else:
            # During evaluation: use fixed slope (mean of lower and upper)
            neg_slope = (self.lower + self.upper) / 2
            return torch.where(x >= 0, x, neg_slope * x)
        #》
        
        
class MyRReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lower, upper, training):
        ctx.save_for_backward(input)
        ctx.lower = lower
        ctx.upper = upper
        ctx.training = training
        
        if training:
            # During training: sample random slope from uniform distribution
            neg_slope = torch.empty_like(input).uniform_(lower, upper)
            ctx.save_for_backward(input, neg_slope)
            return torch.where(input >= 0, input, neg_slope * input)
        else:
            # During evaluation: use fixed slope (mean of lower and upper)
            neg_slope = (lower + upper) / 2
            ctx.neg_slope = neg_slope
            return torch.where(input >= 0, input, neg_slope * input)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.training:
            input, neg_slope = ctx.saved_tensors
            # Gradient: 1 if x >= 0 else the sampled negative slope
            grad_mask = torch.where(input >= 0, torch.ones_like(input), neg_slope)
        else:
            input, = ctx.saved_tensors
            neg_slope = ctx.neg_slope
            # Gradient: 1 if x >= 0 else the fixed negative slope
            grad_mask = torch.where(input >= 0, torch.ones_like(input), neg_slope)
        
        grad_input = grad_output * grad_mask
        return grad_input, None, None, None

class MyRReLU(nn.Module):
    def __init__(self, lower=1.0/8, upper=1.0/3):
        super(MyRReLU, self).__init__()
        self.lower = lower
        self.upper = upper
    
    def forward(self, x):
        return MyRReLUFunction.apply(x, self.lower, self.upper, self.training) 