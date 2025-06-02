import torch
from torch import nn

class CustomSoftplus(nn.Module):
    def __init__(self, beta=1.0, threshold=20.0):
        super(CustomSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold
        
    def forward(self, x):
        # TODO，前向传播，返回softplus(x), log(1 + exp(beta * x)) / beta
        '''《pass》'''
        #《
        scaled_x = self.beta * x
        return torch.where(scaled_x > self.threshold, x, torch.log(1.0 + torch.exp(scaled_x)) / self.beta)
        #》
        
        
class MySoftplusFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, beta, threshold):
        scaled_input = beta * input
        ctx.save_for_backward(input, scaled_input)
        ctx.beta = beta
        ctx.threshold = threshold
        # Softplus: log(1 + exp(beta * x)) / beta
        # For numerical stability, if beta * x > threshold, return x
        return torch.where(scaled_input > threshold, input, torch.log(1.0 + torch.exp(scaled_input)) / beta)

    @staticmethod
    def backward(ctx, grad_output):
        input, scaled_input = ctx.saved_tensors
        beta = ctx.beta
        threshold = ctx.threshold
        # Gradient: sigmoid(beta * x) = 1 / (1 + exp(-beta * x))
        # From C code: dst[i] = src0[i] / (1.0f + dst[i]); where dst[i] = exp(-src1[i])
        grad_mask = torch.where(scaled_input > threshold, torch.ones_like(input), 
                               1.0 / (1.0 + torch.exp(-scaled_input)))
        grad_input = grad_output * grad_mask
        return grad_input, None, None

class MySoftplus(nn.Module):
    def __init__(self, beta=1.0, threshold=20.0):
        super(MySoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold
    
    def forward(self, x):
        return MySoftplusFunction.apply(x, self.beta, self.threshold) 