import torch
from torch import nn

class CustomLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(CustomLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        
    def forward(self, x):
        # TODO，前向传播，返回leakyrelu(x), x if x > 0 else negative_slope * x
        '''《pass》'''
        #《
        return torch.where(x > 0, x, self.negative_slope * x)
        #》
        
        
class MyLeakyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, negative_slope):
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        # LeakyReLU: x if x > 0 else negative_slope * x
        return torch.where(input > 0, input, negative_slope * input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        negative_slope = ctx.negative_slope
        # Gradient: 1 if x > 0 else negative_slope
        # From C code: dst[i] = src1[i] > 0.0f ? src0[i] : alpha * src0[i];
        grad_mask = torch.where(input > 0, torch.ones_like(input), negative_slope * torch.ones_like(input))
        grad_input = grad_output * grad_mask
        return grad_input, None

class MyLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(MyLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x):
        return MyLeakyReLUFunction.apply(x, self.negative_slope) 