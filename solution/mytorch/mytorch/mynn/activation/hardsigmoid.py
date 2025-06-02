import torch
from torch import nn

class CustomHardSigmoid(nn.Module):
    def __init__(self):
        super(CustomHardSigmoid, self).__init__()
        
    def forward(self, x):
        # TODO，前向传播，返回hardsigmoid(x), (x + 3).clamp(0, 6) / 6
        '''《pass》'''
        #《
        return torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0
        #》
        
        
class MyHardSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # HardSigmoid: (x + 3).clamp(0, 6) / 6
        return torch.clamp(input + 3.0, min=0.0, max=6.0) / 6.0

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Gradient is 1/6 if input is in (-3, 3), otherwise 0
        grad_input = grad_output * ((input > -3.0) & (input < 3.0)).float() / 6.0
        return grad_input

class MyHardSigmoid(nn.Module):
    def __init__(self):
        super(MyHardSigmoid, self).__init__()
    
    def forward(self, x):
        return MyHardSigmoidFunction.apply(x) 