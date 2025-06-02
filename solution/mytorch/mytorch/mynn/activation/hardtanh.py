import torch
from torch import nn

class CustomHardTanh(nn.Module):
    def __init__(self, min_val=-1.0, max_val=1.0):
        super(CustomHardTanh, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        
    def forward(self, x):
        # TODO，前向传播，返回hardtanh(x), clip values between min_val and max_val
        '''《pass》'''
        #《
        return torch.clamp(x, min=self.min_val, max=self.max_val)
        #》
        
        
class MyHardTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min_val, max_val):
        ctx.save_for_backward(input)
        ctx.min_val = min_val
        ctx.max_val = max_val
        # HardTanh: clip values between min_val and max_val
        return torch.clamp(input, min=min_val, max=max_val)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        min_val = ctx.min_val
        max_val = ctx.max_val
        # Gradient is grad_output if input is in [min_val, max_val], otherwise 0
        grad_input = grad_output * ((input >= min_val) & (input <= max_val)).float()
        return grad_input, None, None

class MyHardTanh(nn.Module):
    def __init__(self, min_val=-1.0, max_val=1.0):
        super(MyHardTanh, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
    
    def forward(self, x):
        return MyHardTanhFunction.apply(x, self.min_val, self.max_val) 