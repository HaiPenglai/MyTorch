import torch
from torch import nn

class CustomHardSwish(nn.Module):
    def __init__(self):
        super(CustomHardSwish, self).__init__()
        
    def forward(self, x):
        # TODO，前向传播，返回hardswish(x), x * ReLU6(x + 3) / 6
        '''《pass》'''
        #《
        return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0
        #》
        
        
class MyHardSwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # HardSwish: x * ReLU6(x + 3) / 6
        return input * torch.clamp(input + 3.0, min=0.0, max=6.0) / 6.0

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Gradient computation based on the C code: 
        # tmp = (src1[i] > 3.0f ? 1.0f : (src1[i] < -3.0f ? 0.0f : (2.0f * src1[i] + 3.0f) / 6.0f))
        grad_mask = torch.where(input > 3.0, torch.ones_like(input),
                               torch.where(input < -3.0, torch.zeros_like(input),
                                          (2.0 * input + 3.0) / 6.0))
        grad_input = grad_output * grad_mask
        return grad_input

class MyHardSwish(nn.Module):
    def __init__(self):
        super(MyHardSwish, self).__init__()
    
    def forward(self, x):
        return MyHardSwishFunction.apply(x) 