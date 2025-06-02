import torch
from torch import nn

class CustomReLU6(nn.Module):
    def __init__(self):
        super(CustomReLU6, self).__init__()
        
    def forward(self, x):
        # TODO，前向传播，返回relu6(x), clip values between 0 and 6
        '''《pass》'''
        #《
        return torch.clamp(x, min=0, max=6)
        #》
        
        
class MyReLU6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # ReLU6: clip values between 0 and 6
        return torch.clamp(input, min=0, max=6)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Gradient is grad_output if input is in (0, 6], otherwise 0
        grad_input = grad_output * ((input > 0) & (input <= 6)).float()
        return grad_input

class MyReLU6(nn.Module):
    def __init__(self):
        super(MyReLU6, self).__init__()
    
    def forward(self, x):
        return MyReLU6Function.apply(x) 