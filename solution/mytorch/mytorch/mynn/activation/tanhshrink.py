import torch
from torch import nn

class CustomTanhshrink(nn.Module):
    def __init__(self):
        super(CustomTanhshrink, self).__init__()
        
    def forward(self, x):
        # TODO，前向传播，返回tanhshrink(x), x - tanh(x)
        '''《pass》'''
        #《
        return torch.nn.functional.tanhshrink(x)
        #》
        
        
class MyTanhshrinkFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        tanh_input = torch.tanh(input)
        ctx.save_for_backward(tanh_input)
        # Tanhshrink: x - tanh(x)
        return input - tanh_input

    @staticmethod
    def backward(ctx, grad_output):
        tanh_input, = ctx.saved_tensors
        # Derivative: d/dx[x - tanh(x)] = 1 - sech^2(x) = 1 - (1 - tanh^2(x)) = tanh^2(x)
        grad_input = grad_output * tanh_input.pow(2)
        return grad_input

class MyTanhshrink(nn.Module):
    def __init__(self):
        super(MyTanhshrink, self).__init__()
    
    def forward(self, x):
        return MyTanhshrinkFunction.apply(x) 