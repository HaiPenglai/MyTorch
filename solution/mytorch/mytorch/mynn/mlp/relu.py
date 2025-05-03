import torch
from torch import nn

class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()
        
    def forward(self, x):
        # TODO，前向传播，返回relu(x), 使用大于算子要简单一点
        '''《pass》'''
        #《
        return x * (x > 0)
        #》
        
        
class MyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.maximum(torch.zeros_like(input), input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (input > 0).float()
        return grad_input

class MyReLU(nn.Module):
    def __init__(self):
        super(MyReLU, self).__init__()
    
    def forward(self, x):
        return MyReLUFunction.apply(x)