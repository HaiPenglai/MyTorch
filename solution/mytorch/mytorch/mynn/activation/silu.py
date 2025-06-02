import torch
from torch import nn

class CustomSiLU(nn.Module):
    def __init__(self):
        super(CustomSiLU, self).__init__()
        
    def forward(self, x):
        # TODO，前向传播，返回silu(x), x * sigmoid(x)
        '''《pass》'''
        #《
        return x * torch.sigmoid(x)
        #》
        
        
class MySiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        sigmoid_x = torch.sigmoid(input)
        ctx.save_for_backward(input, sigmoid_x)
        # SiLU/Swish: x * sigmoid(x)
        return input * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        input, sigmoid_x = ctx.saved_tensors
        # Derivative of x * sigmoid(x) is sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        # = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        grad_input = grad_output * sigmoid_x * (1.0 + input * (1.0 - sigmoid_x))
        return grad_input

class MySiLU(nn.Module):
    def __init__(self):
        super(MySiLU, self).__init__()
    
    def forward(self, x):
        return MySiLUFunction.apply(x) 