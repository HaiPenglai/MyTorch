import torch
from torch import nn

class CustomMish(nn.Module):
    def __init__(self):
        super(CustomMish, self).__init__()
        
    def forward(self, x):
        # TODO，前向传播，返回mish(x), x * tanh(softplus(x))
        '''《pass》'''
        #《
        return x * torch.tanh(torch.nn.functional.softplus(x))
        #》
        
        
class MyMishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Mish: x * tanh(softplus(x))
        softplus_x = torch.nn.functional.softplus(input)
        tanh_softplus_x = torch.tanh(softplus_x)
        output = input * tanh_softplus_x
        
        # Save for backward
        ctx.save_for_backward(input, softplus_x, tanh_softplus_x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, softplus_x, tanh_softplus_x = ctx.saved_tensors
        
        # Derivative of mish: d/dx[x * tanh(softplus(x))]
        # = tanh(softplus(x)) + x * (1 - tanh²(softplus(x))) * sigmoid(x)
        sigmoid_x = torch.sigmoid(input)
        sech2_softplus = 1 - tanh_softplus_x.pow(2)  # sech²(softplus(x)) = 1 - tanh²(softplus(x))
        
        grad_input = grad_output * (tanh_softplus_x + input * sech2_softplus * sigmoid_x)
        return grad_input

class MyMish(nn.Module):
    def __init__(self):
        super(MyMish, self).__init__()
    
    def forward(self, x):
        return MyMishFunction.apply(x) 