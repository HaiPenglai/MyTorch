import torch
import torch.nn as nn

class CustomTanh(nn.Module):
    def __init__(self):
        super(CustomTanh, self).__init__()
        
    def forward(self, x):
        # TODO，tanh前向传播，提示：指数先算出来摆着
        '''《pass》'''
        #《
        exp_pos = torch.exp(x)
        exp_neg = torch.exp(-x)
        return (exp_pos - exp_neg) / (exp_pos + exp_neg)
        #》



class MyTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.tanh(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = grad_output * (1 - output**2)
        return grad_input

class MyTanh(nn.Module):
    def __init__(self):
        super(MyTanh, self).__init__()
    
    def forward(self, x):
        return MyTanhFunction.apply(x)