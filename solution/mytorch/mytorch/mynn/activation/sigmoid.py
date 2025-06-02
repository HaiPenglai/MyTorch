import torch
from torch import nn

class CustomSigmoid(nn.Module):
    def __init__(self):
        super(CustomSigmoid, self).__init__()
        
    def forward(self, x):
        # TODO, sigmoid前向传播, 提示：torch当中自带exp
        '''《pass》'''
        #《
        return 1 / (1 + torch.exp(-x))
        #》
        
class MySigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = 1 / (1 + torch.exp(-input))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = grad_output * (output * (1 - output))
        return grad_input

class MySigmoid(nn.Module):
    def __init__(self):
        super(MySigmoid, self).__init__()
    
    def forward(self, x):
        return MySigmoidFunction.apply(x)