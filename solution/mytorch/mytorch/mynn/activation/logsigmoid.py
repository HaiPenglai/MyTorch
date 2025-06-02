import torch
from torch import nn

class CustomLogSigmoid(nn.Module):
    def __init__(self):
        super(CustomLogSigmoid, self).__init__()
        
    def forward(self, x):
        # TODO，前向传播，返回logsigmoid(x), log(sigmoid(x)) = -log(1 + exp(-x))
        '''《pass》'''
        #《
        return -torch.log(1.0 + torch.exp(-x))
        #》
        
        
class MyLogSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # For numerical stability, use torch.nn.functional.logsigmoid approach
        # logsigmoid(x) = -softplus(-x) = -log(1 + exp(-x))
        # But for large positive x, use x - softplus(x) = x - log(1 + exp(x))
        neg_input = -input
        ctx.save_for_backward(input, neg_input)
        return -torch.log(1.0 + torch.exp(neg_input))

    @staticmethod
    def backward(ctx, grad_output):
        input, neg_input = ctx.saved_tensors
        # Gradient of logsigmoid(x) is sigmoid(-x) = 1/(1 + exp(x))
        grad_input = grad_output / (1.0 + torch.exp(input))
        return grad_input

class MyLogSigmoid(nn.Module):
    def __init__(self):
        super(MyLogSigmoid, self).__init__()
    
    def forward(self, x):
        return MyLogSigmoidFunction.apply(x) 