import torch
from torch import nn

class CustomSELU(nn.Module):
    def __init__(self):
        super(CustomSELU, self).__init__()
        # Fixed constants from PyTorch source
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        
    def forward(self, x):
        # TODO，前向传播，返回selu(x), scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
        '''《pass》'''
        #《
        return self.scale * (torch.max(torch.zeros_like(x), x) + torch.min(torch.zeros_like(x), self.alpha * (torch.exp(x) - 1)))
        #》
        
        
class MySELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Fixed constants
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        ctx.scale = scale
        
        # SELU: scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
        return scale * (torch.max(torch.zeros_like(input), input) + torch.min(torch.zeros_like(input), alpha * (torch.exp(input) - 1)))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        scale = ctx.scale
        
        # Gradient: scale * (1 if x > 0, else alpha * exp(x))
        grad_mask = torch.where(input > 0, torch.ones_like(input), alpha * torch.exp(input))
        grad_input = grad_output * scale * grad_mask
        return grad_input

class MySELU(nn.Module):
    def __init__(self):
        super(MySELU, self).__init__()
    
    def forward(self, x):
        return MySELUFunction.apply(x) 