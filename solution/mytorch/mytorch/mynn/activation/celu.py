import torch
from torch import nn

class CustomCELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(CustomCELU, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        # TODO，前向传播，返回celu(x), max(0,x) + min(0, alpha * (exp(x/alpha) - 1))
        '''《pass》'''
        #《
        return torch.max(torch.zeros_like(x), x) + torch.min(torch.zeros_like(x), self.alpha * (torch.exp(x / self.alpha) - 1))
        #》
        
        
class MyCELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        # CELU: max(0,x) + min(0, alpha * (exp(x/alpha) - 1))
        return torch.max(torch.zeros_like(input), input) + torch.min(torch.zeros_like(input), alpha * (torch.exp(input / alpha) - 1))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        
        # Gradient: 1 if x > 0, else exp(x/alpha)
        grad_mask = torch.where(input > 0, torch.ones_like(input), torch.exp(input / alpha))
        grad_input = grad_output * grad_mask
        return grad_input, None

class MyCELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(MyCELU, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return MyCELUFunction.apply(x, self.alpha) 