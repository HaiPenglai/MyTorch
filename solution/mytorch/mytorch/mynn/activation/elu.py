import torch
from torch import nn

class CustomELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(CustomELU, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        # TODO，前向传播，返回elu(x), x if x > 0 else alpha * (exp(x) - 1)
        '''《pass》'''
        #《
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))
        #》
        
        
class MyELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        # ELU: x if x > 0 else alpha * (exp(x) - 1)
        return torch.where(input > 0, input, alpha * (torch.exp(input) - 1))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        # Gradient: 1 if x > 0 else alpha * exp(x)
        # From C code: dst[i] = (src1[i] > 0.0f ? src0[i] : alpha * expm1(src1[i]) * src0[i])
        # But expm1(x) = exp(x) - 1, so derivative is alpha * exp(x)
        grad_mask = torch.where(input > 0, torch.ones_like(input), alpha * torch.exp(input))
        grad_input = grad_output * grad_mask
        return grad_input, None

class MyELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(MyELU, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return MyELUFunction.apply(x, self.alpha) 