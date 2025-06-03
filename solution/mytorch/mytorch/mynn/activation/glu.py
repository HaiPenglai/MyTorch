import torch
from torch import nn

class CustomGLU(nn.Module):
    def __init__(self, dim=-1):
        super(CustomGLU, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        # TODO，前向传播，返回glu(x), GLU(a, b) = a ⊗ σ(b) where a is first half, b is second half
        '''《pass》'''
        #《
        return torch.nn.functional.glu(x, dim=self.dim)
        #》
        
        
class MyGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        # GLU: split input into two halves along dim, then a * sigmoid(b)
        # where a is first half and b is second half
        input_size = input.size(dim)
        half_size = input_size // 2
        
        # Split input into two halves
        a = input.narrow(dim, 0, half_size)
        b = input.narrow(dim, half_size, half_size)
        
        sigmoid_b = torch.sigmoid(b)
        output = a * sigmoid_b
        
        # Save for backward
        ctx.save_for_backward(a, b, sigmoid_b)
        ctx.dim = dim
        ctx.input_size = input_size
        ctx.half_size = half_size
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        a, b, sigmoid_b = ctx.saved_tensors
        dim = ctx.dim
        input_size = ctx.input_size
        half_size = ctx.half_size
        
        # Gradient w.r.t. a: sigmoid(b)
        grad_a = grad_output * sigmoid_b
        
        # Gradient w.r.t. b: a * sigmoid(b) * (1 - sigmoid(b))
        grad_b = grad_output * a * sigmoid_b * (1 - sigmoid_b)
        
        # Concatenate gradients along the split dimension
        grad_input = torch.cat([grad_a, grad_b], dim=dim)
        
        return grad_input, None

class MyGLU(nn.Module):
    def __init__(self, dim=-1):
        super(MyGLU, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return MyGLUFunction.apply(x, self.dim) 