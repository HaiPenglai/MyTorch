import torch
from torch import nn

class CustomSoftmax2d(nn.Module):
    def __init__(self):
        super(CustomSoftmax2d, self).__init__()
        
    def forward(self, x):
        # TODO，前向传播，返回softmax2d(x), applies softmax over channels for each spatial location
        '''《pass》'''
        #《
        if x.dim() not in (3, 4):
            raise ValueError(f"Softmax2d: expected input to be 3D or 4D, got {x.dim()}D instead")
        return torch.nn.functional.softmax(x, dim=-3)
        #》
        
        
class MySoftmax2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if input.dim() not in (3, 4):
            raise ValueError(f"Softmax2d: expected input to be 3D or 4D, got {input.dim()}D instead")
        
        # Apply softmax along channel dimension (-3)
        # For 3D: (C, H, W) -> softmax along dim 0
        # For 4D: (N, C, H, W) -> softmax along dim 1 (which is -3)
        dim = -3
        
        # Numerical stability: subtract max
        max_vals = input.max(dim=dim, keepdim=True)[0]
        shifted = input - max_vals
        exp_vals = torch.exp(shifted)
        sum_exp = exp_vals.sum(dim=dim, keepdim=True)
        output = exp_vals / sum_exp
        
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        dim = ctx.dim
        
        # Softmax gradient: softmax_i * (grad_i - sum(softmax_j * grad_j))
        sum_term = (output * grad_output).sum(dim=dim, keepdim=True)
        grad_input = output * (grad_output - sum_term)
        
        return grad_input

class MySoftmax2d(nn.Module):
    def __init__(self):
        super(MySoftmax2d, self).__init__()
    
    def forward(self, x):
        return MySoftmax2dFunction.apply(x) 