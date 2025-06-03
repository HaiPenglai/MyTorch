import torch
from torch import nn

class CustomSoftmin(nn.Module):
    def __init__(self, dim=None):
        super(CustomSoftmin, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        # TODO，前向传播，返回softmin(x), exp(-x_i) / sum_j(exp(-x_j))
        '''《pass》'''
        #《
        return torch.nn.functional.softmin(x, dim=self.dim)
        #》
        
        
class MySoftminFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        # Softmin: exp(-x_i) / sum_j(exp(-x_j))
        # This is equivalent to softmax(-x)
        neg_input = -input
        
        # Apply softmax to -input for numerical stability
        if dim is None:
            # Flatten and apply softmax
            neg_input_flat = neg_input.view(-1)
            max_val = neg_input_flat.max()
            shifted = neg_input_flat - max_val
            exp_vals = torch.exp(shifted)
            sum_exp = exp_vals.sum()
            output_flat = exp_vals / sum_exp
            output = output_flat.view(input.shape)
        else:
            # Apply along specified dimension
            max_vals = neg_input.max(dim=dim, keepdim=True)[0]
            shifted = neg_input - max_vals
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
        
        # Derivative of softmin is same as softmax but with negative sign on input
        # d/dx_i[softmin(x)] = softmin_i * (δ_ij - softmin_j) * (-1)
        # = -softmin_i * (δ_ij - softmin_j)
        
        if dim is None:
            # Flatten case
            output_flat = output.view(-1)
            grad_flat = grad_output.view(-1)
            sum_term = (output_flat * grad_flat).sum()
            grad_input_flat = -output_flat * (grad_flat - sum_term)
            grad_input = grad_input_flat.view(grad_output.shape)
        else:
            sum_term = (output * grad_output).sum(dim=dim, keepdim=True)
            grad_input = -output * (grad_output - sum_term)
        
        return grad_input, None

class MySoftmin(nn.Module):
    def __init__(self, dim=None):
        super(MySoftmin, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return MySoftminFunction.apply(x, self.dim) 