import torch
from torch import nn

class CustomLogSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(CustomLogSoftmax, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        # TODO，前向传播，返回logsoftmax(x), log(softmax(x)) = x - log(sum(exp(x)))
        '''《pass》'''
        #《
        # For numerical stability: x - log(sum(exp(x - max(x)))) - max(x) = x - max(x) - log(sum(exp(x - max(x))))
        max_vals = torch.max(x, dim=self.dim, keepdim=True)[0]
        shifted_x = x - max_vals
        return shifted_x - torch.log(torch.sum(torch.exp(shifted_x), dim=self.dim, keepdim=True))
        #》
        
        
class MyLogSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        # For numerical stability
        max_vals = torch.max(input, dim=dim, keepdim=True)[0]
        shifted_input = input - max_vals
        exp_shifted = torch.exp(shifted_input)
        sum_exp = torch.sum(exp_shifted, dim=dim, keepdim=True)
        log_sum_exp = torch.log(sum_exp)
        output = shifted_input - log_sum_exp
        
        # Save for backward
        softmax_output = exp_shifted / sum_exp
        ctx.save_for_backward(softmax_output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output, = ctx.saved_tensors
        dim = ctx.dim
        
        # Gradient of log_softmax is: grad_output - softmax * sum(grad_output)
        grad_sum = torch.sum(grad_output, dim=dim, keepdim=True)
        grad_input = grad_output - softmax_output * grad_sum
        return grad_input, None

class MyLogSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(MyLogSoftmax, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return MyLogSoftmaxFunction.apply(x, self.dim) 