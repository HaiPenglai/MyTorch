import torch
from torch import nn
import math

class CustomGELU(nn.Module):
    def __init__(self, approximate=True):
        super(CustomGELU, self).__init__()
        self.approximate = approximate
        
    def forward(self, x):
        # TODO，前向传播，返回gelu(x), 0.5 * x * (1 + tanh or erf)
        '''《pass》'''
        #《
        if self.approximate:
            # Approximate GELU: 0.5 * x * (1 + tanh((2/pi)^0.5 * (x + 0.044715*x^3)))
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        else:
            # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
        #》
        
        
class MyGELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, approximate):
        ctx.save_for_backward(input)
        ctx.approximate = approximate
        if approximate:
            # Approximate GELU: 0.5 * x * (1 + tanh((2/pi)^0.5 * (x + 0.044715*x^3)))
            return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
        else:
            # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            return 0.5 * input * (1.0 + torch.erf(input / math.sqrt(2.0)))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        approximate = ctx.approximate
        
        if approximate:
            # Approximate gradient using tanh approximation
            inner = math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))
            tanh_inner = torch.tanh(inner)
            grad_tanh = 1.0 - torch.pow(tanh_inner, 2)
            grad_inner = math.sqrt(2.0 / math.pi) * (1.0 + 3 * 0.044715 * torch.pow(input, 2))
            grad_input = grad_output * (0.5 * (1.0 + tanh_inner) + 0.5 * input * grad_tanh * grad_inner)
        else:
            # Exact gradient: from C code
            # dst[i] = src0[i] * ((0.5 * (1.0 + erf(src1[i] / 1.4142135623730951))) +
            #                     (src1[i] * exp(-0.5 * src1[i] * src1[i]) / 2.5066282746));
            sqrt_2 = 1.4142135623730951
            sqrt_2pi = 2.5066282746
            erf_term = 0.5 * (1.0 + torch.erf(input / sqrt_2))
            exp_term = input * torch.exp(-0.5 * input * input) / sqrt_2pi
            grad_input = grad_output * (erf_term + exp_term)
        
        return grad_input, None

class MyGELU(nn.Module):
    def __init__(self, approximate=True):
        super(MyGELU, self).__init__()
        self.approximate = approximate
    
    def forward(self, x):
        return MyGELUFunction.apply(x, self.approximate) 