import torch
from torch import nn

class CustomSoftShrink(nn.Module):
    def __init__(self, lambd=0.5):
        super(CustomSoftShrink, self).__init__()
        self.lambd = lambd
        
    def forward(self, x):
        # TODO，前向传播，返回softshrink(x), sign(x) * max(|x| - lambd, 0)
        '''《pass》'''
        #《
        return torch.where(x > self.lambd, x - self.lambd,
                          torch.where(x < -self.lambd, x + self.lambd, torch.zeros_like(x)))
        #》
        
        
class MySoftShrinkFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambd):
        ctx.save_for_backward(input)
        ctx.lambd = lambd
        # SoftShrink: sign(x) * max(|x| - lambd, 0)
        # From C code: dst[i] = (src[i] > lambd) ? (src[i] - lambd) : ((src[i] < neg_lambd) ? (src[i] + lambd) : (0));
        return torch.where(input > lambd, input - lambd,
                          torch.where(input < -lambd, input + lambd, torch.zeros_like(input)))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        lambd = ctx.lambd
        # Gradient: 1 if |x| > lambd else 0 (same as HardShrink)
        # From C code: dst[i] = (src1[i] >= neg_lambd && src1[i] <= lambd) ? 0 : src0[i];
        grad_mask = torch.where((input >= -lambd) & (input <= lambd), torch.zeros_like(input), torch.ones_like(input))
        grad_input = grad_output * grad_mask
        return grad_input, None

class MySoftShrink(nn.Module):
    def __init__(self, lambd=0.5):
        super(MySoftShrink, self).__init__()
        self.lambd = lambd
    
    def forward(self, x):
        return MySoftShrinkFunction.apply(x, self.lambd) 