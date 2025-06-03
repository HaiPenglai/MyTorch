import torch
from torch import nn

class CustomThreshold(nn.Module):
    def __init__(self, threshold=0.0, value=0.0):
        super(CustomThreshold, self).__init__()
        self.threshold = threshold
        self.value = value
        
    def forward(self, x):
        # TODO，前向传播，返回threshold(x), x if x > threshold else value
        '''《pass》'''
        #《
        return torch.where(x > self.threshold, x, self.value)
        #》
        
        
class MyThresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, value):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.value = value
        # Threshold: x if x > threshold else value
        return torch.where(input > threshold, input, value)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        # Gradient: 1 if x > threshold else 0
        grad_mask = (input > threshold).float()
        grad_input = grad_output * grad_mask
        return grad_input, None, None

class MyThreshold(nn.Module):
    def __init__(self, threshold=0.0, value=0.0):
        super(MyThreshold, self).__init__()
        self.threshold = threshold
        self.value = value
    
    def forward(self, x):
        return MyThresholdFunction.apply(x, self.threshold, self.value) 