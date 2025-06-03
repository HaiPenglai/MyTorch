import torch
from torch import nn

class CustomPReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super(CustomPReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.full((num_parameters,), init))
        
    def forward(self, x):
        # TODO，前向传播，返回prelu(x), max(0,x) + weight * min(0,x)
        '''《pass》'''
        #《
        return torch.nn.functional.prelu(x, self.weight)
        #》
        
        
class MyPReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        
        # PReLU: max(0,x) + weight * min(0,x)
        # Equivalent to: x if x > 0 else weight * x
        positive = torch.clamp(input, min=0)
        negative = torch.clamp(input, max=0)
        
        # Handle broadcasting for weight
        if weight.numel() == 1:
            # Single parameter for all channels
            output = positive + weight * negative
        else:
            # Per-channel parameters
            # Weight shape should match the channel dimension
            weight_shape = [1] * input.dim()
            weight_shape[1] = weight.size(0)  # Assume channel is dim 1
            weight_reshaped = weight.view(weight_shape)
            output = positive + weight_reshaped * negative
            
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        
        # Gradient w.r.t. input
        mask = (input > 0).float()
        if weight.numel() == 1:
            grad_input = grad_output * (mask + weight * (1 - mask))
        else:
            weight_shape = [1] * input.dim()
            weight_shape[1] = weight.size(0)
            weight_reshaped = weight.view(weight_shape)
            grad_input = grad_output * (mask + weight_reshaped * (1 - mask))
        
        # Gradient w.r.t. weight
        negative_mask = (input <= 0).float()
        if weight.numel() == 1:
            # Single parameter case - ensure gradient has same shape as weight [1]
            grad_weight = (grad_output * input * negative_mask).sum().view(1)
        else:
            # Per-channel case: sum over all dimensions except channel
            grad_weight = (grad_output * input * negative_mask).sum(dim=[d for d in range(input.dim()) if d != 1])
            # Ensure it has the right shape
            if grad_weight.dim() == 0:
                grad_weight = grad_weight.view(1)
            elif grad_weight.shape[0] != weight.shape[0]:
                grad_weight = grad_weight.view(weight.shape)
        
        return grad_input, grad_weight

class MyPReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super(MyPReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.full((num_parameters,), init))
    
    def forward(self, x):
        return MyPReLUFunction.apply(x, self.weight) 