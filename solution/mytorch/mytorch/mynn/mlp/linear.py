import torch
import torch.nn as nn
import math
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # TODO，创建一个可训练的权重矩阵W，注意X@W.T要可以运算
        '''《pass》'''
        #《
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        #》
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        # TODO使用何凯明均匀初始化, 原地修改
        '''《pass》'''
        #《
        nn.init.kaiming_uniform_(self.weight)
        #》
        if self.bias is not None:
            nn.init.zeros_(self.bias)  # 偏置初始化为零
    
    def forward(self, x):
        # TODO进行X@W.T， 如果没有偏置就到此为止了
        '''《
        output = None
        》'''
        #《
        output = x @ self.weight.T
        #》
        if self.bias is not None:
            # TODO加上偏置
            '''《pass》'''
            #《
            output += self.bias
            #》
        return output

class MyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.t(), input)
        grad_bias = grad_output.sum(dim=0) if bias is not None else None
        return grad_input, grad_weight, grad_bias

class MyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(MyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return MyLinearFunction.apply(x, self.weight, self.bias)