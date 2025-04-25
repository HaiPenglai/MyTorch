import torch
import torch.nn as nn
import math
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 定义权重和偏置
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)  # 偏置初始化为零
    
    def forward(self, x):
        output = x @ self.weight.T  # 矩阵乘法
        if self.bias is not None:
            output += self.bias      # 添加偏置
        return output
    
