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
        # TODO使用何凯明均匀初始化
        '''《pass》'''
        #《
        nn.init.kaiming_uniform_(self.weight)
        #》
        if self.bias is not None:
            nn.init.zeros_(self.bias)  # 偏置初始化为零
    
    def forward(self, x):
        # TODO进行X@W.T+b
        '''《pass》'''
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
    
