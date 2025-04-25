import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # 定义权重和偏置
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # 添加填充
        input_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        # 提取输入和权重的维度
        batch_size, in_channels, in_height, in_width = input_padded.shape
        out_channels, _, kernel_height, kernel_width = self.weight.shape

        # 计算输出维度
        out_height = (in_height - kernel_height) // self.stride + 1
        out_width = (in_width - kernel_width) // self.stride + 1

        # 初始化输出
        output = torch.zeros((batch_size, out_channels, out_height, out_width), device=x.device)

        # 执行卷积操作
        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(out_height):
                    for l in range(out_width):
                        h_start = k * self.stride
                        h_end = h_start + kernel_height
                        w_start = l * self.stride
                        w_end = w_start + kernel_width
                        output[i, j, k, l] = torch.sum(
                            input_padded[i, :, h_start:h_end, w_start:w_end] * self.weight[j]
                        ) + (self.bias[j] if self.bias is not None else 0)

        return output

