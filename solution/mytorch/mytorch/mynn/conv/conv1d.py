import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 定义权重和偏置
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
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
        batch_size, in_channels, width = x.shape
        
        # 应用padding
        if self.padding > 0:
            input_padded = F.pad(x, (self.padding, self.padding), "constant", 0)
        else:
            input_padded = x

        # 计算输出宽度
        output_width = ((width + 2 * self.padding - self.kernel_size) // self.stride) + 1

        # 初始化输出张量
        output = torch.zeros(batch_size, self.out_channels, output_width, device=x.device)

        # 执行卷积操作
        for i in range(self.out_channels):
            for j in range(output_width):
                start = j * self.stride
                end = start + self.kernel_size
                # 对所有输入通道执行卷积并求和
                output[:, i, j] = torch.sum(
                    input_padded[:, :, start:end] * self.weight[i, :, :].unsqueeze(0), 
                    dim=(1, 2)
                ) + (self.bias[i] if self.bias is not None else 0)

        return output