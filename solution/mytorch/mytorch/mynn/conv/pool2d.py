import torch
import torch.nn as nn

class CustomMaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(CustomMaxPool2d, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        
    def forward(self, x):
        # 提取输入特征图的维度
        batch_size, channels, in_height, in_width = x.shape

        # 计算输出特征图的维度
        out_height = (in_height - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (in_width - self.kernel_size[1]) // self.stride[1] + 1

        # 初始化输出特征图
        output = torch.zeros((batch_size, channels, out_height, out_width), device=x.device)

        # 执行最大池化操作
        for i in range(batch_size):
            for j in range(channels):
                for k in range(out_height):
                    for l in range(out_width):
                        h_start = k * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = l * self.stride[1]
                        w_end = w_start + self.kernel_size[1]

                        # 在当前池化窗口中提取最大值
                        window = x[i, j, h_start:h_end, w_start:w_end]
                        output[i, j, k, l] = torch.max(window)

        return output