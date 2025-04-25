import torch
import torch.nn as nn
from mytorch.mynn import CustomMaxPool1d

def test_maxpool1d():
    # 配置参数
    torch.manual_seed(42)
    batch_size, channels, width = 2, 300, 100
    kernel_size, stride, padding = 2, 4, 0

    # 创建输入张量
    input_tensor = torch.randn(batch_size, channels, width)

    # 原生MaxPool1d层
    maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    output_native = maxpool(input_tensor)
    print("nn.MaxPool1d Output:")
    print("Shape:", output_native.shape)
    print("First few elements:", output_native[0, 0, :5])

    # 自定义MaxPool1d层
    custom_maxpool = CustomMaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    output_custom = custom_maxpool(input_tensor)
    print("\nCustomMaxPool1d Output:")
    print("Shape:", output_custom.shape)
    print("First few elements:", output_custom[0, 0, :5])

    # 比较结果
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)
    print("形状是否一致:", output_custom.shape == output_native.shape)

    # 打印更多比较信息
    print("\nnn.MaxPool1d Output (sample):")
    print(output_native[0, :3, :3])
    print("\nCustomMaxPool1d Output (sample):")
    print(output_custom[0, :3, :3])

if __name__ == '__main__':
    test_maxpool1d()