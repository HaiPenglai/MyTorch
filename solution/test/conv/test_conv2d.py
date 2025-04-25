import torch
import torch.nn as nn
from mytorch.mynn import CustomConv2d

def test_conv2d():
    # 配置参数
    torch.manual_seed(42)
    in_channels, out_channels = 3, 2
    kernel_size = 3
    batch_size = 2
    input_size = (4, 4)

    # 原生卷积层
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
    torch.save(conv.state_dict(), 'conv.pth')
    print("conv state_dict:")
    print("weight:", conv.weight.data.size())
    print("bias:", conv.bias.data.size() if conv.bias is not None else None)

    # 自定义卷积层加载参数
    custom_conv = CustomConv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
    custom_conv.load_state_dict(torch.load('conv.pth', weights_only=True))
    print("\ncustom_conv state_dict:")
    print("weight:", custom_conv.weight.data.size())
    print("bias:", custom_conv.bias.data.size() if custom_conv.bias is not None else None)

    # 验证前向传播
    x = torch.randn(batch_size, in_channels, *input_size)
    output_native = conv(x)
    output_custom = custom_conv(x)
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)

    # 打印部分结果对比
    print("\nnn.Conv2d Output (first few elements):")
    print(output_native[0, 0, :2, :2])
    print("\nCustomConv2d Output (first few elements):")
    print(output_custom[0, 0, :2, :2])

if __name__ == '__main__':
    test_conv2d()