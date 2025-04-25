import torch
import torch.nn as nn
from mytorch.mynn import CustomConv1d

def test_conv1d():
    # 配置参数
    torch.manual_seed(42)
    in_channels, out_channels = 300, 64
    kernel_size = 3
    batch_size = 2
    width = 100

    # 原生Conv1d层
    conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1)
    torch.save(conv1d.state_dict(), 'conv1d.pth')
    print("conv1d state_dict:")
    print("weight:", conv1d.weight.data.size())
    print("bias:", conv1d.bias.data.size() if conv1d.bias is not None else None)

    # 自定义Conv1d层加载参数
    custom_conv1d = CustomConv1d(in_channels, out_channels, kernel_size, stride=1, padding=1)
    custom_conv1d.load_state_dict(torch.load('conv1d.pth', weights_only=True))
    print("\ncustom_conv1d state_dict:")
    print("weight:", custom_conv1d.weight.data.size())
    print("bias:", custom_conv1d.bias.data.size() if custom_conv1d.bias is not None else None)

    # 验证前向传播
    x = torch.randn(batch_size, in_channels, width)
    output_native = conv1d(x)
    output_custom = custom_conv1d(x)
    is_close = torch.allclose(output_custom, output_native, atol=1e-4)
    print("\n前向传播结果是否一致:", is_close)

    # 打印部分结果对比
    print("\nnn.Conv1d Output (first few elements):")
    print(output_native[0, 0, :5])
    print("\nCustomConv1d Output (first few elements):")
    print(output_custom[0, 0, :5])

if __name__ == '__main__':
    test_conv1d()