import torch
import torch.nn as nn
from mytorch.mynn import CustomMaxPool2d

def test_maxpool2d():
    # 创建测试输入数据
    conv_output = torch.tensor([[[[ 0.5914, -0.8443,  0.3207,  0.3029],
                                [ 0.6956, -0.2633, -0.2755,  0.0091],
                                [ 1.0091,  0.0539, -0.4332,  0.3565],
                                [-0.0718, -0.2377,  0.0800,  0.7624]],

                               [[-0.2488, -0.2749, -1.1166, -0.2491],
                                [ 0.5504,  0.3816,  0.2963,  0.2610],
                                [-0.0412, -0.0039, -0.4768, -0.0611],
                                [ 0.7517,  0.1665, -0.2231, -0.3370]]],

                              [[[-0.2135,  0.4644, -0.2044,  0.5666],
                                [-0.0925, -0.2376, -0.2448,  0.6950],
                                [-0.0976,  0.7593, -1.6869,  1.1621],
                                [ 0.2258,  0.2534, -0.2848, -0.0522]],

                               [[-0.0054, -0.7709,  0.0086, -0.3171],
                                [ 0.6791,  0.1246, -0.1360,  0.1951],
                                [ 0.0818, -0.3583, -0.7911, -1.8213],
                                [-0.1488,  0.4026, -0.3277,  0.3289]]]])

    # 原生MaxPool2d层
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    output_native = maxpool(conv_output)
    print("nn.MaxPool2d Output:")
    print(output_native.size())
    print(output_native)

    # 自定义MaxPool2d层
    custom_maxpool = CustomMaxPool2d(kernel_size=2, stride=2)
    output_custom = custom_maxpool(conv_output)
    print("\nCustomMaxPool2d Output:")
    print(output_custom.size())
    print(output_custom)

    # 比较结果
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)

    # 打印部分结果对比
    print("\nnn.MaxPool2d Output (first few elements):")
    print(output_native[0, 0, :2, :2])
    print("\nCustomMaxPool2d Output (first few elements):")
    print(output_custom[0, 0, :2, :2])

if __name__ == '__main__':
    test_maxpool2d()