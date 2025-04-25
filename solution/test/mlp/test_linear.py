import torch
import torch.nn as nn
from mytorch.mynn import CustomLinear
from mytorch.myutils import print_state_dict

def test_linear():
    # 配置参数
    torch.manual_seed(42)
    in_features, out_features = 5, 3
    batch_size = 2

    # 原生线性层
    linear = nn.Linear(in_features, out_features)
    torch.save(linear.state_dict(), 'linear.pth')
    print("linear state_dict:")
    print_state_dict(linear)

    # 自定义线性层加载参数
    custom_linear = CustomLinear(in_features, out_features)
    custom_linear.load_state_dict(torch.load('linear.pth', weights_only=True))
    print("custom_linear state_dict:")
    print_state_dict(custom_linear)

    # 验证前向传播
    x = torch.randn(batch_size, in_features)
    output_native = linear(x)
    output_custom = custom_linear(x)
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)

if __name__ == '__main__':
    test_linear()