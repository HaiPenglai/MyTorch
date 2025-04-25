import torch
import torch.nn as nn
from mytorch.mynn import CustomReLU

def test_relu():
    # 配置参数
    torch.manual_seed(42)
    
    # 创建输入张量
    input_tensor = torch.randn(2, 3, 3)
    print("Input Tensor:")
    print(input_tensor)
    
    # 原生ReLU层
    relu = nn.ReLU()
    output_native = relu(input_tensor)
    print("\nnn.ReLU Output:")
    print(output_native)
    
    # 自定义ReLU层
    custom_relu = CustomReLU()
    output_custom = custom_relu(input_tensor)
    print("\nCustomReLU Output:")
    print(output_custom)
    
    # 比较结果
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)
    
    # 测试边界情况
    test_input = torch.tensor([[1, -2, 3], [0, 4, -5]])
    print("\nTest Input (边界情况):")
    print(test_input)
    print("\nnn.ReLU Output (边界情况):")
    print(relu(test_input))
    print("\nCustomReLU Output (边界情况):")
    print(custom_relu(test_input))

if __name__ == '__main__':
    test_relu()