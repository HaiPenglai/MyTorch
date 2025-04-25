import torch
import torch.nn as nn
from mytorch.mynn import CustomTanh

def test_tanh():
    # 配置参数
    torch.manual_seed(42)
    
    # 创建输入张量
    input_tensor = torch.randn(2, 3, 3)
    print("Input Tensor:")
    print(input_tensor)
    
    # 原生Tanh层
    tanh = nn.Tanh()
    output_native = tanh(input_tensor)
    print("\nnn.Tanh Output:")
    print(output_native)
    
    # 自定义Tanh层
    custom_tanh = CustomTanh()
    output_custom = custom_tanh(input_tensor)
    print("\nCustomTanh Output:")
    print(output_custom)
    
    # 比较结果
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)
    
    # 测试边界情况
    test_input = torch.tensor([[-100, -1, 0], [1, 100, 0.5]])
    print("\nTest Input (边界情况):")
    print(test_input)
    print("\nnn.Tanh Output (边界情况):")
    print(tanh(test_input))
    print("\nCustomTanh Output (边界情况):")
    print(custom_tanh(test_input))
    
    # 测试极端值
    extreme_input = torch.tensor([[-1e6, 0, 1e6]])
    print("\nExtreme Input:")
    print(extreme_input)
    print("\nnn.Tanh Extreme Output:")
    print(tanh(extreme_input))
    print("\nCustomTanh Extreme Output:")
    print(custom_tanh(extreme_input))

if __name__ == '__main__':
    test_tanh()