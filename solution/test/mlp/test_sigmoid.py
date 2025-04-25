import torch
import torch.nn as nn
from mytorch.mynn import CustomSigmoid

def test_sigmoid():
    # 配置参数
    torch.manual_seed(42)
    
    # 创建输入张量
    input_tensor = torch.randn(2, 2)
    print("Input Tensor:")
    print(input_tensor)
    
    # 原生Sigmoid层
    sigmoid = nn.Sigmoid()
    output_native = sigmoid(input_tensor)
    print("\nnn.Sigmoid Output:")
    print(output_native)
    
    # 自定义Sigmoid层
    custom_sigmoid = CustomSigmoid()
    output_custom = custom_sigmoid(input_tensor)
    print("\nCustomSigmoid Output:")
    print(output_custom)
    
    # 比较结果
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)
    
    # 测试边界情况
    test_input = torch.tensor([[-100, 0], [1, 2]])
    print("\nTest Input (边界情况):")
    print(test_input)
    print("\nnn.Sigmoid Output (边界情况):")
    print(sigmoid(test_input))
    print("\nCustomSigmoid Output (边界情况):")
    print(custom_sigmoid(test_input))
    
    # 测试极端值
    extreme_input = torch.tensor([[-1000, 1000]])
    print("\nExtreme Input:")
    print(extreme_input)
    print("\nnn.Sigmoid Extreme Output:")
    print(sigmoid(extreme_input))
    print("\nCustomSigmoid Extreme Output:")
    print(custom_sigmoid(extreme_input))

if __name__ == '__main__':
    test_sigmoid()