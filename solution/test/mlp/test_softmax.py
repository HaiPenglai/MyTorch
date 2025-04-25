import torch
import torch.nn as nn
from mytorch.mynn import CustomSoftmax

def test_softmax():
    # 测试数据
    input_tensor = torch.tensor([[-1, 0, 1], [0, 1, 2]]).float()
    print("Input Tensor:")
    print(input_tensor)
    
    # 测试dim=1的情况
    print("\nTesting dim=1:")
    # 原生Softmax
    softmax = nn.Softmax(dim=1)
    output_native = softmax(input_tensor)
    print("\nnn.Softmax Output (dim=1):")
    print(output_native)
    
    # 自定义Softmax
    custom_softmax = CustomSoftmax(dim=1)
    output_custom = custom_softmax(input_tensor)
    print("\nCustomSoftmax Output (dim=1):")
    print(output_custom)
    
    # 比较结果
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)
    
    # 测试dim=0的情况
    print("\nTesting dim=0:")
    # 原生Softmax
    softmax = nn.Softmax(dim=0)
    output_native = softmax(input_tensor)
    print("\nnn.Softmax Output (dim=0):")
    print(output_native)
    
    # 自定义Softmax
    custom_softmax = CustomSoftmax(dim=0)
    output_custom = custom_softmax(input_tensor)
    print("\nCustomSoftmax Output (dim=0):")
    print(output_custom)
    
    # 比较结果
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)
    
    # 测试极端值
    print("\nTesting Extreme Values:")
    extreme_input = torch.tensor([[1000, 1001, 1002]]).float()
    print("Input:", extreme_input)
    
    # 原生Softmax
    output_native = nn.Softmax(dim=1)(extreme_input)
    print("\nnn.Softmax Output:")
    print(output_native)
    
    # 自定义Softmax
    output_custom = CustomSoftmax(dim=1)(extreme_input)
    print("\nCustomSoftmax Output:")
    print(output_custom)
    
    # 比较结果
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)

if __name__ == '__main__':
    test_softmax()