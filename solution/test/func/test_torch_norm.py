import torch
from mytorch.mynn import CustomNorm

def test_norm():
    # 测试数据
    tensor = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    print("Input Tensor:")
    print(tensor)
    
    # 测试各种范数
    norms_to_test = [0, 1, 2, float('inf'), -float('inf'), 1.5]
    
    for p in norms_to_test:
        print(f"\nTesting p={p} norm:")
        
        # 原生torch.norm
        try:
            native_norm = torch.norm(tensor, p=p)
            print(f"torch.norm (p={p}):", native_norm)
        except Exception as e:
            print(f"torch.norm (p={p}) failed:", str(e))
            native_norm = None
        
        # 自定义norm
        try:
            custom_norm = CustomNorm(p=p)(tensor)
            print(f"CustomNorm (p={p}):", custom_norm)
        except Exception as e:
            print(f"CustomNorm (p={p}) failed:", str(e))
            custom_norm = None
        
        # 比较结果
        if native_norm is not None and custom_norm is not None:
            is_close = torch.allclose(custom_norm, native_norm, atol=1e-6)
            print("结果是否一致:", is_close)
    
    # 测试带维度的范数
    print("\nTesting with dim=1:")
    p = 2
    native_norm = torch.norm(tensor, p=p, dim=1)
    custom_norm = CustomNorm(p=p, dim=1)(tensor)
    print("torch.norm:", native_norm)
    print("CustomNorm:", custom_norm)
    print("结果是否一致:", torch.allclose(custom_norm, native_norm, atol=1e-6))
    
    # 测试keepdim
    print("\nTesting with keepdim=True:")
    native_norm = torch.norm(tensor, p=p, dim=1, keepdim=True)
    custom_norm = CustomNorm(p=p, dim=1, keepdim=True)(tensor)
    print("torch.norm:", native_norm)
    print("CustomNorm:", custom_norm)
    print("结果是否一致:", torch.allclose(custom_norm, native_norm, atol=1e-6))

if __name__ == '__main__':
    test_norm()