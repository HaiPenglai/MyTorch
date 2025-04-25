import torch
import torch.nn as nn
from mytorch.mynn import CustomLayerNorm
from mytorch.myutils import print_state_dict

def test_layernorm():
    # 配置参数
    torch.manual_seed(42)
    normalized_shape = [10]  # 标准化维度（最后一维）
    input_shape = (2, 5, 10)  # Batch size = 2, Sequence length = 5, Feature size = 10

    # 原生 LayerNorm
    layer_norm = nn.LayerNorm(normalized_shape)
    torch.save(layer_norm.state_dict(), 'layernorm.pth')
    print("layer_norm", layer_norm)
    print_state_dict(layer_norm)

    # 自定义 LayerNorm 加载参数
    custom_layer_norm = CustomLayerNorm(normalized_shape)
    custom_layer_norm.load_state_dict(torch.load('layernorm.pth', weights_only=True))
    print("custom_layer_norm:")
    print_state_dict(custom_layer_norm)

    # 验证前向传播
    x = torch.randn(input_shape)
    output_native = layer_norm(x)
    output_custom = custom_layer_norm(x)
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)

if __name__ == '__main__':
    test_layernorm()