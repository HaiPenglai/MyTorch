import torch
import torch.nn as nn
from mytorch.mynn import CustomTransformerEncoderLayer

if __name__ == '__main__':
    torch.manual_seed(42)  
    # 定义模型参数
    d_model = 6
    n_heads = 2
    dim_feedforward = 10
    dropout = 0

    # 创建 PyTorch 的 TransformerEncoderLayer 实例
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

    # 保存 PyTorch 的 TransformerEncoderLayer 的 state_dict
    torch.save(encoder_layer.state_dict(), 'transformer_encoder_layer.pth')

    # 创建自定义的 TransformerEncoderLayer 实例
    custom_encoder_layer = CustomTransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                         dim_feedforward=dim_feedforward, dropout=dropout)

    # 加载保存的 state_dict
    custom_encoder_layer.load_state_dict(torch.load('transformer_encoder_layer.pth', weights_only=True))

    # 打印 state_dict 以验证命名一致
    print("Custom TransformerEncoderLayer state_dict:")
    for key, value in custom_encoder_layer.state_dict().items():
        print(f"{key}: {value.shape}")

    print("\nnn.TransformerEncoderLayer state_dict:")
    for key, value in encoder_layer.state_dict().items():
        print(f"{key}: {value.shape}")
        
    # 随机生成输入数据
    input_data = torch.randn(2, 3, 6)

    # 使用 PyTorch 的 TransformerEncoderLayer
    output = encoder_layer(input_data)

    # 使用自定义的 TransformerEncoderLayer
    custom_output = custom_encoder_layer(input_data)

    # 比较结果
    print("output", output)
    print("custom_output", custom_output)
    print("\nOutput close:", torch.allclose(custom_output, output, atol=1e-4))
    print(torch.max(torch.abs(custom_output - output)))
    
