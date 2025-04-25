import torch
import torch.nn as nn
from mytorch.mynn import CustomTransformerEncoderLayer, CustomTransformerEncoder  # 引入自定义的 TransformerEncoderLayer

if __name__ == '__main__':
    torch.manual_seed(42)
    
    # 定义模型参数
    d_model = 300
    n_heads = 10
    dim_feedforward = 512
    dropout = 0.0
    num_layers = 2

    # 创建 PyTorch 的 TransformerEncoderLayer 实例
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

    # 创建 PyTorch 的 TransformerEncoder 实例
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    # 保存 PyTorch 的 TransformerEncoder 的 state_dict
    torch.save(transformer_encoder.state_dict(), 'transformer_encoder.pth')

    # 创建自定义的 TransformerEncoderLayer 实例
    custom_encoder_layer = CustomTransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                         dim_feedforward=dim_feedforward, dropout=dropout)

    # 创建自定义的 TransformerEncoder 实例
    custom_transformer_encoder = CustomTransformerEncoder(custom_encoder_layer, num_layers=num_layers)

    # 加载保存的 state_dict
    # custom_transformer_encoder.load_state_dict(torch.load('transformer_encoder.pth', weights_only=True))
    custom_transformer_encoder.load_state_dict(transformer_encoder.state_dict())

    # 打印 state_dict 以验证命名一致
    print("Custom TransformerEncoder state_dict:")
    for key, value in custom_transformer_encoder.state_dict().items():
        print(f"{key}: {value.shape}")

    print("\nnn.TransformerEncoder state_dict:")
    for key, value in transformer_encoder.state_dict().items():
        print(f"{key}: {value.shape}")

    # 随机生成输入数据
    # input_data = torch.randn(64, 64, 300)
    input_data = torch.load("embedded_in_transformer.pt", weights_only=True)['embedded']
    print("input_data shape", input_data.shape)

    # 使用 PyTorch 的 TransformerEncoder
    output = transformer_encoder(input_data)

    # 使用自定义的 TransformerEncoder
    custom_output = custom_transformer_encoder(input_data)

    # 比较结果
    # print("output", output)
    # print("custom_output", custom_output)
    print("\nOutput close:", torch.allclose(custom_output, output, atol=1e-4))
    print(torch.max(torch.abs(custom_output - output)))
