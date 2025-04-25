import torch
import torch.nn as nn
from mytorch.mynn import CustomMultiheadAttention

if __name__ == '__main__':
    # 验证与官方实现的兼容性
    torch.manual_seed(42)
    embed_dim = 6
    num_heads = 2
    
    # 创建对比模型（设置 batch_first=True）
    official_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0, batch_first=True)
    custom_attn = CustomMultiheadAttention(embed_dim, num_heads)
    
    torch.save(official_attn.state_dict(), "transformer_multihead_attention.pth")
    
    # 参数拷贝（确保投影矩阵正确对应）
    custom_attn.load_state_dict(torch.load("transformer_multihead_attention.pth", weights_only=True))

    # 测试数据（自注意力）
    X = torch.randn(2, 3, 6)  # [batch, seq, feat]
    Xq = Xk = Xv = X

    # 官方实现（自注意力）
    official_output, official_attn_weights = official_attn(Xq, Xk, Xv)

    # 自定义实现（自注意力）
    custom_output, custom_attn_weights = custom_attn(Xq, Xk, Xv)

    # 比较自注意力结果
    print("自注意力测试结果：")
    print("官方实现 output 形状:", official_output.shape)
    print("自定义实现 output 形状:", custom_output.shape)
    print("官方实现 attn_weights 形状:", official_attn_weights.shape)
    print("自定义实现 attn_weights 形状:", custom_attn_weights.shape)

    # 比较 output
    output_diff = torch.max(torch.abs(custom_output - official_output))
    print("\noutput 最大差异:", output_diff)
    print("output 结果一致:", torch.allclose(custom_output, official_output, atol=1e-5))

    # 比较 attn_weights
    attn_diff = torch.max(torch.abs(custom_attn_weights - official_attn_weights))
    print("\nattn_weights 最大差异:", attn_diff)
    print("attn_weights 结果一致:", torch.allclose(custom_attn_weights, official_attn_weights, atol=1e-5))

    # 测试交叉注意力
    Xq = torch.randn(2, 3, 6)
    Xk = torch.randn(2, 5, 6)  # 不同序列长度
    Xv = Xk

    # 官方实现（交叉注意力）
    official_cross_output, official_cross_attn_weights = official_attn(Xq, Xk, Xv)

    # 自定义实现（交叉注意力）
    custom_cross_output, custom_cross_attn_weights = custom_attn(Xq, Xk, Xv)

    # 比较交叉注意力结果
    print("\n交叉注意力测试结果：")
    print("官方实现 output 形状:", official_cross_output.shape)
    print("自定义实现 output 形状:", custom_cross_output.shape)
    print("官方实现 attn_weights 形状:", official_cross_attn_weights.shape)
    print("自定义实现 attn_weights 形状:", custom_cross_attn_weights.shape)

    # 比较 output
    cross_output_diff = torch.max(torch.abs(custom_cross_output - official_cross_output))
    print("\noutput 最大差异:", cross_output_diff)
    print("output 结果一致:", torch.allclose(custom_cross_output, official_cross_output, atol=1e-5))

    # 比较 attn_weights
    cross_attn_diff = torch.max(torch.abs(custom_cross_attn_weights - official_cross_attn_weights))
    print("\nattn_weights 最大差异:", cross_attn_diff)
    print("attn_weights 结果一致:", torch.allclose(custom_cross_attn_weights, official_cross_attn_weights, atol=1e-5))