import torch
import torch.nn as nn
from mytorch.mynn import CustomEmbedding
from mytorch.myutils import print_state_dict

def test_embedding():
    # 配置参数
    torch.manual_seed(42)
    num_embeddings, embedding_dim = 10, 4
    batch_size, seq_len = 2, 5

    # 原生嵌入层
    embedding = nn.Embedding(num_embeddings, embedding_dim)
    torch.save(embedding.state_dict(), 'embedding.pth')
    print("embedding state_dict:")
    print_state_dict(embedding)

    # 自定义嵌入层加载参数
    custom_embedding = CustomEmbedding(num_embeddings, embedding_dim)
    custom_embedding.load_state_dict(torch.load('embedding.pth', weights_only=True))
    print("custom_embedding state_dict:")
    print_state_dict(custom_embedding)

    # 验证前向传播
    x = torch.randint(0, num_embeddings, (batch_size, seq_len))
    output_native = embedding(x)
    output_custom = custom_embedding(x)
    print("output_native", output_custom)
    print("output_custom", output_custom)
    is_close = torch.allclose(output_custom, output_native, atol=1e-6)
    print("\n前向传播结果是否一致:", is_close)

if __name__ == '__main__':
    test_embedding()