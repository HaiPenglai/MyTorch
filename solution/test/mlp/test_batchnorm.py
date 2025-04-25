import torch
import torch.nn as nn
from mytorch.mynn import CustomBatchNorm
from mytorch.myutils import print_state_dict

def test_batchnorm():
    # 配置参数
    torch.manual_seed(42)
    num_features = 10  # 特征维度
    input_shape = (2, 10, 5, 5)  # Batch size = 2, Channels = 10, Height = 5, Width = 5

    # 原生 BatchNorm
    batch_norm = nn.BatchNorm2d(num_features)
    torch.save(batch_norm.state_dict(), 'batchnorm.pth')
    print("batch_norm:", batch_norm)
    print_state_dict(batch_norm)

    # 自定义 BatchNorm 加载参数
    custom_batch_norm = CustomBatchNorm(num_features)
    custom_batch_norm.load_state_dict(torch.load('batchnorm.pth', weights_only=True))
    print("\ncustom_batch_norm:")
    print_state_dict(custom_batch_norm)

    # 验证前向传播
    x = torch.randn(input_shape)
    
    # 训练模式
    for _ in range(3):
        batch_norm.train()
        custom_batch_norm.train()
        output_native_train = batch_norm(x)
        output_custom_train = custom_batch_norm(x)
        is_close_train = torch.allclose(output_custom_train, output_native_train, atol=1e-6)
        print("训练模式前向传播结果是否一致:", is_close_train)
        print(custom_batch_norm.num_batches_tracked)
    
    # 评估模式
    batch_norm.eval()
    custom_batch_norm.eval()
    output_native_eval = batch_norm(x)
    output_custom_eval = custom_batch_norm(x)
    is_close_eval = torch.allclose(output_custom_eval, output_native_eval, atol=1e-6)
    print("评估模式前向传播结果是否一致:", is_close_eval)

if __name__ == '__main__':
    test_batchnorm()