import torch
import torch.nn as nn
from mytorch.mynn import CustomBidirectionalLSTM

if __name__ == '__main__':
    # 创建并保存一个 nn.LSTM 模型
    torch.manual_seed(42)
    input_size = 3
    hidden_size = 10
    num_layers = 2
    lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
    torch.save(lstm.state_dict(), 'bidirectional_lstm_model.pth')

    # 创建自定义双向 LSTM 模型并加载参数
    custom_bidirectional_lstm = CustomBidirectionalLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    custom_bidirectional_lstm.load_state_dict(torch.load('bidirectional_lstm_model.pth', weights_only=True))

    # 打印 state_dict 以验证命名一致
    print("Custom Bidirectional LSTM state_dict:")
    for key, value in custom_bidirectional_lstm.state_dict().items():
        print(f"{key}: {value.shape}")

    print("\nnn.LSTM state_dict:")
    for key, value in lstm.state_dict().items():
        print(f"{key}: {value.shape}")

    # 验证前向传播结果
    input_data = torch.randn(2, 5, 3)  # Batch size = 2, Sequence length = 5, Input size = 3

    # 使用 nn.LSTM
    output, (hidden, cell) = lstm(input_data)

    # 使用自定义双向 LSTM
    custom_output, (custom_hidden, custom_cell) = custom_bidirectional_lstm(input_data)

    # 比较结果
    print("output", output)
    print("custom_output", custom_output)
    print("\nOutput close:", torch.allclose(custom_output, output, atol=1e-4))
    print("Hidden state close:", torch.allclose(custom_hidden, hidden, atol=1e-4))
    print("Cell state close:", torch.allclose(custom_cell, cell, atol=1e-4))