import torch
import torch.nn as nn
import math

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(CustomRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # 定义每一层的参数
        for layer in range(num_layers):
            # 输入到隐藏的权重和偏置
            setattr(self, f'weight_ih_l{layer}', nn.Parameter(torch.Tensor(hidden_size, input_size if layer == 0 else hidden_size)))
            setattr(self, f'bias_ih_l{layer}', nn.Parameter(torch.Tensor(hidden_size)))
            # 隐藏到隐藏的权重和偏置
            setattr(self, f'weight_hh_l{layer}', nn.Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, f'bias_hh_l{layer}', nn.Parameter(torch.Tensor(hidden_size)))

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            # 初始化权重和偏置
            nn.init.kaiming_uniform_(getattr(self, f'weight_ih_l{layer}'), a=math.sqrt(5))
            nn.init.kaiming_uniform_(getattr(self, f'weight_hh_l{layer}'), a=math.sqrt(5))
            nn.init.zeros_(getattr(self, f'bias_ih_l{layer}'))
            nn.init.zeros_(getattr(self, f'bias_hh_l{layer}'))

    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        hidden_size = self.hidden_size
        num_layers = self.num_layers

        # 初始化隐藏状态
        h_prev = [torch.zeros(batch_size, hidden_size) for _ in range(num_layers)]

        # 存储每一层的最终输出
        layer_outputs = []

        # 对于每一层
        for layer in range(num_layers):
            layer_input = X if layer == 0 else layer_outputs[-1]
            W_xh = getattr(self, f'weight_ih_l{layer}')
            W_hh = getattr(self, f'weight_hh_l{layer}')
            b_h = getattr(self, f'bias_ih_l{layer}') + getattr(self, f'bias_hh_l{layer}')

            outputs = []
            for t in range(seq_len):
                x_t = layer_input[:, t, :]
                h_t = torch.tanh(x_t @ W_xh.T + h_prev[layer] @ W_hh.T + b_h)
                outputs.append(h_t.unsqueeze(1))
                h_prev[layer] = h_t
            layer_outputs.append(torch.cat(outputs, dim=1))

        return layer_outputs[-1], torch.stack(h_prev, dim=0)


