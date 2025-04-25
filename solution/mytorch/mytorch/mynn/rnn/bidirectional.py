import torch
import torch.nn as nn
import math

class CustomBidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(CustomBidirectionalRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # 定义每一层的参数
        for layer in range(num_layers):
            # 前向传播的参数
            setattr(self, f'weight_ih_l{layer}', nn.Parameter(torch.Tensor(hidden_size, input_size if layer == 0 else hidden_size * 2)))
            setattr(self, f'bias_ih_l{layer}', nn.Parameter(torch.Tensor(hidden_size)))
            setattr(self, f'weight_hh_l{layer}', nn.Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, f'bias_hh_l{layer}', nn.Parameter(torch.Tensor(hidden_size)))

            # 反向传播的参数
            setattr(self, f'weight_ih_l{layer}_reverse', nn.Parameter(torch.Tensor(hidden_size, input_size if layer == 0 else hidden_size * 2)))
            setattr(self, f'bias_ih_l{layer}_reverse', nn.Parameter(torch.Tensor(hidden_size)))
            setattr(self, f'weight_hh_l{layer}_reverse', nn.Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, f'bias_hh_l{layer}_reverse', nn.Parameter(torch.Tensor(hidden_size)))

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            # 初始化前向传播的权重和偏置
            nn.init.kaiming_uniform_(getattr(self, f'weight_ih_l{layer}'), a=math.sqrt(5))
            nn.init.kaiming_uniform_(getattr(self, f'weight_hh_l{layer}'), a=math.sqrt(5))
            nn.init.zeros_(getattr(self, f'bias_ih_l{layer}'))
            nn.init.zeros_(getattr(self, f'bias_hh_l{layer}'))

            # 初始化反向传播的权重和偏置
            nn.init.kaiming_uniform_(getattr(self, f'weight_ih_l{layer}_reverse'), a=math.sqrt(5))
            nn.init.kaiming_uniform_(getattr(self, f'weight_hh_l{layer}_reverse'), a=math.sqrt(5))
            nn.init.zeros_(getattr(self, f'bias_ih_l{layer}_reverse'))
            nn.init.zeros_(getattr(self, f'bias_hh_l{layer}_reverse'))

    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        hidden_size = self.hidden_size
        num_layers = self.num_layers

        # 初始化前向和反向的隐藏状态
        h_prev_forward = [torch.zeros(batch_size, hidden_size) for _ in range(num_layers)]
        h_prev_reverse = [torch.zeros(batch_size, hidden_size) for _ in range(num_layers)]

        # 存储每一层的最终输出
        layer_outputs_forward = []
        layer_outputs_reverse = []

        # 对于每一层
        for layer in range(num_layers):
            layer_input = X if layer == 0 else torch.cat([layer_outputs_forward[-1], layer_outputs_reverse[-1]], dim=-1)

            # 前向传播
            W_xh_forward = getattr(self, f'weight_ih_l{layer}')
            W_hh_forward = getattr(self, f'weight_hh_l{layer}')
            b_h_forward = getattr(self, f'bias_ih_l{layer}') + getattr(self, f'bias_hh_l{layer}')

            outputs_forward = []
            for t in range(seq_len):
                x_t = layer_input[:, t, :]
                h_t = torch.tanh(x_t @ W_xh_forward.T + h_prev_forward[layer] @ W_hh_forward.T + b_h_forward)
                outputs_forward.append(h_t.unsqueeze(1))
                h_prev_forward[layer] = h_t
            layer_outputs_forward.append(torch.cat(outputs_forward, dim=1))

            # 反向传播
            W_xh_reverse = getattr(self, f'weight_ih_l{layer}_reverse')
            W_hh_reverse = getattr(self, f'weight_hh_l{layer}_reverse')
            b_h_reverse = getattr(self, f'bias_ih_l{layer}_reverse') + getattr(self, f'bias_hh_l{layer}_reverse')

            outputs_reverse = []
            for t in range(seq_len - 1, -1, -1):
                x_t = layer_input[:, t, :]
                h_t = torch.tanh(x_t @ W_xh_reverse.T + h_prev_reverse[layer] @ W_hh_reverse.T + b_h_reverse)
                outputs_reverse.append(h_t.unsqueeze(1))
                h_prev_reverse[layer] = h_t
            layer_outputs_reverse.append(torch.cat(outputs_reverse[::-1], dim=1))

        # 将前向和反向的输出拼接起来
        final_output = torch.cat([layer_outputs_forward[-1], layer_outputs_reverse[-1]], dim=-1)

        final_hidden = torch.cat([
            torch.stack([h_prev_forward[layer] for layer in range(num_layers)], dim=0),  # 前向各层隐藏状态
            torch.stack([h_prev_reverse[layer] for layer in range(num_layers)], dim=0)   # 反向各层隐藏状态
        ], dim=0).view(2 * num_layers, batch_size, hidden_size)  # 合并为[num_layers*2, batch, hidden]

        # 完整final_hidden计算应修改为：
        hidden_forward = torch.stack(h_prev_forward, dim=0)  # shape: [num_layers, batch, hidden]
        hidden_reverse = torch.stack(h_prev_reverse, dim=0)  # shape: [num_layers, batch, hidden]

        # 将前向和反向的隐藏状态交叉拼接
        final_hidden = torch.empty(2 * num_layers, batch_size, hidden_size)
        final_hidden[0::2] = hidden_forward    # 偶数位置放前向
        final_hidden[1::2] = hidden_reverse     # 奇数位置放反向

        return final_output, final_hidden
