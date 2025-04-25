import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # 合并的投影矩阵（Q,K,V）
        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        self.out_proj.reset_parameters()

    def forward(self, Xq, Xk=None, Xv=None, layer_id=-1):
        # 处理默认值（自注意力情况）
        if Xk is None:
            Xk = Xq
        if Xv is None:
            Xv = Xk
    
        batch_size, seq_len_q, _ = Xq.shape
        _, seq_len_k, _ = Xk.shape
        _, seq_len_v, _ = Xv.shape
    
        # 分割投影参数
        W_q, W_k, W_v = self.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)

        # 分别投影Q/K/V, 3d*2d=>3d
        Q = F.linear(Xq, W_q, b_q)  # [batch, seq_q, embed]
        K = F.linear(Xk, W_k, b_k)  # [batch, seq_k, embed]
        V = F.linear(Xv, W_v, b_v)  # [batch, seq_v, embed]

        # 分割多头
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)
    
        # 计算注意力分数, 本质是：[seq_q, head_dim] @ [head_dim, seq_k] => [seq_q, seq_k]表示qk相似度，多批次计算
        # [batch, num_heads, seq_q, head_dim] @ [batch, num_heads, head_dim, seq_k] => [batch, num_heads, seq_q, seq_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn = F.softmax(scores, dim=-1)
    
        # 合并所有头的注意力分数（求平均）
        attn_merged = attn.mean(dim=1)  # [batch, seq_q, seq_k]
    
        # 应用注意力到V
        context = torch.matmul(attn, V)  # [batch, num_heads, seq_q, head_dim]
        context = context.transpose(1, 2).reshape(batch_size, seq_len_q, self.embed_dim)
    
        # 最终投影
        output = self.out_proj(context)
        return output, attn_merged  # 返回 output 和合并后的 attn_weights
    

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0, batch_first=True):
        super(CustomTransformerEncoderLayer, self).__init__()
        
        # 自定义多头注意力层
        self.self_attn = CustomMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout 层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.activation = F.relu

    def forward(self, src, layer_id=-1, batch_id=-1):
        # 多头注意力
        src2, _ = self.self_attn(src, src, src, layer_id=layer_id) 
        
        # print(src2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)        

        src2 = self.linear1(src)

        src2 = self.dropout(self.activation(src2))
        src2 = self.linear2(src2)

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src
    
    
class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        # 必须使用深度拷贝，否则共享权重导致报错
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, batch_id=-1):
        for id, layer in enumerate(self.layers):
            src = layer(src, layer_id=id, batch_id=batch_id)
        return src



























