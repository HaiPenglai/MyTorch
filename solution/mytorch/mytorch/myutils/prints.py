def print_state_dict(model):
    for key, tensor in model.state_dict().items():
        print(key, tensor.shape, tensor.dtype)
        

def print_params(model):
    print("=" * 25 ,"print_params", "=" * 25)
    for name, param in model.named_parameters():
        print(name)
        print(f"{param.shape}")
        print(param.view(-1)[:10].data)
        print("-" * 50)
    print("=" * 25 ,"end_print_params", "=" * 25)
    
    
def rnn_parameter_count(L, H, D): # L: RNN层数 (必须≥1), H: 隐藏层维度, D: 输入维度 (仅第0层需要)
    layer0 = D * H + H ** 2 + 2 * H
    other_layers = (L - 1) * (2 * H ** 2 + 2 * H)
    return layer0 + other_layers


def bidirectional_rnn_parameter_count(L, H, D): # L: 双向RNN层数, H: 隐藏层维度, D: 输入维度 (仅第0层需要)
    layer0 = 2 * (D*H + H**2 + 2*H)
    other_layers = (L-1) * 2 * (3*H**2 + 2*H)
    return layer0 + other_layers


def gru_parameter_count(L, H, D):
    first_layer = 3 * D * H + 3 * H**2 + 6 * H
    other_layers = (L - 1) * 6 * (H**2 + H)
    return first_layer + other_layers


def lstm_parameter_count(L, H, D):
    first_layer = 4 * D * H + 4 * H**2 + 8 * H
    other_layers = (L - 1) * (8 * H**2 + 8 * H)
    return first_layer + other_layers


def transformer_encoder_param_count(L, H, F):
    per_layer = 4*H**2 + 9*H + 2*F*H + F
    return L * per_layer


def distilbert_param_count(V, max_pos, L, H, F, num_labels):
    emb_word = V * H          # 词嵌入 
    emb_pos = max_pos * H             # 位置嵌入
    emb_norm = 2 * H                  # 层归一化参数
    total_emb = emb_word + emb_pos + emb_norm
    encoder = L * (4*H**2 + 9*H + 2*H*F + F)  # 编码器
    pre_cls = H * H + H              # 预分类层
    cls = H * num_labels + num_labels # 分类层
    total_cls = pre_cls + cls
    return total_emb + encoder + total_cls


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


