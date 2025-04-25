import math
import torch
import torch.nn as nn

class DistilBertConfig:
    def __init__(
        self,
        vocab_size=119547,
        max_position_embeddings=512,
        n_layers=6,
        n_heads=12,
        dim=768,
        hidden_dim=3072,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        initializer_range=0.02,
        seq_classif_dropout=0.2,
        pad_token_id=0,
        num_labels=2,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.initializer_range = initializer_range
        self.seq_classif_dropout = seq_classif_dropout
        self.pad_token_id = pad_token_id
        self.num_labels = num_labels

class Embeddings(nn.Module):
    def __init__(self, config:DistilBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        embeddings = word_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config:DistilBertConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        
        self.q_lin = nn.Linear(config.dim, config.dim)
        self.k_lin = nn.Linear(config.dim, config.dim)
        self.v_lin = nn.Linear(config.dim, config.dim)
        self.out_lin = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        q = self.q_lin(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # 调整mask维度为 [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(1)  
            # 确保mask与scores维度匹配
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        return self.out_lin(context)


class FFN(nn.Module):
    def __init__(self, config:DistilBertConfig):
        super().__init__()
        self.lin1 = nn.Linear(config.dim, config.hidden_dim)
        self.lin2 = nn.Linear(config.hidden_dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config:DistilBertConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.output_layer_norm = nn.LayerNorm(config.dim)
        self.ffn = FFN(config)
        self.sa_layer_norm = nn.LayerNorm(config.dim)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.output_layer_norm(x + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(x)
        return self.sa_layer_norm(x + ffn_output)

class Transformer(nn.Module):
    def __init__(self, config:DistilBertConfig):
        super().__init__()
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def forward(self, x, mask=None):
        for layer in self.layer:
            x = layer(x, mask)
        return x

class DistilBertModel(nn.Module):
    def __init__(self, config:DistilBertConfig):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.transformer = Transformer(config)
        self._init_weights(config)

    def _init_weights(self, config:DistilBertConfig):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embeddings(input_ids)
        return self.transformer(embeddings, attention_mask)

class CustomDistilBert(nn.Module):
    def __init__(self, config:DistilBertConfig):
        super().__init__()
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.init_weights(config)

    def init_weights(self, config:DistilBertConfig):
        nn.init.normal_(self.classifier.weight, std=config.initializer_range)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.distilbert(input_ids, attention_mask)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return (loss, logits) if loss is not None else logits