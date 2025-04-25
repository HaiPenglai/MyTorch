import torch
import torch.nn as nn
import math

class CustomEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(CustomEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 定义嵌入矩阵
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        scale = math.sqrt(1.0 / self.num_embeddings)
        nn.init.uniform_(self.weight, -scale, scale)  # 均匀分布初始化
    
    def forward(self, x):
        return self.weight[x]  # 直接通过索引获取嵌入向量
    
    

    def __init__(self, num_embeddings, embedding_dim):
        super(CustomEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 定义嵌入矩阵
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        scale = math.sqrt(1.0 / self.num_embeddings)
        nn.init.uniform_(self.weight, -scale, scale)
    
    def forward(self, x):
        return self.weight[x]
    