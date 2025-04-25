import torch
import torch.nn as nn

class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(CustomBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1) # 刚开始(x-0)/1相当于不变
            self.num_batches_tracked.zero_() # forward了多少batch

    def forward(self, x):
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            
        if self.training:
            # 计算当前batch的均值和方差
            mean = x.mean([0, *range(2, x.dim())], keepdim=True) # 特征图形状是[N, C, H, W], 那么mean, var的形状是[1, C, 1, 1]
            var = x.var([0, *range(2, x.dim())], keepdim=True, unbiased=False)
            
            if self.track_running_stats:
                # 指数移动平均更新running mean和running var, 原来的值乘以0.9
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            # 使用running mean和running var, 形状变[1, C, 1, 1]
            mean = self.running_mean.view(1, -1, *([1] * (x.dim() - 2)))
            var = self.running_var.view(1, -1, *([1] * (x.dim() - 2)))
            
        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.affine:
            # 仿射变换
            weight = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
            bias = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
            return x_normalized * weight + bias
        else:
            return x_normalized