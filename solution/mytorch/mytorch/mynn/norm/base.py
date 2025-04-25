import torch
import torch.nn as nn

class CustomNorm(nn.Module):
    def __init__(self, p=2, dim=None, keepdim=False):
        super(CustomNorm, self).__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.p == float('inf'):
            # 无穷范数 (最大绝对值)
            return torch.max(torch.abs(x), dim=self.dim, keepdim=self.keepdim).values
        elif self.p == -float('inf'):
            # 负无穷范数 (最小绝对值)
            return torch.min(torch.abs(x), dim=self.dim, keepdim=self.keepdim).values
        elif self.p == 0:
            # 0范数 (非零元素个数)，转换为与输入相同的类型
            if self.dim is not None:
                raise ValueError("0-norm does not support dim argument")
            return torch.sum(x != 0).to(x.dtype)  # 转换为输入的类型
        elif self.p == 1:
            # 1范数 (绝对值之和)
            return torch.sum(torch.abs(x), dim=self.dim, keepdim=self.keepdim)
        elif self.p == 2:
            # 2范数 (欧几里得范数)
            return torch.sqrt(torch.sum(torch.pow(x, 2), dim=self.dim, keepdim=self.keepdim))
        else:
            # 一般p范数
            return torch.pow(torch.sum(torch.pow(torch.abs(x), self.p), 
                                     dim=self.dim, keepdim=self.keepdim), 1/self.p)