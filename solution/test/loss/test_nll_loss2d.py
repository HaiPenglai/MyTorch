import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomNLLLoss2d

class TestCustomNLLLoss2d(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 创建2x2图像的log_softmax输出 (batch_size=2, channels=2, height=2, width=2)
        self.pred = torch.tensor([
            [[[-0.5, -1.1], [-0.2, -0.8]], [[-1.3, -0.4], [-0.9, -0.1]]],
            [[[-0.7, -0.3], [-1.2, -0.6]], [[-0.8, -0.2], [-1.1, -0.4]]]
        ])
        # 创建对应的目标标签 (batch_size=2, height=2, width=2)
        self.target = torch.tensor([
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]]
        ])

    def test_mean_reduction(self):
        criterion = nn.NLLLoss2d(reduction='mean')
        custom_criterion = CustomNLLLoss2d(reduction='mean')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.NLLLoss2d(reduction='sum')
        custom_criterion = CustomNLLLoss2d(reduction='sum')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.NLLLoss2d(reduction='none')
        custom_criterion = CustomNLLLoss2d(reduction='none')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 