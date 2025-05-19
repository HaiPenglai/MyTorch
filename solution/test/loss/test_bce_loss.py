import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomBCELoss

class TestCustomBCELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 创建测试数据：2个样本，每个样本3个类别的概率值
        self.probs = torch.tensor([[0.9, 0.2, 0.1], 
                                 [0.6, 0.8, 0.3]])
        # 创建对应的目标值（0-1之间的值）
        self.targets = torch.tensor([[0.8, 0.2, 0.1],
                                   [0.3, 0.9, 0.4]])

    def test_mean_reduction(self):
        criterion = nn.BCELoss(reduction='mean')
        custom_criterion = CustomBCELoss(reduction='mean')
        
        loss_native = criterion(self.probs, self.targets)
        loss_custom = custom_criterion(self.probs, self.targets)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.BCELoss(reduction='sum')
        custom_criterion = CustomBCELoss(reduction='sum')
        
        loss_native = criterion(self.probs, self.targets)
        loss_custom = custom_criterion(self.probs, self.targets)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.BCELoss(reduction='none')
        custom_criterion = CustomBCELoss(reduction='none')
        
        loss_native = criterion(self.probs, self.targets)
        loss_custom = custom_criterion(self.probs, self.targets)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_edge_cases(self):
        # 测试边界情况（概率为0或1）
        custom_criterion = CustomBCELoss(reduction='mean')
        
        # 概率接近0的情况
        probs = torch.tensor([[1e-7, 0.5]])
        targets = torch.tensor([[1.0, 0.5]])
        loss = custom_criterion(probs, targets)
        self.assertFalse(torch.isnan(loss))
        
        # 概率接近1的情况
        probs = torch.tensor([[1-1e-7, 0.5]])
        targets = torch.tensor([[0.0, 0.5]])
        loss = custom_criterion(probs, targets)
        self.assertFalse(torch.isnan(loss))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)