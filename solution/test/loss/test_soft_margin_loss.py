import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomSoftMarginLoss

class TestCustomSoftMarginLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 创建预测值和目标值
        self.pred = torch.tensor([1.0, -1.0, 0.5, -0.5])
        self.target = torch.tensor([1.0, 1.0, -1.0, -1.0])

    def test_mean_reduction(self):
        criterion = nn.SoftMarginLoss(reduction='mean')
        custom_criterion = CustomSoftMarginLoss(reduction='mean')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.SoftMarginLoss(reduction='sum')
        custom_criterion = CustomSoftMarginLoss(reduction='sum')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.SoftMarginLoss(reduction='none')
        custom_criterion = CustomSoftMarginLoss(reduction='none')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_edge_cases(self):
        # 测试边界情况：预测值和目标值完全匹配
        pred = torch.tensor([1.0, -1.0])
        target = torch.tensor([1.0, -1.0])
        
        criterion = nn.SoftMarginLoss(reduction='mean')
        custom_criterion = CustomSoftMarginLoss(reduction='mean')
        
        loss_native = criterion(pred, target)
        loss_custom = custom_criterion(pred, target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 