import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomHuberLoss

class TestCustomHuberLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 创建预测值和目标值
        self.pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.target = torch.tensor([1.5, 2.5, 2.5, 3.5])

    def test_mean_reduction(self):
        criterion = nn.HuberLoss(delta=1.0, reduction='mean')
        custom_criterion = CustomHuberLoss(delta=1.0, reduction='mean')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.HuberLoss(delta=1.0, reduction='sum')
        custom_criterion = CustomHuberLoss(delta=1.0, reduction='sum')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.HuberLoss(delta=1.0, reduction='none')
        custom_criterion = CustomHuberLoss(delta=1.0, reduction='none')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_different_delta(self):
        delta = 0.5
        criterion = nn.HuberLoss(delta=delta, reduction='mean')
        custom_criterion = CustomHuberLoss(delta=delta, reduction='mean')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 