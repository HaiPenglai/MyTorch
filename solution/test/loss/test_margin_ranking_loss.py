import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomMarginRankingLoss

class TestCustomMarginRankingLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x1 = torch.tensor([3.0, 1.5, 2.0])
        self.x2 = torch.tensor([2.0, 2.5, 1.0])
        self.target = torch.tensor([1, -1, 1])  # 第一个和第三个样本x1应该大于x2，第二个样本x1应该小于x2

    def test_mean_reduction(self):
        criterion = nn.MarginRankingLoss(reduction='mean')
        custom_criterion = CustomMarginRankingLoss(reduction='mean')
        
        loss_native = criterion(self.x1, self.x2, self.target)
        loss_custom = custom_criterion(self.x1, self.x2, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.MarginRankingLoss(reduction='sum')
        custom_criterion = CustomMarginRankingLoss(reduction='sum')
        
        loss_native = criterion(self.x1, self.x2, self.target)
        loss_custom = custom_criterion(self.x1, self.x2, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.MarginRankingLoss(reduction='none')
        custom_criterion = CustomMarginRankingLoss(reduction='none')
        
        loss_native = criterion(self.x1, self.x2, self.target)
        loss_custom = custom_criterion(self.x1, self.x2, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_with_margin(self):
        margin = 0.5
        criterion = nn.MarginRankingLoss(margin=margin, reduction='mean')
        custom_criterion = CustomMarginRankingLoss(margin=margin, reduction='mean')
        
        loss_native = criterion(self.x1, self.x2, self.target)
        loss_custom = custom_criterion(self.x1, self.x2, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 