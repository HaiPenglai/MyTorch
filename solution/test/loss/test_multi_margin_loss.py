import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomMultiMarginLoss

class TestCustomMultiMarginLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input = torch.tensor([[1.2, 0.5, 3.0], [0.8, 2.1, -0.3]])
        self.target = torch.tensor([2, 1])  # 第一个样本的正确类别是2，第二个样本的正确类别是1

    def test_mean_reduction(self):
        criterion = nn.MultiMarginLoss(reduction='mean')
        custom_criterion = CustomMultiMarginLoss(reduction='mean')
        
        loss_native = criterion(self.input, self.target)
        loss_custom = custom_criterion(self.input, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.MultiMarginLoss(reduction='sum')
        custom_criterion = CustomMultiMarginLoss(reduction='sum')
        
        loss_native = criterion(self.input, self.target)
        loss_custom = custom_criterion(self.input, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.MultiMarginLoss(reduction='none')
        custom_criterion = CustomMultiMarginLoss(reduction='none')
        
        loss_native = criterion(self.input, self.target)
        loss_custom = custom_criterion(self.input, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_with_margin_and_p(self):
        margin = 1.0
        p = 2
        criterion = nn.MultiMarginLoss(margin=margin, p=p, reduction='mean')
        custom_criterion = CustomMultiMarginLoss(margin=margin, p=p, reduction='mean')
        
        loss_native = criterion(self.input, self.target)
        loss_custom = custom_criterion(self.input, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_with_weight(self):
        weight = torch.tensor([1.0, 2.0, 1.0])
        criterion = nn.MultiMarginLoss(weight=weight, reduction='mean')
        custom_criterion = CustomMultiMarginLoss(weight=weight, reduction='mean')
        
        loss_native = criterion(self.input, self.target)
        loss_custom = custom_criterion(self.input, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 