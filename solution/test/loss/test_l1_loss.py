import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomL1Loss

class TestCustomL1Loss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.pred = torch.tensor([[3.5, 7.2, 1.6], [2.1, 4.3, 0.8]])
        self.target = torch.tensor([[2.9, 5.3, 0.9], [2.0, 4.0, 1.0]])

    def test_mean_reduction(self):
        criterion = nn.L1Loss(reduction='mean')
        custom_criterion = CustomL1Loss(reduction='mean')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.L1Loss(reduction='sum')
        custom_criterion = CustomL1Loss(reduction='sum')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.L1Loss(reduction='none')
        custom_criterion = CustomL1Loss(reduction='none')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 