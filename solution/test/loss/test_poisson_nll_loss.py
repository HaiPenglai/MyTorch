import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomPoissonNLLLoss

class TestCustomPoissonNLLLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.pred = torch.tensor([2.0, 5.0])
        self.target = torch.tensor([3.0, 4.0])

    def test_mean_reduction_log_input(self):
        criterion = nn.PoissonNLLLoss(reduction='mean', log_input=True)
        custom_criterion = CustomPoissonNLLLoss(reduction='mean', log_input=True)
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction_log_input(self):
        criterion = nn.PoissonNLLLoss(reduction='sum', log_input=True)
        custom_criterion = CustomPoissonNLLLoss(reduction='sum', log_input=True)
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction_log_input(self):
        criterion = nn.PoissonNLLLoss(reduction='none', log_input=True)
        custom_criterion = CustomPoissonNLLLoss(reduction='none', log_input=True)
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_full_loss(self):
        criterion = nn.PoissonNLLLoss(reduction='mean', log_input=True, full=True)
        custom_criterion = CustomPoissonNLLLoss(reduction='mean', log_input=True, full=True)
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 