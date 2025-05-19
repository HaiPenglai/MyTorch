import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomGaussianNLLLoss

class TestCustomGaussianNLLLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.pred_mean = torch.tensor([1.0, 2.0])
        self.pred_var = torch.tensor([0.5, 1.0])
        self.target = torch.tensor([1.2, 1.8])

    def test_mean_reduction(self):
        criterion = nn.GaussianNLLLoss(reduction='mean')
        custom_criterion = CustomGaussianNLLLoss(reduction='mean')
        
        loss_native = criterion(self.pred_mean, self.target, self.pred_var)
        loss_custom = custom_criterion(self.pred_mean, self.pred_var, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.GaussianNLLLoss(reduction='sum')
        custom_criterion = CustomGaussianNLLLoss(reduction='sum')
        
        loss_native = criterion(self.pred_mean, self.target, self.pred_var)
        loss_custom = custom_criterion(self.pred_mean, self.pred_var, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.GaussianNLLLoss(reduction='none')
        custom_criterion = CustomGaussianNLLLoss(reduction='none')
        
        loss_native = criterion(self.pred_mean, self.target, self.pred_var)
        loss_custom = custom_criterion(self.pred_mean, self.pred_var, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_full_loss(self):
        criterion = nn.GaussianNLLLoss(reduction='mean', full=True)
        custom_criterion = CustomGaussianNLLLoss(reduction='mean', full=True)
        
        loss_native = criterion(self.pred_mean, self.target, self.pred_var)
        loss_custom = custom_criterion(self.pred_mean, self.pred_var, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 