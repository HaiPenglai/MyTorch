import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomNLLLoss

class TestCustomNLLLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 创建log_softmax输出
        self.pred = torch.tensor([[-1.2, -0.9], [-2.1, -0.3]])
        self.target = torch.tensor([1, 0])

    def test_mean_reduction(self):
        criterion = nn.NLLLoss(reduction='mean')
        custom_criterion = CustomNLLLoss(reduction='mean')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.NLLLoss(reduction='sum')
        custom_criterion = CustomNLLLoss(reduction='sum')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.NLLLoss(reduction='none')
        custom_criterion = CustomNLLLoss(reduction='none')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 