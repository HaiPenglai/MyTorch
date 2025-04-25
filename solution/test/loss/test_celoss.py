import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomCrossEntropyLoss

class TestCustomCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
        self.targets = torch.tensor([1, 2])

    def test_mean_reduction(self):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        custom_criterion = CustomCrossEntropyLoss(reduction='mean')
        
        loss_native = criterion(self.logits, self.targets)
        loss_custom = custom_criterion(self.logits, self.targets)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.CrossEntropyLoss(reduction='sum')
        custom_criterion = CustomCrossEntropyLoss(reduction='sum')
        
        loss_native = criterion(self.logits, self.targets)
        loss_custom = custom_criterion(self.logits, self.targets)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.CrossEntropyLoss(reduction='none')
        custom_criterion = CustomCrossEntropyLoss(reduction='none')
        
        loss_native = criterion(self.logits, self.targets)
        loss_custom = custom_criterion(self.logits, self.targets)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)