import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomCosineEmbeddingLoss

class TestCustomCosineEmbeddingLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x1 = torch.tensor([[1.0, 0.5], [0.8, 0.3]])
        self.x2 = torch.tensor([[0.8, 0.3], [0.4, 0.9]])
        self.target = torch.tensor([1, -1])  # 第一个样本相似，第二个样本不相似

    def test_mean_reduction(self):
        criterion = nn.CosineEmbeddingLoss(reduction='mean')
        custom_criterion = CustomCosineEmbeddingLoss(reduction='mean')
        
        loss_native = criterion(self.x1, self.x2, self.target)
        loss_custom = custom_criterion(self.x1, self.x2, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.CosineEmbeddingLoss(reduction='sum')
        custom_criterion = CustomCosineEmbeddingLoss(reduction='sum')
        
        loss_native = criterion(self.x1, self.x2, self.target)
        loss_custom = custom_criterion(self.x1, self.x2, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.CosineEmbeddingLoss(reduction='none')
        custom_criterion = CustomCosineEmbeddingLoss(reduction='none')
        
        loss_native = criterion(self.x1, self.x2, self.target)
        loss_custom = custom_criterion(self.x1, self.x2, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_with_margin(self):
        margin = 0.5
        criterion = nn.CosineEmbeddingLoss(margin=margin, reduction='mean')
        custom_criterion = CustomCosineEmbeddingLoss(margin=margin, reduction='mean')
        
        loss_native = criterion(self.x1, self.x2, self.target)
        loss_custom = custom_criterion(self.x1, self.x2, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 