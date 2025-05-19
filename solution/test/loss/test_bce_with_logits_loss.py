import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomBCEWithLogitsLoss

class TestCustomBCEWithLogitsLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 创建测试数据：2个样本，每个样本3个类别的logits
        self.logits = torch.tensor([[2.0, 1.0, 0.1], 
                                  [0.5, 2.0, 0.3]])
        # 创建对应的目标值（0-1之间的值）
        self.targets = torch.tensor([[0.8, 0.2, 0.1],
                                   [0.3, 0.9, 0.4]])

    def test_mean_reduction(self):
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        custom_criterion = CustomBCEWithLogitsLoss(reduction='mean')
        
        loss_native = criterion(self.logits, self.targets)
        loss_custom = custom_criterion(self.logits, self.targets)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        custom_criterion = CustomBCEWithLogitsLoss(reduction='sum')
        
        loss_native = criterion(self.logits, self.targets)
        loss_custom = custom_criterion(self.logits, self.targets)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        custom_criterion = CustomBCEWithLogitsLoss(reduction='none')
        
        loss_native = criterion(self.logits, self.targets)
        loss_custom = custom_criterion(self.logits, self.targets)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 