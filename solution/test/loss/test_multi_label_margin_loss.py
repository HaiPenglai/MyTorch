import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomMultiLabelMarginLoss

class TestCustomMultiLabelMarginLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 创建预测值和目标值
        self.pred = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.2, 0.3, 0.5, 0.9]])
        self.target = torch.tensor([[3, 0, -1, 1], [0, 1, -1, 3]])

    def test_mean_reduction(self):
        criterion = nn.MultiLabelMarginLoss(reduction='mean')
        custom_criterion = CustomMultiLabelMarginLoss(reduction='mean')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.MultiLabelMarginLoss(reduction='sum')
        custom_criterion = CustomMultiLabelMarginLoss(reduction='sum')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.MultiLabelMarginLoss(reduction='none')
        custom_criterion = CustomMultiLabelMarginLoss(reduction='none')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        # print(loss_custom, loss_native)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 