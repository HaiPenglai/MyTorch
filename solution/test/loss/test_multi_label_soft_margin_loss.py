import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomMultiLabelSoftMarginLoss

class TestMultiLabelSoftMarginLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 测试数据：2个样本，每个样本2个标签
        self.input = torch.tensor([[0.8, -1.2], [1.0, -0.5]], dtype=torch.float32)
        self.target = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)
        
        # 使用PyTorch原生计算作为基准
        self.native_criterion = nn.MultiLabelSoftMarginLoss(reduction='none')
        self.expected_loss_none = self.native_criterion(self.input, self.target)
        self.expected_loss_mean = self.expected_loss_none.mean()
        self.expected_loss_sum = self.expected_loss_none.sum()

    def test_forward(self):
        criterion = CustomMultiLabelSoftMarginLoss(reduction='mean')
        loss = criterion(self.input, self.target)
        self.assertTrue(torch.allclose(loss, self.expected_loss_mean, atol=1e-6))

    def test_mean_reduction(self):
        criterion_native = nn.MultiLabelSoftMarginLoss(reduction='mean')
        criterion_custom = CustomMultiLabelSoftMarginLoss(reduction='mean')
        
        loss_native = criterion_native(self.input, self.target)
        loss_custom = criterion_custom(self.input, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion_native = nn.MultiLabelSoftMarginLoss(reduction='sum')
        criterion_custom = CustomMultiLabelSoftMarginLoss(reduction='sum')
        
        loss_native = criterion_native(self.input, self.target)
        loss_custom = criterion_custom(self.input, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion_native = nn.MultiLabelSoftMarginLoss(reduction='none')
        criterion_custom = CustomMultiLabelSoftMarginLoss(reduction='none')
        
        loss_native = criterion_native(self.input, self.target)
        loss_custom = criterion_custom(self.input, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_with_weight(self):
        weight = torch.tensor([1.0, 2.0])
        criterion_native = nn.MultiLabelSoftMarginLoss(weight=weight, reduction='mean')
        criterion_custom = CustomMultiLabelSoftMarginLoss(weight=weight, reduction='mean')
        
        loss_native = criterion_native(self.input, self.target)
        loss_custom = criterion_custom(self.input, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)