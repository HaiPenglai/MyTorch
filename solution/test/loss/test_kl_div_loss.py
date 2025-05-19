import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomKLDivLoss

class TestCustomKLDivLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 创建预测分布（log形式）
        self.pred = torch.log_softmax(torch.tensor([[0.2, 0.8], [0.6, 0.4]]), dim=-1)
        # 创建目标分布（概率形式）
        self.target = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
        # 创建目标分布（log形式）
        self.log_target = torch.log(self.target)

    def test_mean_reduction(self):
        criterion = nn.KLDivLoss(reduction='mean')
        custom_criterion = CustomKLDivLoss(reduction='mean')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_batchmean_reduction(self):
        criterion = nn.KLDivLoss(reduction='batchmean')
        custom_criterion = CustomKLDivLoss(reduction='batchmean')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.KLDivLoss(reduction='sum')
        custom_criterion = CustomKLDivLoss(reduction='sum')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.KLDivLoss(reduction='none')
        custom_criterion = CustomKLDivLoss(reduction='none')
        
        loss_native = criterion(self.pred, self.target)
        loss_custom = custom_criterion(self.pred, self.target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_log_target(self):
        criterion = nn.KLDivLoss(reduction='mean', log_target=True)
        custom_criterion = CustomKLDivLoss(reduction='mean', log_target=True)
        
        loss_native = criterion(self.pred, self.log_target)
        loss_custom = custom_criterion(self.pred, self.log_target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_log_target_batchmean(self):
        criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
        custom_criterion = CustomKLDivLoss(reduction='batchmean', log_target=True)
        
        loss_native = criterion(self.pred, self.log_target)
        loss_custom = custom_criterion(self.pred, self.log_target)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 