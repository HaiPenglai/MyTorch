import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomCTCLoss

class TestCustomCTCLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 更真实的变长序列场景
        self.log_probs = torch.randn(10, 3, 5).log_softmax(-1)  # (T=10, N=3, C=5)
        
        # 样本1: 实际长度8，标签长度4
        # 样本2: 实际长度5，标签长度2 (用padding缩短)
        # 样本3: 实际长度10，标签长度3
        self.input_lengths = torch.tensor([8, 5, 10], dtype=torch.long)
        self.target_lengths = torch.tensor([4, 2, 3], dtype=torch.long)
        
        self.targets = torch.tensor(
            [1, 3, 2, 4,  # 样本1
             0, 2,       # 样本2 
             2, 1, 4],    # 样本3
            dtype=torch.long
        )

    def test_mean_reduction(self):
        criterion = nn.CTCLoss(reduction='mean')
        custom_criterion = CustomCTCLoss(reduction='mean')
        
        loss_native = criterion(
            self.log_probs, self.targets,
            self.input_lengths, self.target_lengths
        )
        loss_custom = custom_criterion(
            self.log_probs, self.targets,
            self.input_lengths, self.target_lengths
        )
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.CTCLoss(reduction='sum')
        custom_criterion = CustomCTCLoss(reduction='sum')
        
        loss_native = criterion(
            self.log_probs, self.targets,
            self.input_lengths, self.target_lengths
        )
        loss_custom = custom_criterion(
            self.log_probs, self.targets,
            self.input_lengths, self.target_lengths
        )
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.CTCLoss(reduction='none')
        custom_criterion = CustomCTCLoss(reduction='none')
        
        loss_native = criterion(
            self.log_probs, self.targets,
            self.input_lengths, self.target_lengths
        )
        loss_custom = custom_criterion(
            self.log_probs, self.targets,
            self.input_lengths, self.target_lengths
        )
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)