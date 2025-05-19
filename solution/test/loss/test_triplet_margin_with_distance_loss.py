import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import mytorch
from mytorch.mynn import CustomTripletMarginWithDistanceLoss

class TestCustomTripletMarginWithDistanceLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.anchor = torch.tensor([[1.0, 0.5], [0.8, 0.3]])
        self.positive = torch.tensor([[1.1, 0.6], [0.9, 0.4]])
        self.negative = torch.tensor([[0.3, 0.8], [0.4, 0.9]])
        
        # 自定义距离函数
        def cosine_distance(x, y):
            return 1 - F.cosine_similarity(x, y)
        self.cosine_distance = cosine_distance

    def test_default_distance(self):
        criterion = nn.TripletMarginWithDistanceLoss(reduction='mean')
        custom_criterion = CustomTripletMarginWithDistanceLoss(reduction='mean')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_custom_distance(self):
        criterion = nn.TripletMarginWithDistanceLoss(distance_function=self.cosine_distance, reduction='mean')
        custom_criterion = CustomTripletMarginWithDistanceLoss(distance_function=self.cosine_distance, reduction='mean')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_mean_reduction(self):
        criterion = nn.TripletMarginWithDistanceLoss(reduction='mean')
        custom_criterion = CustomTripletMarginWithDistanceLoss(reduction='mean')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.TripletMarginWithDistanceLoss(reduction='sum')
        custom_criterion = CustomTripletMarginWithDistanceLoss(reduction='sum')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.TripletMarginWithDistanceLoss(reduction='none')
        custom_criterion = CustomTripletMarginWithDistanceLoss(reduction='none')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_with_margin(self):
        margin = 0.5
        criterion = nn.TripletMarginWithDistanceLoss(margin=margin, reduction='mean')
        custom_criterion = CustomTripletMarginWithDistanceLoss(margin=margin, reduction='mean')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_with_swap(self):
        criterion = nn.TripletMarginWithDistanceLoss(swap=True, reduction='mean')
        custom_criterion = CustomTripletMarginWithDistanceLoss(swap=True, reduction='mean')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)