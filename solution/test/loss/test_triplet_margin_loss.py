import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomTripletMarginLoss

class TestCustomTripletMarginLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.anchor = torch.tensor([[1.0, 0.5], [0.8, 0.3]])
        self.positive = torch.tensor([[1.1, 0.6], [0.9, 0.4]])
        self.negative = torch.tensor([[0.3, 0.8], [0.4, 0.9]])

    def test_mean_reduction(self):
        criterion = nn.TripletMarginLoss(reduction='mean')
        custom_criterion = CustomTripletMarginLoss(reduction='mean')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_sum_reduction(self):
        criterion = nn.TripletMarginLoss(reduction='sum')
        custom_criterion = CustomTripletMarginLoss(reduction='sum')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_none_reduction(self):
        criterion = nn.TripletMarginLoss(reduction='none')
        custom_criterion = CustomTripletMarginLoss(reduction='none')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_with_margin_and_p(self):
        margin = 0.5
        p = 1
        criterion = nn.TripletMarginLoss(margin=margin, p=p, reduction='mean')
        custom_criterion = CustomTripletMarginLoss(margin=margin, p=p, reduction='mean')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

    def test_with_swap(self):
        criterion = nn.TripletMarginLoss(swap=True, reduction='mean')
        custom_criterion = CustomTripletMarginLoss(swap=True, reduction='mean')
        
        loss_native = criterion(self.anchor, self.positive, self.negative)
        loss_custom = custom_criterion(self.anchor, self.positive, self.negative)
        self.assertTrue(torch.allclose(loss_custom, loss_native, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 