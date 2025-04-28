import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomSoftmax

class TestCustomSoftmax(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.large_input = torch.tensor([[1000, 1001, 1002], [0, 1, 2]])
        self.multi_dim_input = torch.randn(2, 3, 4)

    def test_large_values(self):
        custom_softmax = CustomSoftmax(dim=-1)
        output_custom = custom_softmax(self.large_input)
        
        self.assertFalse(torch.isnan(output_custom).any())
        self.assertFalse(torch.isinf(output_custom).any())
        
        sums = torch.sum(output_custom, dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))

    def test_dimensions(self):
        custom_softmax = CustomSoftmax(dim=-1)
        native_softmax = nn.Softmax(dim=-1)
        output_custom = custom_softmax(self.multi_dim_input)
        output_native = native_softmax(self.multi_dim_input)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

        custom_softmax_dim0 = CustomSoftmax(dim=0)
        native_softmax_dim0 = nn.Softmax(dim=0)
        output_custom_dim0 = custom_softmax_dim0(self.multi_dim_input)
        output_native_dim0 = native_softmax_dim0(self.multi_dim_input)
        self.assertTrue(torch.allclose(output_custom_dim0, output_native_dim0, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)