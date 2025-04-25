import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomMaxPool1d

class TestCustomMaxPool1d(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.channels = 300
        self.width = 100
        self.kernel_size = 2
        self.stride = 4
        self.padding = 0
        
        self.input_tensor = torch.randn(self.batch_size, self.channels, self.width)
        
        self.maxpool = nn.MaxPool1d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        
        self.custom_maxpool = CustomMaxPool1d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

    def test_CustomMaxPool1d_forward(self):
        output_native = self.maxpool(self.input_tensor)
        output_custom = self.custom_maxpool(self.input_tensor)
        
        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-6),
            "Forward pass outputs differ"
        )
        
        self.assertEqual(
            output_custom.shape, output_native.shape,
            "Output shapes differ"
        )

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)