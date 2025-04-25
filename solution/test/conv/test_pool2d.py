import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomMaxPool2d

class TestCustomMaxPool2d(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.channels = 3
        self.height = 4
        self.width = 4
        self.kernel_size = 2
        self.stride = 2
        
        self.input_tensor = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        self.maxpool = nn.MaxPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride
        )
        
        self.custom_maxpool = CustomMaxPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride
        )

    def test_CustomMaxPool2d_forward(self):
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