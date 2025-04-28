import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomTanh

class TestCustomTanh(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        self.test_input = torch.tensor([[1, -2, 3], [0, 4, -5]])
        
    def test_CustomTanh_forward(self):
        tanh = nn.Tanh()
        custom_tanh = CustomTanh()
        
        output_native = tanh(self.input_tensor)
        output_custom = custom_tanh(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)