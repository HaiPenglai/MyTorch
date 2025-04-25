import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomLinear

class TestCustomLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.in_features, self.out_features = 5, 3
        self.batch_size = 2
        
        self.linear = nn.Linear(self.in_features, self.out_features)
        torch.save(self.linear.state_dict(), 'linear.pth')
        
        self.custom_linear = CustomLinear(self.in_features, self.out_features)
        self.custom_linear.load_state_dict(torch.load('linear.pth', weights_only=True))
        
        self.x = torch.randn(self.batch_size, self.in_features)

    def test_CustomLinear_loading(self):
        native_state = self.linear.state_dict()
        custom_state = self.custom_linear.state_dict()
        
        self.assertEqual(set(native_state.keys()), set(custom_state.keys()),
                       "Parameter names don't match")
        
        for name in native_state:
            self.assertTrue(torch.allclose(native_state[name], custom_state[name], atol=1e-6),
                          f"Parameter {name} values don't match")

    def test_CustomLinear_forward(self):
        output_native = self.linear(self.x)
        output_custom = self.custom_linear(self.x)
        
        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-6),
            "Forward pass results differ"
        )

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)