import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomLayerNorm

class TestCustomLayerNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.normalized_shape = [10]
        self.input_shape = (2, 5, 10)
        
        self.layer_norm = nn.LayerNorm(self.normalized_shape)
        torch.save(self.layer_norm.state_dict(), 'layernorm.pth')
        
        self.custom_layer_norm = CustomLayerNorm(self.normalized_shape)
        self.custom_layer_norm.load_state_dict(torch.load('layernorm.pth', weights_only=True))
        
        self.x = torch.randn(self.input_shape)
        
    def test_CustomLayerNorm_loading(self):
        native_state = self.layer_norm.state_dict()
        custom_state = self.custom_layer_norm.state_dict()
        
        self.assertEqual(set(native_state.keys()), set(custom_state.keys()),
                       "Parameter names don't match")
        
        for name in native_state:
            self.assertTrue(torch.allclose(native_state[name], custom_state[name], atol=1e-6),
                          f"Parameter {name} values don't match")
            
    def test_CustomLayerNorm_forward(self):
        output_native = self.layer_norm(self.x)
        output_custom = self.custom_layer_norm(self.x)
        
        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-6),
            "Forward pass outputs differ"
        )

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)