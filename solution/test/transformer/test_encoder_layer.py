import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomTransformerEncoderLayer

class TestCustomTransformerEncoderLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 6
        self.n_heads = 2
        self.dim_feedforward = 10
        self.dropout = 0
        self.batch_size = 2
        self.seq_len = 3
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        torch.save(self.encoder_layer.state_dict(), 'transformer_encoder_layer.pth')
        
        self.custom_encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        self.custom_encoder_layer.load_state_dict(
            torch.load('transformer_encoder_layer.pth', weights_only=True)
        )
        
        self.input_data = torch.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_CustomTransformerEncoderLayer_loading(self):
        native_state = self.encoder_layer.state_dict()
        custom_state = self.custom_encoder_layer.state_dict()
        
        self.assertEqual(set(native_state.keys()), set(custom_state.keys()),
                        "Parameter names don't match")
        
        for name in native_state:
            self.assertTrue(torch.allclose(native_state[name], custom_state[name], atol=1e-6),
                          f"Parameter {name} values don't match")
    
    def test_CustomTransformerEncoderLayer_forward(self):
        output_native = self.encoder_layer(self.input_data)
        
        output_custom = self.custom_encoder_layer(self.input_data)
        
        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-4),
            "Forward pass outputs differ"
        )

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)