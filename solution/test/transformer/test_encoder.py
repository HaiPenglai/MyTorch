import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomTransformerEncoderLayer, CustomTransformerEncoder

class TestCustomTransformerEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
        self.d_model = 300
        self.n_heads = 10
        self.dim_feedforward = 512
        self.dropout = 0.0
        self.num_layers = 2
        self.batch_size = 2
        self.seq_len = 10
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        custom_encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        self.custom_transformer_encoder = CustomTransformerEncoder(
            custom_encoder_layer,
            num_layers=self.num_layers
        )
        
        self.custom_transformer_encoder.load_state_dict(self.transformer_encoder.state_dict())
        
        self.input_data = torch.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_CustomTransformerEncoder_loading(self):
        native_state = self.transformer_encoder.state_dict()
        custom_state = self.custom_transformer_encoder.state_dict()
        
        self.assertEqual(set(native_state.keys()), set(custom_state.keys()),
                       "Parameter names don't match")
        
        for name in native_state:
            self.assertTrue(torch.allclose(native_state[name], custom_state[name], atol=1e-6),
                          f"Parameter {name} values don't match")
    
    def test_CustomTransformerEncoder_forward(self):
        output_native = self.transformer_encoder(self.input_data)
        output_custom = self.custom_transformer_encoder(self.input_data)
        
        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-4),
            "Forward pass outputs differ"
        )

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)