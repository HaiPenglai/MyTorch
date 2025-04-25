import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomMultiheadAttention

class TestCustomMultiheadAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.embed_dim = 6
        self.num_heads = 2
        self.batch_size = 2
        self.seq_len = 3
        self.kv_seq_len = 5
        
        self.official_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0,
            batch_first=True
        )
        torch.save(self.official_attn.state_dict(), 'multihead_attention.pth')
        
        self.custom_attn = CustomMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )
        self.custom_attn.load_state_dict(torch.load('multihead_attention.pth', weights_only=True))
        
        self.X = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.X_cross_kv = torch.randn(self.batch_size, self.kv_seq_len, self.embed_dim)
    
    def test_CustomMultiheadAttention_loading(self):
        native_state = self.official_attn.state_dict()
        custom_state = self.custom_attn.state_dict()
        
        self.assertEqual(set(native_state.keys()), set(custom_state.keys()),
                        "Parameter names don't match")
        
        for name in native_state:
            self.assertTrue(torch.allclose(native_state[name], custom_state[name], atol=1e-6),
                           f"Parameter {name} values don't match")
    
    def test_CustomMultiheadAttention_self_attention(self):
        output_native, weights_native = self.official_attn(self.X, self.X, self.X)
        output_custom, weights_custom = self.custom_attn(self.X, self.X, self.X)
        
        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-5),
            "Self-attention outputs differ"
        )
        
        self.assertTrue(
            torch.allclose(weights_custom, weights_native, atol=1e-5),
            "Self-attention weights differ"
        )
    
    def test_CustomMultiheadAttention_cross_attention(self):
        output_native, weights_native = self.official_attn(self.X, self.X_cross_kv, self.X_cross_kv)
        output_custom, weights_custom = self.custom_attn(self.X, self.X_cross_kv, self.X_cross_kv)
        
        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-5),
            "Cross-attention outputs differ"
        )
        
        self.assertTrue(
            torch.allclose(weights_custom, weights_native, atol=1e-5),
            "Cross-attention weights differ"
        )

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)