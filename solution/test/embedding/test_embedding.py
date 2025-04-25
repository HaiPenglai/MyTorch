import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomEmbedding

class TestCustomEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_embeddings = 10
        self.embedding_dim = 4
        self.batch_size = 2
        self.seq_len = 5
        
        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim
        )
        torch.save(self.embedding.state_dict(), 'embedding.pth')
        
        self.custom_embedding = CustomEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim
        )
        self.custom_embedding.load_state_dict(torch.load('embedding.pth', weights_only=True))
        
        self.x = torch.randint(0, self.num_embeddings, (self.batch_size, self.seq_len))

    def test_CustomEmbedding_loading(self):
        native_state = self.embedding.state_dict()
        custom_state = self.custom_embedding.state_dict()
        
        self.assertEqual(set(native_state.keys()), set(custom_state.keys()),
                       "Parameter names don't match")
        
        for name in native_state:
            self.assertTrue(torch.allclose(native_state[name], custom_state[name], atol=1e-6),
                          f"Parameter {name} values don't match")
          
    def test_CustomEmbedding_forward(self):
        output_native = self.embedding(self.x)
        output_custom = self.custom_embedding(self.x)
        
        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-6),
            "Forward pass outputs differ"
        )

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)