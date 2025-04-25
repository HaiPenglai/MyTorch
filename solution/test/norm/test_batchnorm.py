import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomBatchNorm

class TestCustomBatchNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_features = 10
        self.input_shape = (2, 10, 5, 5)
        
        self.batch_norm = nn.BatchNorm2d(self.num_features)
        torch.save(self.batch_norm.state_dict(), 'batchnorm.pth')
        
        self.custom_batch_norm = CustomBatchNorm(self.num_features)
        self.custom_batch_norm.load_state_dict(torch.load('batchnorm.pth', weights_only=True))
        
        self.x = torch.randn(self.input_shape)
        
    def test_CustomBatchNorm_loading(self):
        native_state = self.batch_norm.state_dict()
        custom_state = self.custom_batch_norm.state_dict()
        
        self.assertEqual(set(native_state.keys()), set(custom_state.keys()),
                       "Parameter names don't match")
        
        for name in native_state:
            self.assertTrue(torch.allclose(native_state[name], custom_state[name], atol=1e-6),
                          f"Parameter {name} values don't match")
            
    def test_CustomBatchNorm_train_mode(self):
        self.batch_norm.train()
        self.custom_batch_norm.train()
        
        for _ in range(3):
            output_native = self.batch_norm(self.x)
            output_custom = self.custom_batch_norm(self.x)
            self.assertTrue(
                torch.allclose(output_custom, output_native, atol=1e-6),
                "Train mode forward pass outputs differ"
            )
            
    def test_CustomBatchNorm_eval_mode(self):
        self.batch_norm.eval()
        self.custom_batch_norm.eval()
        
        output_native = self.batch_norm(self.x)
        output_custom = self.custom_batch_norm(self.x)
        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-6),
            "Eval mode forward pass outputs differ"
        )

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)