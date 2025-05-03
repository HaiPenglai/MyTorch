import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomLinear, MyLinear

class TestCustomLinear(unittest.TestCase):
    def setUp(self): # 每个测试都独自setup，互不干扰
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



class TestMyLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_dim, self.output_dim = 5, 3
        self.batch_size = 2
        
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.my_linear = MyLinear(self.input_dim, self.output_dim)
        
        with torch.no_grad():
            self.my_linear.weight.copy_(self.linear.weight)
            if self.my_linear.bias is not None:
                self.my_linear.bias.copy_(self.linear.bias)
        
        self.x = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        self.grad_output = torch.randn(self.batch_size, self.output_dim)

    def _zero_grads(self):
        self.linear.zero_grad()
        self.my_linear.zero_grad()
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyLinear_loading(self):
        for p1, p2 in zip(self.linear.parameters(), self.my_linear.parameters()):
            self.assertTrue(torch.allclose(p1, p2, atol=1e-6))

    def test_MyLinear_forward(self):
        output_native = self.linear(self.x)
        output_custom = self.my_linear(self.x)
        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-6),
            "Forward outputs differ"
        )

    def test_MyLinear_backward(self):
        output_native = self.linear(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_input_grad = self.x.grad.clone()
        native_weight_grad = self.linear.weight.grad.clone()
        native_bias_grad = self.linear.bias.grad.clone() if self.linear.bias is not None else None
        
        self._zero_grads()
        
        output_custom = self.my_linear(self.x)
        output_custom.backward(self.grad_output)
        
        self.assertTrue(
            torch.allclose(self.x.grad, native_input_grad, atol=1e-6),
            "Input gradients mismatch"
        )
        self.assertTrue(
            torch.allclose(self.my_linear.weight.grad, native_weight_grad, atol=1e-6),
            "Weight gradients mismatch"
        )
        if self.my_linear.bias is not None:
            self.assertTrue(
                torch.allclose(self.my_linear.bias.grad, native_bias_grad, atol=1e-6),
                "Bias gradients mismatch"
            )
            

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)