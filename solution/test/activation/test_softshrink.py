import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomSoftShrink, MySoftShrink

class TestCustomSoftShrink(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomSoftShrink_forward(self):
        softshrink = nn.Softshrink()
        custom_softshrink = CustomSoftShrink()
        
        output_native = softshrink(self.input_tensor)
        output_custom = custom_softshrink(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomSoftShrink_forward_custom_lambd(self):
        lambd = 0.3
        softshrink = nn.Softshrink(lambd=lambd)
        custom_softshrink = CustomSoftShrink(lambd=lambd)
        
        output_native = softshrink(self.input_tensor)
        output_custom = custom_softshrink(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMySoftShrink(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.softshrink = nn.Softshrink()
        self.my_softshrink = MySoftShrink()

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MySoftShrink_forward(self):
        output_native = self.softshrink(self.x)
        output_custom = self.my_softshrink(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MySoftShrink_backward(self):
        output_native = self.softshrink(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_softshrink(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 