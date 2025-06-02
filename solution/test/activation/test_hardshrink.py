import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomHardShrink, MyHardShrink

class TestCustomHardShrink(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomHardShrink_forward(self):
        hardshrink = nn.Hardshrink()
        custom_hardshrink = CustomHardShrink()
        
        output_native = hardshrink(self.input_tensor)
        output_custom = custom_hardshrink(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomHardShrink_forward_custom_lambd(self):
        lambd = 0.3
        hardshrink = nn.Hardshrink(lambd=lambd)
        custom_hardshrink = CustomHardShrink(lambd=lambd)
        
        output_native = hardshrink(self.input_tensor)
        output_custom = custom_hardshrink(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyHardShrink(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.hardshrink = nn.Hardshrink()
        self.my_hardshrink = MyHardShrink()

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyHardShrink_forward(self):
        output_native = self.hardshrink(self.x)
        output_custom = self.my_hardshrink(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyHardShrink_backward(self):
        output_native = self.hardshrink(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_hardshrink(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 