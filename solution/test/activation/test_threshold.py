import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomThreshold, MyThreshold

class TestCustomThreshold(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomThreshold_forward(self):
        threshold = nn.Threshold(0.0, 0.0)
        custom_threshold = CustomThreshold(0.0, 0.0)
        
        output_native = threshold(self.input_tensor)
        output_custom = custom_threshold(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomThreshold_forward_custom_params(self):
        thresh, value = 0.1, -1.0
        threshold = nn.Threshold(thresh, value)
        custom_threshold = CustomThreshold(thresh, value)
        
        output_native = threshold(self.input_tensor)
        output_custom = custom_threshold(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyThreshold(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.threshold = nn.Threshold(0.0, 0.0)
        self.my_threshold = MyThreshold(0.0, 0.0)

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyThreshold_forward(self):
        output_native = self.threshold(self.x)
        output_custom = self.my_threshold(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyThreshold_backward(self):
        output_native = self.threshold(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_threshold(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 