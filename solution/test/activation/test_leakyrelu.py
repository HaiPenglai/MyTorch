import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomLeakyReLU, MyLeakyReLU

class TestCustomLeakyReLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomLeakyReLU_forward(self):
        leakyrelu = nn.LeakyReLU()
        custom_leakyrelu = CustomLeakyReLU()
        
        output_native = leakyrelu(self.input_tensor)
        output_custom = custom_leakyrelu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomLeakyReLU_forward_custom_slope(self):
        negative_slope = 0.1
        leakyrelu = nn.LeakyReLU(negative_slope=negative_slope)
        custom_leakyrelu = CustomLeakyReLU(negative_slope=negative_slope)
        
        output_native = leakyrelu(self.input_tensor)
        output_custom = custom_leakyrelu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyLeakyReLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.leakyrelu = nn.LeakyReLU()
        self.my_leakyrelu = MyLeakyReLU()

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyLeakyReLU_forward(self):
        output_native = self.leakyrelu(self.x)
        output_custom = self.my_leakyrelu(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyLeakyReLU_backward(self):
        output_native = self.leakyrelu(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_leakyrelu(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 