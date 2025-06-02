import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomReLU6, MyReLU6

class TestCustomReLU6(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        self.test_input = torch.tensor([[1, -2, 3], [0, 4, -5], [7, 8, 6]], dtype=torch.float32)
        
    def test_CustomReLU6_forward(self):
        relu6 = nn.ReLU6()
        custom_relu6 = CustomReLU6()
        
        output_native = relu6(self.input_tensor)
        output_custom = custom_relu6(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomReLU6_forward_manual(self):
        custom_relu6 = CustomReLU6()
        output = custom_relu6(self.test_input)
        expected = torch.tensor([[1, 0, 3], [0, 4, 0], [6, 6, 6]], dtype=torch.float32)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))


class TestMyReLU6(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.relu6 = nn.ReLU6()
        self.my_relu6 = MyReLU6()

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyReLU6_forward(self):
        output_native = self.relu6(self.x)
        output_custom = self.my_relu6(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyReLU6_backward(self):
        output_native = self.relu6(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_relu6(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 