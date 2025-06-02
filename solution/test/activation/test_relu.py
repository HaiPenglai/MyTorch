import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomReLU, MyReLU

class TestCustomReLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        self.test_input = torch.tensor([[1, -2, 3], [0, 4, -5]])
        
    def test_CustomReLU_forward(self):
        relu = nn.ReLU()
        custom_relu = CustomReLU()
        
        output_native = relu(self.input_tensor)
        output_custom = custom_relu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyReLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.relu = nn.ReLU()
        self.my_relu = MyReLU()

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyReLU_forward(self):
        output_native = self.relu(self.x)
        output_custom = self.my_relu(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyReLU_backward(self):
        output_native = self.relu(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_relu(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)