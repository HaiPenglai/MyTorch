import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomLogSoftmax, MyLogSoftmax

class TestCustomLogSoftmax(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 5)
        
    def test_CustomLogSoftmax_forward(self):
        logsoftmax = nn.LogSoftmax(dim=-1)
        custom_logsoftmax = CustomLogSoftmax(dim=-1)
        
        output_native = logsoftmax(self.input_tensor)
        output_custom = custom_logsoftmax(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomLogSoftmax_forward_dim1(self):
        logsoftmax = nn.LogSoftmax(dim=1)
        custom_logsoftmax = CustomLogSoftmax(dim=1)
        
        output_native = logsoftmax(self.input_tensor)
        output_custom = custom_logsoftmax(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyLogSoftmax(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 5, requires_grad=True)
        self.grad_output = torch.randn(2, 5)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.my_logsoftmax = MyLogSoftmax(dim=-1)

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyLogSoftmax_forward(self):
        output_native = self.logsoftmax(self.x)
        output_custom = self.my_logsoftmax(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyLogSoftmax_backward(self):
        output_native = self.logsoftmax(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_logsoftmax(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 