import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomSoftmin, MySoftmin

class TestCustomSoftmin(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 4)
        
    def test_CustomSoftmin_forward(self):
        softmin = nn.Softmin(dim=1)
        custom_softmin = CustomSoftmin(dim=1)
        
        output_native = softmin(self.input_tensor)
        output_custom = custom_softmin(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomSoftmin_forward_no_dim(self):
        softmin = nn.Softmin(dim=None)
        custom_softmin = CustomSoftmin(dim=None)
        
        output_native = softmin(self.input_tensor)
        output_custom = custom_softmin(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomSoftmin_forward_dim_2(self):
        softmin = nn.Softmin(dim=2)
        custom_softmin = CustomSoftmin(dim=2)
        
        output_native = softmin(self.input_tensor)
        output_custom = custom_softmin(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMySoftmin(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.softmin = nn.Softmin(dim=1)
        self.my_softmin = MySoftmin(dim=1)

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MySoftmin_forward(self):
        output_native = self.softmin(self.x)
        output_custom = self.my_softmin(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MySoftmin_backward(self):
        output_native = self.softmin(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_softmin(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 