import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomSoftplus, MySoftplus

class TestCustomSoftplus(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomSoftplus_forward(self):
        softplus = nn.Softplus()
        custom_softplus = CustomSoftplus()
        
        output_native = softplus(self.input_tensor)
        output_custom = custom_softplus(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomSoftplus_forward_custom_params(self):
        beta, threshold = 2.0, 10.0
        softplus = nn.Softplus(beta=beta, threshold=threshold)
        custom_softplus = CustomSoftplus(beta=beta, threshold=threshold)
        
        output_native = softplus(self.input_tensor)
        output_custom = custom_softplus(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMySoftplus(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.softplus = nn.Softplus()
        self.my_softplus = MySoftplus()

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MySoftplus_forward(self):
        output_native = self.softplus(self.x)
        output_custom = self.my_softplus(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MySoftplus_backward(self):
        output_native = self.softplus(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_softplus(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 