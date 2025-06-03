import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomCELU, MyCELU

class TestCustomCELU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomCELU_forward(self):
        celu = nn.CELU()
        custom_celu = CustomCELU()
        
        output_native = celu(self.input_tensor)
        output_custom = custom_celu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomCELU_forward_custom_alpha(self):
        alpha = 0.5
        celu = nn.CELU(alpha=alpha)
        custom_celu = CustomCELU(alpha=alpha)
        
        output_native = celu(self.input_tensor)
        output_custom = custom_celu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyCELU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.celu = nn.CELU()
        self.my_celu = MyCELU()

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyCELU_forward(self):
        output_native = self.celu(self.x)
        output_custom = self.my_celu(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyCELU_backward(self):
        output_native = self.celu(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_celu(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 