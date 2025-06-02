import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomELU, MyELU

class TestCustomELU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomELU_forward(self):
        elu = nn.ELU()
        custom_elu = CustomELU()
        
        output_native = elu(self.input_tensor)
        output_custom = custom_elu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomELU_forward_custom_alpha(self):
        alpha = 0.5
        elu = nn.ELU(alpha=alpha)
        custom_elu = CustomELU(alpha=alpha)
        
        output_native = elu(self.input_tensor)
        output_custom = custom_elu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyELU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.elu = nn.ELU()
        self.my_elu = MyELU()

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyELU_forward(self):
        output_native = self.elu(self.x)
        output_custom = self.my_elu(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyELU_backward(self):
        output_native = self.elu(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_elu(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 