import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomGELU, MyGELU

class TestCustomGELU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomGELU_forward_approximate(self):
        gelu = nn.GELU(approximate='tanh')
        custom_gelu = CustomGELU(approximate=True)
        
        output_native = gelu(self.input_tensor)
        output_custom = custom_gelu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-5))
        
    def test_CustomGELU_forward_exact(self):
        gelu = nn.GELU(approximate='none')
        custom_gelu = CustomGELU(approximate=False)
        
        output_native = gelu(self.input_tensor)
        output_custom = custom_gelu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyGELU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyGELU_forward_approximate(self):
        gelu = nn.GELU(approximate='tanh')
        my_gelu = MyGELU(approximate=True)
        
        output_native = gelu(self.x)
        output_custom = my_gelu(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-5))

    def test_MyGELU_backward_approximate(self):
        gelu = nn.GELU(approximate='tanh')
        my_gelu = MyGELU(approximate=True)
        
        output_native = gelu(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = my_gelu(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-4))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 