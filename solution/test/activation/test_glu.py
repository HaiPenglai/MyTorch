import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomGLU, MyGLU

class TestCustomGLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # GLU requires input size to be even along the specified dimension
        self.input_tensor = torch.randn(2, 4, 3)  # 4 is even
        
    def test_CustomGLU_forward(self):
        glu = nn.GLU(dim=1)
        custom_glu = CustomGLU(dim=1)
        
        output_native = glu(self.input_tensor)
        output_custom = custom_glu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomGLU_forward_dim_neg1(self):
        input_tensor = torch.randn(2, 3, 6)  # Last dim is even
        glu = nn.GLU(dim=-1)
        custom_glu = CustomGLU(dim=-1)
        
        output_native = glu(input_tensor)
        output_custom = custom_glu(input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyGLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 4, requires_grad=True)  # 4 is even
        self.grad_output = torch.randn(2, 2)  # Output size is half of input
        self.glu = nn.GLU(dim=1)
        self.my_glu = MyGLU(dim=1)

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyGLU_forward(self):
        output_native = self.glu(self.x)
        output_custom = self.my_glu(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyGLU_backward(self):
        output_native = self.glu(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_glu(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 