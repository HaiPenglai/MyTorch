import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomMish, MyMish

class TestCustomMish(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomMish_forward(self):
        mish = nn.Mish()
        custom_mish = CustomMish()
        
        output_native = mish(self.input_tensor)
        output_custom = custom_mish(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyMish(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.mish = nn.Mish()
        self.my_mish = MyMish()

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyMish_forward(self):
        output_native = self.mish(self.x)
        output_custom = self.my_mish(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyMish_backward(self):
        output_native = self.mish(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_mish(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-5))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 