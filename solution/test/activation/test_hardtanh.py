import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomHardTanh, MyHardTanh

class TestCustomHardTanh(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomHardTanh_forward(self):
        hardtanh = nn.Hardtanh()
        custom_hardtanh = CustomHardTanh()
        
        output_native = hardtanh(self.input_tensor)
        output_custom = custom_hardtanh(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomHardTanh_forward_custom_range(self):
        min_val, max_val = -2.0, 3.0
        hardtanh = nn.Hardtanh(min_val=min_val, max_val=max_val)
        custom_hardtanh = CustomHardTanh(min_val=min_val, max_val=max_val)
        
        output_native = hardtanh(self.input_tensor)
        output_custom = custom_hardtanh(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyHardTanh(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.hardtanh = nn.Hardtanh()
        self.my_hardtanh = MyHardTanh()

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyHardTanh_forward(self):
        output_native = self.hardtanh(self.x)
        output_custom = self.my_hardtanh(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyHardTanh_backward(self):
        output_native = self.hardtanh(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_hardtanh(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 