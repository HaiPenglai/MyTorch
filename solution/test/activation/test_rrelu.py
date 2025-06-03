import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomRReLU, MyRReLU

class TestCustomRReLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomRReLU_forward_eval(self):
        # Test in evaluation mode where behavior is deterministic
        rrelu = nn.RReLU()
        custom_rrelu = CustomRReLU()
        
        rrelu.eval()
        custom_rrelu.eval()
        
        output_native = rrelu(self.input_tensor)
        output_custom = custom_rrelu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomRReLU_forward_custom_params_eval(self):
        lower, upper = 0.1, 0.2
        rrelu = nn.RReLU(lower=lower, upper=upper)
        custom_rrelu = CustomRReLU(lower=lower, upper=upper)
        
        rrelu.eval()
        custom_rrelu.eval()
        
        output_native = rrelu(self.input_tensor)
        output_custom = custom_rrelu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyRReLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.rrelu = nn.RReLU()
        self.my_rrelu = MyRReLU()

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()

    def test_MyRReLU_forward_eval(self):
        # Test in evaluation mode
        self.rrelu.eval()
        self.my_rrelu.eval()
        
        output_native = self.rrelu(self.x)
        output_custom = self.my_rrelu(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyRReLU_backward_eval(self):
        # Test gradients in evaluation mode
        self.rrelu.eval()
        self.my_rrelu.eval()
        
        output_native = self.rrelu(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_rrelu(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 