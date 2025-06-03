import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomPReLU, MyPReLU

class TestCustomPReLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor = torch.randn(2, 3, 3)
        
    def test_CustomPReLU_forward_single_param(self):
        prelu = nn.PReLU()
        custom_prelu = CustomPReLU()
        
        output_native = prelu(self.input_tensor)
        output_custom = custom_prelu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomPReLU_forward_multi_param(self):
        num_params = 3  # Match channel dimension
        prelu = nn.PReLU(num_parameters=num_params)
        custom_prelu = CustomPReLU(num_parameters=num_params)
        
        # Set same initial weights
        custom_prelu.weight.data.copy_(prelu.weight.data)
        
        output_native = prelu(self.input_tensor)
        output_custom = custom_prelu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomPReLU_forward_custom_init(self):
        init_val = 0.1
        prelu = nn.PReLU(init=init_val)
        custom_prelu = CustomPReLU(init=init_val)
        
        output_native = prelu(self.input_tensor)
        output_custom = custom_prelu(self.input_tensor)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))


class TestMyPReLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.prelu = nn.PReLU()
        self.my_prelu = MyPReLU()
        
        # Set same weights
        self.my_prelu.weight.data.copy_(self.prelu.weight.data)

    def _zero_grads(self):
        if self.x.grad is not None:
            self.x.grad.zero_()
        self.prelu.weight.grad = None
        self.my_prelu.weight.grad = None

    def test_MyPReLU_forward(self):
        output_native = self.prelu(self.x)
        output_custom = self.my_prelu(self.x)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MyPReLU_backward(self):
        output_native = self.prelu(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_input_grad = self.x.grad.clone()
        native_weight_grad = self.prelu.weight.grad.clone()
        self._zero_grads()
        
        output_custom = self.my_prelu(self.x)
        output_custom.backward(self.grad_output)
        
        self.assertTrue(torch.allclose(self.x.grad, native_input_grad, atol=1e-6))
        self.assertTrue(torch.allclose(self.my_prelu.weight.grad, native_weight_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 