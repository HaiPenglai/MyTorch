import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomSoftmax, MySoftmax

class TestCustomSoftmax(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.large_input = torch.tensor([[1000, 1001, 1002], [0, 1, 2]])
        self.multi_dim_input = torch.randn(2, 3, 4)

    def test_large_values(self):
        custom_softmax = CustomSoftmax(dim=-1)
        output_custom = custom_softmax(self.large_input)
        
        self.assertFalse(torch.isnan(output_custom).any())
        self.assertFalse(torch.isinf(output_custom).any())
        
        sums = torch.sum(output_custom, dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))

    def test_dimensions(self):
        custom_softmax = CustomSoftmax(dim=-1)
        native_softmax = nn.Softmax(dim=-1)
        output_custom = custom_softmax(self.multi_dim_input)
        output_native = native_softmax(self.multi_dim_input)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

        custom_softmax_dim0 = CustomSoftmax(dim=0)
        native_softmax_dim0 = nn.Softmax(dim=0)
        output_custom_dim0 = custom_softmax_dim0(self.multi_dim_input)
        output_native_dim0 = native_softmax_dim0(self.multi_dim_input)
        self.assertTrue(torch.allclose(output_custom_dim0, output_native_dim0, atol=1e-6))

class TestMySoftmax(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, requires_grad=True)
        self.grad_output = torch.randn(2, 3)
        self.large_input = torch.tensor([[1000, 1001, 1002], [0, 1, 2]])

    def test_MySoftmax_forward(self):
        # 普通输入
        softmax = nn.Softmax(dim=1)
        my_softmax = MySoftmax(dim=1)
        self.assertTrue(torch.allclose(my_softmax(self.x), softmax(self.x), atol=1e-6))

        # 大值稳定性测试
        output = MySoftmax(dim=1)(self.large_input)
        self.assertFalse(torch.isnan(output).any())
        self.assertTrue(torch.allclose(torch.sum(output, dim=1), torch.ones(2), atol=1e-6))

    def test_MySoftmax_backward(self):
        softmax = nn.Softmax(dim=1)
        my_softmax = MySoftmax(dim=1)
        
        output_native = softmax(self.x)
        output_native.backward(self.grad_output, retain_graph=True)
        native_grad = self.x.grad.clone()
        self.x.grad.zero_()
        
        output_custom = my_softmax(self.x)
        output_custom.backward(self.grad_output)
        self.assertTrue(torch.allclose(self.x.grad, native_grad, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)