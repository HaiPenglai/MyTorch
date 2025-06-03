import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomSoftmax2d, MySoftmax2d

class TestCustomSoftmax2d(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_tensor_3d = torch.randn(3, 4, 5)  # (C, H, W)
        self.input_tensor_4d = torch.randn(2, 3, 4, 5)  # (N, C, H, W)
        
    def test_CustomSoftmax2d_forward_3d(self):
        softmax2d = nn.Softmax2d()
        custom_softmax2d = CustomSoftmax2d()
        
        output_native = softmax2d(self.input_tensor_3d)
        output_custom = custom_softmax2d(self.input_tensor_3d)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomSoftmax2d_forward_4d(self):
        softmax2d = nn.Softmax2d()
        custom_softmax2d = CustomSoftmax2d()
        
        output_native = softmax2d(self.input_tensor_4d)
        output_custom = custom_softmax2d(self.input_tensor_4d)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_CustomSoftmax2d_invalid_dims(self):
        custom_softmax2d = CustomSoftmax2d()
        invalid_input = torch.randn(2, 3)  # 2D tensor
        
        with self.assertRaises(ValueError):
            custom_softmax2d(invalid_input)


class TestMySoftmax2d(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x_3d = torch.randn(3, 4, 5, requires_grad=True)
        self.x_4d = torch.randn(2, 3, 4, 5, requires_grad=True)
        self.grad_output_3d = torch.randn(3, 4, 5)
        self.grad_output_4d = torch.randn(2, 3, 4, 5)
        self.softmax2d = nn.Softmax2d()
        self.my_softmax2d = MySoftmax2d()

    def _zero_grads(self, tensor):
        if tensor.grad is not None:
            tensor.grad.zero_()

    def test_MySoftmax2d_forward_3d(self):
        output_native = self.softmax2d(self.x_3d)
        output_custom = self.my_softmax2d(self.x_3d)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MySoftmax2d_forward_4d(self):
        output_native = self.softmax2d(self.x_4d)
        output_custom = self.my_softmax2d(self.x_4d)
        self.assertTrue(torch.allclose(output_custom, output_native, atol=1e-6))

    def test_MySoftmax2d_backward_3d(self):
        output_native = self.softmax2d(self.x_3d)
        output_native.backward(self.grad_output_3d, retain_graph=True)
        native_grad = self.x_3d.grad.clone()
        self._zero_grads(self.x_3d)
        
        output_custom = self.my_softmax2d(self.x_3d)
        output_custom.backward(self.grad_output_3d)
        self.assertTrue(torch.allclose(self.x_3d.grad, native_grad, atol=1e-6))

    def test_MySoftmax2d_backward_4d(self):
        output_native = self.softmax2d(self.x_4d)
        output_native.backward(self.grad_output_4d, retain_graph=True)
        native_grad = self.x_4d.grad.clone()
        self._zero_grads(self.x_4d)
        
        output_custom = self.my_softmax2d(self.x_4d)
        output_custom.backward(self.grad_output_4d)
        self.assertTrue(torch.allclose(self.x_4d.grad, native_grad, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__) 