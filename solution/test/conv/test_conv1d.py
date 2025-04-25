import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomConv1d

class TestCustomConv1d(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.in_channels = 300
        self.out_channels = 64
        self.kernel_size = 3
        self.batch_size = 2
        self.width = 100
        self.stride = 1
        self.padding = 1

        self.conv1d = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        torch.save(self.conv1d.state_dict(), 'conv1d.pth')

        self.custom_conv1d = CustomConv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        self.custom_conv1d.load_state_dict(torch.load('conv1d.pth', weights_only=True))

        self.x = torch.randn(self.batch_size, self.in_channels, self.width)

    def test_CustomConv1d_loading(self):
        native_state = self.conv1d.state_dict()
        custom_state = self.custom_conv1d.state_dict()

        self.assertEqual(set(native_state.keys()), set(custom_state.keys()),
                       "Parameter names don't match")

        for name in native_state:
            self.assertTrue(torch.allclose(native_state[name], custom_state[name], atol=1e-6),
                          f"Parameter {name} values don't match")

    def test_CustomConv1d_forward(self):
        output_native = self.conv1d(self.x)
        output_custom = self.custom_conv1d(self.x)

        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-4),
            "Forward pass outputs differ"
        )

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)