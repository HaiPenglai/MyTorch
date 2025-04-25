import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMaxPool1d(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super(CustomMaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        
    def forward(self, x):
        # Step 1: Padding
        if self.padding > 0:
            pad_tensor = F.pad(x, (self.padding, self.padding), 'constant', 0)
        else:
            pad_tensor = x

        # Step 2: Calculate the width after padding
        batch_size, channels, padded_width = pad_tensor.shape

        # Step 3: Calculate output size
        output_width = (padded_width - self.kernel_size) // self.stride + 1

        # Initialize the output tensor
        output = torch.zeros((batch_size, channels, output_width), device=x.device)

        # Step 4: Apply the pooling operation
        for i in range(output_width):
            # Calculate the start and end of the window
            start = i * self.stride
            end = start + self.kernel_size
            # Select the window and apply max pooling
            output[:, :, i] = pad_tensor[:, :, start:end].max(dim=2)[0]

        return output