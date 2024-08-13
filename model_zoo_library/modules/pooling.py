import torch.nn as nn

class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        in_h, in_w = x.shape[2:]
        out_h, out_w = self.output_size

        stride_h = in_h // out_h
        stride_w = in_w // out_w

        kernel_h = in_h - (out_h - 1) * stride_h
        kernel_w = in_w - (out_w - 1) * stride_w

        return torch.nn.functional.avg_pool2d(x, (kernel_h, kernel_w), (stride_h, stride_w))
