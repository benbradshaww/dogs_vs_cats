import torch.nn as nn
from torch.nn import (AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Linear, Sigmoid,
                      Softmax)


class SE(nn.Module):
    def __init__(self, input_channels, reduction_rate=16):
        super(SE, self).__init__()
        self.global_avg_pool = AdaptiveAvgPool2d(1)

        reduction_dim = max(1, input_channels // reduction_rate)

        self.fc1 = Linear(in_features=input_channels, out_features=reduction_dim, bias=False)

        self.fc2 = Linear(in_features=reduction_dim, out_features=input_channels, bias=False)

        self.sigmoid = Sigmoid()

        self.swish = lambda x: x * self.sigmoid(x)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        y = self.global_avg_pool(x).view(batch_size, channels)

        y = self.swish(self.fc1(y))
        y = self.sigmoid(self.fc2(y))

        y = y.view(batch_size, channels, 1, 1)

        return x * y


class MBConv(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        padding,
        expansion_ratio,
        stride,
        squeeze: bool = False,
        reduction_rate: int = 16,
    ):
        super().__init__()

        # Expansion
        self.expansion_ratio = expansion_ratio
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = int(input_channels * expansion_ratio)

        self.expansion_conv = Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.expansion_normalisation = BatchNorm2d(num_features=self.hidden_channels)

        # Depthwise convolution
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            groups=self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=stride,
            bias=True,
        )

        self.depthwise_normalisation = BatchNorm2d(num_features=self.hidden_channels)

        # Squeeze and excitation
        self.squeeze = squeeze
        if self.squeeze:
            self.squeeze_and_excitation = SE(self.hidden_channels, reduction_rate=reduction_rate)

        # Pointwise Convolution
        self.output_channels = output_channels
        self.pointwise_conv = Conv2d(
            in_channels=self.hidden_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.pointwise_normalisation = BatchNorm2d(num_features=output_channels)

        self.sigmoid = Sigmoid()
        self.swish = lambda x: x * self.sigmoid(x)

    def forward(self, x):

        x = self.expansion_conv(x)
        x = self.expansion_normalisation(x)
        x = self.swish(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_normalisation(x)
        x = self.swish(x)

        if self.squeeze:
            x = self.squeeze_and_excitation(x)

        x = self.pointwise_conv(x)
        x = self.pointwise_normalisation(x)
        x = self.swish(x)

        return x


class efficient_net_b0(nn.Module):

    def __init__(self):
        super().__init__()

        self.depth = 16
        self.num_classes = 2
        self.output_channels = [16] + (2 * [24]) + (2 * [40]) + (3 * [80]) + (3 * [112]) + (4 * [192]) + [320]
        self.input_channels = [32] + self.output_channels[:-1]
        self.strides = [1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1]
        self.kernel_sizes = (3 * [3]) + (2 * [5]) + (3 * [3]) + (7 * [5]) + (1 * [3])

        self.basic_conv = Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )

        self.final_conv = Conv2d(
            in_channels=self.output_channels[-1],
            out_channels=1280,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.final_norm = BatchNorm2d(1280)

        self.global_avg_pool = AdaptiveAvgPool2d(1)

        self.fc = Linear(in_features=1280, out_features=self.num_classes, bias=False)

        self.softmax = Softmax()

    def forward(self, x):

        x = self.basic_conv(x)

        for i in range(self.depth):
            mb_conv = MBConv(
                input_channels=self.input_channels[i],
                output_channels=self.output_channels[i],
                kernel_size=self.kernel_sizes[i],
                expansion_ratio=6,
                stride=self.strides[i],
                padding=self.kernel_sizes[i] // 2,
                squeeze=True,
                reduction_rate=16,
            )
            x = mb_conv(x)

        x = self.final_conv(x)
        x = self.final_norm(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.softmax(x)

        return x
