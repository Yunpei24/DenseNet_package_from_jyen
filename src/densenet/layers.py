"""
Core building blocks for the DenseNet architecture.

This module contains the implementation of the essential layers used in DenseNet,
namely the Bottleneck Layer and the Transition Layer, as described in the paper
"Densely Connected Convolutional Networks" by Huang et al.

Classes:
    DenseNetBottleneckLayer: A bottleneck layer with a 1x1 conv followed by a 3x3 conv.
    TransitionLayer: A layer used to downsample feature maps between dense blocks.
"""
import torch
import torch.nn as nn

class DenseNetBottleneckLayer(nn.Module):
    """
    Implements a DenseNet bottleneck layer.

    This layer consists of two convolutional steps:
    1. A 1x1 convolution to reduce the number of feature maps (bottleneck).
    2. A 3x3 convolution which produces `growth_rate` new feature maps.

    The sequence of operations is: BatchNorm -> ReLU -> Conv(1x1) -> BatchNorm -> ReLU -> Conv(3x3).
    A dropout layer can be optionally added after the 3x3 convolution.
    The output of this layer is concatenated with the input tensor along the channel dimension.

    Args:
        in_channels (int): Number of input channels.
        growth_rate (int): The number of new channels to add (k). The 3x3 conv
                           will have `growth_rate` output channels.
        dropout_rate (float, optional): The probability for the dropout layer.
                                        If 0, no dropout is applied. Defaults to 0.
    """
    def __init__(self, in_channels: int, growth_rate: int, dropout_rate: float = 0):
        super(DenseNetBottleneckLayer, self).__init__()
        inter_channels = 4 * growth_rate
        self.dropout_rate = dropout_rate

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the bottleneck layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor, which is the concatenation of the
                          input tensor and the new feature maps.
        """
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))

        if self.dropout_rate > 0:
            out = self.dropout(out)

        return torch.cat([x, out], 1)


class TransitionLayer(nn.Module):
    """
    A transition layer to connect two consecutive dense blocks.

    This layer performs two main functions:
    1. It reduces the number of channels using a 1x1 convolution (compression).
    2. It downsamples the spatial dimensions of the feature maps using average pooling.

    The sequence of operations is: BatchNorm -> ReLU -> Conv(1x1) -> AvgPool2d.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after compression.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the transition layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The downsampled output tensor.
        """
        out = self.conv1(self.relu1(self.bn1(x)))
        return self.avg_pool(out)