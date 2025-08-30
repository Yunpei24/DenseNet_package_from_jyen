"""
DenseNet Model Architecture.

This module assembles the building blocks from `layers.py` into the full
DenseNet model. It includes the main `DenseNet` class and factory functions
to instantiate specific configurations (e.g., DenseNet-121, DenseNet-169).

Classes:
    DenseNet: The main class for the DenseNet architecture.

Functions:
    densenet_cifar: Creates a DenseNet model for CIFAR datasets.
    densenet121: Creates a DenseNet-121 model for ImageNet.
    densenet169: Creates a DenseNet-169 model for ImageNet.
    densenet201: Creates a DenseNet-201 model for ImageNet.
    densenet161: Creates a DenseNet-161 model for ImageNet.
"""
from typing import List, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DenseNetBottleneckLayer, TransitionLayer

class DenseNet(nn.Module):
    """
    The complete DenseNet architecture.

    Args:
        block (Type[DenseNetBottleneckLayer]): The type of layer to use in dense blocks.
        nblocks (List[int]): A list containing the number of layers in each of the four dense blocks.
        growth_rate (int): The growth rate (k). Defaults to 32.
        reduction (float): The compression factor (theta) for transition layers. Defaults to 0.5.
        num_classes (int): The number of output classes. Defaults to 1000 (for ImageNet).
        dropout_rate (float): The dropout rate to apply within bottleneck layers. Defaults to 0.
        init_weights (bool): If True, initializes weights using Kaiming normalization. Defaults to True.
        dataset (str): The dataset being used ('cifar' or 'imagenet'). This affects the initial convolution layer.
                       Defaults to "imagenet".
    """
    def __init__(self,
                 block: Type[DenseNetBottleneckLayer],
                 nblocks: List[int],
                 growth_rate: int = 32,
                 reduction: float = 0.5,
                 num_classes: int = 1000,
                 dropout_rate: float = 0,
                 init_weights: bool = True,
                 dataset: str = "imagenet"):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate
        self.dataset = dataset
        
        num_planes = 2 * growth_rate  # Number of channels after initial convolution

        if self.dataset == "cifar":
            # Initial convolution for smaller CIFAR images (32x32)
            self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        else:
            # Initial convolution for larger ImageNet images (224x224)
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        
        # --- Dense Block 1 ---
        self.dense1 = self._make_dense_block(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(num_planes * reduction)
        self.trans1 = TransitionLayer(num_planes, out_planes)
        num_planes = out_planes
        
        # --- Dense Block 2 ---
        self.dense2 = self._make_dense_block(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(num_planes * reduction)
        self.trans2 = TransitionLayer(num_planes, out_planes)
        num_planes = out_planes
        
        # --- Dense Block 3 ---
        self.dense3 = self._make_dense_block(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        
        if self.dataset != "cifar":
            # Add a fourth dense block and transition layer for ImageNet models
            out_planes = int(num_planes * reduction)
            self.trans3 = TransitionLayer(num_planes, out_planes)
            num_planes = out_planes
            
            # --- Dense Block 4 ---
            self.dense4 = self._make_dense_block(block, num_planes, nblocks[3])
            num_planes += nblocks[3] * growth_rate
            
        # --- Final layers ---
        self.bn = nn.BatchNorm2d(num_planes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(num_planes, num_classes)
        
        if init_weights:
            self._initialize_weights()

    def _make_dense_block(self, block: Type[DenseNetBottleneckLayer], in_planes: int, nblock: int) -> nn.Sequential:
        """Helper function to create a dense block."""
        layers = []
        for _ in range(nblock):
            layers.append(block(in_planes, self.growth_rate, self.dropout_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initializes the model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the DenseNet model."""
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))

        if self.dataset != "cifar":
            out = self.trans3(self.dense3(out))
            out = self.dense4(out)
        else:
            out = self.dense3(out)
            
        out = self.avg_pool(F.relu(self.bn(out)))
        out = torch.flatten(out, 1)
        return self.linear(out)

# --- Factory Functions ---

def densenet_cifar(growth_rate: int = 12, dropout_rate: float = 0, num_classes: int = 10) -> DenseNet:
    """
    Creates a DenseNet model for CIFAR-10/100.
    This configuration has 3 dense blocks with 16 layers each.
    """
    return DenseNet(DenseNetBottleneckLayer, [16, 16, 16], growth_rate=growth_rate,
                    dropout_rate=dropout_rate, num_classes=num_classes, dataset="cifar")

def densenet121(num_classes: int = 1000) -> DenseNet:
    """Creates a DenseNet-121 model for ImageNet."""
    return DenseNet(DenseNetBottleneckLayer, [6, 12, 24, 16], growth_rate=32,
                    num_classes=num_classes, dataset="imagenet")

def densenet169(num_classes: int = 1000) -> DenseNet:
    """Creates a DenseNet-169 model for ImageNet."""
    return DenseNet(DenseNetBottleneckLayer, [6, 12, 32, 32], growth_rate=32,
                    num_classes=num_classes, dataset="imagenet")

def densenet201(num_classes: int = 1000) -> DenseNet:
    """Creates a DenseNet-201 model for ImageNet."""
    return DenseNet(DenseNetBottleneckLayer, [6, 12, 48, 32], growth_rate=32,
                    num_classes=num_classes, dataset="imagenet")

def densenet161(num_classes: int = 1000) -> DenseNet:
    """Creates a DenseNet-161 model for ImageNet."""
    return DenseNet(DenseNetBottleneckLayer, [6, 12, 36, 24], growth_rate=48,
                    num_classes=num_classes, dataset="imagenet")