"""
densenet - A PyTorch implementation of Densely Connected Convolutional Networks.
"""
from .layers import DenseNetBottleneckLayer, TransitionLayer
from .model import DenseNet, densenet_cifar, densenet121, densenet169, densenet201, densenet161
from .datasets import get_cifar_loaders, get_imagenet_transforms
from .trainer import train_one_epoch, evaluate_cifar, evaluate_imagenet
from .utils import prepare_imagenet_validation_set, setup_wandb

__version__ = "0.1.0"
__author__ = "Joshua Nikiema"
__all__ = [
    "DenseNet",
    "DenseNetBottleneckLayer",
    "TransitionLayer",
    "densenet_cifar",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "get_cifar_loaders",
    "get_imagenet_transforms",
    "train_one_epoch",
    "evaluate_cifar",
    "evaluate_imagenet",
    "prepare_imagenet_validation_set",
    "setup_wandb"
]