"""
Data loading and transformation utilities.

This module provides functions for creating data transformations and
DataLoaders for CIFAR and ImageNet datasets.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from typing import Tuple

def get_cifar_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns standard data augmentation transforms for CIFAR datasets.

    The normalization statistics are standard for CIFAR-10 but work well
    for CIFAR-100 too.

    Returns:
        tuple[transforms.Compose, transforms.Compose]: A tuple containing the
        training and testing transforms, respectively.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test

def get_cifar_loaders(
    dataset_name: str,
    batch_size: int = 64,
    num_workers: int = 2
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Creates and returns DataLoaders for the specified CIFAR dataset.

    Args:
        dataset_name (str): The name of the dataset. Must be 'cifar10' or 'cifar100'.
        batch_size (int): The batch size for the loaders.
        num_workers (int): The number of workers for data loading.

    Returns:
        tuple[DataLoader, DataLoader, int]: A tuple containing:
            - The training data loader.
            - The testing data loader.
            - The number of classes in the dataset.
            
    Raises:
        ValueError: If the dataset_name is not 'cifar10' or 'cifar100'.
    """
    if dataset_name == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
        num_classes = 10
    elif dataset_name == 'cifar100':
        dataset_class = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please choose 'cifar10' or 'cifar100'.")

    transform_train, transform_test = get_cifar_transforms()
    
    trainset = dataset_class(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testset = dataset_class(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return trainloader, testloader, num_classes
# --------------------------------------------------------

def get_imagenet_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns standard data augmentation transforms for the ImageNet dataset.

    Returns:
        tuple[transforms.Compose, transforms.Compose]: A tuple containing the
        training and validation transforms.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return transform_train, transform_val