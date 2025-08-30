"""
Training and evaluation logic for DenseNet models.

This module contains functions to handle the training loop, validation,
and metric calculation for both CIFAR and ImageNet datasets.
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

def train_one_epoch(epoch: int,
                    model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: Optimizer,
                    criterion: nn.Module,
                    device: str,
                    log_interval: int = 10):
    """
    Performs a single training epoch for a model.

    Args:
        epoch (int): The current epoch number.
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): The DataLoader for the training data.
        optimizer (Optimizer): The optimizer to use.
        criterion (nn.Module): The loss function.
        device (str): The device to train on ('cuda' or 'cpu').
        log_interval (int): How often to print training progress.

    Returns:
        tuple[float, float]: A tuple containing the average loss and accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    lr = optimizer.param_groups[0]["lr"]
    print(f'\nEpoch: {epoch} | Learning Rate: {lr:.5f}')
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (epoch % log_interval == 0) or (epoch == 0):
            acc = 100. * correct / total
            progress_bar.set_description(f'Loss: {running_loss/(i+1):.3f} | Acc: {acc:.3f}%')

    avg_loss = running_loss / len(dataloader)
    avg_acc = 100. * correct / total
    return avg_loss, avg_acc


def evaluate_cifar(epoch: int,
                   model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: str,
                   log_interval: int = 10) -> Tuple[float, float, float]:
    """
    Evaluates the model on the test/validation set.

    Args:
        epoch (int): The current epoch number.
        model (nn.Module): The model to be evaluated.
        dataloader (DataLoader): The DataLoader for the test/validation data.
        criterion (nn.Module): The loss function.
        device (str): The device to evaluate on ('cuda' or 'cpu').
        log_interval (int): How often to print evaluation results.

    Returns:
        tuple[float, float]: A tuple containing the average loss and accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    avg_loss = test_loss / len(dataloader)
    avg_acc = 100. * correct / total
    error_rate = 100. - avg_acc
    
    if (epoch % log_interval == 0) or (epoch == 0):
        print(f"\n--- Epoch {epoch} Test Results ---")
        print(f"Accuracy: {avg_acc:.2f}% | Error Rate: {error_rate:.2f}% | Loss: {avg_loss:.4f}")
        print("--------------------------------\n")

    return avg_loss, avg_acc, error_rate

def evaluate_imagenet(model: nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      criterion: nn.Module,
                      device: str) -> Tuple[float, float, float]:
    """
    Evaluates the model on the ImageNet validation set.
    Calculates Top-1 and Top-5 accuracy.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The validation DataLoader.
        criterion (nn.Module): The loss function.
        device (str): The device to run evaluation on.

    Returns:
        tuple[float, float, float]: A tuple of (average loss, top-1 accuracy, top-5 accuracy).
    """
    model.eval()
    val_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Calculate Top-1 and Top-5 accuracy
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            correct_top1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
            correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            total += targets.size(0)

            top1_acc_batch = 100. * correct_top1 / total
            top5_acc_batch = 100. * correct_top5 / total
            progress_bar.set_postfix(top1=f'{top1_acc_batch:.2f}%', top5=f'{top5_acc_batch:.2f}%')

    avg_loss = val_loss / len(dataloader)
    top1_acc = 100. * correct_top1 / total
    top5_acc = 100. * correct_top5 / total

    print("\n--- Validation Results ---")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}% ({int(correct_top1)}/{total})")
    print(f"Top-5 Accuracy: {top5_acc:.2f}% ({int(correct_top5)}/{total})")
    print("--------------------------\n")
    
    return avg_loss, top1_acc, top5_acc