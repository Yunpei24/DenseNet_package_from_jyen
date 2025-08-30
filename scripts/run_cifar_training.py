"""
Main script to train a DenseNet model on a CIFAR dataset.

This script handles the entire training pipeline:
- Universal W&B login (environment variable > Kaggle secrets).
- Argument parsing for hyperparameters (dataset, growth rate, epochs, etc.).
- Dynamic data loading for CIFAR-10 or CIFAR-100.
- Model, optimizer, and scheduler setup.
- The main training and evaluation loop.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Import from our custom library
from densenet import (
    densenet_cifar,
    get_cifar_loaders,
    train_one_epoch,
    evaluate_cifar,
    setup_wandb
)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="DenseNet CIFAR Training")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset to use (cifar10 or cifar100)')
    parser.add_argument('--growth-rate', '-k', type=int, default=12,
                        help='Growth rate (k) for DenseNet (default: 12)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train (default: 300)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer (default: 1e-4)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    return parser.parse_args()

def main():
    """Main function to run the training process."""
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    run, use_wandb = setup_wandb(args, project_name=f"densenet-{args.dataset}")

    # --- DataLoaders ---
    print(f"Loading dataset: {args.dataset}...")
    trainloader, testloader, num_classes = get_cifar_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size
    )
    print(f"Dataset loaded with {num_classes} classes.")

    # Model, Criterion, Optimizer, Scheduler
    model = densenet_cifar(growth_rate=args.growth_rate, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )

    start_epoch = 0

    # --- RESUME FROM CHECKPOINT ---
    if use_wandb and run.resumed:
        print("Resuming training from a W&B checkpoint...")
        try:
            artifact_name = f"{run.project}/{run.name}-checkpoint:latest"
            artifact = run.use_artifact(artifact_name)
            artifact_dir = artifact.download(root=args.output_dir)
            checkpoint_path = os.path.join(artifact_dir, "checkpoint.pth")
            
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
        except Exception as e:
            print(f"Could not load checkpoint. Starting from scratch. Error: {e}")

    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(epoch, model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = evaluate_cifar(epoch, model, testloader, criterion, device)
        scheduler.step()

        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                       "test_loss": test_loss, "test_accuracy": test_acc,
                       "learning_rate": scheduler.get_last_lr()[0]})

            # --- SAVE CHECKPOINT ---
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                print(f"Saving checkpoint at epoch {epoch}...")
                local_path = os.path.join(args.output_dir, "checkpoint.pth")
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()}, local_path)
                
                artifact = wandb.Artifact(name=f"{run.name}-checkpoint", type="model",
                                          description=f"Checkpoint at epoch {epoch}")
                artifact.add_file(local_path)
                run.log_artifact(artifact)
                print("Checkpoint saved to W&B Artifacts.")

    print("--- Training finished ---")
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()