"""
Main script to train a DenseNet model on the ImageNet dataset.

This script handles the entire ImageNet training pipeline:
- Universal W&B login.
- Argument parsing for ImageNet-specific paths and model architectures.
- Data preparation for the Kaggle ImageNet directory structure.
- Model, optimizer, and scheduler setup for large-scale training.
- Training loop with Top-1 and Top-5 evaluation.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb

# Import from our custom library
from densenet import (
    densenet121, densenet169, densenet201, densenet161,
    get_imagenet_transforms,
    prepare_imagenet_validation_set,
    train_one_epoch,
    evaluate_imagenet,
    setup_wandb
)

def parse_args():
    """Parses command-line arguments for ImageNet training."""
    parser = argparse.ArgumentParser(description="DenseNet ImageNet Training")
    parser.add_argument('--input-dir', type=str, required=True, help='Path to ImageNet dataset root (Kaggle format)')
    parser.add_argument('--output-dir', type=str, default='./', help='Path to writable directory for sorted validation set')
    parser.add_argument('--model-arch', type=str, default='densenet121',
                        choices=['densenet121', 'densenet169', 'densenet201', 'densenet161'],
                        help='DenseNet architecture to use')
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')

    parser.add_argument('--save-every', type=int, default=5, help='Save a checkpoint every N epochs')

    parser.add_argument('--resume-id', type=str, default=None, help='W&B run ID to resume from a checkpoint')
    parser.add_argument('--wb-project-name', type=str, default='densenet-imagenet', help='W&B project name')
    return parser.parse_args()

def main():
    """Main function to run the ImageNet training process."""
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training {args.model_arch} on ImageNet using device: {device}")

    if not args.no_wandb:
        if args.wb_project_name is None:
            args.wb_project_name = "densenet-imagenet"

    run, use_wandb = setup_wandb(args, project_name=args.wb_project_name)

    # --- Data Preparation ---
    print("Preparing datasets (this may take a while for the first run)...")
    sorted_val_path = prepare_imagenet_validation_set(args.input_dir, args.output_dir)
    train_dir = os.path.join(args.input_dir, 'ILSVRC/Data/CLS-LOC/train')
    transform_train, transform_val = get_imagenet_transforms()

    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    val_dataset = torchvision.datasets.ImageFolder(root=sorted_val_path, transform=transform_val)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes in the dataset.")

    # --- Model, Criterion, Optimizer, Scheduler ---
    print(f"Creating model: {args.model_arch}")
    model_factory = {
        'densenet121': densenet121, 'densenet169': densenet169,
        'densenet201': densenet201, 'densenet161': densenet161
    }
    model = model_factory[args.model_arch](num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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
        train_loss, train_acc = train_one_epoch(epoch, model, trainloader, optimizer, criterion, device, log_interval=1)
        val_loss, top1, top5 = evaluate_imagenet(model, valloader, criterion, device)
        scheduler.step()

        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                       "val_loss": val_loss, "top1_accuracy": top1, "top5_accuracy": top5,
                       "learning_rate": scheduler.get_last_lr()[0]})

            # --- SAVE CHECKPOINT ---
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                print(f"Saving checkpoint at epoch {epoch}...")
                local_path = os.path.join(args.output_dir, "checkpoint.pth")
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()}, local_path)
                
                artifact = wandb.Artifact(name=f"{run.name}-checkpoint", type="model",
                                          description=f"Checkpoint for {args.model_arch} at epoch {epoch}")
                artifact.add_file(local_path)
                run.log_artifact(artifact)
                print("Checkpoint saved to W&B Artifacts.")
    
    print("--- Training finished ---")
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()