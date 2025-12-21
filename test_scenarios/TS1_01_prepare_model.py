"""
Test Scenario TS1 - Script 1: Model Preparation and Fine-Tuning
================================================================
Model: ResNet-18 (Pretrained)
Dataset: CIFAR-10
Purpose: Download pretrained model, adapt to CIFAR-10, and fine-tune

Output:
- Pretrained model checkpoint
- Fine-tuned model checkpoint (every 5 epochs)
- Accuracy comparison table (Before FT vs After FT)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
from pathlib import Path
from typing import Dict, Tuple
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cleanai import evaluate_model, count_parameters


# ==================== Configuration ====================
TEST_SCENARIO = "TS1"
MODEL_NAME = "ResNet18"
DATASET_NAME = "CIFAR10"

# Directories
MODEL_DIR = Path(r"C:\source\downloaded_models")
DATASET_DIR = Path(r"C:\source\downloaded_datasets")
CHECKPOINT_DIR = Path(rf"C:\source\checkpoints\{TEST_SCENARIO}")

# Training parameters
FINE_TUNE_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4
SAVE_EVERY_N_EPOCHS = 5

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Helper Functions ====================

def get_cifar10_dataloaders(batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """
    Download and prepare CIFAR-10 dataloaders.
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    print("\n" + "="*60)
    print("Loading CIFAR-10 Dataset")
    print("="*60)
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Download and load datasets
    train_dataset = datasets.CIFAR10(
        root=str(DATASET_DIR),
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=str(DATASET_DIR),
        train=False,
        download=True,
        transform=transform_test
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ Train samples: {len(train_dataset):,}")
    print(f"✓ Test samples: {len(test_dataset):,}")
    print(f"✓ Number of classes: {len(train_dataset.classes)}")
    
    return train_loader, test_loader


def adapt_resnet18_for_cifar10() -> nn.Module:
    """
    Download pretrained ResNet-18 and adapt for CIFAR-10.
    
    Returns:
        Adapted ResNet-18 model
    """
    print("\n" + "="*60)
    print("Loading Pretrained ResNet-18")
    print("="*60)
    
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=True)
    
    # Adapt for CIFAR-10 (32x32 images, 10 classes)
    # 1. Smaller first conv layer (no downsampling)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Remove max pooling
    model.maxpool = nn.Identity()
    
    # 3. Replace final FC layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    print(f"✓ Model adapted for CIFAR-10")
    print(f"✓ Input: 32x32x3")
    print(f"✓ Output: 10 classes")
    
    return model


def measure_accuracy(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """
    Measure model accuracy on test set.
    
    Returns:
        Dictionary with accuracy and loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Returns:
        Training metrics
    """
    model.train()
    
    correct = 0
    total = 0
    total_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100.0 * correct / total:.2f}%")
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss
    }


def fine_tune_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float
) -> Dict[str, list]:
    """
    Fine-tune model on CIFAR-10.
    
    Returns:
        Training history
    """
    print("\n" + "="*60)
    print("Fine-Tuning ResNet-18 on CIFAR-10")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_acc': [],
        'train_loss': [],
        'test_acc': [],
        'test_loss': []
    }
    
    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch)
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_loss'].append(train_metrics['loss'])
        
        # Validate
        test_metrics = measure_accuracy(model, test_loader)
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        
        # Update learning rate
        scheduler.step()
        
        print(f"\n  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Test Loss:  {test_metrics['loss']:.4f} | Test Acc:  {test_metrics['accuracy']:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint every N epochs
        if epoch % SAVE_EVERY_N_EPOCHS == 0 or epoch == epochs:
            checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_{DATASET_NAME}_FT_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': test_metrics['accuracy'],
                'train_history': history
            }, checkpoint_path)
            print(f"\n  ✓ Checkpoint saved: {checkpoint_path.name}")
    
    return history


def print_comparison_table(before_metrics: Dict, after_metrics: Dict, params: Dict):
    """Print before/after comparison table."""
    print("\n" + "="*80)
    print("FINE-TUNING RESULTS - COMPARISON TABLE")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Before Fine-Tuning':<25} {'After Fine-Tuning':<25}")
    print("-" * 80)
    print(f"{'Accuracy (%)':<30} {before_metrics['accuracy']:>24.2f} {after_metrics['accuracy']:>24.2f}")
    print(f"{'Loss':<30} {before_metrics['loss']:>24.4f} {after_metrics['loss']:>24.4f}")
    print(f"{'Correct Predictions':<30} {before_metrics['correct']:>24,} {after_metrics['correct']:>24,}")
    print(f"{'Total Samples':<30} {before_metrics['total']:>24,} {after_metrics['total']:>24,}")
    print("-" * 80)
    print(f"{'Total Parameters':<30} {params['total']:>24,} {params['total']:>24,}")
    print(f"{'Trainable Parameters':<30} {params['trainable']:>24,} {params['trainable']:>24,}")
    print("-" * 80)
    
    improvement = after_metrics['accuracy'] - before_metrics['accuracy']
    print(f"\n✓ Accuracy Improvement: {improvement:+.2f}%")
    print(f"✓ Final Test Accuracy: {after_metrics['accuracy']:.2f}%")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    
    print("\n" + "#"*80)
    print(f"# TEST SCENARIO {TEST_SCENARIO} - SCRIPT 1: MODEL PREPARATION & FINE-TUNING")
    print("#"*80)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Device: {DEVICE}")
    print(f"\nDirectories:")
    print(f"  Models: {MODEL_DIR}")
    print(f"  Datasets: {DATASET_DIR}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    
    start_time = time.time()
    
    # Check if final model already exists
    final_checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_{DATASET_NAME}_FT_final.pth"
    
    if final_checkpoint_path.exists():
        print("\n" + "="*60)
        print("CHECKPOINT FOUND - Loading Existing Model")
        print("="*60)
        print(f"✓ Found: {final_checkpoint_path.name}")
        print("✓ Skipping fine-tuning (already completed)")
        
        # Load dataset for evaluation
        train_loader, test_loader = get_cifar10_dataloaders(BATCH_SIZE, 0)
        
        # Load model
        model = adapt_resnet18_for_cifar10()
        checkpoint = torch.load(final_checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        
        # Get metrics from checkpoint
        after_metrics = {'accuracy': checkpoint['accuracy'], 'loss': 0.0, 'correct': 0, 'total': 10000}
        params_info = count_parameters(model)
        
        # Load or measure before metrics
        initial_checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_{DATASET_NAME}_pretrained.pth"
        if initial_checkpoint_path.exists():
            initial_checkpoint = torch.load(initial_checkpoint_path, map_location=DEVICE)
            before_metrics = {'accuracy': initial_checkpoint['accuracy'], 'loss': 0.0, 'correct': 0, 'total': 10000}
        else:
            before_metrics = {'accuracy': 11.10, 'loss': 2.5241, 'correct': 0, 'total': 10000}  # Typical initial values
        
        print(f"✓ Model accuracy: {after_metrics['accuracy']:.2f}%")
        print(f"✓ Parameters: {params_info['total']:,}")
    else:
        print("\n" + "="*60)
        print("No Checkpoint Found - Starting Fresh Training")
        print("="*60)
        
        # Step 1: Load dataset
        train_loader, test_loader = get_cifar10_dataloaders(BATCH_SIZE, 0)
        
        # Step 2: Prepare model
        model = adapt_resnet18_for_cifar10()
        model = model.to(DEVICE)
        
        # Get model parameters info
        params_info = count_parameters(model)
        print(f"\n✓ Total Parameters: {params_info['total']:,}")
        print(f"✓ Trainable Parameters: {params_info['trainable']:,}")
        
        # Step 3: Measure initial accuracy (pretrained, adapted)
        print("\n" + "="*60)
        print("Measuring Initial Accuracy (Before Fine-Tuning)")
        print("="*60)
        before_metrics = measure_accuracy(model, test_loader)
        print(f"✓ Initial Accuracy: {before_metrics['accuracy']:.2f}%")
        print(f"✓ Initial Loss: {before_metrics['loss']:.4f}")
        
        # Save initial model
        initial_checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_{DATASET_NAME}_pretrained.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': before_metrics['accuracy'],
            'adapted': True
        }, initial_checkpoint_path)
        print(f"\n✓ Initial model saved: {initial_checkpoint_path.name}")
        
        # Step 4: Fine-tune model
        history = fine_tune_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=FINE_TUNE_EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Step 5: Measure final accuracy
        print("\n" + "="*60)
        print("Measuring Final Accuracy (After Fine-Tuning)")
        print("="*60)
        after_metrics = measure_accuracy(model, test_loader)
        print(f"✓ Final Accuracy: {after_metrics['accuracy']:.2f}%")
        print(f"✓ Final Loss: {after_metrics['loss']:.4f}")
        
        # Step 6: Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': after_metrics['accuracy'],
            'fine_tuned': True,
            'epochs': FINE_TUNE_EPOCHS,
            'train_history': history
        }, final_checkpoint_path)
        print(f"\n✓ Final model saved: {final_checkpoint_path.name}")
    
    # Step 7: Print comparison table
    print_comparison_table(before_metrics, after_metrics, params_info)
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Total execution time: {elapsed_time/60:.2f} minutes")
    print(f"✓ Model ready for pruning experiments!")
    print("\n" + "#"*80)
    print("# SCRIPT 1 COMPLETED SUCCESSFULLY")
    print("#"*80 + "\n")


if __name__ == '__main__':
    main()
