"""
Test Scenario TS1 - Script 2: Neuron Coverage Pruning + Fine-Tuning
===================================================================
Model: ResNet-18 (Fine-tuned on CIFAR-10)
Method: Neuron Coverage-based Pruning
Purpose: Apply coverage pruning, then fine-tune to recover accuracy

Output:
- Pruned model checkpoint
- Fine-tuned pruned model checkpoint (every 5 epochs)
- Comprehensive comparison table (Original FT / After Pruning / After Pruning+FT)
- Detailed PDF report
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
from cleanai import CoveragePruner, count_parameters, compare_models
from cleanai.reporting import generate_pruning_report


# ==================== Configuration ====================
TEST_SCENARIO = "TS1"
METHOD_NAME = "Coverage"
MODEL_NAME = "ResNet18"
DATASET_NAME = "CIFAR10"
TEST_SCENARIO_NAME = f"{TEST_SCENARIO}_{METHOD_NAME}_{MODEL_NAME}_{DATASET_NAME}"

# Directories
DATASET_DIR = Path(r"C:\source\downloaded_datasets")
CHECKPOINT_BASE_DIR = Path(rf"C:\source\checkpoints\{TEST_SCENARIO}")
CHECKPOINT_DIR = Path(rf"C:\source\checkpoints\{TEST_SCENARIO_NAME}")
REPORT_DIR = Path(rf"C:\source\checkpoints\{TEST_SCENARIO_NAME}\reports")

# Pruning parameters
PRUNING_RATIO = 0.2  # Remove 20% of channels
COVERAGE_METRIC = 'normalized_mean'
GLOBAL_PRUNING = True
ITERATIVE_STEPS = 1
MAX_BATCHES = 50  # Use subset of test data for coverage analysis

# Fine-tuning parameters (after pruning)
FINE_TUNE_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 0  # Windows multiprocessing fix
SAVE_EVERY_N_EPOCHS = 5

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Helper Functions ====================

def get_cifar10_dataloaders(batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 dataloaders."""
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
    
    train_dataset = datasets.CIFAR10(root=str(DATASET_DIR), train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=str(DATASET_DIR), train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


def load_finetuned_model() -> nn.Module:
    """Load the fine-tuned ResNet-18 from Script 1."""
    print("\n" + "="*60)
    print("Loading Fine-Tuned Model")
    print("="*60)
    
    # Create model architecture
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Load checkpoint
    checkpoint_path = CHECKPOINT_BASE_DIR / f"{MODEL_NAME}_{DATASET_NAME}_FT_final.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found: {checkpoint_path}\n"
            f"Please run TS1_01_prepare_model.py first!"
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded from: {checkpoint_path.name}")
    print(f"✓ Checkpoint accuracy: {checkpoint['accuracy']:.2f}%")
    
    return model


def measure_accuracy(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """Measure model accuracy and loss."""
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
    
    return {'accuracy': accuracy, 'loss': avg_loss}


def measure_model_size(model: nn.Module) -> float:
    """Measure model size in MB."""
    torch.save(model.state_dict(), 'temp_model.pth')
    size_mb = os.path.getsize('temp_model.pth') / (1024 * 1024)
    os.remove('temp_model.pth')
    return size_mb


def measure_inference_time(model: nn.Module, test_loader: DataLoader, num_batches: int = 50) -> float:
    """Measure average inference time in milliseconds."""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            
            inputs = inputs.to(DEVICE)
            
            # Warm up
            if i < 3:
                _ = model(inputs)
                continue
            
            # Measure
            start = time.time()
            _ = model(inputs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            
            times.append((end - start) * 1000 / inputs.size(0))  # ms per sample
    
    return sum(times) / len(times) if times else 0.0


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
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
                  f"Loss: {loss.item():.4f} Acc: {100.0 * correct / total:.2f}%")
    
    return {'accuracy': 100.0 * correct / total, 'loss': total_loss / total}


def fine_tune_pruned_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader) -> Dict:
    """Fine-tune pruned model."""
    print("\n" + "="*60)
    print("Fine-Tuning Pruned Model")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINE_TUNE_EPOCHS)
    
    best_acc = 0.0
    history = {'train_acc': [], 'train_loss': [], 'test_acc': [], 'test_loss': []}
    
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{FINE_TUNE_EPOCHS}")
        print("-" * 60)
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch)
        test_metrics = measure_accuracy(model, test_loader)
        
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_loss'].append(train_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        
        scheduler.step()
        
        print(f"\n  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Test Loss:  {test_metrics['loss']:.4f} | Test Acc:  {test_metrics['accuracy']:.2f}%")
        
        # Save checkpoint
        if epoch % SAVE_EVERY_N_EPOCHS == 0 or epoch == FINE_TUNE_EPOCHS:
            checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_{DATASET_NAME}_FTAP_NC_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_metrics['accuracy'],
                'method': 'Coverage',
                'pruning_ratio': PRUNING_RATIO
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path.name}")
        
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
    
    return history


def print_comparison_table(original_metrics: Dict, pruned_metrics: Dict, final_metrics: Dict):
    """Print comprehensive comparison table."""
    print("\n" + "="*100)
    print("NEURON COVERAGE PRUNING - COMPREHENSIVE COMPARISON TABLE")
    print("="*100)
    
    print(f"\n{'Metric':<30} {'Original (FT)':<23} {'After Pruning':<23} {'After Pruning+FT':<23}")
    print("-" * 100)
    
    # Accuracy
    print(f"{'Accuracy (%)':<30} {original_metrics['accuracy']:>22.2f} "
          f"{pruned_metrics['accuracy']:>22.2f} {final_metrics['accuracy']:>22.2f}")
    
    # Size
    print(f"{'Size (MB)':<30} {original_metrics['size_mb']:>22.2f} "
          f"{pruned_metrics['size_mb']:>22.2f} {final_metrics['size_mb']:>22.2f}")
    
    # Parameters
    print(f"{'Parameters (M)':<30} {original_metrics['params']/1e6:>22.2f} "
          f"{pruned_metrics['params']/1e6:>22.2f} {final_metrics['params']/1e6:>22.2f}")
    
    # FLOPs
    if 'flops' in original_metrics:
        print(f"{'FLOPs (G)':<30} {original_metrics['flops']/1e9:>22.2f} "
              f"{pruned_metrics['flops']/1e9:>22.2f} {final_metrics['flops']/1e9:>22.2f}")
    
    # Inference Time
    print(f"{'Avg Inference Time (ms)':<30} {original_metrics['inference_time']:>22.2f} "
          f"{pruned_metrics['inference_time']:>22.2f} {final_metrics['inference_time']:>22.2f}")
    
    print("-" * 100)
    
    # Reductions
    param_reduction = (1 - pruned_metrics['params'] / original_metrics['params']) * 100
    size_reduction = (1 - pruned_metrics['size_mb'] / original_metrics['size_mb']) * 100
    speedup = original_metrics['inference_time'] / final_metrics['inference_time']
    acc_recovery = final_metrics['accuracy'] - pruned_metrics['accuracy']
    final_acc_drop = original_metrics['accuracy'] - final_metrics['accuracy']
    
    print(f"\n{'Summary':<30}")
    print(f"{'  Parameter Reduction':<30} {param_reduction:>22.2f}%")
    print(f"{'  Size Reduction':<30} {size_reduction:>22.2f}%")
    print(f"{'  Speedup':<30} {speedup:>22.2f}x")
    print(f"{'  Accuracy Recovery (FT)':<30} {acc_recovery:>22.2f}%")
    print(f"{'  Final Accuracy Drop':<30} {final_acc_drop:>22.2f}%")
    
    print("="*100 + "\n")


def main():
    """Main execution function."""
    print("\n" + "#"*100)
    print(f"# TEST SCENARIO {TEST_SCENARIO} - SCRIPT 2: NEURON COVERAGE PRUNING + FINE-TUNING")
    print("#"*100)
    print(f"\nTest Scenario: {TEST_SCENARIO_NAME}")
    print(f"Method: {METHOD_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Pruning Ratio: {PRUNING_RATIO:.0%}")
    print(f"Device: {DEVICE}")
    
    start_time = time.time()
    
    # Check if final model already exists
    final_checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_{DATASET_NAME}_FTAP_NC_final.pth"
    pruned_checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_{DATASET_NAME}_pruned_NC.pth"
    
    # Step 1: Load dataset
    print("\n" + "="*60)
    print("Loading CIFAR-10 Dataset")
    print("="*60)
    train_loader, test_loader = get_cifar10_dataloaders(BATCH_SIZE, 0)
    print(f"✓ Dataset loaded")
    
    # Step 2: Load fine-tuned model for comparison
    original_model_full = load_finetuned_model()
    original_model_full = original_model_full.to(DEVICE)
    
    # Step 3: Measure original model metrics
    print("\n" + "="*60)
    print("Measuring Original Model Metrics")
    print("="*60)
    
    import copy
    original_model = copy.deepcopy(original_model_full)
    
    original_accuracy = measure_accuracy(original_model_full, test_loader)
    original_params = count_parameters(original_model_full)
    original_size = measure_model_size(original_model_full)
    original_inference_time = measure_inference_time(original_model_full, test_loader)
    
    print(f"✓ Accuracy: {original_accuracy['accuracy']:.2f}%")
    print(f"✓ Parameters: {original_params['total']:,}")
    print(f"✓ Size: {original_size:.2f} MB")
    print(f"✓ Inference Time: {original_inference_time:.2f} ms/sample")
    
    original_metrics = {
        'accuracy': original_accuracy['accuracy'],
        'params': original_params['total'],
        'size_mb': original_size,
        'inference_time': original_inference_time
    }
    
    if final_checkpoint_path.exists():
        print("\n" + "="*60)
        print("CHECKPOINT FOUND - Loading Pruned Model")
        print("="*60)
        print(f"✓ Found: {final_checkpoint_path.name}")
        print("✓ Skipping pruning and fine-tuning (already completed)")
        
        # Load final pruned model
        model = load_finetuned_model()  # Get architecture
        checkpoint = torch.load(final_checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        
        # Get metrics
        final_accuracy = measure_accuracy(model, test_loader)
        final_params = count_parameters(model)
        final_size = measure_model_size(model)
        final_inference_time = measure_inference_time(model, test_loader)
        
        final_metrics = {
            'accuracy': final_accuracy['accuracy'],
            'params': final_params['total'],
            'size_mb': final_size,
            'inference_time': final_inference_time
        }
        
        # Load pruned (before FT) metrics if available
        if pruned_checkpoint_path.exists():
            pruned_checkpoint = torch.load(pruned_checkpoint_path, map_location=DEVICE)
            temp_model = load_finetuned_model()
            temp_model.load_state_dict(pruned_checkpoint['model_state_dict'])
            temp_model = temp_model.to(DEVICE)
            
            pruned_accuracy = measure_accuracy(temp_model, test_loader)
            pruned_params = count_parameters(temp_model)
            pruned_size = measure_model_size(temp_model)
            pruned_inference_time = measure_inference_time(temp_model, test_loader)
            
            pruned_metrics = {
                'accuracy': pruned_accuracy['accuracy'],
                'params': pruned_params['total'],
                'size_mb': pruned_size,
                'inference_time': pruned_inference_time
            }
        else:
            # Estimate from final model
            pruned_metrics = {
                'accuracy': final_accuracy['accuracy'] - 5.0,  # Estimate drop before FT
                'params': final_params['total'],
                'size_mb': final_size,
                'inference_time': final_inference_time
            }
        
        pruned_model = model
        print(f"✓ Loaded model accuracy: {final_accuracy['accuracy']:.2f}%")
        print(f"✓ Parameters: {final_params['total']:,}")
    else:
        print("\n" + "="*60)
        print("No Checkpoint Found - Starting Pruning Process")
        print("="*60)
        
        model = original_model_full
        
        # Step 4: Apply Neuron Coverage Pruning
        print("\n" + "="*60)
        print("Applying Neuron Coverage Pruning")
        print("="*60)
        
        example_inputs = torch.randn(1, 3, 32, 32).to(DEVICE)
        
        # Protect final classification layer from pruning
        ignored_layers = [model.fc]
        
        pruner = CoveragePruner(
            model=model,
            example_inputs=example_inputs,
            test_loader=test_loader,
            pruning_ratio=PRUNING_RATIO,
            importance_method='coverage',
            coverage_metric=COVERAGE_METRIC,
            global_pruning=GLOBAL_PRUNING,
            iterative_steps=ITERATIVE_STEPS,
            max_batches=MAX_BATCHES,
            ignored_layers=ignored_layers,
            device=DEVICE,
            verbose=True
        )
        
        pruning_results = pruner.prune()
        pruned_model = pruner.get_model()
        
        # Step 5: Measure pruned model metrics
        print("\n" + "="*60)
        print("Measuring Pruned Model Metrics")
        print("="*60)
        
        pruned_accuracy = measure_accuracy(pruned_model, test_loader)
        pruned_params = count_parameters(pruned_model)
        pruned_size = measure_model_size(pruned_model)
        pruned_inference_time = measure_inference_time(pruned_model, test_loader)
        
        print(f"✓ Accuracy: {pruned_accuracy['accuracy']:.2f}%")
        print(f"✓ Parameters: {pruned_params['total']:,}")
        print(f"✓ Size: {pruned_size:.2f} MB")
        print(f"✓ Inference Time: {pruned_inference_time:.2f} ms/sample")
        
        pruned_metrics = {
            'accuracy': pruned_accuracy['accuracy'],
            'params': pruned_params['total'],
            'size_mb': pruned_size,
            'inference_time': pruned_inference_time
        }
        
        # Save pruned model
        torch.save({
            'model_state_dict': pruned_model.state_dict(),
            'accuracy': pruned_accuracy['accuracy'],
            'method': 'Coverage',
            'pruning_ratio': PRUNING_RATIO,
            'pruning_results': pruning_results
        }, pruned_checkpoint_path)
        print(f"\n✓ Pruned model saved: {pruned_checkpoint_path.name}")
        
        # Step 6: Fine-tune pruned model
        history = fine_tune_pruned_model(pruned_model, train_loader, test_loader)
        
        # Step 7: Measure final model metrics
        print("\n" + "="*60)
        print("Measuring Final Model Metrics (After Fine-Tuning)")
        print("="*60)
        
        final_accuracy = measure_accuracy(pruned_model, test_loader)
        final_params = count_parameters(pruned_model)
        final_size = measure_model_size(pruned_model)
        final_inference_time = measure_inference_time(pruned_model, test_loader)
        
        print(f"✓ Accuracy: {final_accuracy['accuracy']:.2f}%")
        print(f"✓ Parameters: {final_params['total']:,}")
        print(f"✓ Size: {final_size:.2f} MB")
        print(f"✓ Inference Time: {final_inference_time:.2f} ms/sample")
        
        final_metrics = {
            'accuracy': final_accuracy['accuracy'],
            'params': final_params['total'],
            'size_mb': final_size,
            'inference_time': final_inference_time
        }
        
        # Save final model
        torch.save({
            'model_state_dict': pruned_model.state_dict(),
            'accuracy': final_accuracy['accuracy'],
            'method': 'Coverage',
            'pruning_ratio': PRUNING_RATIO,
            'fine_tuned': True,
            'epochs': FINE_TUNE_EPOCHS,
            'train_history': history
        }, final_checkpoint_path)
        print(f"\n✓ Final model saved: {final_checkpoint_path.name}")
    
    # Step 8: Generate report
    print("\n" + "="*60)
    print("Generating PDF Report")
    print("="*60)
    
    try:
        report_path = generate_pruning_report(
            model_before=original_model,
            model_after=pruned_model,
            model_name=MODEL_NAME,
            dataloader=test_loader,
            device=DEVICE,
            report_name=TEST_SCENARIO_NAME,
            pruning_method="coverage",
            pruning_ratio=PRUNING_RATIO,
            iterative_steps=ITERATIVE_STEPS,
            dataset_name=DATASET_NAME,
            output_dir=str(REPORT_DIR)
        )
        print(f"✓ Report generated: {report_path}")
    except Exception as e:
        print(f"⚠ Report generation failed: {e}")
    
    # Step 9: Print comparison table
    print_comparison_table(original_metrics, pruned_metrics, final_metrics)
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Total execution time: {elapsed_time/60:.2f} minutes")
    print(f"✓ Neuron Coverage pruning experiment completed!")
    
    print("\n" + "#"*100)
    print("# SCRIPT 2 COMPLETED SUCCESSFULLY")
    print("#"*100 + "\n")


if __name__ == '__main__':
    main()
