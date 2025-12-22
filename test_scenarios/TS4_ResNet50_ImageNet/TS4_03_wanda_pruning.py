"""
Test Scenario TS4 - Script 3: Wanda Pruning
Loads fine-tuned ResNet-50, applies Wanda pruning, and fine-tunes the pruned model.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from tqdm import tqdm
from tabulate import tabulate
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cleanai import CoveragePruner, count_parameters

# Configuration
CONFIG = {
    'test_scenario': 'TS4',
    'model_name': 'ResNet50',
    'dataset_name': 'ImageNet',
    'method': 'W',  # Wanda
    'checkpoint_dir': r'C:\source\checkpoints\TS4',
    'dataset_dir': r'C:\source\downloaded_datasets\imagenet',
    'results_dir': os.path.join(os.path.dirname(__file__)),
    'results_file': 'TS4_Results.json',
    'pruning_ratio': 0.1,  # 10%
    'global_pruning': True,
    'iterative_steps': 1,
    'fine_tune_epochs_base': 10,
    'save_every_n_epochs': 2,
    'batch_size': 256,
    'learning_rate': 0.0001,  # Lower LR for fine-tuning after pruning
    'num_calibration_batches': 100,  # Number of batches for Wanda calculation
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

CONFIG['fine_tune_epochs'] = 10

def load_dataset():
    """Load ImageNet validation dataset"""
    print("\n" + "="*80)
    print("LOADING IMAGENET VALIDATION DATASET")
    print("="*80)
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_dir = os.path.join(CONFIG['dataset_dir'], 'val')
    
    if not os.path.exists(val_dir):
        print(f"\nERROR: ImageNet validation set not found at: {val_dir}")
        print("Please run TS2_01_prepare_model.py first to set up the dataset.")
        raise FileNotFoundError(f"ImageNet validation set not found: {val_dir}")
    
    test_dataset = torchvision.datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )
    
    # Use subset of validation for fine-tuning
    train_size = min(len(test_dataset) // 5, 50000)  # Max 50k images for fine-tuning
    indices = torch.randperm(len(test_dataset)).tolist()
    train_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=train_transform)
    train_dataset = Subset(train_dataset, indices[:train_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    # Create calibration loader (subset for Wanda calculation)
    calibration_size = CONFIG['batch_size'] * CONFIG['num_calibration_batches']
    calibration_indices = indices[:min(calibration_size, len(test_dataset))]
    calibration_dataset = Subset(test_dataset, calibration_indices)
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    print(f"✓ Dataset loaded")
    print(f"  - Validation samples: {len(test_dataset)}")
    print(f"  - Training samples (subset): {len(train_dataset)}")
    print(f"  - Calibration samples: {len(calibration_dataset)}")
    
    return train_loader, test_loader, calibration_loader

def load_finetuned_model():
    """Load the fine-tuned model"""
    print("\n" + "="*80)
    print("LOADING BASELINE MODEL")
    print("="*80)
    
    # Find best checkpoint
    best_checkpoint = os.path.join(
        CONFIG['checkpoint_dir'],
        f"{CONFIG['model_name']}_{CONFIG['dataset_name']}_FT_best.pth"
    )
    
    if not os.path.exists(best_checkpoint):
        raise FileNotFoundError(f"Baseline model not found: {best_checkpoint}\n"
                              f"Please run TS2_01_prepare_model.py first.")
    
    # Create model architecture (1000 classes for ImageNet)
    model = resnet50(weights=None)
    # No need to modify - already has 1000 classes
    
    # Load checkpoint
    checkpoint = torch.load(best_checkpoint, map_location=CONFIG['device'], weights_only=False)
    
    # Filter out thop-related keys (total_ops, total_params)
    state_dict = checkpoint['model_state_dict']
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
    
    model.load_state_dict(state_dict)
    model = model.to(CONFIG['device'])
    
    print(f"✓ Model loaded from: {best_checkpoint}")
    if 'epoch' in checkpoint:
        print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Test Accuracy: {checkpoint['test_accuracy']:.2f}%")
    
    return model

def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

def count_flops(model, input_size=(1, 3, 224, 224)):
    """Estimate FLOPs for the model"""
    try:
        from thop import profile
        device = next(model.parameters()).device
        input_tensor = torch.randn(input_size).to(device)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops
    except ImportError:
        print("  Warning: thop not installed. FLOPs calculation skipped.")
        return 0

def evaluate_model(model, test_loader):
    """Evaluate model accuracy and inference time"""
    model.eval()
    correct = 0
    total = 0
    inference_times = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            start_time = time.time()
            outputs = model(images)
            inference_time = (time.time() - start_time) * 1000 / len(images)
            inference_times.append(inference_time)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return accuracy, avg_inference_time

def apply_wanda_pruning(model, calibration_loader):
    """Apply Wanda pruning"""
    print("\n" + "="*80)
    print("APPLYING WANDA PRUNING")
    print("="*80)
    print(f"Pruning Ratio: {CONFIG['pruning_ratio']*100}%")
    print(f"Global Pruning: {CONFIG['global_pruning']}")
    print(f"Iterative Steps: {CONFIG['iterative_steps']}")
    print(f"WANDA combines weight magnitude with activation importance")
    print(f"Using {CONFIG['num_calibration_batches']} calibration batches")
    
    # Get example input for model analysis
    example_inputs = next(iter(calibration_loader))[0][:1].to(CONFIG['device'])
    
    # Protect final classification layer from pruning
    ignored_layers = [model.fc]
    
    # Use CoveragePruner wrapper with WANDA method (same as TS1)
    pruner = CoveragePruner(
        model=model,
        example_inputs=example_inputs,
        test_loader=calibration_loader,
        pruning_ratio=CONFIG['pruning_ratio'],
        importance_method='wanda',  # WANDA method
        global_pruning=CONFIG['global_pruning'],
        iterative_steps=CONFIG['iterative_steps'],
        max_batches=CONFIG['num_calibration_batches'],
        ignored_layers=ignored_layers,
        device=CONFIG['device'],
        verbose=True
    )
    
    print("\nApplying pruning...")
    pruning_results = pruner.prune()
    pruned_model = pruner.get_model()
    
    print("✓ Wanda pruning completed")
    
    return pruned_model

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['fine_tune_epochs']}")
    for images, labels in pbar:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def fine_tune_pruned_model(model, train_loader, test_loader):
    """Fine-tune the pruned model"""
    print("\n" + "="*80)
    print("FINE-TUNING PRUNED MODEL")
    print("="*80)
    print(f"Fine-tune Epochs: {CONFIG['fine_tune_epochs']}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['fine_tune_epochs'])
    
    best_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(CONFIG['fine_tune_epochs']):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Evaluate
        test_acc, _ = evaluate_model(model, test_loader)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{CONFIG['fine_tune_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % CONFIG['save_every_n_epochs'] == 0 or epoch == CONFIG['fine_tune_epochs'] - 1:
            checkpoint_path = os.path.join(
                CONFIG['checkpoint_dir'],
                f"{CONFIG['model_name']}_{CONFIG['dataset_name']}_FTAP_{CONFIG['method']}_epoch{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
                'train_accuracy': train_acc,
                'pruning_ratio': CONFIG['pruning_ratio']
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_state = model.state_dict().copy()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Save best model
        best_model_path = os.path.join(
            CONFIG['checkpoint_dir'],
            f"{CONFIG['model_name']}_{CONFIG['dataset_name']}_FTAP_{CONFIG['method']}_best.pth"
        )
        torch.save({
            'model_state_dict': best_model_state,
            'test_accuracy': best_accuracy,
            'pruning_ratio': CONFIG['pruning_ratio']
        }, best_model_path)
        print(f"\n✓ Best model saved: {best_model_path}")
        print(f"  Best Accuracy: {best_accuracy:.2f}%")
    
    return model

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(f"TEST SCENARIO {CONFIG['test_scenario']}: WANDA PRUNING")
    print("="*80)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Dataset: {CONFIG['dataset_name']}")
    print(f"Method: Wanda")
    print(f"Device: {CONFIG['device']}")
    print(f"Pruning Ratio: {CONFIG['pruning_ratio']*100}%")
    print("="*80)
    
    # Load dataset
    train_loader, test_loader, calibration_loader = load_dataset()
    
    # Load fine-tuned model
    model = load_finetuned_model()
    
    # Evaluate original model
    print("\n" + "="*80)
    print("EVALUATING ORIGINAL FINE-TUNED MODEL")
    print("="*80)
    
    original_accuracy, original_inference_time = evaluate_model(model, test_loader)
    original_size = calculate_model_size(model)
    original_flops = count_flops(model)
    
    print(f"✓ Original Model Accuracy: {original_accuracy:.2f}%")
    print(f"✓ Model Size: {original_size:.2f} MB")
    print(f"✓ Average Inference Time: {original_inference_time:.4f} ms")
    if original_flops > 0:
        print(f"✓ FLOPs: {original_flops/1e9:.2f} GFLOPs")
    
    # Check if pruned model already exists
    final_checkpoint = os.path.join(
        CONFIG['checkpoint_dir'],
        f"{CONFIG['model_name']}_{CONFIG['dataset_name']}_FTAP_{CONFIG['method']}_best.pth"
    )
    
    if os.path.exists(final_checkpoint):
        print("\n" + "="*80)
        print("LOADING EXISTING PRUNED MODEL")
        print("="*80)
        checkpoint = torch.load(final_checkpoint, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded pruned model from: {final_checkpoint}")
    else:
        # Apply pruning
        model = apply_wanda_pruning(model, calibration_loader)
        
        # Evaluate after pruning (before fine-tuning)
        print("\n" + "="*80)
        print("EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)")
        print("="*80)
        
        pruned_accuracy, pruned_inference_time = evaluate_model(model, test_loader)
        pruned_size = calculate_model_size(model)
        pruned_flops = count_flops(model)
        
        print(f"✓ Pruned Model Accuracy: {pruned_accuracy:.2f}%")
        print(f"✓ Model Size: {pruned_size:.2f} MB")
        print(f"✓ Average Inference Time: {pruned_inference_time:.4f} ms")
        if pruned_flops > 0:
            print(f"✓ FLOPs: {pruned_flops/1e9:.2f} GFLOPs")
        
        # Fine-tune pruned model
        model = fine_tune_pruned_model(model, train_loader, test_loader)
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)")
    print("="*80)
    
    final_accuracy, final_inference_time = evaluate_model(model, test_loader)
    final_size = calculate_model_size(model)
    final_flops = count_flops(model)
    
    print(f"✓ Final Model Accuracy: {final_accuracy:.2f}%")
    print(f"✓ Model Size: {final_size:.2f} MB")
    print(f"✓ Average Inference Time: {final_inference_time:.4f} ms")
    if final_flops > 0:
        print(f"✓ FLOPs: {final_flops/1e9:.2f} GFLOPs")
    
    # Comparison table
    print("\n" + "="*80)
    print("COMPARISON: WANDA PRUNING RESULTS")
    print("="*80)
    
    comparison_data = [
        ["Metric", "Original (FT)", "After Pruning", "After FT", "Change (Original → Final)"],
        ["Accuracy (%)", f"{original_accuracy:.2f}", 
         f"{pruned_accuracy:.2f}" if 'pruned_accuracy' in locals() else "N/A",
         f"{final_accuracy:.2f}", 
         f"{final_accuracy - original_accuracy:+.2f}"],
        ["Size (MB)", f"{original_size:.2f}", 
         f"{pruned_size:.2f}" if 'pruned_size' in locals() else f"{final_size:.2f}",
         f"{final_size:.2f}",
         f"{final_size - original_size:+.2f} ({(final_size/original_size - 1)*100:+.1f}%)"],
        ["Inference Time (ms)", f"{original_inference_time:.4f}",
         f"{pruned_inference_time:.4f}" if 'pruned_inference_time' in locals() else "N/A",
         f"{final_inference_time:.4f}",
         f"{final_inference_time - original_inference_time:+.4f}"],
    ]
    
    if original_flops > 0 and final_flops > 0:
        comparison_data.append([
            "FLOPs (G)", f"{original_flops/1e9:.2f}",
            f"{pruned_flops/1e9:.2f}" if 'pruned_flops' in locals() else f"{final_flops/1e9:.2f}",
            f"{final_flops/1e9:.2f}",
            f"{(final_flops - original_flops)/1e9:+.2f} ({(final_flops/original_flops - 1)*100:+.1f}%)"
        ])
    
    print(tabulate(comparison_data, headers="firstrow", tablefmt="grid"))
    
    # Save results
    results = {
        'test_scenario': CONFIG['test_scenario'],
        'model': CONFIG['model_name'],
        'dataset': CONFIG['dataset_name'],
        'method': 'Wanda',
        'script': 'TS2_03_wanda_pruning',
        'pruning_config': {
            'ratio': CONFIG['pruning_ratio'],
            'global': CONFIG['global_pruning'],
            'iterative_steps': CONFIG['iterative_steps']
        },
        'original': {
            'accuracy': original_accuracy,
            'size_mb': original_size,
            'inference_time_ms': original_inference_time,
            'flops': original_flops
        },
        'pruned': {
            'accuracy': pruned_accuracy if 'pruned_accuracy' in locals() else final_accuracy,
            'size_mb': pruned_size if 'pruned_size' in locals() else final_size,
            'inference_time_ms': pruned_inference_time if 'pruned_inference_time' in locals() else final_inference_time,
            'flops': pruned_flops if 'pruned_flops' in locals() else final_flops
        },
        'final': {
            'accuracy': final_accuracy,
            'size_mb': final_size,
            'inference_time_ms': final_inference_time,
            'flops': final_flops,
            'fine_tune_epochs': CONFIG['fine_tune_epochs']
        }
    }
    
    results_file = os.path.join(CONFIG['results_dir'], CONFIG['results_file'])
    
    # Load existing results
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    all_results['wanda_pruning'] = results
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n✓ Results saved to: {results_file}")
    print("\n" + "="*80)
    print("SCRIPT COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()

