"""
WANDA Pruning Example

This example demonstrates how to use the WANDA (Weight AND Activation)
importance criterion for neural network pruning.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import sys
sys.path.append('..')

from cleanai import CoveragePruner, evaluate_model, count_parameters, print_model_summary


def get_cifar10_dataloaders(batch_size=128):
    """Get CIFAR-10 dataloaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return test_loader


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained model
    print("\nLoading ResNet-18 pretrained model...")
    model = models.resnet18(pretrained=True)
    model = model.to(device)
    
    # Get data
    print("Loading CIFAR-10 test data...")
    test_loader = get_cifar10_dataloaders(batch_size=128)
    
    # Evaluate original model
    print("\n" + "="*60)
    print("ORIGINAL MODEL")
    print("="*60)
    print_model_summary(model, (1, 3, 224, 224))
    original_params = count_parameters(model)
    print(f"\nTotal Parameters: {original_params:,}")
    
    # Evaluate on CIFAR-10
    print("\nEvaluating on CIFAR-10...")
    original_acc = evaluate_model(model, test_loader, device)
    print(f"Original Accuracy: {original_acc:.2f}%")
    
    # ============================================
    # WANDA PRUNING
    # ============================================
    print("\n" + "="*60)
    print("WANDA PRUNING (Weight × Activation)")
    print("="*60)
    
    # Create example input for graph tracing
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    
    # Create WANDA pruner
    pruner = CoveragePruner(
        model=model,
        example_inputs=example_inputs,
        test_loader=test_loader,
        pruning_ratio=0.3,  # Prune 30% of channels
        importance_method='wanda',  # Use WANDA method
        global_pruning=True,
        iterative_steps=1,
        max_batches=50,  # Use 50 batches for calibration
        device=device,
        verbose=True
    )
    
    # Execute pruning
    print("\nExecuting WANDA pruning...")
    pruned_model = pruner.prune()
    
    # Evaluate pruned model
    print("\n" + "="*60)
    print("PRUNED MODEL (WANDA)")
    print("="*60)
    print_model_summary(pruned_model, (1, 3, 224, 224))
    pruned_params = count_parameters(pruned_model)
    print(f"\nTotal Parameters: {pruned_params:,}")
    print(f"Reduction: {(1 - pruned_params/original_params)*100:.2f}%")
    
    # Evaluate pruned model
    print("\nEvaluating pruned model on CIFAR-10...")
    pruned_acc = evaluate_model(pruned_model, test_loader, device)
    print(f"Pruned Accuracy: {pruned_acc:.2f}%")
    print(f"Accuracy Drop: {original_acc - pruned_acc:.2f}%")
    
    # ============================================
    # COMPARISON WITH OTHER METHODS
    # ============================================
    print("\n" + "="*60)
    print("COMPARISON: WANDA vs COVERAGE vs MAGNITUDE")
    print("="*60)
    
    methods = {
        'WANDA': 'wanda',
        'Coverage': 'coverage',
        'Magnitude': 'magnitude'
    }
    
    results = {}
    
    for method_name, method_key in methods.items():
        print(f"\n--- Testing {method_name} ---")
        
        # Reload model
        test_model = models.resnet18(pretrained=True).to(device)
        
        # Create pruner
        test_pruner = CoveragePruner(
            model=test_model,
            example_inputs=example_inputs,
            test_loader=test_loader,
            pruning_ratio=0.3,
            importance_method=method_key,
            global_pruning=True,
            iterative_steps=1,
            max_batches=50 if method_key != 'magnitude' else None,
            device=device,
            verbose=False
        )
        
        # Prune
        pruned = test_pruner.prune()
        
        # Evaluate
        params = count_parameters(pruned)
        acc = evaluate_model(pruned, test_loader, device)
        
        results[method_name] = {
            'params': params,
            'accuracy': acc,
            'reduction': (1 - params/original_params) * 100
        }
        
        print(f"  Parameters: {params:,} ({results[method_name]['reduction']:.2f}% reduction)")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Accuracy Drop: {original_acc - acc:.2f}%")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<15} {'Params':<15} {'Reduction':<12} {'Accuracy':<10} {'Drop':<10}")
    print("-" * 60)
    print(f"{'Original':<15} {original_params:<15,} {'-':<12} {original_acc:<10.2f} {'-':<10}")
    for method_name, data in results.items():
        print(f"{method_name:<15} {data['params']:<15,} {data['reduction']:<12.2f}% "
              f"{data['accuracy']:<10.2f} {original_acc - data['accuracy']:<10.2f}%")
    
    print("\n" + "="*60)
    print("WANDA typically provides better accuracy retention than")
    print("pure magnitude or activation-based methods by considering")
    print("both weight importance and activation patterns.")
    print("="*60)


if __name__ == "__main__":
    main()
