"""
Evaluation Utilities Module

This module provides functions for model evaluation, accuracy measurement,
and performance benchmarking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import time


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    criterion: Optional[nn.Module] = None,
    verbose: bool = True
) -> float:
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run on
        criterion: Optional loss function
        verbose: Print progress
        
    Returns:
        Accuracy as percentage (0-100)
    """
    model.eval()
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch.to(device)
                targets = None
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss if criterion provided
            if criterion is not None and targets is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
            
            # Compute accuracy
            if targets is not None and outputs.dim() > 1:
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
            
            total_samples += inputs.size(0)
            
            if verbose and (batch_idx + 1) % 50 == 0:
                print(f"  Evaluated {batch_idx + 1}/{len(data_loader)} batches", end='\r')
    
    if verbose and total_samples > 0:
        print()  # New line
    
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
    
    return accuracy


def evaluate_with_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model with both accuracy and loss.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run on
        verbose: Print progress
        
    Returns:
        Dictionary with accuracy and loss
    """
    model.eval()
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            inputs, targets = batch[0].to(device), batch[1].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)
            
            if verbose and (batch_idx + 1) % 50 == 0:
                print(f"  Evaluated {batch_idx + 1}/{len(data_loader)} batches", end='\r')
    
    if verbose:
        print()
    
    return {
        'accuracy': (total_correct / total_samples) * 100,
        'loss': total_loss / total_samples,
        'total_samples': total_samples
    }


def measure_inference_time(
    model: nn.Module,
    example_inputs: torch.Tensor,
    device: torch.device,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Measure model inference time.
    
    Args:
        model: PyTorch model
        example_inputs: Example input tensor
        device: Device to run on
        num_runs: Number of inference runs for measurement
        warmup_runs: Number of warmup runs before measurement
        
    Returns:
        Dictionary with timing statistics (in milliseconds)
    """
    model.eval()
    model.to(device)
    example_inputs = example_inputs.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(example_inputs)
    
    # Synchronize for accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(example_inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    times_tensor = torch.tensor(times)
    
    return {
        'mean_ms': float(times_tensor.mean()),
        'std_ms': float(times_tensor.std()),
        'min_ms': float(times_tensor.min()),
        'max_ms': float(times_tensor.max()),
        'median_ms': float(times_tensor.median()),
    }


def compare_models(
    original_model: nn.Module,
    pruned_model: nn.Module,
    test_loader: DataLoader,
    example_inputs: torch.Tensor,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Dict[str, any]:
    """
    Comprehensive comparison between original and pruned models.
    
    Args:
        original_model: Original unpruned model
        pruned_model: Pruned model
        test_loader: Test data loader
        example_inputs: Example input for timing
        device: Device to run on
        
    Returns:
        Dictionary with comprehensive comparison metrics
    """
    from .model_utils import count_parameters, count_flops
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Parameters
    orig_params = count_parameters(original_model)
    pruned_params = count_parameters(pruned_model)
    param_reduction = (orig_params - pruned_params) / orig_params
    
    print(f"\nParameters:")
    print(f"  Original: {orig_params:,}")
    print(f"  Pruned:   {pruned_params:,}")
    print(f"  Reduced:  {orig_params - pruned_params:,} ({param_reduction*100:.2f}%)")
    
    # FLOPs
    orig_flops = count_flops(original_model, example_inputs)
    pruned_flops = count_flops(pruned_model, example_inputs)
    flops_reduction = (orig_flops['gflops'] - pruned_flops['gflops']) / orig_flops['gflops'] if orig_flops['gflops'] > 0 else 0
    
    print(f"\nFLOPs:")
    print(f"  Original: {orig_flops['gflops']:.2f} GFLOPs")
    print(f"  Pruned:   {pruned_flops['gflops']:.2f} GFLOPs")
    print(f"  Reduced:  {flops_reduction*100:.2f}%")
    
    # Accuracy
    print(f"\nEvaluating accuracy...")
    orig_acc = evaluate_model(original_model, test_loader, device, verbose=False)
    pruned_acc = evaluate_model(pruned_model, test_loader, device, verbose=False)
    acc_drop = orig_acc - pruned_acc
    
    print(f"  Original: {orig_acc:.2f}%")
    print(f"  Pruned:   {pruned_acc:.2f}%")
    print(f"  Drop:     {acc_drop:.2f}%")
    
    # Inference time
    print(f"\nMeasuring inference time...")
    orig_time = measure_inference_time(original_model, example_inputs, device, num_runs=50)
    pruned_time = measure_inference_time(pruned_model, example_inputs, device, num_runs=50)
    speedup = orig_time['mean_ms'] / pruned_time['mean_ms'] if pruned_time['mean_ms'] > 0 else 1.0
    
    print(f"  Original: {orig_time['mean_ms']:.2f} ± {orig_time['std_ms']:.2f} ms")
    print(f"  Pruned:   {pruned_time['mean_ms']:.2f} ± {pruned_time['std_ms']:.2f} ms")
    print(f"  Speedup:  {speedup:.2f}x")
    
    print("="*60)
    
    return {
        'params': {
            'original': orig_params,
            'pruned': pruned_params,
            'reduction': param_reduction
        },
        'flops': {
            'original': orig_flops['gflops'],
            'pruned': pruned_flops['gflops'],
            'reduction': flops_reduction
        },
        'accuracy': {
            'original': orig_acc,
            'pruned': pruned_acc,
            'drop': acc_drop
        },
        'inference_time': {
            'original_ms': orig_time['mean_ms'],
            'pruned_ms': pruned_time['mean_ms'],
            'speedup': speedup
        }
    }
