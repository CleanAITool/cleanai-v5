"""
Utility Functions for Coverage-Based Pruning

This module provides helper functions for model evaluation, visualization,
and other common tasks in the pruning pipeline.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List, Any
import time
from pathlib import Path


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with total and trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def count_flops(
    model: nn.Module,
    example_inputs: torch.Tensor,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Count FLOPs and MACs using torch_pruning utility.
    
    Args:
        model: PyTorch model
        example_inputs: Example input tensor
        verbose: Print detailed information
        
    Returns:
        Dictionary with MACs and parameter counts
    """
    try:
        import torch_pruning as tp
        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        
        results = {
            'macs': macs,
            'params': params,
            'gflops': macs * 2 / 1e9  # MACs to GFLOPs (1 MAC ≈ 2 FLOPs)
        }
        
        if verbose:
            print(f"MACs: {macs:,} ({macs/1e9:.2f}G)")
            print(f"Params: {params:,} ({params/1e6:.2f}M)")
            print(f"GFLOPs: {results['gflops']:.2f}")
        
        return results
    except Exception as e:
        print(f"Warning: Could not count FLOPs: {e}")
        return {'macs': 0, 'params': count_parameters(model)['total'], 'gflops': 0}


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


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        criterion: Loss function (optional)
        device: Device to run on
        verbose: Print progress
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
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
            
            # Compute loss
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
    
    results = {
        'total_samples': total_samples,
    }
    
    if criterion is not None and total_samples > 0:
        results['avg_loss'] = total_loss / total_samples
    
    if total_samples > 0:
        results['accuracy'] = total_correct / total_samples
    
    if verbose:
        print()  # New line after progress
    
    return results


def compare_models(
    original_model: nn.Module,
    pruned_model: nn.Module,
    example_inputs: torch.Tensor,
    test_loader: Optional[DataLoader] = None,
    criterion: Optional[nn.Module] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Dict[str, Any]:
    """
    Compare original and pruned models across multiple metrics.
    
    Args:
        original_model: Original unpruned model
        pruned_model: Pruned model
        example_inputs: Example input for FLOPs counting
        test_loader: Test data loader for accuracy evaluation (optional)
        criterion: Loss function for evaluation (optional)
        device: Device to run on
        
    Returns:
        Dictionary with comparison metrics
    """
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    # Parameter counts
    original_params = count_parameters(original_model)
    pruned_params = count_parameters(pruned_model)
    
    param_reduction = (original_params['total'] - pruned_params['total']) / original_params['total']
    
    print(f"\nParameters:")
    print(f"  Original:  {original_params['total']:,}")
    print(f"  Pruned:    {pruned_params['total']:,}")
    print(f"  Reduction: {param_reduction:.2%}")
    
    # FLOPs count
    original_flops = count_flops(original_model, example_inputs)
    pruned_flops = count_flops(pruned_model, example_inputs)
    
    flops_reduction = (original_flops['macs'] - pruned_flops['macs']) / original_flops['macs'] if original_flops['macs'] > 0 else 0
    
    print(f"\nFLOPs:")
    print(f"  Original:  {original_flops['gflops']:.2f} GFLOPs")
    print(f"  Pruned:    {pruned_flops['gflops']:.2f} GFLOPs")
    print(f"  Reduction: {flops_reduction:.2%}")
    
    # Inference time
    print(f"\nMeasuring inference time...")
    original_time = measure_inference_time(original_model, example_inputs, device, num_runs=50)
    pruned_time = measure_inference_time(pruned_model, example_inputs, device, num_runs=50)
    
    speedup = original_time['mean_ms'] / pruned_time['mean_ms'] if pruned_time['mean_ms'] > 0 else 0
    
    print(f"\nInference Time (mean ± std):")
    print(f"  Original:  {original_time['mean_ms']:.2f} ± {original_time['std_ms']:.2f} ms")
    print(f"  Pruned:    {pruned_time['mean_ms']:.2f} ± {pruned_time['std_ms']:.2f} ms")
    print(f"  Speedup:   {speedup:.2f}x")
    
    results = {
        'parameters': {
            'original': original_params['total'],
            'pruned': pruned_params['total'],
            'reduction': param_reduction
        },
        'flops': {
            'original_gflops': original_flops['gflops'],
            'pruned_gflops': pruned_flops['gflops'],
            'reduction': flops_reduction
        },
        'inference_time': {
            'original_ms': original_time['mean_ms'],
            'pruned_ms': pruned_time['mean_ms'],
            'speedup': speedup
        }
    }
    
    # Accuracy evaluation if test_loader provided
    if test_loader is not None:
        print(f"\nEvaluating accuracy...")
        
        original_acc = evaluate_model(original_model, test_loader, criterion, device, verbose=False)
        pruned_acc = evaluate_model(pruned_model, test_loader, criterion, device, verbose=False)
        
        acc_drop = original_acc.get('accuracy', 0) - pruned_acc.get('accuracy', 0)
        
        print(f"\nAccuracy:")
        print(f"  Original:  {original_acc.get('accuracy', 0):.4f}")
        print(f"  Pruned:    {pruned_acc.get('accuracy', 0):.4f}")
        print(f"  Drop:      {acc_drop:.4f} ({acc_drop*100:.2f}%)")
        
        results['accuracy'] = {
            'original': original_acc.get('accuracy', 0),
            'pruned': pruned_acc.get('accuracy', 0),
            'drop': acc_drop
        }
    
    print("="*60 + "\n")
    
    return results


def save_model(
    model: nn.Module,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save pruned model with metadata.
    
    Args:
        model: PyTorch model to save
        save_path: Path to save the model
        metadata: Optional metadata to save with the model
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear gradients to reduce file size
    model.zero_grad()
    
    # Prepare save dict
    save_dict = {
        'model': model,
        'state_dict': model.state_dict(),
    }
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, save_path)
    print(f"Model saved to: {save_path}")


def load_model(
    load_path: str,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
    """
    Load pruned model with metadata.
    
    Args:
        load_path: Path to load the model from
        device: Device to load the model to
        
    Returns:
        Tuple of (model, metadata)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(load_path, map_location=device)
    
    model = checkpoint.get('model')
    metadata = checkpoint.get('metadata')
    
    print(f"Model loaded from: {load_path}")
    
    if metadata is not None:
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    return model, metadata


def print_model_summary(model: nn.Module, example_inputs: torch.Tensor) -> None:
    """
    Print a summary of the model architecture and size.
    
    Args:
        model: PyTorch model
        example_inputs: Example input tensor
    """
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    
    # Parameter count
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total:       {params['total']:,} ({params['total']/1e6:.2f}M)")
    print(f"  Trainable:   {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")
    
    # FLOPs
    flops = count_flops(model, example_inputs)
    print(f"\nComputational Cost:")
    print(f"  MACs:   {flops['macs']:,} ({flops['macs']/1e9:.2f}G)")
    print(f"  GFLOPs: {flops['gflops']:.2f}")
    
    # Layer count by type
    layer_types = {}
    for module in model.modules():
        module_type = type(module).__name__
        layer_types[module_type] = layer_types.get(module_type, 0) + 1
    
    print(f"\nLayer Types:")
    for layer_type, count in sorted(layer_types.items(), key=lambda x: -x[1])[:10]:
        print(f"  {layer_type}: {count}")
    
    print("="*60 + "\n")
