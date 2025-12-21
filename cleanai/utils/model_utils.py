"""
Model Utilities Module

This module provides utility functions for model analysis, parameter counting,
and FLOPs computation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from pathlib import Path


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with 'total' and 'trainable' parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameter_stats(model: nn.Module) -> Dict[str, int]:
    """
    Get detailed parameter statistics for a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def count_flops(
    model: nn.Module,
    example_inputs: torch.Tensor,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Count FLOPs and MACs using torch_pruning utility.
    
    Args:
        model: PyTorch model
        example_inputs: Example input tensor
        verbose: Print detailed information
        
    Returns:
        Dictionary with MACs, parameters, and GFLOPs
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
        return {'macs': 0, 'params': count_parameters(model), 'gflops': 0}


def print_model_summary(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: torch.device = torch.device('cpu')
) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        device: Device to run on
    """
    try:
        from torchinfo import summary
        summary(model, input_size=input_size, device=device)
    except ImportError:
        print("torchinfo not installed. Showing basic info:")
        print(f"Total parameters: {count_parameters(model):,}")
        print("\nModel architecture:")
        print(model)


def save_model(
    model: nn.Module,
    save_path: str,
    metadata: Dict = None
) -> None:
    """
    Save model weights and optional metadata.
    
    Args:
        model: PyTorch model
        save_path: Path to save the model
        metadata: Optional metadata dictionary
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"Model saved to: {save_path}")


def load_model(
    model: nn.Module,
    load_path: str,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """
    Load model weights from file.
    
    Args:
        model: PyTorch model (architecture must match)
        load_path: Path to load the model from
        device: Device to load the model on
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded from: {load_path}")
    return model


def compare_models_params(
    original_model: nn.Module,
    pruned_model: nn.Module,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compare parameter counts between original and pruned models.
    
    Args:
        original_model: Original model
        pruned_model: Pruned model
        verbose: Print comparison
        
    Returns:
        Dictionary with comparison metrics
    """
    original_params = count_parameters(original_model)
    pruned_params = count_parameters(pruned_model)
    reduction = (original_params - pruned_params) / original_params
    
    results = {
        'original_params': original_params,
        'pruned_params': pruned_params,
        'reduction_ratio': reduction,
        'reduction_percent': reduction * 100
    }
    
    if verbose:
        print(f"\nParameter Comparison:")
        print(f"  Original: {original_params:,}")
        print(f"  Pruned:   {pruned_params:,}")
        print(f"  Reduced:  {original_params - pruned_params:,} ({reduction*100:.2f}%)")
    
    return results
