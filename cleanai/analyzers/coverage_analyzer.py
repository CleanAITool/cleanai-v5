"""
Neuron Coverage Analyzer Module

This module provides functionality to analyze neuron activation patterns
across test data and compute coverage metrics for each channel/neuron.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import gc


class ActivationHook:
    """Hook class to capture layer activations during forward pass."""
    
    def __init__(self, module: nn.Module, layer_name: str):
        """
        Initialize activation hook.
        
        Args:
            module: PyTorch module to attach hook to
            layer_name: Name identifier for the layer
        """
        self.module = module
        self.layer_name = layer_name
        # Use running statistics instead of storing all activations
        self.running_sum = None
        self.running_sq_sum = None
        self.sample_count = 0
        self.hook = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        Hook function to capture outputs and update running statistics.
        
        Args:
            module: The module being hooked
            input: Input tensors to the module
            output: Output tensor from the module
        """
        with torch.no_grad():
            # Process immediately instead of storing
            batch_stats = self._compute_batch_stats(output.detach())
            
            if self.running_sum is None:
                self.running_sum = batch_stats.cpu()
                self.sample_count = output.size(0)
            else:
                self.running_sum = self.running_sum + batch_stats.cpu()
                self.sample_count += output.size(0)
    
    def _compute_batch_stats(self, output: torch.Tensor) -> torch.Tensor:
        """
        Compute batch statistics for channel-wise coverage.
        
        Args:
            output: Output tensor [N, C, ...]
            
        Returns:
            Sum of activations per channel [C]
        """
        num_channels = output.shape[1]
        if output.dim() > 2:
            # Conv layers: average over spatial dimensions, sum over batch
            stats = output.view(output.shape[0], num_channels, -1).mean(dim=2).sum(dim=0)
        else:
            # Linear layers: sum over batch
            stats = output.sum(dim=0)
        return stats
    
    def get_mean_activation(self) -> Optional[torch.Tensor]:
        """Get mean activation per channel."""
        if self.sample_count == 0 or self.running_sum is None:
            return None
        return self.running_sum / self.sample_count
    
    def clear(self) -> None:
        """Clear stored statistics to free memory."""
        self.running_sum = None
        self.running_sq_sum = None
        self.sample_count = 0
    
    def remove(self) -> None:
        """Remove the hook from the module."""
        self.hook.remove()


class CoverageAnalyzer:
    """
    Analyzes neuron coverage based on activation patterns from test data.
    
    This class collects activations from specified layers during inference
    on test data and computes coverage metrics for each channel/neuron.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize coverage analyzer.
        
        Args:
            model: PyTorch model to analyze
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.hooks: Dict[str, ActivationHook] = {}
        self.coverage_scores: Dict[str, torch.Tensor] = {}
        
        # Target layer types for pruning
        self.target_types = (nn.Conv2d, nn.Linear)
        
    def register_hooks(self) -> None:
        """Register forward hooks on all target layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, self.target_types):
                self.hooks[name] = ActivationHook(module, name)
                
        print(f"Registered hooks on {len(self.hooks)} layers")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
    
    def collect_activations(
        self,
        test_loader: DataLoader,
        max_batches: Optional[int] = None
    ) -> None:
        """
        Collect activations from test data.
        
        Args:
            test_loader: DataLoader containing test samples
            max_batches: Maximum number of batches to process (None for all)
        """
        self.model.eval()
        self.model.to(self.device)
        
        print(f"Collecting activations from test data...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    if len(batch) < 1:
                        continue  # Skip empty batch
                    inputs = batch[0]
                else:
                    inputs = batch
                
                # Safety check for None inputs
                if inputs is None:
                    continue
                
                inputs = inputs.to(self.device)
                
                # Forward pass to trigger hooks
                try:
                    outputs = self.model(inputs)
                    # Clean up immediately
                    del outputs, inputs
                except Exception as e:
                    print(f"  Warning: Error processing batch {batch_idx}: {e}")
                    continue
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches")
                    # Clear GPU cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Activation collection complete")
    
    def compute_neuron_coverage(
        self,
        metric: str = 'normalized_mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute neuron coverage scores for each layer.
        
        Args:
            metric: Coverage metric to use:
                - 'normalized_mean': Average activation normalized by global max
                - 'frequency': Proportion of samples where neuron is active
                - 'mean_absolute': Absolute mean activation value
                - 'combined': Weighted combination of mean and frequency
        
        Returns:
            Dictionary mapping layer names to coverage scores (one per channel)
        """
        print(f"\nComputing neuron coverage using metric: {metric}")
        
        for layer_name, hook in self.hooks.items():
            mean_activation = hook.get_mean_activation()
            
            if mean_activation is None:
                print(f"  Warning: No activations collected for {layer_name}")
                continue
            
            # Compute coverage based on metric (using pre-computed mean)
            if metric == 'normalized_mean':
                # Normalize by global maximum
                global_max = mean_activation.max()
                if global_max > 0:
                    coverage = mean_activation / global_max
                else:
                    coverage = mean_activation
            elif metric == 'mean_absolute':
                coverage = torch.abs(mean_activation)
            elif metric == 'frequency':
                # Approximate frequency from mean (>0 indicates activation)
                coverage = (mean_activation > 0).float()
            elif metric == 'combined':
                # Simplified combined metric using mean
                global_max = mean_activation.max()
                if global_max > 0:
                    norm_mean = mean_activation / global_max
                else:
                    norm_mean = mean_activation
                coverage = torch.abs(norm_mean)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            self.coverage_scores[layer_name] = coverage
            
            print(f"  {layer_name}: coverage shape {coverage.shape}, "
                  f"min={coverage.min():.6f}, max={coverage.max():.6f}, "
                  f"mean={coverage.mean():.6f}")
        
        return self.coverage_scores
    
    def clear_activations(self) -> None:
        """Clear all stored activations to free memory."""
        for hook in self.hooks.values():
            hook.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_coverage_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary of coverage scores.
        
        Returns:
            Dictionary with statistics for each layer
        """
        stats = {}
        
        for layer_name, scores in self.coverage_scores.items():
            stats[layer_name] = {
                'min': float(scores.min()),
                'max': float(scores.max()),
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'num_channels': len(scores),
                'zero_coverage_count': int((scores == 0).sum())
            }
        
        return stats
    
    def analyze(
        self,
        test_loader: DataLoader,
        metric: str = 'normalized_mean',
        max_batches: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Complete analysis pipeline: register hooks, collect data, compute coverage.
        
        Args:
            test_loader: DataLoader with test samples
            metric: Coverage metric to use
            max_batches: Maximum number of batches to process
            
        Returns:
            Dictionary mapping layer names to coverage scores
        """
        try:
            self.register_hooks()
            self.collect_activations(test_loader, max_batches)
            coverage_scores = self.compute_neuron_coverage(metric)
            return coverage_scores
        finally:
            self.clear_activations()
            # Note: We keep hooks registered for later use in pruning
