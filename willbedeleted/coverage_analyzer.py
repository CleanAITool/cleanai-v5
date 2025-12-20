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
        self.activations: List[torch.Tensor] = []
        self.hook = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        Hook function to capture outputs.
        
        Args:
            module: The module being hooked
            input: Input tensors to the module
            output: Output tensor from the module
        """
        # Detach and move to CPU to save memory
        self.activations.append(output.detach().cpu())
    
    def clear(self) -> None:
        """Clear stored activations to free memory."""
        self.activations.clear()
    
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
                    inputs = batch[0]
                else:
                    inputs = batch
                
                inputs = inputs.to(self.device)
                
                # Forward pass to trigger hooks
                _ = self.model(inputs)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches")
        
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
            if len(hook.activations) == 0:
                print(f"  Warning: No activations collected for {layer_name}")
                continue
            
            # Stack all activations: [num_batches * batch_size, C, ...]
            all_activations = torch.cat(hook.activations, dim=0)
            
            # Compute coverage based on metric
            if metric == 'normalized_mean':
                coverage = self._compute_normalized_mean(all_activations)
            elif metric == 'frequency':
                coverage = self._compute_frequency(all_activations)
            elif metric == 'mean_absolute':
                coverage = self._compute_mean_absolute(all_activations)
            elif metric == 'combined':
                coverage = self._compute_combined(all_activations)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            self.coverage_scores[layer_name] = coverage
            
            print(f"  {layer_name}: coverage shape {coverage.shape}, "
                  f"min={coverage.min():.6f}, max={coverage.max():.6f}, "
                  f"mean={coverage.mean():.6f}")
        
        return self.coverage_scores
    
    def _compute_normalized_mean(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized mean activation per channel.
        
        Args:
            activations: Tensor of shape [N, C, ...] where N is number of samples
            
        Returns:
            Coverage scores of shape [C]
        """
        # For Conv2d: [N, C, H, W] -> average over N, H, W -> [C]
        # For Linear: [N, C] -> average over N -> [C]
        
        num_channels = activations.shape[1]
        
        # Reshape to [N, C, -1] and take mean over spatial dimensions
        if activations.dim() > 2:
            activations = activations.view(activations.shape[0], num_channels, -1)
            channel_activations = activations.mean(dim=2)  # [N, C]
        else:
            channel_activations = activations  # [N, C]
        
        # Compute mean across all samples
        mean_activation = channel_activations.mean(dim=0)  # [C]
        
        # Normalize by global maximum
        global_max = channel_activations.max()
        if global_max > 0:
            normalized = mean_activation / global_max
        else:
            normalized = mean_activation
        
        return normalized
    
    def _compute_frequency(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute activation frequency (proportion of samples where channel is active).
        
        Args:
            activations: Tensor of shape [N, C, ...]
            
        Returns:
            Coverage scores of shape [C]
        """
        num_samples = activations.shape[0]
        num_channels = activations.shape[1]
        
        # Reshape to [N, C, -1]
        if activations.dim() > 2:
            activations = activations.view(num_samples, num_channels, -1)
            # A channel is "active" in a sample if any spatial position is > 0
            channel_active = (activations > 0).any(dim=2).float()  # [N, C]
        else:
            channel_active = (activations > 0).float()  # [N, C]
        
        # Frequency = proportion of samples where channel is active
        frequency = channel_active.mean(dim=0)  # [C]
        
        return frequency
    
    def _compute_mean_absolute(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute mean absolute activation value per channel.
        
        Args:
            activations: Tensor of shape [N, C, ...]
            
        Returns:
            Coverage scores of shape [C]
        """
        num_channels = activations.shape[1]
        
        # Take absolute value
        abs_activations = torch.abs(activations)
        
        # Reshape to [N, C, -1]
        if abs_activations.dim() > 2:
            abs_activations = abs_activations.view(abs_activations.shape[0], num_channels, -1)
            channel_activations = abs_activations.mean(dim=2)  # [N, C]
        else:
            channel_activations = abs_activations  # [N, C]
        
        # Mean across all samples
        mean_abs = channel_activations.mean(dim=0)  # [C]
        
        return mean_abs
    
    def _compute_combined(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute combined coverage metric (mean × frequency).
        
        Args:
            activations: Tensor of shape [N, C, ...]
            
        Returns:
            Coverage scores of shape [C]
        """
        normalized_mean = self._compute_normalized_mean(activations)
        frequency = self._compute_frequency(activations)
        
        # Combined metric: geometric mean
        combined = torch.sqrt(normalized_mean * frequency)
        
        return combined
    
    def clear_activations(self) -> None:
        """Clear all stored activations to free memory."""
        for hook in self.hooks.values():
            hook.clear()
    
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
