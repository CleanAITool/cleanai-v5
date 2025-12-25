"""
Neuron Coverage-based Importance Module

This module implements a custom importance criterion for Torch-Pruning
based on neuron coverage metrics computed from test data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import torch_pruning as tp
from ..analyzers.coverage_analyzer import CoverageAnalyzer


class NeuronCoverageImportance(tp.importance.Importance):
    """
    Importance estimator based on neuron coverage from test data.
    
    This class computes importance scores for pruning based on how active
    each neuron/channel is on test data. Channels with higher coverage
    (more active) are considered more important and will be kept.
    Channels with lower coverage (less active) will be pruned first.
    
    The importance score equals the coverage:
        importance = coverage
    
    This means higher coverage -> higher importance -> kept (not pruned).
    """
    
    def __init__(
        self,
        test_loader: DataLoader,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        coverage_metric: str = 'normalized_mean',
        max_batches: Optional[int] = None,
        epsilon: float = 1e-8
    ):
        """
        Initialize neuron coverage importance estimator.
        
        Args:
            test_loader: DataLoader containing test samples for coverage analysis
            device: Device to run inference on
            coverage_metric: Coverage metric to use:
                - 'normalized_mean': Average activation normalized by global max
                - 'frequency': Proportion of samples where neuron is active
                - 'mean_absolute': Absolute mean activation value
                - 'combined': Weighted combination of mean and frequency
            max_batches: Maximum number of batches to process (None for all)
            epsilon: Small constant to prevent division by zero
        """
        super().__init__()
        
        self.test_loader = test_loader
        self.device = device
        self.coverage_metric = coverage_metric
        self.max_batches = max_batches
        self.epsilon = epsilon
        
        # Coverage analyzer instance (will be initialized per model)
        self.analyzer: Optional[CoverageAnalyzer] = None
        self.coverage_scores: Dict[str, torch.Tensor] = {}
        self._coverage_computed = False
    
    def _ensure_coverage_computed(self, model: nn.Module) -> None:
        """
        Ensure coverage has been computed for the model.
        
        Args:
            model: PyTorch model to analyze
        """
        if self._coverage_computed:
            return
        
        print("\n" + "="*60)
        print("Computing Neuron Coverage on Test Data")
        print("="*60)
        
        # Initialize analyzer
        self.analyzer = CoverageAnalyzer(model, self.device)
        
        # Compute coverage scores
        self.coverage_scores = self.analyzer.analyze(
            test_loader=self.test_loader,
            metric=self.coverage_metric,
            max_batches=self.max_batches
        )
        
        # Print statistics
        print("\n" + "-"*60)
        print("Coverage Statistics:")
        print("-"*60)
        stats = self.analyzer.get_coverage_statistics()
        for layer_name, layer_stats in stats.items():
            print(f"\n{layer_name}:")
            print(f"  Channels: {layer_stats['num_channels']}")
            print(f"  Coverage - Min: {layer_stats['min']:.6f}, "
                  f"Max: {layer_stats['max']:.6f}, "
                  f"Mean: {layer_stats['mean']:.6f}")
            print(f"  Zero coverage neurons: {layer_stats['zero_coverage_count']}")
        print("="*60 + "\n")
        
        self._coverage_computed = True
    
    def _get_module_name(self, model: nn.Module, target_module: nn.Module) -> Optional[str]:
        """
        Find the name of a module in the model.
        
        Args:
            model: The model containing the module
            target_module: The module to find
            
        Returns:
            Module name or None if not found
        """
        for name, module in model.named_modules():
            if module is target_module:
                return name
        return None
    
    @torch.no_grad()
    def __call__(self, group: tp.dependency.Group) -> torch.Tensor:
        """
        Compute importance scores for a pruning group based on neuron coverage.
        
        Args:
            group: A Torch-Pruning dependency group containing layers to be pruned
            
        Returns:
            1-D tensor of importance scores (one per channel/neuron in the group)
            Higher importance = higher priority to KEEP (not prune)
        """
        # Get the model from the dependency graph
        model = group._DG.model
        
        # Ensure coverage has been computed
        self._ensure_coverage_computed(model)
        
        # Get the root layer of the group (the one being pruned)
        root_dep = group[0][0]  # First dependency in group
        root_module = root_dep.target.module
        root_idxs = group[0][1]  # Indices for this group
        
        # Find module name in the model
        module_name = self._get_module_name(model, root_module)
        
        if module_name is None:
            # Fallback: use magnitude-based importance
            print(f"Warning: Could not find module in model, using magnitude fallback")
            return self._magnitude_fallback(root_module, root_idxs)
        
        # Get coverage scores for this layer
        if module_name not in self.coverage_scores:
            print(f"Warning: No coverage scores for {module_name}, using magnitude fallback")
            return self._magnitude_fallback(root_module, root_idxs)
        
        coverage = self.coverage_scores[module_name]
        
        # Extract scores for the specific indices in this group
        # Convert indices to plain list if needed
        if hasattr(root_idxs[0], 'idx'):
            plain_idxs = [idx.idx for idx in root_idxs]
        else:
            plain_idxs = root_idxs
        
        channel_coverage = coverage[plain_idxs]
        
        # IMPORTANT: In Torch-Pruning, higher importance = higher priority to KEEP
        # Therefore: higher coverage = higher importance (keep active neurons)
        # Lower coverage neurons will be pruned first
        importance_scores = channel_coverage
        
        # Normalize to [0, 1] range for consistency
        if importance_scores.max() > importance_scores.min():
            importance_scores = (importance_scores - importance_scores.min()) / \
                              (importance_scores.max() - importance_scores.min())
        
        return importance_scores
    
    def _magnitude_fallback(
        self,
        module: nn.Module,
        idxs: list
    ) -> torch.Tensor:
        """
        Fallback to magnitude-based importance if coverage is unavailable.
        
        Args:
            module: The module to compute importance for
            idxs: Indices of channels to score
            
        Returns:
            Importance scores based on weight magnitude
        """
        # Get weight tensor
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data  # [out_channels, in_channels, k, k]
            # Compute L2 norm for each output channel
            if hasattr(idxs[0], 'idx'):
                plain_idxs = [idx.idx for idx in idxs]
            else:
                plain_idxs = idxs
            
            channel_weights = weight[plain_idxs]
            importance = torch.norm(channel_weights.view(len(plain_idxs), -1), p=2, dim=1)
            
        elif isinstance(module, nn.Linear):
            weight = module.weight.data  # [out_features, in_features]
            if hasattr(idxs[0], 'idx'):
                plain_idxs = [idx.idx for idx in idxs]
            else:
                plain_idxs = idxs
            
            neuron_weights = weight[plain_idxs]
            importance = torch.norm(neuron_weights, p=2, dim=1)
        else:
            # Default: uniform importance
            importance = torch.ones(len(idxs))
        
        # IMPORTANT: In Torch-Pruning, higher importance = keep
        # So higher magnitude = higher importance (standard magnitude-based pruning)
        # Normalize the importance scores
        if importance.max() > 0:
            importance = importance / importance.max()
        
        return importance
