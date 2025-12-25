"""
WANDA (Pruning by Weights AND Activations) Importance Module

This module implements the WANDA importance criterion which combines
weight magnitude with activation magnitude for effective pruning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import torch_pruning as tp
from ..analyzers.coverage_analyzer import CoverageAnalyzer


class WandaImportance(tp.importance.Importance):
    """
    WANDA (Pruning by Weights AND Activations) Importance Estimator.
    
    This method combines weight magnitude with activation magnitude to compute
    importance scores. It's particularly effective for pruning neural networks
    without requiring gradients or fine-tuning.
    
    Reference: "A Simple and Effective Pruning Approach for Large Language Models"
    https://arxiv.org/abs/2306.11695
    
    Importance Score = |Weight| × |Activation|
    
    Key Benefits:
    - Training-free (no gradients needed)
    - Fast one-shot pruning
    - Better than magnitude-only or activation-only methods
    - Effective for both CNNs and LLMs
    """
    
    def __init__(
        self,
        test_loader: DataLoader,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        max_batches: Optional[int] = None,
        epsilon: float = 1e-8,
        weight_norm: str = 'l2',  # 'l1' or 'l2'
        activation_metric: str = 'mean_abs'  # 'mean_abs', 'max', 'l2_norm'
    ):
        """
        Initialize WANDA importance estimator.
        
        Args:
            test_loader: DataLoader containing calibration samples
            device: Device to run inference on
            max_batches: Maximum number of batches to process (None for all)
            epsilon: Small constant to prevent division by zero
            weight_norm: Norm to use for weight magnitude ('l1' or 'l2')
            activation_metric: Metric to compute activation magnitude
                - 'mean_abs': Mean of absolute activations
                - 'max': Maximum absolute activation
                - 'l2_norm': L2 norm of activations
        """
        super().__init__()
        
        self.test_loader = test_loader
        self.device = device
        self.max_batches = max_batches
        self.epsilon = epsilon
        self.weight_norm = weight_norm
        self.activation_metric = activation_metric
        
        # Coverage analyzer to collect activations
        self.analyzer: Optional[CoverageAnalyzer] = None
        self.activation_scores: Dict[str, torch.Tensor] = {}
        self._activations_computed = False
    
    def _ensure_activations_computed(self, model: nn.Module) -> None:
        """
        Ensure activations have been computed for the model.
        
        Args:
            model: PyTorch model to analyze
        """
        if self._activations_computed:
            return
        
        print("\n" + "="*60)
        print("WANDA: Computing Weight × Activation Importance")
        print("="*60)
        
        # Initialize analyzer
        self.analyzer = CoverageAnalyzer(model, self.device)
        
        # Use coverage analyzer to collect activations
        # We'll compute mean absolute activation per channel
        coverage_scores = self.analyzer.analyze(
            test_loader=self.test_loader,
            metric='mean_absolute',  # Get mean absolute activations
            max_batches=self.max_batches
        )
        
        # Store activation scores
        self.activation_scores = coverage_scores
        
        # Print statistics
        print("\n" + "-"*60)
        print("Activation Statistics:")
        print("-"*60)
        stats = self.analyzer.get_coverage_statistics()
        for layer_name, layer_stats in stats.items():
            print(f"\n{layer_name}:")
            print(f"  Channels: {layer_stats['num_channels']}")
            print(f"  Activation - Min: {layer_stats['min']:.6f}, "
                  f"Max: {layer_stats['max']:.6f}, "
                  f"Mean: {layer_stats['mean']:.6f}")
        print("="*60 + "\n")
        
        self._activations_computed = True
    
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
    
    def _compute_weight_importance(
        self,
        module: nn.Module,
        idxs: list
    ) -> torch.Tensor:
        """
        Compute weight magnitude importance for given channels/neurons.
        
        Args:
            module: The module to compute importance for
            idxs: Indices of channels/neurons to score
            
        Returns:
            Weight magnitude scores
        """
        # Extract plain indices
        if hasattr(idxs[0], 'idx'):
            plain_idxs = [idx.idx for idx in idxs]
        else:
            plain_idxs = idxs
        
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data  # [out_channels, in_channels, k, k]
            channel_weights = weight[plain_idxs].flatten(1)  # [selected_channels, in*k*k]
            
        elif isinstance(module, nn.Linear):
            weight = module.weight.data  # [out_features, in_features]
            channel_weights = weight[plain_idxs]  # [selected_neurons, in_features]
            
        else:
            # Unsupported layer type, return uniform scores
            return torch.ones(len(plain_idxs), device=self.device)
        
        # Compute weight magnitude
        if self.weight_norm == 'l1':
            weight_scores = torch.abs(channel_weights).mean(dim=1)
        elif self.weight_norm == 'l2':
            weight_scores = torch.norm(channel_weights, p=2, dim=1)
        else:
            weight_scores = torch.norm(channel_weights, p=2, dim=1)
        
        return weight_scores
    
    @torch.no_grad()
    def __call__(self, group: tp.dependency.Group) -> torch.Tensor:
        """
        Compute WANDA importance scores: |Weight| × |Activation|
        
        Args:
            group: A Torch-Pruning dependency group
            
        Returns:
            1-D tensor of importance scores (lower = more important for pruning)
        """
        # Get the model from the dependency graph
        model = group._DG.model
        
        # Ensure activations have been computed
        self._ensure_activations_computed(model)
        
        # Get the root layer of the group
        root_dep = group[0][0]
        root_module = root_dep.target.module
        root_idxs = group[0][1]
        
        # Find module name in the model
        module_name = self._get_module_name(model, root_module)
        
        if module_name is None or module_name not in self.activation_scores:
            print(f"Warning: No activation data for layer, using magnitude fallback")
            return self._compute_weight_importance(root_module, root_idxs)
        
        # Get activation scores
        activation_scores = self.activation_scores[module_name]
        
        # Extract indices
        if hasattr(root_idxs[0], 'idx'):
            plain_idxs = [idx.idx for idx in root_idxs]
        else:
            plain_idxs = root_idxs
        
        channel_activations = activation_scores[plain_idxs]
        
        # Get weight scores
        weight_scores = self._compute_weight_importance(root_module, root_idxs)
        
        # Ensure both tensors are on the same device
        channel_activations = channel_activations.to(weight_scores.device)
        
        # WANDA: Multiply weight magnitude with activation magnitude
        wanda_scores = weight_scores * channel_activations
        
        # Normalize to [0, 1] range
        if wanda_scores.max() > wanda_scores.min():
            wanda_scores = (wanda_scores - wanda_scores.min()) / \
                          (wanda_scores.max() - wanda_scores.min() + self.epsilon)
        
        # IMPORTANT: In Torch-Pruning, higher importance = keep
        # Higher WANDA score (weight×activation) = more important = keep
        # Lower WANDA score = less important = prune first
        importance_scores = wanda_scores
        
        return importance_scores
