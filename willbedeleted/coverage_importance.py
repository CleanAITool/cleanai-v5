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
from coverage_analyzer import CoverageAnalyzer


class NeuronCoverageImportance(tp.importance.Importance):
    """
    Importance estimator based on neuron coverage from test data.
    
    This class computes importance scores for pruning based on how active
    each neuron/channel is on test data. Channels with lower coverage
    (less active) are considered less important and will be pruned first.
    
    The importance score is computed as the inverse of coverage:
        importance = 1 / (coverage + epsilon)
    
    This means lower coverage -> higher importance for pruning.
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
            Higher importance = higher priority for pruning (inverse of coverage)
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
        
        # Convert coverage to importance (inverse relationship)
        # Lower coverage = higher importance for pruning
        importance_scores = 1.0 / (channel_coverage + self.epsilon)
        
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
        
        # For magnitude, we want to prune small weights, so invert
        # (small magnitude = high importance for pruning)
        max_imp = importance.max()
        if max_imp > 0:
            importance = max_imp - importance
        
        return importance


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
        
        # WANDA: Multiply weight magnitude with activation magnitude
        wanda_scores = weight_scores * channel_activations
        
        # Normalize to [0, 1] range
        if wanda_scores.max() > wanda_scores.min():
            wanda_scores = (wanda_scores - wanda_scores.min()) / \
                          (wanda_scores.max() - wanda_scores.min() + self.epsilon)
        
        # Invert: lower WANDA score = higher importance for pruning
        # We want to prune channels with low weight×activation product
        importance_scores = 1.0 - wanda_scores
        
        return importance_scores


class AdaptiveNeuronCoverageImportance(NeuronCoverageImportance):
    """
    Adaptive version that can update coverage scores during iterative pruning.
    
    This allows the importance criterion to adapt as the model changes
    during iterative pruning steps.
    """
    
    def __init__(
        self,
        test_loader: DataLoader,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        coverage_metric: str = 'normalized_mean',
        max_batches: Optional[int] = None,
        epsilon: float = 1e-8,
        recompute_every_n_steps: int = 1
    ):
        """
        Initialize adaptive coverage importance estimator.
        
        Args:
            test_loader: DataLoader containing test samples
            device: Device to run inference on
            coverage_metric: Coverage metric to use
            max_batches: Maximum number of batches to process
            epsilon: Small constant to prevent division by zero
            recompute_every_n_steps: Recompute coverage every N pruning steps
        """
        super().__init__(test_loader, device, coverage_metric, max_batches, epsilon)
        
        self.recompute_every_n_steps = recompute_every_n_steps
        self.current_step = 0
    
    def step(self) -> None:
        """
        Increment step counter and potentially trigger coverage recomputation.
        """
        self.current_step += 1
        
        # Reset coverage computation flag to force recomputation
        if self.current_step % self.recompute_every_n_steps == 0:
            print(f"\nRecomputing coverage at step {self.current_step}...")
            self._coverage_computed = False
            if self.analyzer is not None:
                self.analyzer.remove_hooks()
                self.analyzer.clear_activations()
    
    def reset(self) -> None:
        """Reset step counter and coverage computation state."""
        self.current_step = 0
        self._coverage_computed = False
        if self.analyzer is not None:
            self.analyzer.remove_hooks()
            self.analyzer.clear_activations()
