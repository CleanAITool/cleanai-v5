"""
Coverage-Based Pruner Module

This module implements a high-level pruner that uses neuron coverage
metrics to guide the structured pruning process.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any, Callable
import torch_pruning as tp
from ..importance.coverage import NeuronCoverageImportance
from ..importance.adaptive import AdaptiveNeuronCoverageImportance
from ..importance.wanda import WandaImportance


class CoveragePruner:
    """
    High-level pruner using neuron coverage for importance estimation.
    
    This class wraps Torch-Pruning's BasePruner with coverage-based importance
    criterion, providing an easy-to-use interface for coverage-guided pruning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        test_loader: DataLoader,
        pruning_ratio: float = 0.5,
        importance_method: str = 'coverage',  # 'coverage', 'wanda', 'magnitude'
        coverage_metric: str = 'normalized_mean',
        global_pruning: bool = False,
        iterative_steps: int = 1,
        max_batches: Optional[int] = None,
        ignored_layers: Optional[List[nn.Module]] = None,
        round_to: Optional[int] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose: bool = True,
        adaptive: bool = False,
        pruning_ratio_dict: Optional[Dict[nn.Module, float]] = None,
        customized_pruners: Optional[Dict[Any, tp.pruner.function.BasePruningFunc]] = None,
        unwrapped_parameters: Optional[Dict[nn.Parameter, int]] = None,
    ):
        """
        Initialize coverage-based pruner.
        
        Args:
            model: PyTorch model to prune
            example_inputs: Dummy input tensor for graph tracing (e.g., torch.randn(1, 3, 224, 224))
            test_loader: DataLoader containing test samples for coverage analysis
            pruning_ratio: Global pruning ratio (0.0 - 1.0). E.g., 0.3 means remove 30% of channels
            importance_method: Importance estimation method:
                - 'coverage': Neuron coverage-based (activation patterns)
                - 'wanda': WANDA (Weight AND Activation)
                - 'magnitude': Standard magnitude-based (fallback)
            coverage_metric: Coverage metric to use (for coverage/wanda methods):
                - 'normalized_mean': Average activation normalized by global max
                - 'frequency': Proportion of samples where neuron is active
                - 'mean_absolute': Absolute mean activation value
                - 'combined': Weighted combination of mean and frequency
            global_pruning: If True, prune globally across all layers; if False, uniform per-layer
            iterative_steps: Number of iterative pruning steps to reach target ratio
            max_batches: Maximum number of test batches to process (None for all)
            ignored_layers: List of layers to exclude from pruning
            round_to: Round number of channels to nearest multiple (e.g., 8)
            device: Device to run computations on
            verbose: Print detailed progress information
            adaptive: Use adaptive importance that recomputes coverage during iterative pruning
            pruning_ratio_dict: Layer-specific pruning ratios (overrides global ratio)
            customized_pruners: Custom pruning functions for specific layer types
            unwrapped_parameters: Unwrapped parameters that need special handling
        """
        self.model = model
        self.example_inputs = example_inputs
        self.test_loader = test_loader
        self.pruning_ratio = pruning_ratio
        self.importance_method = importance_method
        self.coverage_metric = coverage_metric
        self.global_pruning = global_pruning
        self.iterative_steps = iterative_steps
        self.max_batches = max_batches
        self.device = device
        self.verbose = verbose
        self.adaptive = adaptive
        
        # Move model to device
        self.model.to(self.device)
        self.example_inputs = self.example_inputs.to(self.device)
        
        # Create importance criterion based on method
        if importance_method == 'wanda':
            # WANDA: Weight × Activation
            self.importance = WandaImportance(
                test_loader=test_loader,
                device=device,
                max_batches=max_batches,
                weight_norm='l2',
                activation_metric='mean_abs'
            )
            if self.verbose:
                print("Using WANDA importance (Weight × Activation)")
                
        elif importance_method == 'coverage':
            # Coverage-based (activation patterns only)
            if adaptive and iterative_steps > 1:
                self.importance = AdaptiveNeuronCoverageImportance(
                    test_loader=test_loader,
                    device=device,
                    coverage_metric=coverage_metric,
                    max_batches=max_batches,
                    recompute_every_n_steps=1
                )
                if self.verbose:
                    print("Using adaptive coverage importance (recomputes every step)")
            else:
                self.importance = NeuronCoverageImportance(
                    test_loader=test_loader,
                    device=device,
                    coverage_metric=coverage_metric,
                    max_batches=max_batches
                )
                if self.verbose:
                    print("Using static coverage importance (computes once)")
                    
        elif importance_method == 'magnitude':
            # Standard magnitude-based pruning
            self.importance = tp.importance.MagnitudeImportance(p=2)
            if self.verbose:
                print("Using magnitude-based importance (L2 norm)")
        else:
            raise ValueError(f"Unknown importance method: {importance_method}")
        
        # Prepare ignored layers list
        if ignored_layers is None:
            ignored_layers = []
        
        # Create the Torch-Pruning MetaPruner (now called BasePruner in newer versions)
        # We use the compatibility layer for older API
        try:
            # Try new API first (v1.3.0+)
            from torch_pruning.pruner.algorithms import MetaPruner
            pruner_class = MetaPruner
        except ImportError:
            # Fallback to direct BasePruner
            from torch_pruning.pruner.algorithms.base_pruner import BasePruner
            pruner_class = BasePruner
        
        if self.verbose:
            print(f"\nInitializing Torch-Pruning with:")
            print(f"  Importance method: {importance_method}")
            print(f"  Pruning ratio: {pruning_ratio:.2%}")
            if importance_method in ['coverage', 'wanda']:
                print(f"  Coverage metric: {coverage_metric}")
            print(f"  Global pruning: {global_pruning}")
            print(f"  Iterative steps: {iterative_steps}")
            print(f"  Device: {device}")
        
        self.pruner = pruner_class(
            model=model,
            example_inputs=example_inputs,
            importance=self.importance,
            global_pruning=global_pruning,
            pruning_ratio=pruning_ratio,
            pruning_ratio_dict=pruning_ratio_dict,
            iterative_steps=iterative_steps,
            ignored_layers=ignored_layers,
            round_to=round_to,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
        )
        
        # Track pruning progress
        self.current_step = 0
        self.pruning_history: List[Dict[str, Any]] = []
    
    def step(self) -> Dict[str, Any]:
        """
        Perform one pruning step.
        
        Returns:
            Dictionary containing metrics about this pruning step
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Pruning Step {self.current_step + 1}/{self.iterative_steps}")
            print(f"{'='*60}")
        
        # Record model stats before pruning
        before_params = sum(p.numel() for p in self.model.parameters())
        
        # Perform pruning step
        self.pruner.step()
        
        # Record model stats after pruning
        after_params = sum(p.numel() for p in self.model.parameters())
        params_removed = before_params - after_params
        params_removed_ratio = params_removed / before_params if before_params > 0 else 0
        
        # Record step info
        step_info = {
            'step': self.current_step,
            'before_params': before_params,
            'after_params': after_params,
            'params_removed': params_removed,
            'params_removed_ratio': params_removed_ratio,
        }
        
        self.pruning_history.append(step_info)
        
        if self.verbose:
            print(f"\nPruning Results:")
            print(f"  Parameters before: {before_params:,}")
            print(f"  Parameters after:  {after_params:,}")
            print(f"  Parameters removed: {params_removed:,} ({params_removed_ratio:.2%})")
        
        self.current_step += 1
        
        # Update adaptive importance if applicable
        if self.adaptive and hasattr(self.importance, 'step'):
            self.importance.step()
        
        return step_info
    
    def prune(self) -> Dict[str, Any]:
        """
        Perform all pruning steps at once.
        
        Returns:
            Dictionary containing overall pruning metrics
        """
        initial_params = sum(p.numel() for p in self.model.parameters())
        
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"Starting Coverage-Based Pruning")
            print(f"{'#'*60}")
            print(f"Initial parameters: {initial_params:,}")
        
        # Perform all iterative steps
        for _ in range(self.iterative_steps):
            self.step()
        
        final_params = sum(p.numel() for p in self.model.parameters())
        total_removed = initial_params - final_params
        total_removed_ratio = total_removed / initial_params if initial_params > 0 else 0
        
        overall_info = {
            'initial_params': initial_params,
            'final_params': final_params,
            'total_removed': total_removed,
            'total_removed_ratio': total_removed_ratio,
            'iterative_steps': self.iterative_steps,
            'step_history': self.pruning_history
        }
        
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"Pruning Complete")
            print(f"{'#'*60}")
            print(f"Initial parameters:  {initial_params:,}")
            print(f"Final parameters:    {final_params:,}")
            print(f"Total removed:       {total_removed:,} ({total_removed_ratio:.2%})")
            print(f"Target pruning ratio: {self.pruning_ratio:.2%}")
        
        return overall_info
    
    def get_model(self) -> nn.Module:
        """
        Get the pruned model.
        
        Returns:
            Pruned PyTorch model
        """
        return self.model
    
    def get_dependency_graph(self) -> tp.dependency.DependencyGraph:
        """
        Get the dependency graph used by the pruner.
        
        Returns:
            DependencyGraph instance
        """
        return self.pruner.DG
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: Optional[Callable] = None,
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None
    ) -> Dict[str, float]:
        """
        Validate the pruned model.
        
        Args:
            val_loader: Validation DataLoader
            criterion: Loss function (optional)
            metric_fn: Metric function that takes (predictions, targets) and returns score
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        if self.verbose:
            print("\nValidating pruned model...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    if len(batch) < 2:
                        continue  # Skip invalid batch
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs = batch
                    targets = None
                
                # Safety check for None inputs
                if inputs is None:
                    continue
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss if criterion provided
                if criterion is not None and targets is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                
                # Compute accuracy for classification
                if targets is not None:
                    if outputs.dim() > 1 and outputs.size(1) > 1:
                        _, predicted = outputs.max(1)
                        total_correct += predicted.eq(targets).sum().item()
                    
                    all_predictions.append(outputs.cpu())
                    all_targets.append(targets.cpu())
                
                total_samples += inputs.size(0)
                
                if self.verbose and (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches")
        
        results = {'total_samples': total_samples}
        
        if criterion is not None and total_samples > 0:
            results['avg_loss'] = total_loss / total_samples
        
        if total_samples > 0 and len(all_targets) > 0:
            results['accuracy'] = total_correct / total_samples
        
        # Compute custom metric if provided
        if metric_fn is not None and len(all_predictions) > 0:
            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)
            results['custom_metric'] = metric_fn(all_predictions, all_targets)
        
        if self.verbose:
            print("\nValidation Results:")
            for key, value in results.items():
                if key != 'total_samples':
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value}")
        
        return results
    
    def generate_report(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        dataloader: DataLoader,
        report_name: Optional[str] = None,
        dataset_name: str = "Custom Dataset",
        output_dir: str = "reports"
    ) -> str:
        """
        Generate a comprehensive PDF report of the pruning process.
        
        Args:
            model_before: Original model before pruning
            model_after: Pruned model after pruning
            dataloader: DataLoader for evaluation
            report_name: Custom report name (if None, auto-generated)
            dataset_name: Name of the dataset used
            output_dir: Output directory for the report
            
        Returns:
            Path to the generated PDF report
            
        Example:
            ```python
            # After pruning
            pruned_model = pruner.prune(pruning_ratio=0.5)
            
            # Generate report
            report_path = pruner.generate_report(
                model_before=original_model,
                model_after=pruned_model,
                dataloader=test_loader,
                report_name="my_pruning_analysis"
            )
            ```
        """
        from ..reporting import generate_pruning_report
        
        return generate_pruning_report(
            model_before=model_before,
            model_after=model_after,
            model_name=self.model.__class__.__name__,
            dataloader=dataloader,
            device=self.device,
            report_name=report_name,
            coverage_analyzer=self.coverage_analyzer,
            pruning_method=self.importance_method,
            pruning_ratio=self.pruning_ratio,
            iterative_steps=self.iterative_steps,
            dataset_name=dataset_name,
            output_dir=output_dir
        )
