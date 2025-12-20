"""
Adaptive Neuron Coverage-based Importance Module

This module implements an adaptive version of coverage importance that
can update coverage scores during iterative pruning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from .coverage import NeuronCoverageImportance


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
