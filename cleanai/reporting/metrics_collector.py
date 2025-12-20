"""
Metrics Collector Module

This module collects comprehensive metrics before and after pruning for reporting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import time
import numpy as np
from collections import OrderedDict

from ..utils import count_parameters, count_flops, evaluate_model, measure_inference_time


class MetricsCollector:
    """
    Collects comprehensive metrics for pruning analysis and reporting.
    """
    
    def __init__(self):
        """
        Initialize metrics collector.
        """
        self.metrics: Dict[str, Any] = {
            'model_info': {},
            'before_pruning': {},
            'after_pruning': {},
            'coverage_analysis': {},
            'pruning_decisions': {},
            'layer_details': {},
            'risks': []
        }
    
    def collect_model_info(
        self,
        model_type: str = "CNN",
        dataset_name: str = "Unknown",
        batch_size: int = 32,
        framework: str = "PyTorch"
    ) -> None:
        """
        Collect basic model information.
        
        Args:
            model_type: Type of model (CNN, Transformer, RNN, etc.)
            dataset_name: Dataset name
            batch_size: Batch size used
            framework: Framework name
        """
        self.metrics['model_info'] = {
            'name': self.model_name,
            'type': model_type,
            'framework': framework,
            'dataset': dataset_name,
            'batch_size': batch_size,
            'input_shape': tuple(self.example_inputs.shape),
            'device': str(self.device),
            'total_layers': sum(1 for _ in self.model.modules() if isinstance(_, (nn.Conv2d, nn.Linear)))
        }
    
    def collect_before_pruning(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        model_name: str = "UnnamedModel",
        dataset_name: str = "Custom Dataset",
        example_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Collect metrics before pruning.
        
        Args:
            model: Model to analyze
            dataloader: Test data loader for accuracy evaluation
            device: Device for computation
            model_name: Name of the model
            dataset_name: Name of the dataset
            example_inputs: Example inputs for FLOPs calculation
            
        Returns:
            Dictionary with before-pruning metrics
        """
        print("\n" + "="*60)
        print("Collecting Pre-Pruning Metrics")
        print("="*60)
        
        metrics = {}
        metrics['model_name'] = model_name
        metrics['dataset_name'] = dataset_name
        metrics['model_type'] = model.__class__.__name__
        
        # Parameters
        params = count_parameters(model)
        metrics['parameters'] = params
        metrics['model_size_mb'] = (params * 4) / (1024 * 1024)  # Assuming float32
        
        print(f"Parameters: {params:,}")
        print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
        
        # FLOPs
        if example_inputs is not None:
            flops_info = count_flops(model, example_inputs.to(device))
            metrics['macs'] = flops_info['macs']
            metrics['gflops'] = flops_info['gflops']
            print(f"GFLOPs: {metrics['gflops']:.2f}")
        else:
            metrics['gflops'] = 0
        
        # Accuracy
        if dataloader is not None:
            print("\nEvaluating accuracy...")
            accuracy = evaluate_model(model, dataloader, device, verbose=False)
            metrics['accuracy'] = accuracy
            print(f"Accuracy: {accuracy:.2f}%")
        else:
            metrics['accuracy'] = 0
        
        # Inference time
        if example_inputs is not None:
            print("\nMeasuring inference time...")
            timing = measure_inference_time(
                model, 
                example_inputs.to(device), 
                device,
                num_runs=100,
                warmup_runs=10
            )
            metrics['inference_time_ms'] = timing['mean_ms']
            metrics['inference_std_ms'] = timing['std_ms']
            print(f"Inference Time: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
        else:
            metrics['inference_time_ms'] = 0
        
        # Layer count
        metrics['total_layers'] = sum(1 for _ in model.modules() if isinstance(_, (nn.Conv2d, nn.Linear)))
        metrics['architecture_summary'] = f"{model.__class__.__name__} with {metrics['total_layers']} layers"
        
        print("="*60 + "\n")
        
        return metrics
    
    def collect_after_pruning(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        before_metrics: Dict[str, Any],
        example_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Collect metrics after pruning and compute comparisons.
        
        Args:
            model_before: Original model
            model_after: Pruned model
            dataloader: Test data loader
            device: Device for computation
            before_metrics: Metrics from before pruning
            example_inputs: Example inputs for FLOPs
            
        Returns:
            Dictionary with after-pruning metrics and comparisons
        """
        print("\n" + "="*60)
        print("Collecting Post-Pruning Metrics")
        print("="*60)
        
        metrics = {}
        
        # Parameters
        params = count_parameters(model_after)
        metrics['parameters'] = params
        metrics['model_size_mb'] = (params * 4) / (1024 * 1024)
        
        print(f"Parameters: {params:,}")
        print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
        
        # FLOPs
        if example_inputs is not None:
            flops_info = count_flops(model_after, example_inputs.to(device))
            metrics['macs'] = flops_info['macs']
            metrics['gflops'] = flops_info['gflops']
            print(f"GFLOPs: {metrics['gflops']:.2f}")
        else:
            metrics['gflops'] = 0
        
        # Accuracy
        if dataloader is not None:
            print("\nEvaluating accuracy...")
            accuracy = evaluate_model(model_after, dataloader, device, verbose=False)
            metrics['accuracy'] = accuracy
            print(f"Accuracy: {accuracy:.2f}%")
        else:
            metrics['accuracy'] = 0
        
        # Inference time
        if example_inputs is not None:
            print("\nMeasuring inference time...")
            timing = measure_inference_time(
                model_after,
                example_inputs.to(device),
                device,
                num_runs=100,
                warmup_runs=10
            )
            metrics['inference_time_ms'] = timing['mean_ms']
            metrics['inference_std_ms'] = timing['std_ms']
            print(f"Inference Time: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
        else:
            metrics['inference_time_ms'] = 0
        
        # Compute reductions
        if before_metrics['parameters'] > 0:
            metrics['param_reduction_pct'] = ((before_metrics['parameters'] - params) / before_metrics['parameters']) * 100
        else:
            metrics['param_reduction_pct'] = 0
            
        if before_metrics['model_size_mb'] > 0:
            metrics['size_reduction_pct'] = ((before_metrics['model_size_mb'] - metrics['model_size_mb']) / before_metrics['model_size_mb']) * 100
        else:
            metrics['size_reduction_pct'] = 0
            
        if before_metrics.get('gflops', 0) > 0:
            metrics['flops_reduction_pct'] = ((before_metrics['gflops'] - metrics['gflops']) / before_metrics['gflops']) * 100
        else:
            metrics['flops_reduction_pct'] = 0
            
        metrics['accuracy_drop'] = before_metrics.get('accuracy', 0) - metrics.get('accuracy', 0)
        
        if metrics['inference_time_ms'] > 0:
            metrics['speedup'] = before_metrics.get('inference_time_ms', 0) / metrics['inference_time_ms']
        else:
            metrics['speedup'] = 1.0
        
        print(f"\nReductions:")
        print(f"  Parameters: -{metrics['param_reduction_pct']:.2f}%")
        print(f"  Model Size: -{metrics['size_reduction_pct']:.2f}%")
        print(f"  FLOPs: -{metrics['flops_reduction_pct']:.2f}%")
        print(f"  Accuracy Drop: {metrics['accuracy_drop']:.2f}%")
        print(f"  Speedup: {metrics['speedup']:.2f}x")
        
        print("="*60 + "\n")
        
        return metrics
        metrics['inference_std_ms'] = timing['std_ms']
        print(f"Inference Time: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
        
        # Layer structure
        metrics['layer_structure'] = self._analyze_layer_structure(pruned_model)
        
        # Compute reductions
        if 'before_pruning' in self.metrics:
            before = self.metrics['before_pruning']
            metrics['param_reduction_pct'] = ((before['parameters'] - params) / before['parameters']) * 100
            metrics['size_reduction_pct'] = ((before['model_size_mb'] - metrics['model_size_mb']) / before['model_size_mb']) * 100
            
            if 'gflops' in before and 'gflops' in metrics:
                metrics['flops_reduction_pct'] = ((before['gflops'] - metrics['gflops']) / before['gflops']) * 100
            
            if 'accuracy' in before and 'accuracy' in metrics:
                metrics['accuracy_drop'] = before['accuracy'] - metrics['accuracy']
            
            if 'inference_time_ms' in before:
                metrics['speedup'] = before['inference_time_ms'] / metrics['inference_time_ms']
        
        self.metrics['after_pruning'] = metrics
        print("="*60 + "\n")
        
        return metrics
    
    def collect_coverage_analysis(
        self,
        coverage_scores: Dict[str, torch.Tensor],
        coverage_stats: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Collect coverage analysis metrics.
        
        Args:
            coverage_scores: Layer-wise coverage scores
            coverage_stats: Coverage statistics
        """
        self.metrics['coverage_analysis'] = {
            'layer_coverage': {},
            'overall_stats': {},
            'low_coverage_layers': []
        }
        
        all_coverages = []
        
        for layer_name, scores in coverage_scores.items():
            stats = coverage_stats.get(layer_name, {})
            
            self.metrics['coverage_analysis']['layer_coverage'][layer_name] = {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'zero_count': int((scores == 0).sum()),
                'num_channels': len(scores)
            }
            
            all_coverages.extend(scores.cpu().numpy().tolist())
            
            # Identify low coverage layers
            if stats.get('mean', 0) < 0.1:  # Threshold
                self.metrics['coverage_analysis']['low_coverage_layers'].append({
                    'layer': layer_name,
                    'mean_coverage': stats.get('mean', 0),
                    'zero_neurons': stats.get('zero_coverage_count', 0)
                })
        
        # Overall statistics
        if all_coverages:
            all_coverages = np.array(all_coverages)
            self.metrics['coverage_analysis']['overall_stats'] = {
                'mean': float(all_coverages.mean()),
                'std': float(all_coverages.std()),
                'median': float(np.median(all_coverages)),
                'min': float(all_coverages.min()),
                'max': float(all_coverages.max())
            }
    
    def collect_pruning_decisions(
        self,
        pruning_ratio: float,
        importance_method: str,
        global_pruning: bool,
        iterative_steps: int,
        layer_wise_ratios: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Collect pruning decision parameters.
        
        Args:
            pruning_ratio: Overall pruning ratio
            importance_method: Importance method used
            global_pruning: Whether global pruning was used
            iterative_steps: Number of iterative steps
            layer_wise_ratios: Layer-specific pruning ratios
        """
        self.metrics['pruning_decisions'] = {
            'strategy': 'Structured Channel Pruning',
            'method': importance_method,
            'pruning_ratio': pruning_ratio,
            'global_pruning': global_pruning,
            'iterative_steps': iterative_steps,
            'layer_wise_ratios': layer_wise_ratios or {}
        }
        return self.metrics['pruning_decisions']
    
    def analyze_risks(self) -> List[Dict[str, str]]:
        """
        Analyze potential risks from pruning.
        
        Returns:
            List of risk warnings
        """
        risks = []
        
        if 'after_pruning' in self.metrics and 'before_pruning' in self.metrics:
            after = self.metrics['after_pruning']
            before = self.metrics['before_pruning']
            
            # High accuracy drop
            if 'accuracy_drop' in after and after['accuracy_drop'] > 5.0:
                risks.append({
                    'level': 'HIGH',
                    'category': 'Accuracy Degradation',
                    'message': f"Significant accuracy drop of {after['accuracy_drop']:.2f}% detected. Model may have lost critical features."
                })
            elif 'accuracy_drop' in after and after['accuracy_drop'] > 2.0:
                risks.append({
                    'level': 'MEDIUM',
                    'category': 'Accuracy Degradation',
                    'message': f"Moderate accuracy drop of {after['accuracy_drop']:.2f}% observed."
                })
            
            # Over-pruning
            if 'param_reduction_pct' in after and after['param_reduction_pct'] > 70:
                risks.append({
                    'level': 'HIGH',
                    'category': 'Over-Pruning',
                    'message': f"Aggressive pruning of {after['param_reduction_pct']:.1f}% may compromise model capacity."
                })
        
        # Low coverage layers
        if 'coverage_analysis' in self.metrics:
            low_cov = self.metrics['coverage_analysis'].get('low_coverage_layers', [])
            if len(low_cov) > 3:
                risks.append({
                    'level': 'MEDIUM',
                    'category': 'Coverage Imbalance',
                    'message': f"{len(low_cov)} layers have very low coverage, indicating potential dead neurons."
                })
        
        self.metrics['risks'] = risks
        return risks
    
    def _analyze_layer_structure(self, model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Analyze layer-wise structure.
        
        Args:
            model: Model to analyze (default: self.model)
            
        Returns:
            Layer structure information
        """
        if model is None:
            model = self.model
        
        structure = {
            'conv_layers': [],
            'linear_layers': [],
            'total_conv_channels': 0,
            'total_linear_neurons': 0
        }
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                structure['conv_layers'].append({
                    'name': name,
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size
                })
                structure['total_conv_channels'] += module.out_channels
                
            elif isinstance(module, nn.Linear):
                structure['linear_layers'].append({
                    'name': name,
                    'in_features': module.in_features,
                    'out_features': module.out_features
                })
                structure['total_linear_neurons'] += module.out_features
        
        return structure
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self.metrics
