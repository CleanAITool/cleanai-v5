"""
Utilities Module

This module provides utility functions for model analysis, evaluation, and benchmarking.
"""

from .model_utils import (
    count_parameters,
    count_trainable_parameters,
    get_parameter_stats,
    count_flops,
    print_model_summary,
    save_model,
    load_model,
    compare_models_params
)

from .evaluation import (
    evaluate_model,
    evaluate_with_loss,
    measure_inference_time,
    compare_models
)

from .quantization import (
    ModelQuantizer,
    quantize_model
)

__all__ = [
    # Model utilities
    'count_parameters',
    'count_trainable_parameters',
    'get_parameter_stats',
    'count_flops',
    'print_model_summary',
    'save_model',
    'load_model',
    'compare_models_params',
    
    # Evaluation utilities
    'evaluate_model',
    'evaluate_with_loss',
    'measure_inference_time',
    'compare_models',
    
    # Quantization utilities
    'ModelQuantizer',
    'quantize_model',
]
