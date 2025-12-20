"""
CleanAI - Coverage-Based Neural Network Pruning Framework

A modular framework for pruning neural networks using neuron coverage analysis
and various importance metrics including WANDA.
"""

__version__ = "0.1.0"

# Import main components
from .importance import (
    NeuronCoverageImportance,
    WandaImportance,
    AdaptiveNeuronCoverageImportance
)

from .analyzers import CoverageAnalyzer
from .pruners import CoveragePruner

from .utils import (
    count_parameters,
    evaluate_model,
    compare_models,
    print_model_summary
)

from .reporting import (
    PruningReportGenerator,
    generate_pruning_report
)

__all__ = [
    # Importance metrics
    'NeuronCoverageImportance',
    'WandaImportance',
    'AdaptiveNeuronCoverageImportance',
    
    # Analyzers
    'CoverageAnalyzer',
    
    # Pruners
    'CoveragePruner',
    
    # Utilities
    'count_parameters',
    'evaluate_model',
    'compare_models',
    'print_model_summary',
    
    # Reporting
    'PruningReportGenerator',
    'generate_pruning_report',
]
