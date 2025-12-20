"""
Analyzers Module

This module provides tools for analyzing neural network activations and coverage.
"""

from .coverage_analyzer import CoverageAnalyzer, ActivationHook

__all__ = [
    'CoverageAnalyzer',
    'ActivationHook',
]
