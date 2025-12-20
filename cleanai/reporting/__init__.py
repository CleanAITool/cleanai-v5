"""
Reporting Module

This module provides comprehensive PDF reporting capabilities for neural network pruning analysis.

Main Components:
    - PruningReportGenerator: Main class for generating reports
    - MetricsCollector: Collects metrics before/after pruning
    - ReportVisualizations: Creates charts and visualizations
    - PDFReportBuilder: Low-level PDF building

Quick Usage:
    ```python
    from cleanai.reporting import generate_pruning_report
    
    # Generate report
    report_path = generate_pruning_report(
        model_before=original_model,
        model_after=pruned_model,
        model_name="ResNet50",
        dataloader=test_loader,
        device=device,
        report_name="my_pruning_report",
        pruning_method="coverage",
        pruning_ratio=0.5
    )
    ```

Advanced Usage:
    ```python
    from cleanai.reporting import PruningReportGenerator
    
    # Create generator
    generator = PruningReportGenerator(
        model_before=original_model,
        model_after=pruned_model,
        model_name="ResNet50",
        dataset_name="ImageNet"
    )
    
    # Collect metrics
    generator.collect_metrics(
        dataloader=test_loader,
        device=device,
        coverage_analyzer=analyzer,
        pruning_method="coverage",
        pruning_ratio=0.5
    )
    
    # Generate report
    report_path = generator.generate_report(report_name="custom_report")
    ```
"""

from .report_generator import PruningReportGenerator, generate_pruning_report
from .metrics_collector import MetricsCollector
from .visualizations import ReportVisualizations
from .pdf_builder import PDFReportBuilder

__all__ = [
    'PruningReportGenerator',
    'generate_pruning_report',
    'MetricsCollector',
    'ReportVisualizations',
    'PDFReportBuilder',
]
