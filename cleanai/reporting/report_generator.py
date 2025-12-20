"""
Main Report Generator Module

This module coordinates all reporting components to generate comprehensive
pruning analysis reports.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from datetime import datetime
import os
from reportlab.lib.units import inch

from .metrics_collector import MetricsCollector
from .visualizations import ReportVisualizations
from .pdf_builder import PDFReportBuilder


class PruningReportGenerator:
    """
    Generates comprehensive PDF reports for pruning analysis.
    """
    
    def __init__(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        model_name: str,
        dataset_name: str = "Custom Dataset",
        output_dir: str = "reports"
    ):
        """
        Initialize report generator.
        
        Args:
            model_before: Model before pruning
            model_after: Model after pruning
            model_name: Name of the model
            dataset_name: Name of the dataset
            output_dir: Output directory for reports
        """
        self.model_before = model_before
        self.model_after = model_after
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.visualizer = ReportVisualizations()
        
        # Storage for collected metrics
        self.before_metrics = None
        self.after_metrics = None
        self.coverage_metrics = None
        self.pruning_info = None
        self.risks = []
    
    def collect_metrics(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        coverage_analyzer = None,
        pruning_method: str = "coverage",
        pruning_ratio: float = 0.5,
        iterative_steps: int = 1
    ):
        """
        Collect all metrics for the report.
        
        Args:
            dataloader: DataLoader for evaluation
            device: Device for computation
            coverage_analyzer: Coverage analyzer instance
            pruning_method: Pruning method used
            pruning_ratio: Pruning ratio applied
            iterative_steps: Number of iterative steps
        """
        print("📊 Collecting metrics...")
        
        # Create example inputs
        example_inputs = next(iter(dataloader))[0][:1] if dataloader else None
        
        # Collect before-pruning metrics
        print("  - Before pruning metrics...")
        self.before_metrics = self.metrics_collector.collect_before_pruning(
            model=self.model_before,
            dataloader=dataloader,
            device=device,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            example_inputs=example_inputs
        )
        
        # Collect after-pruning metrics
        print("  - After pruning metrics...")
        self.after_metrics = self.metrics_collector.collect_after_pruning(
            model_before=self.model_before,
            model_after=self.model_after,
            dataloader=dataloader,
            device=device,
            before_metrics=self.before_metrics,
            example_inputs=example_inputs
        )
        
        # Collect coverage analysis if available
        if coverage_analyzer is not None:
            print("  - Coverage analysis...")
            self.coverage_metrics = self.metrics_collector.collect_coverage_analysis(
                coverage_analyzer=coverage_analyzer,
                model=self.model_before
            )
        
        # Collect pruning info
        print("  - Pruning decisions...")
        self.pruning_info = self.metrics_collector.collect_pruning_decisions(
            pruning_ratio=pruning_ratio,
            importance_method=pruning_method,
            global_pruning=False,
            iterative_steps=iterative_steps
        )
        
        # Analyze risks
        print("  - Risk analysis...")
        self.risks = self.metrics_collector.analyze_risks()
        
        print("✅ Metrics collection complete!\n")
    
    def generate_report(self, report_name: Optional[str] = None) -> str:
        """
        Generate comprehensive PDF report.
        
        Args:
            report_name: Custom report name (optional)
            
        Returns:
            Path to generated PDF file
        """
        if self.before_metrics is None or self.after_metrics is None:
            raise ValueError("Please collect metrics first using collect_metrics()")
        
        # Generate filename
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"pruning_report_{self.model_name}_{timestamp}"
        
        # Ensure .pdf extension
        if not report_name.endswith('.pdf'):
            report_name += '.pdf'
        
        output_path = os.path.join(self.output_dir, report_name)
        
        print(f"📝 Generating report: {report_name}\n")
        
        # Initialize PDF builder
        pdf = PDFReportBuilder(
            filename=output_path,
            title="Neural Network Pruning Report",
            author="CleanAI Framework"
        )
        
        # 1. Cover page
        print("  ✓ Creating cover page...")
        pdf.add_cover_page(
            model_name=self.model_name,
            model_type=self.before_metrics.get('model_type', 'Unknown'),
            dataset=self.dataset_name,
            pruning_ratio=self.after_metrics.get('param_reduction_pct', 0)
        )
        
        # 2. Executive summary
        print("  ✓ Adding executive summary...")
        pdf.add_executive_summary(
            before=self.before_metrics,
            after=self.after_metrics
        )
        
        # 3. Model & Experiment Info
        print("  ✓ Adding model information...")
        self._add_model_info_section(pdf)
        
        # 4. Coverage Analysis (if available)
        if self.coverage_metrics is not None:
            print("  ✓ Adding coverage analysis...")
            self._add_coverage_section(pdf)
        
        # 5. Pruning Decision Mechanism
        print("  ✓ Adding pruning decisions...")
        self._add_pruning_decisions_section(pdf)
        
        # 6. Post-Pruning Structure
        print("  ✓ Adding post-pruning structure...")
        self._add_post_pruning_structure(pdf)
        
        # 7. Performance Comparison
        print("  ✓ Adding performance comparison...")
        self._add_performance_comparison(pdf)
        
        # 8. Risk Analysis
        print("  ✓ Adding risk analysis...")
        self._add_risk_analysis(pdf)
        
        # Build PDF
        pdf.build()
        
        return output_path
    
    def _add_model_info_section(self, pdf: PDFReportBuilder):
        """Add model information section."""
        pdf.add_section("2. Model & Experiment Information")
        
        # Model details
        pdf.add_subsection("Model Architecture")
        model_info = [
            ['Property', 'Value'],
            ['Model Name', self.model_name],
            ['Model Type', self.before_metrics.get('model_type', 'Unknown')],
            ['Total Layers', str(self.before_metrics.get('total_layers', 0))],
            ['Dataset', self.dataset_name],
        ]
        pdf.add_table(model_info, col_widths=[2*inch, 4*inch])
        
        # Architecture summary
        arch_text = self.before_metrics.get('architecture_summary', '')
        if arch_text:
            pdf.add_text("<b>Architecture Summary:</b>")
            pdf.add_text(arch_text)
    
    def _add_coverage_section(self, pdf: PDFReportBuilder):
        """Add coverage analysis section."""
        pdf.add_section("3. Pre-Pruning Coverage Analysis")
        
        pdf.add_text(
            "Coverage analysis identifies which neurons are actively contributing to model predictions. "
            "Neurons with low coverage are candidates for pruning."
        )
        
        # Overall coverage stats
        pdf.add_subsection("Overall Coverage Statistics")
        coverage_data = [
            ['Metric', 'Value'],
            ['Average Coverage', f"{self.coverage_metrics['overall']['average_coverage']:.2%}"],
            ['Median Coverage', f"{self.coverage_metrics['overall']['median_coverage']:.2%}"],
            ['Min Coverage', f"{self.coverage_metrics['overall']['min_coverage']:.2%}"],
            ['Max Coverage', f"{self.coverage_metrics['overall']['max_coverage']:.2%}"],
        ]
        pdf.add_table(coverage_data, col_widths=[3*inch, 3*inch])
        
        # Coverage visualization
        if self.coverage_metrics['layer_stats']:
            coverage_chart = self.visualizer.create_layer_coverage_bar_chart(
                self.coverage_metrics['layer_stats']
            )
            pdf.add_image(coverage_chart, width=6*inch)
            
            # Coverage heatmap
            heatmap = self.visualizer.create_coverage_heatmap(
                self.coverage_metrics['layer_stats']
            )
            pdf.add_image(heatmap, width=6*inch)
        
        # Low coverage layers
        low_cov = self.coverage_metrics.get('low_coverage_layers', [])
        if low_cov:
            pdf.add_subsection("Layers with Low Coverage (< 50%)")
            low_cov_data = [['Layer Name', 'Coverage %']]
            for layer_info in low_cov[:10]:  # Top 10
                low_cov_data.append([
                    layer_info['name'],
                    f"{layer_info['coverage']:.1f}%"
                ])
            pdf.add_table(low_cov_data, col_widths=[4*inch, 2*inch])
    
    def _add_pruning_decisions_section(self, pdf: PDFReportBuilder):
        """Add pruning decision mechanism section."""
        pdf.add_section("4. Pruning Decision Mechanism")
        
        pdf.add_text(
            "This section explains the pruning strategy and importance criteria used to determine "
            "which parameters to prune."
        )
        
        # Pruning configuration
        pdf.add_subsection("Pruning Configuration")
        config_data = [
            ['Parameter', 'Value'],
            ['Pruning Method', self.pruning_info['method']],
            ['Target Pruning Ratio', f"{self.pruning_info['pruning_ratio']:.1%}"],
            ['Iterative Steps', str(self.pruning_info['iterative_steps'])],
            ['Actual Parameters Pruned', f"{self.after_metrics['param_reduction_pct']:.2f}%"],
        ]
        pdf.add_table(config_data, col_widths=[3*inch, 3*inch])
        
        # Method explanation
        pdf.add_subsection("Importance Criterion Explanation")
        method = self.pruning_info['method']
        
        if method == 'coverage':
            explanation = """
            <b>Coverage-Based Pruning:</b> This method uses neuron activation patterns to determine importance.
            Neurons that are rarely activated across the dataset are considered less important and are 
            prioritized for pruning. This ensures that frequently-used pathways in the network are preserved.
            """
        elif method == 'wanda':
            explanation = """
            <b>WANDA (Weight AND Activation):</b> This method combines weight magnitude and activation 
            statistics to determine importance. The importance score is calculated as: 
            Importance = |Weight| × |Activation|. This captures both the structural importance (weights) 
            and functional importance (activations) of each parameter.
            """
        elif method == 'magnitude':
            explanation = """
            <b>Magnitude-Based Pruning:</b> This classic method prunes parameters based solely on their 
            absolute weight values. Parameters with smaller magnitudes are considered less important and 
            are removed first. This is a simple but effective baseline method.
            """
        else:
            explanation = f"<b>Pruning Method:</b> {method}"
        
        pdf.add_text(explanation)
        
        # Layer-wise pruning distribution
        if 'layer_pruning_distribution' in self.pruning_info:
            pdf.add_subsection("Layer-wise Pruning Distribution")
            dist_data = [['Layer Name', 'Parameters Pruned (%)']]
            for layer_name, pct in self.pruning_info['layer_pruning_distribution'].items():
                dist_data.append([layer_name, f"{pct:.2f}%"])
            pdf.add_table(dist_data, col_widths=[4*inch, 2*inch])
    
    def _add_post_pruning_structure(self, pdf: PDFReportBuilder):
        """Add post-pruning structure section."""
        pdf.add_section("5. Post-Pruning Model Structure")
        
        pdf.add_text(
            "Analysis of the pruned model architecture and remaining parameters."
        )
        
        # Comparison table
        pdf.add_subsection("Architecture Comparison")
        comparison = self.visualizer.create_comparison_bar_chart(
            self.before_metrics,
            self.after_metrics
        )
        pdf.add_image(comparison, width=6*inch)
        
        # Pie chart for reduction
        pie_chart = self.visualizer.create_reduction_pie_chart(
            self.after_metrics['param_reduction_pct']
        )
        pdf.add_image(pie_chart, width=5*inch)
    
    def _add_performance_comparison(self, pdf: PDFReportBuilder):
        """Add performance comparison section."""
        pdf.add_section("6. Performance Comparison")
        
        pdf.add_text(
            "Detailed comparison of model performance before and after pruning."
        )
        
        # Speedup chart
        if 'speedup' in self.after_metrics:
            speedup_chart = self.visualizer.create_speedup_chart(
                self.before_metrics['inference_time_ms'],
                self.after_metrics['inference_time_ms'],
                self.after_metrics['speedup']
            )
            pdf.add_image(speedup_chart, width=6*inch)
        
        # Accuracy comparison
        pdf.add_subsection("Accuracy Analysis")
        acc_data = [
            ['Metric', 'Before', 'After', 'Change'],
            [
                'Top-1 Accuracy',
                f"{self.before_metrics.get('accuracy', 0):.2f}%",
                f"{self.after_metrics.get('accuracy', 0):.2f}%",
                f"-{self.after_metrics.get('accuracy_drop', 0):.2f}%"
            ],
        ]
        
        if 'top5_accuracy' in self.before_metrics:
            acc_data.append([
                'Top-5 Accuracy',
                f"{self.before_metrics['top5_accuracy']:.2f}%",
                f"{self.after_metrics.get('top5_accuracy', 0):.2f}%",
                f"-{self.after_metrics.get('top5_accuracy_drop', 0):.2f}%"
            ])
        
        pdf.add_table(acc_data, col_widths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        
        # Efficiency metrics
        pdf.add_subsection("Efficiency Gains")
        efficiency_data = [
            ['Metric', 'Improvement'],
            ['Parameter Reduction', f"{self.after_metrics['param_reduction_pct']:.2f}%"],
            ['Model Size Reduction', f"{self.after_metrics['size_reduction_pct']:.2f}%"],
            ['FLOPs Reduction', f"{self.after_metrics.get('flops_reduction_pct', 0):.2f}%"],
            ['Speedup', f"{self.after_metrics.get('speedup', 1):.2f}x"],
        ]
        pdf.add_table(efficiency_data, col_widths=[3*inch, 3*inch])
    
    def _add_risk_analysis(self, pdf: PDFReportBuilder):
        """Add risk analysis section."""
        pdf.add_section("7. Risk Analysis & Reliability")
        
        pdf.add_text(
            "Assessment of potential risks and reliability concerns with the pruned model."
        )
        
        pdf.add_risk_warnings(self.risks)
        
        # Recommendations
        pdf.add_subsection("Recommendations")
        
        if not self.risks:
            pdf.add_text(
                "✓ The pruned model shows no significant risks. It is ready for deployment.",
                'Success'
            )
        else:
            # Count risk levels
            high_risks = sum(1 for r in self.risks if r['level'] == 'HIGH')
            medium_risks = sum(1 for r in self.risks if r['level'] == 'MEDIUM')
            
            if high_risks > 0:
                pdf.add_text(
                    f"⚠️ {high_risks} high-priority risk(s) detected. "
                    "Consider reducing pruning ratio or using fine-tuning to recover accuracy.",
                    'Warning'
                )
            
            if medium_risks > 0:
                pdf.add_text(
                    f"⚡ {medium_risks} medium-priority warning(s) detected. "
                    "Monitor model performance closely in production.",
                    'Warning'
                )
        
        # Final summary
        pdf.add_subsection("Conclusion")
        
        accuracy_drop = self.after_metrics.get('accuracy_drop', 0)
        param_reduction = self.after_metrics.get('param_reduction_pct', 0)
        
        if accuracy_drop < 1.0 and param_reduction > 30:
            conclusion = """
            <b>Excellent Result:</b> The pruning operation achieved significant parameter reduction 
            with minimal accuracy loss. The pruned model is highly efficient and maintains strong 
            performance characteristics.
            """
        elif accuracy_drop < 3.0 and param_reduction > 20:
            conclusion = """
            <b>Good Result:</b> The pruning operation successfully reduced model size while maintaining 
            acceptable accuracy levels. The pruned model offers a good balance between efficiency and performance.
            """
        else:
            conclusion = """
            <b>Moderate Result:</b> The pruning operation reduced model complexity but with some performance 
            trade-offs. Consider fine-tuning or adjusting the pruning ratio for better results.
            """
        
        pdf.add_text(conclusion)


# Convenience function
def generate_pruning_report(
    model_before: nn.Module,
    model_after: nn.Module,
    model_name: str,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    report_name: Optional[str] = None,
    coverage_analyzer = None,
    pruning_method: str = "coverage",
    pruning_ratio: float = 0.5,
    iterative_steps: int = 1,
    dataset_name: str = "Custom Dataset",
    output_dir: str = "reports"
) -> str:
    """
    Generate a comprehensive pruning report (convenience function).
    
    Args:
        model_before: Model before pruning
        model_after: Model after pruning
        model_name: Name of the model
        dataloader: DataLoader for evaluation
        device: Device for computation
        report_name: Custom report name (optional)
        coverage_analyzer: Coverage analyzer instance (optional)
        pruning_method: Pruning method used
        pruning_ratio: Pruning ratio applied
        iterative_steps: Number of iterative steps
        dataset_name: Name of the dataset
        output_dir: Output directory for reports
        
    Returns:
        Path to generated PDF report
    """
    generator = PruningReportGenerator(
        model_before=model_before,
        model_after=model_after,
        model_name=model_name,
        dataset_name=dataset_name,
        output_dir=output_dir
    )
    
    generator.collect_metrics(
        dataloader=dataloader,
        device=device,
        coverage_analyzer=coverage_analyzer,
        pruning_method=pruning_method,
        pruning_ratio=pruning_ratio,
        iterative_steps=iterative_steps
    )
    
    return generator.generate_report(report_name=report_name)
