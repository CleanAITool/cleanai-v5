"""
Visualizations Module

This module creates charts and visualizations for pruning reports.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import io
from PIL import Image

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ReportVisualizations:
    """
    Creates visualizations for pruning reports.
    """
    
    def __init__(self, dpi: int = 150):
        """
        Initialize visualizations generator.
        
        Args:
            dpi: DPI for saved figures
        """
        self.dpi = dpi
        self.figures = {}
    
    def create_comparison_bar_chart(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
        metrics: List[str] = ['parameters', 'gflops', 'inference_time_ms', 'accuracy'],
        title: str = "Before vs After Pruning"
    ) -> Image.Image:
        """
        Create side-by-side comparison bar chart.
        
        Args:
            before: Before-pruning metrics
            after: After-pruning metrics
            metrics: Metrics to compare
            title: Chart title
            
        Returns:
            PIL Image
        """
        fig, ax = plt.subplots(figsize=(5, 3))
        
        metric_labels = {
            'parameters': 'Parameters (M)',
            'gflops': 'GFLOPs',
            'inference_time_ms': 'Latency (ms)',
            'accuracy': 'Accuracy (%)',
            'model_size_mb': 'Size (MB)'
        }
        
        x = np.arange(len(metrics))
        width = 0.35
        
        before_vals = []
        after_vals = []
        labels = []
        
        for metric in metrics:
            if metric in before and metric in after:
                before_val = before[metric]
                after_val = after[metric]
                
                # Normalize for visualization
                if metric == 'parameters':
                    before_val = before_val / 1e6  # Convert to millions
                    after_val = after_val / 1e6
                
                before_vals.append(before_val)
                after_vals.append(after_val)
                labels.append(metric_labels.get(metric, metric))
        
        bars1 = ax.bar(x - width/2, before_vals, width, label='Before', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, after_vals, width, label='After', color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
    
    def create_reduction_pie_chart(
        self,
        reduction_pct: float,
        title: str = "Parameter Reduction"
    ) -> Image.Image:
        """
        Create pie chart showing reduction percentage.
        
        Args:
            reduction_pct: Reduction percentage
            title: Chart title
            
        Returns:
            PIL Image
        """
        fig, ax = plt.subplots(figsize=(5, 3))
        
        remaining_pct = 100 - reduction_pct
        sizes = [reduction_pct, remaining_pct]
        labels = [f'Pruned\n{reduction_pct:.1f}%', f'Remaining\n{remaining_pct:.1f}%']
        colors = ['#e74c3c', '#2ecc71']
        explode = (0.1, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='', shadow=True, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
    
    def create_layer_coverage_bar_chart(
        self,
        layer_coverage: Dict[str, Dict[str, float]],
        top_n: int = 20,
        title: str = "Layer-wise Coverage Analysis"
    ) -> Image.Image:
        """
        Create bar chart of layer-wise coverage.
        
        Args:
            layer_coverage: Layer coverage data
            top_n: Number of top layers to show
            title: Chart title
            
        Returns:
            PIL Image
        """
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # Sort layers by coverage
        sorted_layers = sorted(
            layer_coverage.items(),
            key=lambda x: x[1].get('mean', 0),
            reverse=True
        )[:top_n]
        
        layer_names = [name.split('.')[-1] if '.' in name else name for name, _ in sorted_layers]
        coverage_means = [data.get('mean', 0) for _, data in sorted_layers]
        
        # Color code by coverage level
        colors = ['#2ecc71' if cov > 0.5 else '#f39c12' if cov > 0.2 else '#e74c3c' 
                  for cov in coverage_means]
        
        bars = ax.barh(range(len(layer_names)), coverage_means, color=colors, alpha=0.7)
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names)
        ax.set_xlabel('Mean Coverage', fontweight='bold')
        ax.set_ylabel('Layer', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, coverage_means)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=8)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='High (>0.5)'),
            Patch(facecolor='#f39c12', label='Medium (0.2-0.5)'),
            Patch(facecolor='#e74c3c', label='Low (<0.2)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
    
    def create_coverage_heatmap(
        self,
        coverage_data: Dict[str, torch.Tensor],
        max_layers: int = 30,
        max_channels: int = 64,
        title: str = "Coverage Heatmap"
    ) -> Image.Image:
        """
        Create heatmap of coverage across layers and channels.
        
        Args:
            coverage_data: Layer-wise coverage tensors
            max_layers: Maximum layers to show
            max_channels: Maximum channels per layer
            title: Chart title
            
        Returns:
            PIL Image
        """
        import torch
        
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # Prepare data matrix
        layer_names = list(coverage_data.keys())[:max_layers]
        
        matrix = []
        for layer_name in layer_names:
            scores = coverage_data[layer_name].cpu().numpy()
            if len(scores) > max_channels:
                # Sample channels
                indices = np.linspace(0, len(scores)-1, max_channels, dtype=int)
                scores = scores[indices]
            elif len(scores) < max_channels:
                # Pad with zeros
                scores = np.pad(scores, (0, max_channels - len(scores)), constant_values=0)
            matrix.append(scores)
        
        matrix = np.array(matrix)
        
        # Create heatmap
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # Labels
        ax.set_xlabel('Channel Index', fontweight='bold')
        ax.set_ylabel('Layer', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Y-axis labels (layer names)
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels([name.split('.')[-1] for name in layer_names], fontsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage Score', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
    
    def create_speedup_chart(
        self,
        before_latency: float,
        after_latency: float,
        title: str = "Inference Speedup"
    ) -> Image.Image:
        """
        Create speedup comparison chart.
        
        Args:
            before_latency: Before latency in ms
            after_latency: After latency in ms
            title: Chart title
            
        Returns:
            PIL Image
        """
        fig, ax = plt.subplots(figsize=(5, 3))
        
        speedup = before_latency / after_latency
        
        categories = ['Before\nPruning', 'After\nPruning']
        latencies = [before_latency, after_latency]
        colors = ['#e74c3c', '#2ecc71']
        
        bars = ax.bar(categories, latencies, color=colors, alpha=0.7, width=0.6)
        
        ax.set_ylabel('Latency (ms)', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, latencies):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                   f'{val:.2f} ms', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add speedup annotation
        ax.text(0.5, max(latencies) * 0.8,
               f'Speedup: {speedup:.2f}x',
               ha='center', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
    
    def create_accuracy_vs_pruning_chart(
        self,
        pruning_ratios: List[float],
        accuracies: List[float],
        title: str = "Accuracy vs Pruning Ratio"
    ) -> Image.Image:
        """
        Create line chart of accuracy vs pruning ratio.
        
        Args:
            pruning_ratios: List of pruning ratios
            accuracies: Corresponding accuracies
            title: Chart title
            
        Returns:
            PIL Image
        """
        fig, ax = plt.subplots(figsize=(5, 3))
        
        ax.plot(pruning_ratios, accuracies, marker='o', linewidth=2, 
               markersize=8, color='#3498db', label='Accuracy')
        ax.fill_between(pruning_ratios, accuracies, alpha=0.3, color='#3498db')
        
        ax.set_xlabel('Pruning Ratio (%)', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Annotate points
        for ratio, acc in zip(pruning_ratios, accuracies):
            ax.annotate(f'{acc:.1f}%', (ratio, acc),
                       textcoords="offset points", xytext=(0,10),
                       ha='center', fontsize=8)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
