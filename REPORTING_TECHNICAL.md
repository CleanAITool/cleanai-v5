# CleanAI Reporting System - Technical Overview

## Introduction

The CleanAI reporting system provides comprehensive, automated PDF report generation for neural network pruning analysis. This document provides a technical overview of the system architecture and implementation.

## System Architecture

### Components

The reporting system consists of 4 main modules:

```
cleanai/reporting/
├── __init__.py              # Module exports
├── metrics_collector.py     # Metrics collection
├── visualizations.py        # Chart generation
├── pdf_builder.py           # PDF construction
└── report_generator.py      # Main orchestrator
```

### Component Responsibilities

#### 1. MetricsCollector (`metrics_collector.py`)

**Purpose**: Collects all metrics before and after pruning

**Key Methods**:

- `collect_before_pruning()`: Pre-pruning metrics (params, FLOPs, accuracy, inference time)
- `collect_after_pruning()`: Post-pruning metrics and comparisons
- `collect_coverage_analysis()`: Coverage statistics and layer analysis
- `collect_pruning_info()`: Pruning configuration and decisions
- `analyze_risks()`: Risk detection and warnings

**Output**: Dictionary containing all collected metrics

**Example**:

```python
collector = MetricsCollector()
before = collector.collect_before_pruning(model, dataloader, device, "ResNet50", "ImageNet")
# Returns: {parameters, gflops, accuracy, inference_time_ms, model_size_mb, ...}
```

#### 2. ReportVisualizations (`visualizations.py`)

**Purpose**: Generate charts and visualizations for the report

**Key Methods**:

- `create_comparison_bar_chart()`: Before/after comparison
- `create_reduction_pie_chart()`: Parameter reduction visualization
- `create_layer_coverage_bar_chart()`: Layer-wise coverage
- `create_coverage_heatmap()`: 2D coverage visualization
- `create_speedup_chart()`: Latency improvement
- `create_accuracy_vs_pruning_chart()`: Accuracy curves

**Output**: PIL Image objects ready for PDF embedding

**Technical Details**:

- Uses matplotlib with Agg backend (non-interactive)
- Generates high-resolution images (300 DPI)
- Color scheme: Professional blue/gray palette
- Returns PIL Image objects for PDF embedding

**Example**:

```python
visualizer = ReportVisualizations()
chart = visualizer.create_comparison_bar_chart(before_metrics, after_metrics)
# Returns: PIL Image object
```

#### 3. PDFReportBuilder (`pdf_builder.py`)

**Purpose**: Low-level PDF construction using ReportLab

**Key Methods**:

- `add_cover_page()`: Title page with model info
- `add_executive_summary()`: Summary with comparison table
- `add_section()`: Section headers
- `add_table()`: Data tables with styling
- `add_image()`: Embed images (charts)
- `add_risk_warnings()`: Risk analysis section
- `build()`: Generate final PDF

**Technical Details**:

- Uses ReportLab library for PDF generation
- Page size: A4
- Margins: 0.75 inches
- Font: Helvetica family
- Custom styles for headers, warnings, success messages

**Example**:

```python
pdf = PDFReportBuilder("report.pdf", title="Pruning Report")
pdf.add_cover_page(model_name, model_type, dataset, pruning_ratio)
pdf.add_section("Performance Analysis")
pdf.add_image(chart_image, width=6*inch)
pdf.build()
```

#### 4. PruningReportGenerator (`report_generator.py`)

**Purpose**: Main orchestrator that coordinates all components

**Key Methods**:

- `collect_metrics()`: Orchestrates metrics collection
- `generate_report()`: Generates complete PDF report

**Workflow**:

1. Initialize components (MetricsCollector, ReportVisualizations, PDFReportBuilder)
2. Collect all metrics using MetricsCollector
3. Generate visualizations using ReportVisualizations
4. Build PDF using PDFReportBuilder
5. Return path to generated report

**Example**:

```python
generator = PruningReportGenerator(model_before, model_after, "ResNet50")
generator.collect_metrics(dataloader, device, coverage_analyzer, "coverage", 0.5)
report_path = generator.generate_report(report_name="my_report")
```

## Report Structure

### 7 Main Sections

#### Section 1: Cover Page & Executive Summary

- Model name, type, dataset
- Timestamp and framework version
- Comparison table: Parameters, Size, FLOPs, Accuracy, Inference Time

#### Section 2: Model & Experiment Information

- Model architecture details
- Total layers
- Dataset information
- Architecture summary

#### Section 3: Pre-Pruning Coverage Analysis (Optional)

- Overall coverage statistics (average, median, min, max)
- Layer-wise coverage bar chart
- Coverage heatmap
- Low-coverage layers table

#### Section 4: Pruning Decision Mechanism

- Pruning method explanation
- Importance criterion details (Coverage, WANDA, Magnitude)
- Configuration parameters
- Layer-wise pruning distribution

#### Section 5: Post-Pruning Model Structure

- Architecture comparison bar chart
- Parameter reduction pie chart
- Remaining vs. pruned parameters

#### Section 6: Performance Comparison

- Speedup chart
- Accuracy comparison table
- Efficiency gains summary
- FLOPs reduction metrics

#### Section 7: Risk Analysis & Reliability

- Automatic risk detection
- Warning levels (HIGH/MEDIUM)
- Recommendations
- Final conclusion

## Data Flow

```
User Code
    ↓
generate_pruning_report() or PruningReportGenerator
    ↓
collect_metrics()
    ↓
┌─────────────────┐
│ MetricsCollector│
└─────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Before Metrics                        │
│ - Parameters, FLOPs, Size             │
│ - Accuracy, Inference Time            │
│ - Architecture Summary                │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ After Metrics                         │
│ - Remaining Parameters                │
│ - Reduction Percentages               │
│ - Accuracy Drop, Speedup              │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Coverage Analysis (if available)      │
│ - Average/Median Coverage             │
│ - Layer Statistics                    │
│ - Low-Coverage Layers                 │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Risk Analysis                         │
│ - Accuracy Drop Warnings              │
│ - Over-Pruning Detection              │
│ - Coverage Risks                      │
└──────────────────────────────────────┘
    ↓
generate_report(report_name)
    ↓
┌──────────────────┐       ┌──────────────────┐
│ Visualizations   │◄──────│ PDFReportBuilder │
│ - Bar Charts     │       │ - Cover Page     │
│ - Pie Charts     │       │ - Sections       │
│ - Heatmaps       │       │ - Tables         │
└──────────────────┘       └──────────────────┘
    ↓
📄 Final PDF Report
```

## Risk Detection System

The system automatically detects and reports:

### Accuracy-Based Risks

| Condition          | Level  | Message                          |
| ------------------ | ------ | -------------------------------- |
| Accuracy drop > 5% | HIGH   | Significant accuracy degradation |
| Accuracy drop > 2% | MEDIUM | Moderate accuracy decrease       |
| Accuracy drop < 1% | None   | Minimal impact                   |

### Pruning Ratio Risks

| Condition   | Level  | Message                                 |
| ----------- | ------ | --------------------------------------- |
| Ratio > 70% | HIGH   | Aggressive pruning may affect stability |
| Ratio > 50% | MEDIUM | High pruning ratio, monitor closely     |

### Coverage Risks

| Condition            | Level  | Message                             |
| -------------------- | ------ | ----------------------------------- |
| Layer coverage < 30% | MEDIUM | Low coverage may affect reliability |
| Layer coverage < 50% | LOW    | Below-average coverage detected     |

## Visualization Details

### Chart Specifications

#### Comparison Bar Chart

- Type: Grouped bar chart
- Metrics: Parameters, Size, FLOPs, Accuracy, Inference Time
- Colors: Blue (before), Orange (after)
- Resolution: 300 DPI
- Size: 10x6 inches

#### Reduction Pie Chart

- Type: Pie chart with explosion
- Segments: Pruned (red), Remaining (green)
- Shows: Percentage reduction
- Resolution: 300 DPI
- Size: 8x8 inches

#### Coverage Heatmap

- Type: 2D heatmap
- Colormap: RdYlGn (Red-Yellow-Green)
- Shows: Layer-wise coverage distribution
- Resolution: 300 DPI
- Size: 12x8 inches

#### Speedup Chart

- Type: Horizontal bar chart with line
- Shows: Before/after inference time + speedup factor
- Colors: Blue (before), Green (after), Red line (speedup)
- Resolution: 300 DPI
- Size: 10x6 inches

## Integration Points

### Integration with CoveragePruner

The reporting system is tightly integrated with `CoveragePruner`:

```python
pruner = CoveragePruner(...)
pruner.analyze_coverage(...)
pruned_model = pruner.prune(pruning_ratio=0.5)

# Generate report directly
report_path = pruner.generate_report(
    model_before=original_model,
    model_after=pruned_model,
    dataloader=test_loader,
    report_name="my_report"
)
```

### Standalone Usage

Can also be used independently:

```python
from cleanai.reporting import generate_pruning_report

report_path = generate_pruning_report(
    model_before=model1,
    model_after=model2,
    model_name="ResNet50",
    dataloader=test_loader,
    device=device,
    report_name="standalone_report"
)
```

## Performance Considerations

### Metrics Collection

- **Time**: ~30 seconds for 1000 samples
- **Memory**: ~200MB additional (for activations)
- **Optimization**: Use subset of test data for faster generation

### Visualization Generation

- **Time**: ~5 seconds per chart
- **Memory**: ~50MB per chart
- **Optimization**: Charts generated sequentially, memory released after each

### PDF Generation

- **Time**: ~10 seconds for full report
- **File Size**: 2-5 MB typical (with charts)
- **Optimization**: Automatic image compression

### Total Time

- **Typical**: 1-2 minutes for full report
- **Fast Mode**: Use smaller dataset (500 samples)

## Error Handling

### Common Issues

1. **Missing Dependencies**

   - Automatic check for reportlab, matplotlib, seaborn
   - Clear error messages with installation instructions

2. **Insufficient Data**

   - Validates dataloader has samples
   - Warns if sample size too small

3. **Memory Issues**

   - Batch processing for large models
   - Automatic garbage collection

4. **Chart Generation Failures**
   - Graceful degradation (report without chart)
   - Error logging for debugging

## Customization Options

### Report Name

```python
# Auto-generated
report_path = generator.generate_report()
# → "pruning_report_ResNet50_20240115_143052.pdf"

# Custom name
report_path = generator.generate_report(report_name="my_experiment")
# → "my_experiment.pdf"
```

### Output Directory

```python
generator = PruningReportGenerator(..., output_dir="custom_reports")
```

### Chart Styling

```python
visualizer = ReportVisualizations()
# Modify color schemes (future feature)
# visualizer.set_color_scheme("professional")
```

## Future Enhancements

Planned features:

1. **HTML Reports**: Web-based interactive reports
2. **Custom Templates**: User-defined report layouts
3. **Multi-Model Comparison**: Compare multiple pruning methods
4. **Export Formats**: Markdown, LaTeX support
5. **Interactive Charts**: JavaScript-based visualizations
6. **Cloud Storage**: Direct upload to S3, GCS
7. **Email Integration**: Auto-send reports via email

## Dependencies

Required libraries:

```python
# Core dependencies
torch >= 2.0.0
torchvision >= 0.15.0
torch-pruning >= 1.3.0
numpy >= 1.21.0

# Reporting dependencies
matplotlib >= 3.5.0      # Chart generation
seaborn >= 0.12.0        # Advanced visualizations
reportlab >= 4.0.0       # PDF generation
Pillow >= 9.0.0          # Image processing
```

## API Reference

### Main Functions

#### `generate_pruning_report()`

Convenience function for one-line report generation.

**Signature**:

```python
def generate_pruning_report(
    model_before: nn.Module,
    model_after: nn.Module,
    model_name: str,
    dataloader: DataLoader,
    device: torch.device,
    report_name: Optional[str] = None,
    coverage_analyzer = None,
    pruning_method: str = "coverage",
    pruning_ratio: float = 0.5,
    iterative_steps: int = 1,
    dataset_name: str = "Custom Dataset",
    output_dir: str = "reports"
) -> str
```

**Returns**: Path to generated PDF

### Class Methods

#### `MetricsCollector.collect_before_pruning()`

Collects metrics before pruning.

**Returns**: Dictionary with keys:

- `parameters`: Total parameter count
- `gflops`: Giga-FLOPs
- `model_size_mb`: Model size in MB
- `accuracy`: Test accuracy
- `inference_time_ms`: Average inference time
- `architecture_summary`: Text summary

#### `ReportVisualizations.create_comparison_bar_chart()`

Creates comparison chart.

**Returns**: PIL Image object

#### `PDFReportBuilder.build()`

Generates final PDF file.

**Returns**: None (writes to file)

## Testing

### Unit Tests

```bash
pytest tests/test_reporting.py
```

### Integration Tests

```bash
python examples/generate_report.py
```

### Manual Testing

1. Run example script
2. Verify PDF opens correctly
3. Check all sections present
4. Validate charts render properly

## Conclusion

The CleanAI reporting system provides a comprehensive, automated solution for documenting and analyzing neural network pruning results. With its modular architecture, extensive visualizations, and professional PDF output, it enables researchers and practitioners to easily share and understand pruning results.

For usage examples, see [REPORTING_GUIDE.md](REPORTING_GUIDE.md).
