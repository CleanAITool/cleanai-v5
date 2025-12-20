# Reporting Feature Documentation

## Overview

CleanAI provides a comprehensive PDF reporting system that automatically generates detailed analysis reports after pruning neural networks. The reports are professional, include visualizations, and are written in English.

## Features

### Report Sections

Each report contains 7 main sections:

1. **Cover Page & Executive Summary**

   - Model name and type
   - Dataset information
   - Pruning ratio and timestamp
   - Key metrics comparison table

2. **Model & Experiment Information**

   - Model architecture details
   - Layer count and structure
   - Experiment configuration

3. **Pre-Pruning Coverage Analysis** (if applicable)

   - Overall coverage statistics
   - Layer-wise coverage visualization
   - Coverage heatmaps
   - Low-coverage layers identification

4. **Pruning Decision Mechanism (Explainability)**

   - Pruning method explanation
   - Importance criterion details
   - Layer-wise pruning distribution
   - Configuration parameters

5. **Post-Pruning Model Structure**

   - Architecture comparison charts
   - Parameter reduction visualization
   - Pie charts showing pruned vs. remaining parameters

6. **Performance Comparison**

   - Speedup analysis with charts
   - Accuracy comparison tables
   - Efficiency gains summary
   - FLOPs reduction metrics

7. **Risk Analysis & Reliability**
   - Automatic risk detection
   - Warning levels (HIGH/MEDIUM)
   - Recommendations
   - Final conclusions

### Visualizations

Reports include multiple types of charts:

- **Bar Charts**: Comparing before/after metrics
- **Pie Charts**: Parameter reduction percentages
- **Heatmaps**: Coverage analysis across layers
- **Line Graphs**: Speedup and efficiency metrics
- **Tables**: Detailed metrics and statistics

## Usage

### Quick Start

The simplest way to generate a report:

```python
from cleanai import CoveragePruner, generate_pruning_report

# ... after pruning your model ...

report_path = generate_pruning_report(
    model_before=original_model,
    model_after=pruned_model,
    model_name="ResNet50",
    dataloader=test_loader,
    device=device,
    report_name="my_analysis_report",  # Custom name from API
    pruning_method="coverage",
    pruning_ratio=0.5
)

print(f"Report saved to: {report_path}")
```

### Using CoveragePruner Integration

You can also generate reports directly from the pruner:

```python
from cleanai import CoveragePruner

# Create pruner
pruner = CoveragePruner(
    model=model,
    example_inputs=example_inputs,
    importance_method='coverage'
)

# Analyze coverage
pruner.analyze_coverage(dataloader=train_loader, device=device)

# Perform pruning
pruned_model = pruner.prune(pruning_ratio=0.5)

# Generate report
report_path = pruner.generate_report(
    model_before=original_model,
    model_after=pruned_model,
    dataloader=test_loader,
    report_name="coverage_pruning_report"
)
```

### Advanced Usage

For more control over the reporting process:

```python
from cleanai.reporting import PruningReportGenerator

# Create report generator
generator = PruningReportGenerator(
    model_before=original_model,
    model_after=pruned_model,
    model_name="ResNet50",
    dataset_name="ImageNet",
    output_dir="my_reports"
)

# Collect metrics
generator.collect_metrics(
    dataloader=test_loader,
    device=device,
    coverage_analyzer=my_analyzer,
    pruning_method="coverage",
    pruning_ratio=0.5,
    iterative_steps=1
)

# Generate report with custom name
report_path = generator.generate_report(
    report_name="detailed_analysis_2024"
)
```

## API Parameters

### `generate_pruning_report()`

Main convenience function for report generation.

**Parameters:**

- `model_before` (nn.Module): Original model before pruning
- `model_after` (nn.Module): Pruned model after pruning
- `model_name` (str): Name of the model
- `dataloader` (DataLoader): DataLoader for evaluation
- `device` (torch.device): Device for computation
- `report_name` (str, optional): Custom report name (without .pdf extension)
- `coverage_analyzer` (optional): CoverageAnalyzer instance for detailed analysis
- `pruning_method` (str): Method used ('coverage', 'wanda', 'magnitude')
- `pruning_ratio` (float): Pruning ratio applied (0.0-1.0)
- `iterative_steps` (int): Number of iterative pruning steps
- `dataset_name` (str): Name of the dataset
- `output_dir` (str): Output directory for reports (default: "reports")

**Returns:**

- `str`: Path to the generated PDF report

### `PruningReportGenerator.collect_metrics()`

Collects all metrics needed for the report.

**Parameters:**

- `dataloader` (DataLoader): DataLoader for evaluation
- `device` (torch.device): Device for computation
- `coverage_analyzer` (optional): CoverageAnalyzer instance
- `pruning_method` (str): Pruning method name
- `pruning_ratio` (float): Target pruning ratio
- `iterative_steps` (int): Number of iterative steps

### `PruningReportGenerator.generate_report()`

Generates the PDF report.

**Parameters:**

- `report_name` (str, optional): Custom report name

**Returns:**

- `str`: Path to generated PDF

## Report Name Convention

If `report_name` is not provided, the system automatically generates names:

```
pruning_report_{model_name}_{timestamp}.pdf
```

Example: `pruning_report_ResNet50_20240115_143052.pdf`

If provided, the name is used directly:

```python
report_name = "my_custom_report"  # Becomes: my_custom_report.pdf
```

## Output Directory

Reports are saved to the specified `output_dir` (default: "reports/"). The directory is created automatically if it doesn't exist.

```python
# Custom output directory
report_path = generate_pruning_report(
    ...,
    output_dir="my_custom_reports"
)
```

## Examples

### Example 1: Basic Report Generation

```python
import torch
from cleanai import CoveragePruner, generate_pruning_report

# Setup
device = torch.device('cuda')
model = MyModel()

# Prune
pruner = CoveragePruner(model, example_inputs, importance_method='coverage')
pruner.analyze_coverage(train_loader, device)
pruned_model = pruner.prune(0.5)

# Generate report
report = generate_pruning_report(
    model_before=model,
    model_after=pruned_model,
    model_name="MyModel",
    dataloader=test_loader,
    device=device,
    report_name="my_first_report"
)
```

### Example 2: WANDA Method with Report

```python
from cleanai import CoveragePruner, generate_pruning_report

# Use WANDA pruning
pruner = CoveragePruner(
    model=model,
    example_inputs=example_inputs,
    importance_method='wanda'  # Use WANDA
)

# Prune
pruned_model = pruner.prune(pruning_ratio=0.3)

# Generate report
report = generate_pruning_report(
    model_before=original_model,
    model_after=pruned_model,
    model_name="ResNet-18",
    dataloader=test_loader,
    device=device,
    report_name="wanda_pruning_report",
    pruning_method="wanda",
    pruning_ratio=0.3
)
```

### Example 3: Iterative Pruning with Report

```python
# Iterative pruning
pruner = CoveragePruner(
    model=model,
    example_inputs=example_inputs,
    importance_method='coverage'
)

pruner.analyze_coverage(train_loader, device)

# Prune in 5 steps
pruned_model = pruner.prune(
    pruning_ratio=0.6,
    iterative_steps=5
)

# Generate detailed report
report = pruner.generate_report(
    model_before=original_model,
    model_after=pruned_model,
    dataloader=test_loader,
    report_name="iterative_pruning_analysis"
)
```

## Metrics Collected

The reporting system automatically collects:

### Before Pruning:

- Total parameters
- Model size (MB)
- GFLOPs
- Inference time
- Accuracy (Top-1 and Top-5)
- Model architecture summary

### After Pruning:

- Remaining parameters
- Parameter reduction percentage
- Model size reduction
- FLOPs reduction
- Speedup factor
- Accuracy drop
- Inference time improvement

### Coverage Analysis (if applicable):

- Average/median/min/max coverage
- Layer-wise coverage statistics
- Low-coverage layers
- Coverage distribution

### Risk Analysis:

- Accuracy drop warnings
- Over-pruning detection
- Coverage-related risks
- Reliability assessment

## Risk Detection

The system automatically detects and reports:

- **HIGH Risk**: Accuracy drop > 5%
- **MEDIUM Risk**: Accuracy drop > 2%
- **HIGH Risk**: Pruning ratio > 70%
- **MEDIUM Risk**: Layers with coverage < 30%

Example warnings in report:

```
⚠️ [HIGH] Accuracy Drop: Model accuracy decreased by 6.5%.
   Consider reducing pruning ratio.

⚡ [MEDIUM] Low Coverage: Layer 'conv3_x' has 25% coverage.
   May affect model reliability.
```

## Requirements

Make sure you have the required dependencies:

```bash
pip install matplotlib seaborn reportlab Pillow
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Output Format

Reports are generated as PDF files with:

- **Page Size**: A4
- **Language**: English
- **Font**: Helvetica (standard PDF font)
- **Colors**: Professional blue/gray theme
- **Charts**: High-resolution PNG embedded images

## Tips

1. **Report Names**: Use descriptive names that indicate the experiment

   ```python
   report_name = f"{model_name}_{method}_ratio{ratio:.0%}_experiment"
   ```

2. **Coverage Analysis**: Always provide `coverage_analyzer` for detailed insights

   ```python
   coverage_analyzer=pruner.coverage_analyzer
   ```

3. **Dataset Size**: Use a representative subset (1000+ samples) for accurate metrics

4. **Memory**: Report generation requires matplotlib rendering; ensure sufficient memory

5. **Custom Output**: Organize reports by experiment
   ```python
   output_dir = f"reports/experiment_{experiment_id}"
   ```

## Troubleshooting

### Issue: "No module named 'reportlab'"

**Solution**: Install reportlab

```bash
pip install reportlab
```

### Issue: Report generation is slow

**Solution**: Use a smaller dataset subset for evaluation

```python
test_subset = torch.utils.data.Subset(test_dataset, range(500))
```

### Issue: Charts not appearing in report

**Solution**: Ensure matplotlib backend is set correctly (automatically handled by the library)

### Issue: Memory error during report generation

**Solution**: Reduce batch size in DataLoader or use smaller test set

## Future Enhancements

Planned features for future versions:

- [ ] HTML report format
- [ ] Interactive charts
- [ ] Multi-model comparison reports
- [ ] Custom chart configurations
- [ ] Report templates
- [ ] Markdown export option

## Support

For issues or questions about reporting:

1. Check this documentation
2. Review example scripts in `examples/`
3. Open an issue on GitHub
4. Check the CleanAI documentation

## License

The reporting feature is part of CleanAI and follows the same license.
