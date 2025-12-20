# CleanAI: Coverage-Based Neural Network Pruning 🚀

A modular and extensible framework for pruning neural networks using neuron coverage analysis and various importance metrics including WANDA.

## ✨ Features

- **Multiple Importance Metrics:**

  - **Coverage-based**: Prune based on neuron activation patterns
  - **WANDA**: Weight × Activation importance (training-free)
  - **Adaptive**: Dynamic recomputation during iterative pruning
  - **Standard methods**: Magnitude, Taylor, Hessian (via Torch-Pruning)

- **📝 Comprehensive Reporting:**

  - **Automatic PDF Reports**: Generate professional analysis reports after pruning
  - **Visual Analytics**: Charts, graphs, and heatmaps for coverage and performance
  - **Risk Analysis**: Automatic detection of potential issues
  - **Explainability**: Detailed explanation of pruning decisions
  - **English Language**: Professional documentation for sharing results

- **Modular Architecture**: Clean separation of analyzers, importance metrics, and pruners
- **Easy to Use**: High-level API for quick pruning experiments
- **Extensible**: Easy to add custom importance metrics
- **Production Ready**: Comprehensive utilities for evaluation and comparison

## 🏗️ Project Structure

```
CleanAI_v5/
├── cleanai/                      # Main package
│   ├── __init__.py
│   ├── importance/               # Importance metrics
│   │   ├── coverage.py          # Coverage-based importance
│   │   ├── wanda.py             # WANDA importance
│   │   └── adaptive.py          # Adaptive coverage
│   ├── analyzers/                # Activation analyzers
│   │   └── coverage_analyzer.py
│   ├── pruners/                  # Pruning algorithms
│   │   └── coverage_pruner.py
│   ├── reporting/                # PDF reporting system 📝
│   │   ├── report_generator.py  # Main report generator
│   │   ├── metrics_collector.py # Metrics collection
│   │   ├── visualizations.py    # Chart generation
│   │   └── pdf_builder.py       # PDF construction
│   └── utils/                    # Utility functions
│       ├── model_utils.py
│       └── evaluation.py
├── examples/                     # Example scripts
│   ├── simple_pruning.py
│   ├── wanda_comparison.py
│   └── generate_report.py       # Report generation example
├── REPORTING_GUIDE.md            # Detailed reporting documentation
├── main.py                       # Main entry point
└── requirements.txt
```

## 📦 Installation

### Requirements

```bash
pip install torch torchvision
pip install torch-pruning
pip install numpy

# For reporting features
pip install matplotlib seaborn reportlab Pillow
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Clone and Setup

```bash
git clone <your-repo-url>
cd CleanAI_v5
```

## 🎯 Quick Start

### Basic Usage

```python
from cleanai import CoveragePruner, evaluate_model, count_parameters
import torch
from torchvision import models

# Load model
model = models.resnet18(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create pruner with coverage-based importance
pruner = CoveragePruner(
    model=model,
    example_inputs=torch.randn(1, 3, 224, 224).to(device),
    test_loader=test_loader,
    pruning_ratio=0.3,              # Prune 30% of channels
    importance_method='coverage',    # or 'wanda', 'magnitude'
    global_pruning=True,
    device=device
)

# Execute pruning
pruned_model = pruner.prune()

# Evaluate
accuracy = evaluate_model(pruned_model, test_loader, device)
print(f"Accuracy: {accuracy:.2f}%")
```

### Using WANDA Importance

```python
# WANDA: Weight × Activation importance
pruner = CoveragePruner(
    model=model,
    example_inputs=example_inputs,
    test_loader=test_loader,
    pruning_ratio=0.3,
    importance_method='wanda',       # WANDA method
    global_pruning=True,
    max_batches=50,                 # Calibration batches
    device=device
)

pruned_model = pruner.prune()
```

### 📊 Generate Comprehensive Reports

Automatically generate professional PDF reports with detailed analysis:

```python
from cleanai import generate_pruning_report

# After pruning, generate a report
report_path = generate_pruning_report(
    model_before=original_model,
    model_after=pruned_model,
    model_name="ResNet-18",
    dataloader=test_loader,
    device=device,
    report_name="my_pruning_analysis",  # Custom name from your API
    pruning_method="coverage",
    pruning_ratio=0.3
)

print(f"Report saved to: {report_path}")
```

**Report includes:**

- Executive summary with key metrics
- Model architecture details
- Coverage analysis with heatmaps
- Pruning decision explanations
- Performance comparison charts
- Risk analysis and recommendations

See [REPORTING_GUIDE.md](REPORTING_GUIDE.md) for detailed documentation.

### Iterative Pruning

```python
    model=model,
    example_inputs=example_inputs,
    test_loader=test_loader,
    pruning_ratio=0.3,  # Remove 30% of channels
    coverage_metric='normalized_mean',
    global_pruning=True,
    iterative_steps=5
)

# Perform pruning
pruner.prune()

# Get pruned model
pruned_model = pruner.get_model()
```

### Command Line Usage

```bash
# Prune ResNet-18 on CIFAR-10 with 30% pruning ratio
python main.py --model resnet18 --dataset cifar10 --pruning-ratio 0.3

# Use global pruning with adaptive coverage
python main.py --model resnet18 --dataset cifar10 \
    --pruning-ratio 0.5 --global-pruning --adaptive \
    --iterative-steps 5

# Use pretrained model and save results
python main.py --model resnet50 --dataset cifar100 \
    --pretrained --pruning-ratio 0.4 \
    --save-path ./pruned_resnet50.pth
```

## 📊 Coverage Metrics

The framework supports multiple coverage metrics:

1. **`normalized_mean`** (default): Average activation normalized by global maximum

   - Good for general use
   - Balances absolute and relative importance

2. **`frequency`**: Proportion of samples where neuron is active

   - Focuses on activation frequency
   - Good for identifying dead neurons

3. **`mean_absolute`**: Absolute mean activation value

   - Direct activation magnitude
   - Similar to traditional magnitude-based methods

4. **`combined`**: Geometric mean of normalized_mean and frequency
   - Best of both worlds
   - Most comprehensive metric

## 🏗️ Architecture

### Module Structure

```
CleanAI_v5/
├── coverage_analyzer.py      # Activation collection and coverage computation
├── coverage_importance.py    # Torch-Pruning compatible importance criterion
├── coverage_pruner.py        # High-level pruner interface
├── utils.py                  # Utility functions (evaluation, comparison)
├── main.py                   # Command-line interface
└── README.md                 # This file
```

### Key Components

#### 1. **CoverageAnalyzer**

- Collects activations from test data using forward hooks
- Computes coverage metrics for each channel/neuron
- Supports multiple coverage metric types

#### 2. **NeuronCoverageImportance**

- Custom importance criterion for Torch-Pruning
- Converts coverage scores to pruning importance
- Lower coverage → Higher pruning priority

#### 3. **CoveragePruner**

- High-level API for coverage-based pruning
- Wraps Torch-Pruning's BasePruner
- Supports global/local and iterative pruning

## 🔬 How It Works

### 1. Coverage Analysis Phase

```python
# Register hooks on target layers (Conv2d, Linear)
analyzer = CoverageAnalyzer(model, device)
analyzer.register_hooks()

# Collect activations from test data
analyzer.collect_activations(test_loader)

# Compute coverage scores
coverage_scores = analyzer.compute_neuron_coverage(metric='normalized_mean')
```

### 2. Importance Computation

```python
# For each channel, convert coverage to importance
importance = 1.0 / (coverage + epsilon)

# Lower coverage → Higher importance for pruning
# This means less active neurons will be pruned first
```

### 3. Structured Pruning

```python
# Torch-Pruning handles dependencies automatically
# When pruning Conv layer output channels:
#   - BatchNorm channels
#   - Next layer input channels
#   - Skip connection channels
# All are pruned together correctly
```

## 📈 Example Results

### ResNet-18 on CIFAR-10

| Metric         | Original | Pruned (30%) | Reduction    |
| -------------- | -------- | ------------ | ------------ |
| Parameters     | 11.17M   | 7.82M        | 30.0%        |
| FLOPs          | 0.56G    | 0.39G        | 30.4%        |
| Inference Time | 2.45ms   | 1.78ms       | 1.38x faster |
| Accuracy       | 95.2%    | 94.1%        | -1.1%        |

_Results may vary based on model initialization and test data_

## 🎛️ Parameters

### Pruning Parameters

- `pruning_ratio` (float, 0.0-1.0): Proportion of channels to remove
- `coverage_metric` (str): Coverage computation method
- `global_pruning` (bool): Global vs uniform per-layer pruning
- `iterative_steps` (int): Number of iterative pruning steps
- `adaptive` (bool): Recompute coverage during iterations

### Advanced Parameters

- `max_batches` (int): Limit test batches for faster analysis
- `round_to` (int): Round channels to multiples (e.g., 8 for GPU efficiency)
- `ignored_layers` (list): Layers to exclude from pruning
- `pruning_ratio_dict` (dict): Layer-specific pruning ratios

## 🔧 Command Line Arguments

```bash
# Model and Dataset
--model             Model architecture (resnet18, resnet34, resnet50, vgg16, mobilenet_v2)
--dataset           Dataset (cifar10, cifar100, mnist)
--checkpoint        Path to pretrained model checkpoint
--pretrained        Use ImageNet pretrained weights

# Pruning Configuration
--pruning-ratio     Pruning ratio (default: 0.3)
--coverage-metric   Coverage metric (default: normalized_mean)
--global-pruning    Enable global pruning
--iterative-steps   Number of pruning iterations (default: 1)
--adaptive          Use adaptive coverage recomputation
--max-batches       Limit test batches for coverage analysis
--round-to          Round channels to multiples

# Other
--batch-size        Batch size (default: 128)
--no-cuda           Disable CUDA
--save-path         Path to save pruned model
```

## 💡 Tips for Best Results

1. **Test Data Selection**: Use representative samples that cover your domain
2. **Coverage Metric**: Start with `normalized_mean`, try `combined` for best results
3. **Iterative Pruning**: Use 3-5 steps for smoother pruning
4. **Global Pruning**: Enable for better cross-layer optimization
5. **Fine-tuning**: Always fine-tune after pruning to recover accuracy

## 🔍 Advanced Usage

### Custom Coverage Metric

```python
class CustomCoverageAnalyzer(CoverageAnalyzer):
    def _compute_custom_metric(self, activations):
        # Your custom coverage computation
        return custom_scores

# Use in pruner
analyzer = CustomCoverageAnalyzer(model)
```

### Layer-Specific Pruning Ratios

```python
pruning_ratio_dict = {
    model.layer1: 0.2,  # 20% pruning
    model.layer2: 0.3,  # 30% pruning
    model.layer3: 0.4,  # 40% pruning
}

pruner = CoveragePruner(
    model=model,
    pruning_ratio_dict=pruning_ratio_dict,
    ...
)
```

### Model Comparison

```python
from utils import compare_models

comparison = compare_models(
    original_model=original_model,
    pruned_model=pruned_model,
    example_inputs=example_inputs,
    test_loader=test_loader
)
```

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{depgraph2023,
  title={DepGraph: Towards Any Structural Pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  booktitle={CVPR},
  year={2023}
}
```

And consider acknowledging this coverage-based pruning method.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is built upon Torch-Pruning, which is licensed under MIT License.

## 🙏 Acknowledgments

- [Torch-Pruning](https://github.com/VainF/Torch-Pruning) by VainF for the excellent structured pruning framework
- PyTorch team for the deep learning framework

## 📞 Contact

For questions or discussions, please open an issue on GitHub.
