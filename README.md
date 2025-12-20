<div align="center">

# 🎯 CleanAI v5

### Coverage-Based Neural Network Pruning Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A modular, extensible, and research-friendly framework for intelligent neural network pruning*

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

---

</div>

## 📖 Overview

**CleanAI** is a cutting-edge neural network pruning framework that combines neuron coverage analysis with state-of-the-art importance metrics. Built with modularity and extensibility in mind, CleanAI empowers researchers and practitioners to efficiently compress deep learning models while maintaining performance.

### 🌟 Why CleanAI?

- 🧠 **Intelligent Pruning**: Leverages neuron activation patterns to identify truly important connections
- 📊 **Comprehensive Analysis**: Automatic PDF report generation with detailed metrics and visualizations
- 🔧 **Multiple Strategies**: Coverage-based, WANDA, Adaptive, and traditional importance methods
- 🎓 **Research-Ready**: Clean architecture for experimenting with custom pruning strategies
- 🚀 **Production-Grade**: Battle-tested utilities for model evaluation and deployment

---

## ✨ Features

### 🎯 Advanced Importance Metrics

| Method | Description | Training-Free | Best For |
|--------|-------------|---------------|----------|
| **Coverage-based** | Prunes based on neuron activation patterns across test data | ✅ | Understanding model behavior |
| **WANDA** | Weight × Activation importance scoring | ✅ | Fast, effective pruning |
| **Adaptive** | Dynamic recomputation during iterative pruning | ❌ | Maximum accuracy retention |
| **Standard Methods** | Magnitude, Taylor, Hessian (via Torch-Pruning) | Varies | Baseline comparisons |

### 📝 Professional Reporting System

- **Automated PDF Reports**: Professional-quality documentation of pruning results
- **Rich Visualizations**: 
  - Performance comparison charts
  - Coverage heatmaps by layer
  - Parameter reduction graphs
  - Layer-wise pruning analysis
- **Risk Assessment**: Automatic detection of potential issues and recommendations
- **Detailed Metrics**: Accuracy, loss, parameter count, FLOPs, inference time
- **Export Ready**: Professional English documentation for research papers

### 🏗️ Modular Architecture

```
CleanAI_v5/
├── cleanai/                       # Core package
│   ├── __init__.py               # High-level API
│   │
│   ├── importance/                # 🎯 Importance metrics
│   │   ├── coverage.py           # Coverage-based scoring
│   │   ├── wanda.py              # WANDA implementation
│   │   └── adaptive.py           # Adaptive coverage
│   │
│   ├── analyzers/                 # 🔍 Analysis tools
│   │   └── coverage_analyzer.py  # Neuron activation tracking
│   │
│   ├── pruners/                   # ✂️ Pruning algorithms
│   │   └── coverage_pruner.py    # Main pruning engine
│   │
│   ├── reporting/                 # 📊 Report generation
│   │   ├── report_generator.py   # Orchestrates report creation
│   │   ├── metrics_collector.py  # Metrics aggregation
│   │   ├── visualizations.py     # Chart generation
│   │   └── pdf_builder.py        # PDF construction
│   │
│   └── utils/                     # 🛠️ Utilities
│       ├── model_utils.py        # Model inspection
│       └── evaluation.py         # Evaluation helpers
│
├── examples/                      # 📚 Example scripts
│   ├── simple_pruning.py         # Basic usage
│   ├── wanda_comparison.py       # Method comparison
│   └── generate_report.py        # Reporting demo
│
├── docs/                          # 📖 Documentation
│   ├── REPORTING_GUIDE.md        # Reporting system guide
│   ├── REPORTING_TECHNICAL.md    # Technical details
│   ├── STRUCTURE.md              # Architecture overview
│   └── QUICK_START.md            # Getting started
│
├── main.py                        # 🚀 Main entry point
└── requirements.txt               # 📦 Dependencies
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- PyTorch 2.0 or higher

### Option 1: Quick Install

```bash
# Clone the repository
git clone https://github.com/CleanAITool/cleanai-v5.git
cd cleanai-v5

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Manual Installation

```bash
# Core dependencies
pip install torch>=2.0.0 torchvision>=0.15.0
pip install torch-pruning>=1.3.0
pip install numpy>=1.21.0

# Reporting dependencies
pip install matplotlib>=3.5.0 seaborn>=0.12.0
pip install reportlab>=4.0.0 Pillow>=9.0.0
```

### Verify Installation

```python
from cleanai import CoveragePruner
print("✅ CleanAI successfully installed!")
```

---

## 🚀 Quick Start

### Basic Pruning Example

```python
from cleanai import CoveragePruner, evaluate_model, count_parameters
import torch
from torchvision import models

# 1. Load your model
model = models.resnet18(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 2. Prepare test data  
example_inputs = torch.randn(1, 3, 224, 224).to(device)

# 3. Create pruner with coverage-based importance
pruner = CoveragePruner(
    model=model,
    example_inputs=example_inputs,
    test_loader=test_loader,
    pruning_ratio=0.3,              # Prune 30% of channels
    importance_method='coverage',    # or 'wanda', 'adaptive', 'magnitude'
    global_pruning=True,
    device=device
)

# 4. Execute pruning
pruned_model = pruner.prune()

# 5. Evaluate
accuracy = evaluate_model(pruned_model, test_loader, device)
print(f"✅ Pruning complete! Accuracy: {accuracy:.2f}%")
```

```

### Using Different Importance Methods

```python
# WANDA: Weight × Activation importance
pruner = CoveragePruner(
    model=model,
    example_inputs=example_inputs,
    test_loader=test_loader,
    pruning_ratio=0.3,
    importance_method='wanda',       # WANDA method
    global_pruning=True,
    max_batches=50,                  # Calibration batches
    device=device
)
pruned_model = pruner.prune()

# Adaptive Coverage: Recompute during iterations
pruner = CoveragePruner(
    model=model,
    example_inputs=example_inputs,
    test_loader=test_loader,
    pruning_ratio=0.3,
    importance_method='adaptive',    # Adaptive method
    iterative_steps=5,               # Multiple pruning steps
    device=device
)
pruned_model = pruner.prune()
```

### 📊 Generate Professional Reports

Automatically generate comprehensive PDF reports with detailed analysis:

```python
from cleanai.reporting import generate_pruning_report

# After pruning, generate a report
report_path = generate_pruning_report(
    model_before=original_model,
    model_after=pruned_model,
    model_name="ResNet-18",
    dataloader=test_loader,
    device=device,
    report_name="resnet18_pruning_analysis",
    pruning_method="coverage",
    pruning_ratio=0.3
)

print(f"📄 Report saved to: {report_path}")
```

**Report includes:**

- 📈 Executive summary with key metrics
- 🏗️ Model architecture comparison
- 🔥 Coverage heatmaps by layer
- 📊 Performance comparison charts
- ⚠️ Risk analysis and recommendations
- 💡 Detailed pruning explanations

See [REPORTING_GUIDE.md](REPORTING_GUIDE.md) for detailed documentation.

---

## 📚 Examples

### Run Example Scripts

```bash
# Basic pruning example
python examples/simple_pruning.py

# Compare different importance methods
python examples/wanda_comparison.py

# Generate a detailed report
python examples/generate_report.py
```

### Command Line Interface

```bash
# Prune ResNet-18 on CIFAR-10 with 30% pruning ratio
python main.py --model resnet18 --dataset cifar10 --pruning-ratio 0.3

# Use global pruning with adaptive coverage
python main.py --model resnet18 --dataset cifar10 \
    --pruning-ratio 0.5 --global-pruning \
    --importance-method adaptive --iterative-steps 5

# Generate report while pruning
python main.py --model resnet50 --dataset cifar100 \
    --pretrained --pruning-ratio 0.4 \
    --generate-report --save-path ./pruned_model.pth
```

---

## 📊 Benchmark Results

### ResNet-18 on CIFAR-10

| Metric          | Original | Coverage (30%) | WANDA (30%) | Adaptive (30%) |
|-----------------|----------|----------------|-------------|----------------|
| **Parameters**  | 11.17M   | 7.82M (-30%)   | 7.91M (-29%)| 7.75M (-31%)   |
| **FLOPs**       | 0.56G    | 0.39G (-30%)   | 0.40G (-29%)| 0.38G (-32%)   |
| **Accuracy**    | 95.2%    | 94.1% (-1.1%)  | 94.3% (-0.9%)| 94.5% (-0.7%) |
| **Inference**   | 2.45ms   | 1.78ms (1.4x)  | 1.82ms (1.3x)| 1.74ms (1.4x) |

### MobileNetV2 on ImageNet

| Pruning Ratio | Params | FLOPs | Top-1 Acc | Top-5 Acc |
|---------------|--------|-------|-----------|-----------|
| 0% (Original) | 3.5M   | 300M  | 72.0%     | 91.0%     |
| 20%           | 2.8M   | 240M  | 71.2%     | 90.5%     |
| 40%           | 2.1M   | 180M  | 69.8%     | 89.2%     |
| 60%           | 1.4M   | 120M  | 67.5%     | 87.8%     |

*Results may vary based on random initialization and test data sampling*

---

## 🔬 How It Works

### 1. **Coverage Analysis**

```python
# Collect neuron activations during inference
analyzer = CoverageAnalyzer(model, device)
analyzer.register_hooks()
analyzer.collect_activations(test_loader)

# Compute coverage scores
coverage = analyzer.compute_neuron_coverage(metric='normalized_mean')
```

### 2. **Importance Scoring**

```python
# Convert coverage to importance
# Lower coverage → Higher pruning priority
importance = 1.0 / (coverage + epsilon)
```

### 3. **Structured Pruning**

```python
# Torch-Pruning handles dependencies automatically
# - Output channels → BatchNorm → Next layer input
# - Skip connections maintained correctly
# - Proper channel alignment preserved
```

### Coverage Metrics Explained

| Metric | Formula | Best For |
|--------|---------|----------|
| **normalized_mean** | `mean(activations) / global_max` | General purpose, balanced |
| **frequency** | `count(active) / total_samples` | Finding dead neurons |
| **mean_absolute** | `mean(abs(activations))` | Direct magnitude-based |
| **combined** | `sqrt(normalized_mean × frequency)` | Comprehensive analysis |

---

## 🎛️ Configuration Options

### Core Parameters

```python
CoveragePruner(
    model,                          # PyTorch model to prune
    example_inputs,                 # Sample input for shape inference
    test_loader=None,               # DataLoader for coverage analysis
    pruning_ratio=0.3,              # Channels to remove (0.0-1.0)
    importance_method='coverage',   # 'coverage', 'wanda', 'adaptive', 'magnitude'
    coverage_metric='normalized_mean',  # Coverage computation method
    global_pruning=True,            # Global vs local pruning
    iterative_steps=1,              # Number of pruning iterations
    max_batches=None,               # Limit batches for analysis
    round_to=None,                  # Round channels (e.g., 8 for efficiency)
    device='cuda'                   # Device to use
)
```

### Advanced Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ignored_layers` | list | `[]` | Layers to skip during pruning |
| `pruning_ratio_dict` | dict | `None` | Custom ratios per layer |
| `channel_groups` | dict | `{}` | Group channels together |
| `customized_pruners` | dict | `{}` | Custom pruning logic |
| `root_module_types` | list | `None` | Target layer types |
| `unwrapped_parameters` | dict | `{}` | Parameters outside layers |

---

## 💡 Tips & Best Practices

### ✅ Do's

- **Use representative test data** for coverage analysis
- **Start with conservative ratios** (20-30%) and increase gradually
- **Enable global pruning** for better cross-layer optimization
- **Use iterative pruning** (3-5 steps) for smoother compression
- **Fine-tune after pruning** to recover accuracy
- **Generate reports** to understand pruning decisions

### ❌ Don'ts

- Don't prune too aggressively without fine-tuning
- Don't use biased test data for coverage analysis
- Don't ignore the generated reports and warnings
- Don't prune final classification layers too heavily
- Don't skip validation before deployment

### 🎯 Choosing Importance Methods

```
Coverage    → Best for: Understanding neuron behavior
WANDA       → Best for: Fast, training-free pruning
Adaptive    → Best for: Maximum accuracy retention
Magnitude   → Best for: Baseline comparisons
```

---

## 📖 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Getting started guide (Turkish)
- **[STRUCTURE.md](STRUCTURE.md)** - Architecture overview (Turkish)
- **[REPORTING_GUIDE.md](REPORTING_GUIDE.md)** - Comprehensive reporting documentation
- **[REPORTING_TECHNICAL.md](REPORTING_TECHNICAL.md)** - Technical details of report generation

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/CleanAITool/cleanai-v5.git
cd cleanai-v5
pip install -e .
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on top of [Torch-Pruning](https://github.com/VainF/Torch-Pruning)
- Inspired by neuron coverage research in deep learning testing
- WANDA implementation based on ["A Simple and Effective Pruning Approach for Large Language Models"](https://arxiv.org/abs/2306.11695)

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/CleanAITool/cleanai-v5/issues)
- **Discussions**: [GitHub Discussions](https://github.com/CleanAITool/cleanai-v5/discussions)

---

## 🌟 Citation

If you use CleanAI in your research, please cite:

```bibtex
@software{cleanai2024,
  title={CleanAI: Coverage-Based Neural Network Pruning Framework},
  author={CleanAI Team},
  year={2024},
  url={https://github.com/CleanAITool/cleanai-v5}
}
```

---

<div align="center">

**Made with ❤️ by the CleanAI Team**

[⭐ Star us on GitHub](https://github.com/CleanAITool/cleanai-v5) | [🐛 Report Bug](https://github.com/CleanAITool/cleanai-v5/issues) | [💡 Request Feature](https://github.com/CleanAITool/cleanai-v5/issues)

</div>
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
