# Test Scenario TS6: EfficientNet-B4 on Food101

## Overview
This test scenario evaluates pruning methods on EfficientNet-B4 using the Food101 dataset.

## Test Configuration
- **Model**: EfficientNet-B4 (PyTorch Pretrained on ImageNet)
- **Dataset**: Food101 (~101,000 images, 101 food categories)
- **Pruning Ratio**: 10%
- **Global Pruning**: True
- **Iterative Steps**: 1
- **Fine-tuning Epochs After Pruning**: 10 epochs

## Directory Structure
- **Model Download**: `C:\source\downloaded_models\`
- **Dataset Download**: `C:\source\downloaded_datasets\food101\`
- **Checkpoints**: `C:\source\checkpoints\TS6\`
- **Reports**: Saved in checkpoint directories

## Dataset Setup
Food101 dataset will be automatically downloaded using torchvision.datasets:
- Dataset size: ~5 GB
- Training images: ~75,750
- Test images: ~25,250
- Number of classes: 101 (food categories)
- Image resolution: Variable (will be resized to 380x380 for EfficientNet-B4)

The dataset will be automatically downloaded to: `C:\source\downloaded_datasets\food101\`

## Scripts
1. **TS5_01_prepare_model.py**: Loads pretrained EfficientNet-B4, fine-tunes on Stanford Dogs
2. **TS5_02_coverage_pruning.py**: Applies Neuron Coverage pruning and fine-tunes
3. **TS5_03_wanda_pruning.py**: Applies Wanda pruning and fine-tunes
4. **TS5_04_magnitude_pruning.py**: Applies Magnitude-based pruning (Torch-Pruning baseline) and fine-tunes
5. **TS5_05_taylor_pruning.py**: Applies Taylor gradient-based pruning (Torch-Pruning) and fine-tunes
6. **TS5_compare_results.py**: Compares results from all methods
7. **TS5_run_all.py**: Runs all scripts in sequence

## Checkpoint Naming Convention
- Baseline model: `EfficientNetB4_Food101_FT_best.pth`
- After NC pruning + FT: `EfficientNetB4_Food101_FTAP_NC_epoch{N}.pth`
- After Wanda pruning + FT: `EfficientNetB4_Food101_FTAP_W_epoch{N}.pth`
- After Magnitude pruning + FT: `EfficientNetB4_Food101_FTAP_MAG_epoch{N}.pth`
- After Taylor pruning + FT: `EfficientNetB4_Food101_FTAP_TAY_epoch{N}.pth`

## Results
Results are saved to: `test_scenarios/TS5_EfficientNetB4_StanfordDogs/TS5_Results.json`

## Performance
- **Baseline Accuracy**: ~75-85% (EfficientNet-B4 fine-tuned on Food101)
- **Evaluation Time**: ~2-3 minutes
- **Pruning Time**: ~5-8 minutes
- **Fine-tuning**: ~30-45 minutes per method (10 epochs)
