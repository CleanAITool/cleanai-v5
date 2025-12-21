# Test Scenario TS3: ResNet-50 on ImageNet

## Overview
This test scenario evaluates pruning methods on ResNet-50 using the ImageNet validation dataset.

## Test Configuration
- **Model**: ResNet-50 (PyTorch Pretrained on ImageNet)
- **Dataset**: ImageNet Validation Set (~50k images, 1000 classes)
- **Pruning Ratio**: 10%
- **Global Pruning**: False
- **Iterative Steps**: 1
- **Fine-tuning Epochs After Pruning**: 10 epochs

## Directory Structure
- **Model Download**: `C:\source\downloaded_models\`
- **Dataset Download**: `C:\source\downloaded_datasets\imagenet\`
- **Checkpoints**: `C:\source\checkpoints\TS3\`
- **Reports**: Saved in checkpoint directories

## Dataset Setup
ImageNet validation set must be downloaded manually:
1. Register at https://image-net.org/download
2. Download ILSVRC2012 validation images (~6.3 GB)
3. Extract to: `C:\source\downloaded_datasets\imagenet\val\`
4. Structure: `imagenet/val/n01440764/ILSVRC2012_val_00000001.JPEG`, etc.

Alternative: Use Kaggle dataset
```bash
kaggle competitions download -c imagenet-object-localization-challenge
```

## Scripts
1. **TS3_01_prepare_model.py**: Loads pretrained ResNet-50, evaluates on ImageNet validation
2. **TS3_02_coverage_pruning.py**: Applies Neuron Coverage pruning and fine-tunes
3. **TS3_03_wanda_pruning.py**: Applies Wanda pruning and fine-tunes
4. **TS3_compare_results.py**: Compares results from all methods
5. **TS3_run_all.py**: Runs all scripts in sequence

## Checkpoint Naming Convention
- Baseline model: `ResNet50_ImageNet_FT_best.pth`
- After NC pruning + FT: `ResNet50_ImageNet_FTAP_NC_epoch{N}.pth`
- After Wanda pruning + FT: `ResNet50_ImageNet_FTAP_W_epoch{N}.pth`

## Results
Results are saved to: `test_scenarios/TS3_ResNet50_ImageNet/TS3_Results.json`

## Performance
- **Baseline Accuracy**: ~76-77% (ResNet-50 on ImageNet)
- **Evaluation Time**: ~1-2 minutes (50k images)
- **Pruning Time**: ~5-10 minutes
- **Fine-tuning**: ~20-40 minutes (10 epochs)
