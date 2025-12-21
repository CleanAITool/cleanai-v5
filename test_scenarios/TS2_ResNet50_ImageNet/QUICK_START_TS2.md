# Quick Start - Test Scenario TS2: ResNet-50 on ImageNet

## Prerequisites
```bash
pip install torch torchvision tqdm tabulate thop
```

## Dataset Setup (Required First!)
Download ImageNet validation set:
1. Go to https://image-net.org/download and register
2. Download ILSVRC2012 validation images (~6.3 GB)
3. Extract to: `C:\source\downloaded_datasets\imagenet\val\`

Or use Kaggle:
```bash
kaggle competitions download -c imagenet-object-localization-challenge
# Extract val folder to C:\source\downloaded_datasets\imagenet\
```

## Running the Test Scenario

### Option 1: Run All Scripts
```bash
python test_scenarios/TS2_ResNet50_ImageNet/TS2_run_all.py
```

### Option 2: Run Scripts Individually

#### Step 1: Prepare Model and Evaluate Baseline
```bash
python test_scenarios/TS2_ResNet50_ImageNet/TS2_01_prepare_model.py
```
Loads pretrained ResNet-50 and evaluates on ImageNet validation set.
**Expected**: ~76-77% Top-1 accuracy, ~1-2 minutes

#### Step 2: Apply Neuron Coverage Pruning
```bash
python test_scenarios/TS2_ResNet50_ImageNet/TS2_02_coverage_pruning.py
```
Applies Neuron Coverage pruning and fine-tunes the pruned model.

#### Step 3: Apply Wanda Pruning
```bash
python test_scenarios/TS2_ResNet50_ImageNet/TS2_03_wanda_pruning.py
```
Applies Wanda pruning and fine-tunes the pruned model.

#### Step 4: Compare Results
```bash
python test_scenarios/TS2_ResNet50_ImageNet/TS2_compare_results.py
```
Displays comparison of all methods.

##Pretrained ResNet-50 baseline evaluation
- Pruned model checkpoints (NC and Wanda)
- Comparison tables with metrics:
  - **Top-1 Accuracy** (baseline ~76-77%)
  - **Model Size (MB)** (~20% reduction expected)
  - **FLOPs** (~20% reduction expected)
  - **Average Inference Time (ms)**
- Results JSON file: `TS2_Results.json`

## Notes
- **No fine-tuning** on baseline (already trained on ImageNet)
- **5 epochs** fine-tuning after pruning
- Native **224x224** resolution
- Uses **validation set only** (~50k images)
- Results JSON file: `TS2_Results.json`
