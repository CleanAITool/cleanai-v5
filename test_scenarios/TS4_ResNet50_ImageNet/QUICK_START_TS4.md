# Quick Start - Test Scenario TS4: ResNet-50 on ImageNet

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
python test_scenarios/TS4_ResNet50_ImageNet/TS4_run_all.py
```

### Option 2: Run Scripts Individually

#### Step 1: Prepare Model and Evaluate Baseline
```bash
python test_scenarios/TS4_ResNet50_ImageNet/TS4_01_prepare_model.py
```
Loads pretrained ResNet-50 and evaluates on ImageNet validation set.
**Expected**: ~76-77% Top-1 accuracy, ~1-2 minutes

#### Step 2: Apply Neuron Coverage Pruning
```bash
python test_scenarios/TS4_ResNet50_ImageNet/TS4_02_coverage_pruning.py
```
Applies Neuron Coverage pruning (10%) and fine-tunes the pruned model (10 epochs).

#### Step 3: Apply Wanda Pruning
```bash
python test_scenarios/TS4_ResNet50_ImageNet/TS4_03_wanda_pruning.py
```
Applies Wanda pruning (10%) and fine-tunes the pruned model (10 epochs).

#### Step 4: Apply Magnitude-Based Pruning (Torch-Pruning Baseline)
```bash
python test_scenarios/TS4_ResNet50_ImageNet/TS4_04_magnitude_pruning.py
```
Applies Torch-Pruning's Magnitude pruning (10%) and fine-tunes the pruned model (10 epochs).

#### Step 5: Apply Taylor Gradient-Based Pruning
```bash
python test_scenarios/TS4_ResNet50_ImageNet/TS4_05_taylor_pruning.py
```
Applies Torch-Pruning's Taylor gradient-based pruning (10%) and fine-tunes the pruned model (10 epochs).

#### Step 6: Compare Results
```bash
python test_scenarios/TS4_ResNet50_ImageNet/TS4_compare_results.py
```
Displays comparison of all methods (Coverage, Wanda, Magnitude, and Taylor).

## Expected Outputs
- Pretrained ResNet-50 baseline evaluation
- Pruned model checkpoints (NC and Wanda)
- Comparison tables with metrics:
  - **Top-1 Accuracy** (baseline ~76-77%)
  - **Model Size (MB)** (~10% reduction expected)
  - **FLOPs** (~10% reduction expected)
  - **Average Inference Time (ms)**
- Results JSON file: `TS4_Results.json`

## Notes
- **No fine-tuning** on baseline (already trained on ImageNet)
- **10 epochs** fine-tuning after pruning
- **10% pruning ratio** (less aggressive than TS2)
- Native **224x224** resolution
- Uses **validation set only** (~50k images)

