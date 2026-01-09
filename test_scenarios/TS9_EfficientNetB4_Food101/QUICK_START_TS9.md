# Quick Start - Test Scenario TS9: EfficientNet-B4 on Food101

## Prerequisites
```bash
pip install torch torchvision tqdm tabulate thop scipy
```

## Dataset Setup
Food101 dataset will be automatically downloaded on first run (~5 GB).
Dataset location: `C:\source\downloaded_datasets\food101\`

## Running the Test Scenario

### Option 1: Run All Scripts
```bash
python test_scenarios/TS9_EfficientNetB4_Food101/TS9_run_all.py
```

### Option 2: Run Scripts Individually

#### Step 1: Prepare Model and Fine-tune on Food101
```bash
python test_scenarios/TS5_EfficientNetB4_StanfordDogs/TS5_01_prepare_model.py
```
Loads pretrained EfficientNet-B4 and fine-tunes on Food101 dataset.
**Expected**: ~75-85% accuracy after fine-tuning, ~30-45 minutes

#### Step 2: Apply Neuron Coverage Pruning
```bash
python test_scenarios/TS5_EfficientNetB4_StanfordDogs/TS5_02_coverage_pruning.py
```
Applies Neuron Coverage pruning (10%) and fine-tunes the pruned model (10 epochs).

#### Step 3: Apply Wanda Pruning
```bash
python test_scenarios/TS5_EfficientNetB4_StanfordDogs/TS5_03_wanda_pruning.py
```
Applies Wanda pruning (10%) and fine-tunes the pruned model (10 epochs).

#### Step 4: Apply Magnitude-Based Pruning (Torch-Pruning Baseline)
```bash
python test_scenarios/TS5_EfficientNetB4_StanfordDogs/TS5_04_magnitude_pruning.py
```
Applies Torch-Pruning's Magnitude pruning (10%) and fine-tunes the pruned model (10 epochs).

#### Step 5: Apply Taylor Gradient-Based Pruning
```bash
python test_scenarios/TS5_EfficientNetB4_StanfordDogs/TS5_05_taylor_pruning.py
```
Applies Torch-Pruning's Taylor gradient-based pruning (10%) and fine-tunes the pruned model (10 epochs).

#### Step 6: Compare Results
```bash
python test_scenarios/TS5_EfficientNetB4_StanfordDogs/TS5_compare_results.py
```
Displays comparison of all methods (Coverage, Wanda, Magnitude, and Taylor).

## Expected Outputs
- Fine-tuned EfficientNet-B4 baseline (~75-85% accuracy)
- Pruned model checkpoints (NC, Wanda, Magnitude, Taylor)
- Comparison tables with metrics:
  - **Accuracy** (baseline ~75-85%)
  - **Model Size (MB)** (~10% reduction expected)
  - **FLOPs** (~10% reduction expected)
  - **Average Inference Time (ms)**
- Results JSON file: `TS9_Results.json`

## Notes
- **15 epochs** fine-tuning on baseline model
- **10 epochs** fine-tuning after pruning
- **10% pruning ratio**
- **380x380** input resolution (EfficientNet-B4 standard)
- **101 classes** (food categories)
- **Automatic dataset download** on first run
