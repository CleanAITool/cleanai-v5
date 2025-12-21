####################################################################################################
# TEST SCENARIO TS1 - MASTER SCRIPT
####################################################################################################

This will run all test scenario scripts in sequence:
  1. TS1_01_prepare_model.py    - Model preparation & fine-tuning
  2. TS1_02_coverage_pruning.py - Neuron Coverage pruning
  3. TS1_03_wanda_pruning.py    - WANDA pruning

Continue? [y/N]: y

====================================================================================================
RUNNING: TS1_01_prepare_model.py
Description: Download pretrained model, adapt to CIFAR-10, and fine-tune
====================================================================================================


################################################################################
# TEST SCENARIO TS1 - SCRIPT 1: MODEL PREPARATION & FINE-TUNING
################################################################################

Model: ResNet18
Dataset: CIFAR10
Device: cuda

Directories:
  Models: C:\source\downloaded_models
  Datasets: C:\source\downloaded_datasets
  Checkpoints: C:\source\checkpoints\TS1

============================================================
CHECKPOINT FOUND - Loading Existing Model
============================================================
✓ Found: ResNet18_CIFAR10_FT_final.pth
✓ Skipping fine-tuning (already completed)

============================================================
Loading CIFAR-10 Dataset
============================================================
Files already downloaded and verified
Files already downloaded and verified
✓ Train samples: 50,000
✓ Test samples: 10,000
✓ Number of classes: 10

============================================================
Loading Pretrained ResNet-18
============================================================
C:\Users\matisse\anaconda3\Lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\matisse\anaconda3\Lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
✓ Model adapted for CIFAR-10
✓ Input: 32x32x3
✓ Output: 10 classes
C:\source\repos\cleanai-v5\test_scenarios\TS1_01_prepare_model.py:366: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(final_checkpoint_path, map_location=DEVICE)
C:\source\repos\cleanai-v5\test_scenarios\TS1_01_prepare_model.py:384: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  initial_checkpoint = torch.load(initial_checkpoint_path, map_location=DEVICE)
✓ Model accuracy: 93.90%
✓ Parameters: 11,173,962

================================================================================
FINE-TUNING RESULTS - COMPARISON TABLE
================================================================================

Metric                         Before Fine-Tuning        After Fine-Tuning
--------------------------------------------------------------------------------
Accuracy (%)                                      11.10                    93.90
Loss                                             0.0000                   0.0000
Correct Predictions                                   0                        0
Total Samples                                    10,000                   10,000
--------------------------------------------------------------------------------
Total Parameters                             11,173,962               11,173,962
Trainable Parameters                         11,173,962               11,173,962
--------------------------------------------------------------------------------

✓ Accuracy Improvement: +82.80%
✓ Final Test Accuracy: 93.90%
================================================================================


✓ Total execution time: 0.02 minutes
✓ Model ready for pruning experiments!

################################################################################
# SCRIPT 1 COMPLETED SUCCESSFULLY
################################################################################


✓ TS1_01_prepare_model.py completed successfully in 0.08 minutes

====================================================================================================
RUNNING: TS1_02_coverage_pruning.py
Description: Apply Neuron Coverage pruning and fine-tune
====================================================================================================


####################################################################################################
# TEST SCENARIO TS1 - SCRIPT 2: NEURON COVERAGE PRUNING + FINE-TUNING
####################################################################################################

Test Scenario: TS1_Coverage_ResNet18_CIFAR10
Method: Coverage
Model: ResNet18
Dataset: CIFAR10
Pruning Ratio: 20%
Device: cuda

============================================================
Loading CIFAR-10 Dataset
============================================================
Files already downloaded and verified
Files already downloaded and verified
✓ Dataset loaded

============================================================
Loading Fine-Tuned Model
============================================================
C:\Users\matisse\anaconda3\Lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\matisse\anaconda3\Lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
C:\source\repos\cleanai-v5\test_scenarios\TS1_02_coverage_pruning.py:118: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
✓ Model loaded from: ResNet18_CIFAR10_FT_final.pth
✓ Checkpoint accuracy: 93.90%

============================================================
Measuring Original Model Metrics
============================================================
✓ Accuracy: 93.90%
✓ Parameters: 11,173,962
✓ Size: 42.70 MB
✓ Inference Time: 0.07 ms/sample

============================================================
No Checkpoint Found - Starting Pruning Process
============================================================

============================================================
Applying Neuron Coverage Pruning
============================================================
Using static coverage importance (computes once)

Initializing Torch-Pruning with:
  Importance method: coverage
  Pruning ratio: 20.00%
  Coverage metric: normalized_mean
  Global pruning: False
  Iterative steps: 1
  Device: cuda

############################################################
Starting Coverage-Based Pruning
############################################################
Initial parameters: 11,173,962

============================================================
Pruning Step 1/1
============================================================

============================================================
Computing Neuron Coverage on Test Data
============================================================
Registered hooks on 21 layers
Collecting activations from test data...
  Processed 10 batches
  Processed 20 batches
Activation collection complete

Computing neuron coverage using metric: normalized_mean
  conv1: coverage shape torch.Size([64]), min=-0.617364, max=1.000000, mean=0.017954
  layer1.0.conv1: coverage shape torch.Size([64]), min=-1.736153, max=1.000000, mean=-0.450367
  layer1.0.conv2: coverage shape torch.Size([64]), min=-1.013410, max=1.000000, mean=-0.130221
  layer1.1.conv1: coverage shape torch.Size([64]), min=-1.718650, max=1.000000, mean=-0.321154
  layer1.1.conv2: coverage shape torch.Size([64]), min=-1.512626, max=1.000000, mean=-0.098711
  layer2.0.conv1: coverage shape torch.Size([128]), min=-1.477176, max=1.000000, mean=-0.184668
  layer2.0.conv2: coverage shape torch.Size([128]), min=-1.725067, max=1.000000, mean=-0.209564
  layer2.0.downsample.0: coverage shape torch.Size([128]), min=-0.623347, max=1.000000, mean=-0.002388
  layer2.1.conv1: coverage shape torch.Size([128]), min=-0.818819, max=1.000000, mean=-0.161714
  layer2.1.conv2: coverage shape torch.Size([128]), min=-0.767161, max=1.000000, mean=-0.094840
  layer3.0.conv1: coverage shape torch.Size([256]), min=-1.571142, max=1.000000, mean=-0.388569
  layer3.0.conv2: coverage shape torch.Size([256]), min=-2.989337, max=1.000000, mean=-0.406000
  layer3.0.downsample.0: coverage shape torch.Size([256]), min=-1.648756, max=1.000000, mean=-0.139548
  layer3.1.conv1: coverage shape torch.Size([256]), min=-1.067401, max=1.000000, mean=-0.133332
  layer3.1.conv2: coverage shape torch.Size([256]), min=-0.633364, max=1.000000, mean=-0.119257
  layer4.0.conv1: coverage shape torch.Size([512]), min=-1.407906, max=1.000000, mean=-0.122900
  layer4.0.conv2: coverage shape torch.Size([512]), min=-1.283595, max=1.000000, mean=-0.000742
  layer4.0.downsample.0: coverage shape torch.Size([512]), min=-1.266103, max=1.000000, mean=-0.044656
  layer4.1.conv1: coverage shape torch.Size([512]), min=-9.261184, max=1.000000, mean=-0.563029
  layer4.1.conv2: coverage shape torch.Size([512]), min=-0.912434, max=1.000000, mean=0.016541
  fc: coverage shape torch.Size([10]), min=-2.898468, max=-1.776384, mean=-2.337661

------------------------------------------------------------
Coverage Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Coverage - Min: -0.617364, Max: 1.000000, Mean: 0.017954
  Zero coverage neurons: 0

layer1.0.conv1:
  Channels: 64
  Coverage - Min: -1.736153, Max: 1.000000, Mean: -0.450367
  Zero coverage neurons: 0

layer1.0.conv2:
  Channels: 64
  Coverage - Min: -1.013410, Max: 1.000000, Mean: -0.130221
  Zero coverage neurons: 0

layer1.1.conv1:
  Channels: 64
  Coverage - Min: -1.718650, Max: 1.000000, Mean: -0.321154
  Zero coverage neurons: 0

layer1.1.conv2:
  Channels: 64
  Coverage - Min: -1.512626, Max: 1.000000, Mean: -0.098711
  Zero coverage neurons: 0

layer2.0.conv1:
  Channels: 128
  Coverage - Min: -1.477176, Max: 1.000000, Mean: -0.184668
  Zero coverage neurons: 0

layer2.0.conv2:
  Channels: 128
  Coverage - Min: -1.725067, Max: 1.000000, Mean: -0.209564
  Zero coverage neurons: 0

layer2.0.downsample.0:
  Channels: 128
  Coverage - Min: -0.623347, Max: 1.000000, Mean: -0.002388
  Zero coverage neurons: 0

layer2.1.conv1:
  Channels: 128
  Coverage - Min: -0.818819, Max: 1.000000, Mean: -0.161714
  Zero coverage neurons: 0

layer2.1.conv2:
  Channels: 128
  Coverage - Min: -0.767161, Max: 1.000000, Mean: -0.094840
  Zero coverage neurons: 0

layer3.0.conv1:
  Channels: 256
  Coverage - Min: -1.571142, Max: 1.000000, Mean: -0.388569
  Zero coverage neurons: 0

layer3.0.conv2:
  Channels: 256
  Coverage - Min: -2.989337, Max: 1.000000, Mean: -0.406000
  Zero coverage neurons: 0

layer3.0.downsample.0:
  Channels: 256
  Coverage - Min: -1.648756, Max: 1.000000, Mean: -0.139548
  Zero coverage neurons: 0

layer3.1.conv1:
  Channels: 256
  Coverage - Min: -1.067401, Max: 1.000000, Mean: -0.133332
  Zero coverage neurons: 0

layer3.1.conv2:
  Channels: 256
  Coverage - Min: -0.633364, Max: 1.000000, Mean: -0.119257
  Zero coverage neurons: 0

layer4.0.conv1:
  Channels: 512
  Coverage - Min: -1.407906, Max: 1.000000, Mean: -0.122900
  Zero coverage neurons: 0

layer4.0.conv2:
  Channels: 512
  Coverage - Min: -1.283595, Max: 1.000000, Mean: -0.000742
  Zero coverage neurons: 0

layer4.0.downsample.0:
  Channels: 512
  Coverage - Min: -1.266103, Max: 1.000000, Mean: -0.044656
  Zero coverage neurons: 0

layer4.1.conv1:
  Channels: 512
  Coverage - Min: -9.261184, Max: 1.000000, Mean: -0.563029
  Zero coverage neurons: 0

layer4.1.conv2:
  Channels: 512
  Coverage - Min: -0.912434, Max: 1.000000, Mean: 0.016541
  Zero coverage neurons: 0

fc:
  Channels: 10
  Coverage - Min: -2.898468, Max: -1.776384, Mean: -2.337661
  Zero coverage neurons: 0
============================================================


Pruning Results:
  Parameters before: 11,173,962
  Parameters after:  7,122,360
  Parameters removed: 4,051,602 (36.26%)

############################################################
Pruning Complete
############################################################
Initial parameters:  11,173,962
Final parameters:    7,122,360
Total removed:       4,051,602 (36.26%)
Target pruning ratio: 20.00%

============================================================
Measuring Pruned Model Metrics
============================================================
✓ Accuracy: 46.96%
✓ Parameters: 7,122,360
✓ Size: 27.24 MB
✓ Inference Time: 0.08 ms/sample

✓ Pruned model saved: ResNet18_CIFAR10_pruned_NC.pth

============================================================
Fine-Tuning Pruned Model
============================================================

Epoch 1/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.4361 Acc: 87.11%
  Batch [100/391] Loss: 0.3851 Acc: 88.38%
  Batch [150/391] Loss: 0.2923 Acc: 88.44%
  Batch [200/391] Loss: 0.3307 Acc: 88.27%
  Batch [250/391] Loss: 0.4177 Acc: 88.35%
  Batch [300/391] Loss: 0.3345 Acc: 88.39%
  Batch [350/391] Loss: 0.3598 Acc: 88.42%

  Train Loss: 0.3450 | Train Acc: 88.48%
  Test Loss:  0.4187 | Test Acc:  86.42%

Epoch 2/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.3136 Acc: 90.12%
  Batch [100/391] Loss: 0.3350 Acc: 89.91%
  Batch [150/391] Loss: 0.3635 Acc: 89.80%
  Batch [200/391] Loss: 0.2812 Acc: 89.82%
  Batch [250/391] Loss: 0.2824 Acc: 89.67%
  Batch [300/391] Loss: 0.3179 Acc: 89.49%
  Batch [350/391] Loss: 0.3628 Acc: 89.45%

  Train Loss: 0.3125 | Train Acc: 89.45%
  Test Loss:  0.4091 | Test Acc:  86.72%

Epoch 3/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.2025 Acc: 90.81%
  Batch [100/391] Loss: 0.2522 Acc: 90.23%
  Batch [150/391] Loss: 0.3153 Acc: 90.45%
  Batch [200/391] Loss: 0.2637 Acc: 90.30%
  Batch [250/391] Loss: 0.2721 Acc: 90.46%
  Batch [300/391] Loss: 0.3829 Acc: 90.27%
  Batch [350/391] Loss: 0.3341 Acc: 90.25%

  Train Loss: 0.2888 | Train Acc: 90.20%
  Test Loss:  0.3638 | Test Acc:  88.01%

Epoch 4/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.2957 Acc: 91.77%
  Batch [100/391] Loss: 0.2366 Acc: 91.26%
  Batch [150/391] Loss: 0.1889 Acc: 91.61%
  Batch [200/391] Loss: 0.2263 Acc: 91.59%
  Batch [250/391] Loss: 0.3065 Acc: 91.52%
  Batch [300/391] Loss: 0.2176 Acc: 91.38%
  Batch [350/391] Loss: 0.1475 Acc: 91.33%

  Train Loss: 0.2566 | Train Acc: 91.40%
  Test Loss:  0.3604 | Test Acc:  88.26%

Epoch 5/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.1679 Acc: 92.81%
  Batch [100/391] Loss: 0.1673 Acc: 92.64%
  Batch [150/391] Loss: 0.2793 Acc: 92.65%
  Batch [200/391] Loss: 0.2138 Acc: 92.68%
  Batch [250/391] Loss: 0.2010 Acc: 92.69%
  Batch [300/391] Loss: 0.2401 Acc: 92.65%
  Batch [350/391] Loss: 0.1989 Acc: 92.70%

  Train Loss: 0.2195 | Train Acc: 92.61%
  Test Loss:  0.3381 | Test Acc:  89.09%
  ✓ Checkpoint saved: ResNet18_CIFAR10_FTAP_NC_epoch5.pth

Epoch 6/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.1755 Acc: 93.44%
  Batch [100/391] Loss: 0.1016 Acc: 93.80%
  Batch [150/391] Loss: 0.2429 Acc: 93.88%
  Batch [200/391] Loss: 0.1509 Acc: 93.82%
  Batch [250/391] Loss: 0.2941 Acc: 93.84%
  Batch [300/391] Loss: 0.0915 Acc: 93.81%
  Batch [350/391] Loss: 0.2169 Acc: 93.74%

  Train Loss: 0.1882 | Train Acc: 93.65%
  Test Loss:  0.2772 | Test Acc:  91.30%

Epoch 7/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.1223 Acc: 95.50%
  Batch [100/391] Loss: 0.1623 Acc: 95.39%
  Batch [150/391] Loss: 0.1481 Acc: 95.27%
  Batch [200/391] Loss: 0.1599 Acc: 95.29%
  Batch [250/391] Loss: 0.1056 Acc: 95.31%
  Batch [300/391] Loss: 0.1229 Acc: 95.27%
  Batch [350/391] Loss: 0.1394 Acc: 95.18%

  Train Loss: 0.1454 | Train Acc: 95.14%
  Test Loss:  0.2540 | Test Acc:  91.90%

Epoch 8/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.2269 Acc: 96.31%
  Batch [100/391] Loss: 0.1621 Acc: 96.27%
  Batch [150/391] Loss: 0.0732 Acc: 96.51%
  Batch [200/391] Loss: 0.1374 Acc: 96.55%
  Batch [250/391] Loss: 0.0970 Acc: 96.50%
  Batch [300/391] Loss: 0.1152 Acc: 96.46%
  Batch [350/391] Loss: 0.1036 Acc: 96.42%

  Train Loss: 0.1070 | Train Acc: 96.41%
  Test Loss:  0.2413 | Test Acc:  92.76%

Epoch 9/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.0692 Acc: 96.92%
  Batch [100/391] Loss: 0.0847 Acc: 97.16%
  Batch [150/391] Loss: 0.0876 Acc: 97.26%
  Batch [200/391] Loss: 0.0685 Acc: 97.34%
  Batch [250/391] Loss: 0.1442 Acc: 97.31%
  Batch [300/391] Loss: 0.0582 Acc: 97.31%
  Batch [350/391] Loss: 0.0711 Acc: 97.34%

  Train Loss: 0.0824 | Train Acc: 97.30%
  Test Loss:  0.2266 | Test Acc:  93.37%

Epoch 10/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.1695 Acc: 97.80%
  Batch [100/391] Loss: 0.0705 Acc: 97.83%
  Batch [150/391] Loss: 0.0226 Acc: 97.85%
  Batch [200/391] Loss: 0.0585 Acc: 97.91%
  Batch [250/391] Loss: 0.0795 Acc: 97.86%
  Batch [300/391] Loss: 0.0372 Acc: 97.89%
  Batch [350/391] Loss: 0.0565 Acc: 97.91%

  Train Loss: 0.0659 | Train Acc: 97.89%
  Test Loss:  0.2188 | Test Acc:  93.63%
  ✓ Checkpoint saved: ResNet18_CIFAR10_FTAP_NC_epoch10.pth

============================================================
Measuring Final Model Metrics (After Fine-Tuning)
============================================================
✓ Accuracy: 93.63%
✓ Parameters: 7,122,360
✓ Size: 27.24 MB
✓ Inference Time: 0.09 ms/sample

✓ Final model saved: ResNet18_CIFAR10_FTAP_NC_final.pth

============================================================
Generating PDF Report
============================================================
📊 Collecting metrics...
  - Before pruning metrics...

============================================================
Collecting Pre-Pruning Metrics
============================================================
Parameters: 7,122,360
Model Size: 27.17 MB
GFLOPs: 0.71

Evaluating accuracy...
Accuracy: 93.63%

Measuring inference time...
Inference Time: 5.48 ± 0.82 ms
============================================================

  - After pruning metrics...

============================================================
Collecting Post-Pruning Metrics
============================================================
Parameters: 7,122,360
Model Size: 27.17 MB
GFLOPs: 0.71

Evaluating accuracy...
Accuracy: 93.63%

Measuring inference time...
Inference Time: 3.61 ± 1.47 ms

Reductions:
  Parameters: -0.00%
  Model Size: -0.00%
  FLOPs: -0.00%
  Accuracy Drop: 0.00%
  Speedup: 1.52x
============================================================

  - Pruning decisions...
  - Risk analysis...
✅ Metrics collection complete!

📝 Generating report: TS1_Coverage_ResNet18_CIFAR10.pdf

  ✓ Creating cover page...
  ✓ Adding executive summary...
  ✓ Adding model information...
  ✓ Adding pruning decisions...
  ✓ Adding post-pruning structure...
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.

✗ TS1_02_coverage_pruning.py failed after 3.86 minutes
Error code: 3

====================================================================================================
RUNNING: TS1_03_wanda_pruning.py
Description: Apply WANDA pruning and fine-tune
====================================================================================================


####################################################################################################
# TEST SCENARIO TS1 - SCRIPT 3: WANDA PRUNING + FINE-TUNING
####################################################################################################

Test Scenario: TS1_Wanda_ResNet18_CIFAR10
Method: Wanda (Weight AND Activation)
Model: ResNet18
Dataset: CIFAR10
Pruning Ratio: 20%
Device: cuda

============================================================
Loading CIFAR-10 Dataset
============================================================
Files already downloaded and verified
Files already downloaded and verified
✓ Dataset loaded

============================================================
Loading Fine-Tuned Model
============================================================
C:\Users\matisse\anaconda3\Lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\matisse\anaconda3\Lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
C:\source\repos\cleanai-v5\test_scenarios\TS1_03_wanda_pruning.py:117: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
✓ Model loaded from: ResNet18_CIFAR10_FT_final.pth
✓ Checkpoint accuracy: 93.90%

============================================================
Measuring Original Model Metrics
============================================================
✓ Accuracy: 93.90%
✓ Parameters: 11,173,962
✓ Size: 42.70 MB
✓ Inference Time: 0.07 ms/sample

============================================================
No Checkpoint Found - Starting Pruning Process
============================================================

============================================================
Applying WANDA Pruning (Weight AND Activation)
============================================================
WANDA combines weight magnitude with activation importance
Using 20 calibration batches from test set
Using WANDA importance (Weight × Activation)

Initializing Torch-Pruning with:
  Importance method: wanda
  Pruning ratio: 20.00%
  Coverage metric: normalized_mean
  Global pruning: False
  Iterative steps: 1
  Device: cuda

############################################################
Starting Coverage-Based Pruning
############################################################
Initial parameters: 11,173,962

============================================================
Pruning Step 1/1
============================================================

============================================================
WANDA: Computing Weight × Activation Importance
============================================================
Registered hooks on 21 layers
Collecting activations from test data...
  Processed 10 batches
  Processed 20 batches
Activation collection complete

Computing neuron coverage using metric: mean_absolute
  conv1: coverage shape torch.Size([64]), min=0.000000, max=0.024225, mean=0.003606
  layer1.0.conv1: coverage shape torch.Size([64]), min=0.000401, max=1.208783, mean=0.422345
  layer1.0.conv2: coverage shape torch.Size([64]), min=0.012695, max=0.518709, mean=0.180777
  layer1.1.conv1: coverage shape torch.Size([64]), min=0.002991, max=1.262466, mean=0.376610
  layer1.1.conv2: coverage shape torch.Size([64]), min=0.000658, max=0.445837, mean=0.119658
  layer2.0.conv1: coverage shape torch.Size([128]), min=0.001836, max=1.046051, mean=0.258808
  layer2.0.conv2: coverage shape torch.Size([128]), min=0.000307, max=0.659228, mean=0.125761
  layer2.0.downsample.0: coverage shape torch.Size([128]), min=0.000246, max=0.554303, mean=0.085276
  layer2.1.conv1: coverage shape torch.Size([128]), min=0.002282, max=1.160483, mean=0.257742
  layer2.1.conv2: coverage shape torch.Size([128]), min=0.000285, max=0.365137, mean=0.054716
  layer3.0.conv1: coverage shape torch.Size([256]), min=0.001148, max=0.677715, mean=0.209004
  layer3.0.conv2: coverage shape torch.Size([256]), min=0.000001, max=0.522699, mean=0.084740
  layer3.0.downsample.0: coverage shape torch.Size([256]), min=0.000001, max=0.169175, mean=0.029508
  layer3.1.conv1: coverage shape torch.Size([256]), min=0.000000, max=0.434856, mean=0.083851
  layer3.1.conv2: coverage shape torch.Size([256]), min=0.000000, max=0.194815, mean=0.028114
  layer4.0.conv1: coverage shape torch.Size([512]), min=0.000000, max=0.214668, mean=0.024424
  layer4.0.conv2: coverage shape torch.Size([512]), min=0.000000, max=0.276125, mean=0.017481
  layer4.0.downsample.0: coverage shape torch.Size([512]), min=0.000000, max=0.100525, mean=0.008787
  layer4.1.conv1: coverage shape torch.Size([512]), min=0.000000, max=0.843143, mean=0.054520
  layer4.1.conv2: coverage shape torch.Size([512]), min=0.000027, max=0.098345, mean=0.020598
  fc: coverage shape torch.Size([10]), min=1.776384, max=2.898468, mean=2.337661

------------------------------------------------------------
Activation Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Activation - Min: 0.000000, Max: 0.024225, Mean: 0.003606

layer1.0.conv1:
  Channels: 64
  Activation - Min: 0.000401, Max: 1.208783, Mean: 0.422345

layer1.0.conv2:
  Channels: 64
  Activation - Min: 0.012695, Max: 0.518709, Mean: 0.180777

layer1.1.conv1:
  Channels: 64
  Activation - Min: 0.002991, Max: 1.262466, Mean: 0.376610

layer1.1.conv2:
  Channels: 64
  Activation - Min: 0.000658, Max: 0.445837, Mean: 0.119658

layer2.0.conv1:
  Channels: 128
  Activation - Min: 0.001836, Max: 1.046051, Mean: 0.258808

layer2.0.conv2:
  Channels: 128
  Activation - Min: 0.000307, Max: 0.659228, Mean: 0.125761

layer2.0.downsample.0:
  Channels: 128
  Activation - Min: 0.000246, Max: 0.554303, Mean: 0.085276

layer2.1.conv1:
  Channels: 128
  Activation - Min: 0.002282, Max: 1.160483, Mean: 0.257742

layer2.1.conv2:
  Channels: 128
  Activation - Min: 0.000285, Max: 0.365137, Mean: 0.054716

layer3.0.conv1:
  Channels: 256
  Activation - Min: 0.001148, Max: 0.677715, Mean: 0.209004

layer3.0.conv2:
  Channels: 256
  Activation - Min: 0.000001, Max: 0.522699, Mean: 0.084740

layer3.0.downsample.0:
  Channels: 256
  Activation - Min: 0.000001, Max: 0.169175, Mean: 0.029508

layer3.1.conv1:
  Channels: 256
  Activation - Min: 0.000000, Max: 0.434856, Mean: 0.083851

layer3.1.conv2:
  Channels: 256
  Activation - Min: 0.000000, Max: 0.194815, Mean: 0.028114

layer4.0.conv1:
  Channels: 512
  Activation - Min: 0.000000, Max: 0.214668, Mean: 0.024424

layer4.0.conv2:
  Channels: 512
  Activation - Min: 0.000000, Max: 0.276125, Mean: 0.017481

layer4.0.downsample.0:
  Channels: 512
  Activation - Min: 0.000000, Max: 0.100525, Mean: 0.008787

layer4.1.conv1:
  Channels: 512
  Activation - Min: 0.000000, Max: 0.843143, Mean: 0.054520

layer4.1.conv2:
  Channels: 512
  Activation - Min: 0.000027, Max: 0.098345, Mean: 0.020598

fc:
  Channels: 10
  Activation - Min: 1.776384, Max: 2.898468, Mean: 2.337661
============================================================


Pruning Results:
  Parameters before: 11,173,962
  Parameters after:  7,122,360
  Parameters removed: 4,051,602 (36.26%)

############################################################
Pruning Complete
############################################################
Initial parameters:  11,173,962
Final parameters:    7,122,360
Total removed:       4,051,602 (36.26%)
Target pruning ratio: 20.00%

============================================================
Measuring Pruned Model Metrics
============================================================
✓ Accuracy: 10.00%
✓ Parameters: 7,122,360
✓ Size: 27.24 MB
✓ Inference Time: 0.09 ms/sample

✓ Pruned model saved: ResNet18_CIFAR10_pruned_Wanda.pth

============================================================
Fine-Tuning Pruned Model
============================================================

Epoch 1/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.7786 Acc: 58.77%
  Batch [100/391] Loss: 0.6709 Acc: 68.41%
  Batch [150/391] Loss: 0.5540 Acc: 72.58%
  Batch [200/391] Loss: 0.4864 Acc: 75.36%
  Batch [250/391] Loss: 0.3586 Acc: 76.74%
  Batch [300/391] Loss: 0.4925 Acc: 78.03%
  Batch [350/391] Loss: 0.4090 Acc: 79.03%

  Train Loss: 0.6019 | Train Acc: 79.61%
  Test Loss:  0.4991 | Test Acc:  83.20%

Epoch 2/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.4432 Acc: 86.14%
  Batch [100/391] Loss: 0.5005 Acc: 86.45%
  Batch [150/391] Loss: 0.3738 Acc: 86.36%
  Batch [200/391] Loss: 0.4378 Acc: 86.47%
  Batch [250/391] Loss: 0.5694 Acc: 86.49%
  Batch [300/391] Loss: 0.3978 Acc: 86.56%
  Batch [350/391] Loss: 0.3103 Acc: 86.60%

  Train Loss: 0.3955 | Train Acc: 86.71%
  Test Loss:  0.4594 | Test Acc:  84.44%

Epoch 3/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.4366 Acc: 88.08%
  Batch [100/391] Loss: 0.2444 Acc: 88.76%
  Batch [150/391] Loss: 0.3415 Acc: 88.52%
  Batch [200/391] Loss: 0.4121 Acc: 88.39%
  Batch [250/391] Loss: 0.2963 Acc: 88.47%
  Batch [300/391] Loss: 0.3946 Acc: 88.48%
  Batch [350/391] Loss: 0.3673 Acc: 88.38%

  Train Loss: 0.3473 | Train Acc: 88.39%
  Test Loss:  0.4243 | Test Acc:  85.89%

Epoch 4/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.2094 Acc: 89.77%
  Batch [100/391] Loss: 0.2240 Acc: 89.77%
  Batch [150/391] Loss: 0.3356 Acc: 89.72%
  Batch [200/391] Loss: 0.3135 Acc: 89.75%
  Batch [250/391] Loss: 0.2284 Acc: 89.67%
  Batch [300/391] Loss: 0.2590 Acc: 89.73%
  Batch [350/391] Loss: 0.3832 Acc: 89.64%

  Train Loss: 0.3013 | Train Acc: 89.68%
  Test Loss:  0.3444 | Test Acc:  88.36%

Epoch 5/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.1602 Acc: 91.53%
  Batch [100/391] Loss: 0.1821 Acc: 91.48%
  Batch [150/391] Loss: 0.3100 Acc: 91.44%
  Batch [200/391] Loss: 0.3013 Acc: 91.38%
  Batch [250/391] Loss: 0.2573 Acc: 91.45%
  Batch [300/391] Loss: 0.2326 Acc: 91.41%
  Batch [350/391] Loss: 0.1977 Acc: 91.43%

  Train Loss: 0.2566 | Train Acc: 91.41%
  Test Loss:  0.3744 | Test Acc:  87.58%
  ✓ Checkpoint saved: ResNet18_CIFAR10_FTAP_W_epoch5.pth

Epoch 6/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.2125 Acc: 92.14%
  Batch [100/391] Loss: 0.2434 Acc: 92.73%
  Batch [150/391] Loss: 0.2117 Acc: 92.86%
  Batch [200/391] Loss: 0.0894 Acc: 92.88%
  Batch [250/391] Loss: 0.1866 Acc: 92.83%
  Batch [300/391] Loss: 0.2518 Acc: 92.85%
  Batch [350/391] Loss: 0.2484 Acc: 92.82%

  Train Loss: 0.2171 | Train Acc: 92.79%
  Test Loss:  0.3070 | Test Acc:  90.11%

Epoch 7/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.1493 Acc: 95.02%
  Batch [100/391] Loss: 0.1749 Acc: 94.75%
  Batch [150/391] Loss: 0.2423 Acc: 94.62%
  Batch [200/391] Loss: 0.1479 Acc: 94.45%
  Batch [250/391] Loss: 0.1754 Acc: 94.35%
  Batch [300/391] Loss: 0.2470 Acc: 94.32%
  Batch [350/391] Loss: 0.1315 Acc: 94.36%

  Train Loss: 0.1752 | Train Acc: 94.27%
  Test Loss:  0.3005 | Test Acc:  90.54%

Epoch 8/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.0738 Acc: 95.30%
  Batch [100/391] Loss: 0.1504 Acc: 95.22%
  Batch [150/391] Loss: 0.1127 Acc: 95.22%
  Batch [200/391] Loss: 0.1120 Acc: 95.41%
  Batch [250/391] Loss: 0.2067 Acc: 95.50%
  Batch [300/391] Loss: 0.1165 Acc: 95.49%
  Batch [350/391] Loss: 0.1542 Acc: 95.47%

  Train Loss: 0.1376 | Train Acc: 95.45%
  Test Loss:  0.2589 | Test Acc:  91.44%

Epoch 9/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.1246 Acc: 96.44%
  Batch [100/391] Loss: 0.0815 Acc: 96.41%
  Batch [150/391] Loss: 0.0742 Acc: 96.49%
  Batch [200/391] Loss: 0.1081 Acc: 96.55%
  Batch [250/391] Loss: 0.1110 Acc: 96.57%
  Batch [300/391] Loss: 0.1387 Acc: 96.61%
  Batch [350/391] Loss: 0.0504 Acc: 96.60%

  Train Loss: 0.1043 | Train Acc: 96.65%
  Test Loss:  0.2419 | Test Acc:  92.47%

Epoch 10/10
------------------------------------------------------------
  Batch [50/391] Loss: 0.0876 Acc: 97.16%
  Batch [100/391] Loss: 0.0695 Acc: 96.97%
  Batch [150/391] Loss: 0.0873 Acc: 97.07%
  Batch [200/391] Loss: 0.0777 Acc: 97.07%
  Batch [250/391] Loss: 0.0935 Acc: 97.12%
  Batch [300/391] Loss: 0.0884 Acc: 97.08%
  Batch [350/391] Loss: 0.0588 Acc: 97.09%

  Train Loss: 0.0882 | Train Acc: 97.13%
  Test Loss:  0.2329 | Test Acc:  92.71%
  ✓ Checkpoint saved: ResNet18_CIFAR10_FTAP_W_epoch10.pth

============================================================
Measuring Final Model Metrics (After Fine-Tuning)
============================================================
✓ Accuracy: 92.71%
✓ Parameters: 7,122,360
✓ Size: 27.24 MB
✓ Inference Time: 0.09 ms/sample

✓ Final model saved: ResNet18_CIFAR10_FTAP_Wanda_final.pth

============================================================
Generating PDF Report
============================================================
📊 Collecting metrics...
  - Before pruning metrics...

============================================================
Collecting Pre-Pruning Metrics
============================================================
Parameters: 11,173,962
Model Size: 42.63 MB
GFLOPs: 1.11

Evaluating accuracy...
Accuracy: 93.90%

Measuring inference time...
Inference Time: 1.44 ± 0.15 ms
============================================================

  - After pruning metrics...

============================================================
Collecting Post-Pruning Metrics
============================================================
Parameters: 7,122,360
Model Size: 27.17 MB
GFLOPs: 0.71

Evaluating accuracy...
Accuracy: 92.71%

Measuring inference time...
Inference Time: 3.26 ± 0.59 ms

Reductions:
  Parameters: -36.26%
  Model Size: -36.26%
  FLOPs: -36.33%
  Accuracy Drop: 1.19%
  Speedup: 0.44x
============================================================

  - Pruning decisions...
  - Risk analysis...
✅ Metrics collection complete!

📝 Generating report: TS1_Wanda_ResNet18_CIFAR10.pdf

  ✓ Creating cover page...
  ✓ Adding executive summary...
  ✓ Adding model information...
  ✓ Adding pruning decisions...
  ✓ Adding post-pruning structure...
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.

✗ TS1_03_wanda_pruning.py failed after 4.04 minutes
Error code: 3

====================================================================================================
EXECUTION SUMMARY
====================================================================================================

Script                                   Status
------------------------------------------------------------
Prepare                                  ✓ SUCCESS
Coverage                                 ✗ FAILED
Wanda                                    ✗ FAILED
------------------------------------------------------------

Total execution time: 7.98 minutes (0.13 hours)

✗ Some scripts failed. Please check the logs above.

####################################################################################################
# MASTER SCRIPT COMPLETED
####################################################################################################