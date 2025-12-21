================================================================================
TEST SCENARIO TS2: NEURON COVERAGE PRUNING
================================================================================
Model: ResNet50
Dataset: ImageNet
Method: Neuron Coverage
Device: cuda
Pruning Ratio: 20.0%
================================================================================

================================================================================
LOADING IMAGENET VALIDATION DATASET
================================================================================
✓ Dataset loaded
  - Validation samples: 49997
  - Training samples (subset): 9999
  - Calibration samples: 25600

================================================================================
LOADING BASELINE MODEL
================================================================================
✓ Model loaded from: C:\source\checkpoints\TS2\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                 
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0280 ms
✓ FLOPs: 4.13 GFLOPs

================================================================================
APPLYING NEURON COVERAGE PRUNING
================================================================================
Pruning Ratio: 20.0%
Global Pruning: False
Iterative Steps: 1
Coverage Metric: normalized_mean
Max Calibration Batches: 100
Using static coverage importance (computes once)

Initializing Torch-Pruning with:
  Importance method: coverage
  Pruning ratio: 20.00%
  Coverage metric: normalized_mean
  Global pruning: False
  Iterative steps: 1
  Device: cuda

Applying pruning...

############################################################
Starting Coverage-Based Pruning
############################################################
Initial parameters: 25,557,032

============================================================
Pruning Step 1/1
============================================================

============================================================
Computing Neuron Coverage on Test Data
============================================================
Registered hooks on 54 layers
Collecting activations from test data...
  Processed 10 batches
  Processed 20 batches
  Processed 30 batches
  Processed 40 batches
  Processed 50 batches
  Processed 60 batches
  Processed 70 batches
  Processed 80 batches
  Processed 90 batches
  Processed 100 batches
Activation collection complete

Computing neuron coverage using metric: normalized_mean
  conv1: coverage shape torch.Size([64]), min=-0.719204, max=1.000000, mean=0.007877
  layer1.0.conv1: coverage shape torch.Size([64]), min=-0.682190, max=1.000000, mean=-0.109365
  layer1.0.conv2: coverage shape torch.Size([64]), min=-2.973959, max=1.000000, mean=0.121245
  layer1.0.conv3: coverage shape torch.Size([256]), min=-0.757356, max=1.000000, mean=-0.036555
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=-1.999018, max=1.000000, mean=-0.093973
  layer1.1.conv1: coverage shape torch.Size([64]), min=-1.894950, max=1.000000, mean=-0.058573
  layer1.1.conv2: coverage shape torch.Size([64]), min=-1.528996, max=1.000000, mean=-0.103237
  layer1.1.conv3: coverage shape torch.Size([256]), min=-1.595073, max=1.000000, mean=-0.106023
  layer1.2.conv1: coverage shape torch.Size([64]), min=-0.880974, max=1.000000, mean=-0.115776
  layer1.2.conv2: coverage shape torch.Size([64]), min=-1.833899, max=1.000000, mean=-0.215270
  layer1.2.conv3: coverage shape torch.Size([256]), min=-1.091684, max=1.000000, mean=-0.034237
  layer2.0.conv1: coverage shape torch.Size([128]), min=-1.019013, max=1.000000, mean=-0.088990
  layer2.0.conv2: coverage shape torch.Size([128]), min=-1.425867, max=1.000000, mean=-0.151626
  layer2.0.conv3: coverage shape torch.Size([512]), min=-0.670024, max=1.000000, mean=0.015999
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=-0.822479, max=1.000000, mean=-0.056064
  layer2.1.conv1: coverage shape torch.Size([128]), min=-0.987196, max=1.000000, mean=-0.094125
  layer2.1.conv2: coverage shape torch.Size([128]), min=-0.742981, max=1.000000, mean=0.069063
  layer2.1.conv3: coverage shape torch.Size([512]), min=-0.912296, max=1.000000, mean=-0.044342
  layer2.2.conv1: coverage shape torch.Size([128]), min=-1.438550, max=1.000000, mean=-0.005702
  layer2.2.conv2: coverage shape torch.Size([128]), min=-1.097389, max=1.000000, mean=-0.161774
  layer2.2.conv3: coverage shape torch.Size([512]), min=-0.949783, max=1.000000, mean=-0.002227
  layer2.3.conv1: coverage shape torch.Size([128]), min=-0.599997, max=1.000000, mean=-0.044257
  layer2.3.conv2: coverage shape torch.Size([128]), min=-0.855645, max=1.000000, mean=-0.127679
  layer2.3.conv3: coverage shape torch.Size([512]), min=-1.122202, max=1.000000, mean=-0.051377
  layer3.0.conv1: coverage shape torch.Size([256]), min=-1.506848, max=1.000000, mean=-0.197494
  layer3.0.conv2: coverage shape torch.Size([256]), min=-1.557981, max=1.000000, mean=-0.092344
  layer3.0.conv3: coverage shape torch.Size([1024]), min=-1.174070, max=1.000000, mean=-0.020923
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=-1.309310, max=1.000000, mean=-0.148909
  layer3.1.conv1: coverage shape torch.Size([256]), min=-1.821582, max=1.000000, mean=-0.068483
  layer3.1.conv2: coverage shape torch.Size([256]), min=-2.275647, max=1.000000, mean=-0.071611
  layer3.1.conv3: coverage shape torch.Size([1024]), min=-1.962264, max=1.000000, mean=-0.191248
  layer3.2.conv1: coverage shape torch.Size([256]), min=-1.416377, max=1.000000, mean=-0.248613
  layer3.2.conv2: coverage shape torch.Size([256]), min=-1.240684, max=1.000000, mean=-0.199697
  layer3.2.conv3: coverage shape torch.Size([1024]), min=-1.188975, max=1.000000, mean=-0.051573
  layer3.3.conv1: coverage shape torch.Size([256]), min=-1.151438, max=1.000000, mean=-0.128491
  layer3.3.conv2: coverage shape torch.Size([256]), min=-0.715384, max=1.000000, mean=-0.088865
  layer3.3.conv3: coverage shape torch.Size([1024]), min=-1.389535, max=1.000000, mean=-0.087884
  layer3.4.conv1: coverage shape torch.Size([256]), min=-0.841425, max=1.000000, mean=-0.256141
  layer3.4.conv2: coverage shape torch.Size([256]), min=-1.095347, max=1.000000, mean=-0.254781
  layer3.4.conv3: coverage shape torch.Size([1024]), min=-0.733102, max=1.000000, mean=-0.021730
  layer3.5.conv1: coverage shape torch.Size([256]), min=-1.073064, max=1.000000, mean=-0.174949
  layer3.5.conv2: coverage shape torch.Size([256]), min=-1.387426, max=1.000000, mean=-0.231596
  layer3.5.conv3: coverage shape torch.Size([1024]), min=-1.617904, max=1.000000, mean=-0.265097
  layer4.0.conv1: coverage shape torch.Size([512]), min=-1.412367, max=1.000000, mean=-0.603871
  layer4.0.conv2: coverage shape torch.Size([512]), min=-2.159897, max=1.000000, mean=-0.414398
  layer4.0.conv3: coverage shape torch.Size([2048]), min=-0.937194, max=1.000000, mean=-0.086123
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=-1.224776, max=1.000000, mean=-0.160130
  layer4.1.conv1: coverage shape torch.Size([512]), min=-1.799392, max=1.000000, mean=-0.243296
  layer4.1.conv2: coverage shape torch.Size([512]), min=-1.212236, max=1.000000, mean=-0.206692
  layer4.1.conv3: coverage shape torch.Size([2048]), min=-0.726311, max=1.000000, mean=-0.146386
  layer4.2.conv1: coverage shape torch.Size([512]), min=-1.423435, max=1.000000, mean=-0.225099
  layer4.2.conv2: coverage shape torch.Size([512]), min=-2.209763, max=1.000000, mean=-0.299498
  layer4.2.conv3: coverage shape torch.Size([2048]), min=-2.463499, max=1.000000, mean=-0.834311
  fc: coverage shape torch.Size([1000]), min=-0.777097, max=1.000000, mean=0.000387

------------------------------------------------------------
Coverage Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Coverage - Min: -0.719204, Max: 1.000000, Mean: 0.007877
  Zero coverage neurons: 0

layer1.0.conv1:
  Channels: 64
  Coverage - Min: -0.682190, Max: 1.000000, Mean: -0.109365
  Zero coverage neurons: 0

layer1.0.conv2:
  Channels: 64
  Coverage - Min: -2.973959, Max: 1.000000, Mean: 0.121245
  Zero coverage neurons: 0

layer1.0.conv3:
  Channels: 256
  Coverage - Min: -0.757356, Max: 1.000000, Mean: -0.036555
  Zero coverage neurons: 0

layer1.0.downsample.0:
  Channels: 256
  Coverage - Min: -1.999018, Max: 1.000000, Mean: -0.093973
  Zero coverage neurons: 0

layer1.1.conv1:
  Channels: 64
  Coverage - Min: -1.894950, Max: 1.000000, Mean: -0.058573
  Zero coverage neurons: 0

layer1.1.conv2:
  Channels: 64
  Coverage - Min: -1.528996, Max: 1.000000, Mean: -0.103237
  Zero coverage neurons: 0

layer1.1.conv3:
  Channels: 256
  Coverage - Min: -1.595073, Max: 1.000000, Mean: -0.106023
  Zero coverage neurons: 0

layer1.2.conv1:
  Channels: 64
  Coverage - Min: -0.880974, Max: 1.000000, Mean: -0.115776
  Zero coverage neurons: 0

layer1.2.conv2:
  Channels: 64
  Coverage - Min: -1.833899, Max: 1.000000, Mean: -0.215270
  Zero coverage neurons: 0

layer1.2.conv3:
  Channels: 256
  Coverage - Min: -1.091684, Max: 1.000000, Mean: -0.034237
  Zero coverage neurons: 0

layer2.0.conv1:
  Channels: 128
  Coverage - Min: -1.019013, Max: 1.000000, Mean: -0.088990
  Zero coverage neurons: 0

layer2.0.conv2:
  Channels: 128
  Coverage - Min: -1.425867, Max: 1.000000, Mean: -0.151626
  Zero coverage neurons: 0

layer2.0.conv3:
  Channels: 512
  Coverage - Min: -0.670024, Max: 1.000000, Mean: 0.015999
  Zero coverage neurons: 0

layer2.0.downsample.0:
  Channels: 512
  Coverage - Min: -0.822479, Max: 1.000000, Mean: -0.056064
  Zero coverage neurons: 0

layer2.1.conv1:
  Channels: 128
  Coverage - Min: -0.987196, Max: 1.000000, Mean: -0.094125
  Zero coverage neurons: 0

layer2.1.conv2:
  Channels: 128
  Coverage - Min: -0.742981, Max: 1.000000, Mean: 0.069063
  Zero coverage neurons: 0

layer2.1.conv3:
  Channels: 512
  Coverage - Min: -0.912296, Max: 1.000000, Mean: -0.044342
  Zero coverage neurons: 0

layer2.2.conv1:
  Channels: 128
  Coverage - Min: -1.438550, Max: 1.000000, Mean: -0.005702
  Zero coverage neurons: 0

layer2.2.conv2:
  Channels: 128
  Coverage - Min: -1.097389, Max: 1.000000, Mean: -0.161774
  Zero coverage neurons: 0

layer2.2.conv3:
  Channels: 512
  Coverage - Min: -0.949783, Max: 1.000000, Mean: -0.002227
  Zero coverage neurons: 0

layer2.3.conv1:
  Channels: 128
  Coverage - Min: -0.599997, Max: 1.000000, Mean: -0.044257
  Zero coverage neurons: 0

layer2.3.conv2:
  Channels: 128
  Coverage - Min: -0.855645, Max: 1.000000, Mean: -0.127679
  Zero coverage neurons: 0

layer2.3.conv3:
  Channels: 512
  Coverage - Min: -1.122202, Max: 1.000000, Mean: -0.051377
  Zero coverage neurons: 0

layer3.0.conv1:
  Channels: 256
  Coverage - Min: -1.506848, Max: 1.000000, Mean: -0.197494
  Zero coverage neurons: 0

layer3.0.conv2:
  Channels: 256
  Coverage - Min: -1.557981, Max: 1.000000, Mean: -0.092344
  Zero coverage neurons: 0

layer3.0.conv3:
  Channels: 1024
  Coverage - Min: -1.174070, Max: 1.000000, Mean: -0.020923
  Zero coverage neurons: 0

layer3.0.downsample.0:
  Channels: 1024
  Coverage - Min: -1.309310, Max: 1.000000, Mean: -0.148909
  Zero coverage neurons: 0

layer3.1.conv1:
  Channels: 256
  Coverage - Min: -1.821582, Max: 1.000000, Mean: -0.068483
  Zero coverage neurons: 0

layer3.1.conv2:
  Channels: 256
  Coverage - Min: -2.275647, Max: 1.000000, Mean: -0.071611
  Zero coverage neurons: 0

layer3.1.conv3:
  Channels: 1024
  Coverage - Min: -1.962264, Max: 1.000000, Mean: -0.191248
  Zero coverage neurons: 0

layer3.2.conv1:
  Channels: 256
  Coverage - Min: -1.416377, Max: 1.000000, Mean: -0.248613
  Zero coverage neurons: 0

layer3.2.conv2:
  Channels: 256
  Coverage - Min: -1.240684, Max: 1.000000, Mean: -0.199697
  Zero coverage neurons: 0

layer3.2.conv3:
  Channels: 1024
  Coverage - Min: -1.188975, Max: 1.000000, Mean: -0.051573
  Zero coverage neurons: 0

layer3.3.conv1:
  Channels: 256
  Coverage - Min: -1.151438, Max: 1.000000, Mean: -0.128491
  Zero coverage neurons: 0

layer3.3.conv2:
  Channels: 256
  Coverage - Min: -0.715384, Max: 1.000000, Mean: -0.088865
  Zero coverage neurons: 0

layer3.3.conv3:
  Channels: 1024
  Coverage - Min: -1.389535, Max: 1.000000, Mean: -0.087884
  Zero coverage neurons: 0

layer3.4.conv1:
  Channels: 256
  Coverage - Min: -0.841425, Max: 1.000000, Mean: -0.256141
  Zero coverage neurons: 0

layer3.4.conv2:
  Channels: 256
  Coverage - Min: -1.095347, Max: 1.000000, Mean: -0.254781
  Zero coverage neurons: 0

layer3.4.conv3:
  Channels: 1024
  Coverage - Min: -0.733102, Max: 1.000000, Mean: -0.021730
  Zero coverage neurons: 0

layer3.5.conv1:
  Channels: 256
  Coverage - Min: -1.073064, Max: 1.000000, Mean: -0.174949
  Zero coverage neurons: 0

layer3.5.conv2:
  Channels: 256
  Coverage - Min: -1.387426, Max: 1.000000, Mean: -0.231596
  Zero coverage neurons: 0

layer3.5.conv3:
  Channels: 1024
  Coverage - Min: -1.617904, Max: 1.000000, Mean: -0.265097
  Zero coverage neurons: 0

layer4.0.conv1:
  Channels: 512
  Coverage - Min: -1.412367, Max: 1.000000, Mean: -0.603871
  Zero coverage neurons: 0

layer4.0.conv2:
  Channels: 512
  Coverage - Min: -2.159897, Max: 1.000000, Mean: -0.414398
  Zero coverage neurons: 0

layer4.0.conv3:
  Channels: 2048
  Coverage - Min: -0.937194, Max: 1.000000, Mean: -0.086123
  Zero coverage neurons: 0

layer4.0.downsample.0:
  Channels: 2048
  Coverage - Min: -1.224776, Max: 1.000000, Mean: -0.160130
  Zero coverage neurons: 0

layer4.1.conv1:
  Channels: 512
  Coverage - Min: -1.799392, Max: 1.000000, Mean: -0.243296
  Zero coverage neurons: 0

layer4.1.conv2:
  Channels: 512
  Coverage - Min: -1.212236, Max: 1.000000, Mean: -0.206692
  Zero coverage neurons: 0

layer4.1.conv3:
  Channels: 2048
  Coverage - Min: -0.726311, Max: 1.000000, Mean: -0.146386
  Zero coverage neurons: 0

layer4.2.conv1:
  Channels: 512
  Coverage - Min: -1.423435, Max: 1.000000, Mean: -0.225099
  Zero coverage neurons: 0

layer4.2.conv2:
  Channels: 512
  Coverage - Min: -2.209763, Max: 1.000000, Mean: -0.299498
  Zero coverage neurons: 0

layer4.2.conv3:
  Channels: 2048
  Coverage - Min: -2.463499, Max: 1.000000, Mean: -0.834311
  Zero coverage neurons: 0

fc:
  Channels: 1000
  Coverage - Min: -0.777097, Max: 1.000000, Mean: 0.000387
  Zero coverage neurons: 0
============================================================


Pruning Results:
  Parameters before: 25,557,032
  Parameters after:  16,641,302
  Parameters removed: 8,915,730 (34.89%)

############################################################
Pruning Complete
############################################################
Initial parameters:  25,557,032
Final parameters:    16,641,302
Total removed:       8,915,730 (34.89%)
Target pruning ratio: 20.00%
✓ Neuron Coverage pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.18%                 
✓ Model Size: 63.64 MB
✓ Average Inference Time: 1.1886 ms
✓ FLOPs: 2.66 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 5
Epoch 1/5: 100%|██████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.04s/it, loss=4.0876, acc=25.74%] 
Epoch 1/5 - Train Loss: 4.0876, Train Acc: 25.74%, Test Acc: 43.45%                 
Epoch 2/5: 100%|██████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.9316, acc=40.16%] 
Epoch 2/5 - Train Loss: 2.9316, Train Acc: 40.16%, Test Acc: 49.89%                 
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_NC_epoch2.pth
Epoch 3/5: 100%|██████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.5051, acc=46.96%]
Epoch 3/5 - Train Loss: 2.5051, Train Acc: 46.96%, Test Acc: 52.62%                 
Epoch 4/5: 100%|██████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.3415, acc=50.44%] 
Epoch 4/5 - Train Loss: 2.3415, Train Acc: 50.44%, Test Acc: 54.25%                 
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_NC_epoch4.pth
Epoch 5/5: 100%|██████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.2072, acc=52.19%]
Epoch 5/5 - Train Loss: 2.2072, Train Acc: 52.19%, Test Acc: 54.36%                 
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_NC_epoch5.pth

✓ Best model saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_NC_best.pth
  Best Accuracy: 54.36%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 54.36%
✓ Model Size: 63.64 MB
✓ Average Inference Time: 1.1905 ms
✓ FLOPs: 2.66 GFLOPs

================================================================================
COMPARISON: NEURON COVERAGE PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |          78.59  |          0.18   |    54.36   | -24.23                      |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |          97.7   |         63.64   |    63.64   | -34.05 (-34.9%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |           0.028 |          1.1886 |     1.1905 | +1.1624                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |           4.13  |          2.66   |     2.66   | -1.48 (-35.7%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: C:\source\repos\cleanai-v5\test_scenarios\TS2_ResNet50_ImageNet\TS2_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================




*****************************************



================================================================================
TEST SCENARIO TS2: WANDA PRUNING
================================================================================
Model: ResNet50
Dataset: ImageNet
Method: Wanda
Device: cuda
Pruning Ratio: 20.0%
================================================================================

================================================================================
LOADING IMAGENET VALIDATION DATASET
================================================================================
✓ Dataset loaded
  - Validation samples: 49997
  - Training samples (subset): 9999
  - Calibration samples: 25600

================================================================================
LOADING BASELINE MODEL
================================================================================
✓ Model loaded from: C:\source\checkpoints\TS2\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                       
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0256 ms
✓ FLOPs: 4.13 GFLOPs

================================================================================
APPLYING WANDA PRUNING
================================================================================
Pruning Ratio: 20.0%
Global Pruning: False
Iterative Steps: 1
WANDA combines weight magnitude with activation importance
Using 100 calibration batches
Using WANDA importance (Weight × Activation)

Initializing Torch-Pruning with:
  Importance method: wanda
  Pruning ratio: 20.00%
  Coverage metric: normalized_mean
  Global pruning: False
  Iterative steps: 1
  Device: cuda

Applying pruning...

############################################################
Starting Coverage-Based Pruning
############################################################
Initial parameters: 25,557,032

============================================================
Pruning Step 1/1
============================================================

============================================================
WANDA: Computing Weight × Activation Importance
============================================================
Registered hooks on 54 layers
Collecting activations from test data...
  Processed 10 batches
  Processed 20 batches
  Processed 30 batches
  Processed 40 batches
  Processed 50 batches
  Processed 60 batches
  Processed 70 batches
  Processed 80 batches
  Processed 90 batches
  Processed 100 batches
Activation collection complete

Computing neuron coverage using metric: mean_absolute
  conv1: coverage shape torch.Size([64]), min=0.000029, max=0.486398, mean=0.033889
  layer1.0.conv1: coverage shape torch.Size([64]), min=0.869121, max=44.501053, mean=9.874876
  layer1.0.conv2: coverage shape torch.Size([64]), min=0.030763, max=25.874931, mean=3.914886
  layer1.0.conv3: coverage shape torch.Size([256]), min=0.000000, max=10.946317, mean=2.153108
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=0.000000, max=61.130165, mean=5.198342
  layer1.1.conv1: coverage shape torch.Size([64]), min=0.000000, max=6.716150, mean=1.258657
  layer1.1.conv2: coverage shape torch.Size([64]), min=0.124915, max=6.317972, mean=1.653726
  layer1.1.conv3: coverage shape torch.Size([256]), min=0.001513, max=7.034842, mean=1.303543
  layer1.2.conv1: coverage shape torch.Size([64]), min=0.025437, max=4.760034, mean=1.477240
  layer1.2.conv2: coverage shape torch.Size([64]), min=0.068989, max=6.584223, mean=1.331989
  layer1.2.conv3: coverage shape torch.Size([256]), min=0.002860, max=5.845745, mean=0.939681
  layer2.0.conv1: coverage shape torch.Size([128]), min=0.000701, max=5.212509, mean=1.523029
  layer2.0.conv2: coverage shape torch.Size([128]), min=0.003431, max=6.548773, mean=1.467630
  layer2.0.conv3: coverage shape torch.Size([512]), min=0.000002, max=8.117665, mean=1.116465
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=0.000001, max=8.568792, mean=1.163652
  layer2.1.conv1: coverage shape torch.Size([128]), min=0.083003, max=7.683653, mean=2.209352
  layer2.1.conv2: coverage shape torch.Size([128]), min=0.016350, max=10.115569, mean=1.949980
  layer2.1.conv3: coverage shape torch.Size([512]), min=0.000731, max=2.396634, mean=0.467582
  layer2.2.conv1: coverage shape torch.Size([128]), min=0.006929, max=5.483965, mean=1.217462
  layer2.2.conv2: coverage shape torch.Size([128]), min=0.005829, max=6.203212, mean=1.373860
  layer2.2.conv3: coverage shape torch.Size([512]), min=0.000911, max=3.077017, mean=0.579634
  layer2.3.conv1: coverage shape torch.Size([128]), min=0.028715, max=6.824648, mean=1.272722
  layer2.3.conv2: coverage shape torch.Size([128]), min=0.011197, max=5.937040, mean=1.380960
  layer2.3.conv3: coverage shape torch.Size([512]), min=0.000592, max=3.384454, mean=0.602578
  layer3.0.conv1: coverage shape torch.Size([256]), min=0.000962, max=6.384319, mean=1.402823
  layer3.0.conv2: coverage shape torch.Size([256]), min=0.002286, max=4.462298, mean=0.892436
  layer3.0.conv3: coverage shape torch.Size([1024]), min=0.000121, max=4.704535, mean=0.895213
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=0.000004, max=4.077571, mean=0.831644
  layer3.1.conv1: coverage shape torch.Size([256]), min=0.002179, max=9.750409, mean=1.413310
  layer3.1.conv2: coverage shape torch.Size([256]), min=0.000935, max=11.072261, mean=1.027892
  layer3.1.conv3: coverage shape torch.Size([1024]), min=0.000004, max=2.091472, mean=0.318433
  layer3.2.conv1: coverage shape torch.Size([256]), min=0.001873, max=5.951838, mean=1.282979
  layer3.2.conv2: coverage shape torch.Size([256]), min=0.003157, max=3.081629, mean=0.708762
  layer3.2.conv3: coverage shape torch.Size([1024]), min=0.000420, max=2.031153, mean=0.287359
  layer3.3.conv1: coverage shape torch.Size([256]), min=0.001105, max=5.972992, mean=1.163763
  layer3.3.conv2: coverage shape torch.Size([256]), min=0.010760, max=5.390451, mean=0.820844
  layer3.3.conv3: coverage shape torch.Size([1024]), min=0.001061, max=1.958604, mean=0.335641
  layer3.4.conv1: coverage shape torch.Size([256]), min=0.007420, max=3.740199, mean=1.143794
  layer3.4.conv2: coverage shape torch.Size([256]), min=0.012998, max=3.055632, mean=0.863535
  layer3.4.conv3: coverage shape torch.Size([1024]), min=0.000343, max=2.885662, mean=0.299713
  layer3.5.conv1: coverage shape torch.Size([256]), min=0.007438, max=5.942120, mean=1.247725
  layer3.5.conv2: coverage shape torch.Size([256]), min=0.000032, max=3.802198, mean=0.845825
  layer3.5.conv3: coverage shape torch.Size([1024]), min=0.000977, max=2.128817, mean=0.429822
  layer4.0.conv1: coverage shape torch.Size([512]), min=0.044276, max=3.692292, mean=1.628484
  layer4.0.conv2: coverage shape torch.Size([512]), min=0.000118, max=4.823173, mean=0.953119
  layer4.0.conv3: coverage shape torch.Size([2048]), min=0.000260, max=5.294143, mean=0.589577
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=0.000377, max=3.844552, mean=0.576692
  layer4.1.conv1: coverage shape torch.Size([512]), min=0.103880, max=9.322746, mean=1.404531
  layer4.1.conv2: coverage shape torch.Size([512]), min=0.000546, max=5.474304, mean=1.051825
  layer4.1.conv3: coverage shape torch.Size([2048]), min=0.001326, max=2.934864, mean=0.447551
  layer4.2.conv1: coverage shape torch.Size([512]), min=0.084502, max=7.501315, mean=1.341115
  layer4.2.conv2: coverage shape torch.Size([512]), min=0.045472, max=4.702025, mean=0.664837
  layer4.2.conv3: coverage shape torch.Size([2048]), min=0.000004, max=1.059897, mean=0.361030
  fc: coverage shape torch.Size([1000]), min=0.000065, max=0.280175, mean=0.060093

------------------------------------------------------------
Activation Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Activation - Min: 0.000029, Max: 0.486398, Mean: 0.033889

layer1.0.conv1:
  Channels: 64
  Activation - Min: 0.869121, Max: 44.501053, Mean: 9.874876

layer1.0.conv2:
  Channels: 64
  Activation - Min: 0.030763, Max: 25.874931, Mean: 3.914886

layer1.0.conv3:
  Channels: 256
  Activation - Min: 0.000000, Max: 10.946317, Mean: 2.153108

layer1.0.downsample.0:
  Channels: 256
  Activation - Min: 0.000000, Max: 61.130165, Mean: 5.198342

layer1.1.conv1:
  Channels: 64
  Activation - Min: 0.000000, Max: 6.716150, Mean: 1.258657

layer1.1.conv2:
  Channels: 64
  Activation - Min: 0.124915, Max: 6.317972, Mean: 1.653726

layer1.1.conv3:
  Channels: 256
  Activation - Min: 0.001513, Max: 7.034842, Mean: 1.303543

layer1.2.conv1:
  Channels: 64
  Activation - Min: 0.025437, Max: 4.760034, Mean: 1.477240

layer1.2.conv2:
  Channels: 64
  Activation - Min: 0.068989, Max: 6.584223, Mean: 1.331989

layer1.2.conv3:
  Channels: 256
  Activation - Min: 0.002860, Max: 5.845745, Mean: 0.939681

layer2.0.conv1:
  Channels: 128
  Activation - Min: 0.000701, Max: 5.212509, Mean: 1.523029

layer2.0.conv2:
  Channels: 128
  Activation - Min: 0.003431, Max: 6.548773, Mean: 1.467630

layer2.0.conv3:
  Channels: 512
  Activation - Min: 0.000002, Max: 8.117665, Mean: 1.116465

layer2.0.downsample.0:
  Channels: 512
  Activation - Min: 0.000001, Max: 8.568792, Mean: 1.163652

layer2.1.conv1:
  Channels: 128
  Activation - Min: 0.083003, Max: 7.683653, Mean: 2.209352

layer2.1.conv2:
  Channels: 128
  Activation - Min: 0.016350, Max: 10.115569, Mean: 1.949980

layer2.1.conv3:
  Channels: 512
  Activation - Min: 0.000731, Max: 2.396634, Mean: 0.467582

layer2.2.conv1:
  Channels: 128
  Activation - Min: 0.006929, Max: 5.483965, Mean: 1.217462

layer2.2.conv2:
  Channels: 128
  Activation - Min: 0.005829, Max: 6.203212, Mean: 1.373860

layer2.2.conv3:
  Channels: 512
  Activation - Min: 0.000911, Max: 3.077017, Mean: 0.579634

layer2.3.conv1:
  Channels: 128
  Activation - Min: 0.028715, Max: 6.824648, Mean: 1.272722

layer2.3.conv2:
  Channels: 128
  Activation - Min: 0.011197, Max: 5.937040, Mean: 1.380960

layer2.3.conv3:
  Channels: 512
  Activation - Min: 0.000592, Max: 3.384454, Mean: 0.602578

layer3.0.conv1:
  Channels: 256
  Activation - Min: 0.000962, Max: 6.384319, Mean: 1.402823

layer3.0.conv2:
  Channels: 256
  Activation - Min: 0.002286, Max: 4.462298, Mean: 0.892436

layer3.0.conv3:
  Channels: 1024
  Activation - Min: 0.000121, Max: 4.704535, Mean: 0.895213

layer3.0.downsample.0:
  Channels: 1024
  Activation - Min: 0.000004, Max: 4.077571, Mean: 0.831644

layer3.1.conv1:
  Channels: 256
  Activation - Min: 0.002179, Max: 9.750409, Mean: 1.413310

layer3.1.conv2:
  Channels: 256
  Activation - Min: 0.000935, Max: 11.072261, Mean: 1.027892

layer3.1.conv3:
  Channels: 1024
  Activation - Min: 0.000004, Max: 2.091472, Mean: 0.318433

layer3.2.conv1:
  Channels: 256
  Activation - Min: 0.001873, Max: 5.951838, Mean: 1.282979

layer3.2.conv2:
  Channels: 256
  Activation - Min: 0.003157, Max: 3.081629, Mean: 0.708762

layer3.2.conv3:
  Channels: 1024
  Activation - Min: 0.000420, Max: 2.031153, Mean: 0.287359

layer3.3.conv1:
  Channels: 256
  Activation - Min: 0.001105, Max: 5.972992, Mean: 1.163763

layer3.3.conv2:
  Channels: 256
  Activation - Min: 0.010760, Max: 5.390451, Mean: 0.820844

layer3.3.conv3:
  Channels: 1024
  Activation - Min: 0.001061, Max: 1.958604, Mean: 0.335641

layer3.4.conv1:
  Channels: 256
  Activation - Min: 0.007420, Max: 3.740199, Mean: 1.143794

layer3.4.conv2:
  Channels: 256
  Activation - Min: 0.012998, Max: 3.055632, Mean: 0.863535

layer3.4.conv3:
  Channels: 1024
  Activation - Min: 0.000343, Max: 2.885662, Mean: 0.299713

layer3.5.conv1:
  Channels: 256
  Activation - Min: 0.007438, Max: 5.942120, Mean: 1.247725

layer3.5.conv2:
  Channels: 256
  Activation - Min: 0.000032, Max: 3.802198, Mean: 0.845825

layer3.5.conv3:
  Channels: 1024
  Activation - Min: 0.000977, Max: 2.128817, Mean: 0.429822

layer4.0.conv1:
  Channels: 512
  Activation - Min: 0.044276, Max: 3.692292, Mean: 1.628484

layer4.0.conv2:
  Channels: 512
  Activation - Min: 0.000118, Max: 4.823173, Mean: 0.953119

layer4.0.conv3:
  Channels: 2048
  Activation - Min: 0.000260, Max: 5.294143, Mean: 0.589577

layer4.0.downsample.0:
  Channels: 2048
  Activation - Min: 0.000377, Max: 3.844552, Mean: 0.576692

layer4.1.conv1:
  Channels: 512
  Activation - Min: 0.103880, Max: 9.322746, Mean: 1.404531

layer4.1.conv2:
  Channels: 512
  Activation - Min: 0.000546, Max: 5.474304, Mean: 1.051825

layer4.1.conv3:
  Channels: 2048
  Activation - Min: 0.001326, Max: 2.934864, Mean: 0.447551

layer4.2.conv1:
  Channels: 512
  Activation - Min: 0.084502, Max: 7.501315, Mean: 1.341115

layer4.2.conv2:
  Channels: 512
  Activation - Min: 0.045472, Max: 4.702025, Mean: 0.664837

layer4.2.conv3:
  Channels: 2048
  Activation - Min: 0.000004, Max: 1.059897, Mean: 0.361030

fc:
  Channels: 1000
  Activation - Min: 0.000065, Max: 0.280175, Mean: 0.060093
============================================================


Pruning Results:
  Parameters before: 25,557,032
  Parameters after:  16,641,302
  Parameters removed: 8,915,730 (34.89%)

############################################################
Pruning Complete
############################################################
Initial parameters:  25,557,032
Final parameters:    16,641,302
Total removed:       8,915,730 (34.89%)
Target pruning ratio: 20.00%
✓ Wanda pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.10%                                                                                                          
✓ Model Size: 63.64 MB
✓ Average Inference Time: 1.1914 ms
✓ FLOPs: 2.66 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 5
Epoch 1/5: 100%|███████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.04s/it, loss=5.8248, acc=8.60%] 
Epoch 1/5 - Train Loss: 5.8248, Train Acc: 8.60%, Test Acc: 21.97%                                                                      
Epoch 2/5: 100%|██████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.04s/it, loss=4.1967, acc=23.19%] 
Epoch 2/5 - Train Loss: 4.1967, Train Acc: 23.19%, Test Acc: 35.04%                                                                     
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_W_epoch2.pth
Epoch 3/5: 100%|██████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=3.4814, acc=33.25%]
Epoch 3/5 - Train Loss: 3.4814, Train Acc: 33.25%, Test Acc: 40.10%                                                                     
Epoch 4/5: 100%|██████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=3.1352, acc=37.38%] 
Epoch 4/5 - Train Loss: 3.1352, Train Acc: 37.38%, Test Acc: 42.35%                                                                     
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_W_epoch4.pth
Epoch 5/5: 100%|██████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.04s/it, loss=2.9976, acc=38.98%]
Epoch 5/5 - Train Loss: 2.9976, Train Acc: 38.98%, Test Acc: 42.77%                                                                     
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_W_epoch5.pth

✓ Best model saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_W_best.pth
  Best Accuracy: 42.77%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 42.77%                                                                                                          
✓ Model Size: 63.64 MB
✓ Average Inference Time: 1.1926 ms
✓ FLOPs: 2.66 GFLOPs

================================================================================
COMPARISON: WANDA PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.1    |    42.77   | -35.83                      |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         63.64   |    63.64   | -34.05 (-34.9%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0256 |          1.1914 |     1.1926 | +1.1670                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          2.66   |     2.66   | -1.48 (-35.7%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: C:\source\repos\cleanai-v5\test_scenarios\TS2_ResNet50_ImageNet\TS2_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================