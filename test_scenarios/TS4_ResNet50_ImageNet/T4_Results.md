================================================================================
TEST SCENARIO TS4: NEURON COVERAGE PRUNING
================================================================================
Model: ResNet50
Dataset: ImageNet
Method: Neuron Coverage
Device: cuda
Pruning Ratio: 10.0%
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
✓ Model loaded from: C:\source\checkpoints\TS4\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                                                
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0268 ms
✓ FLOPs: 4.13 GFLOPs

================================================================================
APPLYING NEURON COVERAGE PRUNING
================================================================================
Pruning Ratio: 10.0%
Global Pruning: True
Iterative Steps: 1
Coverage Metric: normalized_mean
Max Calibration Batches: 100
Using static coverage importance (computes once)

Initializing Torch-Pruning with:
  Importance method: coverage
  Pruning ratio: 10.00%
  Coverage metric: normalized_mean
  Global pruning: True
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
  conv1: coverage shape torch.Size([64]), min=-0.718954, max=1.000000, mean=0.007614
  layer1.0.conv1: coverage shape torch.Size([64]), min=-0.680115, max=1.000000, mean=-0.109365
  layer1.0.conv2: coverage shape torch.Size([64]), min=-2.974313, max=1.000000, mean=0.121044
  layer1.0.conv3: coverage shape torch.Size([256]), min=-0.757804, max=1.000000, mean=-0.036483
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=-1.998723, max=1.000000, mean=-0.093962
  layer1.1.conv1: coverage shape torch.Size([64]), min=-1.895565, max=1.000000, mean=-0.058904
  layer1.1.conv2: coverage shape torch.Size([64]), min=-1.530866, max=1.000000, mean=-0.103485
  layer1.1.conv3: coverage shape torch.Size([256]), min=-1.593896, max=1.000000, mean=-0.105969
  layer1.2.conv1: coverage shape torch.Size([64]), min=-0.879173, max=1.000000, mean=-0.115885
  layer1.2.conv2: coverage shape torch.Size([64]), min=-1.831158, max=1.000000, mean=-0.215036
  layer1.2.conv3: coverage shape torch.Size([256]), min=-1.091810, max=1.000000, mean=-0.034232
  layer2.0.conv1: coverage shape torch.Size([128]), min=-1.019249, max=1.000000, mean=-0.088596
  layer2.0.conv2: coverage shape torch.Size([128]), min=-1.425465, max=1.000000, mean=-0.151363
  layer2.0.conv3: coverage shape torch.Size([512]), min=-0.669394, max=1.000000, mean=0.016073
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=-0.821694, max=1.000000, mean=-0.056010
  layer2.1.conv1: coverage shape torch.Size([128]), min=-0.986198, max=1.000000, mean=-0.094161
  layer2.1.conv2: coverage shape torch.Size([128]), min=-0.744114, max=1.000000, mean=0.069158
  layer2.1.conv3: coverage shape torch.Size([512]), min=-0.913280, max=1.000000, mean=-0.044202
  layer2.2.conv1: coverage shape torch.Size([128]), min=-1.439117, max=1.000000, mean=-0.005171
  layer2.2.conv2: coverage shape torch.Size([128]), min=-1.097031, max=1.000000, mean=-0.161657
  layer2.2.conv3: coverage shape torch.Size([512]), min=-0.949644, max=1.000000, mean=-0.002174
  layer2.3.conv1: coverage shape torch.Size([128]), min=-0.600136, max=1.000000, mean=-0.044019
  layer2.3.conv2: coverage shape torch.Size([128]), min=-0.853848, max=1.000000, mean=-0.127369
  layer2.3.conv3: coverage shape torch.Size([512]), min=-1.122902, max=1.000000, mean=-0.051298
  layer3.0.conv1: coverage shape torch.Size([256]), min=-1.499959, max=1.000000, mean=-0.196011
  layer3.0.conv2: coverage shape torch.Size([256]), min=-1.561777, max=1.000000, mean=-0.092353
  layer3.0.conv3: coverage shape torch.Size([1024]), min=-1.174681, max=1.000000, mean=-0.020675
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=-1.307675, max=1.000000, mean=-0.148913
  layer3.1.conv1: coverage shape torch.Size([256]), min=-1.815807, max=1.000000, mean=-0.068195
  layer3.1.conv2: coverage shape torch.Size([256]), min=-2.275792, max=1.000000, mean=-0.071581
  layer3.1.conv3: coverage shape torch.Size([1024]), min=-1.961987, max=1.000000, mean=-0.191076
  layer3.2.conv1: coverage shape torch.Size([256]), min=-1.418970, max=1.000000, mean=-0.248766
  layer3.2.conv2: coverage shape torch.Size([256]), min=-1.241372, max=1.000000, mean=-0.199390
  layer3.2.conv3: coverage shape torch.Size([1024]), min=-1.188973, max=1.000000, mean=-0.051410
  layer3.3.conv1: coverage shape torch.Size([256]), min=-1.148061, max=1.000000, mean=-0.128177
  layer3.3.conv2: coverage shape torch.Size([256]), min=-0.715396, max=1.000000, mean=-0.088963
  layer3.3.conv3: coverage shape torch.Size([1024]), min=-1.389674, max=1.000000, mean=-0.087941
  layer3.4.conv1: coverage shape torch.Size([256]), min=-0.841987, max=1.000000, mean=-0.256058
  layer3.4.conv2: coverage shape torch.Size([256]), min=-1.097893, max=1.000000, mean=-0.255326
  layer3.4.conv3: coverage shape torch.Size([1024]), min=-0.732733, max=1.000000, mean=-0.021667
  layer3.5.conv1: coverage shape torch.Size([256]), min=-1.074096, max=1.000000, mean=-0.174825
  layer3.5.conv2: coverage shape torch.Size([256]), min=-1.386848, max=1.000000, mean=-0.231770
  layer3.5.conv3: coverage shape torch.Size([1024]), min=-1.615617, max=1.000000, mean=-0.264795
  layer4.0.conv1: coverage shape torch.Size([512]), min=-1.412687, max=1.000000, mean=-0.603330
  layer4.0.conv2: coverage shape torch.Size([512]), min=-2.151814, max=1.000000, mean=-0.413078
  layer4.0.conv3: coverage shape torch.Size([2048]), min=-0.937191, max=1.000000, mean=-0.086070
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=-1.223646, max=1.000000, mean=-0.160038
  layer4.1.conv1: coverage shape torch.Size([512]), min=-1.799041, max=1.000000, mean=-0.243389
  layer4.1.conv2: coverage shape torch.Size([512]), min=-1.212894, max=1.000000, mean=-0.206920
  layer4.1.conv3: coverage shape torch.Size([2048]), min=-0.725732, max=1.000000, mean=-0.146385
  layer4.2.conv1: coverage shape torch.Size([512]), min=-1.423287, max=1.000000, mean=-0.225309
  layer4.2.conv2: coverage shape torch.Size([512]), min=-2.209335, max=1.000000, mean=-0.299463
  layer4.2.conv3: coverage shape torch.Size([2048]), min=-2.462011, max=1.000000, mean=-0.831448
  fc: coverage shape torch.Size([1000]), min=-0.821214, max=1.000000, mean=0.000405

------------------------------------------------------------
Coverage Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Coverage - Min: -0.718954, Max: 1.000000, Mean: 0.007614
  Zero coverage neurons: 0

layer1.0.conv1:
  Channels: 64
  Coverage - Min: -0.680115, Max: 1.000000, Mean: -0.109365
  Zero coverage neurons: 0

layer1.0.conv2:
  Channels: 64
  Coverage - Min: -2.974313, Max: 1.000000, Mean: 0.121044
  Zero coverage neurons: 0

layer1.0.conv3:
  Channels: 256
  Coverage - Min: -0.757804, Max: 1.000000, Mean: -0.036483
  Zero coverage neurons: 0

layer1.0.downsample.0:
  Channels: 256
  Coverage - Min: -1.998723, Max: 1.000000, Mean: -0.093962
  Zero coverage neurons: 0

layer1.1.conv1:
  Channels: 64
  Coverage - Min: -1.895565, Max: 1.000000, Mean: -0.058904
  Zero coverage neurons: 0

layer1.1.conv2:
  Channels: 64
  Coverage - Min: -1.530866, Max: 1.000000, Mean: -0.103485
  Zero coverage neurons: 0

layer1.1.conv3:
  Channels: 256
  Coverage - Min: -1.593896, Max: 1.000000, Mean: -0.105969
  Zero coverage neurons: 0

layer1.2.conv1:
  Channels: 64
  Coverage - Min: -0.879173, Max: 1.000000, Mean: -0.115885
  Zero coverage neurons: 0

layer1.2.conv2:
  Channels: 64
  Coverage - Min: -1.831158, Max: 1.000000, Mean: -0.215036
  Zero coverage neurons: 0

layer1.2.conv3:
  Channels: 256
  Coverage - Min: -1.091810, Max: 1.000000, Mean: -0.034232
  Zero coverage neurons: 0

layer2.0.conv1:
  Channels: 128
  Coverage - Min: -1.019249, Max: 1.000000, Mean: -0.088596
  Zero coverage neurons: 0

layer2.0.conv2:
  Channels: 128
  Coverage - Min: -1.425465, Max: 1.000000, Mean: -0.151363
  Zero coverage neurons: 0

layer2.0.conv3:
  Channels: 512
  Coverage - Min: -0.669394, Max: 1.000000, Mean: 0.016073
  Zero coverage neurons: 0

layer2.0.downsample.0:
  Channels: 512
  Coverage - Min: -0.821694, Max: 1.000000, Mean: -0.056010
  Zero coverage neurons: 0

layer2.1.conv1:
  Channels: 128
  Coverage - Min: -0.986198, Max: 1.000000, Mean: -0.094161
  Zero coverage neurons: 0

layer2.1.conv2:
  Channels: 128
  Coverage - Min: -0.744114, Max: 1.000000, Mean: 0.069158
  Zero coverage neurons: 0

layer2.1.conv3:
  Channels: 512
  Coverage - Min: -0.913280, Max: 1.000000, Mean: -0.044202
  Zero coverage neurons: 0

layer2.2.conv1:
  Channels: 128
  Coverage - Min: -1.439117, Max: 1.000000, Mean: -0.005171
  Zero coverage neurons: 0

layer2.2.conv2:
  Channels: 128
  Coverage - Min: -1.097031, Max: 1.000000, Mean: -0.161657
  Zero coverage neurons: 0

layer2.2.conv3:
  Channels: 512
  Coverage - Min: -0.949644, Max: 1.000000, Mean: -0.002174
  Zero coverage neurons: 0

layer2.3.conv1:
  Channels: 128
  Coverage - Min: -0.600136, Max: 1.000000, Mean: -0.044019
  Zero coverage neurons: 0

layer2.3.conv2:
  Channels: 128
  Coverage - Min: -0.853848, Max: 1.000000, Mean: -0.127369
  Zero coverage neurons: 0

layer2.3.conv3:
  Channels: 512
  Coverage - Min: -1.122902, Max: 1.000000, Mean: -0.051298
  Zero coverage neurons: 0

layer3.0.conv1:
  Channels: 256
  Coverage - Min: -1.499959, Max: 1.000000, Mean: -0.196011
  Zero coverage neurons: 0

layer3.0.conv2:
  Channels: 256
  Coverage - Min: -1.561777, Max: 1.000000, Mean: -0.092353
  Zero coverage neurons: 0

layer3.0.conv3:
  Channels: 1024
  Coverage - Min: -1.174681, Max: 1.000000, Mean: -0.020675
  Zero coverage neurons: 0

layer3.0.downsample.0:
  Channels: 1024
  Coverage - Min: -1.307675, Max: 1.000000, Mean: -0.148913
  Zero coverage neurons: 0

layer3.1.conv1:
  Channels: 256
  Coverage - Min: -1.815807, Max: 1.000000, Mean: -0.068195
  Zero coverage neurons: 0

layer3.1.conv2:
  Channels: 256
  Coverage - Min: -2.275792, Max: 1.000000, Mean: -0.071581
  Zero coverage neurons: 0

layer3.1.conv3:
  Channels: 1024
  Coverage - Min: -1.961987, Max: 1.000000, Mean: -0.191076
  Zero coverage neurons: 0

layer3.2.conv1:
  Channels: 256
  Coverage - Min: -1.418970, Max: 1.000000, Mean: -0.248766
  Zero coverage neurons: 0

layer3.2.conv2:
  Channels: 256
  Coverage - Min: -1.241372, Max: 1.000000, Mean: -0.199390
  Zero coverage neurons: 0

layer3.2.conv3:
  Channels: 1024
  Coverage - Min: -1.188973, Max: 1.000000, Mean: -0.051410
  Zero coverage neurons: 0

layer3.3.conv1:
  Channels: 256
  Coverage - Min: -1.148061, Max: 1.000000, Mean: -0.128177
  Zero coverage neurons: 0

layer3.3.conv2:
  Channels: 256
  Coverage - Min: -0.715396, Max: 1.000000, Mean: -0.088963
  Zero coverage neurons: 0

layer3.3.conv3:
  Channels: 1024
  Coverage - Min: -1.389674, Max: 1.000000, Mean: -0.087941
  Zero coverage neurons: 0

layer3.4.conv1:
  Channels: 256
  Coverage - Min: -0.841987, Max: 1.000000, Mean: -0.256058
  Zero coverage neurons: 0

layer3.4.conv2:
  Channels: 256
  Coverage - Min: -1.097893, Max: 1.000000, Mean: -0.255326
  Zero coverage neurons: 0

layer3.4.conv3:
  Channels: 1024
  Coverage - Min: -0.732733, Max: 1.000000, Mean: -0.021667
  Zero coverage neurons: 0

layer3.5.conv1:
  Channels: 256
  Coverage - Min: -1.074096, Max: 1.000000, Mean: -0.174825
  Zero coverage neurons: 0

layer3.5.conv2:
  Channels: 256
  Coverage - Min: -1.386848, Max: 1.000000, Mean: -0.231770
  Zero coverage neurons: 0

layer3.5.conv3:
  Channels: 1024
  Coverage - Min: -1.615617, Max: 1.000000, Mean: -0.264795
  Zero coverage neurons: 0

layer4.0.conv1:
  Channels: 512
  Coverage - Min: -1.412687, Max: 1.000000, Mean: -0.603330
  Zero coverage neurons: 0

layer4.0.conv2:
  Channels: 512
  Coverage - Min: -2.151814, Max: 1.000000, Mean: -0.413078
  Zero coverage neurons: 0

layer4.0.conv3:
  Channels: 2048
  Coverage - Min: -0.937191, Max: 1.000000, Mean: -0.086070
  Zero coverage neurons: 0

layer4.0.downsample.0:
  Channels: 2048
  Coverage - Min: -1.223646, Max: 1.000000, Mean: -0.160038
  Zero coverage neurons: 0

layer4.1.conv1:
  Channels: 512
  Coverage - Min: -1.799041, Max: 1.000000, Mean: -0.243389
  Zero coverage neurons: 0

layer4.1.conv2:
  Channels: 512
  Coverage - Min: -1.212894, Max: 1.000000, Mean: -0.206920
  Zero coverage neurons: 0

layer4.1.conv3:
  Channels: 2048
  Coverage - Min: -0.725732, Max: 1.000000, Mean: -0.146385
  Zero coverage neurons: 0

layer4.2.conv1:
  Channels: 512
  Coverage - Min: -1.423287, Max: 1.000000, Mean: -0.225309
  Zero coverage neurons: 0

layer4.2.conv2:
  Channels: 512
  Coverage - Min: -2.209335, Max: 1.000000, Mean: -0.299463
  Zero coverage neurons: 0

layer4.2.conv3:
  Channels: 2048
  Coverage - Min: -2.462011, Max: 1.000000, Mean: -0.831448
  Zero coverage neurons: 0

fc:
  Channels: 1000
  Coverage - Min: -0.821214, Max: 1.000000, Mean: 0.000405
  Zero coverage neurons: 0
============================================================


Pruning Results:
  Parameters before: 25,557,032
  Parameters after:  21,650,982
  Parameters removed: 3,906,050 (15.28%)

############################################################
Pruning Complete
############################################################
Initial parameters:  25,557,032
Final parameters:    21,650,982
Total removed:       3,906,050 (15.28%)
Target pruning ratio: 10.00%
✓ Neuron Coverage pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.11%                                                                                                                                   
✓ Model Size: 82.78 MB
✓ Average Inference Time: 1.3332 ms
✓ FLOPs: 3.20 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:30<00:00,  2.27s/it, loss=3.3259, acc=37.81%] 
Epoch 1/10 - Train Loss: 3.3259, Train Acc: 37.81%, Test Acc: 60.06%                                                                                             
Epoch 2/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.22s/it, loss=2.0507, acc=56.07%] 
Epoch 2/10 - Train Loss: 2.0507, Train Acc: 56.07%, Test Acc: 64.88%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_NC_epoch2.pth
Epoch 3/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.22s/it, loss=1.6999, acc=62.25%]
Epoch 3/10 - Train Loss: 1.6999, Train Acc: 62.25%, Test Acc: 67.31%                                                                                             
Epoch 4/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.21s/it, loss=1.4702, acc=67.02%] 
Epoch 4/10 - Train Loss: 1.4702, Train Acc: 67.02%, Test Acc: 68.85%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_NC_epoch4.pth
Epoch 5/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.22s/it, loss=1.3098, acc=69.88%]
Epoch 5/10 - Train Loss: 1.3098, Train Acc: 69.88%, Test Acc: 69.70%                                                                                             
Epoch 6/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.21s/it, loss=1.2184, acc=71.78%] 
Epoch 6/10 - Train Loss: 1.2184, Train Acc: 71.78%, Test Acc: 70.25%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_NC_epoch6.pth
Epoch 7/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.22s/it, loss=1.1130, acc=74.94%]
Epoch 7/10 - Train Loss: 1.1130, Train Acc: 74.94%, Test Acc: 70.78%                                                                                             
Epoch 8/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:29<00:00,  2.24s/it, loss=1.0637, acc=75.43%] 
Epoch 8/10 - Train Loss: 1.0637, Train Acc: 75.43%, Test Acc: 71.14%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_NC_epoch8.pth
Epoch 9/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:29<00:00,  2.24s/it, loss=1.0450, acc=76.07%]
Epoch 9/10 - Train Loss: 1.0450, Train Acc: 76.07%, Test Acc: 71.25%                                                                                             
Epoch 10/10: 100%|█████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:29<00:00,  2.24s/it, loss=1.0050, acc=76.96%] 
Epoch 10/10 - Train Loss: 1.0050, Train Acc: 76.96%, Test Acc: 71.23%                                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_NC_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_NC_best.pth
  Best Accuracy: 71.25%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 71.23%                                                                                                                                   
✓ Model Size: 82.78 MB
✓ Average Inference Time: 2.0268 ms
✓ FLOPs: 3.20 GFLOPs

================================================================================
COMPARISON: NEURON COVERAGE PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.11   |    71.23   | -7.36                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         82.78   |    82.78   | -14.92 (-15.3%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0268 |          1.3332 |     2.0268 | +2.0000                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.2    |     3.2    | -0.93 (-22.5%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS4_ResNet50_ImageNet\TS4_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================

====================================================================================================
✓ TS4_02_coverage_pruning.py completed successfully
  Elapsed time: 2361.99 seconds
====================================================================================================


====================================================================================================
SCRIPT 3/6: TS4_03_wanda_pruning.py
====================================================================================================

====================================================================================================
RUNNING: TS4_03_wanda_pruning.py
====================================================================================================


================================================================================
TEST SCENARIO TS4: WANDA PRUNING
================================================================================
Model: ResNet50
Dataset: ImageNet
Method: Wanda
Device: cuda
Pruning Ratio: 10.0%
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
✓ Model loaded from: C:\source\checkpoints\TS4\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                                                
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0302 ms
✓ FLOPs: 4.13 GFLOPs

================================================================================
APPLYING WANDA PRUNING
================================================================================
Pruning Ratio: 10.0%
Global Pruning: True
Iterative Steps: 1
WANDA combines weight magnitude with activation importance
Using 100 calibration batches
Using WANDA importance (Weight × Activation)

Initializing Torch-Pruning with:
  Importance method: wanda
  Pruning ratio: 10.00%
  Coverage metric: normalized_mean
  Global pruning: True
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
  conv1: coverage shape torch.Size([64]), min=0.000018, max=0.463468, mean=0.031987
  layer1.0.conv1: coverage shape torch.Size([64]), min=0.865138, max=44.623802, mean=9.893398
  layer1.0.conv2: coverage shape torch.Size([64]), min=0.042248, max=25.857676, mean=3.915015
  layer1.0.conv3: coverage shape torch.Size([256]), min=0.000000, max=10.937715, mean=2.151909
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=0.000000, max=61.284496, mean=5.207543
  layer1.1.conv1: coverage shape torch.Size([64]), min=0.000000, max=6.706448, mean=1.258923
  layer1.1.conv2: coverage shape torch.Size([64]), min=0.125391, max=6.323277, mean=1.657119
  layer1.1.conv3: coverage shape torch.Size([256]), min=0.001693, max=7.037649, mean=1.304591
  layer1.2.conv1: coverage shape torch.Size([64]), min=0.037270, max=4.759827, mean=1.477857
  layer1.2.conv2: coverage shape torch.Size([64]), min=0.069914, max=6.591675, mean=1.333339
  layer1.2.conv3: coverage shape torch.Size([256]), min=0.002854, max=5.847116, mean=0.939919
  layer2.0.conv1: coverage shape torch.Size([128]), min=0.001647, max=5.208591, mean=1.523976
  layer2.0.conv2: coverage shape torch.Size([128]), min=0.007672, max=6.560421, mean=1.469620
  layer2.0.conv3: coverage shape torch.Size([512]), min=0.000002, max=8.119966, mean=1.117013
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=0.000001, max=8.556812, mean=1.163959
  layer2.1.conv1: coverage shape torch.Size([128]), min=0.081494, max=7.689734, mean=2.209903
  layer2.1.conv2: coverage shape torch.Size([128]), min=0.018275, max=10.108135, mean=1.950714
  layer2.1.conv3: coverage shape torch.Size([512]), min=0.000199, max=2.395819, mean=0.467776
  layer2.2.conv1: coverage shape torch.Size([128]), min=0.003893, max=5.491566, mean=1.217828
  layer2.2.conv2: coverage shape torch.Size([128]), min=0.009580, max=6.201503, mean=1.374674
  layer2.2.conv3: coverage shape torch.Size([512]), min=0.000004, max=3.077399, mean=0.579489
  layer2.3.conv1: coverage shape torch.Size([128]), min=0.024303, max=6.830197, mean=1.273417
  layer2.3.conv2: coverage shape torch.Size([128]), min=0.014007, max=5.937801, mean=1.381140
  layer2.3.conv3: coverage shape torch.Size([512]), min=0.000581, max=3.385068, mean=0.602516
  layer3.0.conv1: coverage shape torch.Size([256]), min=0.001163, max=6.382040, mean=1.403026
  layer3.0.conv2: coverage shape torch.Size([256]), min=0.002148, max=4.463396, mean=0.893358
  layer3.0.conv3: coverage shape torch.Size([1024]), min=0.000289, max=4.704841, mean=0.894998
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=0.000252, max=4.078314, mean=0.832225
  layer3.1.conv1: coverage shape torch.Size([256]), min=0.003413, max=9.755636, mean=1.414216
  layer3.1.conv2: coverage shape torch.Size([256]), min=0.002481, max=11.071131, mean=1.028253
  layer3.1.conv3: coverage shape torch.Size([1024]), min=0.000697, max=2.092113, mean=0.318635
  layer3.2.conv1: coverage shape torch.Size([256]), min=0.001509, max=5.953524, mean=1.283567
  layer3.2.conv2: coverage shape torch.Size([256]), min=0.005995, max=3.086114, mean=0.709235
  layer3.2.conv3: coverage shape torch.Size([1024]), min=0.000021, max=2.032020, mean=0.287338
  layer3.3.conv1: coverage shape torch.Size([256]), min=0.001715, max=5.968800, mean=1.165074
  layer3.3.conv2: coverage shape torch.Size([256]), min=0.010337, max=5.399460, mean=0.821056
  layer3.3.conv3: coverage shape torch.Size([1024]), min=0.000937, max=1.960116, mean=0.335788
  layer3.4.conv1: coverage shape torch.Size([256]), min=0.013600, max=3.743222, mean=1.144817
  layer3.4.conv2: coverage shape torch.Size([256]), min=0.012862, max=3.060057, mean=0.864609
  layer3.4.conv3: coverage shape torch.Size([1024]), min=0.000353, max=2.885176, mean=0.299672
  layer3.5.conv1: coverage shape torch.Size([256]), min=0.001476, max=5.941960, mean=1.248652
  layer3.5.conv2: coverage shape torch.Size([256]), min=0.001557, max=3.803387, mean=0.847094
  layer3.5.conv3: coverage shape torch.Size([1024]), min=0.001370, max=2.130053, mean=0.430069
  layer4.0.conv1: coverage shape torch.Size([512]), min=0.043075, max=3.698700, mean=1.628966
  layer4.0.conv2: coverage shape torch.Size([512]), min=0.003745, max=4.827356, mean=0.953804
  layer4.0.conv3: coverage shape torch.Size([2048]), min=0.000625, max=5.294218, mean=0.589508
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=0.000608, max=3.843331, mean=0.577079
  layer4.1.conv1: coverage shape torch.Size([512]), min=0.095942, max=9.324087, mean=1.402962
  layer4.1.conv2: coverage shape torch.Size([512]), min=0.001281, max=5.473158, mean=1.052745
  layer4.1.conv3: coverage shape torch.Size([2048]), min=0.000060, max=2.933403, mean=0.447230
  layer4.2.conv1: coverage shape torch.Size([512]), min=0.083176, max=7.499516, mean=1.339079
  layer4.2.conv2: coverage shape torch.Size([512]), min=0.047346, max=4.692945, mean=0.663935
  layer4.2.conv3: coverage shape torch.Size([2048]), min=0.000653, max=1.060666, mean=0.361493
  fc: coverage shape torch.Size([1000]), min=0.000028, max=0.278331, mean=0.059975

------------------------------------------------------------
Activation Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Activation - Min: 0.000018, Max: 0.463468, Mean: 0.031987

layer1.0.conv1:
  Channels: 64
  Activation - Min: 0.865138, Max: 44.623802, Mean: 9.893398

layer1.0.conv2:
  Channels: 64
  Activation - Min: 0.042248, Max: 25.857676, Mean: 3.915015

layer1.0.conv3:
  Channels: 256
  Activation - Min: 0.000000, Max: 10.937715, Mean: 2.151909

layer1.0.downsample.0:
  Channels: 256
  Activation - Min: 0.000000, Max: 61.284496, Mean: 5.207543

layer1.1.conv1:
  Channels: 64
  Activation - Min: 0.000000, Max: 6.706448, Mean: 1.258923

layer1.1.conv2:
  Channels: 64
  Activation - Min: 0.125391, Max: 6.323277, Mean: 1.657119

layer1.1.conv3:
  Channels: 256
  Activation - Min: 0.001693, Max: 7.037649, Mean: 1.304591

layer1.2.conv1:
  Channels: 64
  Activation - Min: 0.037270, Max: 4.759827, Mean: 1.477857

layer1.2.conv2:
  Channels: 64
  Activation - Min: 0.069914, Max: 6.591675, Mean: 1.333339

layer1.2.conv3:
  Channels: 256
  Activation - Min: 0.002854, Max: 5.847116, Mean: 0.939919

layer2.0.conv1:
  Channels: 128
  Activation - Min: 0.001647, Max: 5.208591, Mean: 1.523976

layer2.0.conv2:
  Channels: 128
  Activation - Min: 0.007672, Max: 6.560421, Mean: 1.469620

layer2.0.conv3:
  Channels: 512
  Activation - Min: 0.000002, Max: 8.119966, Mean: 1.117013

layer2.0.downsample.0:
  Channels: 512
  Activation - Min: 0.000001, Max: 8.556812, Mean: 1.163959

layer2.1.conv1:
  Channels: 128
  Activation - Min: 0.081494, Max: 7.689734, Mean: 2.209903

layer2.1.conv2:
  Channels: 128
  Activation - Min: 0.018275, Max: 10.108135, Mean: 1.950714

layer2.1.conv3:
  Channels: 512
  Activation - Min: 0.000199, Max: 2.395819, Mean: 0.467776

layer2.2.conv1:
  Channels: 128
  Activation - Min: 0.003893, Max: 5.491566, Mean: 1.217828

layer2.2.conv2:
  Channels: 128
  Activation - Min: 0.009580, Max: 6.201503, Mean: 1.374674

layer2.2.conv3:
  Channels: 512
  Activation - Min: 0.000004, Max: 3.077399, Mean: 0.579489

layer2.3.conv1:
  Channels: 128
  Activation - Min: 0.024303, Max: 6.830197, Mean: 1.273417

layer2.3.conv2:
  Channels: 128
  Activation - Min: 0.014007, Max: 5.937801, Mean: 1.381140

layer2.3.conv3:
  Channels: 512
  Activation - Min: 0.000581, Max: 3.385068, Mean: 0.602516

layer3.0.conv1:
  Channels: 256
  Activation - Min: 0.001163, Max: 6.382040, Mean: 1.403026

layer3.0.conv2:
  Channels: 256
  Activation - Min: 0.002148, Max: 4.463396, Mean: 0.893358

layer3.0.conv3:
  Channels: 1024
  Activation - Min: 0.000289, Max: 4.704841, Mean: 0.894998

layer3.0.downsample.0:
  Channels: 1024
  Activation - Min: 0.000252, Max: 4.078314, Mean: 0.832225

layer3.1.conv1:
  Channels: 256
  Activation - Min: 0.003413, Max: 9.755636, Mean: 1.414216

layer3.1.conv2:
  Channels: 256
  Activation - Min: 0.002481, Max: 11.071131, Mean: 1.028253

layer3.1.conv3:
  Channels: 1024
  Activation - Min: 0.000697, Max: 2.092113, Mean: 0.318635

layer3.2.conv1:
  Channels: 256
  Activation - Min: 0.001509, Max: 5.953524, Mean: 1.283567

layer3.2.conv2:
  Channels: 256
  Activation - Min: 0.005995, Max: 3.086114, Mean: 0.709235

layer3.2.conv3:
  Channels: 1024
  Activation - Min: 0.000021, Max: 2.032020, Mean: 0.287338

layer3.3.conv1:
  Channels: 256
  Activation - Min: 0.001715, Max: 5.968800, Mean: 1.165074

layer3.3.conv2:
  Channels: 256
  Activation - Min: 0.010337, Max: 5.399460, Mean: 0.821056

layer3.3.conv3:
  Channels: 1024
  Activation - Min: 0.000937, Max: 1.960116, Mean: 0.335788

layer3.4.conv1:
  Channels: 256
  Activation - Min: 0.013600, Max: 3.743222, Mean: 1.144817

layer3.4.conv2:
  Channels: 256
  Activation - Min: 0.012862, Max: 3.060057, Mean: 0.864609

layer3.4.conv3:
  Channels: 1024
  Activation - Min: 0.000353, Max: 2.885176, Mean: 0.299672

layer3.5.conv1:
  Channels: 256
  Activation - Min: 0.001476, Max: 5.941960, Mean: 1.248652

layer3.5.conv2:
  Channels: 256
  Activation - Min: 0.001557, Max: 3.803387, Mean: 0.847094

layer3.5.conv3:
  Channels: 1024
  Activation - Min: 0.001370, Max: 2.130053, Mean: 0.430069

layer4.0.conv1:
  Channels: 512
  Activation - Min: 0.043075, Max: 3.698700, Mean: 1.628966

layer4.0.conv2:
  Channels: 512
  Activation - Min: 0.003745, Max: 4.827356, Mean: 0.953804

layer4.0.conv3:
  Channels: 2048
  Activation - Min: 0.000625, Max: 5.294218, Mean: 0.589508

layer4.0.downsample.0:
  Channels: 2048
  Activation - Min: 0.000608, Max: 3.843331, Mean: 0.577079

layer4.1.conv1:
  Channels: 512
  Activation - Min: 0.095942, Max: 9.324087, Mean: 1.402962

layer4.1.conv2:
  Channels: 512
  Activation - Min: 0.001281, Max: 5.473158, Mean: 1.052745

layer4.1.conv3:
  Channels: 2048
  Activation - Min: 0.000060, Max: 2.933403, Mean: 0.447230

layer4.2.conv1:
  Channels: 512
  Activation - Min: 0.083176, Max: 7.499516, Mean: 1.339079

layer4.2.conv2:
  Channels: 512
  Activation - Min: 0.047346, Max: 4.692945, Mean: 0.663935

layer4.2.conv3:
  Channels: 2048
  Activation - Min: 0.000653, Max: 1.060666, Mean: 0.361493

fc:
  Channels: 1000
  Activation - Min: 0.000028, Max: 0.278331, Mean: 0.059975
============================================================


Pruning Results:
  Parameters before: 25,557,032
  Parameters after:  21,468,829
  Parameters removed: 4,088,203 (16.00%)

############################################################
Pruning Complete
############################################################
Initial parameters:  25,557,032
Final parameters:    21,468,829
Total removed:       4,088,203 (16.00%)
Target pruning ratio: 10.00%
✓ Wanda pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.11%                                                                                                                                   
✓ Model Size: 82.07 MB
✓ Average Inference Time: 1.1839 ms
✓ FLOPs: 3.17 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:42<00:00,  1.06s/it, loss=3.2447, acc=37.17%] 
Epoch 1/10 - Train Loss: 3.2447, Train Acc: 37.17%, Test Acc: 55.40%                                                                                             
Epoch 2/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:42<00:00,  1.05s/it, loss=2.3064, acc=50.95%] 
Epoch 2/10 - Train Loss: 2.3064, Train Acc: 50.95%, Test Acc: 60.22%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_W_epoch2.pth
Epoch 3/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.05s/it, loss=1.9209, acc=57.75%]
Epoch 3/10 - Train Loss: 1.9209, Train Acc: 57.75%, Test Acc: 62.78%                                                                                             
Epoch 4/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.6790, acc=62.70%] 
Epoch 4/10 - Train Loss: 1.6790, Train Acc: 62.70%, Test Acc: 64.69%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_W_epoch4.pth
Epoch 5/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.5137, acc=65.62%]
Epoch 5/10 - Train Loss: 1.5137, Train Acc: 65.62%, Test Acc: 66.21%                                                                                             
Epoch 6/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.4123, acc=68.06%] 
Epoch 6/10 - Train Loss: 1.4123, Train Acc: 68.06%, Test Acc: 66.99%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_W_epoch6.pth
Epoch 7/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:40<00:00,  1.02s/it, loss=1.3269, acc=69.55%]
Epoch 7/10 - Train Loss: 1.3269, Train Acc: 69.55%, Test Acc: 67.46%                                                                                             
Epoch 8/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.2301, acc=71.11%] 
Epoch 8/10 - Train Loss: 1.2301, Train Acc: 71.11%, Test Acc: 67.86%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_W_epoch8.pth
Epoch 9/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:40<00:00,  1.02s/it, loss=1.2445, acc=71.69%]
Epoch 9/10 - Train Loss: 1.2445, Train Acc: 71.69%, Test Acc: 67.91%                                                                                             
Epoch 10/10: 100%|█████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.1985, acc=72.16%] 
Epoch 10/10 - Train Loss: 1.1985, Train Acc: 72.16%, Test Acc: 68.13%                                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_W_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_W_best.pth
  Best Accuracy: 68.13%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 68.13%                                                                                                                                   
✓ Model Size: 82.07 MB
✓ Average Inference Time: 1.1832 ms
✓ FLOPs: 3.17 GFLOPs

================================================================================
COMPARISON: WANDA PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.11   |    68.13   | -10.46                      |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         82.07   |    82.07   | -15.62 (-16.0%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0302 |          1.1839 |     1.1832 | +1.1530                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.17   |     3.17   | -0.97 (-23.4%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS4_ResNet50_ImageNet\TS4_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================

====================================================================================================
✓ TS4_03_wanda_pruning.py completed successfully
  Elapsed time: 1492.88 seconds
====================================================================================================


====================================================================================================
SCRIPT 4/6: TS4_04_magnitude_pruning.py
====================================================================================================

====================================================================================================
RUNNING: TS4_04_magnitude_pruning.py
====================================================================================================


================================================================================
TEST SCENARIO TS4: MAGNITUDE-BASED PRUNING
================================================================================
Model: ResNet50
Dataset: ImageNet
Method: Magnitude (Torch-Pruning)
Device: cuda
Pruning Ratio: 10.0%
================================================================================

================================================================================
LOADING IMAGENET VALIDATION DATASET
================================================================================
✓ Dataset loaded
  - Validation samples: 49997
  - Training samples (subset): 9999

================================================================================
LOADING BASELINE MODEL
================================================================================
✓ Model loaded from: C:\source\checkpoints\TS4\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                                                
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0309 ms
✓ FLOPs: 4.13 GFLOPs

================================================================================
APPLYING MAGNITUDE-BASED PRUNING
================================================================================
Pruning Ratio: 10.0%
Global Pruning: True
Iterative Steps: 1
Magnitude pruning uses L2 norm of weights to determine importance
This is Torch-Pruning's most robust baseline method
Using magnitude-based importance (L2 norm)

Initializing Torch-Pruning with:
  Importance method: magnitude
  Pruning ratio: 10.00%
  Global pruning: True
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

Pruning Results:
  Parameters before: 25,557,032
  Parameters after:  21,303,919
  Parameters removed: 4,253,113 (16.64%)

############################################################
Pruning Complete
############################################################
Initial parameters:  25,557,032
Final parameters:    21,303,919
Total removed:       4,253,113 (16.64%)
Target pruning ratio: 10.00%
✓ Magnitude-based pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.12%                                                                                                                                   
✓ Model Size: 81.44 MB
✓ Average Inference Time: 0.0219 ms
✓ FLOPs: 3.05 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.02it/s, loss=3.6052, acc=34.17%] 
Epoch 1/10 - Train Loss: 3.6052, Train Acc: 34.17%, Test Acc: 55.90%                                                                                             
Epoch 2/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.02it/s, loss=2.3821, acc=50.80%] 
Epoch 2/10 - Train Loss: 2.3821, Train Acc: 50.80%, Test Acc: 62.27%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_MAG_epoch2.pth
Epoch 3/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.02it/s, loss=2.0160, acc=56.78%]
Epoch 3/10 - Train Loss: 2.0160, Train Acc: 56.78%, Test Acc: 64.53%                                                                                             
Epoch 4/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.03it/s, loss=1.7624, acc=61.33%] 
Epoch 4/10 - Train Loss: 1.7624, Train Acc: 61.33%, Test Acc: 66.11%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_MAG_epoch4.pth
Epoch 5/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.02it/s, loss=1.6183, acc=63.87%]
Epoch 5/10 - Train Loss: 1.6183, Train Acc: 63.87%, Test Acc: 67.19%
Epoch 6/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.02it/s, loss=1.5259, acc=66.29%] 
Epoch 6/10 - Train Loss: 1.5259, Train Acc: 66.29%, Test Acc: 68.04%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_MAG_epoch6.pth
Epoch 7/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.01it/s, loss=1.3940, acc=68.85%]
Epoch 7/10 - Train Loss: 1.3940, Train Acc: 68.85%, Test Acc: 68.43%                                                                                             
Epoch 8/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.02it/s, loss=1.3690, acc=69.28%] 
Epoch 8/10 - Train Loss: 1.3690, Train Acc: 69.28%, Test Acc: 68.79%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_MAG_epoch8.pth
Epoch 9/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.02it/s, loss=1.3280, acc=70.27%]
Epoch 9/10 - Train Loss: 1.3280, Train Acc: 70.27%, Test Acc: 68.99%                                                                                             
Epoch 10/10: 100%|█████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.03it/s, loss=1.3162, acc=70.35%] 
Epoch 10/10 - Train Loss: 1.3162, Train Acc: 70.35%, Test Acc: 69.10%                                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_MAG_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_MAG_best.pth
  Best Accuracy: 69.10%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 69.10%                                                                                                                                   
✓ Model Size: 81.44 MB
✓ Average Inference Time: 0.0161 ms
✓ FLOPs: 3.05 GFLOPs

================================================================================
COMPARISON: MAGNITUDE-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.12   |    69.1    | -9.50                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         81.44   |    81.44   | -16.25 (-16.6%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0309 |          0.0219 |     0.0161 | -0.0148                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.05   |     3.05   | -1.09 (-26.3%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS4_ResNet50_ImageNet\TS4_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================

====================================================================================================
✓ TS4_04_magnitude_pruning.py completed successfully
  Elapsed time: 1480.75 seconds
====================================================================================================


====================================================================================================
SCRIPT 5/6: TS4_05_taylor_pruning.py
====================================================================================================

====================================================================================================
RUNNING: TS4_05_taylor_pruning.py
====================================================================================================


================================================================================
TEST SCENARIO TS4: TAYLOR GRADIENT-BASED PRUNING
================================================================================
Model: ResNet50
Dataset: ImageNet
Method: Taylor (Gradient-based)
Device: cuda
Pruning Ratio: 10.0%
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
✓ Model loaded from: C:\source\checkpoints\TS4\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                                                
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0335 ms
✓ FLOPs: 4.13 GFLOPs

================================================================================
APPLYING TAYLOR GRADIENT-BASED PRUNING
================================================================================
Pruning Ratio: 10.0%
Global Pruning: True
Iterative Steps: 1
Taylor pruning uses first-order gradient information
Importance = |weight × gradient| (Taylor expansion approximation)
Using 100 calibration batches

Computing Taylor importance scores (requires gradients)...
Computing gradients:  99%|███████████████████████████████████████████████████████████████████████████████████████████████████▉ | 99/100 [19:41<00:11, 11.94s/it] 
✓ Gradients computed on 100 batches
Using GroupTaylorImportance

Initializing Torch-Pruning with Taylor importance...

Applying pruning...

✓ Taylor pruning completed
  Parameters before: 25,557,032
  Parameters after: 21,554,836
  Parameters removed: 4,002,196 (15.66%)

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.59%                                                                                                                                   
✓ Model Size: 82.41 MB
✓ Average Inference Time: 0.0204 ms
✓ FLOPs: 3.22 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [09:01<00:00, 13.54s/it, loss=1.8511, acc=59.33%] 
Epoch 1/10 - Train Loss: 1.8511, Train Acc: 59.33%, Test Acc: 72.83%                                                                                             
Epoch 2/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [09:00<00:00, 13.50s/it, loss=1.3868, acc=67.82%] 
Epoch 2/10 - Train Loss: 1.3868, Train Acc: 67.82%, Test Acc: 74.15%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_TAY_epoch2.pth
Epoch 3/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [09:00<00:00, 13.51s/it, loss=1.1796, acc=72.30%]
Epoch 3/10 - Train Loss: 1.1796, Train Acc: 72.30%, Test Acc: 75.15%                                                                                             
Epoch 4/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [09:03<00:00, 13.60s/it, loss=1.0469, acc=75.45%] 
Epoch 4/10 - Train Loss: 1.0469, Train Acc: 75.45%, Test Acc: 75.78%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_TAY_epoch4.pth
Epoch 5/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [08:54<00:00, 13.37s/it, loss=0.9439, acc=78.16%]
Epoch 5/10 - Train Loss: 0.9439, Train Acc: 78.16%, Test Acc: 76.17%                                                                                             
Epoch 6/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [08:54<00:00, 13.36s/it, loss=0.8705, acc=79.77%] 
Epoch 6/10 - Train Loss: 0.8705, Train Acc: 79.77%, Test Acc: 76.40%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_TAY_epoch6.pth
Epoch 7/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [08:55<00:00, 13.40s/it, loss=0.8218, acc=81.07%]
Epoch 7/10 - Train Loss: 0.8218, Train Acc: 81.07%, Test Acc: 76.59%                                                                                             
Epoch 8/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [08:54<00:00, 13.36s/it, loss=0.7850, acc=81.70%] 
Epoch 8/10 - Train Loss: 0.7850, Train Acc: 81.70%, Test Acc: 76.62%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_TAY_epoch8.pth
Epoch 9/10: 100%|██████████████████████████████████████████████████████████████████████████████████████| 40/40 [08:55<00:00, 13.38s/it, loss=0.7762, acc=82.40%]
Epoch 9/10 - Train Loss: 0.7762, Train Acc: 82.40%, Test Acc: 76.67%
Epoch 10/10: 100%|█████████████████████████████████████████████████████████████████████████████████████| 40/40 [08:54<00:00, 13.37s/it, loss=0.7824, acc=82.23%] 
Epoch 10/10 - Train Loss: 0.7824, Train Acc: 82.23%, Test Acc: 76.83%
  ✓ Checkpoint saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_TAY_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS4\ResNet50_ImageNet_FTAP_TAY_best.pth
  Best Accuracy: 76.83%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 76.83%                                                                                                                                   
✓ Model Size: 82.41 MB
✓ Average Inference Time: 0.0210 ms
✓ FLOPs: 3.22 GFLOPs

================================================================================
COMPARISON: TAYLOR GRADIENT-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.59   |     76.83  | -1.77                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         82.41   |     82.41  | -15.28 (-15.6%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0335 |          0.0204 |      0.021 | -0.0126                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.22   |      3.22  | -0.91 (-22.1%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS4_ResNet50_ImageNet\TS4_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================

====================================================================================================
✓ TS4_05_taylor_pruning.py completed successfully
  Elapsed time: 13103.33 seconds
====================================================================================================