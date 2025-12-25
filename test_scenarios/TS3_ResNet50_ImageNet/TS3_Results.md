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
  conv1: coverage shape torch.Size([64]), min=-0.720004, max=1.000000, mean=0.007926
  layer1.0.conv1: coverage shape torch.Size([64]), min=-0.680586, max=1.000000, mean=-0.109383
  layer1.0.conv2: coverage shape torch.Size([64]), min=-2.973305, max=1.000000, mean=0.120890
  layer1.0.conv3: coverage shape torch.Size([256]), min=-0.757535, max=1.000000, mean=-0.036517
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=-1.998923, max=1.000000, mean=-0.093963
  layer1.1.conv1: coverage shape torch.Size([64]), min=-1.896286, max=1.000000, mean=-0.059186
  layer1.1.conv2: coverage shape torch.Size([64]), min=-1.531336, max=1.000000, mean=-0.103501
  layer1.1.conv3: coverage shape torch.Size([256]), min=-1.593806, max=1.000000, mean=-0.106017
  layer1.2.conv1: coverage shape torch.Size([64]), min=-0.881639, max=1.000000, mean=-0.116242
  layer1.2.conv2: coverage shape torch.Size([64]), min=-1.831529, max=1.000000, mean=-0.215139
  layer1.2.conv3: coverage shape torch.Size([256]), min=-1.092057, max=1.000000, mean=-0.034290
  layer2.0.conv1: coverage shape torch.Size([128]), min=-1.019418, max=1.000000, mean=-0.088866
  layer2.0.conv2: coverage shape torch.Size([128]), min=-1.424594, max=1.000000, mean=-0.151112
  layer2.0.conv3: coverage shape torch.Size([512]), min=-0.669595, max=1.000000, mean=0.016052
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=-0.821532, max=1.000000, mean=-0.056077
  layer2.1.conv1: coverage shape torch.Size([128]), min=-0.985447, max=1.000000, mean=-0.094155
  layer2.1.conv2: coverage shape torch.Size([128]), min=-0.743826, max=1.000000, mean=0.069059
  layer2.1.conv3: coverage shape torch.Size([512]), min=-0.911751, max=1.000000, mean=-0.044249
  layer2.2.conv1: coverage shape torch.Size([128]), min=-1.438058, max=1.000000, mean=-0.005520
  layer2.2.conv2: coverage shape torch.Size([128]), min=-1.097991, max=1.000000, mean=-0.161955
  layer2.2.conv3: coverage shape torch.Size([512]), min=-0.948909, max=1.000000, mean=-0.002189
  layer2.3.conv1: coverage shape torch.Size([128]), min=-0.600136, max=1.000000, mean=-0.044146
  layer2.3.conv2: coverage shape torch.Size([128]), min=-0.854221, max=1.000000, mean=-0.127514
  layer2.3.conv3: coverage shape torch.Size([512]), min=-1.123452, max=1.000000, mean=-0.051435
  layer3.0.conv1: coverage shape torch.Size([256]), min=-1.501619, max=1.000000, mean=-0.196377
  layer3.0.conv2: coverage shape torch.Size([256]), min=-1.561093, max=1.000000, mean=-0.092558
  layer3.0.conv3: coverage shape torch.Size([1024]), min=-1.175297, max=1.000000, mean=-0.020690
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=-1.308764, max=1.000000, mean=-0.148950
  layer3.1.conv1: coverage shape torch.Size([256]), min=-1.818992, max=1.000000, mean=-0.068391
  layer3.1.conv2: coverage shape torch.Size([256]), min=-2.274943, max=1.000000, mean=-0.071647
  layer3.1.conv3: coverage shape torch.Size([1024]), min=-1.966147, max=1.000000, mean=-0.191713
  layer3.2.conv1: coverage shape torch.Size([256]), min=-1.419641, max=1.000000, mean=-0.249070
  layer3.2.conv2: coverage shape torch.Size([256]), min=-1.241966, max=1.000000, mean=-0.199610
  layer3.2.conv3: coverage shape torch.Size([1024]), min=-1.188899, max=1.000000, mean=-0.051525
  layer3.3.conv1: coverage shape torch.Size([256]), min=-1.146034, max=1.000000, mean=-0.127977
  layer3.3.conv2: coverage shape torch.Size([256]), min=-0.716645, max=1.000000, mean=-0.088994
  layer3.3.conv3: coverage shape torch.Size([1024]), min=-1.390667, max=1.000000, mean=-0.088042
  layer3.4.conv1: coverage shape torch.Size([256]), min=-0.843379, max=1.000000, mean=-0.256466
  layer3.4.conv2: coverage shape torch.Size([256]), min=-1.097269, max=1.000000, mean=-0.255354
  layer3.4.conv3: coverage shape torch.Size([1024]), min=-0.732784, max=1.000000, mean=-0.021692
  layer3.5.conv1: coverage shape torch.Size([256]), min=-1.072305, max=1.000000, mean=-0.174813
  layer3.5.conv2: coverage shape torch.Size([256]), min=-1.386566, max=1.000000, mean=-0.231804
  layer3.5.conv3: coverage shape torch.Size([1024]), min=-1.614175, max=1.000000, mean=-0.264913
  layer4.0.conv1: coverage shape torch.Size([512]), min=-1.413600, max=1.000000, mean=-0.604115
  layer4.0.conv2: coverage shape torch.Size([512]), min=-2.155089, max=1.000000, mean=-0.413041
  layer4.0.conv3: coverage shape torch.Size([2048]), min=-0.937507, max=1.000000, mean=-0.086188
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=-1.224976, max=1.000000, mean=-0.160217
  layer4.1.conv1: coverage shape torch.Size([512]), min=-1.798767, max=1.000000, mean=-0.243097
  layer4.1.conv2: coverage shape torch.Size([512]), min=-1.212782, max=1.000000, mean=-0.206794
  layer4.1.conv3: coverage shape torch.Size([2048]), min=-0.725702, max=1.000000, mean=-0.146485
  layer4.2.conv1: coverage shape torch.Size([512]), min=-1.424415, max=1.000000, mean=-0.225094
  layer4.2.conv2: coverage shape torch.Size([512]), min=-2.208051, max=1.000000, mean=-0.299194
  layer4.2.conv3: coverage shape torch.Size([2048]), min=-2.460291, max=1.000000, mean=-0.836446
  fc: coverage shape torch.Size([1000]), min=-0.803133, max=1.000000, mean=0.000401

------------------------------------------------------------
Coverage Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Coverage - Min: -0.720004, Max: 1.000000, Mean: 0.007926
  Zero coverage neurons: 0

layer1.0.conv1:
  Channels: 64
  Coverage - Min: -0.680586, Max: 1.000000, Mean: -0.109383
  Zero coverage neurons: 0

layer1.0.conv2:
  Channels: 64
  Coverage - Min: -2.973305, Max: 1.000000, Mean: 0.120890
  Zero coverage neurons: 0

layer1.0.conv3:
  Channels: 256
  Coverage - Min: -0.757535, Max: 1.000000, Mean: -0.036517
  Zero coverage neurons: 0

layer1.0.downsample.0:
  Channels: 256
  Coverage - Min: -1.998923, Max: 1.000000, Mean: -0.093963
  Zero coverage neurons: 0

layer1.1.conv1:
  Channels: 64
  Coverage - Min: -1.896286, Max: 1.000000, Mean: -0.059186
  Zero coverage neurons: 0

layer1.1.conv2:
  Channels: 64
  Coverage - Min: -1.531336, Max: 1.000000, Mean: -0.103501
  Zero coverage neurons: 0

layer1.1.conv3:
  Channels: 256
  Coverage - Min: -1.593806, Max: 1.000000, Mean: -0.106017
  Zero coverage neurons: 0

layer1.2.conv1:
  Channels: 64
  Coverage - Min: -0.881639, Max: 1.000000, Mean: -0.116242
  Zero coverage neurons: 0

layer1.2.conv2:
  Channels: 64
  Coverage - Min: -1.831529, Max: 1.000000, Mean: -0.215139
  Zero coverage neurons: 0

layer1.2.conv3:
  Channels: 256
  Coverage - Min: -1.092057, Max: 1.000000, Mean: -0.034290
  Zero coverage neurons: 0

layer2.0.conv1:
  Channels: 128
  Coverage - Min: -1.019418, Max: 1.000000, Mean: -0.088866
  Zero coverage neurons: 0

layer2.0.conv2:
  Channels: 128
  Coverage - Min: -1.424594, Max: 1.000000, Mean: -0.151112
  Zero coverage neurons: 0

layer2.0.conv3:
  Channels: 512
  Coverage - Min: -0.669595, Max: 1.000000, Mean: 0.016052
  Zero coverage neurons: 0

layer2.0.downsample.0:
  Channels: 512
  Coverage - Min: -0.821532, Max: 1.000000, Mean: -0.056077
  Zero coverage neurons: 0

layer2.1.conv1:
  Channels: 128
  Coverage - Min: -0.985447, Max: 1.000000, Mean: -0.094155
  Zero coverage neurons: 0

layer2.1.conv2:
  Channels: 128
  Coverage - Min: -0.743826, Max: 1.000000, Mean: 0.069059
  Zero coverage neurons: 0

layer2.1.conv3:
  Channels: 512
  Coverage - Min: -0.911751, Max: 1.000000, Mean: -0.044249
  Zero coverage neurons: 0

layer2.2.conv1:
  Channels: 128
  Coverage - Min: -1.438058, Max: 1.000000, Mean: -0.005520
  Zero coverage neurons: 0

layer2.2.conv2:
  Channels: 128
  Coverage - Min: -1.097991, Max: 1.000000, Mean: -0.161955
  Zero coverage neurons: 0

layer2.2.conv3:
  Channels: 512
  Coverage - Min: -0.948909, Max: 1.000000, Mean: -0.002189
  Zero coverage neurons: 0

layer2.3.conv1:
  Channels: 128
  Coverage - Min: -0.600136, Max: 1.000000, Mean: -0.044146
  Zero coverage neurons: 0

layer2.3.conv2:
  Channels: 128
  Coverage - Min: -0.854221, Max: 1.000000, Mean: -0.127514
  Zero coverage neurons: 0

layer2.3.conv3:
  Channels: 512
  Coverage - Min: -1.123452, Max: 1.000000, Mean: -0.051435
  Zero coverage neurons: 0

layer3.0.conv1:
  Channels: 256
  Coverage - Min: -1.501619, Max: 1.000000, Mean: -0.196377
  Zero coverage neurons: 0

layer3.0.conv2:
  Channels: 256
  Coverage - Min: -1.561093, Max: 1.000000, Mean: -0.092558
  Zero coverage neurons: 0

layer3.0.conv3:
  Channels: 1024
  Coverage - Min: -1.175297, Max: 1.000000, Mean: -0.020690
  Zero coverage neurons: 0

layer3.0.downsample.0:
  Channels: 1024
  Coverage - Min: -1.308764, Max: 1.000000, Mean: -0.148950
  Zero coverage neurons: 0

layer3.1.conv1:
  Channels: 256
  Coverage - Min: -1.818992, Max: 1.000000, Mean: -0.068391
  Zero coverage neurons: 0

layer3.1.conv2:
  Channels: 256
  Coverage - Min: -2.274943, Max: 1.000000, Mean: -0.071647
  Zero coverage neurons: 0

layer3.1.conv3:
  Channels: 1024
  Coverage - Min: -1.966147, Max: 1.000000, Mean: -0.191713
  Zero coverage neurons: 0

layer3.2.conv1:
  Channels: 256
  Coverage - Min: -1.419641, Max: 1.000000, Mean: -0.249070
  Zero coverage neurons: 0

layer3.2.conv2:
  Channels: 256
  Coverage - Min: -1.241966, Max: 1.000000, Mean: -0.199610
  Zero coverage neurons: 0

layer3.2.conv3:
  Channels: 1024
  Coverage - Min: -1.188899, Max: 1.000000, Mean: -0.051525
  Zero coverage neurons: 0

layer3.3.conv1:
  Channels: 256
  Coverage - Min: -1.146034, Max: 1.000000, Mean: -0.127977
  Zero coverage neurons: 0

layer3.3.conv2:
  Channels: 256
  Coverage - Min: -0.716645, Max: 1.000000, Mean: -0.088994
  Zero coverage neurons: 0

layer3.3.conv3:
  Channels: 1024
  Coverage - Min: -1.390667, Max: 1.000000, Mean: -0.088042
  Zero coverage neurons: 0

layer3.4.conv1:
  Channels: 256
  Coverage - Min: -0.843379, Max: 1.000000, Mean: -0.256466
  Zero coverage neurons: 0

layer3.4.conv2:
  Channels: 256
  Coverage - Min: -1.097269, Max: 1.000000, Mean: -0.255354
  Zero coverage neurons: 0

layer3.4.conv3:
  Channels: 1024
  Coverage - Min: -0.732784, Max: 1.000000, Mean: -0.021692
  Zero coverage neurons: 0

layer3.5.conv1:
  Channels: 256
  Coverage - Min: -1.072305, Max: 1.000000, Mean: -0.174813
  Zero coverage neurons: 0

layer3.5.conv2:
  Channels: 256
  Coverage - Min: -1.386566, Max: 1.000000, Mean: -0.231804
  Zero coverage neurons: 0

layer3.5.conv3:
  Channels: 1024
  Coverage - Min: -1.614175, Max: 1.000000, Mean: -0.264913
  Zero coverage neurons: 0

layer4.0.conv1:
  Channels: 512
  Coverage - Min: -1.413600, Max: 1.000000, Mean: -0.604115
  Zero coverage neurons: 0

layer4.0.conv2:
  Channels: 512
  Coverage - Min: -2.155089, Max: 1.000000, Mean: -0.413041
  Zero coverage neurons: 0

layer4.0.conv3:
  Channels: 2048
  Coverage - Min: -0.937507, Max: 1.000000, Mean: -0.086188
  Zero coverage neurons: 0

layer4.0.downsample.0:
  Channels: 2048
  Coverage - Min: -1.224976, Max: 1.000000, Mean: -0.160217
  Zero coverage neurons: 0

layer4.1.conv1:
  Channels: 512
  Coverage - Min: -1.798767, Max: 1.000000, Mean: -0.243097
  Zero coverage neurons: 0

layer4.1.conv2:
  Channels: 512
  Coverage - Min: -1.212782, Max: 1.000000, Mean: -0.206794
  Zero coverage neurons: 0

layer4.1.conv3:
  Channels: 2048
  Coverage - Min: -0.725702, Max: 1.000000, Mean: -0.146485
  Zero coverage neurons: 0

layer4.2.conv1:
  Channels: 512
  Coverage - Min: -1.424415, Max: 1.000000, Mean: -0.225094
  Zero coverage neurons: 0

layer4.2.conv2:
  Channels: 512
  Coverage - Min: -2.208051, Max: 1.000000, Mean: -0.299194
  Zero coverage neurons: 0

layer4.2.conv3:
  Channels: 2048
  Coverage - Min: -2.460291, Max: 1.000000, Mean: -0.836446
  Zero coverage neurons: 0

fc:
  Channels: 1000
  Coverage - Min: -0.803133, Max: 1.000000, Mean: 0.000401
  Zero coverage neurons: 0
============================================================


Pruning Results:
  Parameters before: 25,557,032
  Parameters after:  20,837,770
  Parameters removed: 4,719,262 (18.47%)

############################################################
Pruning Complete
############################################################
Initial parameters:  25,557,032
Final parameters:    20,837,770
Total removed:       4,719,262 (18.47%)
Target pruning ratio: 10.00%
✓ Neuron Coverage pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.17%                                                                                                                  
✓ Model Size: 79.67 MB
✓ Average Inference Time: 1.3353 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.20s/it, loss=3.2464, acc=42.35%] 
Epoch 1/10 - Train Loss: 3.2464, Train Acc: 42.35%, Test Acc: 60.88%                                                                            
Epoch 2/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:27<00:00,  2.18s/it, loss=1.9860, acc=56.79%] 
Epoch 2/10 - Train Loss: 1.9860, Train Acc: 56.79%, Test Acc: 65.39%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_epoch2.pth
Epoch 3/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:27<00:00,  2.18s/it, loss=1.6419, acc=63.29%]
Epoch 3/10 - Train Loss: 1.6419, Train Acc: 63.29%, Test Acc: 67.94%                                                                            
Epoch 4/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:27<00:00,  2.18s/it, loss=1.4585, acc=67.60%] 
Epoch 4/10 - Train Loss: 1.4585, Train Acc: 67.60%, Test Acc: 69.41%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_epoch4.pth
Epoch 5/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:27<00:00,  2.18s/it, loss=1.2849, acc=71.02%]
Epoch 5/10 - Train Loss: 1.2849, Train Acc: 71.02%, Test Acc: 70.57%                                                                            
Epoch 6/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:27<00:00,  2.18s/it, loss=1.1503, acc=73.39%] 
Epoch 6/10 - Train Loss: 1.1503, Train Acc: 73.39%, Test Acc: 71.08%
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_epoch6.pth
Epoch 7/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:27<00:00,  2.18s/it, loss=1.1266, acc=74.44%]
Epoch 7/10 - Train Loss: 1.1266, Train Acc: 74.44%, Test Acc: 71.78%                                                                            
Epoch 8/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:27<00:00,  2.19s/it, loss=1.0644, acc=75.29%] 
Epoch 8/10 - Train Loss: 1.0644, Train Acc: 75.29%, Test Acc: 71.87%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_epoch8.pth
Epoch 9/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:27<00:00,  2.19s/it, loss=1.0236, acc=76.78%]
Epoch 9/10 - Train Loss: 1.0236, Train Acc: 76.78%, Test Acc: 72.04%
Epoch 10/10: 100%|████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.20s/it, loss=1.0047, acc=77.16%] 
Epoch 10/10 - Train Loss: 1.0047, Train Acc: 77.16%, Test Acc: 72.13%                                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_best.pth
  Best Accuracy: 72.13%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 72.13%                                                                                                                  
✓ Model Size: 79.67 MB
✓ Average Inference Time: 1.3348 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
COMPARISON: NEURON COVERAGE PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.17   |    72.13   | -6.46                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         79.67   |    79.67   | -18.02 (-18.4%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0286 |          1.3353 |     1.3348 | +1.3062                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.34   |     3.34   | -0.79 (-19.1%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS3_ResNet50_ImageNet\TS3_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================

====================================================================================================
✓ TS3_02_coverage_pruning.py completed successfully
  Elapsed time: 1944.61 seconds
====================================================================================================


====================================================================================================
SCRIPT 3/6: TS3_03_wanda_pruning.py
====================================================================================================

====================================================================================================
RUNNING: TS3_03_wanda_pruning.py
====================================================================================================


================================================================================
TEST SCENARIO TS3: WANDA PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS3\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                               
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0297 ms
✓ FLOPs: 4.13 GFLOPs

================================================================================
APPLYING WANDA PRUNING
================================================================================
Pruning Ratio: 10.0%
Global Pruning: False
Iterative Steps: 1
WANDA combines weight magnitude with activation importance
Using 100 calibration batches
Using WANDA importance (Weight × Activation)

Initializing Torch-Pruning with:
  Importance method: wanda
  Pruning ratio: 10.00%
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
  conv1: coverage shape torch.Size([64]), min=0.000004, max=0.443245, mean=0.030429
  layer1.0.conv1: coverage shape torch.Size([64]), min=0.865630, max=44.575459, mean=9.885008
  layer1.0.conv2: coverage shape torch.Size([64]), min=0.057777, max=25.868952, mean=3.915439
  layer1.0.conv3: coverage shape torch.Size([256]), min=0.000000, max=10.938977, mean=2.151672
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=0.000000, max=61.220440, mean=5.205881
  layer1.1.conv1: coverage shape torch.Size([64]), min=0.000000, max=6.717833, mean=1.259037
  layer1.1.conv2: coverage shape torch.Size([64]), min=0.127224, max=6.329598, mean=1.655253
  layer1.1.conv3: coverage shape torch.Size([256]), min=0.001577, max=7.035923, mean=1.304046
  layer1.2.conv1: coverage shape torch.Size([64]), min=0.032990, max=4.760635, mean=1.476930
  layer1.2.conv2: coverage shape torch.Size([64]), min=0.068031, max=6.588127, mean=1.332326
  layer1.2.conv3: coverage shape torch.Size([256]), min=0.002880, max=5.846415, mean=0.939725
  layer2.0.conv1: coverage shape torch.Size([128]), min=0.003974, max=5.203104, mean=1.523301
  layer2.0.conv2: coverage shape torch.Size([128]), min=0.010778, max=6.546328, mean=1.468158
  layer2.0.conv3: coverage shape torch.Size([512]), min=0.000002, max=8.117187, mean=1.116648
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=0.000001, max=8.571273, mean=1.163367
  layer2.1.conv1: coverage shape torch.Size([128]), min=0.079455, max=7.677267, mean=2.209382
  layer2.1.conv2: coverage shape torch.Size([128]), min=0.016673, max=10.108121, mean=1.949660
  layer2.1.conv3: coverage shape torch.Size([512]), min=0.000254, max=2.399243, mean=0.467550
  layer2.2.conv1: coverage shape torch.Size([128]), min=0.004266, max=5.483771, mean=1.217956
  layer2.2.conv2: coverage shape torch.Size([128]), min=0.009791, max=6.195937, mean=1.373424
  layer2.2.conv3: coverage shape torch.Size([512]), min=0.000742, max=3.078044, mean=0.579468
  layer2.3.conv1: coverage shape torch.Size([128]), min=0.020459, max=6.827806, mean=1.273086
  layer2.3.conv2: coverage shape torch.Size([128]), min=0.014054, max=5.929280, mean=1.380274
  layer2.3.conv3: coverage shape torch.Size([512]), min=0.000620, max=3.385374, mean=0.602324
  layer3.0.conv1: coverage shape torch.Size([256]), min=0.001778, max=6.382233, mean=1.402707
  layer3.0.conv2: coverage shape torch.Size([256]), min=0.003137, max=4.452528, mean=0.891638
  layer3.0.conv3: coverage shape torch.Size([1024]), min=0.000289, max=4.705019, mean=0.894972
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=0.000052, max=4.074196, mean=0.831523
  layer3.1.conv1: coverage shape torch.Size([256]), min=0.000947, max=9.747677, mean=1.413814
  layer3.1.conv2: coverage shape torch.Size([256]), min=0.001789, max=11.068872, mean=1.027918
  layer3.1.conv3: coverage shape torch.Size([1024]), min=0.000099, max=2.091313, mean=0.318415
  layer3.2.conv1: coverage shape torch.Size([256]), min=0.002474, max=5.961559, mean=1.283086
  layer3.2.conv2: coverage shape torch.Size([256]), min=0.001378, max=3.087141, mean=0.708575
  layer3.2.conv3: coverage shape torch.Size([1024]), min=0.000332, max=2.029816, mean=0.287176
  layer3.3.conv1: coverage shape torch.Size([256]), min=0.003799, max=5.973809, mean=1.164546
  layer3.3.conv2: coverage shape torch.Size([256]), min=0.008534, max=5.394253, mean=0.820155
  layer3.3.conv3: coverage shape torch.Size([1024]), min=0.000243, max=1.959607, mean=0.335717
  layer3.4.conv1: coverage shape torch.Size([256]), min=0.015064, max=3.747923, mean=1.144357
  layer3.4.conv2: coverage shape torch.Size([256]), min=0.006332, max=3.057916, mean=0.864061
  layer3.4.conv3: coverage shape torch.Size([1024]), min=0.000012, max=2.885544, mean=0.299634
  layer3.5.conv1: coverage shape torch.Size([256]), min=0.001733, max=5.947008, mean=1.248163
  layer3.5.conv2: coverage shape torch.Size([256]), min=0.001790, max=3.802512, mean=0.846158
  layer3.5.conv3: coverage shape torch.Size([1024]), min=0.000609, max=2.125455, mean=0.429316
  layer4.0.conv1: coverage shape torch.Size([512]), min=0.043782, max=3.700032, mean=1.627956
  layer4.0.conv2: coverage shape torch.Size([512]), min=0.005087, max=4.816246, mean=0.952102
  layer4.0.conv3: coverage shape torch.Size([2048]), min=0.000057, max=5.297569, mean=0.589436
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=0.000481, max=3.841373, mean=0.576780
  layer4.1.conv1: coverage shape torch.Size([512]), min=0.101079, max=9.320873, mean=1.404253
  layer4.1.conv2: coverage shape torch.Size([512]), min=0.000661, max=5.465142, mean=1.052096
  layer4.1.conv3: coverage shape torch.Size([2048]), min=0.000656, max=2.934222, mean=0.447105
  layer4.2.conv1: coverage shape torch.Size([512]), min=0.086660, max=7.496107, mean=1.340167
  layer4.2.conv2: coverage shape torch.Size([512]), min=0.047703, max=4.693573, mean=0.663996
  layer4.2.conv3: coverage shape torch.Size([2048]), min=0.000014, max=1.060815, mean=0.361376
  fc: coverage shape torch.Size([1000]), min=0.000117, max=0.280146, mean=0.059986

------------------------------------------------------------
Activation Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Activation - Min: 0.000004, Max: 0.443245, Mean: 0.030429

layer1.0.conv1:
  Channels: 64
  Activation - Min: 0.865630, Max: 44.575459, Mean: 9.885008

layer1.0.conv2:
  Channels: 64
  Activation - Min: 0.057777, Max: 25.868952, Mean: 3.915439

layer1.0.conv3:
  Channels: 256
  Activation - Min: 0.000000, Max: 10.938977, Mean: 2.151672

layer1.0.downsample.0:
  Channels: 256
  Activation - Min: 0.000000, Max: 61.220440, Mean: 5.205881

layer1.1.conv1:
  Channels: 64
  Activation - Min: 0.000000, Max: 6.717833, Mean: 1.259037

layer1.1.conv2:
  Channels: 64
  Activation - Min: 0.127224, Max: 6.329598, Mean: 1.655253

layer1.1.conv3:
  Channels: 256
  Activation - Min: 0.001577, Max: 7.035923, Mean: 1.304046

layer1.2.conv1:
  Channels: 64
  Activation - Min: 0.032990, Max: 4.760635, Mean: 1.476930

layer1.2.conv2:
  Channels: 64
  Activation - Min: 0.068031, Max: 6.588127, Mean: 1.332326

layer1.2.conv3:
  Channels: 256
  Activation - Min: 0.002880, Max: 5.846415, Mean: 0.939725

layer2.0.conv1:
  Channels: 128
  Activation - Min: 0.003974, Max: 5.203104, Mean: 1.523301

layer2.0.conv2:
  Channels: 128
  Activation - Min: 0.010778, Max: 6.546328, Mean: 1.468158

layer2.0.conv3:
  Channels: 512
  Activation - Min: 0.000002, Max: 8.117187, Mean: 1.116648

layer2.0.downsample.0:
  Channels: 512
  Activation - Min: 0.000001, Max: 8.571273, Mean: 1.163367

layer2.1.conv1:
  Channels: 128
  Activation - Min: 0.079455, Max: 7.677267, Mean: 2.209382

layer2.1.conv2:
  Channels: 128
  Activation - Min: 0.016673, Max: 10.108121, Mean: 1.949660

layer2.1.conv3:
  Channels: 512
  Activation - Min: 0.000254, Max: 2.399243, Mean: 0.467550

layer2.2.conv1:
  Channels: 128
  Activation - Min: 0.004266, Max: 5.483771, Mean: 1.217956

layer2.2.conv2:
  Channels: 128
  Activation - Min: 0.009791, Max: 6.195937, Mean: 1.373424

layer2.2.conv3:
  Channels: 512
  Activation - Min: 0.000742, Max: 3.078044, Mean: 0.579468

layer2.3.conv1:
  Channels: 128
  Activation - Min: 0.020459, Max: 6.827806, Mean: 1.273086

layer2.3.conv2:
  Channels: 128
  Activation - Min: 0.014054, Max: 5.929280, Mean: 1.380274

layer2.3.conv3:
  Channels: 512
  Activation - Min: 0.000620, Max: 3.385374, Mean: 0.602324

layer3.0.conv1:
  Channels: 256
  Activation - Min: 0.001778, Max: 6.382233, Mean: 1.402707

layer3.0.conv2:
  Channels: 256
  Activation - Min: 0.003137, Max: 4.452528, Mean: 0.891638

layer3.0.conv3:
  Channels: 1024
  Activation - Min: 0.000289, Max: 4.705019, Mean: 0.894972

layer3.0.downsample.0:
  Channels: 1024
  Activation - Min: 0.000052, Max: 4.074196, Mean: 0.831523

layer3.1.conv1:
  Channels: 256
  Activation - Min: 0.000947, Max: 9.747677, Mean: 1.413814

layer3.1.conv2:
  Channels: 256
  Activation - Min: 0.001789, Max: 11.068872, Mean: 1.027918

layer3.1.conv3:
  Channels: 1024
  Activation - Min: 0.000099, Max: 2.091313, Mean: 0.318415

layer3.2.conv1:
  Channels: 256
  Activation - Min: 0.002474, Max: 5.961559, Mean: 1.283086

layer3.2.conv2:
  Channels: 256
  Activation - Min: 0.001378, Max: 3.087141, Mean: 0.708575

layer3.2.conv3:
  Channels: 1024
  Activation - Min: 0.000332, Max: 2.029816, Mean: 0.287176

layer3.3.conv1:
  Channels: 256
  Activation - Min: 0.003799, Max: 5.973809, Mean: 1.164546

layer3.3.conv2:
  Channels: 256
  Activation - Min: 0.008534, Max: 5.394253, Mean: 0.820155

layer3.3.conv3:
  Channels: 1024
  Activation - Min: 0.000243, Max: 1.959607, Mean: 0.335717

layer3.4.conv1:
  Channels: 256
  Activation - Min: 0.015064, Max: 3.747923, Mean: 1.144357

layer3.4.conv2:
  Channels: 256
  Activation - Min: 0.006332, Max: 3.057916, Mean: 0.864061

layer3.4.conv3:
  Channels: 1024
  Activation - Min: 0.000012, Max: 2.885544, Mean: 0.299634

layer3.5.conv1:
  Channels: 256
  Activation - Min: 0.001733, Max: 5.947008, Mean: 1.248163

layer3.5.conv2:
  Channels: 256
  Activation - Min: 0.001790, Max: 3.802512, Mean: 0.846158

layer3.5.conv3:
  Channels: 1024
  Activation - Min: 0.000609, Max: 2.125455, Mean: 0.429316

layer4.0.conv1:
  Channels: 512
  Activation - Min: 0.043782, Max: 3.700032, Mean: 1.627956

layer4.0.conv2:
  Channels: 512
  Activation - Min: 0.005087, Max: 4.816246, Mean: 0.952102

layer4.0.conv3:
  Channels: 2048
  Activation - Min: 0.000057, Max: 5.297569, Mean: 0.589436

layer4.0.downsample.0:
  Channels: 2048
  Activation - Min: 0.000481, Max: 3.841373, Mean: 0.576780

layer4.1.conv1:
  Channels: 512
  Activation - Min: 0.101079, Max: 9.320873, Mean: 1.404253

layer4.1.conv2:
  Channels: 512
  Activation - Min: 0.000661, Max: 5.465142, Mean: 1.052096

layer4.1.conv3:
  Channels: 2048
  Activation - Min: 0.000656, Max: 2.934222, Mean: 0.447105

layer4.2.conv1:
  Channels: 512
  Activation - Min: 0.086660, Max: 7.496107, Mean: 1.340167

layer4.2.conv2:
  Channels: 512
  Activation - Min: 0.047703, Max: 4.693573, Mean: 0.663996

layer4.2.conv3:
  Channels: 2048
  Activation - Min: 0.000014, Max: 1.060815, Mean: 0.361376

fc:
  Channels: 1000
  Activation - Min: 0.000117, Max: 0.280146, Mean: 0.059986
============================================================


Pruning Results:
  Parameters before: 25,557,032
  Parameters after:  20,837,770
  Parameters removed: 4,719,262 (18.47%)

############################################################
Pruning Complete
############################################################
Initial parameters:  25,557,032
Final parameters:    20,837,770
Total removed:       4,719,262 (18.47%)
Target pruning ratio: 10.00%
✓ Wanda pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.47%                                                                                                                  
✓ Model Size: 79.67 MB
✓ Average Inference Time: 1.3394 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:29<00:00,  2.23s/it, loss=2.1537, acc=54.83%] 
Epoch 1/10 - Train Loss: 2.1537, Train Acc: 54.83%, Test Acc: 67.95%                                                                            
Epoch 2/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.21s/it, loss=1.6199, acc=63.34%] 
Epoch 2/10 - Train Loss: 1.6199, Train Acc: 63.34%, Test Acc: 70.26%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_epoch2.pth
Epoch 3/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.21s/it, loss=1.3720, acc=68.51%]
Epoch 3/10 - Train Loss: 1.3720, Train Acc: 68.51%, Test Acc: 71.80%                                                                            
Epoch 4/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.21s/it, loss=1.2164, acc=72.06%] 
Epoch 4/10 - Train Loss: 1.2164, Train Acc: 72.06%, Test Acc: 72.77%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_epoch4.pth
Epoch 5/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.21s/it, loss=1.1151, acc=74.11%]
Epoch 5/10 - Train Loss: 1.1151, Train Acc: 74.11%, Test Acc: 73.53%                                                                            
Epoch 6/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.21s/it, loss=1.0112, acc=76.33%] 
Epoch 6/10 - Train Loss: 1.0112, Train Acc: 76.33%, Test Acc: 73.93%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_epoch6.pth
Epoch 7/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.21s/it, loss=0.9475, acc=78.42%]
Epoch 7/10 - Train Loss: 0.9475, Train Acc: 78.42%, Test Acc: 74.23%                                                                            
Epoch 8/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.21s/it, loss=0.8886, acc=79.28%] 
Epoch 8/10 - Train Loss: 0.8886, Train Acc: 79.28%, Test Acc: 74.33%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_epoch8.pth
Epoch 9/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.21s/it, loss=0.8588, acc=80.09%]
Epoch 9/10 - Train Loss: 0.8588, Train Acc: 80.09%, Test Acc: 74.49%                                                                            
Epoch 10/10: 100%|████████████████████████████████████████████████████████████████████| 40/40 [01:28<00:00,  2.21s/it, loss=0.8425, acc=80.30%] 
Epoch 10/10 - Train Loss: 0.8425, Train Acc: 80.30%, Test Acc: 74.50%                                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_best.pth
  Best Accuracy: 74.50%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 74.50%                                                                                                                  
✓ Model Size: 79.67 MB
✓ Average Inference Time: 1.3399 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
COMPARISON: WANDA PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.47   |    74.5    | -4.09                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         79.67   |    79.67   | -18.02 (-18.4%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0297 |          1.3394 |     1.3399 | +1.3102                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.34   |     3.34   | -0.79 (-19.1%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS3_ResNet50_ImageNet\TS3_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================

====================================================================================================
✓ TS3_03_wanda_pruning.py completed successfully
  Elapsed time: 2043.07 seconds
====================================================================================================


====================================================================================================
SCRIPT 4/6: TS3_04_magnitude_pruning.py
====================================================================================================

====================================================================================================
RUNNING: TS3_04_magnitude_pruning.py
====================================================================================================


================================================================================
TEST SCENARIO TS3: MAGNITUDE-BASED PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS3\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                               
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0294 ms
✓ FLOPs: 4.13 GFLOPs

================================================================================
APPLYING MAGNITUDE-BASED PRUNING
================================================================================
Pruning Ratio: 10.0%
Global Pruning: False
Iterative Steps: 1
Magnitude pruning uses L2 norm of weights to determine importance
This is Torch-Pruning's most robust baseline method
Using magnitude-based importance (L2 norm)

Initializing Torch-Pruning with:
  Importance method: magnitude
  Pruning ratio: 10.00%
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

Pruning Results:
  Parameters before: 25,557,032
  Parameters after:  20,837,770
  Parameters removed: 4,719,262 (18.47%)

############################################################
Pruning Complete
############################################################
Initial parameters:  25,557,032
Final parameters:    20,837,770
Total removed:       4,719,262 (18.47%)
Target pruning ratio: 10.00%
✓ Magnitude-based pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.36%                                                                                                                  
✓ Model Size: 79.67 MB
✓ Average Inference Time: 0.0260 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [03:56<00:00,  5.92s/it, loss=2.2211, acc=55.94%] 
Epoch 1/10 - Train Loss: 2.2211, Train Acc: 55.94%, Test Acc: 70.14%                                                                            
Epoch 2/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [04:00<00:00,  6.02s/it, loss=1.5628, acc=65.37%] 
Epoch 2/10 - Train Loss: 1.5628, Train Acc: 65.37%, Test Acc: 72.55%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_epoch2.pth
Epoch 3/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [03:59<00:00,  6.00s/it, loss=1.3601, acc=69.69%]
Epoch 3/10 - Train Loss: 1.3601, Train Acc: 69.69%, Test Acc: 73.63%                                                                            
Epoch 4/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [04:00<00:00,  6.01s/it, loss=1.1720, acc=73.29%] 
Epoch 4/10 - Train Loss: 1.1720, Train Acc: 73.29%, Test Acc: 74.36%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_epoch4.pth
Epoch 5/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [04:00<00:00,  6.01s/it, loss=1.1013, acc=75.21%]
Epoch 5/10 - Train Loss: 1.1013, Train Acc: 75.21%, Test Acc: 74.48%                                                                            
Epoch 6/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [04:00<00:00,  6.01s/it, loss=1.0151, acc=76.78%] 
Epoch 6/10 - Train Loss: 1.0151, Train Acc: 76.78%, Test Acc: 75.03%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_epoch6.pth
Epoch 7/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [03:59<00:00,  5.98s/it, loss=0.9576, acc=77.90%]
Epoch 7/10 - Train Loss: 0.9576, Train Acc: 77.90%, Test Acc: 75.42%                                                                            
Epoch 8/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [04:00<00:00,  6.01s/it, loss=0.8888, acc=79.10%] 
Epoch 8/10 - Train Loss: 0.8888, Train Acc: 79.10%, Test Acc: 75.52%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_epoch8.pth
Epoch 9/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [03:59<00:00,  6.00s/it, loss=0.8625, acc=79.64%]
Epoch 9/10 - Train Loss: 0.8625, Train Acc: 79.64%, Test Acc: 75.60%
Epoch 10/10: 100%|████████████████████████████████████████████████████████████████████| 40/40 [04:00<00:00,  6.00s/it, loss=0.8809, acc=80.11%] 
Epoch 10/10 - Train Loss: 0.8809, Train Acc: 80.11%, Test Acc: 75.55%                                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_best.pth
  Best Accuracy: 75.60%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 75.55%                                                                                                                  
✓ Model Size: 79.67 MB
✓ Average Inference Time: 0.0139 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
COMPARISON: MAGNITUDE-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |           0.36  |    75.55   | -3.05                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |          79.67  |    79.67   | -18.02 (-18.4%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0294 |           0.026 |     0.0139 | -0.0155                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |           3.34  |     3.34   | -0.79 (-19.1%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS3_ResNet50_ImageNet\TS3_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================

====================================================================================================
✓ TS3_04_magnitude_pruning.py completed successfully
  Elapsed time: 3460.40 seconds
====================================================================================================


====================================================================================================
SCRIPT 5/6: TS3_05_taylor_pruning.py
====================================================================================================

====================================================================================================
RUNNING: TS3_05_taylor_pruning.py
====================================================================================================


================================================================================
TEST SCENARIO TS3: TAYLOR GRADIENT-BASED PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS3\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                               
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0271 ms
✓ FLOPs: 4.13 GFLOPs

================================================================================
APPLYING TAYLOR GRADIENT-BASED PRUNING
================================================================================
Pruning Ratio: 10.0%
Global Pruning: False
Iterative Steps: 1
Taylor pruning uses first-order gradient information
Importance = |weight × gradient| (Taylor expansion approximation)
Using 100 calibration batches

Computing Taylor importance scores (requires gradients)...
Computing gradients:  99%|███████████████████████████████████████████████████████████████████████████████████▏| 99/100 [16:52<00:10, 10.22s/it] 
✓ Gradients computed on 100 batches
Using GroupTaylorImportance

Initializing Torch-Pruning with Taylor importance...

Applying pruning...

✓ Taylor pruning completed
  Parameters before: 25,557,032
  Parameters after: 20,837,770
  Parameters removed: 4,719,262 (18.47%)

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 6.10%                                                                                                                  
✓ Model Size: 79.67 MB
✓ Average Inference Time: 0.0176 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [10:28<00:00, 15.72s/it, loss=1.9651, acc=57.82%] 
Epoch 1/10 - Train Loss: 1.9651, Train Acc: 57.82%, Test Acc: 70.08%                                                                            
Epoch 2/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [10:34<00:00, 15.86s/it, loss=1.5044, acc=66.39%] 
Epoch 2/10 - Train Loss: 1.5044, Train Acc: 66.39%, Test Acc: 72.06%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_epoch2.pth
Epoch 3/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [10:40<00:00, 16.01s/it, loss=1.2691, acc=70.12%]
Epoch 3/10 - Train Loss: 1.2691, Train Acc: 70.12%, Test Acc: 73.25%                                                                            
Epoch 4/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [10:29<00:00, 15.73s/it, loss=1.0993, acc=74.12%] 
Epoch 4/10 - Train Loss: 1.0993, Train Acc: 74.12%, Test Acc: 74.01%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_epoch4.pth
Epoch 5/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [10:37<00:00, 15.94s/it, loss=0.9937, acc=77.12%]
Epoch 5/10 - Train Loss: 0.9937, Train Acc: 77.12%, Test Acc: 74.57%                                                                            
Epoch 6/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [10:28<00:00, 15.71s/it, loss=0.9240, acc=78.33%] 
Epoch 6/10 - Train Loss: 0.9240, Train Acc: 78.33%, Test Acc: 74.90%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_epoch6.pth
Epoch 7/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [10:37<00:00, 15.94s/it, loss=0.8457, acc=80.24%]
Epoch 7/10 - Train Loss: 0.8457, Train Acc: 80.24%, Test Acc: 75.20%                                                                            
Epoch 8/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [10:28<00:00, 15.71s/it, loss=0.8042, acc=80.83%] 
Epoch 8/10 - Train Loss: 0.8042, Train Acc: 80.83%, Test Acc: 75.32%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_epoch8.pth
Epoch 9/10: 100%|█████████████████████████████████████████████████████████████████████| 40/40 [10:39<00:00, 15.99s/it, loss=0.8303, acc=81.20%]
Epoch 9/10 - Train Loss: 0.8303, Train Acc: 81.20%, Test Acc: 75.52%                                                                            
Epoch 10/10: 100%|████████████████████████████████████████████████████████████████████| 40/40 [10:31<00:00, 15.80s/it, loss=0.7955, acc=81.21%] 
Epoch 10/10 - Train Loss: 0.7955, Train Acc: 81.21%, Test Acc: 75.54%                                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_best.pth
  Best Accuracy: 75.54%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 75.54%                                                                                                                  
✓ Model Size: 79.67 MB
✓ Average Inference Time: 0.0181 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
COMPARISON: TAYLOR GRADIENT-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          6.1    |    75.54   | -3.06                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         79.67   |    79.67   | -18.02 (-18.4%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0271 |          0.0176 |     0.0181 | -0.0090                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.34   |     3.34   | -0.79 (-19.1%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS3_ResNet50_ImageNet\TS3_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================

====================================================================================================
✓ TS3_05_taylor_pruning.py completed successfully
  Elapsed time: 14578.25 seconds
====================================================================================================