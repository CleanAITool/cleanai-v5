================================================================================
TEST SCENARIO TS7: NEURON COVERAGE PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS7\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                    
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0261 ms
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
  conv1: coverage shape torch.Size([64]), min=-0.719299, max=1.000000, mean=0.007813
  layer1.0.conv1: coverage shape torch.Size([64]), min=-0.682198, max=1.000000, mean=-0.109375
  layer1.0.conv2: coverage shape torch.Size([64]), min=-2.975588, max=1.000000, mean=0.121303
  layer1.0.conv3: coverage shape torch.Size([256]), min=-0.757383, max=1.000000, mean=-0.036521
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=-1.998949, max=1.000000, mean=-0.093982
  layer1.1.conv1: coverage shape torch.Size([64]), min=-1.894118, max=1.000000, mean=-0.058185
  layer1.1.conv2: coverage shape torch.Size([64]), min=-1.527905, max=1.000000, mean=-0.103044
  layer1.1.conv3: coverage shape torch.Size([256]), min=-1.595068, max=1.000000, mean=-0.105994
  layer1.2.conv1: coverage shape torch.Size([64]), min=-0.879295, max=1.000000, mean=-0.115572
  layer1.2.conv2: coverage shape torch.Size([64]), min=-1.833173, max=1.000000, mean=-0.215166
  layer1.2.conv3: coverage shape torch.Size([256]), min=-1.091851, max=1.000000, mean=-0.034205
  layer2.0.conv1: coverage shape torch.Size([128]), min=-1.017522, max=1.000000, mean=-0.088532
  layer2.0.conv2: coverage shape torch.Size([128]), min=-1.425962, max=1.000000, mean=-0.151666
  layer2.0.conv3: coverage shape torch.Size([512]), min=-0.669799, max=1.000000, mean=0.016014
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=-0.821847, max=1.000000, mean=-0.055918
  layer2.1.conv1: coverage shape torch.Size([128]), min=-0.988149, max=1.000000, mean=-0.094178
  layer2.1.conv2: coverage shape torch.Size([128]), min=-0.743290, max=1.000000, mean=0.069141
  layer2.1.conv3: coverage shape torch.Size([512]), min=-0.911349, max=1.000000, mean=-0.043976
  layer2.2.conv1: coverage shape torch.Size([128]), min=-1.437878, max=1.000000, mean=-0.005417
  layer2.2.conv2: coverage shape torch.Size([128]), min=-1.097275, max=1.000000, mean=-0.161687
  layer2.2.conv3: coverage shape torch.Size([512]), min=-0.949308, max=1.000000, mean=-0.002144
  layer2.3.conv1: coverage shape torch.Size([128]), min=-0.599401, max=1.000000, mean=-0.044096
  layer2.3.conv2: coverage shape torch.Size([128]), min=-0.855327, max=1.000000, mean=-0.127570
  layer2.3.conv3: coverage shape torch.Size([512]), min=-1.122539, max=1.000000, mean=-0.051312
  layer3.0.conv1: coverage shape torch.Size([256]), min=-1.504441, max=1.000000, mean=-0.196712
  layer3.0.conv2: coverage shape torch.Size([256]), min=-1.559325, max=1.000000, mean=-0.092353
  layer3.0.conv3: coverage shape torch.Size([1024]), min=-1.174423, max=1.000000, mean=-0.020844
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=-1.315241, max=1.000000, mean=-0.149406
  layer3.1.conv1: coverage shape torch.Size([256]), min=-1.814898, max=1.000000, mean=-0.068272
  layer3.1.conv2: coverage shape torch.Size([256]), min=-2.276280, max=1.000000, mean=-0.071583
  layer3.1.conv3: coverage shape torch.Size([1024]), min=-1.962636, max=1.000000, mean=-0.191074
  layer3.2.conv1: coverage shape torch.Size([256]), min=-1.419594, max=1.000000, mean=-0.248784
  layer3.2.conv2: coverage shape torch.Size([256]), min=-1.242712, max=1.000000, mean=-0.199763
  layer3.2.conv3: coverage shape torch.Size([1024]), min=-1.188858, max=1.000000, mean=-0.051403
  layer3.3.conv1: coverage shape torch.Size([256]), min=-1.150611, max=1.000000, mean=-0.128206
  layer3.3.conv2: coverage shape torch.Size([256]), min=-0.715961, max=1.000000, mean=-0.088912
  layer3.3.conv3: coverage shape torch.Size([1024]), min=-1.390462, max=1.000000, mean=-0.087795
  layer3.4.conv1: coverage shape torch.Size([256]), min=-0.842542, max=1.000000, mean=-0.256077
  layer3.4.conv2: coverage shape torch.Size([256]), min=-1.096814, max=1.000000, mean=-0.255149
  layer3.4.conv3: coverage shape torch.Size([1024]), min=-0.733167, max=1.000000, mean=-0.021725
  layer3.5.conv1: coverage shape torch.Size([256]), min=-1.073503, max=1.000000, mean=-0.174853
  layer3.5.conv2: coverage shape torch.Size([256]), min=-1.386695, max=1.000000, mean=-0.231584
  layer3.5.conv3: coverage shape torch.Size([1024]), min=-1.615893, max=1.000000, mean=-0.264700
  layer4.0.conv1: coverage shape torch.Size([512]), min=-1.413804, max=1.000000, mean=-0.603347
  layer4.0.conv2: coverage shape torch.Size([512]), min=-2.157458, max=1.000000, mean=-0.413463
  layer4.0.conv3: coverage shape torch.Size([2048]), min=-0.937220, max=1.000000, mean=-0.086108
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=-1.223801, max=1.000000, mean=-0.160032
  layer4.1.conv1: coverage shape torch.Size([512]), min=-1.796372, max=1.000000, mean=-0.243067
  layer4.1.conv2: coverage shape torch.Size([512]), min=-1.215733, max=1.000000, mean=-0.206999
  layer4.1.conv3: coverage shape torch.Size([2048]), min=-0.726145, max=1.000000, mean=-0.146462
  layer4.2.conv1: coverage shape torch.Size([512]), min=-1.422721, max=1.000000, mean=-0.225042
  layer4.2.conv2: coverage shape torch.Size([512]), min=-2.207188, max=1.000000, mean=-0.299027
  layer4.2.conv3: coverage shape torch.Size([2048]), min=-2.470708, max=1.000000, mean=-0.834878
  fc: coverage shape torch.Size([1000]), min=-0.821834, max=1.000000, mean=0.000404

------------------------------------------------------------
Coverage Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Coverage - Min: -0.719299, Max: 1.000000, Mean: 0.007813
  Zero coverage neurons: 0

layer1.0.conv1:
  Channels: 64
  Coverage - Min: -0.682198, Max: 1.000000, Mean: -0.109375
  Zero coverage neurons: 0

layer1.0.conv2:
  Channels: 64
  Coverage - Min: -2.975588, Max: 1.000000, Mean: 0.121303
  Zero coverage neurons: 0

layer1.0.conv3:
  Channels: 256
  Coverage - Min: -0.757383, Max: 1.000000, Mean: -0.036521
  Zero coverage neurons: 0

layer1.0.downsample.0:
  Channels: 256
  Coverage - Min: -1.998949, Max: 1.000000, Mean: -0.093982
  Zero coverage neurons: 0

layer1.1.conv1:
  Channels: 64
  Coverage - Min: -1.894118, Max: 1.000000, Mean: -0.058185
  Zero coverage neurons: 0

layer1.1.conv2:
  Channels: 64
  Coverage - Min: -1.527905, Max: 1.000000, Mean: -0.103044
  Zero coverage neurons: 0

layer1.1.conv3:
  Channels: 256
  Coverage - Min: -1.595068, Max: 1.000000, Mean: -0.105994
  Zero coverage neurons: 0

layer1.2.conv1:
  Channels: 64
  Coverage - Min: -0.879295, Max: 1.000000, Mean: -0.115572
  Zero coverage neurons: 0

layer1.2.conv2:
  Channels: 64
  Coverage - Min: -1.833173, Max: 1.000000, Mean: -0.215166
  Zero coverage neurons: 0

layer1.2.conv3:
  Channels: 256
  Coverage - Min: -1.091851, Max: 1.000000, Mean: -0.034205
  Zero coverage neurons: 0

layer2.0.conv1:
  Channels: 128
  Coverage - Min: -1.017522, Max: 1.000000, Mean: -0.088532
  Zero coverage neurons: 0

layer2.0.conv2:
  Channels: 128
  Coverage - Min: -1.425962, Max: 1.000000, Mean: -0.151666
  Zero coverage neurons: 0

layer2.0.conv3:
  Channels: 512
  Coverage - Min: -0.669799, Max: 1.000000, Mean: 0.016014
  Zero coverage neurons: 0

layer2.0.downsample.0:
  Channels: 512
  Coverage - Min: -0.821847, Max: 1.000000, Mean: -0.055918
  Zero coverage neurons: 0

layer2.1.conv1:
  Channels: 128
  Coverage - Min: -0.988149, Max: 1.000000, Mean: -0.094178
  Zero coverage neurons: 0

layer2.1.conv2:
  Channels: 128
  Coverage - Min: -0.743290, Max: 1.000000, Mean: 0.069141
  Zero coverage neurons: 0

layer2.1.conv3:
  Channels: 512
  Coverage - Min: -0.911349, Max: 1.000000, Mean: -0.043976
  Zero coverage neurons: 0

layer2.2.conv1:
  Channels: 128
  Coverage - Min: -1.437878, Max: 1.000000, Mean: -0.005417
  Zero coverage neurons: 0

layer2.2.conv2:
  Channels: 128
  Coverage - Min: -1.097275, Max: 1.000000, Mean: -0.161687
  Zero coverage neurons: 0

layer2.2.conv3:
  Channels: 512
  Coverage - Min: -0.949308, Max: 1.000000, Mean: -0.002144
  Zero coverage neurons: 0

layer2.3.conv1:
  Channels: 128
  Coverage - Min: -0.599401, Max: 1.000000, Mean: -0.044096
  Zero coverage neurons: 0

layer2.3.conv2:
  Channels: 128
  Coverage - Min: -0.855327, Max: 1.000000, Mean: -0.127570
  Zero coverage neurons: 0

layer2.3.conv3:
  Channels: 512
  Coverage - Min: -1.122539, Max: 1.000000, Mean: -0.051312
  Zero coverage neurons: 0

layer3.0.conv1:
  Channels: 256
  Coverage - Min: -1.504441, Max: 1.000000, Mean: -0.196712
  Zero coverage neurons: 0

layer3.0.conv2:
  Channels: 256
  Coverage - Min: -1.559325, Max: 1.000000, Mean: -0.092353
  Zero coverage neurons: 0

layer3.0.conv3:
  Channels: 1024
  Coverage - Min: -1.174423, Max: 1.000000, Mean: -0.020844
  Zero coverage neurons: 0

layer3.0.downsample.0:
  Channels: 1024
  Coverage - Min: -1.315241, Max: 1.000000, Mean: -0.149406
  Zero coverage neurons: 0

layer3.1.conv1:
  Channels: 256
  Coverage - Min: -1.814898, Max: 1.000000, Mean: -0.068272
  Zero coverage neurons: 0

layer3.1.conv2:
  Channels: 256
  Coverage - Min: -2.276280, Max: 1.000000, Mean: -0.071583
  Zero coverage neurons: 0

layer3.1.conv3:
  Channels: 1024
  Coverage - Min: -1.962636, Max: 1.000000, Mean: -0.191074
  Zero coverage neurons: 0

layer3.2.conv1:
  Channels: 256
  Coverage - Min: -1.419594, Max: 1.000000, Mean: -0.248784
  Zero coverage neurons: 0

layer3.2.conv2:
  Channels: 256
  Coverage - Min: -1.242712, Max: 1.000000, Mean: -0.199763
  Zero coverage neurons: 0

layer3.2.conv3:
  Channels: 1024
  Coverage - Min: -1.188858, Max: 1.000000, Mean: -0.051403
  Zero coverage neurons: 0

layer3.3.conv1:
  Channels: 256
  Coverage - Min: -1.150611, Max: 1.000000, Mean: -0.128206
  Zero coverage neurons: 0

layer3.3.conv2:
  Channels: 256
  Coverage - Min: -0.715961, Max: 1.000000, Mean: -0.088912
  Zero coverage neurons: 0

layer3.3.conv3:
  Channels: 1024
  Coverage - Min: -1.390462, Max: 1.000000, Mean: -0.087795
  Zero coverage neurons: 0

layer3.4.conv1:
  Channels: 256
  Coverage - Min: -0.842542, Max: 1.000000, Mean: -0.256077
  Zero coverage neurons: 0

layer3.4.conv2:
  Channels: 256
  Coverage - Min: -1.096814, Max: 1.000000, Mean: -0.255149
  Zero coverage neurons: 0

layer3.4.conv3:
  Channels: 1024
  Coverage - Min: -0.733167, Max: 1.000000, Mean: -0.021725
  Zero coverage neurons: 0

layer3.5.conv1:
  Channels: 256
  Coverage - Min: -1.073503, Max: 1.000000, Mean: -0.174853
  Zero coverage neurons: 0

layer3.5.conv2:
  Channels: 256
  Coverage - Min: -1.386695, Max: 1.000000, Mean: -0.231584
  Zero coverage neurons: 0

layer3.5.conv3:
  Channels: 1024
  Coverage - Min: -1.615893, Max: 1.000000, Mean: -0.264700
  Zero coverage neurons: 0

layer4.0.conv1:
  Channels: 512
  Coverage - Min: -1.413804, Max: 1.000000, Mean: -0.603347
  Zero coverage neurons: 0

layer4.0.conv2:
  Channels: 512
  Coverage - Min: -2.157458, Max: 1.000000, Mean: -0.413463
  Zero coverage neurons: 0

layer4.0.conv3:
  Channels: 2048
  Coverage - Min: -0.937220, Max: 1.000000, Mean: -0.086108
  Zero coverage neurons: 0

layer4.0.downsample.0:
  Channels: 2048
  Coverage - Min: -1.223801, Max: 1.000000, Mean: -0.160032
  Zero coverage neurons: 0

layer4.1.conv1:
  Channels: 512
  Coverage - Min: -1.796372, Max: 1.000000, Mean: -0.243067
  Zero coverage neurons: 0

layer4.1.conv2:
  Channels: 512
  Coverage - Min: -1.215733, Max: 1.000000, Mean: -0.206999
  Zero coverage neurons: 0

layer4.1.conv3:
  Channels: 2048
  Coverage - Min: -0.726145, Max: 1.000000, Mean: -0.146462
  Zero coverage neurons: 0

layer4.2.conv1:
  Channels: 512
  Coverage - Min: -1.422721, Max: 1.000000, Mean: -0.225042
  Zero coverage neurons: 0

layer4.2.conv2:
  Channels: 512
  Coverage - Min: -2.207188, Max: 1.000000, Mean: -0.299027
  Zero coverage neurons: 0

layer4.2.conv3:
  Channels: 2048
  Coverage - Min: -2.470708, Max: 1.000000, Mean: -0.834878
  Zero coverage neurons: 0

fc:
  Channels: 1000
  Coverage - Min: -0.821834, Max: 1.000000, Mean: 0.000404
  Zero coverage neurons: 0
============================================================


Pruning Results:
  Parameters before: 25,557,032
  Parameters after:  21,650,039
  Parameters removed: 3,906,993 (15.29%)

############################################################
Pruning Complete
############################################################
Initial parameters:  25,557,032
Final parameters:    21,650,039
Total removed:       3,906,993 (15.29%)
Target pruning ratio: 10.00%
✓ Neuron Coverage pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.11%                                                                                       
✓ Model Size: 82.78 MB
✓ Average Inference Time: 1.3365 ms
✓ FLOPs: 3.21 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|██████████████████████████████████████████| 40/40 [01:25<00:00,  2.14s/it, loss=3.3562, acc=37.88%]
Epoch 1/10 - Train Loss: 3.3562, Train Acc: 37.88%, Test Acc: 59.38%
Epoch 2/10: 100%|██████████████████████████████████████████| 40/40 [01:22<00:00,  2.07s/it, loss=2.0977, acc=54.86%]
Epoch 2/10 - Train Loss: 2.0977, Train Acc: 54.86%, Test Acc: 64.59%
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_NC_epoch2.pth
Epoch 3/10: 100%|██████████████████████████████████████████| 40/40 [01:23<00:00,  2.08s/it, loss=1.7466, acc=61.39%]
Epoch 3/10 - Train Loss: 1.7466, Train Acc: 61.39%, Test Acc: 67.29%
Epoch 4/10: 100%|██████████████████████████████████████████| 40/40 [01:22<00:00,  2.07s/it, loss=1.5185, acc=65.80%]
Epoch 4/10 - Train Loss: 1.5185, Train Acc: 65.80%, Test Acc: 68.64%
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_NC_epoch4.pth
Epoch 5/10: 100%|██████████████████████████████████████████| 40/40 [01:22<00:00,  2.07s/it, loss=1.3545, acc=69.08%]
Epoch 5/10 - Train Loss: 1.3545, Train Acc: 69.08%, Test Acc: 69.63%
Epoch 6/10: 100%|██████████████████████████████████████████| 40/40 [01:23<00:00,  2.08s/it, loss=1.2356, acc=71.72%]
Epoch 6/10 - Train Loss: 1.2356, Train Acc: 71.72%, Test Acc: 70.23%
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_NC_epoch6.pth
Epoch 7/10: 100%|██████████████████████████████████████████| 40/40 [01:22<00:00,  2.07s/it, loss=1.1902, acc=73.04%]
Epoch 7/10 - Train Loss: 1.1902, Train Acc: 73.04%, Test Acc: 70.65%
Epoch 8/10: 100%|██████████████████████████████████████████| 40/40 [01:22<00:00,  2.07s/it, loss=1.1271, acc=75.00%]
Epoch 8/10 - Train Loss: 1.1271, Train Acc: 75.00%, Test Acc: 71.01%
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_NC_epoch8.pth
Epoch 9/10: 100%|██████████████████████████████████████████| 40/40 [01:23<00:00,  2.09s/it, loss=1.0917, acc=75.35%]
Epoch 9/10 - Train Loss: 1.0917, Train Acc: 75.35%, Test Acc: 71.18%
Epoch 10/10: 100%|█████████████████████████████████████████| 40/40 [01:23<00:00,  2.08s/it, loss=1.0764, acc=75.02%]
Epoch 10/10 - Train Loss: 1.0764, Train Acc: 75.02%, Test Acc: 71.16%
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_NC_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_NC_best.pth
  Best Accuracy: 71.18%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 71.16%                                                                                                                
✓ Model Size: 82.78 MB
✓ Average Inference Time: 1.8076 ms
✓ FLOPs: 3.21 GFLOPs

================================================================================
COMPARISON: NEURON COVERAGE PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.11   |    71.16   | -7.43                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         82.78   |    82.78   | -14.92 (-15.3%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0261 |          1.3365 |     1.8076 | +1.7814                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.21   |     3.21   | -0.93 (-22.4%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS7_ResNet50_ImageNet\TS7_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================




***************************************



================================================================================
TEST SCENARIO TS7: WANDA PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS7\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                             
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0262 ms
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
  conv1: coverage shape torch.Size([64]), min=0.000023, max=0.439966, mean=0.030284
  layer1.0.conv1: coverage shape torch.Size([64]), min=0.866231, max=44.563385, mean=9.882047
  layer1.0.conv2: coverage shape torch.Size([64]), min=0.060523, max=25.869665, mean=3.917229
  layer1.0.conv3: coverage shape torch.Size([256]), min=0.000000, max=10.936882, mean=2.151631
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=0.000000, max=61.205597, mean=5.205736
  layer1.1.conv1: coverage shape torch.Size([64]), min=0.000000, max=6.712589, mean=1.258943
  layer1.1.conv2: coverage shape torch.Size([64]), min=0.127245, max=6.327180, mean=1.655192
  layer1.1.conv3: coverage shape torch.Size([256]), min=0.001589, max=7.035775, mean=1.304068
  layer1.2.conv1: coverage shape torch.Size([64]), min=0.030672, max=4.762389, mean=1.477386
  layer1.2.conv2: coverage shape torch.Size([64]), min=0.064876, max=6.586328, mean=1.332125
  layer1.2.conv3: coverage shape torch.Size([256]), min=0.002894, max=5.845990, mean=0.939756
  layer2.0.conv1: coverage shape torch.Size([128]), min=0.003312, max=5.204557, mean=1.523264
  layer2.0.conv2: coverage shape torch.Size([128]), min=0.014571, max=6.543492, mean=1.467194
  layer2.0.conv3: coverage shape torch.Size([512]), min=0.000002, max=8.115936, mean=1.116410
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=0.000001, max=8.563728, mean=1.163486
  layer2.1.conv1: coverage shape torch.Size([128]), min=0.079182, max=7.679285, mean=2.209431
  layer2.1.conv2: coverage shape torch.Size([128]), min=0.022476, max=10.109801, mean=1.950451
  layer2.1.conv3: coverage shape torch.Size([512]), min=0.000288, max=2.399324, mean=0.467649
  layer2.2.conv1: coverage shape torch.Size([128]), min=0.007613, max=5.483105, mean=1.217980
  layer2.2.conv2: coverage shape torch.Size([128]), min=0.011287, max=6.193237, mean=1.373259
  layer2.2.conv3: coverage shape torch.Size([512]), min=0.000564, max=3.078230, mean=0.579465
  layer2.3.conv1: coverage shape torch.Size([128]), min=0.021050, max=6.826830, mean=1.273093
  layer2.3.conv2: coverage shape torch.Size([128]), min=0.013255, max=5.927760, mean=1.379885
  layer2.3.conv3: coverage shape torch.Size([512]), min=0.000561, max=3.384736, mean=0.602306
  layer3.0.conv1: coverage shape torch.Size([256]), min=0.000202, max=6.383198, mean=1.402857
  layer3.0.conv2: coverage shape torch.Size([256]), min=0.000213, max=4.450257, mean=0.891210
  layer3.0.conv3: coverage shape torch.Size([1024]), min=0.000279, max=4.704382, mean=0.894875
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=0.000251, max=4.072598, mean=0.831258
  layer3.1.conv1: coverage shape torch.Size([256]), min=0.003626, max=9.746452, mean=1.413345
  layer3.1.conv2: coverage shape torch.Size([256]), min=0.001958, max=11.066037, mean=1.027974
  layer3.1.conv3: coverage shape torch.Size([1024]), min=0.000111, max=2.090846, mean=0.318360
  layer3.2.conv1: coverage shape torch.Size([256]), min=0.000866, max=5.962078, mean=1.282869
  layer3.2.conv2: coverage shape torch.Size([256]), min=0.004096, max=3.085822, mean=0.708354
  layer3.2.conv3: coverage shape torch.Size([1024]), min=0.000465, max=2.030093, mean=0.287256
  layer3.3.conv1: coverage shape torch.Size([256]), min=0.001409, max=5.974561, mean=1.164230
  layer3.3.conv2: coverage shape torch.Size([256]), min=0.010209, max=5.393601, mean=0.820337
  layer3.3.conv3: coverage shape torch.Size([1024]), min=0.000529, max=1.958202, mean=0.335661
  layer3.4.conv1: coverage shape torch.Size([256]), min=0.011464, max=3.742772, mean=1.143913
  layer3.4.conv2: coverage shape torch.Size([256]), min=0.010574, max=3.057929, mean=0.864038
  layer3.4.conv3: coverage shape torch.Size([1024]), min=0.000015, max=2.884928, mean=0.299612
  layer3.5.conv1: coverage shape torch.Size([256]), min=0.002398, max=5.940731, mean=1.247435
  layer3.5.conv2: coverage shape torch.Size([256]), min=0.000910, max=3.803958, mean=0.846353
  layer3.5.conv3: coverage shape torch.Size([1024]), min=0.001264, max=2.126465, mean=0.429383
  layer4.0.conv1: coverage shape torch.Size([512]), min=0.042930, max=3.695797, mean=1.627567
  layer4.0.conv2: coverage shape torch.Size([512]), min=0.007911, max=4.814509, mean=0.951578
  layer4.0.conv3: coverage shape torch.Size([2048]), min=0.000147, max=5.296698, mean=0.589151
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=0.000602, max=3.842167, mean=0.576766
  layer4.1.conv1: coverage shape torch.Size([512]), min=0.111470, max=9.317658, mean=1.403164
  layer4.1.conv2: coverage shape torch.Size([512]), min=0.001620, max=5.475863, mean=1.052990
  layer4.1.conv3: coverage shape torch.Size([2048]), min=0.000093, max=2.933833, mean=0.447328
  layer4.2.conv1: coverage shape torch.Size([512]), min=0.088492, max=7.494927, mean=1.340458
  layer4.2.conv2: coverage shape torch.Size([512]), min=0.047812, max=4.695520, mean=0.664283
  layer4.2.conv3: coverage shape torch.Size([2048]), min=0.000295, max=1.061600, mean=0.361308
  fc: coverage shape torch.Size([1000]), min=0.000196, max=0.281779, mean=0.059968

------------------------------------------------------------
Activation Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Activation - Min: 0.000023, Max: 0.439966, Mean: 0.030284

layer1.0.conv1:
  Channels: 64
  Activation - Min: 0.866231, Max: 44.563385, Mean: 9.882047

layer1.0.conv2:
  Channels: 64
  Activation - Min: 0.060523, Max: 25.869665, Mean: 3.917229

layer1.0.conv3:
  Channels: 256
  Activation - Min: 0.000000, Max: 10.936882, Mean: 2.151631

layer1.0.downsample.0:
  Channels: 256
  Activation - Min: 0.000000, Max: 61.205597, Mean: 5.205736

layer1.1.conv1:
  Channels: 64
  Activation - Min: 0.000000, Max: 6.712589, Mean: 1.258943

layer1.1.conv2:
  Channels: 64
  Activation - Min: 0.127245, Max: 6.327180, Mean: 1.655192

layer1.1.conv3:
  Channels: 256
  Activation - Min: 0.001589, Max: 7.035775, Mean: 1.304068

layer1.2.conv1:
  Channels: 64
  Activation - Min: 0.030672, Max: 4.762389, Mean: 1.477386

layer1.2.conv2:
  Channels: 64
  Activation - Min: 0.064876, Max: 6.586328, Mean: 1.332125

layer1.2.conv3:
  Channels: 256
  Activation - Min: 0.002894, Max: 5.845990, Mean: 0.939756

layer2.0.conv1:
  Channels: 128
  Activation - Min: 0.003312, Max: 5.204557, Mean: 1.523264

layer2.0.conv2:
  Channels: 128
  Activation - Min: 0.014571, Max: 6.543492, Mean: 1.467194

layer2.0.conv3:
  Channels: 512
  Activation - Min: 0.000002, Max: 8.115936, Mean: 1.116410

layer2.0.downsample.0:
  Channels: 512
  Activation - Min: 0.000001, Max: 8.563728, Mean: 1.163486

layer2.1.conv1:
  Channels: 128
  Activation - Min: 0.079182, Max: 7.679285, Mean: 2.209431

layer2.1.conv2:
  Channels: 128
  Activation - Min: 0.022476, Max: 10.109801, Mean: 1.950451

layer2.1.conv3:
  Channels: 512
  Activation - Min: 0.000288, Max: 2.399324, Mean: 0.467649

layer2.2.conv1:
  Channels: 128
  Activation - Min: 0.007613, Max: 5.483105, Mean: 1.217980

layer2.2.conv2:
  Channels: 128
  Activation - Min: 0.011287, Max: 6.193237, Mean: 1.373259

layer2.2.conv3:
  Channels: 512
  Activation - Min: 0.000564, Max: 3.078230, Mean: 0.579465

layer2.3.conv1:
  Channels: 128
  Activation - Min: 0.021050, Max: 6.826830, Mean: 1.273093

layer2.3.conv2:
  Channels: 128
  Activation - Min: 0.013255, Max: 5.927760, Mean: 1.379885

layer2.3.conv3:
  Channels: 512
  Activation - Min: 0.000561, Max: 3.384736, Mean: 0.602306

layer3.0.conv1:
  Channels: 256
  Activation - Min: 0.000202, Max: 6.383198, Mean: 1.402857

layer3.0.conv2:
  Channels: 256
  Activation - Min: 0.000213, Max: 4.450257, Mean: 0.891210

layer3.0.conv3:
  Channels: 1024
  Activation - Min: 0.000279, Max: 4.704382, Mean: 0.894875

layer3.0.downsample.0:
  Channels: 1024
  Activation - Min: 0.000251, Max: 4.072598, Mean: 0.831258

layer3.1.conv1:
  Channels: 256
  Activation - Min: 0.003626, Max: 9.746452, Mean: 1.413345

layer3.1.conv2:
  Channels: 256
  Activation - Min: 0.001958, Max: 11.066037, Mean: 1.027974

layer3.1.conv3:
  Channels: 1024
  Activation - Min: 0.000111, Max: 2.090846, Mean: 0.318360

layer3.2.conv1:
  Channels: 256
  Activation - Min: 0.000866, Max: 5.962078, Mean: 1.282869

layer3.2.conv2:
  Channels: 256
  Activation - Min: 0.004096, Max: 3.085822, Mean: 0.708354

layer3.2.conv3:
  Channels: 1024
  Activation - Min: 0.000465, Max: 2.030093, Mean: 0.287256

layer3.3.conv1:
  Channels: 256
  Activation - Min: 0.001409, Max: 5.974561, Mean: 1.164230

layer3.3.conv2:
  Channels: 256
  Activation - Min: 0.010209, Max: 5.393601, Mean: 0.820337

layer3.3.conv3:
  Channels: 1024
  Activation - Min: 0.000529, Max: 1.958202, Mean: 0.335661

layer3.4.conv1:
  Channels: 256
  Activation - Min: 0.011464, Max: 3.742772, Mean: 1.143913

layer3.4.conv2:
  Channels: 256
  Activation - Min: 0.010574, Max: 3.057929, Mean: 0.864038

layer3.4.conv3:
  Channels: 1024
  Activation - Min: 0.000015, Max: 2.884928, Mean: 0.299612

layer3.5.conv1:
  Channels: 256
  Activation - Min: 0.002398, Max: 5.940731, Mean: 1.247435

layer3.5.conv2:
  Channels: 256
  Activation - Min: 0.000910, Max: 3.803958, Mean: 0.846353

layer3.5.conv3:
  Channels: 1024
  Activation - Min: 0.001264, Max: 2.126465, Mean: 0.429383

layer4.0.conv1:
  Channels: 512
  Activation - Min: 0.042930, Max: 3.695797, Mean: 1.627567

layer4.0.conv2:
  Channels: 512
  Activation - Min: 0.007911, Max: 4.814509, Mean: 0.951578

layer4.0.conv3:
  Channels: 2048
  Activation - Min: 0.000147, Max: 5.296698, Mean: 0.589151

layer4.0.downsample.0:
  Channels: 2048
  Activation - Min: 0.000602, Max: 3.842167, Mean: 0.576766

layer4.1.conv1:
  Channels: 512
  Activation - Min: 0.111470, Max: 9.317658, Mean: 1.403164

layer4.1.conv2:
  Channels: 512
  Activation - Min: 0.001620, Max: 5.475863, Mean: 1.052990

layer4.1.conv3:
  Channels: 2048
  Activation - Min: 0.000093, Max: 2.933833, Mean: 0.447328

layer4.2.conv1:
  Channels: 512
  Activation - Min: 0.088492, Max: 7.494927, Mean: 1.340458

layer4.2.conv2:
  Channels: 512
  Activation - Min: 0.047812, Max: 4.695520, Mean: 0.664283

layer4.2.conv3:
  Channels: 2048
  Activation - Min: 0.000295, Max: 1.061600, Mean: 0.361308

fc:
  Channels: 1000
  Activation - Min: 0.000196, Max: 0.281779, Mean: 0.059968
============================================================


Pruning Results:
  Parameters before: 25,557,032
  Parameters after:  21,481,254
  Parameters removed: 4,075,778 (15.95%)

############################################################
Pruning Complete
############################################################
Initial parameters:  25,557,032
Final parameters:    21,481,254
Total removed:       4,075,778 (15.95%)
Target pruning ratio: 10.00%
✓ Wanda pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.18%                                                                                                                
✓ Model Size: 82.12 MB
✓ Average Inference Time: 1.1776 ms
✓ FLOPs: 3.16 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.04s/it, loss=3.4075, acc=34.55%] 
Epoch 1/10 - Train Loss: 3.4075, Train Acc: 34.55%, Test Acc: 53.95%                                                                          
Epoch 2/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.3230, acc=50.68%] 
Epoch 2/10 - Train Loss: 2.3230, Train Acc: 50.68%, Test Acc: 59.49%                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_W_epoch2.pth
Epoch 3/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.0082, acc=56.41%]
Epoch 3/10 - Train Loss: 2.0082, Train Acc: 56.41%, Test Acc: 62.31%                                                                          
Epoch 4/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.6942, acc=62.34%] 
Epoch 4/10 - Train Loss: 1.6942, Train Acc: 62.34%, Test Acc: 64.24%                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_W_epoch4.pth
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.5319, acc=64.70%]
Epoch 5/10 - Train Loss: 1.5319, Train Acc: 64.70%, Test Acc: 65.24%                                                                          
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.4751, acc=66.88%] 
Epoch 6/10 - Train Loss: 1.4751, Train Acc: 66.88%, Test Acc: 66.15%                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_W_epoch6.pth
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.3264, acc=69.45%]
Epoch 7/10 - Train Loss: 1.3264, Train Acc: 69.45%, Test Acc: 66.91%                                                                          
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.2848, acc=71.09%] 
Epoch 8/10 - Train Loss: 1.2848, Train Acc: 71.09%, Test Acc: 67.26%                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_W_epoch8.pth
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.2576, acc=71.19%]
Epoch 9/10 - Train Loss: 1.2576, Train Acc: 71.19%, Test Acc: 67.32%                                                                          
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=1.2204, acc=72.07%] 
Epoch 10/10 - Train Loss: 1.2204, Train Acc: 72.07%, Test Acc: 67.35%                                                                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_W_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_W_best.pth
  Best Accuracy: 67.35%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 67.35%                                                                                                                
✓ Model Size: 82.12 MB
✓ Average Inference Time: 1.1796 ms
✓ FLOPs: 3.16 GFLOPs

================================================================================
COMPARISON: WANDA PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.18   |    67.35   | -11.25                      |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         82.12   |    82.12   | -15.58 (-15.9%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0262 |          1.1776 |     1.1796 | +1.1534                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.16   |     3.16   | -0.97 (-23.5%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS7_ResNet50_ImageNet\TS7_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================




**************************



================================================================================
TEST SCENARIO TS7: MAGNITUDE-BASED PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS7\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                             
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0281 ms
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
✓ Average Inference Time: 0.0253 ms
✓ FLOPs: 3.05 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.01it/s, loss=3.6642, acc=33.39%] 
Epoch 1/10 - Train Loss: 3.6642, Train Acc: 33.39%, Test Acc: 55.98%                                                                          
Epoch 2/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.02it/s, loss=2.3618, acc=50.49%] 
Epoch 2/10 - Train Loss: 2.3618, Train Acc: 50.49%, Test Acc: 62.03%
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_MAG_epoch2.pth
Epoch 3/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.04it/s, loss=2.0094, acc=57.04%]
Epoch 3/10 - Train Loss: 2.0094, Train Acc: 57.04%, Test Acc: 64.81%                                                                          
Epoch 4/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.04it/s, loss=1.7708, acc=61.39%] 
Epoch 4/10 - Train Loss: 1.7708, Train Acc: 61.39%, Test Acc: 66.37%
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_MAG_epoch4.pth
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.05it/s, loss=1.6209, acc=64.40%]
Epoch 5/10 - Train Loss: 1.6209, Train Acc: 64.40%, Test Acc: 67.47%                                                                          
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.04it/s, loss=1.4695, acc=66.77%] 
Epoch 6/10 - Train Loss: 1.4695, Train Acc: 66.77%, Test Acc: 68.27%                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_MAG_epoch6.pth
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.05it/s, loss=1.4179, acc=68.30%]
Epoch 7/10 - Train Loss: 1.4179, Train Acc: 68.30%, Test Acc: 68.67%                                                                          
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.04it/s, loss=1.3387, acc=70.05%] 
Epoch 8/10 - Train Loss: 1.3387, Train Acc: 70.05%, Test Acc: 68.96%                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_MAG_epoch8.pth
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.05it/s, loss=1.3150, acc=70.44%]
Epoch 9/10 - Train Loss: 1.3150, Train Acc: 70.44%, Test Acc: 69.12%                                                                          
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.04it/s, loss=1.2956, acc=71.30%] 
Epoch 10/10 - Train Loss: 1.2956, Train Acc: 71.30%, Test Acc: 69.22%                                                                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_MAG_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_MAG_best.pth
  Best Accuracy: 69.22%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 69.22%                                                                                                                
✓ Model Size: 81.44 MB
✓ Average Inference Time: 0.0143 ms
✓ FLOPs: 3.05 GFLOPs

================================================================================
COMPARISON: MAGNITUDE-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.12   |    69.22   | -9.38                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         81.44   |    81.44   | -16.25 (-16.6%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0281 |          0.0253 |     0.0143 | -0.0137                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.05   |     3.05   | -1.09 (-26.3%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS7_ResNet50_ImageNet\TS7_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================



***********************


================================================================================
TEST SCENARIO TS7: TAYLOR GRADIENT-BASED PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS7\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                             
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0287 ms
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
Computing gradients:  99%|█████████████████████████████████████████████████████████████████████████████████▏| 99/100 [17:10<00:10, 10.41s/it] 
✓ Gradients computed on 100 batches
Using GroupTaylorImportance

Initializing Torch-Pruning with Taylor importance...

Applying pruning...

✓ Taylor pruning completed
  Parameters before: 25,557,032
  Parameters after: 21,548,673
  Parameters removed: 4,008,359 (15.68%)

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 0.88%                                                                                                                
✓ Model Size: 82.39 MB
✓ Average Inference Time: 0.0183 ms
✓ FLOPs: 3.22 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [07:34<00:00, 11.36s/it, loss=1.8801, acc=59.59%] 
Epoch 1/10 - Train Loss: 1.8801, Train Acc: 59.59%, Test Acc: 71.94%                                                                          
Epoch 2/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [07:33<00:00, 11.33s/it, loss=1.4431, acc=67.80%] 
Epoch 2/10 - Train Loss: 1.4431, Train Acc: 67.80%, Test Acc: 73.64%                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_TAY_epoch2.pth
Epoch 3/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [07:33<00:00, 11.33s/it, loss=1.2194, acc=71.68%]
Epoch 3/10 - Train Loss: 1.2194, Train Acc: 71.68%, Test Acc: 74.69%                                                                          
Epoch 4/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [07:33<00:00, 11.35s/it, loss=1.0703, acc=75.09%] 
Epoch 4/10 - Train Loss: 1.0703, Train Acc: 75.09%, Test Acc: 75.42%                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_TAY_epoch4.pth
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [07:33<00:00, 11.35s/it, loss=0.9912, acc=77.43%]
Epoch 5/10 - Train Loss: 0.9912, Train Acc: 77.43%, Test Acc: 75.61%                                                                          
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [07:33<00:00, 11.33s/it, loss=0.8880, acc=78.88%] 
Epoch 6/10 - Train Loss: 0.8880, Train Acc: 78.88%, Test Acc: 76.07%                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_TAY_epoch6.pth
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [07:34<00:00, 11.37s/it, loss=0.8607, acc=80.58%]
Epoch 7/10 - Train Loss: 0.8607, Train Acc: 80.58%, Test Acc: 76.32%                                                                          
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [07:33<00:00, 11.34s/it, loss=0.8242, acc=80.93%] 
Epoch 8/10 - Train Loss: 0.8242, Train Acc: 80.93%, Test Acc: 76.29%                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_TAY_epoch8.pth
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████| 40/40 [07:34<00:00, 11.36s/it, loss=0.7835, acc=82.20%]
Epoch 9/10 - Train Loss: 0.7835, Train Acc: 82.20%, Test Acc: 76.51%                                                                          
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [07:33<00:00, 11.34s/it, loss=0.7778, acc=82.47%] 
Epoch 10/10 - Train Loss: 0.7778, Train Acc: 82.47%, Test Acc: 76.50%                                                                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_TAY_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_TAY_best.pth
  Best Accuracy: 76.51%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 76.50%                                                                                                                
✓ Model Size: 82.39 MB
✓ Average Inference Time: 0.0188 ms
✓ FLOPs: 3.22 GFLOPs

================================================================================
COMPARISON: TAYLOR GRADIENT-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.88   |    76.5    | -2.10                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         82.39   |    82.39   | -15.30 (-15.7%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0287 |          0.0183 |     0.0188 | -0.0098                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.22   |     3.22   | -0.92 (-22.2%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS7_ResNet50_ImageNet\TS7_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================