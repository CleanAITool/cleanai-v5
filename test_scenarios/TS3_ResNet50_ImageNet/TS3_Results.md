================================================================================
TEST SCENARIO TS3: NEURON COVERAGE PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS3\ResNet50_ImageNet_FT_best.pth
  - Epoch: 0
  - Test Accuracy: 78.59%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                      
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0275 ms
✓ FLOPs: 4.13 GFLOPs

================================================================================
APPLYING NEURON COVERAGE PRUNING
================================================================================
Pruning Ratio: 10.0%
Global Pruning: False
Iterative Steps: 1
Coverage Metric: normalized_mean
Max Calibration Batches: 100
Using static coverage importance (computes once)

Initializing Torch-Pruning with:
  Importance method: coverage
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
  conv1: coverage shape torch.Size([64]), min=-0.719283, max=1.000000, mean=0.007738
  layer1.0.conv1: coverage shape torch.Size([64]), min=-0.680190, max=1.000000, mean=-0.109356
  layer1.0.conv2: coverage shape torch.Size([64]), min=-2.973168, max=1.000000, mean=0.121098
  layer1.0.conv3: coverage shape torch.Size([256]), min=-0.757856, max=1.000000, mean=-0.036497
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=-1.998780, max=1.000000, mean=-0.093964
  layer1.1.conv1: coverage shape torch.Size([64]), min=-1.895094, max=1.000000, mean=-0.059053
  layer1.1.conv2: coverage shape torch.Size([64]), min=-1.529547, max=1.000000, mean=-0.103429
  layer1.1.conv3: coverage shape torch.Size([256]), min=-1.593803, max=1.000000, mean=-0.105997
  layer1.2.conv1: coverage shape torch.Size([64]), min=-0.879929, max=1.000000, mean=-0.115961
  layer1.2.conv2: coverage shape torch.Size([64]), min=-1.831300, max=1.000000, mean=-0.215085
  layer1.2.conv3: coverage shape torch.Size([256]), min=-1.091953, max=1.000000, mean=-0.034257
  layer2.0.conv1: coverage shape torch.Size([128]), min=-1.019463, max=1.000000, mean=-0.089013
  layer2.0.conv2: coverage shape torch.Size([128]), min=-1.427794, max=1.000000, mean=-0.151650
  layer2.0.conv3: coverage shape torch.Size([512]), min=-0.669428, max=1.000000, mean=0.016029
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=-0.821975, max=1.000000, mean=-0.056081
  layer2.1.conv1: coverage shape torch.Size([128]), min=-0.986537, max=1.000000, mean=-0.094143
  layer2.1.conv2: coverage shape torch.Size([128]), min=-0.743421, max=1.000000, mean=0.069035
  layer2.1.conv3: coverage shape torch.Size([512]), min=-0.912380, max=1.000000, mean=-0.044287
  layer2.2.conv1: coverage shape torch.Size([128]), min=-1.438262, max=1.000000, mean=-0.005542
  layer2.2.conv2: coverage shape torch.Size([128]), min=-1.098086, max=1.000000, mean=-0.161881
  layer2.2.conv3: coverage shape torch.Size([512]), min=-0.948887, max=1.000000, mean=-0.002181
  layer2.3.conv1: coverage shape torch.Size([128]), min=-0.600289, max=1.000000, mean=-0.044135
  layer2.3.conv2: coverage shape torch.Size([128]), min=-0.854634, max=1.000000, mean=-0.127506
  layer2.3.conv3: coverage shape torch.Size([512]), min=-1.122708, max=1.000000, mean=-0.051406
  layer3.0.conv1: coverage shape torch.Size([256]), min=-1.501991, max=1.000000, mean=-0.196536
  layer3.0.conv2: coverage shape torch.Size([256]), min=-1.555279, max=1.000000, mean=-0.092179
  layer3.0.conv3: coverage shape torch.Size([1024]), min=-1.174786, max=1.000000, mean=-0.020720
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=-1.306983, max=1.000000, mean=-0.148680
  layer3.1.conv1: coverage shape torch.Size([256]), min=-1.816209, max=1.000000, mean=-0.068257
  layer3.1.conv2: coverage shape torch.Size([256]), min=-2.275693, max=1.000000, mean=-0.071589
  layer3.1.conv3: coverage shape torch.Size([1024]), min=-1.960462, max=1.000000, mean=-0.191199
  layer3.2.conv1: coverage shape torch.Size([256]), min=-1.419222, max=1.000000, mean=-0.249022
  layer3.2.conv2: coverage shape torch.Size([256]), min=-1.240412, max=1.000000, mean=-0.199264
  layer3.2.conv3: coverage shape torch.Size([1024]), min=-1.188677, max=1.000000, mean=-0.051439
  layer3.3.conv1: coverage shape torch.Size([256]), min=-1.148759, max=1.000000, mean=-0.128136
  layer3.3.conv2: coverage shape torch.Size([256]), min=-0.715102, max=1.000000, mean=-0.088815
  layer3.3.conv3: coverage shape torch.Size([1024]), min=-1.390955, max=1.000000, mean=-0.087932
  layer3.4.conv1: coverage shape torch.Size([256]), min=-0.842267, max=1.000000, mean=-0.256144
  layer3.4.conv2: coverage shape torch.Size([256]), min=-1.098190, max=1.000000, mean=-0.255260
  layer3.4.conv3: coverage shape torch.Size([1024]), min=-0.732722, max=1.000000, mean=-0.021697
  layer3.5.conv1: coverage shape torch.Size([256]), min=-1.072329, max=1.000000, mean=-0.174897
  layer3.5.conv2: coverage shape torch.Size([256]), min=-1.386984, max=1.000000, mean=-0.231676
  layer3.5.conv3: coverage shape torch.Size([1024]), min=-1.616471, max=1.000000, mean=-0.264879
  layer4.0.conv1: coverage shape torch.Size([512]), min=-1.410840, max=1.000000, mean=-0.603107
  layer4.0.conv2: coverage shape torch.Size([512]), min=-2.159355, max=1.000000, mean=-0.414081
  layer4.0.conv3: coverage shape torch.Size([2048]), min=-0.937292, max=1.000000, mean=-0.086154
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=-1.224207, max=1.000000, mean=-0.160178
  layer4.1.conv1: coverage shape torch.Size([512]), min=-1.798952, max=1.000000, mean=-0.243148
  layer4.1.conv2: coverage shape torch.Size([512]), min=-1.212858, max=1.000000, mean=-0.206669
  layer4.1.conv3: coverage shape torch.Size([2048]), min=-0.725998, max=1.000000, mean=-0.146413
  layer4.2.conv1: coverage shape torch.Size([512]), min=-1.422669, max=1.000000, mean=-0.224944
  layer4.2.conv2: coverage shape torch.Size([512]), min=-2.206686, max=1.000000, mean=-0.299195
  layer4.2.conv3: coverage shape torch.Size([2048]), min=-2.467433, max=1.000000, mean=-0.836297
  fc: coverage shape torch.Size([1000]), min=-0.817519, max=1.000000, mean=0.000403

------------------------------------------------------------
Coverage Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Coverage - Min: -0.719283, Max: 1.000000, Mean: 0.007738
  Zero coverage neurons: 0

layer1.0.conv1:
  Channels: 64
  Coverage - Min: -0.680190, Max: 1.000000, Mean: -0.109356
  Zero coverage neurons: 0

layer1.0.conv2:
  Channels: 64
  Coverage - Min: -2.973168, Max: 1.000000, Mean: 0.121098
  Zero coverage neurons: 0

layer1.0.conv3:
  Channels: 256
  Coverage - Min: -0.757856, Max: 1.000000, Mean: -0.036497
  Zero coverage neurons: 0

layer1.0.downsample.0:
  Channels: 256
  Coverage - Min: -1.998780, Max: 1.000000, Mean: -0.093964
  Zero coverage neurons: 0

layer1.1.conv1:
  Channels: 64
  Coverage - Min: -1.895094, Max: 1.000000, Mean: -0.059053
  Zero coverage neurons: 0

layer1.1.conv2:
  Channels: 64
  Coverage - Min: -1.529547, Max: 1.000000, Mean: -0.103429
  Zero coverage neurons: 0

layer1.1.conv3:
  Channels: 256
  Coverage - Min: -1.593803, Max: 1.000000, Mean: -0.105997
  Zero coverage neurons: 0

layer1.2.conv1:
  Channels: 64
  Coverage - Min: -0.879929, Max: 1.000000, Mean: -0.115961
  Zero coverage neurons: 0

layer1.2.conv2:
  Channels: 64
  Coverage - Min: -1.831300, Max: 1.000000, Mean: -0.215085
  Zero coverage neurons: 0

layer1.2.conv3:
  Channels: 256
  Coverage - Min: -1.091953, Max: 1.000000, Mean: -0.034257
  Zero coverage neurons: 0

layer2.0.conv1:
  Channels: 128
  Coverage - Min: -1.019463, Max: 1.000000, Mean: -0.089013
  Zero coverage neurons: 0

layer2.0.conv2:
  Channels: 128
  Coverage - Min: -1.427794, Max: 1.000000, Mean: -0.151650
  Zero coverage neurons: 0

layer2.0.conv3:
  Channels: 512
  Coverage - Min: -0.669428, Max: 1.000000, Mean: 0.016029
  Zero coverage neurons: 0

layer2.0.downsample.0:
  Channels: 512
  Coverage - Min: -0.821975, Max: 1.000000, Mean: -0.056081
  Zero coverage neurons: 0

layer2.1.conv1:
  Channels: 128
  Coverage - Min: -0.986537, Max: 1.000000, Mean: -0.094143
  Zero coverage neurons: 0

layer2.1.conv2:
  Channels: 128
  Coverage - Min: -0.743421, Max: 1.000000, Mean: 0.069035
  Zero coverage neurons: 0

layer2.1.conv3:
  Channels: 512
  Coverage - Min: -0.912380, Max: 1.000000, Mean: -0.044287
  Zero coverage neurons: 0

layer2.2.conv1:
  Channels: 128
  Coverage - Min: -1.438262, Max: 1.000000, Mean: -0.005542
  Zero coverage neurons: 0

layer2.2.conv2:
  Channels: 128
  Coverage - Min: -1.098086, Max: 1.000000, Mean: -0.161881
  Zero coverage neurons: 0

layer2.2.conv3:
  Channels: 512
  Coverage - Min: -0.948887, Max: 1.000000, Mean: -0.002181
  Zero coverage neurons: 0

layer2.3.conv1:
  Channels: 128
  Coverage - Min: -0.600289, Max: 1.000000, Mean: -0.044135
  Zero coverage neurons: 0

layer2.3.conv2:
  Channels: 128
  Coverage - Min: -0.854634, Max: 1.000000, Mean: -0.127506
  Zero coverage neurons: 0

layer2.3.conv3:
  Channels: 512
  Coverage - Min: -1.122708, Max: 1.000000, Mean: -0.051406
  Zero coverage neurons: 0

layer3.0.conv1:
  Channels: 256
  Coverage - Min: -1.501991, Max: 1.000000, Mean: -0.196536
  Zero coverage neurons: 0

layer3.0.conv2:
  Channels: 256
  Coverage - Min: -1.555279, Max: 1.000000, Mean: -0.092179
  Zero coverage neurons: 0

layer3.0.conv3:
  Channels: 1024
  Coverage - Min: -1.174786, Max: 1.000000, Mean: -0.020720
  Zero coverage neurons: 0

layer3.0.downsample.0:
  Channels: 1024
  Coverage - Min: -1.306983, Max: 1.000000, Mean: -0.148680
  Zero coverage neurons: 0

layer3.1.conv1:
  Channels: 256
  Coverage - Min: -1.816209, Max: 1.000000, Mean: -0.068257
  Zero coverage neurons: 0

layer3.1.conv2:
  Channels: 256
  Coverage - Min: -2.275693, Max: 1.000000, Mean: -0.071589
  Zero coverage neurons: 0

layer3.1.conv3:
  Channels: 1024
  Coverage - Min: -1.960462, Max: 1.000000, Mean: -0.191199
  Zero coverage neurons: 0

layer3.2.conv1:
  Channels: 256
  Coverage - Min: -1.419222, Max: 1.000000, Mean: -0.249022
  Zero coverage neurons: 0

layer3.2.conv2:
  Channels: 256
  Coverage - Min: -1.240412, Max: 1.000000, Mean: -0.199264
  Zero coverage neurons: 0

layer3.2.conv3:
  Channels: 1024
  Coverage - Min: -1.188677, Max: 1.000000, Mean: -0.051439
  Zero coverage neurons: 0

layer3.3.conv1:
  Channels: 256
  Coverage - Min: -1.148759, Max: 1.000000, Mean: -0.128136
  Zero coverage neurons: 0

layer3.3.conv2:
  Channels: 256
  Coverage - Min: -0.715102, Max: 1.000000, Mean: -0.088815
  Zero coverage neurons: 0

layer3.3.conv3:
  Channels: 1024
  Coverage - Min: -1.390955, Max: 1.000000, Mean: -0.087932
  Zero coverage neurons: 0

layer3.4.conv1:
  Channels: 256
  Coverage - Min: -0.842267, Max: 1.000000, Mean: -0.256144
  Zero coverage neurons: 0

layer3.4.conv2:
  Channels: 256
  Coverage - Min: -1.098190, Max: 1.000000, Mean: -0.255260
  Zero coverage neurons: 0

layer3.4.conv3:
  Channels: 1024
  Coverage - Min: -0.732722, Max: 1.000000, Mean: -0.021697
  Zero coverage neurons: 0

layer3.5.conv1:
  Channels: 256
  Coverage - Min: -1.072329, Max: 1.000000, Mean: -0.174897
  Zero coverage neurons: 0

layer3.5.conv2:
  Channels: 256
  Coverage - Min: -1.386984, Max: 1.000000, Mean: -0.231676
  Zero coverage neurons: 0

layer3.5.conv3:
  Channels: 1024
  Coverage - Min: -1.616471, Max: 1.000000, Mean: -0.264879
  Zero coverage neurons: 0

layer4.0.conv1:
  Channels: 512
  Coverage - Min: -1.410840, Max: 1.000000, Mean: -0.603107
  Zero coverage neurons: 0

layer4.0.conv2:
  Channels: 512
  Coverage - Min: -2.159355, Max: 1.000000, Mean: -0.414081
  Zero coverage neurons: 0

layer4.0.conv3:
  Channels: 2048
  Coverage - Min: -0.937292, Max: 1.000000, Mean: -0.086154
  Zero coverage neurons: 0

layer4.0.downsample.0:
  Channels: 2048
  Coverage - Min: -1.224207, Max: 1.000000, Mean: -0.160178
  Zero coverage neurons: 0

layer4.1.conv1:
  Channels: 512
  Coverage - Min: -1.798952, Max: 1.000000, Mean: -0.243148
  Zero coverage neurons: 0

layer4.1.conv2:
  Channels: 512
  Coverage - Min: -1.212858, Max: 1.000000, Mean: -0.206669
  Zero coverage neurons: 0

layer4.1.conv3:
  Channels: 2048
  Coverage - Min: -0.725998, Max: 1.000000, Mean: -0.146413
  Zero coverage neurons: 0

layer4.2.conv1:
  Channels: 512
  Coverage - Min: -1.422669, Max: 1.000000, Mean: -0.224944
  Zero coverage neurons: 0

layer4.2.conv2:
  Channels: 512
  Coverage - Min: -2.206686, Max: 1.000000, Mean: -0.299195
  Zero coverage neurons: 0

layer4.2.conv3:
  Channels: 2048
  Coverage - Min: -2.467433, Max: 1.000000, Mean: -0.836297
  Zero coverage neurons: 0

fc:
  Channels: 1000
  Coverage - Min: -0.817519, Max: 1.000000, Mean: 0.000403
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
✓ Pruned Model Accuracy: 0.21%                                                                                         
✓ Model Size: 79.67 MB
✓ Average Inference Time: 1.3372 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.69s/it, loss=2.2637, acc=52.36%] 
Epoch 1/10 - Train Loss: 2.2637, Train Acc: 52.36%, Test Acc: 66.48%                                                   
Epoch 2/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.6494, acc=63.04%] 
Epoch 2/10 - Train Loss: 1.6494, Train Acc: 63.04%, Test Acc: 69.19%                                                   
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_epoch2.pth
Epoch 3/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.4054, acc=67.59%]
Epoch 3/10 - Train Loss: 1.4054, Train Acc: 67.59%, Test Acc: 70.87%
Epoch 4/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.2894, acc=70.22%] 
Epoch 4/10 - Train Loss: 1.2894, Train Acc: 70.22%, Test Acc: 71.95%
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_epoch4.pth
Epoch 5/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.1317, acc=73.64%]
Epoch 5/10 - Train Loss: 1.1317, Train Acc: 73.64%, Test Acc: 72.49%                                                   
Epoch 6/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.0440, acc=75.96%] 
Epoch 6/10 - Train Loss: 1.0440, Train Acc: 75.96%, Test Acc: 73.26%                                                   
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_epoch6.pth
Epoch 7/10: 100%|████████████████████████████████████████████| 40/40 [01:06<00:00,  1.67s/it, loss=0.9652, acc=77.65%]
Epoch 7/10 - Train Loss: 0.9652, Train Acc: 77.65%, Test Acc: 73.64%                                                   
Epoch 8/10: 100%|████████████████████████████████████████████| 40/40 [01:06<00:00,  1.67s/it, loss=0.9349, acc=78.66%] 
Epoch 8/10 - Train Loss: 0.9349, Train Acc: 78.66%, Test Acc: 73.81%                                                   
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_epoch8.pth
Epoch 9/10: 100%|████████████████████████████████████████████| 40/40 [01:06<00:00,  1.67s/it, loss=0.9014, acc=79.33%]
Epoch 9/10 - Train Loss: 0.9014, Train Acc: 79.33%, Test Acc: 73.87%                                                   
Epoch 10/10: 100%|███████████████████████████████████████████| 40/40 [01:06<00:00,  1.67s/it, loss=0.9215, acc=79.10%] 
Epoch 10/10 - Train Loss: 0.9215, Train Acc: 79.10%, Test Acc: 73.88%                                                  
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_NC_best.pth
  Best Accuracy: 73.88%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 73.88%                                                                                         
✓ Model Size: 79.67 MB
✓ Average Inference Time: 1.3371 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
COMPARISON: NEURON COVERAGE PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.21   |    73.88   | -4.72                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         79.67   |    79.67   | -18.02 (-18.4%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0275 |          1.3372 |     1.3371 | +1.3096                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.34   |     3.34   | -0.79 (-19.1%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: C:\source\repos\cleanai-v5\test_scenarios\TS3_ResNet50_ImageNet\TS3_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================




*******************************************





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
✓ Average Inference Time: 0.0271 ms
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
  conv1: coverage shape torch.Size([64]), min=0.000062, max=0.457685, mean=0.031280
  layer1.0.conv1: coverage shape torch.Size([64]), min=0.867421, max=44.562344, mean=9.883335
  layer1.0.conv2: coverage shape torch.Size([64]), min=0.048937, max=25.866533, mean=3.915803
  layer1.0.conv3: coverage shape torch.Size([256]), min=0.000000, max=10.939842, mean=2.152102
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=0.000000, max=61.208370, mean=5.204352
  layer1.1.conv1: coverage shape torch.Size([64]), min=0.000000, max=6.713943, mean=1.258899
  layer1.1.conv2: coverage shape torch.Size([64]), min=0.127370, max=6.323568, mean=1.655549
  layer1.1.conv3: coverage shape torch.Size([256]), min=0.001651, max=7.036158, mean=1.304104
  layer1.2.conv1: coverage shape torch.Size([64]), min=0.029590, max=4.758059, mean=1.477254
  layer1.2.conv2: coverage shape torch.Size([64]), min=0.069897, max=6.587742, mean=1.332427
  layer1.2.conv3: coverage shape torch.Size([256]), min=0.002872, max=5.846490, mean=0.939758
  layer2.0.conv1: coverage shape torch.Size([128]), min=0.000645, max=5.207532, mean=1.523076
  layer2.0.conv2: coverage shape torch.Size([128]), min=0.014667, max=6.549282, mean=1.468072
  layer2.0.conv3: coverage shape torch.Size([512]), min=0.000002, max=8.116773, mean=1.116648
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=0.000001, max=8.565344, mean=1.163757
  layer2.1.conv1: coverage shape torch.Size([128]), min=0.079849, max=7.683916, mean=2.209462
  layer2.1.conv2: coverage shape torch.Size([128]), min=0.015511, max=10.110276, mean=1.950205
  layer2.1.conv3: coverage shape torch.Size([512]), min=0.000143, max=2.398072, mean=0.467714
  layer2.2.conv1: coverage shape torch.Size([128]), min=0.006555, max=5.486268, mean=1.218076
  layer2.2.conv2: coverage shape torch.Size([128]), min=0.008411, max=6.198031, mean=1.373755
  layer2.2.conv3: coverage shape torch.Size([512]), min=0.001242, max=3.077337, mean=0.579381
  layer2.3.conv1: coverage shape torch.Size([128]), min=0.023961, max=6.827947, mean=1.273080
  layer2.3.conv2: coverage shape torch.Size([128]), min=0.012208, max=5.932532, mean=1.380650
  layer2.3.conv3: coverage shape torch.Size([512]), min=0.000633, max=3.384783, mean=0.602474
  layer3.0.conv1: coverage shape torch.Size([256]), min=0.001804, max=6.383680, mean=1.402852
  layer3.0.conv2: coverage shape torch.Size([256]), min=0.004279, max=4.455744, mean=0.891533
  layer3.0.conv3: coverage shape torch.Size([1024]), min=0.000289, max=4.705327, mean=0.895029
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=0.000121, max=4.077484, mean=0.831802
  layer3.1.conv1: coverage shape torch.Size([256]), min=0.000396, max=9.748751, mean=1.413218
  layer3.1.conv2: coverage shape torch.Size([256]), min=0.001953, max=11.070211, mean=1.028116
  layer3.1.conv3: coverage shape torch.Size([1024]), min=0.000062, max=2.092516, mean=0.318429
  layer3.2.conv1: coverage shape torch.Size([256]), min=0.000476, max=5.960211, mean=1.283052
  layer3.2.conv2: coverage shape torch.Size([256]), min=0.004155, max=3.086331, mean=0.708978
  layer3.2.conv3: coverage shape torch.Size([1024]), min=0.000111, max=2.030435, mean=0.287278
  layer3.3.conv1: coverage shape torch.Size([256]), min=0.000117, max=5.973184, mean=1.164334
  layer3.3.conv2: coverage shape torch.Size([256]), min=0.010427, max=5.394636, mean=0.820471
  layer3.3.conv3: coverage shape torch.Size([1024]), min=0.000754, max=1.959350, mean=0.335739
  layer3.4.conv1: coverage shape torch.Size([256]), min=0.011238, max=3.743127, mean=1.143842
  layer3.4.conv2: coverage shape torch.Size([256]), min=0.008095, max=3.056389, mean=0.863761
  layer3.4.conv3: coverage shape torch.Size([1024]), min=0.000693, max=2.885629, mean=0.299688
  layer3.5.conv1: coverage shape torch.Size([256]), min=0.000835, max=5.945729, mean=1.247592
  layer3.5.conv2: coverage shape torch.Size([256]), min=0.001880, max=3.800623, mean=0.845828
  layer3.5.conv3: coverage shape torch.Size([1024]), min=0.001420, max=2.127687, mean=0.429533
  layer4.0.conv1: coverage shape torch.Size([512]), min=0.043903, max=3.691809, mean=1.628092
  layer4.0.conv2: coverage shape torch.Size([512]), min=0.004971, max=4.814616, mean=0.951439
  layer4.0.conv3: coverage shape torch.Size([2048]), min=0.000055, max=5.297915, mean=0.589517
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=0.000213, max=3.841798, mean=0.576783
  layer4.1.conv1: coverage shape torch.Size([512]), min=0.104963, max=9.322679, mean=1.404315
  layer4.1.conv2: coverage shape torch.Size([512]), min=0.002489, max=5.465999, mean=1.051868
  layer4.1.conv3: coverage shape torch.Size([2048]), min=0.000325, max=2.933958, mean=0.447038
  layer4.2.conv1: coverage shape torch.Size([512]), min=0.086322, max=7.494973, mean=1.339737
  layer4.2.conv2: coverage shape torch.Size([512]), min=0.048915, max=4.689241, mean=0.663621
  layer4.2.conv3: coverage shape torch.Size([2048]), min=0.000395, max=1.061287, mean=0.360751
  fc: coverage shape torch.Size([1000]), min=0.000071, max=0.285496, mean=0.060372

------------------------------------------------------------
Activation Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Activation - Min: 0.000062, Max: 0.457685, Mean: 0.031280

layer1.0.conv1:
  Channels: 64
  Activation - Min: 0.867421, Max: 44.562344, Mean: 9.883335

layer1.0.conv2:
  Channels: 64
  Activation - Min: 0.048937, Max: 25.866533, Mean: 3.915803

layer1.0.conv3:
  Channels: 256
  Activation - Min: 0.000000, Max: 10.939842, Mean: 2.152102

layer1.0.downsample.0:
  Channels: 256
  Activation - Min: 0.000000, Max: 61.208370, Mean: 5.204352

layer1.1.conv1:
  Channels: 64
  Activation - Min: 0.000000, Max: 6.713943, Mean: 1.258899

layer1.1.conv2:
  Channels: 64
  Activation - Min: 0.127370, Max: 6.323568, Mean: 1.655549

layer1.1.conv3:
  Channels: 256
  Activation - Min: 0.001651, Max: 7.036158, Mean: 1.304104

layer1.2.conv1:
  Channels: 64
  Activation - Min: 0.029590, Max: 4.758059, Mean: 1.477254

layer1.2.conv2:
  Channels: 64
  Activation - Min: 0.069897, Max: 6.587742, Mean: 1.332427

layer1.2.conv3:
  Channels: 256
  Activation - Min: 0.002872, Max: 5.846490, Mean: 0.939758

layer2.0.conv1:
  Channels: 128
  Activation - Min: 0.000645, Max: 5.207532, Mean: 1.523076

layer2.0.conv2:
  Channels: 128
  Activation - Min: 0.014667, Max: 6.549282, Mean: 1.468072

layer2.0.conv3:
  Channels: 512
  Activation - Min: 0.000002, Max: 8.116773, Mean: 1.116648

layer2.0.downsample.0:
  Channels: 512
  Activation - Min: 0.000001, Max: 8.565344, Mean: 1.163757

layer2.1.conv1:
  Channels: 128
  Activation - Min: 0.079849, Max: 7.683916, Mean: 2.209462

layer2.1.conv2:
  Channels: 128
  Activation - Min: 0.015511, Max: 10.110276, Mean: 1.950205

layer2.1.conv3:
  Channels: 512
  Activation - Min: 0.000143, Max: 2.398072, Mean: 0.467714

layer2.2.conv1:
  Channels: 128
  Activation - Min: 0.006555, Max: 5.486268, Mean: 1.218076

layer2.2.conv2:
  Channels: 128
  Activation - Min: 0.008411, Max: 6.198031, Mean: 1.373755

layer2.2.conv3:
  Channels: 512
  Activation - Min: 0.001242, Max: 3.077337, Mean: 0.579381

layer2.3.conv1:
  Channels: 128
  Activation - Min: 0.023961, Max: 6.827947, Mean: 1.273080

layer2.3.conv2:
  Channels: 128
  Activation - Min: 0.012208, Max: 5.932532, Mean: 1.380650

layer2.3.conv3:
  Channels: 512
  Activation - Min: 0.000633, Max: 3.384783, Mean: 0.602474

layer3.0.conv1:
  Channels: 256
  Activation - Min: 0.001804, Max: 6.383680, Mean: 1.402852

layer3.0.conv2:
  Channels: 256
  Activation - Min: 0.004279, Max: 4.455744, Mean: 0.891533

layer3.0.conv3:
  Channels: 1024
  Activation - Min: 0.000289, Max: 4.705327, Mean: 0.895029

layer3.0.downsample.0:
  Channels: 1024
  Activation - Min: 0.000121, Max: 4.077484, Mean: 0.831802

layer3.1.conv1:
  Channels: 256
  Activation - Min: 0.000396, Max: 9.748751, Mean: 1.413218

layer3.1.conv2:
  Channels: 256
  Activation - Min: 0.001953, Max: 11.070211, Mean: 1.028116

layer3.1.conv3:
  Channels: 1024
  Activation - Min: 0.000062, Max: 2.092516, Mean: 0.318429

layer3.2.conv1:
  Channels: 256
  Activation - Min: 0.000476, Max: 5.960211, Mean: 1.283052

layer3.2.conv2:
  Channels: 256
  Activation - Min: 0.004155, Max: 3.086331, Mean: 0.708978

layer3.2.conv3:
  Channels: 1024
  Activation - Min: 0.000111, Max: 2.030435, Mean: 0.287278

layer3.3.conv1:
  Channels: 256
  Activation - Min: 0.000117, Max: 5.973184, Mean: 1.164334

layer3.3.conv2:
  Channels: 256
  Activation - Min: 0.010427, Max: 5.394636, Mean: 0.820471

layer3.3.conv3:
  Channels: 1024
  Activation - Min: 0.000754, Max: 1.959350, Mean: 0.335739

layer3.4.conv1:
  Channels: 256
  Activation - Min: 0.011238, Max: 3.743127, Mean: 1.143842

layer3.4.conv2:
  Channels: 256
  Activation - Min: 0.008095, Max: 3.056389, Mean: 0.863761

layer3.4.conv3:
  Channels: 1024
  Activation - Min: 0.000693, Max: 2.885629, Mean: 0.299688

layer3.5.conv1:
  Channels: 256
  Activation - Min: 0.000835, Max: 5.945729, Mean: 1.247592

layer3.5.conv2:
  Channels: 256
  Activation - Min: 0.001880, Max: 3.800623, Mean: 0.845828

layer3.5.conv3:
  Channels: 1024
  Activation - Min: 0.001420, Max: 2.127687, Mean: 0.429533

layer4.0.conv1:
  Channels: 512
  Activation - Min: 0.043903, Max: 3.691809, Mean: 1.628092

layer4.0.conv2:
  Channels: 512
  Activation - Min: 0.004971, Max: 4.814616, Mean: 0.951439

layer4.0.conv3:
  Channels: 2048
  Activation - Min: 0.000055, Max: 5.297915, Mean: 0.589517

layer4.0.downsample.0:
  Channels: 2048
  Activation - Min: 0.000213, Max: 3.841798, Mean: 0.576783

layer4.1.conv1:
  Channels: 512
  Activation - Min: 0.104963, Max: 9.322679, Mean: 1.404315

layer4.1.conv2:
  Channels: 512
  Activation - Min: 0.002489, Max: 5.465999, Mean: 1.051868

layer4.1.conv3:
  Channels: 2048
  Activation - Min: 0.000325, Max: 2.933958, Mean: 0.447038

layer4.2.conv1:
  Channels: 512
  Activation - Min: 0.086322, Max: 7.494973, Mean: 1.339737

layer4.2.conv2:
  Channels: 512
  Activation - Min: 0.048915, Max: 4.689241, Mean: 0.663621

layer4.2.conv3:
  Channels: 2048
  Activation - Min: 0.000395, Max: 1.061287, Mean: 0.360751

fc:
  Channels: 1000
  Activation - Min: 0.000071, Max: 0.285496, Mean: 0.060372
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
✓ Pruned Model Accuracy: 0.09%                                                                                         
✓ Model Size: 79.67 MB
✓ Average Inference Time: 1.3359 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.69s/it, loss=3.3652, acc=39.84%] 
Epoch 1/10 - Train Loss: 3.3652, Train Acc: 39.84%, Test Acc: 57.84%                                                   
Epoch 2/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=2.2057, acc=53.94%] 
Epoch 2/10 - Train Loss: 2.2057, Train Acc: 53.94%, Test Acc: 62.92%                                                   
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_epoch2.pth
Epoch 3/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.8783, acc=59.91%]
Epoch 3/10 - Train Loss: 1.8783, Train Acc: 59.91%, Test Acc: 65.14%                                                   
Epoch 4/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.6606, acc=63.97%] 
Epoch 4/10 - Train Loss: 1.6606, Train Acc: 63.97%, Test Acc: 66.69%                                                   
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_epoch4.pth
Epoch 5/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.4640, acc=67.49%]
Epoch 5/10 - Train Loss: 1.4640, Train Acc: 67.49%, Test Acc: 67.92%                                                   
Epoch 6/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.3395, acc=69.81%] 
Epoch 6/10 - Train Loss: 1.3395, Train Acc: 69.81%, Test Acc: 68.50%                                                   
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_epoch6.pth
Epoch 7/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.2436, acc=72.12%]
Epoch 7/10 - Train Loss: 1.2436, Train Acc: 72.12%, Test Acc: 69.19%                                                   
Epoch 8/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.1790, acc=73.10%] 
Epoch 8/10 - Train Loss: 1.1790, Train Acc: 73.10%, Test Acc: 69.46%
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_epoch8.pth
Epoch 9/10: 100%|████████████████████████████████████████████| 40/40 [01:07<00:00,  1.69s/it, loss=1.2005, acc=73.30%]
Epoch 9/10 - Train Loss: 1.2005, Train Acc: 73.30%, Test Acc: 69.71%                                                   
Epoch 10/10: 100%|███████████████████████████████████████████| 40/40 [01:07<00:00,  1.68s/it, loss=1.1455, acc=74.82%] 
Epoch 10/10 - Train Loss: 1.1455, Train Acc: 74.82%, Test Acc: 69.67%                                                  
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_W_best.pth
  Best Accuracy: 69.71%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 69.67%                                                                                         
✓ Model Size: 79.67 MB
✓ Average Inference Time: 1.3387 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
COMPARISON: WANDA PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.09   |    69.67   | -8.93                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         79.67   |    79.67   | -18.02 (-18.4%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0271 |          1.3359 |     1.3387 | +1.3116                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.34   |     3.34   | -0.79 (-19.1%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS3_ResNet50_ImageNet\TS3_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================




*********************************************




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
✓ Average Inference Time: 0.0225 ms
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
✓ Average Inference Time: 0.0234 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|████████████████████████████████████████████| 40/40 [02:38<00:00,  3.96s/it, loss=2.2202, acc=56.33%]
Epoch 1/10 - Train Loss: 2.2202, Train Acc: 56.33%, Test Acc: 70.00%
Epoch 2/10: 100%|████████████████████████████████████████████| 40/40 [02:40<00:00,  4.00s/it, loss=1.5183, acc=65.23%]
Epoch 2/10 - Train Loss: 1.5183, Train Acc: 65.23%, Test Acc: 72.43%
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_epoch2.pth
Epoch 3/10: 100%|████████████████████████████████████████████| 40/40 [08:21<00:00, 12.54s/it, loss=1.3185, acc=69.81%]
Epoch 3/10 - Train Loss: 1.3185, Train Acc: 69.81%, Test Acc: 73.47%                                                                                          
Epoch 4/10: 100%|███████████████████████████████████████████████████████████████████████████████████| 40/40 [02:56<00:00,  4.41s/it, loss=1.2111, acc=72.47%] 
Epoch 4/10 - Train Loss: 1.2111, Train Acc: 72.47%, Test Acc: 74.09%
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_epoch4.pth
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████████████████████| 40/40 [02:56<00:00,  4.41s/it, loss=1.1315, acc=74.82%]
Epoch 5/10 - Train Loss: 1.1315, Train Acc: 74.82%, Test Acc: 74.59%                                                                                          
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████████████████████| 40/40 [02:56<00:00,  4.40s/it, loss=0.9894, acc=76.67%] 
Epoch 6/10 - Train Loss: 0.9894, Train Acc: 76.67%, Test Acc: 75.02%
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_epoch6.pth
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████████████████████| 40/40 [02:56<00:00,  4.41s/it, loss=0.9392, acc=77.98%]
Epoch 7/10 - Train Loss: 0.9392, Train Acc: 77.98%, Test Acc: 75.27%                                                                                          
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████████████████████| 40/40 [02:56<00:00,  4.40s/it, loss=0.9123, acc=79.03%] 
Epoch 8/10 - Train Loss: 0.9123, Train Acc: 79.03%, Test Acc: 75.49%                                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_epoch8.pth
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████████████████████| 40/40 [02:56<00:00,  4.41s/it, loss=0.9014, acc=78.90%]
Epoch 9/10 - Train Loss: 0.9014, Train Acc: 78.90%, Test Acc: 75.57%                                                                                          
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████████████████████| 40/40 [02:56<00:00,  4.40s/it, loss=0.8605, acc=79.75%] 
Epoch 10/10 - Train Loss: 0.8605, Train Acc: 79.75%, Test Acc: 75.49%
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_MAG_best.pth
  Best Accuracy: 75.57%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 75.49%                                                                                                                                
✓ Model Size: 79.67 MB
✓ Average Inference Time: 0.0138 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
COMPARISON: MAGNITUDE-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.36   |    75.49   | -3.10                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         79.67   |    79.67   | -18.02 (-18.4%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0225 |          0.0234 |     0.0138 | -0.0087                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.34   |     3.34   | -0.79 (-19.1%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: C:\source\repos\cleanai-v5\test_scenarios\TS3_ResNet50_ImageNet\TS3_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================





****************************************






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
✓ Average Inference Time: 0.0264 ms
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
Computing gradients:  99%|████████████████████████████████████████████████████████████████████████████████▏| 99/100 [11:36<00:07,  7.03s/it] 
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
✓ Pruned Model Accuracy: 2.43%                                                                                                               
✓ Model Size: 79.67 MB
✓ Average Inference Time: 0.0158 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [07:23<00:00, 11.09s/it, loss=1.9249, acc=58.33%] 
Epoch 1/10 - Train Loss: 1.9249, Train Acc: 58.33%, Test Acc: 70.24%                                                                         
Epoch 2/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [07:22<00:00, 11.07s/it, loss=1.4407, acc=66.76%] 
Epoch 2/10 - Train Loss: 1.4407, Train Acc: 66.76%, Test Acc: 72.22%                                                                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_epoch2.pth
Epoch 3/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [07:23<00:00, 11.08s/it, loss=1.2279, acc=71.24%]
Epoch 3/10 - Train Loss: 1.2279, Train Acc: 71.24%, Test Acc: 73.15%                                                                         
Epoch 4/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [07:25<00:00, 11.14s/it, loss=1.0729, acc=74.69%] 
Epoch 4/10 - Train Loss: 1.0729, Train Acc: 74.69%, Test Acc: 73.99%                                                                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_epoch4.pth
Epoch 5/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [07:23<00:00, 11.08s/it, loss=0.9578, acc=77.05%]
Epoch 5/10 - Train Loss: 0.9578, Train Acc: 77.05%, Test Acc: 74.41%                                                                         
Epoch 6/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [07:22<00:00, 11.06s/it, loss=0.8944, acc=79.10%] 
Epoch 6/10 - Train Loss: 0.8944, Train Acc: 79.10%, Test Acc: 74.73%                                                                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_epoch6.pth
Epoch 7/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [07:22<00:00, 11.07s/it, loss=0.8348, acc=80.64%]
Epoch 7/10 - Train Loss: 0.8348, Train Acc: 80.64%, Test Acc: 75.06%                                                                         
Epoch 8/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [07:22<00:00, 11.07s/it, loss=0.8167, acc=80.78%] 
Epoch 8/10 - Train Loss: 0.8167, Train Acc: 80.78%, Test Acc: 75.24%                                                                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_epoch8.pth
Epoch 9/10: 100%|██████████████████████████████████████████████████████████████████| 40/40 [07:22<00:00, 11.06s/it, loss=0.8047, acc=81.82%]
Epoch 9/10 - Train Loss: 0.8047, Train Acc: 81.82%, Test Acc: 75.27%                                                                         
Epoch 10/10: 100%|█████████████████████████████████████████████████████████████████| 40/40 [07:22<00:00, 11.06s/it, loss=0.7764, acc=82.10%] 
Epoch 10/10 - Train Loss: 0.7764, Train Acc: 82.10%, Test Acc: 75.30%                                                                        
  ✓ Checkpoint saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS3\ResNet50_ImageNet_FTAP_TAY_best.pth
  Best Accuracy: 75.30%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 75.30%                                                                                                               
✓ Model Size: 79.67 MB
✓ Average Inference Time: 0.0148 ms
✓ FLOPs: 3.34 GFLOPs

================================================================================
COMPARISON: TAYLOR GRADIENT-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          2.43   |    75.3    | -3.29                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         79.67   |    79.67   | -18.02 (-18.4%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0264 |          0.0158 |     0.0148 | -0.0116                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          3.34   |     3.34   | -0.79 (-19.1%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS3_ResNet50_ImageNet\TS3_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================