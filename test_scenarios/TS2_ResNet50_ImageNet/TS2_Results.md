================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 78.59%                                                                                                                                
✓ Model Size: 97.70 MB
✓ Average Inference Time: 0.0298 ms
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
  conv1: coverage shape torch.Size([64]), min=-0.718590, max=1.000000, mean=0.007640
  layer1.0.conv1: coverage shape torch.Size([64]), min=-0.681478, max=1.000000, mean=-0.109354
  layer1.0.conv2: coverage shape torch.Size([64]), min=-2.974997, max=1.000000, mean=0.121175
  layer1.0.conv3: coverage shape torch.Size([256]), min=-0.757488, max=1.000000, mean=-0.036503
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=-1.998849, max=1.000000, mean=-0.093980
  layer1.1.conv1: coverage shape torch.Size([64]), min=-1.894221, max=1.000000, mean=-0.058656
  layer1.1.conv2: coverage shape torch.Size([64]), min=-1.529318, max=1.000000, mean=-0.103326
  layer1.1.conv3: coverage shape torch.Size([256]), min=-1.594771, max=1.000000, mean=-0.106011
  layer1.2.conv1: coverage shape torch.Size([64]), min=-0.880118, max=1.000000, mean=-0.115805
  layer1.2.conv2: coverage shape torch.Size([64]), min=-1.833312, max=1.000000, mean=-0.215189
  layer1.2.conv3: coverage shape torch.Size([256]), min=-1.091855, max=1.000000, mean=-0.034222
  layer2.0.conv1: coverage shape torch.Size([128]), min=-1.018834, max=1.000000, mean=-0.088720
  layer2.0.conv2: coverage shape torch.Size([128]), min=-1.427576, max=1.000000, mean=-0.151698
  layer2.0.conv3: coverage shape torch.Size([512]), min=-0.669662, max=1.000000, mean=0.016033
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=-0.822004, max=1.000000, mean=-0.056024
  layer2.1.conv1: coverage shape torch.Size([128]), min=-0.986802, max=1.000000, mean=-0.094142
  layer2.1.conv2: coverage shape torch.Size([128]), min=-0.743309, max=1.000000, mean=0.069122
  layer2.1.conv3: coverage shape torch.Size([512]), min=-0.912450, max=1.000000, mean=-0.044121
  layer2.2.conv1: coverage shape torch.Size([128]), min=-1.437133, max=1.000000, mean=-0.005323
  layer2.2.conv2: coverage shape torch.Size([128]), min=-1.097141, max=1.000000, mean=-0.161710
  layer2.2.conv3: coverage shape torch.Size([512]), min=-0.948458, max=1.000000, mean=-0.002177
  layer2.3.conv1: coverage shape torch.Size([128]), min=-0.599998, max=1.000000, mean=-0.044062
  layer2.3.conv2: coverage shape torch.Size([128]), min=-0.855008, max=1.000000, mean=-0.127488
  layer2.3.conv3: coverage shape torch.Size([512]), min=-1.122360, max=1.000000, mean=-0.051276
  layer3.0.conv1: coverage shape torch.Size([256]), min=-1.503915, max=1.000000, mean=-0.196626
  layer3.0.conv2: coverage shape torch.Size([256]), min=-1.559527, max=1.000000, mean=-0.092187
  layer3.0.conv3: coverage shape torch.Size([1024]), min=-1.174768, max=1.000000, mean=-0.020766
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=-1.310002, max=1.000000, mean=-0.149015
  layer3.1.conv1: coverage shape torch.Size([256]), min=-1.814959, max=1.000000, mean=-0.068210
  layer3.1.conv2: coverage shape torch.Size([256]), min=-2.277541, max=1.000000, mean=-0.071568
  layer3.1.conv3: coverage shape torch.Size([1024]), min=-1.961913, max=1.000000, mean=-0.191128
  layer3.2.conv1: coverage shape torch.Size([256]), min=-1.421810, max=1.000000, mean=-0.249001
  layer3.2.conv2: coverage shape torch.Size([256]), min=-1.243426, max=1.000000, mean=-0.199575
  layer3.2.conv3: coverage shape torch.Size([1024]), min=-1.188716, max=1.000000, mean=-0.051485
  layer3.3.conv1: coverage shape torch.Size([256]), min=-1.150825, max=1.000000, mean=-0.128283
  layer3.3.conv2: coverage shape torch.Size([256]), min=-0.715742, max=1.000000, mean=-0.088740
  layer3.3.conv3: coverage shape torch.Size([1024]), min=-1.390668, max=1.000000, mean=-0.087931
  layer3.4.conv1: coverage shape torch.Size([256]), min=-0.841428, max=1.000000, mean=-0.255905
  layer3.4.conv2: coverage shape torch.Size([256]), min=-1.098107, max=1.000000, mean=-0.255564
  layer3.4.conv3: coverage shape torch.Size([1024]), min=-0.733089, max=1.000000, mean=-0.021703
  layer3.5.conv1: coverage shape torch.Size([256]), min=-1.073804, max=1.000000, mean=-0.174894
  layer3.5.conv2: coverage shape torch.Size([256]), min=-1.385621, max=1.000000, mean=-0.231508
  layer3.5.conv3: coverage shape torch.Size([1024]), min=-1.614918, max=1.000000, mean=-0.264835
  layer4.0.conv1: coverage shape torch.Size([512]), min=-1.410034, max=1.000000, mean=-0.602901
  layer4.0.conv2: coverage shape torch.Size([512]), min=-2.154461, max=1.000000, mean=-0.413627
  layer4.0.conv3: coverage shape torch.Size([2048]), min=-0.936845, max=1.000000, mean=-0.086041
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=-1.223049, max=1.000000, mean=-0.159978
  layer4.1.conv1: coverage shape torch.Size([512]), min=-1.798352, max=1.000000, mean=-0.243466
  layer4.1.conv2: coverage shape torch.Size([512]), min=-1.213289, max=1.000000, mean=-0.206792
  layer4.1.conv3: coverage shape torch.Size([2048]), min=-0.725346, max=1.000000, mean=-0.146416
  layer4.2.conv1: coverage shape torch.Size([512]), min=-1.424264, max=1.000000, mean=-0.225286
  layer4.2.conv2: coverage shape torch.Size([512]), min=-2.205902, max=1.000000, mean=-0.299054
  layer4.2.conv3: coverage shape torch.Size([2048]), min=-2.470827, max=1.000000, mean=-0.835054
  fc: coverage shape torch.Size([1000]), min=-0.799171, max=1.000000, mean=0.000398

------------------------------------------------------------
Coverage Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Coverage - Min: -0.718590, Max: 1.000000, Mean: 0.007640
  Zero coverage neurons: 0

layer1.0.conv1:
  Channels: 64
  Coverage - Min: -0.681478, Max: 1.000000, Mean: -0.109354
  Zero coverage neurons: 0

layer1.0.conv2:
  Channels: 64
  Coverage - Min: -2.974997, Max: 1.000000, Mean: 0.121175
  Zero coverage neurons: 0

layer1.0.conv3:
  Channels: 256
  Coverage - Min: -0.757488, Max: 1.000000, Mean: -0.036503
  Zero coverage neurons: 0

layer1.0.downsample.0:
  Channels: 256
  Coverage - Min: -1.998849, Max: 1.000000, Mean: -0.093980
  Zero coverage neurons: 0

layer1.1.conv1:
  Channels: 64
  Coverage - Min: -1.894221, Max: 1.000000, Mean: -0.058656
  Zero coverage neurons: 0

layer1.1.conv2:
  Channels: 64
  Coverage - Min: -1.529318, Max: 1.000000, Mean: -0.103326
  Zero coverage neurons: 0

layer1.1.conv3:
  Channels: 256
  Coverage - Min: -1.594771, Max: 1.000000, Mean: -0.106011
  Zero coverage neurons: 0

layer1.2.conv1:
  Channels: 64
  Coverage - Min: -0.880118, Max: 1.000000, Mean: -0.115805
  Zero coverage neurons: 0

layer1.2.conv2:
  Channels: 64
  Coverage - Min: -1.833312, Max: 1.000000, Mean: -0.215189
  Zero coverage neurons: 0

layer1.2.conv3:
  Channels: 256
  Coverage - Min: -1.091855, Max: 1.000000, Mean: -0.034222
  Zero coverage neurons: 0

layer2.0.conv1:
  Channels: 128
  Coverage - Min: -1.018834, Max: 1.000000, Mean: -0.088720
  Zero coverage neurons: 0

layer2.0.conv2:
  Channels: 128
  Coverage - Min: -1.427576, Max: 1.000000, Mean: -0.151698
  Zero coverage neurons: 0

layer2.0.conv3:
  Channels: 512
  Coverage - Min: -0.669662, Max: 1.000000, Mean: 0.016033
  Zero coverage neurons: 0

layer2.0.downsample.0:
  Channels: 512
  Coverage - Min: -0.822004, Max: 1.000000, Mean: -0.056024
  Zero coverage neurons: 0

layer2.1.conv1:
  Channels: 128
  Coverage - Min: -0.986802, Max: 1.000000, Mean: -0.094142
  Zero coverage neurons: 0

layer2.1.conv2:
  Channels: 128
  Coverage - Min: -0.743309, Max: 1.000000, Mean: 0.069122
  Zero coverage neurons: 0

layer2.1.conv3:
  Channels: 512
  Coverage - Min: -0.912450, Max: 1.000000, Mean: -0.044121
  Zero coverage neurons: 0

layer2.2.conv1:
  Channels: 128
  Coverage - Min: -1.437133, Max: 1.000000, Mean: -0.005323
  Zero coverage neurons: 0

layer2.2.conv2:
  Channels: 128
  Coverage - Min: -1.097141, Max: 1.000000, Mean: -0.161710
  Zero coverage neurons: 0

layer2.2.conv3:
  Channels: 512
  Coverage - Min: -0.948458, Max: 1.000000, Mean: -0.002177
  Zero coverage neurons: 0

layer2.3.conv1:
  Channels: 128
  Coverage - Min: -0.599998, Max: 1.000000, Mean: -0.044062
  Zero coverage neurons: 0

layer2.3.conv2:
  Channels: 128
  Coverage - Min: -0.855008, Max: 1.000000, Mean: -0.127488
  Zero coverage neurons: 0

layer2.3.conv3:
  Channels: 512
  Coverage - Min: -1.122360, Max: 1.000000, Mean: -0.051276
  Zero coverage neurons: 0

layer3.0.conv1:
  Channels: 256
  Coverage - Min: -1.503915, Max: 1.000000, Mean: -0.196626
  Zero coverage neurons: 0

layer3.0.conv2:
  Channels: 256
  Coverage - Min: -1.559527, Max: 1.000000, Mean: -0.092187
  Zero coverage neurons: 0

layer3.0.conv3:
  Channels: 1024
  Coverage - Min: -1.174768, Max: 1.000000, Mean: -0.020766
  Zero coverage neurons: 0

layer3.0.downsample.0:
  Channels: 1024
  Coverage - Min: -1.310002, Max: 1.000000, Mean: -0.149015
  Zero coverage neurons: 0

layer3.1.conv1:
  Channels: 256
  Coverage - Min: -1.814959, Max: 1.000000, Mean: -0.068210
  Zero coverage neurons: 0

layer3.1.conv2:
  Channels: 256
  Coverage - Min: -2.277541, Max: 1.000000, Mean: -0.071568
  Zero coverage neurons: 0

layer3.1.conv3:
  Channels: 1024
  Coverage - Min: -1.961913, Max: 1.000000, Mean: -0.191128
  Zero coverage neurons: 0

layer3.2.conv1:
  Channels: 256
  Coverage - Min: -1.421810, Max: 1.000000, Mean: -0.249001
  Zero coverage neurons: 0

layer3.2.conv2:
  Channels: 256
  Coverage - Min: -1.243426, Max: 1.000000, Mean: -0.199575
  Zero coverage neurons: 0

layer3.2.conv3:
  Channels: 1024
  Coverage - Min: -1.188716, Max: 1.000000, Mean: -0.051485
  Zero coverage neurons: 0

layer3.3.conv1:
  Channels: 256
  Coverage - Min: -1.150825, Max: 1.000000, Mean: -0.128283
  Zero coverage neurons: 0

layer3.3.conv2:
  Channels: 256
  Coverage - Min: -0.715742, Max: 1.000000, Mean: -0.088740
  Zero coverage neurons: 0

layer3.3.conv3:
  Channels: 1024
  Coverage - Min: -1.390668, Max: 1.000000, Mean: -0.087931
  Zero coverage neurons: 0

layer3.4.conv1:
  Channels: 256
  Coverage - Min: -0.841428, Max: 1.000000, Mean: -0.255905
  Zero coverage neurons: 0

layer3.4.conv2:
  Channels: 256
  Coverage - Min: -1.098107, Max: 1.000000, Mean: -0.255564
  Zero coverage neurons: 0

layer3.4.conv3:
  Channels: 1024
  Coverage - Min: -0.733089, Max: 1.000000, Mean: -0.021703
  Zero coverage neurons: 0

layer3.5.conv1:
  Channels: 256
  Coverage - Min: -1.073804, Max: 1.000000, Mean: -0.174894
  Zero coverage neurons: 0

layer3.5.conv2:
  Channels: 256
  Coverage - Min: -1.385621, Max: 1.000000, Mean: -0.231508
  Zero coverage neurons: 0

layer3.5.conv3:
  Channels: 1024
  Coverage - Min: -1.614918, Max: 1.000000, Mean: -0.264835
  Zero coverage neurons: 0

layer4.0.conv1:
  Channels: 512
  Coverage - Min: -1.410034, Max: 1.000000, Mean: -0.602901
  Zero coverage neurons: 0

layer4.0.conv2:
  Channels: 512
  Coverage - Min: -2.154461, Max: 1.000000, Mean: -0.413627
  Zero coverage neurons: 0

layer4.0.conv3:
  Channels: 2048
  Coverage - Min: -0.936845, Max: 1.000000, Mean: -0.086041
  Zero coverage neurons: 0

layer4.0.downsample.0:
  Channels: 2048
  Coverage - Min: -1.223049, Max: 1.000000, Mean: -0.159978
  Zero coverage neurons: 0

layer4.1.conv1:
  Channels: 512
  Coverage - Min: -1.798352, Max: 1.000000, Mean: -0.243466
  Zero coverage neurons: 0

layer4.1.conv2:
  Channels: 512
  Coverage - Min: -1.213289, Max: 1.000000, Mean: -0.206792
  Zero coverage neurons: 0

layer4.1.conv3:
  Channels: 2048
  Coverage - Min: -0.725346, Max: 1.000000, Mean: -0.146416
  Zero coverage neurons: 0

layer4.2.conv1:
  Channels: 512
  Coverage - Min: -1.424264, Max: 1.000000, Mean: -0.225286
  Zero coverage neurons: 0

layer4.2.conv2:
  Channels: 512
  Coverage - Min: -2.205902, Max: 1.000000, Mean: -0.299054
  Zero coverage neurons: 0

layer4.2.conv3:
  Channels: 2048
  Coverage - Min: -2.470827, Max: 1.000000, Mean: -0.835054
  Zero coverage neurons: 0

fc:
  Channels: 1000
  Coverage - Min: -0.799171, Max: 1.000000, Mean: 0.000398
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
✓ Pruned Model Accuracy: 0.09%                                                                                                                                   
✓ Model Size: 63.64 MB
✓ Average Inference Time: 1.1882 ms
✓ FLOPs: 2.66 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 5
Epoch 1/5: 100%|███████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:42<00:00,  1.06s/it, loss=5.5799, acc=11.90%] 
Epoch 1/5 - Train Loss: 5.5799, Train Acc: 11.90%, Test Acc: 25.26%                                                                                              
Epoch 2/5: 100%|███████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=3.6743, acc=29.96%] 
Epoch 2/5 - Train Loss: 3.6743, Train Acc: 29.96%, Test Acc: 41.13%                                                                                              
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_NC_epoch2.pth
Epoch 3/5: 100%|███████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=3.0176, acc=39.46%]
Epoch 3/5 - Train Loss: 3.0176, Train Acc: 39.46%, Test Acc: 45.97%                                                                                              
Epoch 4/5: 100%|███████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.7116, acc=44.06%] 
Epoch 4/5 - Train Loss: 2.7116, Train Acc: 44.06%, Test Acc: 47.82%                                                                                              
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_NC_epoch4.pth
Epoch 5/5: 100%|███████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.5796, acc=45.76%]
Epoch 5/5 - Train Loss: 2.5796, Train Acc: 45.76%, Test Acc: 48.44%                                                                                              
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_NC_epoch5.pth

✓ Best model saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_NC_best.pth
  Best Accuracy: 48.44%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 48.44%                                                                                                                                   
✓ Model Size: 63.64 MB
✓ Average Inference Time: 1.1929 ms
✓ FLOPs: 2.66 GFLOPs

================================================================================
COMPARISON: NEURON COVERAGE PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.09   |    48.44   | -30.15                      |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         63.64   |    63.64   | -34.05 (-34.9%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0298 |          1.1882 |     1.1929 | +1.1632                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          2.66   |     2.66   | -1.48 (-35.7%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS2_ResNet50_ImageNet\TS2_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================

====================================================================================================
✓ TS2_02_coverage_pruning.py completed successfully
  Elapsed time: 892.18 seconds
====================================================================================================


====================================================================================================
SCRIPT 3/4: TS2_03_wanda_pruning.py
====================================================================================================

====================================================================================================
RUNNING: TS2_03_wanda_pruning.py
====================================================================================================


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
✓ Average Inference Time: 0.0287 ms
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
  conv1: coverage shape torch.Size([64]), min=0.000072, max=0.513547, mean=0.034473
  layer1.0.conv1: coverage shape torch.Size([64]), min=0.867673, max=44.524899, mean=9.878698
  layer1.0.conv2: coverage shape torch.Size([64]), min=0.007039, max=25.866432, mean=3.913787
  layer1.0.conv3: coverage shape torch.Size([256]), min=0.000000, max=10.946033, mean=2.153580
  layer1.0.downsample.0: coverage shape torch.Size([256]), min=0.000000, max=61.157711, mean=5.196681
  layer1.1.conv1: coverage shape torch.Size([64]), min=0.000000, max=6.707611, mean=1.258582
  layer1.1.conv2: coverage shape torch.Size([64]), min=0.123085, max=6.317640, mean=1.653637
  layer1.1.conv3: coverage shape torch.Size([256]), min=0.001444, max=7.035081, mean=1.303635
  layer1.2.conv1: coverage shape torch.Size([64]), min=0.021889, max=4.766005, mean=1.477814
  layer1.2.conv2: coverage shape torch.Size([64]), min=0.068610, max=6.584062, mean=1.331575
  layer1.2.conv3: coverage shape torch.Size([256]), min=0.002450, max=5.844776, mean=0.939614
  layer2.0.conv1: coverage shape torch.Size([128]), min=0.002160, max=5.208488, mean=1.523770
  layer2.0.conv2: coverage shape torch.Size([128]), min=0.002134, max=6.537640, mean=1.466158
  layer2.0.conv3: coverage shape torch.Size([512]), min=0.000002, max=8.114993, mean=1.116065
  layer2.0.downsample.0: coverage shape torch.Size([512]), min=0.000001, max=8.558477, mean=1.162570
  layer2.1.conv1: coverage shape torch.Size([128]), min=0.080366, max=7.678209, mean=2.208224
  layer2.1.conv2: coverage shape torch.Size([128]), min=0.022826, max=10.115705, mean=1.950466
  layer2.1.conv3: coverage shape torch.Size([512]), min=0.000087, max=2.398087, mean=0.467886
  layer2.2.conv1: coverage shape torch.Size([128]), min=0.009974, max=5.482849, mean=1.217688
  layer2.2.conv2: coverage shape torch.Size([128]), min=0.010913, max=6.196023, mean=1.372780
  layer2.2.conv3: coverage shape torch.Size([512]), min=0.000085, max=3.077775, mean=0.579574
  layer2.3.conv1: coverage shape torch.Size([128]), min=0.022085, max=6.825028, mean=1.273274
  layer2.3.conv2: coverage shape torch.Size([128]), min=0.013530, max=5.928537, mean=1.380155
  layer2.3.conv3: coverage shape torch.Size([512]), min=0.000629, max=3.385095, mean=0.602423
  layer3.0.conv1: coverage shape torch.Size([256]), min=0.004166, max=6.387030, mean=1.403260
  layer3.0.conv2: coverage shape torch.Size([256]), min=0.003687, max=4.447390, mean=0.889897
  layer3.0.conv3: coverage shape torch.Size([1024]), min=0.000117, max=4.702579, mean=0.894866
  layer3.0.downsample.0: coverage shape torch.Size([1024]), min=0.000251, max=4.077830, mean=0.831033
  layer3.1.conv1: coverage shape torch.Size([256]), min=0.004504, max=9.749556, mean=1.412705
  layer3.1.conv2: coverage shape torch.Size([256]), min=0.001212, max=11.067134, mean=1.027710
  layer3.1.conv3: coverage shape torch.Size([1024]), min=0.000345, max=2.091558, mean=0.318516
  layer3.2.conv1: coverage shape torch.Size([256]), min=0.002833, max=5.954094, mean=1.282970
  layer3.2.conv2: coverage shape torch.Size([256]), min=0.004934, max=3.083417, mean=0.708400
  layer3.2.conv3: coverage shape torch.Size([1024]), min=0.000150, max=2.031354, mean=0.287331
  layer3.3.conv1: coverage shape torch.Size([256]), min=0.001658, max=5.975231, mean=1.163958
  layer3.3.conv2: coverage shape torch.Size([256]), min=0.009635, max=5.396166, mean=0.820631
  layer3.3.conv3: coverage shape torch.Size([1024]), min=0.001390, max=1.959153, mean=0.335760
  layer3.4.conv1: coverage shape torch.Size([256]), min=0.008412, max=3.742167, mean=1.143398
  layer3.4.conv2: coverage shape torch.Size([256]), min=0.013324, max=3.055292, mean=0.863581
  layer3.4.conv3: coverage shape torch.Size([1024]), min=0.000076, max=2.885143, mean=0.299669
  layer3.5.conv1: coverage shape torch.Size([256]), min=0.009171, max=5.946611, mean=1.247296
  layer3.5.conv2: coverage shape torch.Size([256]), min=0.000837, max=3.799148, mean=0.845543
  layer3.5.conv3: coverage shape torch.Size([1024]), min=0.001355, max=2.126883, mean=0.429371
  layer4.0.conv1: coverage shape torch.Size([512]), min=0.044654, max=3.696602, mean=1.627988
  layer4.0.conv2: coverage shape torch.Size([512]), min=0.002829, max=4.810726, mean=0.951606
  layer4.0.conv3: coverage shape torch.Size([2048]), min=0.000032, max=5.297407, mean=0.589667
  layer4.0.downsample.0: coverage shape torch.Size([2048]), min=0.001165, max=3.842200, mean=0.576639
  layer4.1.conv1: coverage shape torch.Size([512]), min=0.105291, max=9.321593, mean=1.404464
  layer4.1.conv2: coverage shape torch.Size([512]), min=0.000152, max=5.468699, mean=1.052740
  layer4.1.conv3: coverage shape torch.Size([2048]), min=0.000803, max=2.933120, mean=0.447285
  layer4.2.conv1: coverage shape torch.Size([512]), min=0.085780, max=7.492838, mean=1.339012
  layer4.2.conv2: coverage shape torch.Size([512]), min=0.047416, max=4.689207, mean=0.663592
  layer4.2.conv3: coverage shape torch.Size([2048]), min=0.000526, max=1.059533, mean=0.360627
  fc: coverage shape torch.Size([1000]), min=0.000294, max=0.277118, mean=0.059963

------------------------------------------------------------
Activation Statistics:
------------------------------------------------------------

conv1:
  Channels: 64
  Activation - Min: 0.000072, Max: 0.513547, Mean: 0.034473

layer1.0.conv1:
  Channels: 64
  Activation - Min: 0.867673, Max: 44.524899, Mean: 9.878698

layer1.0.conv2:
  Channels: 64
  Activation - Min: 0.007039, Max: 25.866432, Mean: 3.913787

layer1.0.conv3:
  Channels: 256
  Activation - Min: 0.000000, Max: 10.946033, Mean: 2.153580

layer1.0.downsample.0:
  Channels: 256
  Activation - Min: 0.000000, Max: 61.157711, Mean: 5.196681

layer1.1.conv1:
  Channels: 64
  Activation - Min: 0.000000, Max: 6.707611, Mean: 1.258582

layer1.1.conv2:
  Channels: 64
  Activation - Min: 0.123085, Max: 6.317640, Mean: 1.653637

layer1.1.conv3:
  Channels: 256
  Activation - Min: 0.001444, Max: 7.035081, Mean: 1.303635

layer1.2.conv1:
  Channels: 64
  Activation - Min: 0.021889, Max: 4.766005, Mean: 1.477814

layer1.2.conv2:
  Channels: 64
  Activation - Min: 0.068610, Max: 6.584062, Mean: 1.331575

layer1.2.conv3:
  Channels: 256
  Activation - Min: 0.002450, Max: 5.844776, Mean: 0.939614

layer2.0.conv1:
  Channels: 128
  Activation - Min: 0.002160, Max: 5.208488, Mean: 1.523770

layer2.0.conv2:
  Channels: 128
  Activation - Min: 0.002134, Max: 6.537640, Mean: 1.466158

layer2.0.conv3:
  Channels: 512
  Activation - Min: 0.000002, Max: 8.114993, Mean: 1.116065

layer2.0.downsample.0:
  Channels: 512
  Activation - Min: 0.000001, Max: 8.558477, Mean: 1.162570

layer2.1.conv1:
  Channels: 128
  Activation - Min: 0.080366, Max: 7.678209, Mean: 2.208224

layer2.1.conv2:
  Channels: 128
  Activation - Min: 0.022826, Max: 10.115705, Mean: 1.950466

layer2.1.conv3:
  Channels: 512
  Activation - Min: 0.000087, Max: 2.398087, Mean: 0.467886

layer2.2.conv1:
  Channels: 128
  Activation - Min: 0.009974, Max: 5.482849, Mean: 1.217688

layer2.2.conv2:
  Channels: 128
  Activation - Min: 0.010913, Max: 6.196023, Mean: 1.372780

layer2.2.conv3:
  Channels: 512
  Activation - Min: 0.000085, Max: 3.077775, Mean: 0.579574

layer2.3.conv1:
  Channels: 128
  Activation - Min: 0.022085, Max: 6.825028, Mean: 1.273274

layer2.3.conv2:
  Channels: 128
  Activation - Min: 0.013530, Max: 5.928537, Mean: 1.380155

layer2.3.conv3:
  Channels: 512
  Activation - Min: 0.000629, Max: 3.385095, Mean: 0.602423

layer3.0.conv1:
  Channels: 256
  Activation - Min: 0.004166, Max: 6.387030, Mean: 1.403260

layer3.0.conv2:
  Channels: 256
  Activation - Min: 0.003687, Max: 4.447390, Mean: 0.889897

layer3.0.conv3:
  Channels: 1024
  Activation - Min: 0.000117, Max: 4.702579, Mean: 0.894866

layer3.0.downsample.0:
  Channels: 1024
  Activation - Min: 0.000251, Max: 4.077830, Mean: 0.831033

layer3.1.conv1:
  Channels: 256
  Activation - Min: 0.004504, Max: 9.749556, Mean: 1.412705

layer3.1.conv2:
  Channels: 256
  Activation - Min: 0.001212, Max: 11.067134, Mean: 1.027710

layer3.1.conv3:
  Channels: 1024
  Activation - Min: 0.000345, Max: 2.091558, Mean: 0.318516

layer3.2.conv1:
  Channels: 256
  Activation - Min: 0.002833, Max: 5.954094, Mean: 1.282970

layer3.2.conv2:
  Channels: 256
  Activation - Min: 0.004934, Max: 3.083417, Mean: 0.708400

layer3.2.conv3:
  Channels: 1024
  Activation - Min: 0.000150, Max: 2.031354, Mean: 0.287331

layer3.3.conv1:
  Channels: 256
  Activation - Min: 0.001658, Max: 5.975231, Mean: 1.163958

layer3.3.conv2:
  Channels: 256
  Activation - Min: 0.009635, Max: 5.396166, Mean: 0.820631

layer3.3.conv3:
  Channels: 1024
  Activation - Min: 0.001390, Max: 1.959153, Mean: 0.335760

layer3.4.conv1:
  Channels: 256
  Activation - Min: 0.008412, Max: 3.742167, Mean: 1.143398

layer3.4.conv2:
  Channels: 256
  Activation - Min: 0.013324, Max: 3.055292, Mean: 0.863581

layer3.4.conv3:
  Channels: 1024
  Activation - Min: 0.000076, Max: 2.885143, Mean: 0.299669

layer3.5.conv1:
  Channels: 256
  Activation - Min: 0.009171, Max: 5.946611, Mean: 1.247296

layer3.5.conv2:
  Channels: 256
  Activation - Min: 0.000837, Max: 3.799148, Mean: 0.845543

layer3.5.conv3:
  Channels: 1024
  Activation - Min: 0.001355, Max: 2.126883, Mean: 0.429371

layer4.0.conv1:
  Channels: 512
  Activation - Min: 0.044654, Max: 3.696602, Mean: 1.627988

layer4.0.conv2:
  Channels: 512
  Activation - Min: 0.002829, Max: 4.810726, Mean: 0.951606

layer4.0.conv3:
  Channels: 2048
  Activation - Min: 0.000032, Max: 5.297407, Mean: 0.589667

layer4.0.downsample.0:
  Channels: 2048
  Activation - Min: 0.001165, Max: 3.842200, Mean: 0.576639

layer4.1.conv1:
  Channels: 512
  Activation - Min: 0.105291, Max: 9.321593, Mean: 1.404464

layer4.1.conv2:
  Channels: 512
  Activation - Min: 0.000152, Max: 5.468699, Mean: 1.052740

layer4.1.conv3:
  Channels: 2048
  Activation - Min: 0.000803, Max: 2.933120, Mean: 0.447285

layer4.2.conv1:
  Channels: 512
  Activation - Min: 0.085780, Max: 7.492838, Mean: 1.339012

layer4.2.conv2:
  Channels: 512
  Activation - Min: 0.047416, Max: 4.689207, Mean: 0.663592

layer4.2.conv3:
  Channels: 2048
  Activation - Min: 0.000526, Max: 1.059533, Mean: 0.360627

fc:
  Channels: 1000
  Activation - Min: 0.000294, Max: 0.277118, Mean: 0.059963
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
✓ Pruned Model Accuracy: 0.18%                                                                                                                                   
✓ Model Size: 63.64 MB
✓ Average Inference Time: 1.1923 ms
✓ FLOPs: 2.66 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 5
Epoch 1/5: 100%|███████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.04s/it, loss=3.7757, acc=30.11%] 
Epoch 1/5 - Train Loss: 3.7757, Train Acc: 30.11%, Test Acc: 47.05%                                                                                              
Epoch 2/5: 100%|███████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.7742, acc=42.31%] 
Epoch 2/5 - Train Loss: 2.7742, Train Acc: 42.31%, Test Acc: 52.41%                                                                                              
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_W_epoch2.pth
Epoch 3/5: 100%|███████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.4311, acc=48.55%]
Epoch 3/5 - Train Loss: 2.4311, Train Acc: 48.55%, Test Acc: 54.77%                                                                                              
Epoch 4/5: 100%|███████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.1774, acc=52.72%] 
Epoch 4/5 - Train Loss: 2.1774, Train Acc: 52.72%, Test Acc: 55.98%
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_W_epoch4.pth
Epoch 5/5: 100%|███████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it, loss=2.1101, acc=53.59%]
Epoch 5/5 - Train Loss: 2.1101, Train Acc: 53.59%, Test Acc: 56.45%                                                                                              
  ✓ Checkpoint saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_W_epoch5.pth

✓ Best model saved: C:\source\checkpoints\TS2\ResNet50_ImageNet_FTAP_W_best.pth
  Best Accuracy: 56.45%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 56.45%                                                                                                                                   
✓ Model Size: 63.64 MB
✓ Average Inference Time: 1.1888 ms
✓ FLOPs: 2.66 GFLOPs

================================================================================
COMPARISON: WANDA PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         78.59   |          0.18   |    56.45   | -22.14                      |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         97.7    |         63.64   |    63.64   | -34.05 (-34.9%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.0287 |          1.1923 |     1.1888 | +1.1600                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.13   |          2.66   |     2.66   | -1.48 (-35.7%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS2_ResNet50_ImageNet\TS2_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================

====================================================================================================
✓ TS2_03_wanda_pruning.py completed successfully
  Elapsed time: 867.92 seconds
====================================================================================================


====================================================================================================
SCRIPT 4/4: TS2_compare_results.py
====================================================================================================

====================================================================================================
RUNNING: TS2_compare_results.py
====================================================================================================


====================================================================================================
TEST SCENARIO TS2: RESULTS COMPARISON
====================================================================================================

====================================================================================================
TEST SCENARIO TS2: COMPREHENSIVE COMPARISON
====================================================================================================
+---------------------+-------------------------+-------------------+---------+
| Metric              |   Original (Fine-tuned) |   Neuron Coverage |   Wanda |
+=====================+=========================+===================+=========+
| Accuracy (%)        |                 78.59   |           48.44   | 56.45   |
+---------------------+-------------------------+-------------------+---------+
| Size (MB)           |                 97.7    |           63.64   | 63.64   |
+---------------------+-------------------------+-------------------+---------+
| Inference Time (ms) |                  0.0139 |            1.1929 |  1.1888 |
+---------------------+-------------------------+-------------------+---------+
| FLOPs (GFLOPs)      |                  4.13   |            2.66   |  2.66   |
+---------------------+-------------------------+-------------------+---------+

====================================================================================================
PERCENTAGE CHANGES (compared to original)
====================================================================================================
+---------------------------+------------------------+------------------------+
| Metric                    | Neuron Coverage        | Wanda                  |
+===========================+========================+========================+
| Accuracy Change (%)       | -30.15 (-38.36%)       | -22.14 (-28.17%)       |
+---------------------------+------------------------+------------------------+
| Size Reduction (%)        | -34.05 MB (-34.85%)    | -34.05 MB (-34.85%)    |
+---------------------------+------------------------+------------------------+
| Inference Time Change (%) | +1.1790 ms (+8453.25%) | +1.1748 ms (+8423.44%) |
+---------------------------+------------------------+------------------------+
| FLOPs Reduction (%)       | -1.48 G (-35.72%)      | -1.48 G (-35.72%)      |
+---------------------------+------------------------+------------------------+

====================================================================================================
SUMMARY INSIGHTS
====================================================================================================

Accuracy Comparison:
  ✓ Wanda achieved better accuracy: 56.45% vs 48.44%
    Difference: 8.01%

Model Size Comparison:
  ✓ Both methods produced equal-sized models: 63.64 MB

Overall Assessment:
  Both methods achieved TS2 pruning with 20% target ratio.
  Original model accuracy: 78.59%
  ✓ Wanda preserved accuracy better (loss: 22.14%)

====================================================================================================

====================================================================================================
DETAILED RESULTS
====================================================================================================

--- Model Preparation ---
Pretrained Accuracy: 78.59%
Fine-tuned Accuracy: 78.59%
Improvement: +0.00%
Fine-tuning Epochs: 0

--- Neuron Coverage Pruning ---
Original Accuracy: 78.59%
After Pruning: 0.09%
After Fine-tuning: 48.44%
Pruning Ratio: 20.0%
Fine-tuning Epochs: 5

--- Wanda Pruning ---
Original Accuracy: 78.59%
After Pruning: 0.18%
After Fine-tuning: 56.45%
Pruning Ratio: 20.0%
Fine-tuning Epochs: 5

====================================================================================================

✓ Results comparison completed
====================================================================================================


====================================================================================================
✓ TS2_compare_results.py completed successfully
  Elapsed time: 0.06 seconds
====================================================================================================


====================================================================================================
EXECUTION SUMMARY
====================================================================================================

Total elapsed time: 1923.73 seconds (32.06 minutes)

Scripts executed: 4/4

Results:
  ✓ SUCCESS: TS2_01_prepare_model.py
  ✓ SUCCESS: TS2_02_coverage_pruning.py
  ✓ SUCCESS: TS2_03_wanda_pruning.py
  ✓ SUCCESS: TS2_compare_results.py

====================================================================================================
✓ ALL SCRIPTS COMPLETED SUCCESSFULLY
====================================================================================================