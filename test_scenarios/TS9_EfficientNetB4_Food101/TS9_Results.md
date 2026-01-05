================================================================================
TEST SCENARIO TS6: PREPARE EfficientNetB4 MODEL
================================================================================
Model: EfficientNetB4
Dataset: Food101
Device: cuda
Fine-tune Epochs: 10
================================================================================
✓ Directories created/verified

================================================================================
PREPARING STANFORD DOGS DATASET
================================================================================
Note: Using Food101 dataset (101 food categories) as Stanford Dogs is not built-in.
Food101: 101 classes, ~75,750 training images, ~25,250 test images
✓ Dataset prepared: C:\source\downloaded_datasets\food101
  - Training samples: 75750
  - Test samples: 25250
  - Number of classes: 101 (food categories)
  - Image size: 380x380

================================================================================
LOADING PRETRAINED EFFICIENTNET-B4
================================================================================
✓ EfficientNet-B4 loaded successfully
  - Pretrained on ImageNet
  - Modified for 101 classes (Food101)
  - Model parameters: 17,729,709
  - Device: cuda

================================================================================
EVALUATING PRETRAINED MODEL (BEFORE FINE-TUNING)
================================================================================
✓ Pretrained Model Accuracy: 0.89%                                                                                                                            
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.6028 ms
✓ FLOPs: 4.61 GFLOPs

================================================================================
FINE-TUNING MODEL ON STANFORD DOGS
================================================================================
Epoch 1/10: 100%|███████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:01<00:00,  3.58it/s, loss=2.2554, acc=45.71%] 
Epoch 1/10 - Train Loss: 2.2554, Train Acc: 45.71%, Test Acc: 78.08%                                                                                          
Epoch 2/10: 100%|███████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:01<00:00,  3.58it/s, loss=1.4131, acc=63.97%]
Epoch 2/10 - Train Loss: 1.4131, Train Acc: 63.97%, Test Acc: 83.67%                                                                                          
Epoch 3/10: 100%|███████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:04<00:00,  3.57it/s, loss=1.2070, acc=68.82%]
Epoch 3/10 - Train Loss: 1.2070, Train Acc: 68.82%, Test Acc: 85.85%                                                                                          
Epoch 4/10: 100%|███████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:04<00:00,  3.57it/s, loss=1.0725, acc=72.19%]
Epoch 4/10 - Train Loss: 1.0725, Train Acc: 72.19%, Test Acc: 86.80%                                                                                          
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:04<00:00,  3.57it/s, loss=0.9894, acc=74.20%]
Epoch 5/10 - Train Loss: 0.9894, Train Acc: 74.20%, Test Acc: 87.71%                                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FT_epoch5.pth
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:04<00:00,  3.57it/s, loss=0.9170, acc=76.03%]
Epoch 6/10 - Train Loss: 0.9170, Train Acc: 76.03%, Test Acc: 88.17%                                                                                          
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:04<00:00,  3.57it/s, loss=0.8596, acc=77.29%]
Epoch 7/10 - Train Loss: 0.8596, Train Acc: 77.29%, Test Acc: 88.92%                                                                                          
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:05<00:00,  3.57it/s, loss=0.8388, acc=77.85%]
Epoch 8/10 - Train Loss: 0.8388, Train Acc: 77.85%, Test Acc: 88.82%                                                                                          
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:04<00:00,  3.57it/s, loss=0.8061, acc=78.73%] 
Epoch 9/10 - Train Loss: 0.8061, Train Acc: 78.73%, Test Acc: 89.20%                                                                                          
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:04<00:00,  3.57it/s, loss=0.8005, acc=78.83%]
Epoch 10/10 - Train Loss: 0.8005, Train Acc: 78.83%, Test Acc: 89.15%                                                                                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FT_epoch10.pth
c:\source\repos\cleanai-v5\test_scenarios\TS6_EfficientNetB4_Food101\TS6_01_prepare_model.py:345: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(best_checkpoint, map_location=CONFIG['device'])

================================================================================
FINAL EVALUATION (AFTER FINE-TUNING)
================================================================================
✓ Fine-tuned Model Accuracy: 89.20%                                                                                                                           
✓ Model Size: 68.12 MB
✓ Average Inference Time: 0.5878 ms
✓ FLOPs: 4.61 GFLOPs

================================================================================
COMPARISON: BEFORE vs AFTER FINE-TUNING
================================================================================
+---------------------+--------------------------+-------------------------+----------+
| Metric              |   Pretrained (Before FT) |   Fine-tuned (After FT) |   Change |
+=====================+==========================+=========================+==========+
| Accuracy (%)        |                   0.89   |                 89.2    |   88.31  |
+---------------------+--------------------------+-------------------------+----------+
| Size (MB)           |                  68.11   |                 68.12   |    0     |
+---------------------+--------------------------+-------------------------+----------+
| Inference Time (ms) |                   0.6028 |                  0.5878 |   -0.015 |
+---------------------+--------------------------+-------------------------+----------+
| FLOPs (G)           |                   4.61   |                  4.61   |    0     |
+---------------------+--------------------------+-------------------------+----------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS6_EfficientNetB4_Food101\TS6_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================



*****************************************************



features.5.0.block.2.fc2:
  Channels: 672
  Coverage - Min: -0.184414, Max: 1.000000, Mean: 0.275044
  Zero coverage neurons: 0

features.5.0.block.3.0:
  Channels: 160
  Coverage - Min: -0.810008, Max: 1.000000, Mean: 0.018344
  Zero coverage neurons: 0

features.5.1.block.0.0:
  Channels: 960
  Coverage - Min: -0.595363, Max: 1.000000, Mean: -0.063139
  Zero coverage neurons: 0

features.5.1.block.1.0:
  Channels: 960
  Coverage - Min: -2.566228, Max: 1.000000, Mean: -0.024737
  Zero coverage neurons: 0

features.5.1.block.2.fc1:
  Channels: 40
  Coverage - Min: -0.420268, Max: 1.000000, Mean: 0.221384
  Zero coverage neurons: 0

features.5.1.block.2.fc2:
  Channels: 960
  Coverage - Min: -1.161589, Max: 1.000000, Mean: 0.022454
  Zero coverage neurons: 0

features.5.1.block.3.0:
  Channels: 160
  Coverage - Min: -1.183899, Max: 1.000000, Mean: 0.015322
  Zero coverage neurons: 0

features.5.2.block.0.0:
  Channels: 960
  Coverage - Min: -1.519522, Max: 1.000000, Mean: -0.348893
  Zero coverage neurons: 0

features.5.2.block.1.0:
  Channels: 960
  Coverage - Min: -1.221980, Max: 1.000000, Mean: -0.014436
  Zero coverage neurons: 0

features.5.2.block.2.fc1:
  Channels: 40
  Coverage - Min: -1.076306, Max: 1.000000, Mean: 0.110397
  Zero coverage neurons: 0

features.5.2.block.2.fc2:
  Channels: 960
  Coverage - Min: -1.786856, Max: 1.000000, Mean: -0.007272
  Zero coverage neurons: 0

features.5.2.block.3.0:
  Channels: 160
  Coverage - Min: -1.161999, Max: 1.000000, Mean: -0.042064
  Zero coverage neurons: 0

features.5.3.block.0.0:
  Channels: 960
  Coverage - Min: -1.509642, Max: 1.000000, Mean: -0.327752
  Zero coverage neurons: 0

features.5.3.block.1.0:
  Channels: 960
  Coverage - Min: -1.866239, Max: 1.000000, Mean: -0.013482
  Zero coverage neurons: 0

features.5.3.block.2.fc1:
  Channels: 40
  Coverage - Min: -1.840320, Max: 1.000000, Mean: -0.141377
  Zero coverage neurons: 0

features.5.3.block.2.fc2:
  Channels: 960
  Coverage - Min: -2.181280, Max: 1.000000, Mean: 0.094523
  Zero coverage neurons: 0

features.5.3.block.3.0:
  Channels: 160
  Coverage - Min: -1.602123, Max: 1.000000, Mean: -0.016126
  Zero coverage neurons: 0

features.5.4.block.0.0:
  Channels: 960
  Coverage - Min: -1.658588, Max: 1.000000, Mean: -0.457442
  Zero coverage neurons: 0

features.5.4.block.1.0:
  Channels: 960
  Coverage - Min: -2.633905, Max: 1.000000, Mean: -0.014959
  Zero coverage neurons: 0

features.5.4.block.2.fc1:
  Channels: 40
  Coverage - Min: -2.475442, Max: 1.000000, Mean: 0.029887
  Zero coverage neurons: 0

features.5.4.block.2.fc2:
  Channels: 960
  Coverage - Min: -1.936021, Max: 1.000000, Mean: 0.096230
  Zero coverage neurons: 0

features.5.4.block.3.0:
  Channels: 160
  Coverage - Min: -0.794461, Max: 1.000000, Mean: 0.030464
  Zero coverage neurons: 0

features.5.5.block.0.0:
  Channels: 960
  Coverage - Min: -0.778213, Max: 1.000000, Mean: -0.250445
  Zero coverage neurons: 0

features.5.5.block.1.0:
  Channels: 960
  Coverage - Min: -1.596955, Max: 1.000000, Mean: -0.012935
  Zero coverage neurons: 0

features.5.5.block.2.fc1:
  Channels: 40
  Coverage - Min: -0.529121, Max: 1.000000, Mean: 0.225243
  Zero coverage neurons: 0

features.5.5.block.2.fc2:
  Channels: 960
  Coverage - Min: -1.826166, Max: 1.000000, Mean: 0.046266
  Zero coverage neurons: 0

features.5.5.block.3.0:
  Channels: 160
  Coverage - Min: -0.925347, Max: 1.000000, Mean: -0.005222
  Zero coverage neurons: 0

features.6.0.block.0.0:
  Channels: 960
  Coverage - Min: -1.480554, Max: 1.000000, Mean: -0.432768
  Zero coverage neurons: 0

features.6.0.block.1.0:
  Channels: 960
  Coverage - Min: -49.226097, Max: 1.000000, Mean: -0.585839
  Zero coverage neurons: 0

features.6.0.block.2.fc1:
  Channels: 40
  Coverage - Min: -45.872482, Max: -0.770931, Mean: -7.805642
  Zero coverage neurons: 0

features.6.0.block.2.fc2:
  Channels: 960
  Coverage - Min: 0.227658, Max: 1.000000, Mean: 0.472746
  Zero coverage neurons: 0

features.6.0.block.3.0:
  Channels: 272
  Coverage - Min: -0.945304, Max: 1.000000, Mean: -0.000656
  Zero coverage neurons: 0

features.6.1.block.0.0:
  Channels: 1632
  Coverage - Min: -1.091914, Max: 1.000000, Mean: -0.136582
  Zero coverage neurons: 0

features.6.1.block.1.0:
  Channels: 1632
  Coverage - Min: -3.785107, Max: 1.000000, Mean: -0.023138
  Zero coverage neurons: 0

features.6.1.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.030726, Max: 1.000000, Mean: -0.178551
  Zero coverage neurons: 0

features.6.1.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.153783, Max: 1.000000, Mean: 0.004994
  Zero coverage neurons: 0

features.6.1.block.3.0:
  Channels: 272
  Coverage - Min: -1.279580, Max: 1.000000, Mean: 0.010022
  Zero coverage neurons: 0

features.6.2.block.0.0:
  Channels: 1632
  Coverage - Min: -1.233076, Max: 1.000000, Mean: -0.269918
  Zero coverage neurons: 0

features.6.2.block.1.0:
  Channels: 1632
  Coverage - Min: -2.912089, Max: 1.000000, Mean: -0.052628
  Zero coverage neurons: 0

features.6.2.block.2.fc1:
  Channels: 68
  Coverage - Min: -0.823967, Max: 1.000000, Mean: -0.025618
  Zero coverage neurons: 0

features.6.2.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.411958, Max: 1.000000, Mean: -0.063373
  Zero coverage neurons: 0

features.6.2.block.3.0:
  Channels: 272
  Coverage - Min: -1.731704, Max: 1.000000, Mean: -0.016395
  Zero coverage neurons: 0

features.6.3.block.0.0:
  Channels: 1632
  Coverage - Min: -1.629233, Max: 1.000000, Mean: -0.431744
  Zero coverage neurons: 0

features.6.3.block.1.0:
  Channels: 1632
  Coverage - Min: -2.350585, Max: 1.000000, Mean: -0.033779
  Zero coverage neurons: 0

features.6.3.block.2.fc1:
  Channels: 68
  Coverage - Min: -0.957549, Max: 1.000000, Mean: -0.061994
  Zero coverage neurons: 0

features.6.3.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.826688, Max: 1.000000, Mean: -0.119462
  Zero coverage neurons: 0

features.6.3.block.3.0:
  Channels: 272
  Coverage - Min: -0.757031, Max: 1.000000, Mean: 0.001927
  Zero coverage neurons: 0

features.6.4.block.0.0:
  Channels: 1632
  Coverage - Min: -1.269662, Max: 1.000000, Mean: -0.406494
  Zero coverage neurons: 0

features.6.4.block.1.0:
  Channels: 1632
  Coverage - Min: -0.969305, Max: 1.000000, Mean: -0.024234
  Zero coverage neurons: 0

features.6.4.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.462779, Max: 1.000000, Mean: -0.124799
  Zero coverage neurons: 0

features.6.4.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.329674, Max: 1.000000, Mean: -0.104099
  Zero coverage neurons: 0

features.6.4.block.3.0:
  Channels: 272
  Coverage - Min: -0.651209, Max: 1.000000, Mean: 0.011457
  Zero coverage neurons: 0

features.6.5.block.0.0:
  Channels: 1632
  Coverage - Min: -1.848969, Max: 1.000000, Mean: -0.689163
  Zero coverage neurons: 0

features.6.5.block.1.0:
  Channels: 1632
  Coverage - Min: -1.413017, Max: 1.000000, Mean: -0.050252
  Zero coverage neurons: 0

features.6.5.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.286442, Max: 1.000000, Mean: -0.030977
  Zero coverage neurons: 0

features.6.5.block.2.fc2:
  Channels: 1632
  Coverage - Min: -0.762867, Max: 1.000000, Mean: -0.069276
  Zero coverage neurons: 0

features.6.5.block.3.0:
  Channels: 272
  Coverage - Min: -0.929149, Max: 1.000000, Mean: 0.018805
  Zero coverage neurons: 0

features.6.6.block.0.0:
  Channels: 1632
  Coverage - Min: -2.021627, Max: 1.000000, Mean: -0.774704
  Zero coverage neurons: 0

features.6.6.block.1.0:
  Channels: 1632
  Coverage - Min: -0.663348, Max: 1.000000, Mean: -0.032887
  Zero coverage neurons: 0

features.6.6.block.2.fc1:
  Channels: 68
  Coverage - Min: -0.735706, Max: 1.000000, Mean: -0.056849
  Zero coverage neurons: 0

features.6.6.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.084334, Max: 1.000000, Mean: -0.111441
  Zero coverage neurons: 0

features.6.6.block.3.0:
  Channels: 272
  Coverage - Min: -1.197144, Max: 1.000000, Mean: -0.019196
  Zero coverage neurons: 0

features.6.7.block.0.0:
  Channels: 1632
  Coverage - Min: -0.775907, Max: 1.000000, Mean: -0.294406
  Zero coverage neurons: 0

features.6.7.block.1.0:
  Channels: 1632
  Coverage - Min: -2.746810, Max: 1.000000, Mean: -0.112359
  Zero coverage neurons: 0

features.6.7.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.409478, Max: 1.000000, Mean: -0.203822
  Zero coverage neurons: 0

features.6.7.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.180773, Max: 1.000000, Mean: -0.084658
  Zero coverage neurons: 0

features.6.7.block.3.0:
  Channels: 272
  Coverage - Min: -0.924287, Max: 1.000000, Mean: 0.019417
  Zero coverage neurons: 0

features.7.0.block.0.0:
  Channels: 1632
  Coverage - Min: -2.037144, Max: 1.000000, Mean: -0.346713
  Zero coverage neurons: 0

features.7.0.block.1.0:
  Channels: 1632
  Coverage - Min: -5.806378, Max: 1.000000, Mean: -0.090818
  Zero coverage neurons: 0

features.7.0.block.2.fc1:
  Channels: 68
  Coverage - Min: -0.735690, Max: 1.000000, Mean: -0.242751
  Zero coverage neurons: 0

features.7.0.block.2.fc2:
  Channels: 1632
  Coverage - Min: -0.617710, Max: 1.000000, Mean: 0.166306
  Zero coverage neurons: 0

features.7.0.block.3.0:
  Channels: 448
  Coverage - Min: -1.883672, Max: 1.000000, Mean: -0.011453
  Zero coverage neurons: 0

features.7.1.block.0.0:
  Channels: 2688
  Coverage - Min: -0.808675, Max: 1.000000, Mean: -0.072420
  Zero coverage neurons: 0

features.7.1.block.1.0:
  Channels: 2688
  Coverage - Min: -0.392789, Max: 1.000000, Mean: -0.009857
  Zero coverage neurons: 0

features.7.1.block.2.fc1:
  Channels: 112
  Coverage - Min: -0.879149, Max: 1.000000, Mean: -0.255391
  Zero coverage neurons: 0

features.7.1.block.2.fc2:
  Channels: 2688
  Coverage - Min: -1.182342, Max: 1.000000, Mean: -0.076595
  Zero coverage neurons: 0

features.7.1.block.3.0:
  Channels: 448
  Coverage - Min: -1.944434, Max: 1.000000, Mean: -0.039285
  Zero coverage neurons: 0

features.8.0:
  Channels: 1792
  Coverage - Min: -23.682522, Max: 1.000000, Mean: -12.510130
  Zero coverage neurons: 0

classifier.1:
  Channels: 101
  Coverage - Min: -13.779793, Max: -9.018144, Mean: -11.004374
  Zero coverage neurons: 0
============================================================


Pruning Results:
  Parameters before: 17,729,709
  Parameters after:  12,847,818
  Parameters removed: 4,881,891 (27.54%)

############################################################
Pruning Complete
############################################################
Initial parameters:  17,729,709
Final parameters:    12,847,818
Total removed:       4,881,891 (27.54%)
Target pruning ratio: 10.00%
✓ Neuron Coverage pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 1.00%                         
✓ Model Size: 49.45 MB
✓ Average Inference Time: 5.3863 ms
✓ FLOPs: 3.70 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|████████████████████████████████████████████████████████████████| 313/313 [01:45<00:00,  2.97it/s, loss=2.2700, acc=44.34%] 
Epoch 1/10 - Train Loss: 2.2700, Train Acc: 44.34%, Test Acc: 74.58%                         
Epoch 2/10: 100%|████████████████████████████████████████████████████████████████| 313/313 [01:46<00:00,  2.94it/s, loss=1.7293, acc=56.44%] 
Epoch 2/10 - Train Loss: 1.7293, Train Acc: 56.44%, Test Acc: 77.36%                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_NC_epoch2.pth
Epoch 3/10: 100%|████████████████████████████████████████████████████████████████| 313/313 [01:46<00:00,  2.93it/s, loss=1.5083, acc=62.00%]
Epoch 3/10 - Train Loss: 1.5083, Train Acc: 62.00%, Test Acc: 78.97%                         
Epoch 4/10: 100%|████████████████████████████████████████████████████████████████| 313/313 [01:46<00:00,  2.94it/s, loss=1.3840, acc=64.18%] 
Epoch 4/10 - Train Loss: 1.3840, Train Acc: 64.18%, Test Acc: 79.33%                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_NC_epoch4.pth
Epoch 5/10: 100%|████████████████████████████████████████████████████████████████| 313/313 [01:46<00:00,  2.94it/s, loss=1.3166, acc=65.86%]
Epoch 5/10 - Train Loss: 1.3166, Train Acc: 65.86%, Test Acc: 80.25%                         
Epoch 6/10: 100%|████████████████████████████████████████████████████████████████| 313/313 [01:46<00:00,  2.93it/s, loss=1.2165, acc=67.74%] 
Epoch 6/10 - Train Loss: 1.2165, Train Acc: 67.74%, Test Acc: 80.44%                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_NC_epoch6.pth
Epoch 7/10: 100%|████████████████████████████████████████████████████████████████| 313/313 [01:47<00:00,  2.92it/s, loss=1Epoch 6/10 - Train Loss: 1.2165, Train Acc: 67.74%, Test Acc: 80.44%
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_NC_epoch6.pth
Epoch 7/10: 100%|████████████████████████████████████████████████████████████████| 313/313 [01:47<00:00,  2.92it/s, loss=1.1974, acc=68.46%]
Epoch 7/10 - Train Loss: 1.1974, Train Acc: 68.46%, Test Acc: 80.83%
Epoch 8/10: 100%|████████████████████████████████████████████████████████████████| 313/313 [01:46<00:00,  2.94it/s, loss=1.1692, acc=69.26%]
Epoch 8/10 - Train Loss: 1.1692, Train Acc: 69.26%, Test Acc: 80.74%
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_NC_epoch8.pth
Epoch 9/10: 100%|████████████████████████████████████████████████████████████████| 313/313 [01:46<00:00,  2.93it/s, loss=1.1307, acc=70.76%]
Epoch 9/10 - Train Loss: 1.1307, Train Acc: 70.76%, Test Acc: 80.94%
Epoch 10/10: 100%|███████████████████████████████████████████████████████████████| 313/313 [01:44<00:00,  3.00it/s, loss=1.1172, acc=70.78%]
Epoch 10/10 - Train Loss: 1.1172, Train Acc: 70.78%, Test Acc: 80.95%
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_NC_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_NC_best.pth
  Best Accuracy: 80.95%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
Evaluating:  30%|██████████████████████████▎                                                             | 473/15 :54Evaluating:  30%|██████████████████████████▍                                                             | 475  [00:54Evaluating:  30%|██████████████████████████▌                                                             |    Evaluating:  30%|██████████████████████████▋                                                             | 479/15    Evaluating:  30%|██████████████████████████▊                                                             | 481/15    Evaluating:  31%|██████████████████████████▉                                                             | 483/15    Evaluating:  31%|███████████████████████████                                                             | 485/15    Evaluating:  31%|███████████████████████████▏                                                            | 487/15    Evaluating:  31%|███████████████████████████▎                                                            | 489/15    Evaluating:  31%|███████████████████████████▎                                                            | 491/15    Evaluating:  31%|███████████████████████████▍                                                            | 493/15    Evaluating:  31%|███████████████████████████▌                                                            | 495/15    Evaluating:  31%|███████████████████████████▋                                                            | 497/15    Evaluating:  32%|███████████████████████████▊                                                            | 499/15    Evaluating:  32%|███████████████████████████▉                                                            | 501/15    Evaluating:  32%|████████████████████████████                                                            | 503/15    Evaluating:  32%|████████████████████████████▏                                                           | 505/15    Evaluating:  32%|████████████████████████████▎                                                           | 507/15    Evaluating:  32%|████████████████████████████▎                                                           | 509/15    Evaluating:  32%|████████████████████████████▍                                                           | 511/15    Evaluating:  32%|████████████████████████████▌                                                           | 513/15    Evaluating:  33%|████████████████████████████▋                                                           | 515/15    Evaluating:  33%|████████████████████████████▊                                                           | 517/15    Evaluating:  33%|████████████████████████████▉                                                           | 519/15    Evaluating:  33%|█████████████████████████████                                                           | 521/15    Evaluating:  33%|█████████████████████████████▏                                                          | 523/15    Evaluating:  33%|█████████████████████████████▎                                                          | 525/15    Evaluating:  33%|█████████████████████████████▎                                                          | 527/15    Evaluating:  34%|█████████████████████████████▍                                                          | 529/15    Evaluating:  34%|█████████████████████████████▌                                                          | 531/15    Evaluating:  34%|█████████████████████████████▋                                                          | 533/15    Evaluating:  34%|█████████████████████████████▊                                                          | 535/15    Evaluating:  34%|█████████████████████████████▉                                                          | 537/15    Evaluating:  34%|██████████████████████████████                                                          | 539/15    Evaluating:  34%|██████████████████████████████▏                                                         | 541/15    Evaluating:  34%|██████████████████████████████▎                                                         | 543/15    Evaluating:  35%|██████████████████████████████▎                                                         | 545/15    Evaluating:  35%|██████████████████████████████▍                                                         | 547/15    Evaluating:  35%|██████████████████████████████▌                                                         | 549/15    Evaluating:  35%|██████████████████████████████▋                                                         | 551/15    Evaluating:  35%|██████████████████████████████▊                                                         | 553/15    Evaluating:  35%|██████████████████████████████▉                                                         | 555/15    Evaluating:  35%|███████████████████████████████                                                         | 557/15    Evaluating:  35%|███████████████████████████████▏                                                        | 559/15    Evaluating:  36%|███████████████████████████████▎                                                        | 561/15    Evaluating:  36%|███████████████████████████████▍                                                        | 563/15    Evaluating:  36%|███████████████████████████████▍                                                        | 565/15    Evaluating:  36%|███████████████████████████████▌                                                        | 567/15    Evaluating:  36%|███████████████████████████████▋                                                        | 569/15    Evaluating:  36%|███████████████████████████████▊                                                        | 571/15    Evaluating:  36%|███████████████████████████████▉                                                        | 573/15    Evaluating:  36%|████████████████████████████████                                                        | 575/15    Evaluating:  37%|████████████████████████████████▏                                                       | 577/15    Evaluating:  37%|████████████████████████████████▎                                                       | 579/15    Evaluating:  37%|████████████████████████████████▍                                                       | 581/15    Evaluating:  37%|████████████████████████████████▍                                                       | 583/15    Evaluating:  37%|████████████████████████████████▌                                                       | 585/15    Evaluating:  37%|████████████████████████████████▋                                                       | 587/15    Evaluating:  37%|████████████████████████████████▊                                                       | 589/15    Evaluating:  37%|████████████████████████████████▉                                                       | 591/15    Evaluating:  38%|█████████████████████████████████                                                       | 593/15    Evaluating:  38%|█████████████████████████████████▏                                                      | 595/15    Evaluating:  38%|█████████████████████████████████▎                                                      | 597/15    Evaluating:  38%|█████████████████████████████████▍                                                      | 599/15    Evaluating:  38%|█████████████████████████████████▍                                                      | 601/15    Evaluating:  38%|█████████████████████████████████▌                                                      | 603/15    Evaluating:  38%|█████████████████████████████████▋                                                      | 605/15    Evaluating:  38%|█████████████████████████████████▊                                                      | 607/15    Evaluating:  39%|█████████████████████████████████▉                                                      | 609/15    Evaluating:  39%|██████████████████████████████████                                                      | 611/15    Evaluating:  39%|██████████████████████████████████▏                                                     | 613/15    Evaluating:  39%|██████████████████████████████████▎                                                     | 615/15    Evaluating:  39%|██████████████████████████████████▍                                                     | 617/15    Evaluating:  39%|██████████████████████████████████▍                                                     | 619/15    Evaluating:  39%|██████████████████████████████████▌                                                     | 621/15    Evaluating:  39%|██████████████████████████████████▋                                                     | 623/15    Evaluating:  40%|██████████████████████████████████▊                                                     | 625/15    Evaluating:  40%|██████████████████████████████████▉                                                     | 627/15    Evaluating:  40%|███████████████████████████████████                                                     | 629/15    Evaluating:  40%|███████████████████████████████████▏                                                    | 631/15    Evaluating:  40%|███████████████████████████████████▎                                                    | 633/15    Evaluating:  40%|███████████████████████████████████▍                                                    | 635/15    Evaluating:  40%|███████████████████████████████████▌                                                    | 637/15    Evaluating:  40%|███████████████████████████████████▌                                                    | 639/15    Evaluating:  41%|███████████████████████████████████▋                                                    | 641/15    Evaluating:  41%|███████████████████████████████████▊                                                    | 643/15    Evaluating:  41%|███████████████████████████████████▉                                                    | 645/15    Evaluating:  41%|████████████████████████████████████                                                    | 647/15    Evaluating:  41%|████████████████████████████████████▏                                                   | 649/15    Evaluating:  41%|████████████████████████████████████▎                                                   | 651/15    Evaluating:  41%|████████████████████████████████████▍                                                   | 653/15    Evaluating:  41%|████████████████████████████████████▌                                                   | 655/15    Evaluating:  42%|████████████████████████████████████▌                                                   | 657/15    Evaluating:  42%|████████████████████████████████████▋                                                   | 659/15    Evaluating:  42%|████████████████████████████████████▊                                                   | 661/15    Evaluating:  42%|████████████████████████████████████▉                                                   | 663/15    Evaluating:  42%|█████████████████████████████████████                                                   | 665/15    Evaluating:  42%|█████████████████████████████████████▏                                                  | 667/15    Evaluating:  42%|█████████████████████████████████████▎                                                  | 669/15    Evaluating:  42%|█████████████████████████████████████▍                                                  | 671/15    Evaluating:  43%|█████████████████████████████████████▌                                                  | 673/15    Evaluating:  43%|█████████████████████████████████████▌                                                  | 675/15    Evaluating:  43%|█████████████████████████████████████▋                                                  | 677/15    Evaluating:  43%|█████████████████████████████████████▊                                                  | 679/15    Evaluating:  43%|█████████████████████████████████████▉                                                  | 681/15    Evaluating:  43%|██████████████████████████████████████                                                  | 683/15    Evaluating:  43%|██████████████████████████████████████▏                                                 | 685/15    Evaluating:  44%|██████████████████████████████████████▎                                                 | 687/15    Evaluating:  44%|██████████████████████████████████████▍                                                 | 689/15    Evaluating:  44%|██████████████████████████████████████▌                                                 | 691/15    Evaluating:  44%|██████████████████████████████████████▌                                                 | 693/15    Evaluating:  44%|██████████████████████████████████████▋                                                 | 695/15    Evaluating:  44%|██████████████████████████████████████▊                                                 | 697/15    Evaluating:  44%|██████████████████████████████████████▉                                                 | 699/15    Evaluating:  44%|███████████████████████████████████████                                                 | 701/15    Evaluating:  45%|███████████████████████████████████████▏                                                | 703/15Evaluating:  45%|███████████████████████████████████████▎                                                | 705/15 Evaluating:  45%|███████████████████████████████████████▍                                                | 707/15  Evaluating:  45%|███████████████████████████████████████▌                                                | 709/15   Evaluating:  45%|███████████████████████████████████████▋                                                | 711/15    Evaluating:  45%|███████████████████████████████████████▋                                                | 713/15    Evaluating:  45%|███████████████████████████████████████▊                                                | 715/15    Evaluating:  45%|███████████████████████████████████████▉                                                | 717/15    Evaluating:  46%|████████████████████████████████████████                                                | 719/15    %|████████████████████████████████████████▏                                               | 721/1579 [01Evaluatin    ███████████████████████████▎                                               | 723/1579 [01Evaluating:  46%|███████    █████████████▍                                               | 725/1579 [01Evaluating:  46%|█████████████████████                                                   | 727/1579 [01Evaluating:  46%|███████████████████████████████████                                     | 729/1579 [01Evaluating:  46%|████████████████████████████████████████▋                                             | 729/1579 [01Evaluating:  46%|████████████████████████████████████████▋                                  | 731/1579 [01Evaluating:  46%|████████████████████████████████████████▊                                   | 733/1579 [01Evaluating:  47%|████████████████████████████████████████▉                                   | 735/1579 [01Evaluating:  47%|█████████████████████████████████████████                                           79 [01Evaluating:  47%|█████████████████████████████████████████▏                                              |     aluating:  47%|█████████████████████████████████████████▎                                              | 741/1579    :  47%|█████████████████████████████████████████▍                                              | 743/1579 [01Eval    ████████████████████████████████████████▌                                              | 745/1579 [01Evaluating:     ████████████████████████████████▋                                              | 747/1579 [01Evaluating:  47%|███    ████████████████████████▋                                              | 749/1579 [01Evaluating:  48%|███████████    ██████████████▉                                              | 753/1579 [01Evaluating:  48%|█████████████████████    █████████████                                              | 755/1579 [01Evaluating:  48%|███████████████████████    ███████████▏                                             | 757/1579 [01Evaluating:  48%|█████████████████████████    █████████▎                                             | 759/1579 [01Evaluating:  48%|███████████████████████████    ██████████▍                                             | 761/1579 [01Evaluating:  48%|██████████████████████████    ███████████▌                                             | 763/1579 [01Evaluating:  48%|█████████████████████████    ████████████▋                                             | 765/1579 [01Evaluating:  49%|████████████████████████    █████████████▋                                             | 767/1579 [01:19<01:12, 11.20iEvaluating:  49%|██████    ███████████████████████████████▊                                             | 769Evaluating:  53%|██████████████    █████████████████████████████▊                                         | 841/157Evaluating:  53%|████████████████    ███████████████████████████▉                                         | 843/1579 [01:26<01:04, 11.4Evaluating:  54    ███████████████████████████▉                                         | 843/1579 [01:26<01:04, 11.4Evaluating:  54    %|███████████████████████████████████████████████                                         | 845Evaluating:  54%|█    ██████████████████████████████████████████████▏                                        | 847/1579 [01:26<01:03, 1    1.Evaluating:  54%|███████████████████████████████████████████████▎                                        | 849/    1579 [01:26<01:04, 11.Evaluating:  54%|████████████████████████████████████████████Evaluating:  54%|█████████████    ██████████████████████████████████▌                                        |Evaluating:  54%|████████████████████    ███████████████████████████▋                                        | 855/1579 [01:27<01:03,Evaluating:  54%|████    ███████████████████████████████████████████▊                                        | 857/1579 [01:27<01:03,Evalu    ating:  54%|███████████████████████████████████████████████▊                Evaluating:  55%|████████████████████    ███████████████████████████▉                                        | 861/1579 [01:27<01:0Evaluating:  55%|██████    ██████████████████████████████████████████                                        | 863/1579 [01:28<01:0Evaluatin    g:  55%|████████████████████████████████████████████████▏                                       | 865/1579 [01:28    <01:0Evaluating:  55%|████████████████████████████████████████████████▎                                       | 8    67/1579 [01:28<01:0Evaluating:  55%|████████████████████████████████████████████████▍                                           | 869/1579 [01:28<01:0Evaluating:  55%|████████████████████████████████████████████████▌                                          Evaluating:  55%|████████████████████████████████████████████████▋                                           | 873/1579 [01:2Evaluating:  55%|████████████████████████████████████████████████▊                                           | 875/1579 [01:2Evaluating:  56%|████████████████████████████████Evaluating:  56%|███████    █████████████████████████████████████████▉                                       | 879/1579 [0Evaluating:  56%|██    ███████████████████████████████████████████████                                       | 881/1579 [0Evaluating:  5    6%|█████████████████████████████████████████████████▏                                      | 883/1579 [0Evaluatin    g:  56%|█████████████████████████████████████████████████▎                                      | 885/1579 [0Eval    uating:  56%|█████████████████████████████████████████████████▍   Evaluating:  56%|██████████████████████████████    ███████████████████▌                                      | 889/1579 Evaluating:  56%|███████████████████████████    Evaluating:  57%|█████████████████████████████████████████████████▊                                      | 893/15    Evaluating:  57%|█████████████████████████████████████████████████▉                                      | 895/15    Evaluating:  57%|█████████████████████████████████████████████████▉                                      | 897/15    Evaluating:  57%|██████████████████████████████████████████████████                                      | 899/15    Evaluating:  57%|██████████████████████████████████████████████████▏                                     | 901/15    Evaluating:  57%|██████████████████████████████████████████████████▎                                     | 903/15    Evaluating:  57%|██████████████████████████████████████████████████▍                                     | 905/15    Evaluating:  57%|██████████████████████████████████████████████████▌                                     | 907/15    Evaluating:  58%|██████████████████████████████████████████████████▋                                     | 909/15    Evaluating:  58%|██████████████████████████████████████████████████▊                                     | 911/15    Evaluating:  58%|██████████████████████████████████████████████████▉                                     | 913/15    Evaluating:  58%|██████████████████████████████████████████████████▉                                     | 915/15    Evaluating:  58%|███████████████████████████████████████████████████                                     | 917/15    Evaluating:  58%|███████████████████████████████████████████████████▏                                    | 919/15    Evaluating:  58%|███████████████████████████████████████████████████▎                                    | 921/15    Evaluating:  58%|███████████████████████████████████████████████████▍                                    | 923/15    Evaluating:  59%|███████████████████████████████████████████████████▌                                    | 925/15    Evaluating:  59%|███████████████████████████████████████████████████▋                                    | 927/15    Evaluating:  59%|███████████████████████████████████████████████████▊                                    | 929/15    Evaluating:  59%|███████████████████████████████████████████████████▉                                    | 931/15    Evaluating:  59%|███████████████████████████████████████████████████▉                                    | 933/15    Evaluating:  59%|████████████████████████████████████████████████████                                    | 935/15    Evaluating:  59%|████████████████████████████████████████████████████▏                                   | 937/15    Evaluating:  59%|████████████████████████████████████████████████████▎                                   | 939/15    Evaluating:  60%|████████████████████████████████████████████████████▍                                   | 941/15    Evaluating:  60%|████████████████████████████████████████████████████▌                                   | 943/15    Evaluating:  60%|████████████████████████████████████████████████████▋                                   | 945/15    Evaluating:  60%|████████████████████████████████████████████████████▊                                   | 947/15    Evaluating:  60%|████████████████████████████████████████████████████▉                                   | 949/15    Evaluating:  60%|█████████████████████████████████████████████████████                                   | 951/15    Evaluating:  60%|█████████████████████████████████████████████████████                                   | 953/15    Evaluating:  60%|█████████████████████████████████████████████████████▏                                  | 955/15    Evaluating:  61%|█████████████████████████████████████████████████████▎                                  | 957/15    Evaluating:  61%|█████████████████████████████████████████████████████▍                                  | 959/15    Evaluating:  61%|█████████████████████████████████████████████████████▌                                  | 961/15    Evaluating:  61%|█████████████████████████████████████████████████████▋                                  | 963/15    Evaluating:  61%|█████████████████████████████████████████████████████▊                                  | 965/15    Evaluating:  61%|█████████████████████████████████████████████████████▉                                  | 967/15    Evaluating:  61%|██████████████████████████████████████████████████████                                  | 969/15    Evaluating:  61%|██████████████████████████████████████████████████████                                  | 971/15    Evaluating:  62%|██████████████████████████████████████████████████████▏                                 | 973/15    Evaluating:  62%|██████████████████████████████████████████████████████▎                                 | 975/15    Evaluating:  62%|██████████████████████████████████████████████████████▍                                 | 977/15    Evaluating:  62%|██████████████████████████████████████████████████████▌                                 | 979/15    Evaluating:  62%|██████████████████████████████████████████████████████▋                                 | 981/15    Evaluating:  62%|██████████████████████████████████████████████████████▊                                 | 983/15    Evaluating:  62%|██████████████████████████████████████████████████████▉                                 | 985/15    Evaluating:  63%|███████████████████████████████████████████████████████                                 | 987/15    Evaluating:  63%|███████████████████████████████████████████████████████                                 | 989/15    Evaluating:  63%|███████████████████████████████████████████████████████▏                                | 991/15    Evaluating:  63%|███████████████████████████████████████████████████████▎                                | 993/15    Evaluating:  63%|███████████████████████████████████████████████████████▍                                | 995/15    Evaluating:  63%|███████████████████████████████████████████████████████▌                                | 997/15    Evaluating:  63%|███████████████████████████████████████████████████████▋                                | 999/15    Evaluating:  63%|███████████████████████████████████████████████████████▏                               | 1001/15    Evaluating:  64%|███████████████████████████████████████████████████████▎                               | 1003/15    Evaluating:  64%|███████████████████████████████████████████████████████▎                               | 1005/15    Evaluating:  64%|███████████████████████████████████████████████████████▍                               | 1007/15Evaluating:  64%|███████████████████████████████████████████████████████▌                               | 1009/15 Evaluating:  64%|███████████████████████████████████████████████████████▋                               | 1011/15  Evaluating:  64%|███████████████████████████████████████████████████████▊                               | 1013/15   Evaluating:  64%|███████████████████████████████████████████████████████▉                               | 1015/15    Evaluating:  64%|████████████████████████████████████████████████████████                               | 1017/15    Evaluating:  65%|████████████████████████████████████████████████████████▏                              | 1019/15    Evaluating:  65%|████████████████████████████████████████████████████████▎                              | 1021/15    Evaluating:  65%|████████████████████████████████████████████████████████▎                              | 1023/15    Evaluating:  65%|████████████████████████████████████████████████████████▍                              | 1025/15    Evaluating:  65%|████████████████████████████████████████████████████████▌                              | 1027/15    Evaluating:  65%|████████████████████████████████████████████████████████▋                              | 1029/15    Evaluating:  65%|████████████████████████████████████████████████████████▊                              | 1031/15    Evaluating:  65%|████████████████████████████████████████████████████████▉                              | 1033/15    Evaluating:  66%|█████████████████████████████████████████████████████████                              | 1035/15    Evaluating:  66%|█████████████████████████████████████████████████████████▏                             | 1037/15    Evaluating:  66%|█████████████████████████████████████████████████████████▏                             | 1039/15    Evaluating:  66%|█████████████████████████████████████████████████████████▎                             | 1041/15    Evaluating:  66%|█████████████████████████████████████████████████████████▍                             | 1043/15    Evaluating:  66%|█████████████████████████████████████████████████████████▌                             | 1045/15    Evaluating:  66%|█████████████████████████████████████████████████████████▋                             | 1047/15    Evaluating:  66%|█████████████████████████████████████████████████████████▊                             | 1049/15    Evaluating:  67%|█████████████████████████████████████████████████████████▉                             | 1051/15    Evaluating:  67%|██████████████████████████████████████████████████████████                             | 1053/15    Evaluating:  67%|██████████████████████████████████████████████████████████▏                            | 1055/15    Evaluating:  67%|██████████████████████████████████████████████████████████▏                            | 1057/15    Evaluating:  67%|██████████████████████████████████████████████████████████▎                            | 1059/15    Evaluating:  67%|██████████████████████████████████████████████████████████▍                            | 1061/15    Evaluating:  67%|██████████████████████████████████████████████████████████▌                            | 1063/15    Evaluating:  67%|██████████████████████████████████████████████████████████▋                            | 1065/15    Evaluating:  68%|██████████████████████████████████████████████████████████▊                            | 1067/15    Evaluating:  68%|██████████████████████████████████████████████████████████▉                            | 1069/15    Evaluating:  68%|███████████████████████████████████████████████████████████                            | 1071/15    Evaluating:  68%|███████████████████████████████████████████████████████████                            | 1073/15    Evaluating:  68%|███████████████████████████████████████████████████████████▏                           | 1075/15    Evaluating:  68%|███████████████████████████████████████████████████████████▎                           | 1077/15    Evaluating:  68%|███████████████████████████████████████████████████████████▍                           | 1079/15    Evaluating:  68%|███████████████████████████████████████████████████████████▌                           | 1081/15    Evaluating:  69%|███████████████████████████████████████████████████████████▋                           | 1083/15    Evaluating:  69%|███████████████████████████████████████████████████████████▊                           | 1085/15    Evaluating:  69%|███████████████████████████████████████████████████████████▉                           | 1087/15    Evaluating:  69%|████████████████████████████████████████████████████████████                           | 1089/15    Evaluating:  69%|████████████████████████████████████████████████████████████                           | 1091/15    Evaluating:  69%|████████████████████████████████████████████████████████████▏                          | 1093/15    Evaluating:  69%|████████████████████████████████████████████████████████████▎                          | 1095/15    Evaluating:  69%|████████████████████████████████████████████████████████████▍                          | 1097/15    Evaluating:  70%|████████████████████████████████████████████████████████████▌                          | 1099/15    Evaluating:  70%|████████████████████████████████████████████████████████████▋                          | 1101/15    Evaluating:  70%|████████████████████████████████████████████████████████████▊                          | 1103/15    Evaluating:  70%|████████████████████████████████████████████████████████████▉                          | 1105/15    Evaluating:  70%|████████████████████████████████████████████████████████████▉                          | 1107/15    Evaluating:  70%|█████████████████████████████████████████████████████████████                          | 1109/15    Evaluating:  70%|█████████████████████████████████████████████████████████████▏                         | 1111/15    Evaluating:  70%|█████████████████████████████████████████████████████████████▎                         | 1113/15    Evaluating:  71%|█████████████████████████████████████████████████████████████▍                         | 1115/15    Evaluating:  71%|█████████████████████████████████████████████████████████████▌                         | 1117/15    Evaluating:  71%|█████████████████████████████████████████████████████████████▋                         | 1119/15    Evaluating:  71%|█████████████████████████████████████████████████████████████▊                         | 1121/15    Evaluating:  71%|█████████████████████████████████████████████████████████████▉                         | 1123/15    Evaluating:  71%|█████████████████████████████████████████████████████████████▉                         | 1125/15    Evaluating:  71%|██████████████████████████████████████████████████████████████                         | 1127/15    Evaluating:  72%|██████████████████████████████████████████████████████████████▏                        | 1129/15    Evaluating:  72%|██████████████████████████████████████████████████████████████▎                        | 1131/15    Evaluating:  72%|██████████████████████████████████████████████████████████████▍                        | 1133/15    Evaluating:  72%|██████████████████████████████████████████████████████████████▌                        | 1135/15    Evaluating:  72%|██████████████████████████████████████████████████████████████▋                        | 1137/15    Evaluating:  72%|██████████████████████████████████████████████████████████████▊                        | 1139/15    Evaluating:  72%|██████████████████████████████████████████████████████████████▊                        | 1141/15    Evaluating:  72%|██████████████████████████████████████████████████████████████▉                        | 1143/15    Evaluating:  73%|███████████████████████████████████████████████████████████████                        | 1145/15    Evaluating:  73%|███████████████████████████████████████████████████████████████▏                       | 1147/15    Evaluating:  73%|███████████████████████████████████████████████████████████████▎                       | 1149/15    Evaluating:  73%|███████████████████████████████████████████████████████████████▍                       | 1151/15    Evaluating:  73%|███████████████████████████████████████████████████████████████▌                       | 1153/15    Evaluating:  73%|███████████████████████████████████████████████████████████████▋                       | 1155/15    Evaluating:  73%|███████████████████████████████████████████████████████████████▋                       | 1157/15    Evaluating:  73%|███████████████████████████████████████████████████████████████▊                       | 1159/15    Evaluating:  74%|███████████████████████████████████████████████████████████████▉                       | 1161/15    Evaluating:  74%|████████████████████████████████████████████████████████████████                       | 1163/15    Evaluating:  74%|████████████████████████████████████████████████████████████████▏                      | 1165/15    Evaluating:  74%|████████████████████████████████████████████████████████████████▎                      | 1167/15    Evaluating:  74%|████████████████████████████████████████████████████████████████▍                      | 1169/15    Evaluating:  74%|████████████████████████████████████████████████████████████████▌                      | 1171/15    Evaluating:  74%|████████████████████████████████████████████████████████████████▋                      | 1173/15    Evaluating:  74%|████████████████████████████████████████████████████████████████▋                      | 1175/15    Evaluating:  75%|████████████████████████████████████████████████████████████████▊                      | 1177/15    Evaluating:  75%|████████████████████████████████████████████████████████████████▉                      | 1179/15    Evaluating:  75%|█████████████████████████████████████████████████████████████████                      | 1181/15    Evaluating:  75%|█████████████████████████████████████████████████████████████████▏                     | 1183/15    Evaluating:  75%|█████████████████████████████████████████████████████████████████▎                     | 1185/15    Evaluating:  75%|█████████████████████████████████████████████████████████████████▍                     | 1187/15    Evaluating:  75%|█████████████████████████████████████████████████████████████████▌                     | 1189/15    Evaluating:  75%|█████████████████████████████████████████████████████████████████▌                     | 1191/15    Evaluating:  76%|█████████████████████████████████████████████████████████████████▋                     | 1193/15    Evaluating:  76%|█████████████████████████████████████████████████████████████████▊                     | 1195/15    Evaluating:  76%|█████████████████████████████████████████████████████████████████▉                     | 1197/15    Evaluating:  76%|██████████████████████████████████████████████████████████████████                     | 1199/15    Evaluating:  76%|██████████████████████████████████████████████████████████████████▏                    | 1201/15    Evaluating:  76%|██████████████████████████████████████████████████████████████████▎                    | 1203/15    Evaluating:  76%|██████████████████████████████████████████████████████████████████▍                    | 1205/15    Evaluating:  76%|██████████████████████████████████████████████████████████████████▌                    | 1207/15    Evaluating:  77%|██████████████████████████████████████████████████████████████████▌                    | 1209/15    Evaluating:  77%|██████████████████████████████████████████████████████████████████▋                    | 1211/15Eva ing:  77%|██████████████████████████████████████████████████████████████████▊                    | 1213/15Evaluating 7%|██████████████████████████████████████████████████████████████████▉                    | 1215/15Evaluating:  77%| ███████████████████████████████████████████████████████████████                    | 1217/15Evaluating:  77%|███████ ████████████████████████████████████████████████████████▏                   | 1219/15Evaluating:  77%|██████████████ █████████████████████████████████████████████████▎                   | 1221/15Evaluating:  77%|█████████████████████ ██████████████████████████████████████████▍                   | 1223/15Evaluating:  78%|████████████████████████████ ███████████████████████████████████▍                   | 1225/15Evaluating:  78%|███████████████████████████████████ ████████████████████████████▌                   | 1227/15Evaluating:  78%|██████████████████████████████████████████ █████████████████████▋                   | 1229/15Evaluating:  78%|█████████████████████████████████████████████████ ██████████████▊                   | 1231/15Evaluating:  78%|████████████████████████████████████████████████████████ ███████▉                   | 1233/15Evaluating:  78%|███████████████████████████████████████████████████████████████ █                   | 1235/15Evaluating:  78%|████████████████████████████████████████████████████████████████████▏                  | 1237/15Evaluating:  78%|████████████████████████████████████████████████████████████████████▎                  | 1239/15Evaluating:  79%|████████████████████████████████████████████████████████████████████▍                 | 1241/15Evaluating:  79%|████████████████████████████████████████████████████████████████████▍                   | 1243/15Evaluating:  79%|████████████████████████████████████████████████████████████████████▌                   | 1245/15Evaluating:  79%|████████████████████████████████████████████████████████████████████▋                  |  1247/15Evaluating:  79%|████████████████████████████████████████████████████████████████████▊                  | 12 49/15Evaluating:  79%|████████████████████████████████████████████████████████████████████▉                  | 1251/ 15Evaluating:  79%|█████████████████████████████████████████████████████████████████████                  | 1253/15E valuating:  79%|█████████████████████████████████████████████████████████████████████▏                 | 1255/15Eval uating:  80%|████████████████████████Evaluating:  80%|██████████████████████████████████████████████████████████████ ███████▎                 | 1259/1579 [02:Evaluating:  80%|██████████████████████████████████████████████████████████ ███████████▍                 | 1261/1579 [02:Evaluating:  80%|██████████████████████████████████████████████████████ ███████████████▌                 | 1263/1579 [02:Evaluating:  80%|█████████████████████████████████████████████████████████████████████▋                 | 1265/1579 [02:Evaluating:  80%|█████████████████████████████████████████████████████████████████████▊                 | 1267/1579 [02:Evaluating:  80%|█████████████████████████████████████████████████████████████████████▉                 | 1269/1579 [02:Evaluating:  80%|██████████████████████████████████████████████████████████████████████                 | 1271/1579 [02:Evaluating:  81%|██████████████████████████████████████████████████████████████████████▏                | 1273/1579 [02:Evaluating:  81%|██████████████████████████████████████████████████████████████████████▎                | 1275/1579 [02:Evaluating:  81%|██████████████████████████████████████████████████████████████████████▎                | 1277/1579 [02:Evaluating:  81%|██████████████████████████████████████████████▎                | 1275/1579 [02:Evaluating:  81%|██████████████████████████████████████████████████████████████████████▎                | 1277/1579 [02:Evaluating:  81%|██████████████████████████████████████████████████████████████████████▍                | 1279/1579 [02:Evaluating:  81%|██████████████████████████████████████████████████████████████████████▌                | 1281/1579 [02:Evaluating:  81%|██████████████████████████████████████████████████████████████████████▋                | 1283/1579 [02:Evaluating:  81%|███████████████████████████████Evaluating:  82%|██████████████████████████████████████████████████████████████████████▉                | 1287/1579 [Evaluating:  82%|███████████████████████████████████████████████████████████████████████                | 1289/1579 [Evaluating:  82%|███████████████████████████████████████████████████████████████████████▏    Evaluating:  82%|███████████████████████████████████████████████████████████████████████▏               | 1293/1579 Evaluating:  82%|███████████████████████████████████████████████████████████████████████▎               | 1295/1579 Evaluating:  82%|███████████████████████████████████████████████████████████████████████▍               | 1297/1579 Evaluating:  82%|███████████████████████████████████████████████████████████████████████▌               | 1299/1579 Evaluating:  82%|███████████████████████████████████████████████████████████████████████▋               | 1301/1579 Evaluating:  83%|███████████████████████████████████████████████████████████████████████▊               | 1303/1579 Evaluating:  83%|███████████████████████████████████████████████████████████████████████▉               | 1305/1579 Evaluating:  83%|████████████████████████████████████████████████████████████████████████               | 1307/1579 Evaluating:  83%|████████████████████████████████████████████████████████████████████████               | 1309/1579 Evaluating:  83%|████████████████████████████████████████████████████████████████████████▏              | 1311/1579 [02:06<00:22, 11.69it/s]Evaluating:  83%|████████████████████████████████████████████████████████████████████████▎              | 1313/1579 [Evaluating:  83%|████████████████████████████████████████████████████████████████████████▍              | 1315/1579 [Evaluating:  83%|████████████████████████████████████████████████████████████████████████▌              | 1317/1579 [Evaluating:  84%|████████████████████████████████████████████████████████████████████████▋              | 1319/1579 [Evaluating:  84%|████████████████████████████████████████████████████████████████████████▊              | 1321/1579 [Evaluating:  84%|████████████████████████████████████████████████████████████████████████▉   Evaluating:  84%|█████████████████████████████████████████████████████████████████████████              | 1325/1579 Evaluating:  84%|█████████████████████████████████████████████████████████████████████████              | 1327/1579 Evaluating:  84%|█████████████████████████████████████████████████████████████████████████▏             | 1329/1579 Evaluating:  84%|█████████████████████████████████████████████████████████████████████████▎             | 1331/1579 Evaluating:  84%|█████████████████████████████████████████████████████████████████████████▍             | 1333/1579 Evaluating:  85%|█████████████████████████████████████████████████████████████████████████▌             | 1335/1579 Evaluating:  85%|█████████████████████████████████████████████████████████████████████████▋             | 1337/1579 Evaluating:  85%|█████████████████████████████████████████████████████████████████████████▊             | 1339/1579 Evaluating:  85%|█████████████████████████████████████████████████████████████████████████▉             | 1341/1579 Evaluating:  85%|█████████████████████████████████████████████████████████████████████████▉             | 1343/1579 Evaluating:  85%|██████████████████████████████████████████████████████████████████████████             | 1345/1579 Evaluating:  85%|██████████████████████████████████████████████████████████████████████████▏            | 1347/1579 Evaluating:  85%|██████████████████████████████████████████████████████████████████████████▎            | 1349/1579 Evaluating:  86%|██████████████████████████████████████████████████████████████████████████▍            | 1351/1579 Evaluating:  86%|██████████████████████████████████████████████████████████████████████████▌            | 1353/1579 Evaluating:  86%|██████████████████████████████████████████████████████████████████████████▋            | 1355/1579 Evaluating:  86%|██████████████████████████████████████████████████████████████████████████▊            | 1357/1579 Evaluating:  86%|██████████████████████████████████████████████████████████████████████████▉            | 1359/1579 Evaluating:  86%|██████████████████████████████████████████████████████████████████████████▉            | 1361/1579 Evaluating:  86%|███████████████████████████████████████████████████████████████████████████            | 1363/1579 Evaluating:  86%|███████████████████████████████████████████████████████████████████████████▏           | 1365/1579 Evaluating:  87%|███████████████████████████████████████████████████████████████████████████▎           | 1367/1579 Evaluating:  87%|███████████████████████████████████████████████████████████████████████████▍           | 1369/1579 Evaluating:  87%|███████████████████████████████████████████████████████████████████████████▌           | 1371/1579 Evaluating:  87%|███████████████████████████████████████████████████████████████████████████▋           | 1373/1579 Evaluating:  87%|███████████████████████████████████████████████████████████████████████████▊           | 1375/1579 Evaluating:  87%|███████████████████████████████████████████████████████████████████████████▊           | 1377/1579 Evaluating:  87%|███████████████████████████████████████████████████████████████████████████▉           | 1379/1579 Evaluating:  87%|████████████████████████████████████████████████████████████████████████████           | 1381/1579 Evaluating:  88%|████████████████████████████████████████████████████████████████████████████▏          | 1383/1579 Evaluating:  88%|████████████████████████████████████████████████████████████████████████████▎          | 1385/1579 Evaluating:  88%|████████████████████████████████████████████████████████████████████████████▍          | 1387/1579 Evaluating:  88%|████████████████████████████████████████████████████████████████████████████▌          | 1389/1579 Evaluating:  88%|████████████████████████████████████████████████████████████████████████████▋          | 1391/1579 Evaluating:  88%|████████████████████████████████████████████████████████████████████████████▊          | 1393/1579 Evaluating:  88%|████████████████████████████████████████████████████████████████████████████▊          | 1395/1579 Evaluating:  88%|████████████████████████████████████████████████████████████████████████████▉          | 1397/1579 Evaluating:  89%|█████████████████████████████████████████████████████████████████████████████          | 1399/1579 Evaluating:  89%|█████████████████████████████████████████████████████████████████████████████▏         | 1401/1579 Evaluating:  89%|█████████████████████████████████████████████████████████████████████████████▎         | 1403/1579 Evaluating:  89%|█████████████████████████████████████████████████████████████████████████████▍         | 1405/1579 Evaluating:  89%|█████████████████████████████████████████████████████████████████████████████▌         | 1407/1579 Evaluating:  89%|█████████████████████████████████████████████████████████████████████████████▋         | 1409/1579 Evaluating:  89%|█████████████████████████████████████████████████████████████████████████████▋         | 1411/1579 Evaluating:  89%|█████████████████████████████████████████████████████████████████████████████▊         | 1413/1579 Evaluating:  90%|█████████████████████████████████████████████████████████████████████████████▉         | 1415/1579 Evaluating:  90%|██████████████████████████████████████████████████████████████████████████████         | 1417/1579 Evaluating:  90%|██████████████████████████████████████████████████████████████████████████████▏        | 1419/1579 Evaluating:  90%|██████████████████████████████████████████████████████████████████████████████▎        | 1421/1579 Evaluating:  90%|██████████████████████████████████████████████████████████████████████████████▍        | 1423/1579 Evaluating:  90%|██████████████████████████████████████████████████████████████████████████████▌        | 1425/1579 Evaluating:  90%|██████████████████████████████████████████████████████████████████████████████▋        | 1427/1579 Evaluating:  91%|██████████████████████████████████████████████████████████████████████████████▋        | 1429/1579 Evaluating:  91%|██████████████████████████████████████████████████████████████████████████████▊        | 1431/1579 Evaluating:  91%|██████████████████████████████████████████████████████████████████████████████▉        | 1433/1579 Evaluating:  91%|███████████████████████████████████████████████████████████████████████████████        | 1435/1579 Evaluating:  91%|███████████████████████████████████████████████████████████████████████████████▏       | 1437/1579 Evaluating:  91%|███████████████████████████████████████████████████████████████████████████████▎       | 1439/1579 Evaluating:  91%|███████████████████████████████████████████████████████████████████████████████▍       | 1441/1579 Evaluating:  91%|███████████████████████████████████████████████████████████████████████████████▌       | 1443/1579 Evaluating:  92%|███████████████████████████████████████████████████████████████████████████████▌       | 1445/1579 Evaluating:  92%|███████████████████████████████████████████████████████████████████████████████▋       | 1447/1579 Evaluating:  92%|███████████████████████████████████████████████████████████████████████████████▊       | 1449/1579 Evaluating:  92%|███████████████████████████████████████████████████████████████████████████████▉       | 1451/1579 Evaluating:  92%|████████████████████████████████████████████████████████████████████████████████       | 1453/1579 Evaluating:  92%|████████████████████████████████████████████████████████████████████████████████▏      | 1455/1579 Evaluating:  92%|████████████████████████████████████████████████████████████████████████████████▎      | 1457/1579 Evaluating:  92%|████████████████████████████████████████████████████████████████████████████████▍      | 1459/1579 Evaluating:  93%|████████████████████████████████████████████████████████████████████████████████▍      | 1461/1579 Evaluating:  93%|████████████████████████████████████████████████████████████████████████████████▌      | 1463/1579 Evaluating:  93%|████████████████████████████████████████████████████████████████████████████████▋      | 1465/1579 Evaluating:  93%|████████████████████████████████████████████████████████████████████████████████▊      | 1467/1579 Evaluating:  93%|████████████████████████████████████████████████████████████████████████████████▉      | 1469/1579 Evaluating:  93%|█████████████████████████████████████████████████████████████████████████████████      | 1471/1579 Evaluating:  93%|█████████████████████████████████████████████████████████████████████████████████▏     | 1473/1579 Evaluating:  93%|█████████████████████████████████████████████████████████████████████████████████▎     | 1475/1579 Evaluating:  94%|█████████████████████████████████████████████████████████████████████████████████▍     | 1477/1579 Evaluating:  94%|█████████████████████████████████████████████████████████████████████████████████▍     | 1479/1579 Evaluating:  94%|█████████████████████████████████████████████████████████████████████████████████▌     | 1481/1579 Evaluating:  94%|█████████████████████████████████████████████████████████████████████████████████▋     | 1483/1579 Evaluating:  94%|█████████████████████████████████████████████████████████████████████████████████▊     | 1485/1579 Evaluating:  94%|█████████████████████████████████████████████████████████████████████████████████▉     | 1487/1579 Evaluating:  94%|██████████████████████████████████████████████████████████████████████████████████     | 1489/1579 Evaluating:  94%|██████████████████████████████████████████████████████████████████████████████████▏    | 1491/1579 Evaluating:  95%|██████████████████████████████████████████████████████████████████████████████████▎    | 1493/1579 Evaluating:  95%|██████████████████████████████████████████████████████████████████████████████████▎    | 1495/1579 Evaluating:  95%|██████████████████████████████████████████████████████████████████████████████████▍    | 1497/1579 Evaluating:  95%|██████████████████████████████████████████████████████████████████████████████████▌    | 1499/1579 Evaluating:  95%|██████████████████████████████████████████████████████████████████████████████████▋    | 1501/1579 Evaluating:  95%|██████████████████████████████████████████████████████████████████████████████████▊    | 1503/1579 Evaluating:  95%|██████████████████████████████████████████████████████████████████████████████████▉    | 1505/1579 Evaluating:  95%|███████████████████████████████████████████████████████████████████████████████████    | 1507/1579 Evaluating:  96%|███████████████████████████████████████████████████████████████████████████████████▏   | 1509/1579 Evaluating:  96%|███████████████████████████████████████████████████████████████████████████████████▎   | 1511/1579 Evaluating:  96%|███████████████████████████████████████████████████████████████████████████████████▎   | 1513/1579 Evaluating:  96%|███████████████████████████████████████████████████████████████████████████████████▍   | 1515/1579 Evaluating:  96%|███████████████████████████████████████████████████████████████████████████████████▌   | 1517/1579 Evaluating:  96%|███████████████████████████████████████████████████████████████████████████████████▋   | 1519/1579 Evaluating:  96%|███████████████████████████████████████████████████████████████████████████████████▊   | 1521/1579 Evaluating:  96%|███████████████████████████████████████████████████████████████████████████████████▉   | 1523/1579 Evaluating:  97%|████████████████████████████████████████████████████████████████████████████████████   | 1525/1579 Evaluating:  97%|████████████████████████████████████████████████████████████████████████████████████▏  | 1527/1579 Evaluating:  97%|████████████████████████████████████████████████████████████████████████████████████▏  | 1529/1579 Evaluating:  97%|████████████████████████████████████████████████████████████████████████████████████▎  | 1531/1579 Evaluating:  97%|████████████████████████████████████████████████████████████████████████████████████▍  | 1533/1579 Evaluating:  97%|████████████████████████████████████████████████████████████████████████████████████▌  | 1535/1579 Evaluating:  97%|████████████████████████████████████████████████████████████████████████████████████▋  | 1537/1579 Evaluating:  97%|████████████████████████████████████████████████████████████████████████████████████▊  | 1539/1579 Evaluating:  98%|████████████████████████████████████████████████████████████████████████████████████▉  | 1541/1579 Evaluating:  98%|█████████████████████████████████████████████████████████████████████████████████████  | 1543/1579 Evaluating:  98%|█████████████████████████████████████████████████████████████████████████████████████▏ | 1545/1579 Evaluating:  98%|█████████████████████████████████████████████████████████████████████████████████████▏ | 1547/1579 Evaluating:  98%|█████████████████████████████████████████████████████████████████████████████████████▎ | 1549/1579 Evaluating:  98%|█████████████████████████████████████████████████████████████████████████████████████▍ | 1551/1579 Evaluating:  98%|█████████████████████████████████████████████████████████████████████████████████████▌ | 1553/1579 Evaluating:  98%|█████████████████████████████████████████████████████████████████████████████████████▋ | 1555/1579 Evaluating:  99%|█████████████████████████████████████████████████████████████████████████████████████▊ | 1557/1579 Evaluating:  99%|█████████████████████████████████████████████████████████████████████████████████████▉ | 1559/1579 Evaluating:  99%|██████████████████████████████████████████████████████████████████████████████████████ | 1561/1579 Evaluating:  99%|██████████████████████████████████████████████████████████████████████████████████████ | 1563/1579 Evaluating:  99%|██████████████████████████████████████████████████████████████████████████████████████▏| 1565/1579 Evaluating:  99%|██████████████████████████████████████████████████████████████████████████████████████▎| 1567/1579 Evaluating:  99%|██████████████████████████████████████████████████████████████████████████████████████▍| 1569/1579 Evaluating:  99%|██████████████████████████████████████████████████████████████████████████████████████▌| 1571/1579 Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████▋| 1573/1579 Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████▊| 1575/1579 Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████▉| 1577/1579 Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████| 1579/1579                                                                                                                     
✓ Final Model Accuracy: 80.95%
✓ Model Size: 49.45 MB
✓ Average Inference Time: 5.3230 ms
✓ FLOPs: 3.70 GFLOPs

================================================================================
COMPARISON: NEURON COVERAGE PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         89.2    |          1      |     80.95  | -8.25                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         68.11   |         49.45   |     49.45  | -18.66 (-27.4%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.5975 |          5.3863 |      5.323 | +4.7255                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.61   |          3.7    |      3.7   | -0.91 (-19.7%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS6_EfficientNetB4_Food101\TS6_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================





**************************



================================================================================
TEST SCENARIO TS6: WANDA PRUNING
================================================================================
Model: EfficientNetB4
Dataset: Food101
Method: Wanda
Device: cuda
Pruning Ratio: 10.0%
================================================================================

================================================================================
LOADING FOOD101 DATASET
================================================================================
✓ Dataset loaded
  - Validation samples: 25250
  - Training samples (subset): 5000
  - Calibration samples: 1600

================================================================================
LOADING BASELINE MODEL
================================================================================
✓ Model loaded from: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FT_best.pth
  - Epoch: 9
  - Test Accuracy: 89.20%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 89.20%                                                                                   
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.5935 ms
✓ FLOPs: 4.61 GFLOPs

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
Initial parameters: 17,729,709

============================================================
Pruning Step 1/1
============================================================

============================================================
WANDA: Computing Weight × Activation Importance
============================================================
Registered hooks on 161 layers
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
  features.0.0: coverage shape torch.Size([48]), min=0.000005, max=9.570285, mean=1.188647
  features.1.0.block.0.0: coverage shape torch.Size([48]), min=0.000002, max=70.475891, mean=6.115322
  features.1.0.block.1.fc1: coverage shape torch.Size([12]), min=1.594100, max=16.342318, mean=5.839727
  features.1.0.block.1.fc2: coverage shape torch.Size([48]), min=0.478122, max=12.208261, mean=5.437280
  features.1.0.block.2.0: coverage shape torch.Size([24]), min=8.071848, max=206.758743, mean=72.715538
  features.1.1.block.0.0: coverage shape torch.Size([24]), min=0.009486, max=18.797279, mean=4.467664
  features.1.1.block.1.fc1: coverage shape torch.Size([6]), min=2.373862, max=15.584175, mean=8.972976
  features.1.1.block.1.fc2: coverage shape torch.Size([24]), min=1.510930, max=4.556255, mean=2.943042
  features.1.1.block.2.0: coverage shape torch.Size([24]), min=0.116686, max=103.217537, mean=24.054199
  features.2.0.block.0.0: coverage shape torch.Size([144]), min=0.008392, max=87.453186, mean=15.653802
  features.2.0.block.1.0: coverage shape torch.Size([144]), min=0.043333, max=69.647926, mean=14.064722
  features.2.0.block.2.fc1: coverage shape torch.Size([6]), min=1.584254, max=5.057578, mean=2.893535
  features.2.0.block.2.fc2: coverage shape torch.Size([144]), min=0.145425, max=5.321406, mean=3.026818
  features.2.0.block.3.0: coverage shape torch.Size([32]), min=8.624709, max=263.364838, mean=103.548874
  features.2.1.block.0.0: coverage shape torch.Size([192]), min=0.005800, max=149.439011, mean=22.218931
  features.2.1.block.1.0: coverage shape torch.Size([192]), min=0.000359, max=21.099800, mean=1.370066
  features.2.1.block.2.fc1: coverage shape torch.Size([8]), min=0.875271, max=34.984241, mean=7.151401
  features.2.1.block.2.fc2: coverage shape torch.Size([192]), min=0.006051, max=32.400982, mean=5.846864
  features.2.1.block.3.0: coverage shape torch.Size([32]), min=1.617949, max=129.789108, mean=44.872833
  features.2.2.block.0.0: coverage shape torch.Size([192]), min=0.000006, max=77.460419, mean=18.462326
  features.2.2.block.1.0: coverage shape torch.Size([192]), min=0.000000, max=24.519821, mean=1.771008
  features.2.2.block.2.fc1: coverage shape torch.Size([8]), min=1.268744, max=28.256720, mean=5.331277
  features.2.2.block.2.fc2: coverage shape torch.Size([192]), min=0.114301, max=12.199249, mean=3.344899
  features.2.2.block.3.0: coverage shape torch.Size([32]), min=2.594090, max=51.776978, mean=18.235826
  features.2.3.block.0.0: coverage shape torch.Size([192]), min=0.195222, max=101.027191, mean=24.825638
  features.2.3.block.1.0: coverage shape torch.Size([192]), min=0.000072, max=14.725036, mean=1.471231
  features.2.3.block.2.fc1: coverage shape torch.Size([8]), min=2.440099, max=2.997169, mean=2.630338
  features.2.3.block.2.fc2: coverage shape torch.Size([192]), min=0.122995, max=5.080687, mean=1.629416
  features.2.3.block.3.0: coverage shape torch.Size([32]), min=0.370438, max=36.105156, mean=12.945640
  features.3.0.block.0.0: coverage shape torch.Size([192]), min=0.163784, max=147.474869, mean=39.654003
  features.3.0.block.1.0: coverage shape torch.Size([192]), min=0.002477, max=62.528366, mean=4.223199
  features.3.0.block.2.fc1: coverage shape torch.Size([8]), min=7.869696, max=55.656227, mean=22.707430
  features.3.0.block.2.fc2: coverage shape torch.Size([192]), min=0.417609, max=23.194859, mean=7.138456
  features.3.0.block.3.0: coverage shape torch.Size([56]), min=1.469187, max=229.994095, mean=63.517220
  features.3.1.block.0.0: coverage shape torch.Size([336]), min=0.009970, max=61.449024, mean=7.894484
  features.3.1.block.1.0: coverage shape torch.Size([336]), min=0.000272, max=29.530479, mean=1.591154
  features.3.1.block.2.fc1: coverage shape torch.Size([14]), min=0.187410, max=34.173901, mean=16.716377
  features.3.1.block.2.fc2: coverage shape torch.Size([336]), min=0.034520, max=97.661110, mean=12.623178
  features.3.1.block.3.0: coverage shape torch.Size([56]), min=0.093188, max=27.719746, mean=10.913754
  features.3.2.block.0.0: coverage shape torch.Size([336]), min=0.020835, max=71.044281, mean=12.326323
  features.3.2.block.1.0: coverage shape torch.Size([336]), min=0.000148, max=22.885935, mean=1.529981
  features.3.2.block.2.fc1: coverage shape torch.Size([14]), min=0.284116, max=44.847599, mean=9.800695
  features.3.2.block.2.fc2: coverage shape torch.Size([336]), min=0.310052, max=101.584572, mean=7.622476
  features.3.2.block.3.0: coverage shape torch.Size([56]), min=0.383615, max=26.730564, mean=7.076074
  features.3.3.block.0.0: coverage shape torch.Size([336]), min=0.002118, max=77.930527, mean=19.348635
  features.3.3.block.1.0: coverage shape torch.Size([336]), min=0.000616, max=21.598316, mean=1.616776
  features.3.3.block.2.fc1: coverage shape torch.Size([14]), min=1.199705, max=27.610624, mean=10.267140
  features.3.3.block.2.fc2: coverage shape torch.Size([336]), min=0.973803, max=90.446579, mean=9.577975
  features.3.3.block.3.0: coverage shape torch.Size([56]), min=0.020350, max=11.530581, mean=3.740807
  features.4.0.block.0.0: coverage shape torch.Size([336]), min=0.328755, max=111.859413, mean=26.280407
  features.4.0.block.1.0: coverage shape torch.Size([336]), min=0.000644, max=45.327797, mean=2.406143
  features.4.0.block.2.fc1: coverage shape torch.Size([14]), min=0.024156, max=26.980434, mean=7.907677
  features.4.0.block.2.fc2: coverage shape torch.Size([336]), min=1.961545, max=4.862790, mean=3.064029
  features.4.0.block.3.0: coverage shape torch.Size([112]), min=2.669891, max=135.678177, mean=39.967682
  features.4.1.block.0.0: coverage shape torch.Size([672]), min=0.012347, max=81.473763, mean=8.707880
  features.4.1.block.1.0: coverage shape torch.Size([672]), min=0.000041, max=15.291220, mean=0.760079
  features.4.1.block.2.fc1: coverage shape torch.Size([28]), min=0.258721, max=45.659332, mean=13.795403
  features.4.1.block.2.fc2: coverage shape torch.Size([672]), min=0.131325, max=196.571304, mean=21.193892
  features.4.1.block.3.0: coverage shape torch.Size([112]), min=0.074145, max=9.379838, mean=2.446699
  features.4.2.block.0.0: coverage shape torch.Size([672]), min=0.016897, max=50.005161, mean=5.870855
  features.4.2.block.1.0: coverage shape torch.Size([672]), min=0.000092, max=9.302204, mean=0.889911
  features.4.2.block.2.fc1: coverage shape torch.Size([28]), min=0.007776, max=1.511010, mean=0.654948
  features.4.2.block.2.fc2: coverage shape torch.Size([672]), min=0.002150, max=5.650177, mean=1.270486
  features.4.2.block.3.0: coverage shape torch.Size([112]), min=0.004577, max=1.397341, mean=0.343649
  features.4.3.block.0.0: coverage shape torch.Size([672]), min=0.015618, max=44.913971, mean=8.057437
  features.4.3.block.1.0: coverage shape torch.Size([672]), min=0.000039, max=13.208421, mean=0.461833
  features.4.3.block.2.fc1: coverage shape torch.Size([28]), min=0.118634, max=34.386475, mean=10.594604
  features.4.3.block.2.fc2: coverage shape torch.Size([672]), min=0.021731, max=174.693344, mean=16.628258
  features.4.3.block.3.0: coverage shape torch.Size([112]), min=0.005162, max=5.298434, mean=1.901119
  features.4.4.block.0.0: coverage shape torch.Size([672]), min=0.022949, max=38.949974, mean=12.267711
  features.4.4.block.1.0: coverage shape torch.Size([672]), min=0.000217, max=6.945410, mean=0.647510
  features.4.4.block.2.fc1: coverage shape torch.Size([28]), min=0.227180, max=12.598496, mean=4.539231
  features.4.4.block.2.fc2: coverage shape torch.Size([672]), min=0.006524, max=28.603308, mean=5.991187
  features.4.4.block.3.0: coverage shape torch.Size([112]), min=0.003928, max=2.703108, mean=0.736547
  features.4.5.block.0.0: coverage shape torch.Size([672]), min=0.011053, max=55.061588, mean=14.408757
  features.4.5.block.1.0: coverage shape torch.Size([672]), min=0.000064, max=12.086459, mean=0.872341
  features.4.5.block.2.fc1: coverage shape torch.Size([28]), min=0.011563, max=3.669663, mean=1.387655
  features.4.5.block.2.fc2: coverage shape torch.Size([672]), min=0.009619, max=9.269710, mean=2.064347
  features.4.5.block.3.0: coverage shape torch.Size([112]), min=0.000885, max=1.536901, mean=0.339827
  features.5.0.block.0.0: coverage shape torch.Size([672]), min=0.080569, max=68.459290, mean=14.917103
  features.5.0.block.1.0: coverage shape torch.Size([672]), min=0.000261, max=139.405960, mean=1.706770
  features.5.0.block.2.fc1: coverage shape torch.Size([28]), min=1.117783, max=26.979675, mean=8.286932
  features.5.0.block.2.fc2: coverage shape torch.Size([672]), min=0.030711, max=8.908279, mean=2.464932
  features.5.0.block.3.0: coverage shape torch.Size([160]), min=0.142513, max=101.712639, mean=24.122259
  features.5.1.block.0.0: coverage shape torch.Size([960]), min=0.001786, max=19.853218, mean=2.528156
  features.5.1.block.1.0: coverage shape torch.Size([960]), min=0.000026, max=13.281586, mean=0.753042
  features.5.1.block.2.fc1: coverage shape torch.Size([40]), min=0.003723, max=3.344900, mean=0.906705
  features.5.1.block.2.fc2: coverage shape torch.Size([960]), min=0.000066, max=6.732561, mean=1.531858
  features.5.1.block.3.0: coverage shape torch.Size([160]), min=0.001293, max=1.518489, mean=0.408218
  features.5.2.block.0.0: coverage shape torch.Size([960]), min=0.033074, max=18.225740, mean=4.962642
  features.5.2.block.1.0: coverage shape torch.Size([960]), min=0.000620, max=12.421055, mean=0.612971
  features.5.2.block.2.fc1: coverage shape torch.Size([40]), min=0.013349, max=3.235063, mean=1.170354
  features.5.2.block.2.fc2: coverage shape torch.Size([960]), min=0.004887, max=6.207378, mean=1.423541
  features.5.2.block.3.0: coverage shape torch.Size([160]), min=0.004963, max=1.341853, mean=0.310409
  features.5.3.block.0.0: coverage shape torch.Size([960]), min=0.015266, max=34.361027, mean=8.191531
  features.5.3.block.1.0: coverage shape torch.Size([960]), min=0.000246, max=14.064580, mean=0.651178
  features.5.3.block.2.fc1: coverage shape torch.Size([40]), min=0.037252, max=6.118243, mean=1.891896
  features.5.3.block.2.fc2: coverage shape torch.Size([960]), min=0.004316, max=13.013511, mean=2.135455
  features.5.3.block.3.0: coverage shape torch.Size([160]), min=0.004157, max=1.902496, mean=0.405181
  features.5.4.block.0.0: coverage shape torch.Size([960]), min=0.038357, max=40.201633, mean=12.220301
  features.5.4.block.1.0: coverage shape torch.Size([960]), min=0.000149, max=27.117779, mean=0.654977
  features.5.4.block.2.fc1: coverage shape torch.Size([40]), min=0.003637, max=5.051436, mean=0.952037
  features.5.4.block.2.fc2: coverage shape torch.Size([960]), min=0.005959, max=6.378989, mean=1.410942
  features.5.4.block.3.0: coverage shape torch.Size([160]), min=0.001033, max=0.970343, mean=0.275086
  features.5.5.block.0.0: coverage shape torch.Size([960]), min=0.014091, max=66.410477, mean=18.053423
  features.5.5.block.1.0: coverage shape torch.Size([960]), min=0.000227, max=13.960898, mean=0.569938
  features.5.5.block.2.fc1: coverage shape torch.Size([40]), min=0.053653, max=3.036252, mean=0.876760
  features.5.5.block.2.fc2: coverage shape torch.Size([960]), min=0.002129, max=6.884548, mean=1.084510
  features.5.5.block.3.0: coverage shape torch.Size([160]), min=0.000335, max=1.020957, mean=0.324611
  features.6.0.block.0.0: coverage shape torch.Size([960]), min=0.065138, max=66.367767, mean=21.038359
  features.6.0.block.1.0: coverage shape torch.Size([960]), min=0.000702, max=130.708084, mean=1.732548
  features.6.0.block.2.fc1: coverage shape torch.Size([40]), min=0.767627, max=45.875011, mean=7.804797
  features.6.0.block.2.fc2: coverage shape torch.Size([960]), min=1.113122, max=4.892104, mean=2.312621
  features.6.0.block.3.0: coverage shape torch.Size([272]), min=0.295370, max=108.847824, mean=29.547148
  features.6.1.block.0.0: coverage shape torch.Size([1632]), min=0.005084, max=12.808135, mean=2.370197
  features.6.1.block.1.0: coverage shape torch.Size([1632]), min=0.000128, max=27.753069, mean=0.374088
  features.6.1.block.2.fc1: coverage shape torch.Size([68]), min=0.023853, max=3.583997, mean=1.142831
  features.6.1.block.2.fc2: coverage shape torch.Size([1632]), min=0.000993, max=5.992636, mean=1.388433
  features.6.1.block.3.0: coverage shape torch.Size([272]), min=0.000147, max=3.892729, mean=0.495089
  features.6.2.block.0.0: coverage shape torch.Size([1632]), min=0.044284, max=14.247544, mean=3.783187
  features.6.2.block.1.0: coverage shape torch.Size([1632]), min=0.000072, max=9.955866, mean=0.317556
  features.6.2.block.2.fc1: coverage shape torch.Size([68]), min=0.000178, max=3.249152, mean=0.749737
  features.6.2.block.2.fc2: coverage shape torch.Size([1632]), min=0.001326, max=6.109069, mean=1.092298
  features.6.2.block.3.0: coverage shape torch.Size([272]), min=0.006127, max=2.896483, mean=0.373929
  features.6.3.block.0.0: coverage shape torch.Size([1632]), min=0.031687, max=29.706028, mean=8.363130
  features.6.3.block.1.0: coverage shape torch.Size([1632]), min=0.000029, max=12.323219, mean=0.306187
  features.6.3.block.2.fc1: coverage shape torch.Size([68]), min=0.002466, max=3.552847, mean=0.941424
  features.6.3.block.2.fc2: coverage shape torch.Size([1632]), min=0.001178, max=6.466455, mean=1.283758
  features.6.3.block.3.0: coverage shape torch.Size([272]), min=0.000231, max=1.674850, mean=0.344362
  features.6.4.block.0.0: coverage shape torch.Size([1632]), min=0.004473, max=39.143082, mean=13.355037
  features.6.4.block.1.0: coverage shape torch.Size([1632]), min=0.000329, max=8.501222, mean=0.304415
  features.6.4.block.2.fc1: coverage shape torch.Size([68]), min=0.004725, max=3.373990, mean=0.941068
  features.6.4.block.2.fc2: coverage shape torch.Size([1632]), min=0.000067, max=5.843535, mean=1.141636
  features.6.4.block.3.0: coverage shape torch.Size([272]), min=0.000210, max=1.593228, mean=0.298785
  features.6.5.block.0.0: coverage shape torch.Size([1632]), min=0.042942, max=50.569958, mean=19.592445
  features.6.5.block.1.0: coverage shape torch.Size([1632]), min=0.000132, max=6.486299, mean=0.316726
  features.6.5.block.2.fc1: coverage shape torch.Size([68]), min=0.006536, max=2.743574, mean=0.792182
  features.6.5.block.2.fc2: coverage shape torch.Size([1632]), min=0.000245, max=6.264595, mean=1.108882
  features.6.5.block.3.0: coverage shape torch.Size([272]), min=0.000600, max=1.056913, mean=0.258131
  features.6.6.block.0.0: coverage shape torch.Size([1632]), min=0.004795, max=65.507874, mean=26.304094
  features.6.6.block.1.0: coverage shape torch.Size([1632]), min=0.000047, max=7.217981, mean=0.318080
  features.6.6.block.2.fc1: coverage shape torch.Size([68]), min=0.036047, max=2.752984, mean=0.712102
  features.6.6.block.2.fc2: coverage shape torch.Size([1632]), min=0.000098, max=4.878676, mean=1.090930
  features.6.6.block.3.0: coverage shape torch.Size([272]), min=0.001197, max=1.145360, mean=0.277484
  features.6.7.block.0.0: coverage shape torch.Size([1632]), min=0.069229, max=106.343231, mean=32.830860
  features.6.7.block.1.0: coverage shape torch.Size([1632]), min=0.000217, max=6.320518, mean=0.321550
  features.6.7.block.2.fc1: coverage shape torch.Size([68]), min=0.017275, max=2.714064, mean=0.862422
  features.6.7.block.2.fc2: coverage shape torch.Size([1632]), min=0.002286, max=5.437206, mean=0.986737
  features.6.7.block.3.0: coverage shape torch.Size([272]), min=0.004768, max=1.609838, mean=0.310095
  features.7.0.block.0.0: coverage shape torch.Size([1632]), min=0.080991, max=186.173950, mean=46.136871
  features.7.0.block.1.0: coverage shape torch.Size([1632]), min=0.000000, max=95.248802, mean=1.985910
  features.7.0.block.2.fc1: coverage shape torch.Size([68]), min=0.103895, max=7.645433, mean=2.637768
  features.7.0.block.2.fc2: coverage shape torch.Size([1632]), min=0.007732, max=15.847843, mean=2.936387
  features.7.0.block.3.0: coverage shape torch.Size([448]), min=0.001967, max=9.424610, mean=0.691492
  features.7.1.block.0.0: coverage shape torch.Size([2688]), min=0.000172, max=4.400664, mean=0.858840
  features.7.1.block.1.0: coverage shape torch.Size([2688]), min=0.000087, max=20.638578, mean=0.648986
  features.7.1.block.2.fc1: coverage shape torch.Size([112]), min=0.013543, max=5.164683, mean=1.559041
  features.7.1.block.2.fc2: coverage shape torch.Size([2688]), min=0.000441, max=12.220177, mean=1.955827
  features.7.1.block.3.0: coverage shape torch.Size([448]), min=0.001860, max=2.376958, mean=0.270264
  features.8.0: coverage shape torch.Size([1792]), min=0.104432, max=12.778315, mean=7.067520
  classifier.1: coverage shape torch.Size([101]), min=9.152007, max=14.078601, mean=11.058645

------------------------------------------------------------
Activation Statistics:
------------------------------------------------------------

features.0.0:
  Channels: 48
  Activation - Min: 0.000005, Max: 9.570285, Mean: 1.188647

features.1.0.block.0.0:
  Channels: 48
  Activation - Min: 0.000002, Max: 70.475891, Mean: 6.115322

features.1.0.block.1.fc1:
  Channels: 12
  Activation - Min: 1.594100, Max: 16.342318, Mean: 5.839727

features.1.0.block.1.fc2:
  Channels: 48
  Activation - Min: 0.478122, Max: 12.208261, Mean: 5.437280

features.1.0.block.2.0:
  Channels: 24
  Activation - Min: 8.071848, Max: 206.758743, Mean: 72.715538

features.1.1.block.0.0:
  Channels: 24
  Activation - Min: 0.009486, Max: 18.797279, Mean: 4.467664

features.1.1.block.1.fc1:
  Channels: 6
  Activation - Min: 2.373862, Max: 15.584175, Mean: 8.972976

features.1.1.block.1.fc2:
  Channels: 24
  Activation - Min: 1.510930, Max: 4.556255, Mean: 2.943042

features.1.1.block.2.0:
  Channels: 24
  Activation - Min: 0.116686, Max: 103.217537, Mean: 24.054199

features.2.0.block.0.0:
  Channels: 144
  Activation - Min: 0.008392, Max: 87.453186, Mean: 15.653802

features.2.0.block.1.0:
  Channels: 144
  Activation - Min: 0.043333, Max: 69.647926, Mean: 14.064722

features.2.0.block.2.fc1:
  Channels: 6
  Activation - Min: 1.584254, Max: 5.057578, Mean: 2.893535

features.2.0.block.2.fc2:
  Channels: 144
  Activation - Min: 0.145425, Max: 5.321406, Mean: 3.026818

features.2.0.block.3.0:
  Channels: 32
  Activation - Min: 8.624709, Max: 263.364838, Mean: 103.548874

features.2.1.block.0.0:
  Channels: 192
  Activation - Min: 0.005800, Max: 149.439011, Mean: 22.218931

features.2.1.block.1.0:
  Channels: 192
  Activation - Min: 0.000359, Max: 21.099800, Mean: 1.370066

features.2.1.block.2.fc1:
  Channels: 8
  Activation - Min: 0.875271, Max: 34.984241, Mean: 7.151401

features.2.1.block.2.fc2:
  Channels: 192
  Activation - Min: 0.006051, Max: 32.400982, Mean: 5.846864

features.2.1.block.3.0:
  Channels: 32
  Activation - Min: 1.617949, Max: 129.789108, Mean: 44.872833

features.2.2.block.0.0:
  Channels: 192
  Activation - Min: 0.000006, Max: 77.460419, Mean: 18.462326

features.2.2.block.1.0:
  Channels: 192
  Activation - Min: 0.000000, Max: 24.519821, Mean: 1.771008

features.2.2.block.2.fc1:
  Channels: 8
  Activation - Min: 1.268744, Max: 28.256720, Mean: 5.331277

features.2.2.block.2.fc2:
  Channels: 192
  Activation - Min: 0.114301, Max: 12.199249, Mean: 3.344899

features.2.2.block.3.0:
  Channels: 32
  Activation - Min: 2.594090, Max: 51.776978, Mean: 18.235826

features.2.3.block.0.0:
  Channels: 192
  Activation - Min: 0.195222, Max: 101.027191, Mean: 24.825638

features.2.3.block.1.0:
  Channels: 192
  Activation - Min: 0.000072, Max: 14.725036, Mean: 1.471231

features.2.3.block.2.fc1:
  Channels: 8
  Activation - Min: 2.440099, Max: 2.997169, Mean: 2.630338

features.2.3.block.2.fc2:
  Channels: 192
  Activation - Min: 0.122995, Max: 5.080687, Mean: 1.629416

features.2.3.block.3.0:
  Channels: 32
  Activation - Min: 0.370438, Max: 36.105156, Mean: 12.945640

features.3.0.block.0.0:
  Channels: 192
  Activation - Min: 0.163784, Max: 147.474869, Mean: 39.654003

features.3.0.block.1.0:
  Channels: 192
  Activation - Min: 0.002477, Max: 62.528366, Mean: 4.223199

features.3.0.block.2.fc1:
  Channels: 8
  Activation - Min: 7.869696, Max: 55.656227, Mean: 22.707430

features.3.0.block.2.fc2:
  Channels: 192
  Activation - Min: 0.417609, Max: 23.194859, Mean: 7.138456

features.3.0.block.3.0:
  Channels: 56
  Activation - Min: 1.469187, Max: 229.994095, Mean: 63.517220

features.3.1.block.0.0:
  Channels: 336
  Activation - Min: 0.009970, Max: 61.449024, Mean: 7.894484

features.3.1.block.1.0:
  Channels: 336
  Activation - Min: 0.000272, Max: 29.530479, Mean: 1.591154

features.3.1.block.2.fc1:
  Channels: 14
  Activation - Min: 0.187410, Max: 34.173901, Mean: 16.716377

features.3.1.block.2.fc2:
  Channels: 336
  Activation - Min: 0.034520, Max: 97.661110, Mean: 12.623178

features.3.1.block.3.0:
  Channels: 56
  Activation - Min: 0.093188, Max: 27.719746, Mean: 10.913754

features.3.2.block.0.0:
  Channels: 336
  Activation - Min: 0.020835, Max: 71.044281, Mean: 12.326323

features.3.2.block.1.0:
  Channels: 336
  Activation - Min: 0.000148, Max: 22.885935, Mean: 1.529981

features.3.2.block.2.fc1:
  Channels: 14
  Activation - Min: 0.284116, Max: 44.847599, Mean: 9.800695

features.3.2.block.2.fc2:
  Channels: 336
  Activation - Min: 0.310052, Max: 101.584572, Mean: 7.622476

features.3.2.block.3.0:
  Channels: 56
  Activation - Min: 0.383615, Max: 26.730564, Mean: 7.076074

features.3.3.block.0.0:
  Channels: 336
  Activation - Min: 0.002118, Max: 77.930527, Mean: 19.348635

features.3.3.block.1.0:
  Channels: 336
  Activation - Min: 0.000616, Max: 21.598316, Mean: 1.616776

features.3.3.block.2.fc1:
  Channels: 14
  Activation - Min: 1.199705, Max: 27.610624, Mean: 10.267140

features.3.3.block.2.fc2:
  Channels: 336
  Activation - Min: 0.973803, Max: 90.446579, Mean: 9.577975

features.3.3.block.3.0:
  Channels: 56
  Activation - Min: 0.020350, Max: 11.530581, Mean: 3.740807

features.4.0.block.0.0:
  Channels: 336
  Activation - Min: 0.328755, Max: 111.859413, Mean: 26.280407

features.4.0.block.1.0:
  Channels: 336
  Activation - Min: 0.000644, Max: 45.327797, Mean: 2.406143

features.4.0.block.2.fc1:
  Channels: 14
  Activation - Min: 0.024156, Max: 26.980434, Mean: 7.907677

features.4.0.block.2.fc2:
  Channels: 336
  Activation - Min: 1.961545, Max: 4.862790, Mean: 3.064029

features.4.0.block.3.0:
  Channels: 112
  Activation - Min: 2.669891, Max: 135.678177, Mean: 39.967682

features.4.1.block.0.0:
  Channels: 672
  Activation - Min: 0.012347, Max: 81.473763, Mean: 8.707880

features.4.1.block.1.0:
  Channels: 672
  Activation - Min: 0.000041, Max: 15.291220, Mean: 0.760079

features.4.1.block.2.fc1:
  Channels: 28
  Activation - Min: 0.258721, Max: 45.659332, Mean: 13.795403

features.4.1.block.2.fc2:
  Channels: 672
  Activation - Min: 0.131325, Max: 196.571304, Mean: 21.193892

features.4.1.block.3.0:
  Channels: 112
  Activation - Min: 0.074145, Max: 9.379838, Mean: 2.446699

features.4.2.block.0.0:
  Channels: 672
  Activation - Min: 0.016897, Max: 50.005161, Mean: 5.870855

features.4.2.block.1.0:
  Channels: 672
  Activation - Min: 0.000092, Max: 9.302204, Mean: 0.889911

features.4.2.block.2.fc1:
  Channels: 28
  Activation - Min: 0.007776, Max: 1.511010, Mean: 0.654948

features.4.2.block.2.fc2:
  Channels: 672
  Activation - Min: 0.002150, Max: 5.650177, Mean: 1.270486

features.4.2.block.3.0:
  Channels: 112
  Activation - Min: 0.004577, Max: 1.397341, Mean: 0.343649

features.4.3.block.0.0:
  Channels: 672
  Activation - Min: 0.015618, Max: 44.913971, Mean: 8.057437

features.4.3.block.1.0:
  Channels: 672
  Activation - Min: 0.000039, Max: 13.208421, Mean: 0.461833

features.4.3.block.2.fc1:
  Channels: 28
  Activation - Min: 0.118634, Max: 34.386475, Mean: 10.594604

features.4.3.block.2.fc2:
  Channels: 672
  Activation - Min: 0.021731, Max: 174.693344, Mean: 16.628258

features.4.3.block.3.0:
  Channels: 112
  Activation - Min: 0.005162, Max: 5.298434, Mean: 1.901119

features.4.4.block.0.0:
  Channels: 672
  Activation - Min: 0.022949, Max: 38.949974, Mean: 12.267711

features.4.4.block.1.0:
  Channels: 672
  Activation - Min: 0.000217, Max: 6.945410, Mean: 0.647510

features.4.4.block.2.fc1:
  Channels: 28
  Activation - Min: 0.227180, Max: 12.598496, Mean: 4.539231

features.4.4.block.2.fc2:
  Channels: 672
  Activation - Min: 0.006524, Max: 28.603308, Mean: 5.991187

features.4.4.block.3.0:
  Channels: 112
  Activation - Min: 0.003928, Max: 2.703108, Mean: 0.736547

features.4.5.block.0.0:
  Channels: 672
  Activation - Min: 0.011053, Max: 55.061588, Mean: 14.408757

features.4.5.block.1.0:
  Channels: 672
  Activation - Min: 0.000064, Max: 12.086459, Mean: 0.872341

features.4.5.block.2.fc1:
  Channels: 28
  Activation - Min: 0.011563, Max: 3.669663, Mean: 1.387655

features.4.5.block.2.fc2:
  Channels: 672
  Activation - Min: 0.009619, Max: 9.269710, Mean: 2.064347

features.4.5.block.3.0:
  Channels: 112
  Activation - Min: 0.000885, Max: 1.536901, Mean: 0.339827

features.5.0.block.0.0:
  Channels: 672
  Activation - Min: 0.080569, Max: 68.459290, Mean: 14.917103

features.5.0.block.1.0:
  Channels: 672
  Activation - Min: 0.000261, Max: 139.405960, Mean: 1.706770

features.5.0.block.2.fc1:
  Channels: 28
  Activation - Min: 1.117783, Max: 26.979675, Mean: 8.286932

features.5.0.block.2.fc2:
  Channels: 672
  Activation - Min: 0.030711, Max: 8.908279, Mean: 2.464932

features.5.0.block.3.0:
  Channels: 160
  Activation - Min: 0.142513, Max: 101.712639, Mean: 24.122259

features.5.1.block.0.0:
  Channels: 960
  Activation - Min: 0.001786, Max: 19.853218, Mean: 2.528156

features.5.1.block.1.0:
  Channels: 960
  Activation - Min: 0.000026, Max: 13.281586, Mean: 0.753042

features.5.1.block.2.fc1:
  Channels: 40
  Activation - Min: 0.003723, Max: 3.344900, Mean: 0.906705

features.5.1.block.2.fc2:
  Channels: 960
  Activation - Min: 0.000066, Max: 6.732561, Mean: 1.531858

features.5.1.block.3.0:
  Channels: 160
  Activation - Min: 0.001293, Max: 1.518489, Mean: 0.408218

features.5.2.block.0.0:
  Channels: 960
  Activation - Min: 0.033074, Max: 18.225740, Mean: 4.962642

features.5.2.block.1.0:
  Channels: 960
  Activation - Min: 0.000620, Max: 12.421055, Mean: 0.612971

features.5.2.block.2.fc1:
  Channels: 40
  Activation - Min: 0.013349, Max: 3.235063, Mean: 1.170354

features.5.2.block.2.fc2:
  Channels: 960
  Activation - Min: 0.004887, Max: 6.207378, Mean: 1.423541

features.5.2.block.3.0:
  Channels: 160
  Activation - Min: 0.004963, Max: 1.341853, Mean: 0.310409

features.5.3.block.0.0:
  Channels: 960
  Activation - Min: 0.015266, Max: 34.361027, Mean: 8.191531

features.5.3.block.1.0:
  Channels: 960
  Activation - Min: 0.000246, Max: 14.064580, Mean: 0.651178

features.5.3.block.2.fc1:
  Channels: 40
  Activation - Min: 0.037252, Max: 6.118243, Mean: 1.891896

features.5.3.block.2.fc2:
  Channels: 960
  Activation - Min: 0.004316, Max: 13.013511, Mean: 2.135455

features.5.3.block.3.0:
  Channels: 160
  Activation - Min: 0.004157, Max: 1.902496, Mean: 0.405181

features.5.4.block.0.0:
  Channels: 960
  Activation - Min: 0.038357, Max: 40.201633, Mean: 12.220301

features.5.4.block.1.0:
  Channels: 960
  Activation - Min: 0.000149, Max: 27.117779, Mean: 0.654977

features.5.4.block.2.fc1:
  Channels: 40
  Activation - Min: 0.003637, Max: 5.051436, Mean: 0.952037

features.5.4.block.2.fc2:
  Channels: 960
  Activation - Min: 0.005959, Max: 6.378989, Mean: 1.410942

features.5.4.block.3.0:
  Channels: 160
  Activation - Min: 0.001033, Max: 0.970343, Mean: 0.275086

features.5.5.block.0.0:
  Channels: 960
  Activation - Min: 0.014091, Max: 66.410477, Mean: 18.053423

features.5.5.block.1.0:
  Channels: 960
  Activation - Min: 0.000227, Max: 13.960898, Mean: 0.569938

features.5.5.block.2.fc1:
  Channels: 40
  Activation - Min: 0.053653, Max: 3.036252, Mean: 0.876760

features.5.5.block.2.fc2:
  Channels: 960
  Activation - Min: 0.002129, Max: 6.884548, Mean: 1.084510

features.5.5.block.3.0:
  Channels: 160
  Activation - Min: 0.000335, Max: 1.020957, Mean: 0.324611

features.6.0.block.0.0:
  Channels: 960
  Activation - Min: 0.065138, Max: 66.367767, Mean: 21.038359

features.6.0.block.1.0:
  Channels: 960
  Activation - Min: 0.000702, Max: 130.708084, Mean: 1.732548

features.6.0.block.2.fc1:
  Channels: 40
  Activation - Min: 0.767627, Max: 45.875011, Mean: 7.804797

features.6.0.block.2.fc2:
  Channels: 960
  Activation - Min: 1.113122, Max: 4.892104, Mean: 2.312621

features.6.0.block.3.0:
  Channels: 272
  Activation - Min: 0.295370, Max: 108.847824, Mean: 29.547148

features.6.1.block.0.0:
  Channels: 1632
  Activation - Min: 0.005084, Max: 12.808135, Mean: 2.370197

features.6.1.block.1.0:
  Channels: 1632
  Activation - Min: 0.000128, Max: 27.753069, Mean: 0.374088

features.6.1.block.2.fc1:
  Channels: 68
  Activation - Min: 0.023853, Max: 3.583997, Mean: 1.142831

features.6.1.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000993, Max: 5.992636, Mean: 1.388433

features.6.1.block.3.0:
  Channels: 272
  Activation - Min: 0.000147, Max: 3.892729, Mean: 0.495089

features.6.2.block.0.0:
  Channels: 1632
  Activation - Min: 0.044284, Max: 14.247544, Mean: 3.783187

features.6.2.block.1.0:
  Channels: 1632
  Activation - Min: 0.000072, Max: 9.955866, Mean: 0.317556

features.6.2.block.2.fc1:
  Channels: 68
  Activation - Min: 0.000178, Max: 3.249152, Mean: 0.749737

features.6.2.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.001326, Max: 6.109069, Mean: 1.092298

features.6.2.block.3.0:
  Channels: 272
  Activation - Min: 0.006127, Max: 2.896483, Mean: 0.373929

features.6.3.block.0.0:
  Channels: 1632
  Activation - Min: 0.031687, Max: 29.706028, Mean: 8.363130

features.6.3.block.1.0:
  Channels: 1632
  Activation - Min: 0.000029, Max: 12.323219, Mean: 0.306187

features.6.3.block.2.fc1:
  Channels: 68
  Activation - Min: 0.002466, Max: 3.552847, Mean: 0.941424

features.6.3.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.001178, Max: 6.466455, Mean: 1.283758

features.6.3.block.3.0:
  Channels: 272
  Activation - Min: 0.000231, Max: 1.674850, Mean: 0.344362

features.6.4.block.0.0:
  Channels: 1632
  Activation - Min: 0.004473, Max: 39.143082, Mean: 13.355037

features.6.4.block.1.0:
  Channels: 1632
  Activation - Min: 0.000329, Max: 8.501222, Mean: 0.304415

features.6.4.block.2.fc1:
  Channels: 68
  Activation - Min: 0.004725, Max: 3.373990, Mean: 0.941068

features.6.4.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000067, Max: 5.843535, Mean: 1.141636

features.6.4.block.3.0:
  Channels: 272
  Activation - Min: 0.000210, Max: 1.593228, Mean: 0.298785

features.6.5.block.0.0:
  Channels: 1632
  Activation - Min: 0.042942, Max: 50.569958, Mean: 19.592445

features.6.5.block.1.0:
  Channels: 1632
  Activation - Min: 0.000132, Max: 6.486299, Mean: 0.316726

features.6.5.block.2.fc1:
  Channels: 68
  Activation - Min: 0.006536, Max: 2.743574, Mean: 0.792182

features.6.5.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000245, Max: 6.264595, Mean: 1.108882

features.6.5.block.3.0:
  Channels: 272
  Activation - Min: 0.000600, Max: 1.056913, Mean: 0.258131

features.6.6.block.0.0:
  Channels: 1632
  Activation - Min: 0.004795, Max: 65.507874, Mean: 26.304094

features.6.6.block.1.0:
  Channels: 1632
  Activation - Min: 0.000047, Max: 7.217981, Mean: 0.318080

features.6.6.block.2.fc1:
  Channels: 68
  Activation - Min: 0.036047, Max: 2.752984, Mean: 0.712102

features.6.6.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000098, Max: 4.878676, Mean: 1.090930

features.6.6.block.3.0:
  Channels: 272
  Activation - Min: 0.001197, Max: 1.145360, Mean: 0.277484

features.6.7.block.0.0:
  Channels: 1632
  Activation - Min: 0.069229, Max: 106.343231, Mean: 32.830860

features.6.7.block.1.0:
  Channels: 1632
  Activation - Min: 0.000217, Max: 6.320518, Mean: 0.321550

features.6.7.block.2.fc1:
  Channels: 68
  Activation - Min: 0.017275, Max: 2.714064, Mean: 0.862422

features.6.7.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.002286, Max: 5.437206, Mean: 0.986737

features.6.7.block.3.0:
  Channels: 272
  Activation - Min: 0.004768, Max: 1.609838, Mean: 0.310095

features.7.0.block.0.0:
  Channels: 1632
  Activation - Min: 0.080991, Max: 186.173950, Mean: 46.136871

features.7.0.block.1.0:
  Channels: 1632
  Activation - Min: 0.000000, Max: 95.248802, Mean: 1.985910

features.7.0.block.2.fc1:
  Channels: 68
  Activation - Min: 0.103895, Max: 7.645433, Mean: 2.637768

features.7.0.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.007732, Max: 15.847843, Mean: 2.936387

features.7.0.block.3.0:
  Channels: 448
  Activation - Min: 0.001967, Max: 9.424610, Mean: 0.691492

features.7.1.block.0.0:
  Channels: 2688
  Activation - Min: 0.000172, Max: 4.400664, Mean: 0.858840

features.7.1.block.1.0:
  Channels: 2688
  Activation - Min: 0.000087, Max: 20.638578, Mean: 0.648986

features.7.1.block.2.fc1:
  Channels: 112
  Activation - Min: 0.013543, Max: 5.164683, Mean: 1.559041

features.7.1.block.2.fc2:
  Channels: 2688
  Activation - Min: 0.000441, Max: 12.220177, Mean: 1.955827

features.7.1.block.3.0:
  Channels: 448
  Activation - Min: 0.001860, Max: 2.376958, Mean: 0.270264

features.8.0:
  Channels: 1792
  Activation - Min: 0.104432, Max: 12.778315, Mean: 7.067520

classifier.1:
  Channels: 101
  Activation - Min: 9.152007, Max: 14.078601, Mean: 11.058645
============================================================


Pruning Results:
  Parameters before: 17,729,709
  Parameters after:  15,444,556
  Parameters removed: 2,285,153 (12.89%)

############################################################
Pruning Complete
############################################################
Initial parameters:  17,729,709
Final parameters:    15,444,556
Total removed:       2,285,153 (12.89%)
Target pruning ratio: 10.00%
✓ Wanda pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 1.67%                                                                                      
✓ Model Size: 59.35 MB
✓ Average Inference Time: 5.2605 ms
✓ FLOPs: 4.05 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|███████████████████████████████████████| 313/313 [01:44<00:00,  3.00it/s, loss=1.2187, acc=68.30%] 
Epoch 1/10 - Train Loss: 1.2187, Train Acc: 68.30%, Test Acc: 84.68%                                                
Epoch 2/10: 100%|███████████████████████████████████████| 313/313 [01:43<00:00,  3.02it/s, loss=1.0484, acc=73.34%] 
Epoch 2/10 - Train Loss: 1.0484, Train Acc: 73.34%, Test Acc: 85.14%                                                
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_W_epoch2.pth
Epoch 3/10: 100%|███████████████████████████████████████| 313/313 [01:43<00:00,  3.02it/s, loss=0.9836, acc=73.86%]
Epoch 3/10 - Train Loss: 0.9836, Train Acc: 73.86%, Test Acc: 85.63%
Epoch 4/10: 100%|███████████████████████████████████████| 313/313 [01:43<00:00,  3.03it/s, loss=0.9085, acc=76.54%] 
Epoch 4/10 - Train Loss: 0.9085, Train Acc: 76.54%, Test Acc: 85.33%                                                
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_W_epoch4.pth
Epoch 5/10: 100%|███████████████████████████████████████| 313/313 [01:43<00:00,  3.01it/s, loss=0.8481, acc=77.44%]
Epoch 5/10 - Train Loss: 0.8481, Train Acc: 77.44%, Test Acc: 85.58%                                                
Epoch 6/10: 100%|███████████████████████████████████████| 313/313 [01:43<00:00,  3.02it/s, loss=0.7954, acc=79.40%] 
Epoch 6/10 - Train Loss: 0.7954, Train Acc: 79.40%, Test Acc: 85.48%                                                
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_W_epoch6.pth
Epoch 7/10: 100%|███████████████████████████████████████| 313/313 [01:44<00:00,  3.00it/s, loss=0.7833, acc=79.32%]
Epoch 7/10 - Train Loss: 0.7833, Train Acc: 79.32%, Test Acc: 85.73%                                                
Epoch 8/10: 100%|███████████████████████████████████████| 313/313 [01:41<00:00,  3.08it/s, loss=0.7591, acc=79.94%] 
Epoch 8/10 - Train Loss: 0.7591, Train Acc: 79.94%, Test Acc: 85.80%                                                
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_W_epoch8.pth
Epoch 9/10: 100%|███████████████████████████████████████| 313/313 [01:41<00:00,  3.09it/s, loss=0.7454, acc=80.08%]
Epoch 9/10 - Train Loss: 0.7454, Train Acc: 80.08%, Test Acc: 85.98%                                                
Epoch 10/10: 100%|██████████████████████████████████████| 313/313 [01:41<00:00,  3.07it/s, loss=0.7239, acc=80.56%] 
Epoch 10/10 - Train Loss: 0.7239, Train Acc: 80.56%, Test Acc: 85.93%                                               
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_W_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_W_best.pth
  Best Accuracy: 85.98%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 85.93%                                                                                      
✓ Model Size: 59.35 MB
✓ Average Inference Time: 5.3202 ms
✓ FLOPs: 4.05 GFLOPs

================================================================================
COMPARISON: WANDA PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         89.2    |          1.67   |    85.93   | -3.27                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         68.11   |         59.35   |    59.35   | -8.76 (-12.9%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.5935 |          5.2605 |     5.3202 | +4.7267                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.61   |          4.05   |     4.05   | -0.56 (-12.1%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS6_EfficientNetB4_Food101\TS6_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================



******************************



================================================================================
TEST SCENARIO TS6: MAGNITUDE-BASED PRUNING
================================================================================
Model: EfficientNetB4
Dataset: Food101
Method: Magnitude (Torch-Pruning)
Device: cuda
Pruning Ratio: 10.0%
================================================================================

================================================================================
LOADING FOOD101 DATASET
================================================================================
✓ Dataset loaded
  - Validation samples: 25250
  - Training samples (subset): 5000

================================================================================
LOADING BASELINE MODEL
================================================================================
✓ Model loaded from: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FT_best.pth
  - Epoch: 9
  - Test Accuracy: 89.20%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 89.20%                                                                                              
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.5921 ms
✓ FLOPs: 4.61 GFLOPs

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
Initial parameters: 17,729,709

============================================================
Pruning Step 1/1
============================================================

Pruning Results:
  Parameters before: 17,729,709
  Parameters after:  16,049,077
  Parameters removed: 1,680,632 (9.48%)

############################################################
Pruning Complete
############################################################
Initial parameters:  17,729,709
Final parameters:    16,049,077
Total removed:       1,680,632 (9.48%)
Target pruning ratio: 10.00%
✓ Magnitude-based pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 2.89%                                                                                                 
✓ Model Size: 61.66 MB
✓ Average Inference Time: 0.5888 ms
✓ FLOPs: 3.91 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|██████████████████████████████████████████████████| 313/313 [01:23<00:00,  3.73it/s, loss=1.0899, acc=71.82%] 
Epoch 1/10 - Train Loss: 1.0899, Train Acc: 71.82%, Test Acc: 86.19%                                                           
Epoch 2/10: 100%|██████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.67it/s, loss=0.9481, acc=74.52%] 
Epoch 2/10 - Train Loss: 0.9481, Train Acc: 74.52%, Test Acc: 86.37%                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_MAG_epoch2.pth
Epoch 3/10: 100%|██████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.66it/s, loss=0.8748, acc=76.50%]
Epoch 3/10 - Train Loss: 0.8748, Train Acc: 76.50%, Test Acc: 86.69%                                                           
Epoch 4/10: 100%|██████████████████████████████████████████████████| 313/313 [01:24<00:00,  3.69it/s, loss=0.8086, acc=78.44%] 
Epoch 4/10 - Train Loss: 0.8086, Train Acc: 78.44%, Test Acc: 86.63%                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_MAG_epoch4.pth
Epoch 5/10: 100%|██████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.66it/s, loss=0.7764, acc=78.96%]
Epoch 5/10 - Train Loss: 0.7764, Train Acc: 78.96%, Test Acc: 86.76%                                                           
Epoch 6/10: 100%|██████████████████████████████████████████████████| 313/313 [01:24<00:00,  3.69it/s, loss=0.7524, acc=79.50%] 
Epoch 6/10 - Train Loss: 0.7524, Train Acc: 79.50%, Test Acc: 86.64%                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_MAG_epoch6.pth
Epoch 7/10: 100%|██████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.66it/s, loss=0.6777, acc=81.42%]
Epoch 7/10 - Train Loss: 0.6777, Train Acc: 81.42%, Test Acc: 86.87%                                                           
Epoch 8/10: 100%|██████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.67it/s, loss=0.6686, acc=82.26%] 
Epoch 8/10 - Train Loss: 0.6686, Train Acc: 82.26%, Test Acc: 86.79%                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_MAG_epoch8.pth
Epoch 9/10: 100%|██████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.65it/s, loss=0.6934, acc=80.64%]
Epoch 9/10 - Train Loss: 0.6934, Train Acc: 80.64%, Test Acc: 86.91%                                                           
Epoch 10/10: 100%|█████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.68it/s, loss=0.6605, acc=82.46%] 
Epoch 10/10 - Train Loss: 0.6605, Train Acc: 82.46%, Test Acc: 86.78%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_MAG_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_MAG_best.pth
  Best Accuracy: 86.91%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 86.78%                                                                                                 
✓ Model Size: 61.66 MB
✓ Average Inference Time: 0.5797 ms
✓ FLOPs: 3.91 GFLOPs

================================================================================
COMPARISON: MAGNITUDE-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         89.2    |          2.89   |    86.78   | -2.42                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         68.11   |         61.66   |    61.66   | -6.45 (-9.5%)               |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.5921 |          0.5888 |     0.5797 | -0.0125                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.61   |          3.91   |     3.91   | -0.70 (-15.2%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS6_EfficientNetB4_Food101\TS6_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================



*********************************



================================================================================
TEST SCENARIO TS6: TAYLOR GRADIENT-BASED PRUNING
================================================================================
Model: EfficientNetB4
Dataset: Food101
Method: Taylor (Gradient-based)
Device: cuda
Pruning Ratio: 10.0%
================================================================================

================================================================================
LOADING FOOD101 DATASET
================================================================================
✓ Dataset loaded
  - Validation samples: 25250
  - Training samples (subset): 5000
  - Calibration samples: 1600

================================================================================
LOADING BASELINE MODEL
================================================================================
✓ Model loaded from: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FT_best.pth
  - Epoch: 9
  - Test Accuracy: 89.20%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 89.20%                                                                                              
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.5862 ms
✓ FLOPs: 4.61 GFLOPs

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
Computing gradients:  99%|██████████████████████████████████████████████████████████████████▎| 99/100 [00:40<00:00,  2.45it/s]
✓ Gradients computed on 100 batches
Using GroupTaylorImportance

Initializing Torch-Pruning with Taylor importance...

Applying pruning...

✓ Taylor pruning completed
  Parameters before: 17,729,709
  Parameters after: 15,389,241
  Parameters removed: 2,340,468 (13.20%)

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 19.73%                                                                                                
✓ Model Size: 59.14 MB
✓ Average Inference Time: 0.5869 ms
✓ FLOPs: 4.11 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|██████████████████████████████████████████████████| 313/313 [01:29<00:00,  3.50it/s, loss=0.9849, acc=74.64%] 
Epoch 1/10 - Train Loss: 0.9849, Train Acc: 74.64%, Test Acc: 88.00%                                                           
Epoch 2/10: 100%|██████████████████████████████████████████████████| 313/313 [01:29<00:00,  3.51it/s, loss=0.8716, acc=77.20%] 
Epoch 2/10 - Train Loss: 0.8716, Train Acc: 77.20%, Test Acc: 87.89%                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_TAY_epoch2.pth
Epoch 3/10: 100%|██████████████████████████████████████████████████| 313/313 [01:29<00:00,  3.50it/s, loss=0.8109, acc=78.02%]
Epoch 3/10 - Train Loss: 0.8109, Train Acc: 78.02%, Test Acc: 87.86%                                                           
Epoch 4/10: 100%|██████████████████████████████████████████████████| 313/313 [01:30<00:00,  3.45it/s, loss=0.7613, acc=79.62%] 
Epoch 4/10 - Train Loss: 0.7613, Train Acc: 79.62%, Test Acc: 87.31%                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_TAY_epoch4.pth
Epoch 5/10: 100%|██████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.7261, acc=80.54%]
Epoch 5/10 - Train Loss: 0.7261, Train Acc: 80.54%, Test Acc: 87.87%                                                           
Epoch 6/10: 100%|██████████████████████████████████████████████████| 313/313 [01:30<00:00,  3.45it/s, loss=0.6934, acc=81.58%] 
Epoch 6/10 - Train Loss: 0.6934, Train Acc: 81.58%, Test Acc: 87.93%                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_TAY_epoch6.pth
Epoch 7/10: 100%|██████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.42it/s, loss=0.6696, acc=81.74%]
Epoch 7/10 - Train Loss: 0.6696, Train Acc: 81.74%, Test Acc: 87.87%                                                           
Epoch 8/10: 100%|██████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.6315, acc=83.06%] 
Epoch 8/10 - Train Loss: 0.6315, Train Acc: 83.06%, Test Acc: 87.85%                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_TAY_epoch8.pth
Epoch 9/10: 100%|██████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.42it/s, loss=0.6049, acc=83.26%]
Epoch 9/10 - Train Loss: 0.6049, Train Acc: 83.26%, Test Acc: 87.85%                                                           
Epoch 10/10: 100%|█████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.42it/s, loss=0.6129, acc=83.42%] 
Epoch 10/10 - Train Loss: 0.6129, Train Acc: 83.42%, Test Acc: 87.99%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_TAY_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS6\EfficientNetB4_Food101_FTAP_TAY_best.pth
  Best Accuracy: 88.00%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 87.99%                                                                                                 
✓ Model Size: 59.14 MB
✓ Average Inference Time: 0.5905 ms
✓ FLOPs: 4.11 GFLOPs

================================================================================
COMPARISON: TAYLOR GRADIENT-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         89.2    |         19.73   |    87.99   | -1.21                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         68.11   |         59.14   |    59.14   | -8.97 (-13.2%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.5862 |          0.5869 |     0.5905 | +0.0043                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.61   |          4.11   |     4.11   | -0.50 (-10.9%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS6_EfficientNetB4_Food101\TS6_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================