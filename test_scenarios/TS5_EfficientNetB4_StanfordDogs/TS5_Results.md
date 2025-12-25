================================================================================
TEST SCENARIO TS5: PREPARE EfficientNetB4 MODEL
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
✓ Pretrained Model Accuracy: 0.91%                                                                                                                               
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.6094 ms
✓ FLOPs: 4.61 GFLOPs

================================================================================
FINE-TUNING MODEL ON STANFORD DOGS
================================================================================
Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:01<00:00,  3.58it/s, loss=2.2377, acc=45.84%] 
Epoch 1/10 - Train Loss: 2.2377, Train Acc: 45.84%, Test Acc: 78.96%                                                                                             
Epoch 2/10: 100%|██████████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:01<00:00,  3.58it/s, loss=1.4217, acc=63.70%]
Epoch 2/10 - Train Loss: 1.4217, Train Acc: 63.70%, Test Acc: 83.50%                                                                                             
Epoch 3/10: 100%|██████████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:03<00:00,  3.58it/s, loss=1.2020, acc=68.90%]
Epoch 3/10 - Train Loss: 1.2020, Train Acc: 68.90%, Test Acc: 85.62%                                                                                             
Epoch 4/10: 100%|██████████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:05<00:00,  3.57it/s, loss=1.0803, acc=72.20%]
Epoch 4/10 - Train Loss: 1.0803, Train Acc: 72.20%, Test Acc: 87.39%                                                                                             
Epoch 5/10: 100%|██████████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:04<00:00,  3.57it/s, loss=1.0028, acc=73.94%]
Epoch 5/10 - Train Loss: 1.0028, Train Acc: 73.94%, Test Acc: 87.79%                                                                                             
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FT_epoch5.pth
Epoch 6/10: 100%|██████████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:05<00:00,  3.57it/s, loss=0.9289, acc=75.53%]
Epoch 6/10 - Train Loss: 0.9289, Train Acc: 75.53%, Test Acc: 88.21%                                                                                             
Epoch 7/10: 100%|██████████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:04<00:00,  3.58it/s, loss=0.8715, acc=77.10%]
Epoch 7/10 - Train Loss: 0.8715, Train Acc: 77.10%, Test Acc: 88.78%                                                                                             
Epoch 8/10: 100%|██████████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:04<00:00,  3.58it/s, loss=0.8303, acc=78.10%]
Epoch 8/10 - Train Loss: 0.8303, Train Acc: 78.10%, Test Acc: 88.89%                                                                                             
Epoch 9/10: 100%|██████████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:02<00:00,  3.58it/s, loss=0.8145, acc=78.42%]
Epoch 9/10 - Train Loss: 0.8145, Train Acc: 78.42%, Test Acc: 89.21%                                                                                             
Epoch 10/10: 100%|█████████████████████████████████████████████████████████████████████████████████| 4735/4735 [22:05<00:00,  3.57it/s, loss=0.8097, acc=78.69%]
Epoch 10/10 - Train Loss: 0.8097, Train Acc: 78.69%, Test Acc: 89.06%                                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FT_epoch10.pth
c:\source\repos\cleanai-v5\test_scenarios\TS5_EfficientNetB4_StanfordDogs\TS5_01_prepare_model.py:345: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(best_checkpoint, map_location=CONFIG['device'])

================================================================================
FINAL EVALUATION (AFTER FINE-TUNING)
================================================================================
✓ Fine-tuned Model Accuracy: 89.21%                                                                                                                              
✓ Model Size: 68.12 MB
✓ Average Inference Time: 0.5761 ms
✓ FLOPs: 4.61 GFLOPs

================================================================================
COMPARISON: BEFORE vs AFTER FINE-TUNING
================================================================================
+---------------------+--------------------------+-------------------------+----------+
| Metric              |   Pretrained (Before FT) |   Fine-tuned (After FT) |   Change |
+=====================+==========================+=========================+==========+
| Accuracy (%)        |                   0.91   |                 89.21   |  88.3    |
+---------------------+--------------------------+-------------------------+----------+
| Size (MB)           |                  68.11   |                 68.12   |   0      |
+---------------------+--------------------------+-------------------------+----------+
| Inference Time (ms) |                   0.6094 |                  0.5761 |  -0.0333 |
+---------------------+--------------------------+-------------------------+----------+
| FLOPs (G)           |                   4.61   |                  4.61   |   0      |
+---------------------+--------------------------+-------------------------+----------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS5_EfficientNetB4_StanfordDogs\TS5_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================



************************************




  features.3.0.block.3.0: coverage shape torch.Size([56]), min=-0.779956, max=1.000000, mean=0.015043
  features.3.1.block.0.0: coverage shape torch.Size([336]), min=-0.901222, max=1.000000, mean=0.009629
  features.3.1.block.1.0: coverage shape torch.Size([336]), min=-3.666774, max=1.000000, mean=-0.120686
  features.3.1.block.2.fc1: coverage shape torch.Size([14]), min=-0.024983, max=1.000000, mean=0.382133
  features.3.1.block.2.fc2: coverage shape torch.Size([336]), min=-1.509517, max=1.000000, mean=0.157532
  features.3.1.block.3.0: coverage shape torch.Size([56]), min=-1.029634, max=1.000000, mean=0.033186
  features.3.2.block.0.0: coverage shape torch.Size([336]), min=-2.007234, max=1.000000, mean=-0.217789
  features.3.2.block.1.0: coverage shape torch.Size([336]), min=-1.934926, max=1.000000, mean=-0.071799
  features.3.2.block.2.fc1: coverage shape torch.Size([14]), min=-0.304844, max=1.000000, mean=0.111786
  features.3.2.block.2.fc2: coverage shape torch.Size([336]), min=-2.987664, max=1.000000, mean=0.125695
  features.3.2.block.3.0: coverage shape torch.Size([56]), min=-0.767084, max=1.000000, mean=0.027163
  features.3.3.block.0.0: coverage shape torch.Size([336]), min=-0.996078, max=1.000000, mean=-0.146168
  features.3.3.block.1.0: coverage shape torch.Size([336]), min=-1.741781, max=1.000000, mean=-0.063261
  features.3.3.block.2.fc1: coverage shape torch.Size([14]), min=-0.629044, max=1.000000, mean=-0.006891
  features.3.3.block.2.fc2: coverage shape torch.Size([336]), min=-0.695708, max=1.000000, mean=0.068257
  features.3.3.block.3.0: coverage shape torch.Size([56]), min=-0.734313, max=1.000000, mean=-0.023183
  features.4.0.block.0.0: coverage shape torch.Size([336]), min=-0.751614, max=1.000000, mean=-0.159840
  features.4.0.block.1.0: coverage shape torch.Size([336]), min=-4.024682, max=1.000000, mean=-0.188975
  features.4.0.block.2.fc1: coverage shape torch.Size([14]), min=-0.951480, max=1.000000, mean=-0.156340
  features.4.0.block.2.fc2: coverage shape torch.Size([336]), min=-0.300682, max=1.000000, mean=0.424969
  features.4.0.block.3.0: coverage shape torch.Size([112]), min=-0.998053, max=1.000000, mean=-0.018512
  features.4.1.block.0.0: coverage shape torch.Size([672]), min=-0.921940, max=1.000000, mean=-0.001201
  features.4.1.block.1.0: coverage shape torch.Size([672]), min=-2.885710, max=1.000000, mean=-0.044020
  features.4.1.block.2.fc1: coverage shape torch.Size([28]), min=-0.785222, max=1.000000, mean=0.100551
  features.4.1.block.2.fc2: coverage shape torch.Size([672]), min=-1.284496, max=1.000000, mean=0.021283
  features.4.1.block.3.0: coverage shape torch.Size([112]), min=-0.689937, max=1.000000, mean=0.029136
  features.4.2.block.0.0: coverage shape torch.Size([672]), min=-0.481996, max=1.000000, mean=-0.057298
  features.4.2.block.1.0: coverage shape torch.Size([672]), min=-1.457130, max=1.000000, mean=-0.005181
  features.4.2.block.2.fc1: coverage shape torch.Size([28]), min=-1.032089, max=1.000000, mean=0.233805
  features.4.2.block.2.fc2: coverage shape torch.Size([672]), min=-2.086202, max=1.000000, mean=-0.035682
  features.4.2.block.3.0: coverage shape torch.Size([112]), min=-0.811981, max=1.000000, mean=0.029871
  features.4.3.block.0.0: coverage shape torch.Size([672]), min=-1.564362, max=1.000000, mean=-0.151875
  features.4.3.block.1.0: coverage shape torch.Size([672]), min=-2.400361, max=1.000000, mean=-0.029290
  features.4.3.block.2.fc1: coverage shape torch.Size([28]), min=-1.194702, max=1.000000, mean=0.153651
  features.4.3.block.2.fc2: coverage shape torch.Size([672]), min=-1.207883, max=1.000000, mean=0.033959
  features.4.3.block.3.0: coverage shape torch.Size([112]), min=-0.883339, max=1.000000, mean=0.012968
  features.4.4.block.0.0: coverage shape torch.Size([672]), min=-1.296990, max=1.000000, mean=-0.223133
  features.4.4.block.1.0: coverage shape torch.Size([672]), min=-1.493672, max=1.000000, mean=-0.030342
  features.4.4.block.2.fc1: coverage shape torch.Size([28]), min=-1.428248, max=1.000000, mean=0.000111
  features.4.4.block.2.fc2: coverage shape torch.Size([672]), min=-1.366426, max=1.000000, mean=0.018761
  features.4.4.block.3.0: coverage shape torch.Size([112]), min=-0.937275, max=1.000000, mean=0.036137
  features.4.5.block.0.0: coverage shape torch.Size([672]), min=-1.343356, max=1.000000, mean=-0.240308
  features.4.5.block.1.0: coverage shape torch.Size([672]), min=-0.824092, max=1.000000, mean=-0.011461
  features.4.5.block.2.fc1: coverage shape torch.Size([28]), min=-0.986120, max=1.000000, mean=0.106742
  features.4.5.block.2.fc2: coverage shape torch.Size([672]), min=-1.584063, max=1.000000, mean=-0.035353
  features.4.5.block.3.0: coverage shape torch.Size([112]), min=-0.589859, max=1.000000, mean=0.024118
  features.5.0.block.0.0: coverage shape torch.Size([672]), min=-1.815845, max=1.000000, mean=-0.242697
  features.5.0.block.1.0: coverage shape torch.Size([672]), min=-12.401925, max=1.000000, mean=-0.106766
  features.5.0.block.2.fc1: coverage shape torch.Size([28]), min=-29.670982, max=1.000000, mean=-6.985915
  features.5.0.block.2.fc2: coverage shape torch.Size([672]), min=-0.129877, max=1.000000, mean=0.574895
  features.5.0.block.3.0: coverage shape torch.Size([160]), min=-0.769190, max=1.000000, mean=0.019513
  features.5.1.block.0.0: coverage shape torch.Size([960]), min=-0.617489, max=1.000000, mean=-0.067459
  features.5.1.block.1.0: coverage shape torch.Size([960]), min=-3.228250, max=1.000000, mean=-0.027722
  features.5.1.block.2.fc1: coverage shape torch.Size([40]), min=-0.635393, max=1.000000, mean=0.193263
  features.5.1.block.2.fc2: coverage shape torch.Size([960]), min=-1.124043, max=1.000000, mean=0.015389
  features.5.1.block.3.0: coverage shape torch.Size([160]), min=-1.326142, max=1.000000, mean=0.014626
  features.5.2.block.0.0: coverage shape torch.Size([960]), min=-1.239736, max=1.000000, mean=-0.294037
  features.5.2.block.1.0: coverage shape torch.Size([960]), min=-1.286732, max=1.000000, mean=-0.015357
  features.5.2.block.2.fc1: coverage shape torch.Size([40]), min=-1.143112, max=1.000000, mean=0.102549
  features.5.2.block.2.fc2: coverage shape torch.Size([960]), min=-1.592070, max=1.000000, mean=-0.011988
  features.5.2.block.3.0: coverage shape torch.Size([160]), min=-1.245801, max=1.000000, mean=-0.035745
  features.5.3.block.0.0: coverage shape torch.Size([960]), min=-1.500080, max=1.000000, mean=-0.327488
  features.5.3.block.1.0: coverage shape torch.Size([960]), min=-1.822343, max=1.000000, mean=-0.013802
  features.5.3.block.2.fc1: coverage shape torch.Size([40]), min=-2.071081, max=1.000000, mean=-0.145196
  features.5.3.block.2.fc2: coverage shape torch.Size([960]), min=-2.292564, max=1.000000, mean=0.101168
  features.5.3.block.3.0: coverage shape torch.Size([160]), min=-1.169598, max=1.000000, mean=-0.003924
  features.5.4.block.0.0: coverage shape torch.Size([960]), min=-1.682904, max=1.000000, mean=-0.432897
  features.5.4.block.1.0: coverage shape torch.Size([960]), min=-2.612020, max=1.000000, mean=-0.015485
  features.5.4.block.2.fc1: coverage shape torch.Size([40]), min=-2.272704, max=1.000000, mean=0.023310
  features.5.4.block.2.fc2: coverage shape torch.Size([960]), min=-1.978283, max=1.000000, mean=0.097084
  features.5.4.block.3.0: coverage shape torch.Size([160]), min=-0.864234, max=1.000000, mean=0.027439
  features.5.5.block.0.0: coverage shape torch.Size([960]), min=-0.781913, max=1.000000, mean=-0.253383
  features.5.5.block.1.0: coverage shape torch.Size([960]), min=-1.557560, max=1.000000, mean=-0.012903
  features.5.5.block.2.fc1: coverage shape torch.Size([40]), min=-0.552038, max=1.000000, mean=0.237740
  features.5.5.block.2.fc2: coverage shape torch.Size([960]), min=-1.869839, max=1.000000, mean=0.059842
  features.5.5.block.3.0: coverage shape torch.Size([160]), min=-0.991093, max=1.000000, mean=-0.005094
  features.6.0.block.0.0: coverage shape torch.Size([960]), min=-1.417207, max=1.000000, mean=-0.383867
  features.6.0.block.1.0: coverage shape torch.Size([960]), min=-51.137794, max=1.000000, mean=-0.611389
  features.6.0.block.2.fc1: coverage shape torch.Size([40]), min=-723.093262, max=1.000000, mean=-113.665077
  features.6.0.block.2.fc2: coverage shape torch.Size([960]), min=0.247109, max=1.000000, mean=0.476111
  features.6.0.block.3.0: coverage shape torch.Size([272]), min=-1.005122, max=1.000000, mean=-0.004120
  features.6.1.block.0.0: coverage shape torch.Size([1632]), min=-1.161900, max=1.000000, mean=-0.133986
  features.6.1.block.1.0: coverage shape torch.Size([1632]), min=-3.187070, max=1.000000, mean=-0.019438
  features.6.1.block.2.fc1: coverage shape torch.Size([68]), min=-1.553048, max=1.000000, mean=-0.269945
  features.6.1.block.2.fc2: coverage shape torch.Size([1632]), min=-1.037435, max=1.000000, mean=-0.000561
  features.6.1.block.3.0: coverage shape torch.Size([272]), min=-1.368157, max=1.000000, mean=0.010058
  features.6.2.block.0.0: coverage shape torch.Size([1632]), min=-1.138314, max=1.000000, mean=-0.249873
  features.6.2.block.1.0: coverage shape torch.Size([1632]), min=-3.084178, max=1.000000, mean=-0.058497
  features.6.2.block.2.fc1: coverage shape torch.Size([68]), min=-0.588970, max=1.000000, mean=-0.031584
  features.6.2.block.2.fc2: coverage shape torch.Size([1632]), min=-1.504316, max=1.000000, mean=-0.078002
  features.6.2.block.3.0: coverage shape torch.Size([272]), min=-1.620811, max=1.000000, mean=-0.018574
  features.6.3.block.0.0: coverage shape torch.Size([1632]), min=-1.415623, max=1.000000, mean=-0.337478
  features.6.3.block.1.0: coverage shape torch.Size([1632]), min=-2.778405, max=1.000000, mean=-0.039931
  features.6.3.block.2.fc1: coverage shape torch.Size([68]), min=-0.841016, max=1.000000, mean=-0.054902
  features.6.3.block.2.fc2: coverage shape torch.Size([1632]), min=-1.817256, max=1.000000, mean=-0.128424
  features.6.3.block.3.0: coverage shape torch.Size([272]), min=-0.727282, max=1.000000, mean=-0.003179
  features.6.4.block.0.0: coverage shape torch.Size([1632]), min=-1.283647, max=1.000000, mean=-0.418126
  features.6.4.block.1.0: coverage shape torch.Size([1632]), min=-1.004158, max=1.000000, mean=-0.024139
  features.6.4.block.2.fc1: coverage shape torch.Size([68]), min=-1.176407, max=1.000000, mean=-0.102308
  features.6.4.block.2.fc2: coverage shape torch.Size([1632]), min=-1.221959, max=1.000000, mean=-0.103348
  features.6.4.block.3.0: coverage shape torch.Size([272]), min=-0.694859, max=1.000000, mean=0.013524
  features.6.5.block.0.0: coverage shape torch.Size([1632]), min=-1.782088, max=1.000000, mean=-0.638692
  features.6.5.block.1.0: coverage shape torch.Size([1632]), min=-1.419583, max=1.000000, mean=-0.048225
  features.6.5.block.2.fc1: coverage shape torch.Size([68]), min=-1.438556, max=1.000000, mean=-0.086664
  features.6.5.block.2.fc2: coverage shape torch.Size([1632]), min=-0.869955, max=1.000000, mean=-0.077105
  features.6.5.block.3.0: coverage shape torch.Size([272]), min=-0.943321, max=1.000000, mean=0.024546
  features.6.6.block.0.0: coverage shape torch.Size([1632]), min=-2.170594, max=1.000000, mean=-0.835879
  features.6.6.block.1.0: coverage shape torch.Size([1632]), min=-0.779060, max=1.000000, mean=-0.037098
  features.6.6.block.2.fc1: coverage shape torch.Size([68]), min=-1.101979, max=1.000000, mean=-0.079937
  features.6.6.block.2.fc2: coverage shape torch.Size([1632]), min=-1.153723, max=1.000000, mean=-0.116876
  features.6.6.block.3.0: coverage shape torch.Size([272]), min=-1.232157, max=1.000000, mean=-0.014302
  features.6.7.block.0.0: coverage shape torch.Size([1632]), min=-0.834226, max=1.000000, mean=-0.300899
  features.6.7.block.1.0: coverage shape torch.Size([1632]), min=-2.669134, max=1.000000, mean=-0.110942
  features.6.7.block.2.fc1: coverage shape torch.Size([68]), min=-1.974272, max=1.000000, mean=-0.260768
  features.6.7.block.2.fc2: coverage shape torch.Size([1632]), min=-1.200322, max=1.000000, mean=-0.082640
  features.6.7.block.3.0: coverage shape torch.Size([272]), min=-0.931827, max=1.000000, mean=0.024596
  features.7.0.block.0.0: coverage shape torch.Size([1632]), min=-1.919885, max=1.000000, mean=-0.323010
  features.7.0.block.1.0: coverage shape torch.Size([1632]), min=-5.714458, max=1.000000, mean=-0.089693
  features.7.0.block.2.fc1: coverage shape torch.Size([68]), min=-0.707628, max=1.000000, mean=-0.239016
  features.7.0.block.2.fc2: coverage shape torch.Size([1632]), min=-0.554624, max=1.000000, mean=0.161512
  features.7.0.block.3.0: coverage shape torch.Size([448]), min=-1.525344, max=1.000000, mean=-0.007673
  features.7.1.block.0.0: coverage shape torch.Size([2688]), min=-0.883063, max=1.000000, mean=-0.090375
  features.7.1.block.1.0: coverage shape torch.Size([2688]), min=-0.394311, max=1.000000, mean=-0.010232
  features.7.1.block.2.fc1: coverage shape torch.Size([112]), min=-0.959526, max=1.000000, mean=-0.299260
  features.7.1.block.2.fc2: coverage shape torch.Size([2688]), min=-1.222285, max=1.000000, mean=-0.075400
  features.7.1.block.3.0: coverage shape torch.Size([448]), min=-1.497327, max=1.000000, mean=-0.029387
  features.8.0: coverage shape torch.Size([1792]), min=-14.177032, max=-0.933330, mean=-7.240156
  classifier.1: coverage shape torch.Size([101]), min=-14.330275, max=-9.320189, mean=-11.440582

------------------------------------------------------------
Coverage Statistics:
------------------------------------------------------------

features.0.0:
  Channels: 48
  Coverage - Min: -1.333773, Max: 1.000000, Mean: -0.021201
  Zero coverage neurons: 0

features.1.0.block.0.0:
  Channels: 48
  Coverage - Min: -0.446795, Max: 1.000000, Mean: 0.001776
  Zero coverage neurons: 0

features.1.0.block.1.fc1:
  Channels: 12
  Coverage - Min: -0.641884, Max: 1.000000, Mean: 0.222474
  Zero coverage neurons: 0

features.1.0.block.1.fc2:
  Channels: 48
  Coverage - Min: -0.062799, Max: 1.000000, Mean: 0.439118
  Zero coverage neurons: 0

features.1.0.block.2.0:
  Channels: 24
  Coverage - Min: -1.055990, Max: 1.000000, Mean: 0.178298
  Zero coverage neurons: 0

features.1.1.block.0.0:
  Channels: 24
  Coverage - Min: -0.565429, Max: 1.000000, Mean: 0.087814
  Zero coverage neurons: 0

features.1.1.block.1.fc1:
  Channels: 6
  Coverage - Min: -15.756065, Max: -2.489717, Mean: -9.867497
  Zero coverage neurons: 0

features.1.1.block.1.fc2:
  Channels: 24
  Coverage - Min: -0.336395, Max: 1.000000, Mean: 0.620935
  Zero coverage neurons: 0

features.1.1.block.2.0:
  Channels: 24
  Coverage - Min: -3.121837, Max: 1.000000, Mean: -0.463210
  Zero coverage neurons: 0

features.2.0.block.0.0:
  Channels: 144
  Coverage - Min: -0.833052, Max: 1.000000, Mean: 0.014702
  Zero coverage neurons: 0

features.2.0.block.1.0:
  Channels: 144
  Coverage - Min: -1.101933, Max: 1.000000, Mean: 0.050424
  Zero coverage neurons: 0

features.2.0.block.2.fc1:
  Channels: 6
  Coverage - Min: -7.047021, Max: -1.442365, Mean: -3.522002
  Zero coverage neurons: 0

features.2.0.block.2.fc2:
  Channels: 144
  Coverage - Min: 0.022909, Max: 1.000000, Mean: 0.567718
  Zero coverage neurons: 0

features.2.0.block.3.0:
  Channels: 32
  Coverage - Min: -1.113953, Max: 1.000000, Mean: 0.189860
  Zero coverage neurons: 0

features.2.1.block.0.0:
  Channels: 192
  Coverage - Min: -1.644312, Max: 1.000000, Mean: -0.086887
  Zero coverage neurons: 0

features.2.1.block.1.0:
  Channels: 192
  Coverage - Min: -1.180760, Max: 1.000000, Mean: -0.036302
  Zero coverage neurons: 0

features.2.1.block.2.fc1:
  Channels: 8
  Coverage - Min: -0.121285, Max: 1.000000, Mean: 0.112836
  Zero coverage neurons: 0

features.2.1.block.2.fc2:
  Channels: 192
  Coverage - Min: -1.334660, Max: 1.000000, Mean: 0.152818
  Zero coverage neurons: 0

features.2.1.block.3.0:
  Channels: 32
  Coverage - Min: -1.357444, Max: 1.000000, Mean: -0.128920
  Zero coverage neurons: 0

features.2.2.block.0.0:
  Channels: 192
  Coverage - Min: -1.173677, Max: 1.000000, Mean: -0.104037
  Zero coverage neurons: 0

features.2.2.block.1.0:
  Channels: 192
  Coverage - Min: -1.440624, Max: 1.000000, Mean: -0.030350
  Zero coverage neurons: 0

features.2.2.block.2.fc1:
  Channels: 8
  Coverage - Min: -28.234404, Max: -0.495475, Mean: -5.044511
  Zero coverage neurons: 0

features.2.2.block.2.fc2:
  Channels: 192
  Coverage - Min: -0.889934, Max: 1.000000, Mean: 0.241851
  Zero coverage neurons: 0

features.2.2.block.3.0:
  Channels: 32
  Coverage - Min: -1.024326, Max: 1.000000, Mean: 0.081148
  Zero coverage neurons: 0

features.2.3.block.0.0:
  Channels: 192
  Coverage - Min: -1.267619, Max: 1.000000, Mean: -0.162726
  Zero coverage neurons: 0

features.2.3.block.1.0:
  Channels: 192
  Coverage - Min: -1.382231, Max: 1.000000, Mean: -0.033876
  Zero coverage neurons: 0

features.2.3.block.2.fc1:
  Channels: 8
  Coverage - Min: -2.542885, Max: -2.217818, Mean: -2.406159
  Zero coverage neurons: 0

features.2.3.block.2.fc2:
  Channels: 192
  Coverage - Min: -0.572115, Max: 1.000000, Mean: 0.294209
  Zero coverage neurons: 0

features.2.3.block.3.0:
  Channels: 32
  Coverage - Min: -0.839403, Max: 1.000000, Mean: 0.014359
  Zero coverage neurons: 0

features.3.0.block.0.0:
  Channels: 192
  Coverage - Min: -1.398009, Max: 1.000000, Mean: -0.253850
  Zero coverage neurons: 0

features.3.0.block.1.0:
  Channels: 192
  Coverage - Min: -0.841059, Max: 1.000000, Mean: -0.027589
  Zero coverage neurons: 0

features.3.0.block.2.fc1:
  Channels: 8
  Coverage - Min: -0.258176, Max: 1.000000, Mean: 0.221531
  Zero coverage neurons: 0

features.3.0.block.2.fc2:
  Channels: 192
  Coverage - Min: -0.523610, Max: 1.000000, Mean: 0.289668
  Zero coverage neurons: 0

features.3.0.block.3.0:
  Channels: 56
  Coverage - Min: -0.779956, Max: 1.000000, Mean: 0.015043
  Zero coverage neurons: 0

features.3.1.block.0.0:
  Channels: 336
  Coverage - Min: -0.901222, Max: 1.000000, Mean: 0.009629
  Zero coverage neurons: 0

features.3.1.block.1.0:
  Channels: 336
  Coverage - Min: -3.666774, Max: 1.000000, Mean: -0.120686
  Zero coverage neurons: 0

features.3.1.block.2.fc1:
  Channels: 14
  Coverage - Min: -0.024983, Max: 1.000000, Mean: 0.382133
  Zero coverage neurons: 0

features.3.1.block.2.fc2:
  Channels: 336
  Coverage - Min: -1.509517, Max: 1.000000, Mean: 0.157532
  Zero coverage neurons: 0

features.3.1.block.3.0:
  Channels: 56
  Coverage - Min: -1.029634, Max: 1.000000, Mean: 0.033186
  Zero coverage neurons: 0

features.3.2.block.0.0:
  Channels: 336
  Coverage - Min: -2.007234, Max: 1.000000, Mean: -0.217789
  Zero coverage neurons: 0

features.3.2.block.1.0:
  Channels: 336
  Coverage - Min: -1.934926, Max: 1.000000, Mean: -0.071799
  Zero coverage neurons: 0

features.3.2.block.2.fc1:
  Channels: 14
  Coverage - Min: -0.304844, Max: 1.000000, Mean: 0.111786
  Zero coverage neurons: 0

features.3.2.block.2.fc2:
  Channels: 336
  Coverage - Min: -2.987664, Max: 1.000000, Mean: 0.125695
  Zero coverage neurons: 0

features.3.2.block.3.0:
  Channels: 56
  Coverage - Min: -0.767084, Max: 1.000000, Mean: 0.027163
  Zero coverage neurons: 0

features.3.3.block.0.0:
  Channels: 336
  Coverage - Min: -0.996078, Max: 1.000000, Mean: -0.146168
  Zero coverage neurons: 0

features.3.3.block.1.0:
  Channels: 336
  Coverage - Min: -1.741781, Max: 1.000000, Mean: -0.063261
  Zero coverage neurons: 0

features.3.3.block.2.fc1:
  Channels: 14
  Coverage - Min: -0.629044, Max: 1.000000, Mean: -0.006891
  Zero coverage neurons: 0

features.3.3.block.2.fc2:
  Channels: 336
  Coverage - Min: -0.695708, Max: 1.000000, Mean: 0.068257
  Zero coverage neurons: 0

features.3.3.block.3.0:
  Channels: 56
  Coverage - Min: -0.734313, Max: 1.000000, Mean: -0.023183
  Zero coverage neurons: 0

features.4.0.block.0.0:
  Channels: 336
  Coverage - Min: -0.751614, Max: 1.000000, Mean: -0.159840
  Zero coverage neurons: 0

features.4.0.block.1.0:
  Channels: 336
  Coverage - Min: -4.024682, Max: 1.000000, Mean: -0.188975
  Zero coverage neurons: 0

features.4.0.block.2.fc1:
  Channels: 14
  Coverage - Min: -0.951480, Max: 1.000000, Mean: -0.156340
  Zero coverage neurons: 0

features.4.0.block.2.fc2:
  Channels: 336
  Coverage - Min: -0.300682, Max: 1.000000, Mean: 0.424969
  Zero coverage neurons: 0

features.4.0.block.3.0:
  Channels: 112
  Coverage - Min: -0.998053, Max: 1.000000, Mean: -0.018512
  Zero coverage neurons: 0

features.4.1.block.0.0:
  Channels: 672
  Coverage - Min: -0.921940, Max: 1.000000, Mean: -0.001201
  Zero coverage neurons: 0

features.4.1.block.1.0:
  Channels: 672
  Coverage - Min: -2.885710, Max: 1.000000, Mean: -0.044020
  Zero coverage neurons: 0

features.4.1.block.2.fc1:
  Channels: 28
  Coverage - Min: -0.785222, Max: 1.000000, Mean: 0.100551
  Zero coverage neurons: 0

features.4.1.block.2.fc2:
  Channels: 672
  Coverage - Min: -1.284496, Max: 1.000000, Mean: 0.021283
  Zero coverage neurons: 0

features.4.1.block.3.0:
  Channels: 112
  Coverage - Min: -0.689937, Max: 1.000000, Mean: 0.029136
  Zero coverage neurons: 0

features.4.2.block.0.0:
  Channels: 672
  Coverage - Min: -0.481996, Max: 1.000000, Mean: -0.057298
  Zero coverage neurons: 0

features.4.2.block.1.0:
  Channels: 672
  Coverage - Min: -1.457130, Max: 1.000000, Mean: -0.005181
  Zero coverage neurons: 0

features.4.2.block.2.fc1:
  Channels: 28
  Coverage - Min: -1.032089, Max: 1.000000, Mean: 0.233805
  Zero coverage neurons: 0

features.4.2.block.2.fc2:
  Channels: 672
  Coverage - Min: -2.086202, Max: 1.000000, Mean: -0.035682
  Zero coverage neurons: 0

features.4.2.block.3.0:
  Channels: 112
  Coverage - Min: -0.811981, Max: 1.000000, Mean: 0.029871
  Zero coverage neurons: 0

features.4.3.block.0.0:
  Channels: 672
  Coverage - Min: -1.564362, Max: 1.000000, Mean: -0.151875
  Zero coverage neurons: 0

features.4.3.block.1.0:
  Channels: 672
  Coverage - Min: -2.400361, Max: 1.000000, Mean: -0.029290
  Zero coverage neurons: 0

features.4.3.block.2.fc1:
  Channels: 28
  Coverage - Min: -1.194702, Max: 1.000000, Mean: 0.153651
  Zero coverage neurons: 0

features.4.3.block.2.fc2:
  Channels: 672
  Coverage - Min: -1.207883, Max: 1.000000, Mean: 0.033959
  Zero coverage neurons: 0

features.4.3.block.3.0:
  Channels: 112
  Coverage - Min: -0.883339, Max: 1.000000, Mean: 0.012968
  Zero coverage neurons: 0

features.4.4.block.0.0:
  Channels: 672
  Coverage - Min: -1.296990, Max: 1.000000, Mean: -0.223133
  Zero coverage neurons: 0

features.4.4.block.1.0:
  Channels: 672
  Coverage - Min: -1.493672, Max: 1.000000, Mean: -0.030342
  Zero coverage neurons: 0

features.4.4.block.2.fc1:
  Channels: 28
  Coverage - Min: -1.428248, Max: 1.000000, Mean: 0.000111
  Zero coverage neurons: 0

features.4.4.block.2.fc2:
  Channels: 672
  Coverage - Min: -1.366426, Max: 1.000000, Mean: 0.018761
  Zero coverage neurons: 0

features.4.4.block.3.0:
  Channels: 112
  Coverage - Min: -0.937275, Max: 1.000000, Mean: 0.036137
  Zero coverage neurons: 0

features.4.5.block.0.0:
  Channels: 672
  Coverage - Min: -1.343356, Max: 1.000000, Mean: -0.240308
  Zero coverage neurons: 0

features.4.5.block.1.0:
  Channels: 672
  Coverage - Min: -0.824092, Max: 1.000000, Mean: -0.011461
  Zero coverage neurons: 0

features.4.5.block.2.fc1:
  Channels: 28
  Coverage - Min: -0.986120, Max: 1.000000, Mean: 0.106742
  Zero coverage neurons: 0

features.4.5.block.2.fc2:
  Channels: 672
  Coverage - Min: -1.584063, Max: 1.000000, Mean: -0.035353
  Zero coverage neurons: 0

features.4.5.block.3.0:
  Channels: 112
  Coverage - Min: -0.589859, Max: 1.000000, Mean: 0.024118
  Zero coverage neurons: 0

features.5.0.block.0.0:
  Channels: 672
  Coverage - Min: -1.815845, Max: 1.000000, Mean: -0.242697
  Zero coverage neurons: 0

features.5.0.block.1.0:
  Channels: 672
  Coverage - Min: -12.401925, Max: 1.000000, Mean: -0.106766
  Zero coverage neurons: 0

features.5.0.block.2.fc1:
  Channels: 28
  Coverage - Min: -29.670982, Max: 1.000000, Mean: -6.985915
  Zero coverage neurons: 0

features.5.0.block.2.fc2:
  Channels: 672
  Coverage - Min: -0.129877, Max: 1.000000, Mean: 0.574895
  Zero coverage neurons: 0

features.5.0.block.3.0:
  Channels: 160
  Coverage - Min: -0.769190, Max: 1.000000, Mean: 0.019513
  Zero coverage neurons: 0

features.5.1.block.0.0:
  Channels: 960
  Coverage - Min: -0.617489, Max: 1.000000, Mean: -0.067459
  Zero coverage neurons: 0

features.5.1.block.1.0:
  Channels: 960
  Coverage - Min: -3.228250, Max: 1.000000, Mean: -0.027722
  Zero coverage neurons: 0

features.5.1.block.2.fc1:
  Channels: 40
  Coverage - Min: -0.635393, Max: 1.000000, Mean: 0.193263
  Zero coverage neurons: 0

features.5.1.block.2.fc2:
  Channels: 960
  Coverage - Min: -1.124043, Max: 1.000000, Mean: 0.015389
  Zero coverage neurons: 0

features.5.1.block.3.0:
  Channels: 160
  Coverage - Min: -1.326142, Max: 1.000000, Mean: 0.014626
  Zero coverage neurons: 0

features.5.2.block.0.0:
  Channels: 960
  Coverage - Min: -1.239736, Max: 1.000000, Mean: -0.294037
  Zero coverage neurons: 0

features.5.2.block.1.0:
  Channels: 960
  Coverage - Min: -1.286732, Max: 1.000000, Mean: -0.015357
  Zero coverage neurons: 0

features.5.2.block.2.fc1:
  Channels: 40
  Coverage - Min: -1.143112, Max: 1.000000, Mean: 0.102549
  Zero coverage neurons: 0

features.5.2.block.2.fc2:
  Channels: 960
  Coverage - Min: -1.592070, Max: 1.000000, Mean: -0.011988
  Zero coverage neurons: 0

features.5.2.block.3.0:
  Channels: 160
  Coverage - Min: -1.245801, Max: 1.000000, Mean: -0.035745
  Zero coverage neurons: 0

features.5.3.block.0.0:
  Channels: 960
  Coverage - Min: -1.500080, Max: 1.000000, Mean: -0.327488
  Zero coverage neurons: 0

features.5.3.block.1.0:
  Channels: 960
  Coverage - Min: -1.822343, Max: 1.000000, Mean: -0.013802
  Zero coverage neurons: 0

features.5.3.block.2.fc1:
  Channels: 40
  Coverage - Min: -2.071081, Max: 1.000000, Mean: -0.145196
  Zero coverage neurons: 0

features.5.3.block.2.fc2:
  Channels: 960
  Coverage - Min: -2.292564, Max: 1.000000, Mean: 0.101168
  Zero coverage neurons: 0

features.5.3.block.3.0:
  Channels: 160
  Coverage - Min: -1.169598, Max: 1.000000, Mean: -0.003924
  Zero coverage neurons: 0

features.5.4.block.0.0:
  Channels: 960
  Coverage - Min: -1.682904, Max: 1.000000, Mean: -0.432897
  Zero coverage neurons: 0

features.5.4.block.1.0:
  Channels: 960
  Coverage - Min: -2.612020, Max: 1.000000, Mean: -0.015485
  Zero coverage neurons: 0

features.5.4.block.2.fc1:
  Channels: 40
  Coverage - Min: -2.272704, Max: 1.000000, Mean: 0.023310
  Zero coverage neurons: 0

features.5.4.block.2.fc2:
  Channels: 960
  Coverage - Min: -1.978283, Max: 1.000000, Mean: 0.097084
  Zero coverage neurons: 0

features.5.4.block.3.0:
  Channels: 160
  Coverage - Min: -0.864234, Max: 1.000000, Mean: 0.027439
  Zero coverage neurons: 0

features.5.5.block.0.0:
  Channels: 960
  Coverage - Min: -0.781913, Max: 1.000000, Mean: -0.253383
  Zero coverage neurons: 0

features.5.5.block.1.0:
  Channels: 960
  Coverage - Min: -1.557560, Max: 1.000000, Mean: -0.012903
  Zero coverage neurons: 0

features.5.5.block.2.fc1:
  Channels: 40
  Coverage - Min: -0.552038, Max: 1.000000, Mean: 0.237740
  Zero coverage neurons: 0

features.5.5.block.2.fc2:
  Channels: 960
  Coverage - Min: -1.869839, Max: 1.000000, Mean: 0.059842
  Zero coverage neurons: 0

features.5.5.block.3.0:
  Channels: 160
  Coverage - Min: -0.991093, Max: 1.000000, Mean: -0.005094
  Zero coverage neurons: 0

features.6.0.block.0.0:
  Channels: 960
  Coverage - Min: -1.417207, Max: 1.000000, Mean: -0.383867
  Zero coverage neurons: 0

features.6.0.block.1.0:
  Channels: 960
  Coverage - Min: -51.137794, Max: 1.000000, Mean: -0.611389
  Zero coverage neurons: 0

features.6.0.block.2.fc1:
  Channels: 40
  Coverage - Min: -723.093262, Max: 1.000000, Mean: -113.665077
  Zero coverage neurons: 0

features.6.0.block.2.fc2:
  Channels: 960
  Coverage - Min: 0.247109, Max: 1.000000, Mean: 0.476111
  Zero coverage neurons: 0

features.6.0.block.3.0:
  Channels: 272
  Coverage - Min: -1.005122, Max: 1.000000, Mean: -0.004120
  Zero coverage neurons: 0

features.6.1.block.0.0:
  Channels: 1632
  Coverage - Min: -1.161900, Max: 1.000000, Mean: -0.133986
  Zero coverage neurons: 0

features.6.1.block.1.0:
  Channels: 1632
  Coverage - Min: -3.187070, Max: 1.000000, Mean: -0.019438
  Zero coverage neurons: 0

features.6.1.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.553048, Max: 1.000000, Mean: -0.269945
  Zero coverage neurons: 0

features.6.1.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.037435, Max: 1.000000, Mean: -0.000561
  Zero coverage neurons: 0

features.6.1.block.3.0:
  Channels: 272
  Coverage - Min: -1.368157, Max: 1.000000, Mean: 0.010058
  Zero coverage neurons: 0

features.6.2.block.0.0:
  Channels: 1632
  Coverage - Min: -1.138314, Max: 1.000000, Mean: -0.249873
  Zero coverage neurons: 0

features.6.2.block.1.0:
  Channels: 1632
  Coverage - Min: -3.084178, Max: 1.000000, Mean: -0.058497
  Zero coverage neurons: 0

features.6.2.block.2.fc1:
  Channels: 68
  Coverage - Min: -0.588970, Max: 1.000000, Mean: -0.031584
  Zero coverage neurons: 0

features.6.2.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.504316, Max: 1.000000, Mean: -0.078002
  Zero coverage neurons: 0

features.6.2.block.3.0:
  Channels: 272
  Coverage - Min: -1.620811, Max: 1.000000, Mean: -0.018574
  Zero coverage neurons: 0

features.6.3.block.0.0:
  Channels: 1632
  Coverage - Min: -1.415623, Max: 1.000000, Mean: -0.337478
  Zero coverage neurons: 0

features.6.3.block.1.0:
  Channels: 1632
  Coverage - Min: -2.778405, Max: 1.000000, Mean: -0.039931
  Zero coverage neurons: 0

features.6.3.block.2.fc1:
  Channels: 68
  Coverage - Min: -0.841016, Max: 1.000000, Mean: -0.054902
  Zero coverage neurons: 0

features.6.3.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.817256, Max: 1.000000, Mean: -0.128424
  Zero coverage neurons: 0

features.6.3.block.3.0:
  Channels: 272
  Coverage - Min: -0.727282, Max: 1.000000, Mean: -0.003179
  Zero coverage neurons: 0

features.6.4.block.0.0:
  Channels: 1632
  Coverage - Min: -1.283647, Max: 1.000000, Mean: -0.418126
  Zero coverage neurons: 0

features.6.4.block.1.0:
  Channels: 1632
  Coverage - Min: -1.004158, Max: 1.000000, Mean: -0.024139
  Zero coverage neurons: 0

features.6.4.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.176407, Max: 1.000000, Mean: -0.102308
  Zero coverage neurons: 0

features.6.4.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.221959, Max: 1.000000, Mean: -0.103348
  Zero coverage neurons: 0

features.6.4.block.3.0:
  Channels: 272
  Coverage - Min: -0.694859, Max: 1.000000, Mean: 0.013524
  Zero coverage neurons: 0

features.6.5.block.0.0:
  Channels: 1632
  Coverage - Min: -1.782088, Max: 1.000000, Mean: -0.638692
  Zero coverage neurons: 0

features.6.5.block.1.0:
  Channels: 1632
  Coverage - Min: -1.419583, Max: 1.000000, Mean: -0.048225
  Zero coverage neurons: 0

features.6.5.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.438556, Max: 1.000000, Mean: -0.086664
  Zero coverage neurons: 0

features.6.5.block.2.fc2:
  Channels: 1632
  Coverage - Min: -0.869955, Max: 1.000000, Mean: -0.077105
  Zero coverage neurons: 0

features.6.5.block.3.0:
  Channels: 272
  Coverage - Min: -0.943321, Max: 1.000000, Mean: 0.024546
  Zero coverage neurons: 0

features.6.6.block.0.0:
  Channels: 1632
  Coverage - Min: -2.170594, Max: 1.000000, Mean: -0.835879
  Zero coverage neurons: 0

features.6.6.block.1.0:
  Channels: 1632
  Coverage - Min: -0.779060, Max: 1.000000, Mean: -0.037098
  Zero coverage neurons: 0

features.6.6.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.101979, Max: 1.000000, Mean: -0.079937
  Zero coverage neurons: 0

features.6.6.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.153723, Max: 1.000000, Mean: -0.116876
  Zero coverage neurons: 0

features.6.6.block.3.0:
  Channels: 272
  Coverage - Min: -1.232157, Max: 1.000000, Mean: -0.014302
  Zero coverage neurons: 0

features.6.7.block.0.0:
  Channels: 1632
  Coverage - Min: -0.834226, Max: 1.000000, Mean: -0.300899
  Zero coverage neurons: 0

features.6.7.block.1.0:
  Channels: 1632
  Coverage - Min: -2.669134, Max: 1.000000, Mean: -0.110942
  Zero coverage neurons: 0

features.6.7.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.974272, Max: 1.000000, Mean: -0.260768
  Zero coverage neurons: 0

features.6.7.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.200322, Max: 1.000000, Mean: -0.082640
  Zero coverage neurons: 0

features.6.7.block.3.0:
  Channels: 272
  Coverage - Min: -0.931827, Max: 1.000000, Mean: 0.024596
  Zero coverage neurons: 0

features.7.0.block.0.0:
  Channels: 1632
  Coverage - Min: -1.919885, Max: 1.000000, Mean: -0.323010
  Zero coverage neurons: 0

features.7.0.block.1.0:
  Channels: 1632
  Coverage - Min: -5.714458, Max: 1.000000, Mean: -0.089693
  Zero coverage neurons: 0

features.7.0.block.2.fc1:
  Channels: 68
  Coverage - Min: -0.707628, Max: 1.000000, Mean: -0.239016
  Zero coverage neurons: 0

features.7.0.block.2.fc2:
  Channels: 1632
  Coverage - Min: -0.554624, Max: 1.000000, Mean: 0.161512
  Zero coverage neurons: 0

features.7.0.block.3.0:
  Channels: 448
  Coverage - Min: -1.525344, Max: 1.000000, Mean: -0.007673
  Zero coverage neurons: 0

features.7.1.block.0.0:
  Channels: 2688
  Coverage - Min: -0.883063, Max: 1.000000, Mean: -0.090375
  Zero coverage neurons: 0

features.7.1.block.1.0:
  Channels: 2688
  Coverage - Min: -0.394311, Max: 1.000000, Mean: -0.010232
  Zero coverage neurons: 0

features.7.1.block.2.fc1:
  Channels: 112
  Coverage - Min: -0.959526, Max: 1.000000, Mean: -0.299260
  Zero coverage neurons: 0

features.7.1.block.2.fc2:
  Channels: 2688
  Coverage - Min: -1.222285, Max: 1.000000, Mean: -0.075400
  Zero coverage neurons: 0

features.7.1.block.3.0:
  Channels: 448
  Coverage - Min: -1.497327, Max: 1.000000, Mean: -0.029387
  Zero coverage neurons: 0

features.8.0:
  Channels: 1792
  Coverage - Min: -14.177032, Max: -0.933330, Mean: -7.240156
  Zero coverage neurons: 0

classifier.1:
  Channels: 101
  Coverage - Min: -14.330275, Max: -9.320189, Mean: -11.440582
  Zero coverage neurons: 0
============================================================


Pruning Results:
  Parameters before: 17,729,709
  Parameters after:  14,395,049
  Parameters removed: 3,334,660 (18.81%)

############################################################
Pruning Complete
############################################################
Initial parameters:  17,729,709
Final parameters:    14,395,049
Total removed:       3,334,660 (18.81%)
Target pruning ratio: 10.00%
✓ Neuron Coverage pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 1.11%                                                                                                                  
✓ Model Size: 55.35 MB
✓ Average Inference Time: 5.1063 ms
✓ FLOPs: 3.76 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.10it/s, loss=2.1034, acc=48.62%] 
Epoch 1/10 - Train Loss: 2.1034, Train Acc: 48.62%, Test Acc: 76.95%                                                                            
Epoch 2/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.12it/s, loss=1.6988, acc=57.02%] 
Epoch 2/10 - Train Loss: 1.6988, Train Acc: 57.02%, Test Acc: 78.74%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_NC_epoch2.pth
Epoch 3/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.11it/s, loss=1.4792, acc=61.86%]
Epoch 3/10 - Train Loss: 1.4792, Train Acc: 61.86%, Test Acc: 80.26%                                                                            
Epoch 4/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.11it/s, loss=1.3303, acc=66.10%] 
Epoch 4/10 - Train Loss: 1.3303, Train Acc: 66.10%, Test Acc: 81.08%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_NC_epoch4.pth
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.11it/s, loss=1.2660, acc=67.22%]
Epoch 5/10 - Train Loss: 1.2660, Train Acc: 67.22%, Test Acc: 81.70%                                                                            
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.11it/s, loss=1.1576, acc=70.12%] 
Epoch 6/10 - Train Loss: 1.1576, Train Acc: 70.12%, Test Acc: 82.23%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_NC_epoch6.pth
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.10it/s, loss=1.1526, acc=69.48%]
Epoch 7/10 - Train Loss: 1.1526, Train Acc: 69.48%, Test Acc: 82.40%                                                                            
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.10it/s, loss=1.1050, acc=71.38%] 
Epoch 8/10 - Train Loss: 1.1050, Train Acc: 71.38%, Test Acc: 82.42%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_NC_epoch8.pth
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.11it/s, loss=1.0516, acc=71.76%]
Epoch 9/10 - Train Loss: 1.0516, Train Acc: 71.76%, Test Acc: 82.78%                                                                            
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.11it/s, loss=1.0733, acc=71.78%] 
Epoch 10/10 - Train Loss: 1.0733, Train Acc: 71.78%, Test Acc: 82.69%                                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_NC_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_NC_best.pth
  Best Accuracy: 82.78%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 82.69%                                                                                                                  
✓ Model Size: 55.35 MB
✓ Average Inference Time: 5.1142 ms
✓ FLOPs: 3.76 GFLOPs

================================================================================
COMPARISON: NEURON COVERAGE PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         89.21   |          1.11   |    82.69   | -6.51                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         68.11   |         55.35   |    55.35   | -12.76 (-18.7%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.5976 |          5.1063 |     5.1142 | +4.5166                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.61   |          3.76   |     3.76   | -0.85 (-18.5%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS5_EfficientNetB4_StanfordDogs\TS5_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================



**********************************



================================================================================
TEST SCENARIO TS5: WANDA PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FT_best.pth
  - Epoch: 9
  - Test Accuracy: 89.21%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 89.21%                                                                                                               
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.5898 ms
✓ FLOPs: 4.61 GFLOPs

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
  features.0.0: coverage shape torch.Size([48]), min=0.000010, max=9.543589, mean=1.175885
  features.1.0.block.0.0: coverage shape torch.Size([48]), min=0.000001, max=71.363930, mean=6.196229
  features.1.0.block.1.fc1: coverage shape torch.Size([12]), min=2.213008, max=18.588037, mean=6.480382
  features.1.0.block.1.fc2: coverage shape torch.Size([48]), min=0.013150, max=13.444448, mean=5.942733
  features.1.0.block.2.0: coverage shape torch.Size([24]), min=8.379252, max=202.707291, mean=71.898041
  features.1.1.block.0.0: coverage shape torch.Size([24]), min=0.006973, max=19.292110, mean=3.961229
  features.1.1.block.1.fc1: coverage shape torch.Size([6]), min=2.492083, max=15.853820, mean=9.894175
  features.1.1.block.1.fc2: coverage shape torch.Size([24]), min=1.527064, max=4.539252, mean=2.945816
  features.1.1.block.2.0: coverage shape torch.Size([24]), min=1.192892, max=103.464394, mean=23.983599
  features.2.0.block.0.0: coverage shape torch.Size([144]), min=0.049043, max=84.112755, mean=15.515657
  features.2.0.block.1.0: coverage shape torch.Size([144]), min=0.006556, max=69.707596, mean=14.040387
  features.2.0.block.2.fc1: coverage shape torch.Size([6]), min=1.430914, max=7.124955, mean=3.543283
  features.2.0.block.2.fc2: coverage shape torch.Size([144]), min=0.122095, max=5.332840, mean=3.027450
  features.2.0.block.3.0: coverage shape torch.Size([32]), min=7.144043, max=263.575562, mean=101.624069
  features.2.1.block.0.0: coverage shape torch.Size([192]), min=0.203832, max=143.015549, mean=21.001665
  features.2.1.block.1.0: coverage shape torch.Size([192]), min=0.000460, max=20.473652, mean=1.352044
  features.2.1.block.2.fc1: coverage shape torch.Size([8]), min=1.569479, max=40.032692, mean=7.479882
  features.2.1.block.2.fc2: coverage shape torch.Size([192]), min=0.341097, max=38.590317, mean=5.929050
  features.2.1.block.3.0: coverage shape torch.Size([32]), min=1.891819, max=128.478638, mean=44.913597
  features.2.2.block.0.0: coverage shape torch.Size([192]), min=0.000338, max=74.394455, mean=17.899221
  features.2.2.block.1.0: coverage shape torch.Size([192]), min=0.000003, max=23.985632, mean=1.776366
  features.2.2.block.2.fc1: coverage shape torch.Size([8]), min=0.493672, max=28.279497, mean=5.046848
  features.2.2.block.2.fc2: coverage shape torch.Size([192]), min=0.104489, max=12.220940, mean=3.290544
  features.2.2.block.3.0: coverage shape torch.Size([32]), min=3.079779, max=52.057362, mean=18.444427
  features.2.3.block.0.0: coverage shape torch.Size([192]), min=0.146540, max=98.717461, mean=24.183615
  features.2.3.block.1.0: coverage shape torch.Size([192]), min=0.000077, max=14.663739, mean=1.476071
  features.2.3.block.2.fc1: coverage shape torch.Size([8]), min=2.204707, max=2.541802, mean=2.395243
  features.2.3.block.2.fc2: coverage shape torch.Size([192]), min=0.102524, max=5.015934, mean=1.623722
  features.2.3.block.3.0: coverage shape torch.Size([32]), min=0.138702, max=35.775822, mean=12.902413
  features.3.0.block.0.0: coverage shape torch.Size([192]), min=0.097459, max=145.721451, mean=39.466152
  features.3.0.block.1.0: coverage shape torch.Size([192]), min=0.000097, max=63.454346, mean=4.252139
  features.3.0.block.2.fc1: coverage shape torch.Size([8]), min=5.450554, max=48.401188, mean=21.064997
  features.3.0.block.2.fc2: coverage shape torch.Size([192]), min=0.083362, max=20.301817, mean=6.127841
  features.3.0.block.3.0: coverage shape torch.Size([56]), min=0.858595, max=235.979156, mean=61.938747
  features.3.1.block.0.0: coverage shape torch.Size([336]), min=0.002978, max=58.641388, mean=7.696443
  features.3.1.block.1.0: coverage shape torch.Size([336]), min=0.000088, max=28.824862, mean=1.584996
  features.3.1.block.2.fc1: coverage shape torch.Size([14]), min=0.754659, max=38.841068, mean=15.442009
  features.3.1.block.2.fc2: coverage shape torch.Size([336]), min=0.262587, max=86.974152, mean=12.666800
  features.3.1.block.3.0: coverage shape torch.Size([56]), min=0.021071, max=27.853456, mean=10.873216
  features.3.2.block.0.0: coverage shape torch.Size([336]), min=0.024421, max=69.926529, mean=12.424342
  features.3.2.block.1.0: coverage shape torch.Size([336]), min=0.000671, max=22.855808, mean=1.533637
  features.3.2.block.2.fc1: coverage shape torch.Size([14]), min=0.058282, max=50.264351, mean=10.330870
  features.3.2.block.2.fc2: coverage shape torch.Size([336]), min=0.240110, max=115.111359, mean=8.058840
  features.3.2.block.3.0: coverage shape torch.Size([56]), min=0.584855, max=26.812168, mean=7.095654
  features.3.3.block.0.0: coverage shape torch.Size([336]), min=0.033525, max=77.479393, mean=19.298447
  features.3.3.block.1.0: coverage shape torch.Size([336]), min=0.000801, max=21.374039, mean=1.620977
  features.3.3.block.2.fc1: coverage shape torch.Size([14]), min=0.221091, max=32.047379, mean=10.421250
  features.3.3.block.2.fc2: coverage shape torch.Size([336]), min=0.049499, max=90.307510, mean=10.043493
  features.3.3.block.3.0: coverage shape torch.Size([56]), min=0.200774, max=11.219766, mean=3.778970
  features.4.0.block.0.0: coverage shape torch.Size([336]), min=0.221818, max=106.700623, mean=26.547802
  features.4.0.block.1.0: coverage shape torch.Size([336]), min=0.000190, max=44.366432, mean=2.410887
  features.4.0.block.2.fc1: coverage shape torch.Size([14]), min=2.077830, max=28.458878, mean=13.322555
  features.4.0.block.2.fc2: coverage shape torch.Size([336]), min=0.155690, max=7.399580, mean=3.156038
  features.4.0.block.3.0: coverage shape torch.Size([112]), min=0.020427, max=137.821381, mean=39.806252
  features.4.1.block.0.0: coverage shape torch.Size([672]), min=0.038166, max=73.658180, mean=8.215742
  features.4.1.block.1.0: coverage shape torch.Size([672]), min=0.000439, max=16.050774, mean=0.759308
  features.4.1.block.2.fc1: coverage shape torch.Size([28]), min=0.918786, max=39.550297, mean=13.305115
  features.4.1.block.2.fc2: coverage shape torch.Size([672]), min=0.003424, max=185.706268, mean=19.163393
  features.4.1.block.3.0: coverage shape torch.Size([112]), min=0.037049, max=10.160813, mean=2.513325
  features.4.2.block.0.0: coverage shape torch.Size([672]), min=0.022838, max=48.823978, mean=5.586002
  features.4.2.block.1.0: coverage shape torch.Size([672]), min=0.000038, max=9.240995, mean=0.882491
  features.4.2.block.2.fc1: coverage shape torch.Size([28]), min=0.018105, max=1.267408, mean=0.636035
  features.4.2.block.2.fc2: coverage shape torch.Size([672]), min=0.000641, max=6.236047, mean=1.281504
  features.4.2.block.3.0: coverage shape torch.Size([112]), min=0.009497, max=1.317272, mean=0.319354
  features.4.3.block.0.0: coverage shape torch.Size([672]), min=0.006142, max=43.337242, mean=7.698403
  features.4.3.block.1.0: coverage shape torch.Size([672]), min=0.000422, max=12.850874, mean=0.466811
  features.4.3.block.2.fc1: coverage shape torch.Size([28]), min=0.028621, max=35.604507, mean=10.090437
  features.4.3.block.2.fc2: coverage shape torch.Size([672]), min=0.082854, max=153.077332, mean=14.789365
  features.4.3.block.3.0: coverage shape torch.Size([112]), min=0.002659, max=5.313998, mean=1.900539
  features.4.4.block.0.0: coverage shape torch.Size([672]), min=0.016974, max=39.822266, mean=12.346816
  features.4.4.block.1.0: coverage shape torch.Size([672]), min=0.000167, max=7.346904, mean=0.647383
  features.4.4.block.2.fc1: coverage shape torch.Size([28]), min=0.627229, max=13.304763, mean=4.532600
  features.4.4.block.2.fc2: coverage shape torch.Size([672]), min=0.056739, max=28.782646, mean=5.723851
  features.4.4.block.3.0: coverage shape torch.Size([112]), min=0.003720, max=2.511139, mean=0.662294
  features.4.5.block.0.0: coverage shape torch.Size([672]), min=0.067938, max=54.779911, mean=14.574196
  features.4.5.block.1.0: coverage shape torch.Size([672]), min=0.000299, max=12.534796, mean=0.871697
  features.4.5.block.2.fc1: coverage shape torch.Size([28]), min=0.030882, max=3.372029, mean=1.321679
  features.4.5.block.2.fc2: coverage shape torch.Size([672]), min=0.001946, max=8.302926, mean=1.959165
  features.4.5.block.3.0: coverage shape torch.Size([112]), min=0.001601, max=1.397083, mean=0.318746
  features.5.0.block.0.0: coverage shape torch.Size([672]), min=0.006385, max=74.775764, mean=15.148971
  features.5.0.block.1.0: coverage shape torch.Size([672]), min=0.000172, max=139.789429, mean=1.691830
  features.5.0.block.2.fc1: coverage shape torch.Size([28]), min=0.439604, max=21.014698, mean=5.001423
  features.5.0.block.2.fc2: coverage shape torch.Size([672]), min=0.563461, max=4.354167, mean=2.505469
  features.5.0.block.3.0: coverage shape torch.Size([160]), min=0.220785, max=107.051712, mean=24.518240
  features.5.1.block.0.0: coverage shape torch.Size([960]), min=0.004855, max=19.199223, mean=2.612166
  features.5.1.block.1.0: coverage shape torch.Size([960]), min=0.000418, max=15.215022, mean=0.751695
  features.5.1.block.2.fc1: coverage shape torch.Size([40]), min=0.013287, max=3.934079, mean=0.969760
  features.5.1.block.2.fc2: coverage shape torch.Size([960]), min=0.003543, max=6.953300, mean=1.581387
  features.5.1.block.3.0: coverage shape torch.Size([160]), min=0.004584, max=1.561576, mean=0.408752
  features.5.2.block.0.0: coverage shape torch.Size([960]), min=0.018173, max=17.917385, mean=4.999826
  features.5.2.block.1.0: coverage shape torch.Size([960]), min=0.000088, max=12.644541, mean=0.610979
  features.5.2.block.2.fc1: coverage shape torch.Size([40]), min=0.098460, max=3.056908, mean=1.172840
  features.5.2.block.2.fc2: coverage shape torch.Size([960]), min=0.000407, max=6.258863, mean=1.385204
  features.5.2.block.3.0: coverage shape torch.Size([160]), min=0.001803, max=1.405194, mean=0.307117
  features.5.3.block.0.0: coverage shape torch.Size([960]), min=0.028015, max=33.365509, mean=8.175978
  features.5.3.block.1.0: coverage shape torch.Size([960]), min=0.000081, max=14.231907, mean=0.649765
  features.5.3.block.2.fc1: coverage shape torch.Size([40]), min=0.018182, max=6.709850, mean=1.892068
  features.5.3.block.2.fc2: coverage shape torch.Size([960]), min=0.000257, max=12.826449, mean=2.145785
  features.5.3.block.3.0: coverage shape torch.Size([160]), min=0.003342, max=1.716107, mean=0.404523
  features.5.4.block.0.0: coverage shape torch.Size([960]), min=0.005701, max=42.935768, mean=12.350568
  features.5.4.block.1.0: coverage shape torch.Size([960]), min=0.000059, max=27.330711, mean=0.650841
  features.5.4.block.2.fc1: coverage shape torch.Size([40]), min=0.097898, max=4.861397, mean=0.931747
  features.5.4.block.2.fc2: coverage shape torch.Size([960]), min=0.000617, max=6.656050, mean=1.413185
  features.5.4.block.3.0: coverage shape torch.Size([160]), min=0.007032, max=0.860430, mean=0.273534
  features.5.5.block.0.0: coverage shape torch.Size([960]), min=0.001536, max=64.392479, mean=17.743853
  features.5.5.block.1.0: coverage shape torch.Size([960]), min=0.000083, max=13.813315, mean=0.566461
  features.5.5.block.2.fc1: coverage shape torch.Size([40]), min=0.032838, max=2.854957, mean=0.840708
  features.5.5.block.2.fc2: coverage shape torch.Size([960]), min=0.002807, max=6.485136, mean=1.073431
  features.5.5.block.3.0: coverage shape torch.Size([160]), min=0.002051, max=0.993262, mean=0.318905
  features.6.0.block.0.0: coverage shape torch.Size([960]), min=0.032514, max=69.420334, mean=20.613058
  features.6.0.block.1.0: coverage shape torch.Size([960]), min=0.000006, max=130.430374, mean=1.737655
  features.6.0.block.2.fc1: coverage shape torch.Size([40]), min=0.070094, max=45.857365, mean=7.207217
  features.6.0.block.2.fc2: coverage shape torch.Size([960]), min=1.201670, max=4.857645, mean=2.312980
  features.6.0.block.3.0: coverage shape torch.Size([272]), min=0.100869, max=107.744362, mean=29.567905
  features.6.1.block.0.0: coverage shape torch.Size([1632]), min=0.001115, max=13.742572, mean=2.373191
  features.6.1.block.1.0: coverage shape torch.Size([1632]), min=0.000068, max=27.595055, mean=0.376057
  features.6.1.block.2.fc1: coverage shape torch.Size([68]), min=0.003558, max=4.373805, mean=1.218415
  features.6.1.block.2.fc2: coverage shape torch.Size([1632]), min=0.000140, max=5.571476, mean=1.310373
  features.6.1.block.3.0: coverage shape torch.Size([272]), min=0.004379, max=3.849205, mean=0.479547
  features.6.2.block.0.0: coverage shape torch.Size([1632]), min=0.001921, max=15.486609, mean=3.953124
  features.6.2.block.1.0: coverage shape torch.Size([1632]), min=0.000128, max=9.799919, mean=0.318072
  features.6.2.block.2.fc1: coverage shape torch.Size([68]), min=0.014838, max=3.332206, mean=0.899835
  features.6.2.block.2.fc2: coverage shape torch.Size([1632]), min=0.000142, max=6.385685, mean=1.158854
  features.6.2.block.3.0: coverage shape torch.Size([272]), min=0.001107, max=2.766101, mean=0.367926
  features.6.3.block.0.0: coverage shape torch.Size([1632]), min=0.004518, max=32.607914, mean=8.291677
  features.6.3.block.1.0: coverage shape torch.Size([1632]), min=0.000017, max=12.405740, mean=0.306482
  features.6.3.block.2.fc1: coverage shape torch.Size([68]), min=0.068797, max=3.315123, mean=0.979745
  features.6.3.block.2.fc2: coverage shape torch.Size([1632]), min=0.000828, max=6.485568, mean=1.282154
  features.6.3.block.3.0: coverage shape torch.Size([272]), min=0.006385, max=1.649880, mean=0.355696
  features.6.4.block.0.0: coverage shape torch.Size([1632]), min=0.000960, max=37.620983, mean=13.189050
  features.6.4.block.1.0: coverage shape torch.Size([1632]), min=0.000383, max=8.460897, mean=0.305775
  features.6.4.block.2.fc1: coverage shape torch.Size([68]), min=0.033987, max=2.899823, mean=0.911164
  features.6.4.block.2.fc2: coverage shape torch.Size([1632]), min=0.000111, max=5.942620, mean=1.174892
  features.6.4.block.3.0: coverage shape torch.Size([272]), min=0.001477, max=1.482880, mean=0.284397
  features.6.5.block.0.0: coverage shape torch.Size([1632]), min=0.039744, max=51.074795, mean=19.479034
  features.6.5.block.1.0: coverage shape torch.Size([1632]), min=0.000132, max=6.742333, mean=0.316871
  features.6.5.block.2.fc1: coverage shape torch.Size([68]), min=0.014909, max=2.929404, mean=0.900495
  features.6.5.block.2.fc2: coverage shape torch.Size([1632]), min=0.000886, max=5.857830, mean=1.093653
  features.6.5.block.3.0: coverage shape torch.Size([272]), min=0.001952, max=0.961126, mean=0.253267
  features.6.6.block.0.0: coverage shape torch.Size([1632]), min=0.091865, max=64.639252, mean=26.055731
  features.6.6.block.1.0: coverage shape torch.Size([1632]), min=0.000119, max=6.462984, mean=0.318237
  features.6.6.block.2.fc1: coverage shape torch.Size([68]), min=0.000469, max=3.022286, mean=0.848774
  features.6.6.block.2.fc2: coverage shape torch.Size([1632]), min=0.000819, max=5.026285, mean=1.133789
  features.6.6.block.3.0: coverage shape torch.Size([272]), min=0.004858, max=1.180757, mean=0.288484
  features.6.7.block.0.0: coverage shape torch.Size([1632]), min=0.054504, max=103.854973, mean=32.809284
  features.6.7.block.1.0: coverage shape torch.Size([1632]), min=0.000046, max=6.189996, mean=0.325109
  features.6.7.block.2.fc1: coverage shape torch.Size([68]), min=0.013734, max=3.575205, mean=0.915034
  features.6.7.block.2.fc2: coverage shape torch.Size([1632]), min=0.001849, max=5.105322, mean=0.997810
  features.6.7.block.3.0: coverage shape torch.Size([272]), min=0.000652, max=1.699778, mean=0.317885
  features.7.0.block.0.0: coverage shape torch.Size([1632]), min=0.124344, max=189.461868, mean=46.375145
  features.7.0.block.1.0: coverage shape torch.Size([1632]), min=0.000000, max=95.634827, mean=1.996001
  features.7.0.block.2.fc1: coverage shape torch.Size([68]), min=0.020424, max=7.527972, mean=2.576136
  features.7.0.block.2.fc2: coverage shape torch.Size([1632]), min=0.000084, max=16.855524, mean=2.986217
  features.7.0.block.3.0: coverage shape torch.Size([448]), min=0.002424, max=8.242309, mean=0.676880
  features.7.1.block.0.0: coverage shape torch.Size([2688]), min=0.000143, max=4.506327, mean=0.781999
  features.7.1.block.1.0: coverage shape torch.Size([2688]), min=0.000044, max=20.846529, mean=0.646608
  features.7.1.block.2.fc1: coverage shape torch.Size([112]), min=0.008418, max=4.873343, mean=1.618287
  features.7.1.block.2.fc2: coverage shape torch.Size([2688]), min=0.000328, max=12.961106, mean=1.930181
  features.7.1.block.3.0: coverage shape torch.Size([448]), min=0.000097, max=1.914211, mean=0.262652
  features.8.0: coverage shape torch.Size([1792]), min=1.173469, max=14.359065, mean=7.176383
  classifier.1: coverage shape torch.Size([101]), min=9.213017, max=14.209520, mean=11.377314

------------------------------------------------------------
Activation Statistics:
------------------------------------------------------------

features.0.0:
  Channels: 48
  Activation - Min: 0.000010, Max: 9.543589, Mean: 1.175885

features.1.0.block.0.0:
  Channels: 48
  Activation - Min: 0.000001, Max: 71.363930, Mean: 6.196229

features.1.0.block.1.fc1:
  Channels: 12
  Activation - Min: 2.213008, Max: 18.588037, Mean: 6.480382

features.1.0.block.1.fc2:
  Channels: 48
  Activation - Min: 0.013150, Max: 13.444448, Mean: 5.942733

features.1.0.block.2.0:
  Channels: 24
  Activation - Min: 8.379252, Max: 202.707291, Mean: 71.898041

features.1.1.block.0.0:
  Channels: 24
  Activation - Min: 0.006973, Max: 19.292110, Mean: 3.961229

features.1.1.block.1.fc1:
  Channels: 6
  Activation - Min: 2.492083, Max: 15.853820, Mean: 9.894175

features.1.1.block.1.fc2:
  Channels: 24
  Activation - Min: 1.527064, Max: 4.539252, Mean: 2.945816

features.1.1.block.2.0:
  Channels: 24
  Activation - Min: 1.192892, Max: 103.464394, Mean: 23.983599

features.2.0.block.0.0:
  Channels: 144
  Activation - Min: 0.049043, Max: 84.112755, Mean: 15.515657

features.2.0.block.1.0:
  Channels: 144
  Activation - Min: 0.006556, Max: 69.707596, Mean: 14.040387

features.2.0.block.2.fc1:
  Channels: 6
  Activation - Min: 1.430914, Max: 7.124955, Mean: 3.543283

features.2.0.block.2.fc2:
  Channels: 144
  Activation - Min: 0.122095, Max: 5.332840, Mean: 3.027450

features.2.0.block.3.0:
  Channels: 32
  Activation - Min: 7.144043, Max: 263.575562, Mean: 101.624069

features.2.1.block.0.0:
  Channels: 192
  Activation - Min: 0.203832, Max: 143.015549, Mean: 21.001665

features.2.1.block.1.0:
  Channels: 192
  Activation - Min: 0.000460, Max: 20.473652, Mean: 1.352044

features.2.1.block.2.fc1:
  Channels: 8
  Activation - Min: 1.569479, Max: 40.032692, Mean: 7.479882

features.2.1.block.2.fc2:
  Channels: 192
  Activation - Min: 0.341097, Max: 38.590317, Mean: 5.929050

features.2.1.block.3.0:
  Channels: 32
  Activation - Min: 1.891819, Max: 128.478638, Mean: 44.913597

features.2.2.block.0.0:
  Channels: 192
  Activation - Min: 0.000338, Max: 74.394455, Mean: 17.899221

features.2.2.block.1.0:
  Channels: 192
  Activation - Min: 0.000003, Max: 23.985632, Mean: 1.776366

features.2.2.block.2.fc1:
  Channels: 8
  Activation - Min: 0.493672, Max: 28.279497, Mean: 5.046848

features.2.2.block.2.fc2:
  Channels: 192
  Activation - Min: 0.104489, Max: 12.220940, Mean: 3.290544

features.2.2.block.3.0:
  Channels: 32
  Activation - Min: 3.079779, Max: 52.057362, Mean: 18.444427

features.2.3.block.0.0:
  Channels: 192
  Activation - Min: 0.146540, Max: 98.717461, Mean: 24.183615

features.2.3.block.1.0:
  Channels: 192
  Activation - Min: 0.000077, Max: 14.663739, Mean: 1.476071

features.2.3.block.2.fc1:
  Channels: 8
  Activation - Min: 2.204707, Max: 2.541802, Mean: 2.395243

features.2.3.block.2.fc2:
  Channels: 192
  Activation - Min: 0.102524, Max: 5.015934, Mean: 1.623722

features.2.3.block.3.0:
  Channels: 32
  Activation - Min: 0.138702, Max: 35.775822, Mean: 12.902413

features.3.0.block.0.0:
  Channels: 192
  Activation - Min: 0.097459, Max: 145.721451, Mean: 39.466152

features.3.0.block.1.0:
  Channels: 192
  Activation - Min: 0.000097, Max: 63.454346, Mean: 4.252139

features.3.0.block.2.fc1:
  Channels: 8
  Activation - Min: 5.450554, Max: 48.401188, Mean: 21.064997

features.3.0.block.2.fc2:
  Channels: 192
  Activation - Min: 0.083362, Max: 20.301817, Mean: 6.127841

features.3.0.block.3.0:
  Channels: 56
  Activation - Min: 0.858595, Max: 235.979156, Mean: 61.938747

features.3.1.block.0.0:
  Channels: 336
  Activation - Min: 0.002978, Max: 58.641388, Mean: 7.696443

features.3.1.block.1.0:
  Channels: 336
  Activation - Min: 0.000088, Max: 28.824862, Mean: 1.584996

features.3.1.block.2.fc1:
  Channels: 14
  Activation - Min: 0.754659, Max: 38.841068, Mean: 15.442009

features.3.1.block.2.fc2:
  Channels: 336
  Activation - Min: 0.262587, Max: 86.974152, Mean: 12.666800

features.3.1.block.3.0:
  Channels: 56
  Activation - Min: 0.021071, Max: 27.853456, Mean: 10.873216

features.3.2.block.0.0:
  Channels: 336
  Activation - Min: 0.024421, Max: 69.926529, Mean: 12.424342

features.3.2.block.1.0:
  Channels: 336
  Activation - Min: 0.000671, Max: 22.855808, Mean: 1.533637

features.3.2.block.2.fc1:
  Channels: 14
  Activation - Min: 0.058282, Max: 50.264351, Mean: 10.330870

features.3.2.block.2.fc2:
  Channels: 336
  Activation - Min: 0.240110, Max: 115.111359, Mean: 8.058840

features.3.2.block.3.0:
  Channels: 56
  Activation - Min: 0.584855, Max: 26.812168, Mean: 7.095654

features.3.3.block.0.0:
  Channels: 336
  Activation - Min: 0.033525, Max: 77.479393, Mean: 19.298447

features.3.3.block.1.0:
  Channels: 336
  Activation - Min: 0.000801, Max: 21.374039, Mean: 1.620977

features.3.3.block.2.fc1:
  Channels: 14
  Activation - Min: 0.221091, Max: 32.047379, Mean: 10.421250

features.3.3.block.2.fc2:
  Channels: 336
  Activation - Min: 0.049499, Max: 90.307510, Mean: 10.043493

features.3.3.block.3.0:
  Channels: 56
  Activation - Min: 0.200774, Max: 11.219766, Mean: 3.778970

features.4.0.block.0.0:
  Channels: 336
  Activation - Min: 0.221818, Max: 106.700623, Mean: 26.547802

features.4.0.block.1.0:
  Channels: 336
  Activation - Min: 0.000190, Max: 44.366432, Mean: 2.410887

features.4.0.block.2.fc1:
  Channels: 14
  Activation - Min: 2.077830, Max: 28.458878, Mean: 13.322555

features.4.0.block.2.fc2:
  Channels: 336
  Activation - Min: 0.155690, Max: 7.399580, Mean: 3.156038

features.4.0.block.3.0:
  Channels: 112
  Activation - Min: 0.020427, Max: 137.821381, Mean: 39.806252

features.4.1.block.0.0:
  Channels: 672
  Activation - Min: 0.038166, Max: 73.658180, Mean: 8.215742

features.4.1.block.1.0:
  Channels: 672
  Activation - Min: 0.000439, Max: 16.050774, Mean: 0.759308

features.4.1.block.2.fc1:
  Channels: 28
  Activation - Min: 0.918786, Max: 39.550297, Mean: 13.305115

features.4.1.block.2.fc2:
  Channels: 672
  Activation - Min: 0.003424, Max: 185.706268, Mean: 19.163393

features.4.1.block.3.0:
  Channels: 112
  Activation - Min: 0.037049, Max: 10.160813, Mean: 2.513325

features.4.2.block.0.0:
  Channels: 672
  Activation - Min: 0.022838, Max: 48.823978, Mean: 5.586002

features.4.2.block.1.0:
  Channels: 672
  Activation - Min: 0.000038, Max: 9.240995, Mean: 0.882491

features.4.2.block.2.fc1:
  Channels: 28
  Activation - Min: 0.018105, Max: 1.267408, Mean: 0.636035

features.4.2.block.2.fc2:
  Channels: 672
  Activation - Min: 0.000641, Max: 6.236047, Mean: 1.281504

features.4.2.block.3.0:
  Channels: 112
  Activation - Min: 0.009497, Max: 1.317272, Mean: 0.319354

features.4.3.block.0.0:
  Channels: 672
  Activation - Min: 0.006142, Max: 43.337242, Mean: 7.698403

features.4.3.block.1.0:
  Channels: 672
  Activation - Min: 0.000422, Max: 12.850874, Mean: 0.466811

features.4.3.block.2.fc1:
  Channels: 28
  Activation - Min: 0.028621, Max: 35.604507, Mean: 10.090437

features.4.3.block.2.fc2:
  Channels: 672
  Activation - Min: 0.082854, Max: 153.077332, Mean: 14.789365

features.4.3.block.3.0:
  Channels: 112
  Activation - Min: 0.002659, Max: 5.313998, Mean: 1.900539

features.4.4.block.0.0:
  Channels: 672
  Activation - Min: 0.016974, Max: 39.822266, Mean: 12.346816

features.4.4.block.1.0:
  Channels: 672
  Activation - Min: 0.000167, Max: 7.346904, Mean: 0.647383

features.4.4.block.2.fc1:
  Channels: 28
  Activation - Min: 0.627229, Max: 13.304763, Mean: 4.532600

features.4.4.block.2.fc2:
  Channels: 672
  Activation - Min: 0.056739, Max: 28.782646, Mean: 5.723851

features.4.4.block.3.0:
  Channels: 112
  Activation - Min: 0.003720, Max: 2.511139, Mean: 0.662294

features.4.5.block.0.0:
  Channels: 672
  Activation - Min: 0.067938, Max: 54.779911, Mean: 14.574196

features.4.5.block.1.0:
  Channels: 672
  Activation - Min: 0.000299, Max: 12.534796, Mean: 0.871697

features.4.5.block.2.fc1:
  Channels: 28
  Activation - Min: 0.030882, Max: 3.372029, Mean: 1.321679

features.4.5.block.2.fc2:
  Channels: 672
  Activation - Min: 0.001946, Max: 8.302926, Mean: 1.959165

features.4.5.block.3.0:
  Channels: 112
  Activation - Min: 0.001601, Max: 1.397083, Mean: 0.318746

features.5.0.block.0.0:
  Channels: 672
  Activation - Min: 0.006385, Max: 74.775764, Mean: 15.148971

features.5.0.block.1.0:
  Channels: 672
  Activation - Min: 0.000172, Max: 139.789429, Mean: 1.691830

features.5.0.block.2.fc1:
  Channels: 28
  Activation - Min: 0.439604, Max: 21.014698, Mean: 5.001423

features.5.0.block.2.fc2:
  Channels: 672
  Activation - Min: 0.563461, Max: 4.354167, Mean: 2.505469

features.5.0.block.3.0:
  Channels: 160
  Activation - Min: 0.220785, Max: 107.051712, Mean: 24.518240

features.5.1.block.0.0:
  Channels: 960
  Activation - Min: 0.004855, Max: 19.199223, Mean: 2.612166

features.5.1.block.1.0:
  Channels: 960
  Activation - Min: 0.000418, Max: 15.215022, Mean: 0.751695

features.5.1.block.2.fc1:
  Channels: 40
  Activation - Min: 0.013287, Max: 3.934079, Mean: 0.969760

features.5.1.block.2.fc2:
  Channels: 960
  Activation - Min: 0.003543, Max: 6.953300, Mean: 1.581387

features.5.1.block.3.0:
  Channels: 160
  Activation - Min: 0.004584, Max: 1.561576, Mean: 0.408752

features.5.2.block.0.0:
  Channels: 960
  Activation - Min: 0.018173, Max: 17.917385, Mean: 4.999826

features.5.2.block.1.0:
  Channels: 960
  Activation - Min: 0.000088, Max: 12.644541, Mean: 0.610979

features.5.2.block.2.fc1:
  Channels: 40
  Activation - Min: 0.098460, Max: 3.056908, Mean: 1.172840

features.5.2.block.2.fc2:
  Channels: 960
  Activation - Min: 0.000407, Max: 6.258863, Mean: 1.385204

features.5.2.block.3.0:
  Channels: 160
  Activation - Min: 0.001803, Max: 1.405194, Mean: 0.307117

features.5.3.block.0.0:
  Channels: 960
  Activation - Min: 0.028015, Max: 33.365509, Mean: 8.175978

features.5.3.block.1.0:
  Channels: 960
  Activation - Min: 0.000081, Max: 14.231907, Mean: 0.649765

features.5.3.block.2.fc1:
  Channels: 40
  Activation - Min: 0.018182, Max: 6.709850, Mean: 1.892068

features.5.3.block.2.fc2:
  Channels: 960
  Activation - Min: 0.000257, Max: 12.826449, Mean: 2.145785

features.5.3.block.3.0:
  Channels: 160
  Activation - Min: 0.003342, Max: 1.716107, Mean: 0.404523

features.5.4.block.0.0:
  Channels: 960
  Activation - Min: 0.005701, Max: 42.935768, Mean: 12.350568

features.5.4.block.1.0:
  Channels: 960
  Activation - Min: 0.000059, Max: 27.330711, Mean: 0.650841

features.5.4.block.2.fc1:
  Channels: 40
  Activation - Min: 0.097898, Max: 4.861397, Mean: 0.931747

features.5.4.block.2.fc2:
  Channels: 960
  Activation - Min: 0.000617, Max: 6.656050, Mean: 1.413185

features.5.4.block.3.0:
  Channels: 160
  Activation - Min: 0.007032, Max: 0.860430, Mean: 0.273534

features.5.5.block.0.0:
  Channels: 960
  Activation - Min: 0.001536, Max: 64.392479, Mean: 17.743853

features.5.5.block.1.0:
  Channels: 960
  Activation - Min: 0.000083, Max: 13.813315, Mean: 0.566461

features.5.5.block.2.fc1:
  Channels: 40
  Activation - Min: 0.032838, Max: 2.854957, Mean: 0.840708

features.5.5.block.2.fc2:
  Channels: 960
  Activation - Min: 0.002807, Max: 6.485136, Mean: 1.073431

features.5.5.block.3.0:
  Channels: 160
  Activation - Min: 0.002051, Max: 0.993262, Mean: 0.318905

features.6.0.block.0.0:
  Channels: 960
  Activation - Min: 0.032514, Max: 69.420334, Mean: 20.613058

features.6.0.block.1.0:
  Channels: 960
  Activation - Min: 0.000006, Max: 130.430374, Mean: 1.737655

features.6.0.block.2.fc1:
  Channels: 40
  Activation - Min: 0.070094, Max: 45.857365, Mean: 7.207217

features.6.0.block.2.fc2:
  Channels: 960
  Activation - Min: 1.201670, Max: 4.857645, Mean: 2.312980

features.6.0.block.3.0:
  Channels: 272
  Activation - Min: 0.100869, Max: 107.744362, Mean: 29.567905

features.6.1.block.0.0:
  Channels: 1632
  Activation - Min: 0.001115, Max: 13.742572, Mean: 2.373191

features.6.1.block.1.0:
  Channels: 1632
  Activation - Min: 0.000068, Max: 27.595055, Mean: 0.376057

features.6.1.block.2.fc1:
  Channels: 68
  Activation - Min: 0.003558, Max: 4.373805, Mean: 1.218415

features.6.1.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000140, Max: 5.571476, Mean: 1.310373

features.6.1.block.3.0:
  Channels: 272
  Activation - Min: 0.004379, Max: 3.849205, Mean: 0.479547

features.6.2.block.0.0:
  Channels: 1632
  Activation - Min: 0.001921, Max: 15.486609, Mean: 3.953124

features.6.2.block.1.0:
  Channels: 1632
  Activation - Min: 0.000128, Max: 9.799919, Mean: 0.318072

features.6.2.block.2.fc1:
  Channels: 68
  Activation - Min: 0.014838, Max: 3.332206, Mean: 0.899835

features.6.2.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000142, Max: 6.385685, Mean: 1.158854

features.6.2.block.3.0:
  Channels: 272
  Activation - Min: 0.001107, Max: 2.766101, Mean: 0.367926

features.6.3.block.0.0:
  Channels: 1632
  Activation - Min: 0.004518, Max: 32.607914, Mean: 8.291677

features.6.3.block.1.0:
  Channels: 1632
  Activation - Min: 0.000017, Max: 12.405740, Mean: 0.306482

features.6.3.block.2.fc1:
  Channels: 68
  Activation - Min: 0.068797, Max: 3.315123, Mean: 0.979745

features.6.3.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000828, Max: 6.485568, Mean: 1.282154

features.6.3.block.3.0:
  Channels: 272
  Activation - Min: 0.006385, Max: 1.649880, Mean: 0.355696

features.6.4.block.0.0:
  Channels: 1632
  Activation - Min: 0.000960, Max: 37.620983, Mean: 13.189050

features.6.4.block.1.0:
  Channels: 1632
  Activation - Min: 0.000383, Max: 8.460897, Mean: 0.305775

features.6.4.block.2.fc1:
  Channels: 68
  Activation - Min: 0.033987, Max: 2.899823, Mean: 0.911164

features.6.4.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000111, Max: 5.942620, Mean: 1.174892

features.6.4.block.3.0:
  Channels: 272
  Activation - Min: 0.001477, Max: 1.482880, Mean: 0.284397

features.6.5.block.0.0:
  Channels: 1632
  Activation - Min: 0.039744, Max: 51.074795, Mean: 19.479034

features.6.5.block.1.0:
  Channels: 1632
  Activation - Min: 0.000132, Max: 6.742333, Mean: 0.316871

features.6.5.block.2.fc1:
  Channels: 68
  Activation - Min: 0.014909, Max: 2.929404, Mean: 0.900495

features.6.5.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000886, Max: 5.857830, Mean: 1.093653

features.6.5.block.3.0:
  Channels: 272
  Activation - Min: 0.001952, Max: 0.961126, Mean: 0.253267

features.6.6.block.0.0:
  Channels: 1632
  Activation - Min: 0.091865, Max: 64.639252, Mean: 26.055731

features.6.6.block.1.0:
  Channels: 1632
  Activation - Min: 0.000119, Max: 6.462984, Mean: 0.318237

features.6.6.block.2.fc1:
  Channels: 68
  Activation - Min: 0.000469, Max: 3.022286, Mean: 0.848774

features.6.6.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000819, Max: 5.026285, Mean: 1.133789

features.6.6.block.3.0:
  Channels: 272
  Activation - Min: 0.004858, Max: 1.180757, Mean: 0.288484

features.6.7.block.0.0:
  Channels: 1632
  Activation - Min: 0.054504, Max: 103.854973, Mean: 32.809284

features.6.7.block.1.0:
  Channels: 1632
  Activation - Min: 0.000046, Max: 6.189996, Mean: 0.325109

features.6.7.block.2.fc1:
  Channels: 68
  Activation - Min: 0.013734, Max: 3.575205, Mean: 0.915034

features.6.7.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.001849, Max: 5.105322, Mean: 0.997810

features.6.7.block.3.0:
  Channels: 272
  Activation - Min: 0.000652, Max: 1.699778, Mean: 0.317885

features.7.0.block.0.0:
  Channels: 1632
  Activation - Min: 0.124344, Max: 189.461868, Mean: 46.375145

features.7.0.block.1.0:
  Channels: 1632
  Activation - Min: 0.000000, Max: 95.634827, Mean: 1.996001

features.7.0.block.2.fc1:
  Channels: 68
  Activation - Min: 0.020424, Max: 7.527972, Mean: 2.576136

features.7.0.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000084, Max: 16.855524, Mean: 2.986217

features.7.0.block.3.0:
  Channels: 448
  Activation - Min: 0.002424, Max: 8.242309, Mean: 0.676880

features.7.1.block.0.0:
  Channels: 2688
  Activation - Min: 0.000143, Max: 4.506327, Mean: 0.781999

features.7.1.block.1.0:
  Channels: 2688
  Activation - Min: 0.000044, Max: 20.846529, Mean: 0.646608

features.7.1.block.2.fc1:
  Channels: 112
  Activation - Min: 0.008418, Max: 4.873343, Mean: 1.618287

features.7.1.block.2.fc2:
  Channels: 2688
  Activation - Min: 0.000328, Max: 12.961106, Mean: 1.930181

features.7.1.block.3.0:
  Channels: 448
  Activation - Min: 0.000097, Max: 1.914211, Mean: 0.262652

features.8.0:
  Channels: 1792
  Activation - Min: 1.173469, Max: 14.359065, Mean: 7.176383

classifier.1:
  Channels: 101
  Activation - Min: 9.213017, Max: 14.209520, Mean: 11.377314
============================================================


Pruning Results:
  Parameters before: 17,729,709
  Parameters after:  14,395,049
  Parameters removed: 3,334,660 (18.81%)

############################################################
Pruning Complete
############################################################
Initial parameters:  17,729,709
Final parameters:    14,395,049
Total removed:       3,334,660 (18.81%)
Target pruning ratio: 10.00%
✓ Wanda pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 1.81%                                                                                                                  
✓ Model Size: 55.35 MB
✓ Average Inference Time: 5.2322 ms
✓ FLOPs: 3.76 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.10it/s, loss=1.6446, acc=58.18%] 
Epoch 1/10 - Train Loss: 1.6446, Train Acc: 58.18%, Test Acc: 81.12%                                                                            
Epoch 2/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.11it/s, loss=1.3872, acc=64.20%] 
Epoch 2/10 - Train Loss: 1.3872, Train Acc: 64.20%, Test Acc: 81.69%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_W_epoch2.pth
Epoch 3/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.11it/s, loss=1.2616, acc=66.70%]
Epoch 3/10 - Train Loss: 1.2616, Train Acc: 66.70%, Test Acc: 82.65%                                                                            
Epoch 4/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.10it/s, loss=1.1607, acc=69.60%] 
Epoch 4/10 - Train Loss: 1.1607, Train Acc: 69.60%, Test Acc: 82.14%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_W_epoch4.pth
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:41<00:00,  3.09it/s, loss=1.0688, acc=71.44%]
Epoch 5/10 - Train Loss: 1.0688, Train Acc: 71.44%, Test Acc: 82.37%                                                                            
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:41<00:00,  3.09it/s, loss=1.0515, acc=72.48%] 
Epoch 6/10 - Train Loss: 1.0515, Train Acc: 72.48%, Test Acc: 82.86%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_W_epoch6.pth
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:41<00:00,  3.10it/s, loss=1.0212, acc=72.74%]
Epoch 7/10 - Train Loss: 1.0212, Train Acc: 72.74%, Test Acc: 82.97%                                                                            
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.11it/s, loss=0.9975, acc=73.50%] 
Epoch 8/10 - Train Loss: 0.9975, Train Acc: 73.50%, Test Acc: 83.17%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_W_epoch8.pth
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:40<00:00,  3.11it/s, loss=0.9732, acc=74.46%]
Epoch 9/10 - Train Loss: 0.9732, Train Acc: 74.46%, Test Acc: 83.02%                                                                            
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████| 313/313 [01:41<00:00,  3.07it/s, loss=0.9450, acc=75.08%] 
Epoch 10/10 - Train Loss: 0.9450, Train Acc: 75.08%, Test Acc: 83.30%                                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_W_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_W_best.pth
  Best Accuracy: 83.30%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 83.30%                                                                                                                  
✓ Model Size: 55.35 MB
✓ Average Inference Time: 5.1453 ms
✓ FLOPs: 3.76 GFLOPs

================================================================================
COMPARISON: WANDA PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         89.21   |          1.81   |    83.3    | -5.91                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         68.11   |         55.35   |    55.35   | -12.76 (-18.7%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.5898 |          5.2322 |     5.1453 | +4.5555                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.61   |          3.76   |     3.76   | -0.85 (-18.5%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS5_EfficientNetB4_StanfordDogs\TS5_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================


*******************************




================================================================================
TEST SCENARIO TS5: MAGNITUDE-BASED PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FT_best.pth
  - Epoch: 9
  - Test Accuracy: 89.21%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 89.21%                                                                                                               
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.6050 ms
✓ FLOPs: 4.61 GFLOPs

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
Initial parameters: 17,729,709

============================================================
Pruning Step 1/1
============================================================

Pruning Results:
  Parameters before: 17,729,709
  Parameters after:  14,395,049
  Parameters removed: 3,334,660 (18.81%)

############################################################
Pruning Complete
############################################################
Initial parameters:  17,729,709
Final parameters:    14,395,049
Total removed:       3,334,660 (18.81%)
Target pruning ratio: 10.00%
✓ Magnitude-based pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 27.56%                                                                                                                 
✓ Model Size: 55.35 MB
✓ Average Inference Time: 0.5721 ms
✓ FLOPs: 3.76 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.42it/s, loss=1.2149, acc=68.02%] 
Epoch 1/10 - Train Loss: 1.2149, Train Acc: 68.02%, Test Acc: 84.84%                                                                            
Epoch 2/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:30<00:00,  3.44it/s, loss=1.0736, acc=71.94%] 
Epoch 2/10 - Train Loss: 1.0736, Train Acc: 71.94%, Test Acc: 85.13%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_MAG_epoch2.pth
Epoch 3/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.44it/s, loss=1.0019, acc=73.28%]
Epoch 3/10 - Train Loss: 1.0019, Train Acc: 73.28%, Test Acc: 85.19%                                                                            
Epoch 4/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.9101, acc=75.74%] 
Epoch 4/10 - Train Loss: 0.9101, Train Acc: 75.74%, Test Acc: 85.23%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_MAG_epoch4.pth
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.8609, acc=76.58%]
Epoch 5/10 - Train Loss: 0.8609, Train Acc: 76.58%, Test Acc: 85.49%                                                                            
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.8247, acc=77.66%] 
Epoch 6/10 - Train Loss: 0.8247, Train Acc: 77.66%, Test Acc: 85.62%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_MAG_epoch6.pth
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.44it/s, loss=0.8002, acc=78.60%]
Epoch 7/10 - Train Loss: 0.8002, Train Acc: 78.60%, Test Acc: 85.98%                                                                            
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.44it/s, loss=0.7681, acc=79.28%] 
Epoch 8/10 - Train Loss: 0.7681, Train Acc: 79.28%, Test Acc: 85.94%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_MAG_epoch8.pth
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.44it/s, loss=0.7298, acc=80.32%]
Epoch 9/10 - Train Loss: 0.7298, Train Acc: 80.32%, Test Acc: 85.93%                                                                            
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.7480, acc=80.00%] 
Epoch 10/10 - Train Loss: 0.7480, Train Acc: 80.00%, Test Acc: 85.83%                                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_MAG_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_MAG_best.pth
  Best Accuracy: 85.98%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 85.83%                                                                                                                  
✓ Model Size: 55.35 MB
✓ Average Inference Time: 0.5603 ms
✓ FLOPs: 3.76 GFLOPs

================================================================================
COMPARISON: MAGNITUDE-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |          89.21  |         27.56   |    85.83   | -3.38                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |          68.11  |         55.35   |    55.35   | -12.76 (-18.7%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |           0.605 |          0.5721 |     0.5603 | -0.0446                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |           4.61  |          3.76   |     3.76   | -0.85 (-18.5%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS5_EfficientNetB4_StanfordDogs\TS5_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================





***************************************




================================================================================
TEST SCENARIO TS5: TAYLOR GRADIENT-BASED PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FT_best.pth
  - Epoch: 9
  - Test Accuracy: 89.21%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 89.21%                                                                                                               
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.5873 ms
✓ FLOPs: 4.61 GFLOPs

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
Computing gradients:  99%|███████████████████████████████████████████████████████████████████████████████████▏| 99/100 [00:40<00:00,  2.46it/s] 
✓ Gradients computed on 100 batches
Using GroupTaylorImportance

Initializing Torch-Pruning with Taylor importance...

Applying pruning...

✓ Taylor pruning completed
  Parameters before: 17,729,709
  Parameters after: 14,395,049
  Parameters removed: 3,334,660 (18.81%)

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 34.38%                                                                                                                 
✓ Model Size: 55.35 MB
✓ Average Inference Time: 0.5730 ms
✓ FLOPs: 3.76 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.42it/s, loss=1.0888, acc=72.02%] 
Epoch 1/10 - Train Loss: 1.0888, Train Acc: 72.02%, Test Acc: 86.65%                                                                            
Epoch 2/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.9623, acc=75.12%] 
Epoch 2/10 - Train Loss: 0.9623, Train Acc: 75.12%, Test Acc: 86.69%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_TAY_epoch2.pth
Epoch 3/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.8908, acc=76.28%]
Epoch 3/10 - Train Loss: 0.8908, Train Acc: 76.28%, Test Acc: 86.72%                                                                            
Epoch 4/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.42it/s, loss=0.8562, acc=77.12%] 
Epoch 4/10 - Train Loss: 0.8562, Train Acc: 77.12%, Test Acc: 86.59%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_TAY_epoch4.pth
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.8301, acc=77.28%]
Epoch 5/10 - Train Loss: 0.8301, Train Acc: 77.28%, Test Acc: 86.88%                                                                            
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.7365, acc=80.52%] 
Epoch 6/10 - Train Loss: 0.7365, Train Acc: 80.52%, Test Acc: 86.79%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_TAY_epoch6.pth
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.7176, acc=80.66%]
Epoch 7/10 - Train Loss: 0.7176, Train Acc: 80.66%, Test Acc: 86.65%                                                                            
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.6950, acc=81.46%] 
Epoch 8/10 - Train Loss: 0.6950, Train Acc: 81.46%, Test Acc: 86.75%                                                                            
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_TAY_epoch8.pth
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.6968, acc=81.70%]
Epoch 9/10 - Train Loss: 0.6968, Train Acc: 81.70%, Test Acc: 86.69%                                                                            
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.7082, acc=81.06%] 
Epoch 10/10 - Train Loss: 0.7082, Train Acc: 81.06%, Test Acc: 86.67%                                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_TAY_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS5\EfficientNetB4_Food101_FTAP_TAY_best.pth
  Best Accuracy: 86.88%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 86.67%                                                                                                                  
✓ Model Size: 55.35 MB
✓ Average Inference Time: 0.5560 ms
✓ FLOPs: 3.76 GFLOPs

================================================================================
COMPARISON: TAYLOR GRADIENT-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         89.21   |          34.38  |     86.67  | -2.53                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         68.11   |          55.35  |     55.35  | -12.76 (-18.7%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.5873 |           0.573 |      0.556 | -0.0313                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.61   |           3.76  |      3.76  | -0.85 (-18.5%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS5_EfficientNetB4_StanfordDogs\TS5_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================