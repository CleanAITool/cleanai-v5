================================================================================
TEST SCENARIO TS9: PREPARE EfficientNetB4 MODEL
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
✓ Pretrained Model Accuracy: 1.17%                                                                                                             
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.5962 ms
✓ FLOPs: 4.61 GFLOPs

================================================================================
FINE-TUNING MODEL ON STANFORD DOGS
================================================================================
Epoch 1/10: 100%|████████████████████████████████████████████████████████████████| 4735/4735 [22:05<00:00,  3.57it/s, loss=2.2453, acc=45.87%] 
Epoch 1/10 - Train Loss: 2.2453, Train Acc: 45.87%, Test Acc: 77.69%                                                                           
Epoch 2/10: 100%|████████████████████████████████████████████████████████████████| 4735/4735 [22:07<00:00,  3.57it/s, loss=1.4154, acc=63.85%]
Epoch 2/10 - Train Loss: 1.4154, Train Acc: 63.85%, Test Acc: 83.81%                                                                           
Epoch 3/10: 100%|████████████████████████████████████████████████████████████████| 4735/4735 [22:07<00:00,  3.57it/s, loss=1.2081, acc=68.98%]
Epoch 3/10 - Train Loss: 1.2081, Train Acc: 68.98%, Test Acc: 85.53%                                                                           
Epoch 4/10: 100%|████████████████████████████████████████████████████████████████| 4735/4735 [22:07<00:00,  3.57it/s, loss=1.0762, acc=72.00%]
Epoch 4/10 - Train Loss: 1.0762, Train Acc: 72.00%, Test Acc: 87.05%                                                                           
Epoch 5/10: 100%|████████████████████████████████████████████████████████████████| 4735/4735 [22:07<00:00,  3.57it/s, loss=0.9894, acc=74.19%]
Epoch 5/10 - Train Loss: 0.9894, Train Acc: 74.19%, Test Acc: 87.87%                                                                           
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FT_epoch5.pth
Epoch 6/10: 100%|████████████████████████████████████████████████████████████████| 4735/4735 [22:07<00:00,  3.57it/s, loss=0.9291, acc=75.83%]
Epoch 6/10 - Train Loss: 0.9291, Train Acc: 75.83%, Test Acc: 88.36%                                                                           
Epoch 7/10: 100%|████████████████████████████████████████████████████████████████| 4735/4735 [22:07<00:00,  3.57it/s, loss=0.8719, acc=77.20%]
Epoch 7/10 - Train Loss: 0.8719, Train Acc: 77.20%, Test Acc: 88.61%                                                                           
Epoch 8/10: 100%|████████████████████████████████████████████████████████████████| 4735/4735 [22:07<00:00,  3.57it/s, loss=0.8364, acc=77.89%]
Epoch 8/10 - Train Loss: 0.8364, Train Acc: 77.89%, Test Acc: 89.08%                                                                           
Epoch 9/10: 100%|████████████████████████████████████████████████████████████████| 4735/4735 [22:07<00:00,  3.57it/s, loss=0.8201, acc=78.34%]
Epoch 9/10 - Train Loss: 0.8201, Train Acc: 78.34%, Test Acc: 89.21%                                                                           
Epoch 10/10: 100%|███████████████████████████████████████████████████████████████| 4735/4735 [22:07<00:00,  3.57it/s, loss=0.8053, acc=78.77%]
Epoch 10/10 - Train Loss: 0.8053, Train Acc: 78.77%, Test Acc: 89.38%                                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FT_epoch10.pth
c:\source\repos\cleanai-v5\test_scenarios\TS9_EfficientNetB4_Food101\TS9_01_prepare_model.py:345: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(best_checkpoint, map_location=CONFIG['device'])

================================================================================
FINAL EVALUATION (AFTER FINE-TUNING)
================================================================================
✓ Fine-tuned Model Accuracy: 89.38%                                                                                                            
✓ Model Size: 68.12 MB
✓ Average Inference Time: 0.6114 ms
✓ FLOPs: 4.61 GFLOPs

================================================================================
COMPARISON: BEFORE vs AFTER FINE-TUNING
================================================================================
+---------------------+--------------------------+-------------------------+----------+
| Metric              |   Pretrained (Before FT) |   Fine-tuned (After FT) |   Change |
+=====================+==========================+=========================+==========+
| Accuracy (%)        |                   1.17   |                 89.38   |  88.21   |
+---------------------+--------------------------+-------------------------+----------+
| Size (MB)           |                  68.11   |                 68.12   |   0      |
+---------------------+--------------------------+-------------------------+----------+
| Inference Time (ms) |                   0.5962 |                  0.6114 |   0.0152 |
+---------------------+--------------------------+-------------------------+----------+
| FLOPs (G)           |                   4.61   |                  4.61   |   0      |
+---------------------+--------------------------+-------------------------+----------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS9_EfficientNetB4_Food101\TS9_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================




********************************




 features.3.0.block.2.fc2: coverage shape torch.Size([192]), min=-0.608386, max=1.000000, mean=0.293847
  features.3.0.block.3.0: coverage shape torch.Size([56]), min=-0.761944, max=1.000000, mean=0.014581
  features.3.1.block.0.0: coverage shape torch.Size([336]), min=-0.862992, max=1.000000, mean=0.008858
  features.3.1.block.1.0: coverage shape torch.Size([336]), min=-3.720114, max=1.000000, mean=-0.130605
  features.3.1.block.2.fc1: coverage shape torch.Size([14]), min=-0.043882, max=1.000000, mean=0.263726
  features.3.1.block.2.fc2: coverage shape torch.Size([336]), min=-1.557634, max=1.000000, mean=0.161927
  features.3.1.block.3.0: coverage shape torch.Size([56]), min=-1.018814, max=1.000000, mean=0.042396
  features.3.2.block.0.0: coverage shape torch.Size([336]), min=-1.964223, max=1.000000, mean=-0.221253
  features.3.2.block.1.0: coverage shape torch.Size([336]), min=-1.942446, max=1.000000, mean=-0.073373
  features.3.2.block.2.fc1: coverage shape torch.Size([14]), min=-0.339858, max=1.000000, mean=0.031272
  features.3.2.block.2.fc2: coverage shape torch.Size([336]), min=-2.530282, max=1.000000, mean=0.126706
  features.3.2.block.3.0: coverage shape torch.Size([56]), min=-0.715217, max=1.000000, mean=0.019824
  features.3.3.block.0.0: coverage shape torch.Size([336]), min=-0.982032, max=1.000000, mean=-0.142124
  features.3.3.block.1.0: coverage shape torch.Size([336]), min=-1.701053, max=1.000000, mean=-0.064831
  features.3.3.block.2.fc1: coverage shape torch.Size([14]), min=-0.727502, max=1.000000, mean=-0.017758
  features.3.3.block.2.fc2: coverage shape torch.Size([336]), min=-0.637394, max=1.000000, mean=0.060623
  features.3.3.block.3.0: coverage shape torch.Size([56]), min=-0.777795, max=1.000000, mean=-0.020132
  features.4.0.block.0.0: coverage shape torch.Size([336]), min=-0.682449, max=1.000000, mean=-0.134276
  features.4.0.block.1.0: coverage shape torch.Size([336]), min=-3.700109, max=1.000000, mean=-0.179640
  features.4.0.block.2.fc1: coverage shape torch.Size([14]), min=-405.150085, max=1.000000, mean=-98.463585
  features.4.0.block.2.fc2: coverage shape torch.Size([336]), min=0.406757, max=1.000000, mean=0.626027
  features.4.0.block.3.0: coverage shape torch.Size([112]), min=-1.022181, max=1.000000, mean=-0.019959
  features.4.1.block.0.0: coverage shape torch.Size([672]), min=-0.874641, max=1.000000, mean=-0.002892
  features.4.1.block.1.0: coverage shape torch.Size([672]), min=-3.110874, max=1.000000, mean=-0.047315
  features.4.1.block.2.fc1: coverage shape torch.Size([28]), min=-0.741219, max=1.000000, mean=0.068599
  features.4.1.block.2.fc2: coverage shape torch.Size([672]), min=-1.213916, max=1.000000, mean=0.017306
  features.4.1.block.3.0: coverage shape torch.Size([112]), min=-0.748125, max=1.000000, mean=0.034320
  features.4.2.block.0.0: coverage shape torch.Size([672]), min=-0.516407, max=1.000000, mean=-0.056279
  features.4.2.block.1.0: coverage shape torch.Size([672]), min=-1.512297, max=1.000000, mean=-0.005659
  features.4.2.block.2.fc1: coverage shape torch.Size([28]), min=-0.854404, max=1.000000, mean=0.140019
  features.4.2.block.2.fc2: coverage shape torch.Size([672]), min=-1.751653, max=1.000000, mean=-0.032486
  features.4.2.block.3.0: coverage shape torch.Size([112]), min=-0.856541, max=1.000000, mean=0.031904
  features.4.3.block.0.0: coverage shape torch.Size([672]), min=-1.581999, max=1.000000, mean=-0.147235
  features.4.3.block.1.0: coverage shape torch.Size([672]), min=-2.498396, max=1.000000, mean=-0.030462
  features.4.3.block.2.fc1: coverage shape torch.Size([28]), min=-1.087753, max=1.000000, mean=0.179023
  features.4.3.block.2.fc2: coverage shape torch.Size([672]), min=-1.203735, max=1.000000, mean=0.030681
  features.4.3.block.3.0: coverage shape torch.Size([112]), min=-0.995720, max=1.000000, mean=0.010493
  features.4.4.block.0.0: coverage shape torch.Size([672]), min=-1.094754, max=1.000000, mean=-0.205416
  features.4.4.block.1.0: coverage shape torch.Size([672]), min=-1.545862, max=1.000000, mean=-0.030146
  features.4.4.block.2.fc1: coverage shape torch.Size([28]), min=-1.298033, max=1.000000, mean=0.025193
  features.4.4.block.2.fc2: coverage shape torch.Size([672]), min=-1.314510, max=1.000000, mean=0.020747
  features.4.4.block.3.0: coverage shape torch.Size([112]), min=-1.037407, max=1.000000, mean=0.048701
  features.4.5.block.0.0: coverage shape torch.Size([672]), min=-1.460253, max=1.000000, mean=-0.258299
  features.4.5.block.1.0: coverage shape torch.Size([672]), min=-0.857458, max=1.000000, mean=-0.011762
  features.4.5.block.2.fc1: coverage shape torch.Size([28]), min=-0.794836, max=1.000000, mean=0.134829
  features.4.5.block.2.fc2: coverage shape torch.Size([672]), min=-1.788704, max=1.000000, mean=-0.036161
  features.4.5.block.3.0: coverage shape torch.Size([112]), min=-0.641261, max=1.000000, mean=0.020727
  features.5.0.block.0.0: coverage shape torch.Size([672]), min=-1.552601, max=1.000000, mean=-0.205217
  features.5.0.block.1.0: coverage shape torch.Size([672]), min=-12.117888, max=1.000000, mean=-0.104229
  features.5.0.block.2.fc1: coverage shape torch.Size([28]), min=-21.090221, max=-1.116417, mean=-4.801590
  features.5.0.block.2.fc2: coverage shape torch.Size([672]), min=-0.134993, max=1.000000, mean=0.583149
  features.5.0.block.3.0: coverage shape torch.Size([160]), min=-0.806054, max=1.000000, mean=0.018639
  features.5.1.block.0.0: coverage shape torch.Size([960]), min=-0.654918, max=1.000000, mean=-0.077291
  features.5.1.block.1.0: coverage shape torch.Size([960]), min=-2.921118, max=1.000000, mean=-0.026426
  features.5.1.block.2.fc1: coverage shape torch.Size([40]), min=-0.573725, max=1.000000, mean=0.189454
  features.5.1.block.2.fc2: coverage shape torch.Size([960]), min=-1.239110, max=1.000000, mean=0.014380
  features.5.1.block.3.0: coverage shape torch.Size([160]), min=-1.286026, max=1.000000, mean=0.017282
  features.5.2.block.0.0: coverage shape torch.Size([960]), min=-1.320048, max=1.000000, mean=-0.310327
  features.5.2.block.1.0: coverage shape torch.Size([960]), min=-1.514089, max=1.000000, mean=-0.016168
  features.5.2.block.2.fc1: coverage shape torch.Size([40]), min=-1.056388, max=1.000000, mean=0.113195
  features.5.2.block.2.fc2: coverage shape torch.Size([960]), min=-1.620865, max=1.000000, mean=-0.008136
  features.5.2.block.3.0: coverage shape torch.Size([160]), min=-1.008095, max=1.000000, mean=-0.029908
  features.5.3.block.0.0: coverage shape torch.Size([960]), min=-1.430980, max=1.000000, mean=-0.325479
  features.5.3.block.1.0: coverage shape torch.Size([960]), min=-1.942754, max=1.000000, mean=-0.014296
  features.5.3.block.2.fc1: coverage shape torch.Size([40]), min=-2.240282, max=1.000000, mean=-0.201326
  features.5.3.block.2.fc2: coverage shape torch.Size([960]), min=-2.051208, max=1.000000, mean=0.107156
  features.5.3.block.3.0: coverage shape torch.Size([160]), min=-1.302238, max=1.000000, mean=-0.009454
  features.5.4.block.0.0: coverage shape torch.Size([960]), min=-1.699906, max=1.000000, mean=-0.462631
  features.5.4.block.1.0: coverage shape torch.Size([960]), min=-2.624244, max=1.000000, mean=-0.015291
  features.5.4.block.2.fc1: coverage shape torch.Size([40]), min=-2.688798, max=1.000000, mean=0.028913
  features.5.4.block.2.fc2: coverage shape torch.Size([960]), min=-2.072161, max=1.000000, mean=0.103082
  features.5.4.block.3.0: coverage shape torch.Size([160]), min=-0.746779, max=1.000000, mean=0.028693
  features.5.5.block.0.0: coverage shape torch.Size([960]), min=-0.816896, max=1.000000, mean=-0.258892
  features.5.5.block.1.0: coverage shape torch.Size([960]), min=-1.613489, max=1.000000, mean=-0.013071
  features.5.5.block.2.fc1: coverage shape torch.Size([40]), min=-0.515922, max=1.000000, mean=0.208825
  features.5.5.block.2.fc2: coverage shape torch.Size([960]), min=-1.979397, max=1.000000, mean=0.050194
  features.5.5.block.3.0: coverage shape torch.Size([160]), min=-0.972983, max=1.000000, mean=-0.014513
  features.6.0.block.0.0: coverage shape torch.Size([960]), min=-1.410161, max=1.000000, mean=-0.397137
  features.6.0.block.1.0: coverage shape torch.Size([960]), min=-49.771370, max=1.000000, mean=-0.593414
  features.6.0.block.2.fc1: coverage shape torch.Size([40]), min=-45.987602, max=-1.128847, mean=-7.257628
  features.6.0.block.2.fc2: coverage shape torch.Size([960]), min=0.215724, max=1.000000, mean=0.466978
  features.6.0.block.3.0: coverage shape torch.Size([272]), min=-0.983420, max=1.000000, mean=-0.003449
  features.6.1.block.0.0: coverage shape torch.Size([1632]), min=-1.285282, max=1.000000, mean=-0.183792
  features.6.1.block.1.0: coverage shape torch.Size([1632]), min=-3.661251, max=1.000000, mean=-0.022077
  features.6.1.block.2.fc1: coverage shape torch.Size([68]), min=-1.623496, max=1.000000, mean=-0.280411
  features.6.1.block.2.fc2: coverage shape torch.Size([1632]), min=-1.250241, max=1.000000, mean=-0.001019
  features.6.1.block.3.0: coverage shape torch.Size([272]), min=-1.350474, max=1.000000, mean=0.010158
  features.6.2.block.0.0: coverage shape torch.Size([1632]), min=-1.246834, max=1.000000, mean=-0.266313
  features.6.2.block.1.0: coverage shape torch.Size([1632]), min=-2.951343, max=1.000000, mean=-0.057677
  features.6.2.block.2.fc1: coverage shape torch.Size([68]), min=-0.913789, max=1.000000, mean=-0.031893
  features.6.2.block.2.fc2: coverage shape torch.Size([1632]), min=-1.269012, max=1.000000, mean=-0.080991
  features.6.2.block.3.0: coverage shape torch.Size([272]), min=-1.830985, max=1.000000, mean=-0.016847
  features.6.3.block.0.0: coverage shape torch.Size([1632]), min=-1.407122, max=1.000000, mean=-0.362289
  features.6.3.block.1.0: coverage shape torch.Size([1632]), min=-2.723267, max=1.000000, mean=-0.040035
  features.6.3.block.2.fc1: coverage shape torch.Size([68]), min=-0.819289, max=1.000000, mean=-0.040049
  features.6.3.block.2.fc2: coverage shape torch.Size([1632]), min=-1.722505, max=1.000000, mean=-0.122264
  features.6.3.block.3.0: coverage shape torch.Size([272]), min=-0.762927, max=1.000000, mean=-0.000090
  features.6.4.block.0.0: coverage shape torch.Size([1632]), min=-1.244738, max=1.000000, mean=-0.426962
  features.6.4.block.1.0: coverage shape torch.Size([1632]), min=-0.998189, max=1.000000, mean=-0.025346
  features.6.4.block.2.fc1: coverage shape torch.Size([68]), min=-1.433634, max=1.000000, mean=-0.115490
  features.6.4.block.2.fc2: coverage shape torch.Size([1632]), min=-1.382396, max=1.000000, mean=-0.110953
  features.6.4.block.3.0: coverage shape torch.Size([272]), min=-0.748581, max=1.000000, mean=0.010188
  features.6.5.block.0.0: coverage shape torch.Size([1632]), min=-1.804128, max=1.000000, mean=-0.682681
  features.6.5.block.1.0: coverage shape torch.Size([1632]), min=-1.309167, max=1.000000, mean=-0.048044
  features.6.5.block.2.fc1: coverage shape torch.Size([68]), min=-1.215616, max=1.000000, mean=-0.064648
  features.6.5.block.2.fc2: coverage shape torch.Size([1632]), min=-0.688705, max=1.000000, mean=-0.063830
  features.6.5.block.3.0: coverage shape torch.Size([272]), min=-0.845952, max=1.000000, mean=0.023821
  features.6.6.block.0.0: coverage shape torch.Size([1632]), min=-2.300307, max=1.000000, mean=-0.895013
  features.6.6.block.1.0: coverage shape torch.Size([1632]), min=-0.742820, max=1.000000, mean=-0.034390
  features.6.6.block.2.fc1: coverage shape torch.Size([68]), min=-0.941717, max=1.000000, mean=-0.062588
  features.6.6.block.2.fc2: coverage shape torch.Size([1632]), min=-1.175937, max=1.000000, mean=-0.123424
  features.6.6.block.3.0: coverage shape torch.Size([272]), min=-1.313111, max=1.000000, mean=-0.020432
  features.6.7.block.0.0: coverage shape torch.Size([1632]), min=-0.785075, max=1.000000, mean=-0.306549
  features.6.7.block.1.0: coverage shape torch.Size([1632]), min=-2.628421, max=1.000000, mean=-0.124350
  features.6.7.block.2.fc1: coverage shape torch.Size([68]), min=-1.283596, max=1.000000, mean=-0.159571
  features.6.7.block.2.fc2: coverage shape torch.Size([1632]), min=-1.204297, max=1.000000, mean=-0.085623
  features.6.7.block.3.0: coverage shape torch.Size([272]), min=-0.855855, max=1.000000, mean=0.021431
  features.7.0.block.0.0: coverage shape torch.Size([1632]), min=-1.945723, max=1.000000, mean=-0.326937
  features.7.0.block.1.0: coverage shape torch.Size([1632]), min=-5.585697, max=1.000000, mean=-0.087601
  features.7.0.block.2.fc1: coverage shape torch.Size([68]), min=-0.818484, max=1.000000, mean=-0.256258
  features.7.0.block.2.fc2: coverage shape torch.Size([1632]), min=-0.624918, max=1.000000, mean=0.167847
  features.7.0.block.3.0: coverage shape torch.Size([448]), min=-1.638181, max=1.000000, mean=-0.012693
  features.7.1.block.0.0: coverage shape torch.Size([2688]), min=-0.990905, max=1.000000, mean=-0.115143
  features.7.1.block.1.0: coverage shape torch.Size([2688]), min=-0.382466, max=1.000000, mean=-0.010182
  features.7.1.block.2.fc1: coverage shape torch.Size([112]), min=-0.896514, max=1.000000, mean=-0.287927
  features.7.1.block.2.fc2: coverage shape torch.Size([2688]), min=-1.175002, max=1.000000, mean=-0.066078
  features.7.1.block.3.0: coverage shape torch.Size([448]), min=-1.832581, max=1.000000, mean=-0.033116
  features.8.0: coverage shape torch.Size([1792]), min=-13.567109, max=-2.395609, mean=-7.435386
  classifier.1: coverage shape torch.Size([101]), min=-14.084210, max=-9.408195, mean=-11.340934

------------------------------------------------------------
Coverage Statistics:
------------------------------------------------------------

features.0.0:
  Channels: 48
  Coverage - Min: -1.370113, Max: 1.000000, Mean: -0.020590
  Zero coverage neurons: 0

features.1.0.block.0.0:
  Channels: 48
  Coverage - Min: -0.464612, Max: 1.000000, Mean: 0.001804
  Zero coverage neurons: 0

features.1.0.block.1.fc1:
  Channels: 12
  Coverage - Min: -0.657171, Max: 1.000000, Mean: 0.227549
  Zero coverage neurons: 0

features.1.0.block.1.fc2:
  Channels: 48
  Coverage - Min: -0.099096, Max: 1.000000, Mean: 0.483725
  Zero coverage neurons: 0

features.1.0.block.2.0:
  Channels: 24
  Coverage - Min: -1.063244, Max: 1.000000, Mean: 0.179302
  Zero coverage neurons: 0

features.1.1.block.0.0:
  Channels: 24
  Coverage - Min: -0.550854, Max: 1.000000, Mean: 0.096047
  Zero coverage neurons: 0

features.1.1.block.1.fc1:
  Channels: 6
  Coverage - Min: -17.864712, Max: -2.436741, Mean: -10.322094
  Zero coverage neurons: 0

features.1.1.block.1.fc2:
  Channels: 24
  Coverage - Min: -0.337256, Max: 1.000000, Mean: 0.623510
  Zero coverage neurons: 0

features.1.1.block.2.0:
  Channels: 24
  Coverage - Min: -3.216683, Max: 1.000000, Mean: -0.480281
  Zero coverage neurons: 0

features.2.0.block.0.0:
  Channels: 144
  Coverage - Min: -0.849921, Max: 1.000000, Mean: 0.014162
  Zero coverage neurons: 0

features.2.0.block.1.0:
  Channels: 144
  Coverage - Min: -1.041770, Max: 1.000000, Mean: 0.050544
  Zero coverage neurons: 0

features.2.0.block.2.fc1:
  Channels: 6
  Coverage - Min: -11.726668, Max: -0.942153, Mean: -5.100576
  Zero coverage neurons: 0

features.2.0.block.2.fc2:
  Channels: 144
  Coverage - Min: 0.025777, Max: 1.000000, Mean: 0.564362
  Zero coverage neurons: 0

features.2.0.block.3.0:
  Channels: 32
  Coverage - Min: -1.046850, Max: 1.000000, Mean: 0.183512
  Zero coverage neurons: 0

features.2.1.block.0.0:
  Channels: 192
  Coverage - Min: -1.554276, Max: 1.000000, Mean: -0.090969
  Zero coverage neurons: 0

features.2.1.block.1.0:
  Channels: 192
  Coverage - Min: -1.125548, Max: 1.000000, Mean: -0.033404
  Zero coverage neurons: 0

features.2.1.block.2.fc1:
  Channels: 8
  Coverage - Min: -0.273431, Max: 1.000000, Mean: 0.038269
  Zero coverage neurons: 0

features.2.1.block.2.fc2:
  Channels: 192
  Coverage - Min: -1.237109, Max: 1.000000, Mean: 0.170330
  Zero coverage neurons: 0

features.2.1.block.3.0:
  Channels: 32
  Coverage - Min: -1.352929, Max: 1.000000, Mean: -0.131863
  Zero coverage neurons: 0

features.2.2.block.0.0:
  Channels: 192
  Coverage - Min: -1.198591, Max: 1.000000, Mean: -0.116988
  Zero coverage neurons: 0

features.2.2.block.1.0:
  Channels: 192
  Coverage - Min: -1.599875, Max: 1.000000, Mean: -0.032859
  Zero coverage neurons: 0

features.2.2.block.2.fc1:
  Channels: 8
  Coverage - Min: -28.277876, Max: -1.265039, Mean: -5.408116
  Zero coverage neurons: 0

features.2.2.block.2.fc2:
  Channels: 192
  Coverage - Min: -0.890730, Max: 1.000000, Mean: 0.241708
  Zero coverage neurons: 0

features.2.2.block.3.0:
  Channels: 32
  Coverage - Min: -0.993440, Max: 1.000000, Mean: 0.081630
  Zero coverage neurons: 0

features.2.3.block.0.0:
  Channels: 192
  Coverage - Min: -1.355004, Max: 1.000000, Mean: -0.178252
  Zero coverage neurons: 0

features.2.3.block.1.0:
  Channels: 192
  Coverage - Min: -1.391695, Max: 1.000000, Mean: -0.036774
  Zero coverage neurons: 0

features.2.3.block.2.fc1:
  Channels: 8
  Coverage - Min: -3.090279, Max: -1.832920, Mean: -2.546887
  Zero coverage neurons: 0

features.2.3.block.2.fc2:
  Channels: 192
  Coverage - Min: -0.564984, Max: 1.000000, Mean: 0.291982
  Zero coverage neurons: 0

features.2.3.block.3.0:
  Channels: 32
  Coverage - Min: -0.826006, Max: 1.000000, Mean: 0.013604
  Zero coverage neurons: 0

features.3.0.block.0.0:
  Channels: 192
  Coverage - Min: -1.444459, Max: 1.000000, Mean: -0.243453
  Zero coverage neurons: 0

features.3.0.block.1.0:
  Channels: 192
  Coverage - Min: -0.822763, Max: 1.000000, Mean: -0.027223
  Zero coverage neurons: 0

features.3.0.block.2.fc1:
  Channels: 8
  Coverage - Min: 0.050040, Max: 1.000000, Mean: 0.361643
  Zero coverage neurons: 0

features.3.0.block.2.fc2:
  Channels: 192
  Coverage - Min: -0.608386, Max: 1.000000, Mean: 0.293847
  Zero coverage neurons: 0

features.3.0.block.3.0:
  Channels: 56
  Coverage - Min: -0.761944, Max: 1.000000, Mean: 0.014581
  Zero coverage neurons: 0

features.3.1.block.0.0:
  Channels: 336
  Coverage - Min: -0.862992, Max: 1.000000, Mean: 0.008858
  Zero coverage neurons: 0

features.3.1.block.1.0:
  Channels: 336
  Coverage - Min: -3.720114, Max: 1.000000, Mean: -0.130605
  Zero coverage neurons: 0

features.3.1.block.2.fc1:
  Channels: 14
  Coverage - Min: -0.043882, Max: 1.000000, Mean: 0.263726
  Zero coverage neurons: 0

features.3.1.block.2.fc2:
  Channels: 336
  Coverage - Min: -1.557634, Max: 1.000000, Mean: 0.161927
  Zero coverage neurons: 0

features.3.1.block.3.0:
  Channels: 56
  Coverage - Min: -1.018814, Max: 1.000000, Mean: 0.042396
  Zero coverage neurons: 0

features.3.2.block.0.0:
  Channels: 336
  Coverage - Min: -1.964223, Max: 1.000000, Mean: -0.221253
  Zero coverage neurons: 0

features.3.2.block.1.0:
  Channels: 336
  Coverage - Min: -1.942446, Max: 1.000000, Mean: -0.073373
  Zero coverage neurons: 0

features.3.2.block.2.fc1:
  Channels: 14
  Coverage - Min: -0.339858, Max: 1.000000, Mean: 0.031272
  Zero coverage neurons: 0

features.3.2.block.2.fc2:
  Channels: 336
  Coverage - Min: -2.530282, Max: 1.000000, Mean: 0.126706
  Zero coverage neurons: 0

features.3.2.block.3.0:
  Channels: 56
  Coverage - Min: -0.715217, Max: 1.000000, Mean: 0.019824
  Zero coverage neurons: 0

features.3.3.block.0.0:
  Channels: 336
  Coverage - Min: -0.982032, Max: 1.000000, Mean: -0.142124
  Zero coverage neurons: 0

features.3.3.block.1.0:
  Channels: 336
  Coverage - Min: -1.701053, Max: 1.000000, Mean: -0.064831
  Zero coverage neurons: 0

features.3.3.block.2.fc1:
  Channels: 14
  Coverage - Min: -0.727502, Max: 1.000000, Mean: -0.017758
  Zero coverage neurons: 0

features.3.3.block.2.fc2:
  Channels: 336
  Coverage - Min: -0.637394, Max: 1.000000, Mean: 0.060623
  Zero coverage neurons: 0

features.3.3.block.3.0:
  Channels: 56
  Coverage - Min: -0.777795, Max: 1.000000, Mean: -0.020132
  Zero coverage neurons: 0

features.4.0.block.0.0:
  Channels: 336
  Coverage - Min: -0.682449, Max: 1.000000, Mean: -0.134276
  Zero coverage neurons: 0

features.4.0.block.1.0:
  Channels: 336
  Coverage - Min: -3.700109, Max: 1.000000, Mean: -0.179640
  Zero coverage neurons: 0

features.4.0.block.2.fc1:
  Channels: 14
  Coverage - Min: -405.150085, Max: 1.000000, Mean: -98.463585
  Zero coverage neurons: 0

features.4.0.block.2.fc2:
  Channels: 336
  Coverage - Min: 0.406757, Max: 1.000000, Mean: 0.626027
  Zero coverage neurons: 0

features.4.0.block.3.0:
  Channels: 112
  Coverage - Min: -1.022181, Max: 1.000000, Mean: -0.019959
  Zero coverage neurons: 0

features.4.1.block.0.0:
  Channels: 672
  Coverage - Min: -0.874641, Max: 1.000000, Mean: -0.002892
  Zero coverage neurons: 0

features.4.1.block.1.0:
  Channels: 672
  Coverage - Min: -3.110874, Max: 1.000000, Mean: -0.047315
  Zero coverage neurons: 0

features.4.1.block.2.fc1:
  Channels: 28
  Coverage - Min: -0.741219, Max: 1.000000, Mean: 0.068599
  Zero coverage neurons: 0

features.4.1.block.2.fc2:
  Channels: 672
  Coverage - Min: -1.213916, Max: 1.000000, Mean: 0.017306
  Zero coverage neurons: 0

features.4.1.block.3.0:
  Channels: 112
  Coverage - Min: -0.748125, Max: 1.000000, Mean: 0.034320
  Zero coverage neurons: 0

features.4.2.block.0.0:
  Channels: 672
  Coverage - Min: -0.516407, Max: 1.000000, Mean: -0.056279
  Zero coverage neurons: 0

features.4.2.block.1.0:
  Channels: 672
  Coverage - Min: -1.512297, Max: 1.000000, Mean: -0.005659
  Zero coverage neurons: 0

features.4.2.block.2.fc1:
  Channels: 28
  Coverage - Min: -0.854404, Max: 1.000000, Mean: 0.140019
  Zero coverage neurons: 0

features.4.2.block.2.fc2:
  Channels: 672
  Coverage - Min: -1.751653, Max: 1.000000, Mean: -0.032486
  Zero coverage neurons: 0

features.4.2.block.3.0:
  Channels: 112
  Coverage - Min: -0.856541, Max: 1.000000, Mean: 0.031904
  Zero coverage neurons: 0

features.4.3.block.0.0:
  Channels: 672
  Coverage - Min: -1.581999, Max: 1.000000, Mean: -0.147235
  Zero coverage neurons: 0

features.4.3.block.1.0:
  Channels: 672
  Coverage - Min: -2.498396, Max: 1.000000, Mean: -0.030462
  Zero coverage neurons: 0

features.4.3.block.2.fc1:
  Channels: 28
  Coverage - Min: -1.087753, Max: 1.000000, Mean: 0.179023
  Zero coverage neurons: 0

features.4.3.block.2.fc2:
  Channels: 672
  Coverage - Min: -1.203735, Max: 1.000000, Mean: 0.030681
  Zero coverage neurons: 0

features.4.3.block.3.0:
  Channels: 112
  Coverage - Min: -0.995720, Max: 1.000000, Mean: 0.010493
  Zero coverage neurons: 0

features.4.4.block.0.0:
  Channels: 672
  Coverage - Min: -1.094754, Max: 1.000000, Mean: -0.205416
  Zero coverage neurons: 0

features.4.4.block.1.0:
  Channels: 672
  Coverage - Min: -1.545862, Max: 1.000000, Mean: -0.030146
  Zero coverage neurons: 0

features.4.4.block.2.fc1:
  Channels: 28
  Coverage - Min: -1.298033, Max: 1.000000, Mean: 0.025193
  Zero coverage neurons: 0

features.4.4.block.2.fc2:
  Channels: 672
  Coverage - Min: -1.314510, Max: 1.000000, Mean: 0.020747
  Zero coverage neurons: 0

features.4.4.block.3.0:
  Channels: 112
  Coverage - Min: -1.037407, Max: 1.000000, Mean: 0.048701
  Zero coverage neurons: 0

features.4.5.block.0.0:
  Channels: 672
  Coverage - Min: -1.460253, Max: 1.000000, Mean: -0.258299
  Zero coverage neurons: 0

features.4.5.block.1.0:
  Channels: 672
  Coverage - Min: -0.857458, Max: 1.000000, Mean: -0.011762
  Zero coverage neurons: 0

features.4.5.block.2.fc1:
  Channels: 28
  Coverage - Min: -0.794836, Max: 1.000000, Mean: 0.134829
  Zero coverage neurons: 0

features.4.5.block.2.fc2:
  Channels: 672
  Coverage - Min: -1.788704, Max: 1.000000, Mean: -0.036161
  Zero coverage neurons: 0

features.4.5.block.3.0:
  Channels: 112
  Coverage - Min: -0.641261, Max: 1.000000, Mean: 0.020727
  Zero coverage neurons: 0

features.5.0.block.0.0:
  Channels: 672
  Coverage - Min: -1.552601, Max: 1.000000, Mean: -0.205217
  Zero coverage neurons: 0

features.5.0.block.1.0:
  Channels: 672
  Coverage - Min: -12.117888, Max: 1.000000, Mean: -0.104229
  Zero coverage neurons: 0

features.5.0.block.2.fc1:
  Channels: 28
  Coverage - Min: -21.090221, Max: -1.116417, Mean: -4.801590
  Zero coverage neurons: 0

features.5.0.block.2.fc2:
  Channels: 672
  Coverage - Min: -0.134993, Max: 1.000000, Mean: 0.583149
  Zero coverage neurons: 0

features.5.0.block.3.0:
  Channels: 160
  Coverage - Min: -0.806054, Max: 1.000000, Mean: 0.018639
  Zero coverage neurons: 0

features.5.1.block.0.0:
  Channels: 960
  Coverage - Min: -0.654918, Max: 1.000000, Mean: -0.077291
  Zero coverage neurons: 0

features.5.1.block.1.0:
  Channels: 960
  Coverage - Min: -2.921118, Max: 1.000000, Mean: -0.026426
  Zero coverage neurons: 0

features.5.1.block.2.fc1:
  Channels: 40
  Coverage - Min: -0.573725, Max: 1.000000, Mean: 0.189454
  Zero coverage neurons: 0

features.5.1.block.2.fc2:
  Channels: 960
  Coverage - Min: -1.239110, Max: 1.000000, Mean: 0.014380
  Zero coverage neurons: 0

features.5.1.block.3.0:
  Channels: 160
  Coverage - Min: -1.286026, Max: 1.000000, Mean: 0.017282
  Zero coverage neurons: 0

features.5.2.block.0.0:
  Channels: 960
  Coverage - Min: -1.320048, Max: 1.000000, Mean: -0.310327
  Zero coverage neurons: 0

features.5.2.block.1.0:
  Channels: 960
  Coverage - Min: -1.514089, Max: 1.000000, Mean: -0.016168
  Zero coverage neurons: 0

features.5.2.block.2.fc1:
  Channels: 40
  Coverage - Min: -1.056388, Max: 1.000000, Mean: 0.113195
  Zero coverage neurons: 0

features.5.2.block.2.fc2:
  Channels: 960
  Coverage - Min: -1.620865, Max: 1.000000, Mean: -0.008136
  Zero coverage neurons: 0

features.5.2.block.3.0:
  Channels: 160
  Coverage - Min: -1.008095, Max: 1.000000, Mean: -0.029908
  Zero coverage neurons: 0

features.5.3.block.0.0:
  Channels: 960
  Coverage - Min: -1.430980, Max: 1.000000, Mean: -0.325479
  Zero coverage neurons: 0

features.5.3.block.1.0:
  Channels: 960
  Coverage - Min: -1.942754, Max: 1.000000, Mean: -0.014296
  Zero coverage neurons: 0

features.5.3.block.2.fc1:
  Channels: 40
  Coverage - Min: -2.240282, Max: 1.000000, Mean: -0.201326
  Zero coverage neurons: 0

features.5.3.block.2.fc2:
  Channels: 960
  Coverage - Min: -2.051208, Max: 1.000000, Mean: 0.107156
  Zero coverage neurons: 0

features.5.3.block.3.0:
  Channels: 160
  Coverage - Min: -1.302238, Max: 1.000000, Mean: -0.009454
  Zero coverage neurons: 0

features.5.4.block.0.0:
  Channels: 960
  Coverage - Min: -1.699906, Max: 1.000000, Mean: -0.462631
  Zero coverage neurons: 0

features.5.4.block.1.0:
  Channels: 960
  Coverage - Min: -2.624244, Max: 1.000000, Mean: -0.015291
  Zero coverage neurons: 0

features.5.4.block.2.fc1:
  Channels: 40
  Coverage - Min: -2.688798, Max: 1.000000, Mean: 0.028913
  Zero coverage neurons: 0

features.5.4.block.2.fc2:
  Channels: 960
  Coverage - Min: -2.072161, Max: 1.000000, Mean: 0.103082
  Zero coverage neurons: 0

features.5.4.block.3.0:
  Channels: 160
  Coverage - Min: -0.746779, Max: 1.000000, Mean: 0.028693
  Zero coverage neurons: 0

features.5.5.block.0.0:
  Channels: 960
  Coverage - Min: -0.816896, Max: 1.000000, Mean: -0.258892
  Zero coverage neurons: 0

features.5.5.block.1.0:
  Channels: 960
  Coverage - Min: -1.613489, Max: 1.000000, Mean: -0.013071
  Zero coverage neurons: 0

features.5.5.block.2.fc1:
  Channels: 40
  Coverage - Min: -0.515922, Max: 1.000000, Mean: 0.208825
  Zero coverage neurons: 0

features.5.5.block.2.fc2:
  Channels: 960
  Coverage - Min: -1.979397, Max: 1.000000, Mean: 0.050194
  Zero coverage neurons: 0

features.5.5.block.3.0:
  Channels: 160
  Coverage - Min: -0.972983, Max: 1.000000, Mean: -0.014513
  Zero coverage neurons: 0

features.6.0.block.0.0:
  Channels: 960
  Coverage - Min: -1.410161, Max: 1.000000, Mean: -0.397137
  Zero coverage neurons: 0

features.6.0.block.1.0:
  Channels: 960
  Coverage - Min: -49.771370, Max: 1.000000, Mean: -0.593414
  Zero coverage neurons: 0

features.6.0.block.2.fc1:
  Channels: 40
  Coverage - Min: -45.987602, Max: -1.128847, Mean: -7.257628
  Zero coverage neurons: 0

features.6.0.block.2.fc2:
  Channels: 960
  Coverage - Min: 0.215724, Max: 1.000000, Mean: 0.466978
  Zero coverage neurons: 0

features.6.0.block.3.0:
  Channels: 272
  Coverage - Min: -0.983420, Max: 1.000000, Mean: -0.003449
  Zero coverage neurons: 0

features.6.1.block.0.0:
  Channels: 1632
  Coverage - Min: -1.285282, Max: 1.000000, Mean: -0.183792
  Zero coverage neurons: 0

features.6.1.block.1.0:
  Channels: 1632
  Coverage - Min: -3.661251, Max: 1.000000, Mean: -0.022077
  Zero coverage neurons: 0

features.6.1.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.623496, Max: 1.000000, Mean: -0.280411
  Zero coverage neurons: 0

features.6.1.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.250241, Max: 1.000000, Mean: -0.001019
  Zero coverage neurons: 0

features.6.1.block.3.0:
  Channels: 272
  Coverage - Min: -1.350474, Max: 1.000000, Mean: 0.010158
  Zero coverage neurons: 0

features.6.2.block.0.0:
  Channels: 1632
  Coverage - Min: -1.246834, Max: 1.000000, Mean: -0.266313
  Zero coverage neurons: 0

features.6.2.block.1.0:
  Channels: 1632
  Coverage - Min: -2.951343, Max: 1.000000, Mean: -0.057677
  Zero coverage neurons: 0

features.6.2.block.2.fc1:
  Channels: 68
  Coverage - Min: -0.913789, Max: 1.000000, Mean: -0.031893
  Zero coverage neurons: 0

features.6.2.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.269012, Max: 1.000000, Mean: -0.080991
  Zero coverage neurons: 0

features.6.2.block.3.0:
  Channels: 272
  Coverage - Min: -1.830985, Max: 1.000000, Mean: -0.016847
  Zero coverage neurons: 0

features.6.3.block.0.0:
  Channels: 1632
  Coverage - Min: -1.407122, Max: 1.000000, Mean: -0.362289
  Zero coverage neurons: 0

features.6.3.block.1.0:
  Channels: 1632
  Coverage - Min: -2.723267, Max: 1.000000, Mean: -0.040035
  Zero coverage neurons: 0

features.6.3.block.2.fc1:
  Channels: 68
  Coverage - Min: -0.819289, Max: 1.000000, Mean: -0.040049
  Zero coverage neurons: 0

features.6.3.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.722505, Max: 1.000000, Mean: -0.122264
  Zero coverage neurons: 0

features.6.3.block.3.0:
  Channels: 272
  Coverage - Min: -0.762927, Max: 1.000000, Mean: -0.000090
  Zero coverage neurons: 0

features.6.4.block.0.0:
  Channels: 1632
  Coverage - Min: -1.244738, Max: 1.000000, Mean: -0.426962
  Zero coverage neurons: 0

features.6.4.block.1.0:
  Channels: 1632
  Coverage - Min: -0.998189, Max: 1.000000, Mean: -0.025346
  Zero coverage neurons: 0

features.6.4.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.433634, Max: 1.000000, Mean: -0.115490
  Zero coverage neurons: 0

features.6.4.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.382396, Max: 1.000000, Mean: -0.110953
  Zero coverage neurons: 0

features.6.4.block.3.0:
  Channels: 272
  Coverage - Min: -0.748581, Max: 1.000000, Mean: 0.010188
  Zero coverage neurons: 0

features.6.5.block.0.0:
  Channels: 1632
  Coverage - Min: -1.804128, Max: 1.000000, Mean: -0.682681
  Zero coverage neurons: 0

features.6.5.block.1.0:
  Channels: 1632
  Coverage - Min: -1.309167, Max: 1.000000, Mean: -0.048044
  Zero coverage neurons: 0

features.6.5.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.215616, Max: 1.000000, Mean: -0.064648
  Zero coverage neurons: 0

features.6.5.block.2.fc2:
  Channels: 1632
  Coverage - Min: -0.688705, Max: 1.000000, Mean: -0.063830
  Zero coverage neurons: 0

features.6.5.block.3.0:
  Channels: 272
  Coverage - Min: -0.845952, Max: 1.000000, Mean: 0.023821
  Zero coverage neurons: 0

features.6.6.block.0.0:
  Channels: 1632
  Coverage - Min: -2.300307, Max: 1.000000, Mean: -0.895013
  Zero coverage neurons: 0

features.6.6.block.1.0:
  Channels: 1632
  Coverage - Min: -0.742820, Max: 1.000000, Mean: -0.034390
  Zero coverage neurons: 0

features.6.6.block.2.fc1:
  Channels: 68
  Coverage - Min: -0.941717, Max: 1.000000, Mean: -0.062588
  Zero coverage neurons: 0

features.6.6.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.175937, Max: 1.000000, Mean: -0.123424
  Zero coverage neurons: 0

features.6.6.block.3.0:
  Channels: 272
  Coverage - Min: -1.313111, Max: 1.000000, Mean: -0.020432
  Zero coverage neurons: 0

features.6.7.block.0.0:
  Channels: 1632
  Coverage - Min: -0.785075, Max: 1.000000, Mean: -0.306549
  Zero coverage neurons: 0

features.6.7.block.1.0:
  Channels: 1632
  Coverage - Min: -2.628421, Max: 1.000000, Mean: -0.124350
  Zero coverage neurons: 0

features.6.7.block.2.fc1:
  Channels: 68
  Coverage - Min: -1.283596, Max: 1.000000, Mean: -0.159571
  Zero coverage neurons: 0

features.6.7.block.2.fc2:
  Channels: 1632
  Coverage - Min: -1.204297, Max: 1.000000, Mean: -0.085623
  Zero coverage neurons: 0

features.6.7.block.3.0:
  Channels: 272
  Coverage - Min: -0.855855, Max: 1.000000, Mean: 0.021431
  Zero coverage neurons: 0

features.7.0.block.0.0:
  Channels: 1632
  Coverage - Min: -1.945723, Max: 1.000000, Mean: -0.326937
  Zero coverage neurons: 0

features.7.0.block.1.0:
  Channels: 1632
  Coverage - Min: -5.585697, Max: 1.000000, Mean: -0.087601
  Zero coverage neurons: 0

features.7.0.block.2.fc1:
  Channels: 68
  Coverage - Min: -0.818484, Max: 1.000000, Mean: -0.256258
  Zero coverage neurons: 0

features.7.0.block.2.fc2:
  Channels: 1632
  Coverage - Min: -0.624918, Max: 1.000000, Mean: 0.167847
  Zero coverage neurons: 0

features.7.0.block.3.0:
  Channels: 448
  Coverage - Min: -1.638181, Max: 1.000000, Mean: -0.012693
  Zero coverage neurons: 0

features.7.1.block.0.0:
  Channels: 2688
  Coverage - Min: -0.990905, Max: 1.000000, Mean: -0.115143
  Zero coverage neurons: 0

features.7.1.block.1.0:
  Channels: 2688
  Coverage - Min: -0.382466, Max: 1.000000, Mean: -0.010182
  Zero coverage neurons: 0

features.7.1.block.2.fc1:
  Channels: 112
  Coverage - Min: -0.896514, Max: 1.000000, Mean: -0.287927
  Zero coverage neurons: 0

features.7.1.block.2.fc2:
  Channels: 2688
  Coverage - Min: -1.175002, Max: 1.000000, Mean: -0.066078
  Zero coverage neurons: 0

features.7.1.block.3.0:
  Channels: 448
  Coverage - Min: -1.832581, Max: 1.000000, Mean: -0.033116
  Zero coverage neurons: 0

features.8.0:
  Channels: 1792
  Coverage - Min: -13.567109, Max: -2.395609, Mean: -7.435386
  Zero coverage neurons: 0

classifier.1:
  Channels: 101
  Coverage - Min: -14.084210, Max: -9.408195, Mean: -11.340934
  Zero coverage neurons: 0
============================================================


Pruning Results:
  Parameters before: 17,729,709
  Parameters after:  12,814,772
  Parameters removed: 4,914,937 (27.72%)

############################################################
Pruning Complete
############################################################
Initial parameters:  17,729,709
Final parameters:    12,814,772
Total removed:       4,914,937 (27.72%)
Target pruning ratio: 10.00%
✓ Neuron Coverage pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 1.18%                                                                                                
✓ Model Size: 49.32 MB
✓ Average Inference Time: 5.4122 ms
✓ FLOPs: 3.67 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|█████████████████████████████████████████████████| 313/313 [01:46<00:00,  2.95it/s, loss=2.1807, acc=45.78%] 
Epoch 1/10 - Train Loss: 2.1807, Train Acc: 45.78%, Test Acc: 74.29%                                                          
Epoch 2/10: 100%|█████████████████████████████████████████████████| 313/313 [01:45<00:00,  2.97it/s, loss=1.6985, acc=56.98%] 
Epoch 2/10 - Train Loss: 1.6985, Train Acc: 56.98%, Test Acc: 77.30%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_NC_epoch2.pth
Epoch 3/10: 100%|█████████████████████████████████████████████████| 313/313 [01:45<00:00,  2.97it/s, loss=1.5166, acc=60.86%]
Epoch 3/10 - Train Loss: 1.5166, Train Acc: 60.86%, Test Acc: 78.00%                                                          
Epoch 4/10: 100%|█████████████████████████████████████████████████| 313/313 [01:45<00:00,  2.97it/s, loss=1.3997, acc=64.48%] 
Epoch 4/10 - Train Loss: 1.3997, Train Acc: 64.48%, Test Acc: 79.10%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_NC_epoch4.pth
Epoch 5/10: 100%|█████████████████████████████████████████████████| 313/313 [01:45<00:00,  2.96it/s, loss=1.2973, acc=66.78%]
Epoch 5/10 - Train Loss: 1.2973, Train Acc: 66.78%, Test Acc: 79.87%                                                          
Epoch 6/10: 100%|█████████████████████████████████████████████████| 313/313 [01:43<00:00,  3.02it/s, loss=1.2153, acc=68.84%] 
Epoch 6/10 - Train Loss: 1.2153, Train Acc: 68.84%, Test Acc: 80.00%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_NC_epoch6.pth
Epoch 7/10: 100%|█████████████████████████████████████████████████| 313/313 [01:43<00:00,  3.02it/s, loss=1.1852, acc=68.84%]
Epoch 7/10 - Train Loss: 1.1852, Train Acc: 68.84%, Test Acc: 80.23%                                                          
Epoch 8/10: 100%|█████████████████████████████████████████████████| 313/313 [01:43<00:00,  3.02it/s, loss=1.1496, acc=69.48%] 
Epoch 8/10 - Train Loss: 1.1496, Train Acc: 69.48%, Test Acc: 80.29%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_NC_epoch8.pth
Epoch 9/10: 100%|█████████████████████████████████████████████████| 313/313 [01:43<00:00,  3.03it/s, loss=1.1160, acc=71.32%]
Epoch 9/10 - Train Loss: 1.1160, Train Acc: 71.32%, Test Acc: 80.40%                                                          
Epoch 10/10: 100%|████████████████████████████████████████████████| 313/313 [01:43<00:00,  3.02it/s, loss=1.1286, acc=70.68%] 
Epoch 10/10 - Train Loss: 1.1286, Train Acc: 70.68%, Test Acc: 80.63%
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_NC_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_NC_best.pth
  Best Accuracy: 80.63%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 80.63%                                                                                                
✓ Model Size: 49.32 MB
✓ Average Inference Time: 5.3258 ms
✓ FLOPs: 3.67 GFLOPs

================================================================================
COMPARISON: NEURON COVERAGE PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         89.38   |          1.18   |    80.63   | -8.74                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         68.11   |         49.32   |    49.32   | -18.79 (-27.6%)             |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.5875 |          5.4122 |     5.3258 | +4.7383                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.61   |          3.67   |     3.67   | -0.94 (-20.4%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS9_EfficientNetB4_Food101\TS9_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================



******************************



================================================================================
TEST SCENARIO TS9: WANDA PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FT_best.pth
  - Epoch: 10
  - Test Accuracy: 89.38%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 89.38%                                                                                             
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.5993 ms
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
  features.0.0: coverage shape torch.Size([48]), min=0.000011, max=9.562968, mean=1.174722
  features.1.0.block.0.0: coverage shape torch.Size([48]), min=0.000001, max=72.190231, mean=6.292434
  features.1.0.block.1.fc1: coverage shape torch.Size([12]), min=1.658148, max=17.596046, mean=6.297251
  features.1.0.block.1.fc2: coverage shape torch.Size([48]), min=0.057518, max=11.286330, mean=5.514259
  features.1.0.block.2.0: coverage shape torch.Size([24]), min=8.865083, max=201.871353, mean=71.299019
  features.1.1.block.0.0: coverage shape torch.Size([24]), min=0.005865, max=18.146849, mean=3.735430
  features.1.1.block.1.fc1: coverage shape torch.Size([6]), min=2.440449, max=17.900244, mean=10.329642
  features.1.1.block.1.fc2: coverage shape torch.Size([24]), min=1.524869, max=4.521369, mean=2.946193
  features.1.1.block.2.0: coverage shape torch.Size([24]), min=0.978210, max=103.392441, mean=23.946381
  features.2.0.block.0.0: coverage shape torch.Size([144]), min=0.097046, max=83.683838, mean=15.568424
  features.2.0.block.1.0: coverage shape torch.Size([144]), min=0.000489, max=69.421967, mean=14.393116
  features.2.0.block.2.fc1: coverage shape torch.Size([6]), min=0.923778, max=11.713142, mean=5.096838
  features.2.0.block.2.fc2: coverage shape torch.Size([144]), min=0.138168, max=5.362625, mean=3.026444
  features.2.0.block.3.0: coverage shape torch.Size([32]), min=6.711240, max=266.541595, mean=104.210510
  features.2.1.block.0.0: coverage shape torch.Size([192]), min=0.123149, max=152.537766, mean=23.541395
  features.2.1.block.1.0: coverage shape torch.Size([192]), min=0.000253, max=20.971653, mean=1.384128
  features.2.1.block.2.fc1: coverage shape torch.Size([8]), min=1.751311, max=38.438187, mean=10.304512
  features.2.1.block.2.fc2: coverage shape torch.Size([192]), min=0.232381, max=34.826874, mean=6.397720
  features.2.1.block.3.0: coverage shape torch.Size([32]), min=1.782971, max=129.210251, mean=45.848495
  features.2.2.block.0.0: coverage shape torch.Size([192]), min=0.004084, max=80.429131, mean=19.128473
  features.2.2.block.1.0: coverage shape torch.Size([192]), min=0.000330, max=26.228838, mean=1.786126
  features.2.2.block.2.fc1: coverage shape torch.Size([8]), min=1.265855, max=28.287111, mean=5.407845
  features.2.2.block.2.fc2: coverage shape torch.Size([192]), min=0.106843, max=12.216711, mean=3.286314
  features.2.2.block.3.0: coverage shape torch.Size([32]), min=2.017129, max=51.710049, mean=18.482124
  features.2.3.block.0.0: coverage shape torch.Size([192]), min=0.217518, max=101.588562, mean=25.048094
  features.2.3.block.1.0: coverage shape torch.Size([192]), min=0.000103, max=14.748684, mean=1.478459
  features.2.3.block.2.fc1: coverage shape torch.Size([8]), min=1.867512, max=3.059613, mean=2.539073
  features.2.3.block.2.fc2: coverage shape torch.Size([192]), min=0.155259, max=5.062204, mean=1.626324
  features.2.3.block.3.0: coverage shape torch.Size([32]), min=0.257191, max=36.294430, mean=12.816447
  features.3.0.block.0.0: coverage shape torch.Size([192]), min=0.079393, max=151.153976, mean=39.178329
  features.3.0.block.1.0: coverage shape torch.Size([192]), min=0.000061, max=65.077782, mean=4.326277
  features.3.0.block.2.fc1: coverage shape torch.Size([8]), min=2.382142, max=47.477085, mean=17.168377
  features.3.0.block.2.fc2: coverage shape torch.Size([192]), min=0.054550, max=19.839218, mean=6.136811
  features.3.0.block.3.0: coverage shape torch.Size([56]), min=0.140417, max=243.032181, mean=63.562832
  features.3.1.block.0.0: coverage shape torch.Size([336]), min=0.004189, max=61.975510, mean=8.337664
  features.3.1.block.1.0: coverage shape torch.Size([336]), min=0.000720, max=28.964994, mean=1.633546
  features.3.1.block.2.fc1: coverage shape torch.Size([14]), min=0.832598, max=41.513302, mean=11.697281
  features.3.1.block.2.fc2: coverage shape torch.Size([336]), min=0.423905, max=88.621803, mean=12.737144
  features.3.1.block.3.0: coverage shape torch.Size([56]), min=0.275031, max=26.756033, mean=10.841260
  features.3.2.block.0.0: coverage shape torch.Size([336]), min=0.037167, max=72.581276, mean=13.047283
  features.3.2.block.1.0: coverage shape torch.Size([336]), min=0.000129, max=23.149195, mean=1.548062
  features.3.2.block.2.fc1: coverage shape torch.Size([14]), min=1.079765, max=50.403011, mean=8.372694
  features.3.2.block.2.fc2: coverage shape torch.Size([336]), min=0.737806, max=113.278328, mean=9.114061
  features.3.2.block.3.0: coverage shape torch.Size([56]), min=0.318775, max=27.192673, mean=7.251514
  features.3.3.block.0.0: coverage shape torch.Size([336]), min=0.009027, max=80.866508, mean=19.707226
  features.3.3.block.1.0: coverage shape torch.Size([336]), min=0.000310, max=21.183348, mean=1.650308
  features.3.3.block.2.fc1: coverage shape torch.Size([14]), min=0.311760, max=29.939253, mean=11.286435
  features.3.3.block.2.fc2: coverage shape torch.Size([336]), min=0.924510, max=97.780823, mean=9.870144
  features.3.3.block.3.0: coverage shape torch.Size([56]), min=0.277444, max=10.825496, mean=3.786280
  features.4.0.block.0.0: coverage shape torch.Size([336]), min=0.059058, max=116.772308, mean=26.209223
  features.4.0.block.1.0: coverage shape torch.Size([336]), min=0.000061, max=43.230202, mean=2.420198
  features.4.0.block.2.fc1: coverage shape torch.Size([14]), min=0.063505, max=27.177996, mean=6.617860
  features.4.0.block.2.fc2: coverage shape torch.Size([336]), min=1.991435, max=4.895815, mean=3.064958
  features.4.0.block.3.0: coverage shape torch.Size([112]), min=1.188536, max=139.375839, mean=39.974586
  features.4.1.block.0.0: coverage shape torch.Size([672]), min=0.004093, max=80.252502, mean=9.084398
  features.4.1.block.1.0: coverage shape torch.Size([672]), min=0.000097, max=17.214123, mean=0.778276
  features.4.1.block.2.fc1: coverage shape torch.Size([28]), min=0.266145, max=47.519131, mean=15.584308
  features.4.1.block.2.fc2: coverage shape torch.Size([672]), min=0.202033, max=210.086273, mean=21.920538
  features.4.1.block.3.0: coverage shape torch.Size([112]), min=0.035745, max=9.787390, mean=2.453908
  features.4.2.block.0.0: coverage shape torch.Size([672]), min=0.009788, max=53.985374, mean=6.413076
  features.4.2.block.1.0: coverage shape torch.Size([672]), min=0.000006, max=9.589431, mean=0.889016
  features.4.2.block.2.fc1: coverage shape torch.Size([28]), min=0.083284, max=1.686934, mean=0.695973
  features.4.2.block.2.fc2: coverage shape torch.Size([672]), min=0.016091, max=5.377132, mean=1.248544
  features.4.2.block.3.0: coverage shape torch.Size([112]), min=0.005502, max=1.393682, mean=0.319203
  features.4.3.block.0.0: coverage shape torch.Size([672]), min=0.009169, max=48.953232, mean=8.751328
  features.4.3.block.1.0: coverage shape torch.Size([672]), min=0.000097, max=13.015607, mean=0.473577
  features.4.3.block.2.fc1: coverage shape torch.Size([28]), min=0.177856, max=37.282669, mean=10.785552
  features.4.3.block.2.fc2: coverage shape torch.Size([672]), min=0.018644, max=169.989212, mean=16.129740
  features.4.3.block.3.0: coverage shape torch.Size([112]), min=0.003009, max=5.335489, mean=1.918490
  features.4.4.block.0.0: coverage shape torch.Size([672]), min=0.053486, max=41.394619, mean=13.304329
  features.4.4.block.1.0: coverage shape torch.Size([672]), min=0.000382, max=7.965086, mean=0.653993
  features.4.4.block.2.fc1: coverage shape torch.Size([28]), min=0.233793, max=12.027225, mean=4.916203
  features.4.4.block.2.fc2: coverage shape torch.Size([672]), min=0.022347, max=32.376938, mean=6.425279
  features.4.4.block.3.0: coverage shape torch.Size([112]), min=0.001429, max=2.509109, mean=0.716680
  features.4.5.block.0.0: coverage shape torch.Size([672]), min=0.045414, max=58.000866, mean=15.150898
  features.4.5.block.1.0: coverage shape torch.Size([672]), min=0.000276, max=12.172184, mean=0.876324
  features.4.5.block.2.fc1: coverage shape torch.Size([28]), min=0.122321, max=3.834189, mean=1.475245
  features.4.5.block.2.fc2: coverage shape torch.Size([672]), min=0.008529, max=9.408781, mean=2.139942
  features.4.5.block.3.0: coverage shape torch.Size([112]), min=0.001299, max=1.300971, mean=0.319702
  features.5.0.block.0.0: coverage shape torch.Size([672]), min=0.007012, max=73.086800, mean=15.105392
  features.5.0.block.1.0: coverage shape torch.Size([672]), min=0.000295, max=140.422806, mean=1.706063
  features.5.0.block.2.fc1: coverage shape torch.Size([28]), min=1.122027, max=21.083286, mean=4.792241
  features.5.0.block.2.fc2: coverage shape torch.Size([672]), min=0.542539, max=4.286863, mean=2.501563
  features.5.0.block.3.0: coverage shape torch.Size([160]), min=0.289192, max=105.554962, mean=24.539452
  features.5.1.block.0.0: coverage shape torch.Size([960]), min=0.000650, max=20.010799, mean=2.652662
  features.5.1.block.1.0: coverage shape torch.Size([960]), min=0.000056, max=15.148857, mean=0.763596
  features.5.1.block.2.fc1: coverage shape torch.Size([40]), min=0.008404, max=3.515030, mean=0.950706
  features.5.1.block.2.fc2: coverage shape torch.Size([960]), min=0.002820, max=6.645010, mean=1.556745
  features.5.1.block.3.0: coverage shape torch.Size([160]), min=0.000213, max=1.467956, mean=0.397046
  features.5.2.block.0.0: coverage shape torch.Size([960]), min=0.002534, max=19.678261, mean=5.343886
  features.5.2.block.1.0: coverage shape torch.Size([960]), min=0.000061, max=14.027079, mean=0.609557
  features.5.2.block.2.fc1: coverage shape torch.Size([40]), min=0.038071, max=2.964217, mean=1.154813
  features.5.2.block.2.fc2: coverage shape torch.Size([960]), min=0.005780, max=5.761446, mean=1.371718
  features.5.2.block.3.0: coverage shape torch.Size([160]), min=0.004559, max=1.287904, mean=0.305629
  features.5.3.block.0.0: coverage shape torch.Size([960]), min=0.012498, max=34.498901, mean=8.668063
  features.5.3.block.1.0: coverage shape torch.Size([960]), min=0.000422, max=14.375252, mean=0.643051
  features.5.3.block.2.fc1: coverage shape torch.Size([40]), min=0.000685, max=6.363124, mean=1.946756
  features.5.3.block.2.fc2: coverage shape torch.Size([960]), min=0.001156, max=11.721559, mean=2.079188
  features.5.3.block.3.0: coverage shape torch.Size([160]), min=0.001566, max=1.658578, mean=0.414970
  features.5.4.block.0.0: coverage shape torch.Size([960]), min=0.021133, max=42.729176, mean=12.758574
  features.5.4.block.1.0: coverage shape torch.Size([960]), min=0.000102, max=27.023483, mean=0.654262
  features.5.4.block.2.fc1: coverage shape torch.Size([40]), min=0.020895, max=5.503302, mean=0.946148
  features.5.4.block.2.fc2: coverage shape torch.Size([960]), min=0.000987, max=6.396408, mean=1.419140
  features.5.4.block.3.0: coverage shape torch.Size([160]), min=0.000038, max=0.973633, mean=0.285486
  features.5.5.block.0.0: coverage shape torch.Size([960]), min=0.039469, max=64.498085, mean=18.124432
  features.5.5.block.1.0: coverage shape torch.Size([960]), min=0.000368, max=13.944245, mean=0.570324
  features.5.5.block.2.fc1: coverage shape torch.Size([40]), min=0.042480, max=3.260714, mean=0.850897
  features.5.5.block.2.fc2: coverage shape torch.Size([960]), min=0.000601, max=6.808885, mean=1.082900
  features.5.5.block.3.0: coverage shape torch.Size([160]), min=0.001572, max=1.018314, mean=0.339469
  features.6.0.block.0.0: coverage shape torch.Size([960]), min=0.030957, max=68.985535, mean=21.203688
  features.6.0.block.1.0: coverage shape torch.Size([960]), min=0.000152, max=130.277283, mean=1.730889
  features.6.0.block.2.fc1: coverage shape torch.Size([40]), min=1.118063, max=45.969154, mean=7.257420
  features.6.0.block.2.fc2: coverage shape torch.Size([960]), min=1.067424, max=4.946344, mean=2.309893
  features.6.0.block.3.0: coverage shape torch.Size([272]), min=0.026840, max=109.170822, mean=29.655418
  features.6.1.block.0.0: coverage shape torch.Size([1632]), min=0.000918, max=14.086412, mean=2.678354
  features.6.1.block.1.0: coverage shape torch.Size([1632]), min=0.000079, max=28.540703, mean=0.375044
  features.6.1.block.2.fc1: coverage shape torch.Size([68]), min=0.083609, max=4.366369, mean=1.241827
  features.6.1.block.2.fc2: coverage shape torch.Size([1632]), min=0.000592, max=6.156479, mean=1.326324
  features.6.1.block.3.0: coverage shape torch.Size([272]), min=0.001434, max=3.744389, mean=0.485161
  features.6.2.block.0.0: coverage shape torch.Size([1632]), min=0.030594, max=17.505016, mean=4.384604
  features.6.2.block.1.0: coverage shape torch.Size([1632]), min=0.000044, max=9.528646, mean=0.320457
  features.6.2.block.2.fc1: coverage shape torch.Size([68]), min=0.024379, max=3.258947, mean=0.826033
  features.6.2.block.2.fc2: coverage shape torch.Size([1632]), min=0.000908, max=5.849295, mean=1.126227
  features.6.2.block.3.0: coverage shape torch.Size([272]), min=0.000031, max=3.070688, mean=0.355721
  features.6.3.block.0.0: coverage shape torch.Size([1632]), min=0.002919, max=31.585915, mean=8.893963
  features.6.3.block.1.0: coverage shape torch.Size([1632]), min=0.000007, max=12.107544, mean=0.305763
  features.6.3.block.2.fc1: coverage shape torch.Size([68]), min=0.021548, max=3.244771, mean=0.935487
  features.6.3.block.2.fc2: coverage shape torch.Size([1632]), min=0.001166, max=6.708079, mean=1.291871
  features.6.3.block.3.0: coverage shape torch.Size([272]), min=0.000627, max=1.806907, mean=0.358159
  features.6.4.block.0.0: coverage shape torch.Size([1632]), min=0.014290, max=38.042530, mean=13.735968
  features.6.4.block.1.0: coverage shape torch.Size([1632]), min=0.000413, max=7.927786, mean=0.304457
  features.6.4.block.2.fc1: coverage shape torch.Size([68]), min=0.008204, max=3.330363, mean=0.970715
  features.6.4.block.2.fc2: coverage shape torch.Size([1632]), min=0.000877, max=6.184988, mean=1.192212
  features.6.4.block.3.0: coverage shape torch.Size([272]), min=0.001400, max=1.616360, mean=0.285206
  features.6.5.block.0.0: coverage shape torch.Size([1632]), min=0.005120, max=50.424210, mean=20.058403
  features.6.5.block.1.0: coverage shape torch.Size([1632]), min=0.000061, max=6.155792, mean=0.314514
  features.6.5.block.2.fc1: coverage shape torch.Size([68]), min=0.016619, max=2.776173, mean=0.839603
  features.6.5.block.2.fc2: coverage shape torch.Size([1632]), min=0.002460, max=6.702063, mean=1.096104
  features.6.5.block.3.0: coverage shape torch.Size([272]), min=0.000089, max=1.069080, mean=0.269906
  features.6.6.block.0.0: coverage shape torch.Size([1632]), min=0.135037, max=65.614319, mean=26.663113
  features.6.6.block.1.0: coverage shape torch.Size([1632]), min=0.000635, max=6.783964, mean=0.314609
  features.6.6.block.2.fc1: coverage shape torch.Size([68]), min=0.000578, max=2.902066, mean=0.824757
  features.6.6.block.2.fc2: coverage shape torch.Size([1632]), min=0.001150, max=5.080026, mean=1.146363
  features.6.6.block.3.0: coverage shape torch.Size([272]), min=0.001888, max=1.071183, mean=0.281261
  features.6.7.block.0.0: coverage shape torch.Size([1632]), min=0.033040, max=103.394226, mean=33.210102
  features.6.7.block.1.0: coverage shape torch.Size([1632]), min=0.000144, max=5.364334, mean=0.320570
  features.6.7.block.2.fc1: coverage shape torch.Size([68]), min=0.003284, max=2.884629, mean=0.836860
  features.6.7.block.2.fc2: coverage shape torch.Size([1632]), min=0.000112, max=5.418056, mean=1.012448
  features.6.7.block.3.0: coverage shape torch.Size([272]), min=0.002939, max=1.623473, mean=0.310752
  features.7.0.block.0.0: coverage shape torch.Size([1632]), min=0.016749, max=191.471741, mean=46.656960
  features.7.0.block.1.0: coverage shape torch.Size([1632]), min=0.000000, max=94.969307, mean=1.989396
  features.7.0.block.2.fc1: coverage shape torch.Size([68]), min=0.218438, max=7.339898, mean=2.632897
  features.7.0.block.2.fc2: coverage shape torch.Size([1632]), min=0.004671, max=15.875252, mean=2.925092
  features.7.0.block.3.0: coverage shape torch.Size([448]), min=0.000137, max=8.692631, mean=0.675491
  features.7.1.block.0.0: coverage shape torch.Size([2688]), min=0.000363, max=4.192782, mean=0.792369
  features.7.1.block.1.0: coverage shape torch.Size([2688]), min=0.000000, max=21.017275, mean=0.645600
  features.7.1.block.2.fc1: coverage shape torch.Size([112]), min=0.035696, max=4.641036, mean=1.570205
  features.7.1.block.2.fc2: coverage shape torch.Size([2688]), min=0.000027, max=12.298304, mean=1.938009
  features.7.1.block.3.0: coverage shape torch.Size([448]), min=0.000085, max=2.487521, mean=0.274690
  features.8.0: coverage shape torch.Size([1792]), min=2.676399, max=13.694213, mean=7.448977
  classifier.1: coverage shape torch.Size([101]), min=9.381924, max=14.200382, mean=11.377246

------------------------------------------------------------
Activation Statistics:
------------------------------------------------------------

features.0.0:
  Channels: 48
  Activation - Min: 0.000011, Max: 9.562968, Mean: 1.174722

features.1.0.block.0.0:
  Channels: 48
  Activation - Min: 0.000001, Max: 72.190231, Mean: 6.292434

features.1.0.block.1.fc1:
  Channels: 12
  Activation - Min: 1.658148, Max: 17.596046, Mean: 6.297251

features.1.0.block.1.fc2:
  Channels: 48
  Activation - Min: 0.057518, Max: 11.286330, Mean: 5.514259

features.1.0.block.2.0:
  Channels: 24
  Activation - Min: 8.865083, Max: 201.871353, Mean: 71.299019

features.1.1.block.0.0:
  Channels: 24
  Activation - Min: 0.005865, Max: 18.146849, Mean: 3.735430

features.1.1.block.1.fc1:
  Channels: 6
  Activation - Min: 2.440449, Max: 17.900244, Mean: 10.329642

features.1.1.block.1.fc2:
  Channels: 24
  Activation - Min: 1.524869, Max: 4.521369, Mean: 2.946193

features.1.1.block.2.0:
  Channels: 24
  Activation - Min: 0.978210, Max: 103.392441, Mean: 23.946381

features.2.0.block.0.0:
  Channels: 144
  Activation - Min: 0.097046, Max: 83.683838, Mean: 15.568424

features.2.0.block.1.0:
  Channels: 144
  Activation - Min: 0.000489, Max: 69.421967, Mean: 14.393116

features.2.0.block.2.fc1:
  Channels: 6
  Activation - Min: 0.923778, Max: 11.713142, Mean: 5.096838

features.2.0.block.2.fc2:
  Channels: 144
  Activation - Min: 0.138168, Max: 5.362625, Mean: 3.026444

features.2.0.block.3.0:
  Channels: 32
  Activation - Min: 6.711240, Max: 266.541595, Mean: 104.210510

features.2.1.block.0.0:
  Channels: 192
  Activation - Min: 0.123149, Max: 152.537766, Mean: 23.541395

features.2.1.block.1.0:
  Channels: 192
  Activation - Min: 0.000253, Max: 20.971653, Mean: 1.384128

features.2.1.block.2.fc1:
  Channels: 8
  Activation - Min: 1.751311, Max: 38.438187, Mean: 10.304512

features.2.1.block.2.fc2:
  Channels: 192
  Activation - Min: 0.232381, Max: 34.826874, Mean: 6.397720

features.2.1.block.3.0:
  Channels: 32
  Activation - Min: 1.782971, Max: 129.210251, Mean: 45.848495

features.2.2.block.0.0:
  Channels: 192
  Activation - Min: 0.004084, Max: 80.429131, Mean: 19.128473

features.2.2.block.1.0:
  Channels: 192
  Activation - Min: 0.000330, Max: 26.228838, Mean: 1.786126

features.2.2.block.2.fc1:
  Channels: 8
  Activation - Min: 1.265855, Max: 28.287111, Mean: 5.407845

features.2.2.block.2.fc2:
  Channels: 192
  Activation - Min: 0.106843, Max: 12.216711, Mean: 3.286314

features.2.2.block.3.0:
  Channels: 32
  Activation - Min: 2.017129, Max: 51.710049, Mean: 18.482124

features.2.3.block.0.0:
  Channels: 192
  Activation - Min: 0.217518, Max: 101.588562, Mean: 25.048094

features.2.3.block.1.0:
  Channels: 192
  Activation - Min: 0.000103, Max: 14.748684, Mean: 1.478459

features.2.3.block.2.fc1:
  Channels: 8
  Activation - Min: 1.867512, Max: 3.059613, Mean: 2.539073

features.2.3.block.2.fc2:
  Channels: 192
  Activation - Min: 0.155259, Max: 5.062204, Mean: 1.626324

features.2.3.block.3.0:
  Channels: 32
  Activation - Min: 0.257191, Max: 36.294430, Mean: 12.816447

features.3.0.block.0.0:
  Channels: 192
  Activation - Min: 0.079393, Max: 151.153976, Mean: 39.178329

features.3.0.block.1.0:
  Channels: 192
  Activation - Min: 0.000061, Max: 65.077782, Mean: 4.326277

features.3.0.block.2.fc1:
  Channels: 8
  Activation - Min: 2.382142, Max: 47.477085, Mean: 17.168377

features.3.0.block.2.fc2:
  Channels: 192
  Activation - Min: 0.054550, Max: 19.839218, Mean: 6.136811

features.3.0.block.3.0:
  Channels: 56
  Activation - Min: 0.140417, Max: 243.032181, Mean: 63.562832

features.3.1.block.0.0:
  Channels: 336
  Activation - Min: 0.004189, Max: 61.975510, Mean: 8.337664

features.3.1.block.1.0:
  Channels: 336
  Activation - Min: 0.000720, Max: 28.964994, Mean: 1.633546

features.3.1.block.2.fc1:
  Channels: 14
  Activation - Min: 0.832598, Max: 41.513302, Mean: 11.697281

features.3.1.block.2.fc2:
  Channels: 336
  Activation - Min: 0.423905, Max: 88.621803, Mean: 12.737144

features.3.1.block.3.0:
  Channels: 56
  Activation - Min: 0.275031, Max: 26.756033, Mean: 10.841260

features.3.2.block.0.0:
  Channels: 336
  Activation - Min: 0.037167, Max: 72.581276, Mean: 13.047283

features.3.2.block.1.0:
  Channels: 336
  Activation - Min: 0.000129, Max: 23.149195, Mean: 1.548062

features.3.2.block.2.fc1:
  Channels: 14
  Activation - Min: 1.079765, Max: 50.403011, Mean: 8.372694

features.3.2.block.2.fc2:
  Channels: 336
  Activation - Min: 0.737806, Max: 113.278328, Mean: 9.114061

features.3.2.block.3.0:
  Channels: 56
  Activation - Min: 0.318775, Max: 27.192673, Mean: 7.251514

features.3.3.block.0.0:
  Channels: 336
  Activation - Min: 0.009027, Max: 80.866508, Mean: 19.707226

features.3.3.block.1.0:
  Channels: 336
  Activation - Min: 0.000310, Max: 21.183348, Mean: 1.650308

features.3.3.block.2.fc1:
  Channels: 14
  Activation - Min: 0.311760, Max: 29.939253, Mean: 11.286435

features.3.3.block.2.fc2:
  Channels: 336
  Activation - Min: 0.924510, Max: 97.780823, Mean: 9.870144

features.3.3.block.3.0:
  Channels: 56
  Activation - Min: 0.277444, Max: 10.825496, Mean: 3.786280

features.4.0.block.0.0:
  Channels: 336
  Activation - Min: 0.059058, Max: 116.772308, Mean: 26.209223

features.4.0.block.1.0:
  Channels: 336
  Activation - Min: 0.000061, Max: 43.230202, Mean: 2.420198

features.4.0.block.2.fc1:
  Channels: 14
  Activation - Min: 0.063505, Max: 27.177996, Mean: 6.617860

features.4.0.block.2.fc2:
  Channels: 336
  Activation - Min: 1.991435, Max: 4.895815, Mean: 3.064958

features.4.0.block.3.0:
  Channels: 112
  Activation - Min: 1.188536, Max: 139.375839, Mean: 39.974586

features.4.1.block.0.0:
  Channels: 672
  Activation - Min: 0.004093, Max: 80.252502, Mean: 9.084398

features.4.1.block.1.0:
  Channels: 672
  Activation - Min: 0.000097, Max: 17.214123, Mean: 0.778276

features.4.1.block.2.fc1:
  Channels: 28
  Activation - Min: 0.266145, Max: 47.519131, Mean: 15.584308

features.4.1.block.2.fc2:
  Channels: 672
  Activation - Min: 0.202033, Max: 210.086273, Mean: 21.920538

features.4.1.block.3.0:
  Channels: 112
  Activation - Min: 0.035745, Max: 9.787390, Mean: 2.453908

features.4.2.block.0.0:
  Channels: 672
  Activation - Min: 0.009788, Max: 53.985374, Mean: 6.413076

features.4.2.block.1.0:
  Channels: 672
  Activation - Min: 0.000006, Max: 9.589431, Mean: 0.889016

features.4.2.block.2.fc1:
  Channels: 28
  Activation - Min: 0.083284, Max: 1.686934, Mean: 0.695973

features.4.2.block.2.fc2:
  Channels: 672
  Activation - Min: 0.016091, Max: 5.377132, Mean: 1.248544

features.4.2.block.3.0:
  Channels: 112
  Activation - Min: 0.005502, Max: 1.393682, Mean: 0.319203

features.4.3.block.0.0:
  Channels: 672
  Activation - Min: 0.009169, Max: 48.953232, Mean: 8.751328

features.4.3.block.1.0:
  Channels: 672
  Activation - Min: 0.000097, Max: 13.015607, Mean: 0.473577

features.4.3.block.2.fc1:
  Channels: 28
  Activation - Min: 0.177856, Max: 37.282669, Mean: 10.785552

features.4.3.block.2.fc2:
  Channels: 672
  Activation - Min: 0.018644, Max: 169.989212, Mean: 16.129740

features.4.3.block.3.0:
  Channels: 112
  Activation - Min: 0.003009, Max: 5.335489, Mean: 1.918490

features.4.4.block.0.0:
  Channels: 672
  Activation - Min: 0.053486, Max: 41.394619, Mean: 13.304329

features.4.4.block.1.0:
  Channels: 672
  Activation - Min: 0.000382, Max: 7.965086, Mean: 0.653993

features.4.4.block.2.fc1:
  Channels: 28
  Activation - Min: 0.233793, Max: 12.027225, Mean: 4.916203

features.4.4.block.2.fc2:
  Channels: 672
  Activation - Min: 0.022347, Max: 32.376938, Mean: 6.425279

features.4.4.block.3.0:
  Channels: 112
  Activation - Min: 0.001429, Max: 2.509109, Mean: 0.716680

features.4.5.block.0.0:
  Channels: 672
  Activation - Min: 0.045414, Max: 58.000866, Mean: 15.150898

features.4.5.block.1.0:
  Channels: 672
  Activation - Min: 0.000276, Max: 12.172184, Mean: 0.876324

features.4.5.block.2.fc1:
  Channels: 28
  Activation - Min: 0.122321, Max: 3.834189, Mean: 1.475245

features.4.5.block.2.fc2:
  Channels: 672
  Activation - Min: 0.008529, Max: 9.408781, Mean: 2.139942

features.4.5.block.3.0:
  Channels: 112
  Activation - Min: 0.001299, Max: 1.300971, Mean: 0.319702

features.5.0.block.0.0:
  Channels: 672
  Activation - Min: 0.007012, Max: 73.086800, Mean: 15.105392

features.5.0.block.1.0:
  Channels: 672
  Activation - Min: 0.000295, Max: 140.422806, Mean: 1.706063

features.5.0.block.2.fc1:
  Channels: 28
  Activation - Min: 1.122027, Max: 21.083286, Mean: 4.792241

features.5.0.block.2.fc2:
  Channels: 672
  Activation - Min: 0.542539, Max: 4.286863, Mean: 2.501563

features.5.0.block.3.0:
  Channels: 160
  Activation - Min: 0.289192, Max: 105.554962, Mean: 24.539452

features.5.1.block.0.0:
  Channels: 960
  Activation - Min: 0.000650, Max: 20.010799, Mean: 2.652662

features.5.1.block.1.0:
  Channels: 960
  Activation - Min: 0.000056, Max: 15.148857, Mean: 0.763596

features.5.1.block.2.fc1:
  Channels: 40
  Activation - Min: 0.008404, Max: 3.515030, Mean: 0.950706

features.5.1.block.2.fc2:
  Channels: 960
  Activation - Min: 0.002820, Max: 6.645010, Mean: 1.556745

features.5.1.block.3.0:
  Channels: 160
  Activation - Min: 0.000213, Max: 1.467956, Mean: 0.397046

features.5.2.block.0.0:
  Channels: 960
  Activation - Min: 0.002534, Max: 19.678261, Mean: 5.343886

features.5.2.block.1.0:
  Channels: 960
  Activation - Min: 0.000061, Max: 14.027079, Mean: 0.609557

features.5.2.block.2.fc1:
  Channels: 40
  Activation - Min: 0.038071, Max: 2.964217, Mean: 1.154813

features.5.2.block.2.fc2:
  Channels: 960
  Activation - Min: 0.005780, Max: 5.761446, Mean: 1.371718

features.5.2.block.3.0:
  Channels: 160
  Activation - Min: 0.004559, Max: 1.287904, Mean: 0.305629

features.5.3.block.0.0:
  Channels: 960
  Activation - Min: 0.012498, Max: 34.498901, Mean: 8.668063

features.5.3.block.1.0:
  Channels: 960
  Activation - Min: 0.000422, Max: 14.375252, Mean: 0.643051

features.5.3.block.2.fc1:
  Channels: 40
  Activation - Min: 0.000685, Max: 6.363124, Mean: 1.946756

features.5.3.block.2.fc2:
  Channels: 960
  Activation - Min: 0.001156, Max: 11.721559, Mean: 2.079188

features.5.3.block.3.0:
  Channels: 160
  Activation - Min: 0.001566, Max: 1.658578, Mean: 0.414970

features.5.4.block.0.0:
  Channels: 960
  Activation - Min: 0.021133, Max: 42.729176, Mean: 12.758574

features.5.4.block.1.0:
  Channels: 960
  Activation - Min: 0.000102, Max: 27.023483, Mean: 0.654262

features.5.4.block.2.fc1:
  Channels: 40
  Activation - Min: 0.020895, Max: 5.503302, Mean: 0.946148

features.5.4.block.2.fc2:
  Channels: 960
  Activation - Min: 0.000987, Max: 6.396408, Mean: 1.419140

features.5.4.block.3.0:
  Channels: 160
  Activation - Min: 0.000038, Max: 0.973633, Mean: 0.285486

features.5.5.block.0.0:
  Channels: 960
  Activation - Min: 0.039469, Max: 64.498085, Mean: 18.124432

features.5.5.block.1.0:
  Channels: 960
  Activation - Min: 0.000368, Max: 13.944245, Mean: 0.570324

features.5.5.block.2.fc1:
  Channels: 40
  Activation - Min: 0.042480, Max: 3.260714, Mean: 0.850897

features.5.5.block.2.fc2:
  Channels: 960
  Activation - Min: 0.000601, Max: 6.808885, Mean: 1.082900

features.5.5.block.3.0:
  Channels: 160
  Activation - Min: 0.001572, Max: 1.018314, Mean: 0.339469

features.6.0.block.0.0:
  Channels: 960
  Activation - Min: 0.030957, Max: 68.985535, Mean: 21.203688

features.6.0.block.1.0:
  Channels: 960
  Activation - Min: 0.000152, Max: 130.277283, Mean: 1.730889

features.6.0.block.2.fc1:
  Channels: 40
  Activation - Min: 1.118063, Max: 45.969154, Mean: 7.257420

features.6.0.block.2.fc2:
  Channels: 960
  Activation - Min: 1.067424, Max: 4.946344, Mean: 2.309893

features.6.0.block.3.0:
  Channels: 272
  Activation - Min: 0.026840, Max: 109.170822, Mean: 29.655418

features.6.1.block.0.0:
  Channels: 1632
  Activation - Min: 0.000918, Max: 14.086412, Mean: 2.678354

features.6.1.block.1.0:
  Channels: 1632
  Activation - Min: 0.000079, Max: 28.540703, Mean: 0.375044

features.6.1.block.2.fc1:
  Channels: 68
  Activation - Min: 0.083609, Max: 4.366369, Mean: 1.241827

features.6.1.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000592, Max: 6.156479, Mean: 1.326324

features.6.1.block.3.0:
  Channels: 272
  Activation - Min: 0.001434, Max: 3.744389, Mean: 0.485161

features.6.2.block.0.0:
  Channels: 1632
  Activation - Min: 0.030594, Max: 17.505016, Mean: 4.384604

features.6.2.block.1.0:
  Channels: 1632
  Activation - Min: 0.000044, Max: 9.528646, Mean: 0.320457

features.6.2.block.2.fc1:
  Channels: 68
  Activation - Min: 0.024379, Max: 3.258947, Mean: 0.826033

features.6.2.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000908, Max: 5.849295, Mean: 1.126227

features.6.2.block.3.0:
  Channels: 272
  Activation - Min: 0.000031, Max: 3.070688, Mean: 0.355721

features.6.3.block.0.0:
  Channels: 1632
  Activation - Min: 0.002919, Max: 31.585915, Mean: 8.893963

features.6.3.block.1.0:
  Channels: 1632
  Activation - Min: 0.000007, Max: 12.107544, Mean: 0.305763

features.6.3.block.2.fc1:
  Channels: 68
  Activation - Min: 0.021548, Max: 3.244771, Mean: 0.935487

features.6.3.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.001166, Max: 6.708079, Mean: 1.291871

features.6.3.block.3.0:
  Channels: 272
  Activation - Min: 0.000627, Max: 1.806907, Mean: 0.358159

features.6.4.block.0.0:
  Channels: 1632
  Activation - Min: 0.014290, Max: 38.042530, Mean: 13.735968

features.6.4.block.1.0:
  Channels: 1632
  Activation - Min: 0.000413, Max: 7.927786, Mean: 0.304457

features.6.4.block.2.fc1:
  Channels: 68
  Activation - Min: 0.008204, Max: 3.330363, Mean: 0.970715

features.6.4.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000877, Max: 6.184988, Mean: 1.192212

features.6.4.block.3.0:
  Channels: 272
  Activation - Min: 0.001400, Max: 1.616360, Mean: 0.285206

features.6.5.block.0.0:
  Channels: 1632
  Activation - Min: 0.005120, Max: 50.424210, Mean: 20.058403

features.6.5.block.1.0:
  Channels: 1632
  Activation - Min: 0.000061, Max: 6.155792, Mean: 0.314514

features.6.5.block.2.fc1:
  Channels: 68
  Activation - Min: 0.016619, Max: 2.776173, Mean: 0.839603

features.6.5.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.002460, Max: 6.702063, Mean: 1.096104

features.6.5.block.3.0:
  Channels: 272
  Activation - Min: 0.000089, Max: 1.069080, Mean: 0.269906

features.6.6.block.0.0:
  Channels: 1632
  Activation - Min: 0.135037, Max: 65.614319, Mean: 26.663113

features.6.6.block.1.0:
  Channels: 1632
  Activation - Min: 0.000635, Max: 6.783964, Mean: 0.314609

features.6.6.block.2.fc1:
  Channels: 68
  Activation - Min: 0.000578, Max: 2.902066, Mean: 0.824757

features.6.6.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.001150, Max: 5.080026, Mean: 1.146363

features.6.6.block.3.0:
  Channels: 272
  Activation - Min: 0.001888, Max: 1.071183, Mean: 0.281261

features.6.7.block.0.0:
  Channels: 1632
  Activation - Min: 0.033040, Max: 103.394226, Mean: 33.210102

features.6.7.block.1.0:
  Channels: 1632
  Activation - Min: 0.000144, Max: 5.364334, Mean: 0.320570

features.6.7.block.2.fc1:
  Channels: 68
  Activation - Min: 0.003284, Max: 2.884629, Mean: 0.836860

features.6.7.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.000112, Max: 5.418056, Mean: 1.012448

features.6.7.block.3.0:
  Channels: 272
  Activation - Min: 0.002939, Max: 1.623473, Mean: 0.310752

features.7.0.block.0.0:
  Channels: 1632
  Activation - Min: 0.016749, Max: 191.471741, Mean: 46.656960

features.7.0.block.1.0:
  Channels: 1632
  Activation - Min: 0.000000, Max: 94.969307, Mean: 1.989396

features.7.0.block.2.fc1:
  Channels: 68
  Activation - Min: 0.218438, Max: 7.339898, Mean: 2.632897

features.7.0.block.2.fc2:
  Channels: 1632
  Activation - Min: 0.004671, Max: 15.875252, Mean: 2.925092

features.7.0.block.3.0:
  Channels: 448
  Activation - Min: 0.000137, Max: 8.692631, Mean: 0.675491

features.7.1.block.0.0:
  Channels: 2688
  Activation - Min: 0.000363, Max: 4.192782, Mean: 0.792369

features.7.1.block.1.0:
  Channels: 2688
  Activation - Min: 0.000000, Max: 21.017275, Mean: 0.645600

features.7.1.block.2.fc1:
  Channels: 112
  Activation - Min: 0.035696, Max: 4.641036, Mean: 1.570205

features.7.1.block.2.fc2:
  Channels: 2688
  Activation - Min: 0.000027, Max: 12.298304, Mean: 1.938009

features.7.1.block.3.0:
  Channels: 448
  Activation - Min: 0.000085, Max: 2.487521, Mean: 0.274690

features.8.0:
  Channels: 1792
  Activation - Min: 2.676399, Max: 13.694213, Mean: 7.448977

classifier.1:
  Channels: 101
  Activation - Min: 9.381924, Max: 14.200382, Mean: 11.377246
============================================================


Pruning Results:
  Parameters before: 17,729,709
  Parameters after:  15,439,010
  Parameters removed: 2,290,699 (12.92%)

############################################################
Pruning Complete
############################################################
Initial parameters:  17,729,709
Final parameters:    15,439,010
Total removed:       2,290,699 (12.92%)
Target pruning ratio: 10.00%
✓ Wanda pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 3.20%                                                                                                
✓ Model Size: 59.33 MB
✓ Average Inference Time: 5.3055 ms
✓ FLOPs: 4.04 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|█████████████████████████████████████████████████| 313/313 [01:42<00:00,  3.06it/s, loss=1.1954, acc=68.26%] 
Epoch 1/10 - Train Loss: 1.1954, Train Acc: 68.26%, Test Acc: 84.56%                                                          
Epoch 2/10: 100%|█████████████████████████████████████████████████| 313/313 [01:41<00:00,  3.08it/s, loss=1.0861, acc=71.48%] 
Epoch 2/10 - Train Loss: 1.0861, Train Acc: 71.48%, Test Acc: 85.25%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_W_epoch2.pth
Epoch 3/10: 100%|█████████████████████████████████████████████████| 313/313 [01:41<00:00,  3.08it/s, loss=0.9755, acc=74.36%]
Epoch 3/10 - Train Loss: 0.9755, Train Acc: 74.36%, Test Acc: 85.27%                                                          
Epoch 4/10: 100%|█████████████████████████████████████████████████| 313/313 [01:41<00:00,  3.09it/s, loss=0.9174, acc=75.66%] 
Epoch 4/10 - Train Loss: 0.9174, Train Acc: 75.66%, Test Acc: 85.45%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_W_epoch4.pth
Epoch 5/10: 100%|█████████████████████████████████████████████████| 313/313 [01:42<00:00,  3.04it/s, loss=0.8576, acc=77.10%]
Epoch 5/10 - Train Loss: 0.8576, Train Acc: 77.10%, Test Acc: 85.92%                                                          
Epoch 6/10: 100%|█████████████████████████████████████████████████| 313/313 [01:43<00:00,  3.03it/s, loss=0.8174, acc=78.62%] 
Epoch 6/10 - Train Loss: 0.8174, Train Acc: 78.62%, Test Acc: 85.82%
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_W_epoch6.pth
Epoch 7/10: 100%|█████████████████████████████████████████████████| 313/313 [01:43<00:00,  3.03it/s, loss=0.7995, acc=78.42%]
Epoch 7/10 - Train Loss: 0.7995, Train Acc: 78.42%, Test Acc: 85.68%                                                          
Epoch 8/10: 100%|█████████████████████████████████████████████████| 313/313 [01:43<00:00,  3.04it/s, loss=0.7605, acc=80.00%] 
Epoch 8/10 - Train Loss: 0.7605, Train Acc: 80.00%, Test Acc: 85.91%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_W_epoch8.pth
Epoch 9/10: 100%|█████████████████████████████████████████████████| 313/313 [01:43<00:00,  3.03it/s, loss=0.7287, acc=80.68%]
Epoch 9/10 - Train Loss: 0.7287, Train Acc: 80.68%, Test Acc: 86.06%                                                          
Epoch 10/10: 100%|████████████████████████████████████████████████| 313/313 [01:43<00:00,  3.03it/s, loss=0.7440, acc=80.92%] 
Epoch 10/10 - Train Loss: 0.7440, Train Acc: 80.92%, Test Acc: 85.86%                                                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_W_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_W_best.pth
  Best Accuracy: 86.06%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 85.86%                                                                                                
✓ Model Size: 59.33 MB
✓ Average Inference Time: 5.2931 ms
✓ FLOPs: 4.04 GFLOPs

================================================================================
COMPARISON: WANDA PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         89.38   |          3.2    |    85.86   | -3.52                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         68.11   |         59.33   |    59.33   | -8.78 (-12.9%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.5993 |          5.3055 |     5.2931 | +4.6938                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.61   |          4.04   |     4.04   | -0.57 (-12.3%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS9_EfficientNetB4_Food101\TS9_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================



******************************



================================================================================
TEST SCENARIO TS9: MAGNITUDE-BASED PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FT_best.pth
  - Epoch: 10
  - Test Accuracy: 89.38%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 89.38%                                                                                             
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.6250 ms
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
  Parameters after:  16,050,758
  Parameters removed: 1,678,951 (9.47%)

############################################################
Pruning Complete
############################################################
Initial parameters:  17,729,709
Final parameters:    16,050,758
Total removed:       1,678,951 (9.47%)
Target pruning ratio: 10.00%
✓ Magnitude-based pruning completed

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 2.06%                                                                                                
✓ Model Size: 61.67 MB
✓ Average Inference Time: 0.6081 ms
✓ FLOPs: 3.91 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|█████████████████████████████████████████████████| 313/313 [01:26<00:00,  3.62it/s, loss=1.0709, acc=71.52%] 
Epoch 1/10 - Train Loss: 1.0709, Train Acc: 71.52%, Test Acc: 86.68%                                                          
Epoch 2/10: 100%|█████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.64it/s, loss=0.9645, acc=74.98%] 
Epoch 2/10 - Train Loss: 0.9645, Train Acc: 74.98%, Test Acc: 86.84%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_MAG_epoch2.pth
Epoch 3/10: 100%|█████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.65it/s, loss=0.8792, acc=76.58%]
Epoch 3/10 - Train Loss: 0.8792, Train Acc: 76.58%, Test Acc: 86.72%                                                          
Epoch 4/10: 100%|█████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.65it/s, loss=0.7837, acc=79.30%] 
Epoch 4/10 - Train Loss: 0.7837, Train Acc: 79.30%, Test Acc: 86.87%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_MAG_epoch4.pth
Epoch 5/10: 100%|█████████████████████████████████████████████████| 313/313 [01:26<00:00,  3.64it/s, loss=0.7891, acc=78.82%]
Epoch 5/10 - Train Loss: 0.7891, Train Acc: 78.82%, Test Acc: 87.03%                                                          
Epoch 6/10: 100%|█████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.64it/s, loss=0.6968, acc=81.04%] 
Epoch 6/10 - Train Loss: 0.6968, Train Acc: 81.04%, Test Acc: 86.86%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_MAG_epoch6.pth
Epoch 7/10: 100%|█████████████████████████████████████████████████| 313/313 [01:26<00:00,  3.64it/s, loss=0.6942, acc=81.10%]
Epoch 7/10 - Train Loss: 0.6942, Train Acc: 81.10%, Test Acc: 87.12%                                                          
Epoch 8/10: 100%|█████████████████████████████████████████████████| 313/313 [01:25<00:00,  3.64it/s, loss=0.6662, acc=82.68%] 
Epoch 8/10 - Train Loss: 0.6662, Train Acc: 82.68%, Test Acc: 87.03%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_MAG_epoch8.pth
Epoch 9/10: 100%|█████████████████████████████████████████████████| 313/313 [01:26<00:00,  3.63it/s, loss=0.6517, acc=82.34%]
Epoch 9/10 - Train Loss: 0.6517, Train Acc: 82.34%, Test Acc: 87.16%                                                          
Epoch 10/10: 100%|████████████████████████████████████████████████| 313/313 [01:26<00:00,  3.64it/s, loss=0.6537, acc=82.36%] 
Epoch 10/10 - Train Loss: 0.6537, Train Acc: 82.36%, Test Acc: 87.12%                                                         
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_MAG_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_MAG_best.pth
  Best Accuracy: 87.16%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 87.12%                                                                                                
✓ Model Size: 61.67 MB
✓ Average Inference Time: 0.5877 ms
✓ FLOPs: 3.91 GFLOPs

================================================================================
COMPARISON: MAGNITUDE-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |          89.38  |          2.06   |    87.12   | -2.25                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |          68.11  |         61.67   |    61.67   | -6.45 (-9.5%)               |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |           0.625 |          0.6081 |     0.5877 | -0.0372                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |           4.61  |          3.91   |     3.91   | -0.70 (-15.3%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS9_EfficientNetB4_Food101\TS9_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================



**********************



================================================================================
TEST SCENARIO TS9: TAYLOR GRADIENT-BASED PRUNING
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
✓ Model loaded from: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FT_best.pth
  - Epoch: 10
  - Test Accuracy: 89.38%

================================================================================
EVALUATING ORIGINAL FINE-TUNED MODEL
================================================================================
✓ Original Model Accuracy: 89.38%                                                                                             
✓ Model Size: 68.11 MB
✓ Average Inference Time: 0.5962 ms
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
Computing gradients:  99%|█████████████████████████████████████████████████████████████████▎| 99/100 [00:40<00:00,  2.44it/s] 
✓ Gradients computed on 100 batches
Using GroupTaylorImportance

Initializing Torch-Pruning with Taylor importance...

Applying pruning...

✓ Taylor pruning completed
  Parameters before: 17,729,709
  Parameters after: 15,393,337
  Parameters removed: 2,336,372 (13.18%)

================================================================================
EVALUATING AFTER PRUNING (BEFORE FINE-TUNING)
================================================================================
✓ Pruned Model Accuracy: 35.69%                                                                                               
✓ Model Size: 59.16 MB
✓ Average Inference Time: 0.6056 ms
✓ FLOPs: 4.11 GFLOPs

================================================================================
FINE-TUNING PRUNED MODEL
================================================================================
Fine-tune Epochs: 10
Epoch 1/10: 100%|█████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.43it/s, loss=0.9166, acc=75.96%] 
Epoch 1/10 - Train Loss: 0.9166, Train Acc: 75.96%, Test Acc: 87.84%                                                          
Epoch 2/10: 100%|█████████████████████████████████████████████████| 313/313 [01:30<00:00,  3.45it/s, loss=0.8095, acc=78.46%] 
Epoch 2/10 - Train Loss: 0.8095, Train Acc: 78.46%, Test Acc: 87.91%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_TAY_epoch2.pth
Epoch 3/10: 100%|█████████████████████████████████████████████████| 313/313 [01:30<00:00,  3.45it/s, loss=0.7591, acc=80.34%]
Epoch 3/10 - Train Loss: 0.7591, Train Acc: 80.34%, Test Acc: 87.84%                                                          
Epoch 4/10: 100%|█████████████████████████████████████████████████| 313/313 [01:30<00:00,  3.45it/s, loss=0.7202, acc=80.58%] 
Epoch 4/10 - Train Loss: 0.7202, Train Acc: 80.58%, Test Acc: 87.75%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_TAY_epoch4.pth
Epoch 5/10: 100%|█████████████████████████████████████████████████| 313/313 [01:30<00:00,  3.44it/s, loss=0.6575, acc=83.02%]
Epoch 5/10 - Train Loss: 0.6575, Train Acc: 83.02%, Test Acc: 87.72%                                                          
Epoch 6/10: 100%|█████████████████████████████████████████████████| 313/313 [01:30<00:00,  3.45it/s, loss=0.6397, acc=82.88%] 
Epoch 6/10 - Train Loss: 0.6397, Train Acc: 82.88%, Test Acc: 87.75%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_TAY_epoch6.pth
Epoch 7/10: 100%|█████████████████████████████████████████████████| 313/313 [01:30<00:00,  3.44it/s, loss=0.6237, acc=83.50%]
Epoch 7/10 - Train Loss: 0.6237, Train Acc: 83.50%, Test Acc: 87.92%                                                          
Epoch 8/10: 100%|█████████████████████████████████████████████████| 313/313 [01:30<00:00,  3.45it/s, loss=0.5785, acc=84.96%] 
Epoch 8/10 - Train Loss: 0.5785, Train Acc: 84.96%, Test Acc: 87.87%                                                          
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_TAY_epoch8.pth
Epoch 9/10: 100%|█████████████████████████████████████████████████| 313/313 [01:30<00:00,  3.44it/s, loss=0.5817, acc=84.16%]
Epoch 9/10 - Train Loss: 0.5817, Train Acc: 84.16%, Test Acc: 87.80%                                                          
Epoch 10/10: 100%|████████████████████████████████████████████████| 313/313 [01:31<00:00,  3.44it/s, loss=0.5765, acc=84.80%] 
Epoch 10/10 - Train Loss: 0.5765, Train Acc: 84.80%, Test Acc: 87.97%
  ✓ Checkpoint saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_TAY_epoch10.pth

✓ Best model saved: C:\source\checkpoints\TS9\EfficientNetB4_Food101_FTAP_TAY_best.pth
  Best Accuracy: 87.97%

================================================================================
FINAL EVALUATION (AFTER PRUNING + FINE-TUNING)
================================================================================
✓ Final Model Accuracy: 87.97%                                                                                                
✓ Model Size: 59.16 MB
✓ Average Inference Time: 0.5965 ms
✓ FLOPs: 4.11 GFLOPs

================================================================================
COMPARISON: TAYLOR GRADIENT-BASED PRUNING RESULTS
================================================================================
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Metric              |   Original (FT) |   After Pruning |   After FT | Change (Original → Final)   |
+=====================+=================+=================+============+=============================+
| Accuracy (%)        |         89.38   |         35.69   |    87.97   | -1.41                       |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Size (MB)           |         68.11   |         59.16   |    59.16   | -8.95 (-13.1%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| Inference Time (ms) |          0.5962 |          0.6056 |     0.5965 | +0.0003                     |
+---------------------+-----------------+-----------------+------------+-----------------------------+
| FLOPs (G)           |          4.61   |          4.11   |     4.11   | -0.50 (-10.9%)              |
+---------------------+-----------------+-----------------+------------+-----------------------------+

✓ Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS9_EfficientNetB4_Food101\TS9_Results.json

================================================================================
SCRIPT COMPLETED SUCCESSFULLY
================================================================================