================================================================================
TS7 FP16 QUANTIZATION ANALYSIS
================================================================================

Loading ImageNet validation dataset...
Dataset directory: C:\source\downloaded_datasets\imagenet
Using 5000 samples from validation set

Using device: cuda

################################################################################
# MODEL: Original_FT
# Original Fine-tuned ResNet50
################################################################################

Loading model from: C:\source\checkpoints\TS7\ResNet50_ImageNet_FT_best.pth

--------------------------------------------------------------------------------
ORIGINAL MODEL (FP32)
--------------------------------------------------------------------------------

================================================================================
Evaluating: Original_FT (Original)
================================================================================

1. Measuring accuracy...
  Evaluated 150/157 batches

2. Measuring inference time...
Average inference time per input: 1.0955 ms

3. Measuring model size...
Model size: 97.80 MB

4. Counting parameters...
Total parameters: 25,557,032
Trainable parameters: 25,557,032

--------------------------------------------------------------------------------
APPLYING FP16 QUANTIZATION
--------------------------------------------------------------------------------

Applying FP16 quantization...

============================================================
CONVERTING TO HALF PRECISION (FP16)
============================================================

✓ FP16 conversion completed
  - Original parameters: 25,557,032
  - FP16 parameters: 25,557,032
  - Memory reduction: ~50%
  - Works on: CPU and GPU
  - Note: May have numerical stability issues

--------------------------------------------------------------------------------
QUANTIZED MODEL (FP16)
--------------------------------------------------------------------------------

================================================================================
Evaluating: Original_FT (FP16 Quantized)
================================================================================

1. Measuring accuracy...
  Evaluated 150/157 batches

2. Measuring inference time...
Average inference time per input: 0.5375 ms

3. Measuring model size...
Model size: 48.95 MB

4. Counting parameters...
Total parameters: 25,557,032
Trainable parameters: 25,557,032

Saved quantized checkpoint: C:\source\checkpoints\TS7\ResNet50_ImageNet_FT_best_FP16.pth

################################################################################
# MODEL: Coverage_Pruned_FT
# Coverage Pruning + Fine-tuned
################################################################################

Loading model from: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_NC_best.pth
Loaded pruned model architecture from checkpoint['model']

--------------------------------------------------------------------------------
ORIGINAL MODEL (FP32)
--------------------------------------------------------------------------------

================================================================================
Evaluating: Coverage_Pruned_FT (Original)
================================================================================

1. Measuring accuracy...
  Evaluated 150/157 batches

2. Measuring inference time...
Average inference time per input: 1.2868 ms

3. Measuring model size...
Model size: 82.88 MB

4. Counting parameters...
Total parameters: 21,650,039
Trainable parameters: 21,650,039

--------------------------------------------------------------------------------
APPLYING FP16 QUANTIZATION
--------------------------------------------------------------------------------

Applying FP16 quantization...

============================================================
CONVERTING TO HALF PRECISION (FP16)
============================================================

✓ FP16 conversion completed
  - Original parameters: 21,650,039
  - FP16 parameters: 21,650,039
  - Memory reduction: ~50%
  - Works on: CPU and GPU
  - Note: May have numerical stability issues

--------------------------------------------------------------------------------
QUANTIZED MODEL (FP16)
--------------------------------------------------------------------------------

================================================================================
Evaluating: Coverage_Pruned_FT (FP16 Quantized)
================================================================================

1. Measuring accuracy...
  Evaluated 150/157 batches

2. Measuring inference time...
Average inference time per input: 0.7648 ms

3. Measuring model size...
Model size: 41.49 MB

4. Counting parameters...
Total parameters: 21,650,039
Trainable parameters: 21,650,039

Saved quantized checkpoint: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_NC_best_FP16.pth

################################################################################
# MODEL: Wanda_Pruned_FT
# Wanda Pruning + Fine-tuned
################################################################################

Loading model from: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_W_best.pth
Loaded pruned model architecture from checkpoint['model']

--------------------------------------------------------------------------------
ORIGINAL MODEL (FP32)
--------------------------------------------------------------------------------

================================================================================
Evaluating: Wanda_Pruned_FT (Original)
================================================================================

1. Measuring accuracy...
  Evaluated 150/157 batches

2. Measuring inference time...
Average inference time per input: 1.1014 ms

3. Measuring model size...
Model size: 82.22 MB

4. Counting parameters...
Total parameters: 21,481,254
Trainable parameters: 21,481,254

--------------------------------------------------------------------------------
APPLYING FP16 QUANTIZATION
--------------------------------------------------------------------------------

Applying FP16 quantization...

============================================================
CONVERTING TO HALF PRECISION (FP16)
============================================================

✓ FP16 conversion completed
  - Original parameters: 21,481,254
  - FP16 parameters: 21,481,254
  - Memory reduction: ~50%
  - Works on: CPU and GPU
  - Note: May have numerical stability issues

--------------------------------------------------------------------------------
QUANTIZED MODEL (FP16)
--------------------------------------------------------------------------------

================================================================================
Evaluating: Wanda_Pruned_FT (FP16 Quantized)
================================================================================

1. Measuring accuracy...
  Evaluated 150/157 batches

2. Measuring inference time...
Average inference time per input: 0.6041 ms

3. Measuring model size...
Model size: 41.16 MB

4. Counting parameters...
Total parameters: 21,481,254
Trainable parameters: 21,481,254

Saved quantized checkpoint: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_W_best_FP16.pth

################################################################################
# MODEL: Magnitude_Pruned_FT
# Magnitude Pruning + Fine-tuned
################################################################################

Loading model from: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_MAG_best.pth
Loaded pruned model architecture from checkpoint['model']

--------------------------------------------------------------------------------
ORIGINAL MODEL (FP32)
--------------------------------------------------------------------------------

================================================================================
Evaluating: Magnitude_Pruned_FT (Original)
================================================================================

1. Measuring accuracy...
  Evaluated 150/157 batches

2. Measuring inference time...
Average inference time per input: 0.8469 ms

3. Measuring model size...
Model size: 81.55 MB

4. Counting parameters...
Total parameters: 21,303,919
Trainable parameters: 21,303,919

--------------------------------------------------------------------------------
APPLYING FP16 QUANTIZATION
--------------------------------------------------------------------------------

Applying FP16 quantization...

============================================================
CONVERTING TO HALF PRECISION (FP16)
============================================================

✓ FP16 conversion completed
  - Original parameters: 21,303,919
  - FP16 parameters: 21,303,919
  - Memory reduction: ~50%
  - Works on: CPU and GPU
  - Note: May have numerical stability issues

--------------------------------------------------------------------------------
QUANTIZED MODEL (FP16)
--------------------------------------------------------------------------------

================================================================================
Evaluating: Magnitude_Pruned_FT (FP16 Quantized)
================================================================================

1. Measuring accuracy...
  Evaluated 150/157 batches

2. Measuring inference time...
Average inference time per input: 0.4118 ms

3. Measuring model size...
Model size: 40.83 MB

4. Counting parameters...
Total parameters: 21,303,919
Trainable parameters: 21,303,919

Saved quantized checkpoint: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_MAG_best_FP16.pth

################################################################################
# MODEL: Taylor_Pruned_FT
# Taylor Pruning + Fine-tuned
################################################################################

Loading model from: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_TAY_best.pth
Loaded pruned model architecture from checkpoint['model']

--------------------------------------------------------------------------------
ORIGINAL MODEL (FP32)
--------------------------------------------------------------------------------

================================================================================
Evaluating: Taylor_Pruned_FT (Original)
================================================================================

1. Measuring accuracy...
  Evaluated 150/157 batches

2. Measuring inference time...
Average inference time per input: 1.0226 ms

3. Measuring model size...
Model size: 82.49 MB

4. Counting parameters...
Total parameters: 21,548,673
Trainable parameters: 21,548,673

--------------------------------------------------------------------------------
APPLYING FP16 QUANTIZATION
--------------------------------------------------------------------------------

Applying FP16 quantization...

============================================================
CONVERTING TO HALF PRECISION (FP16)
============================================================

✓ FP16 conversion completed
  - Original parameters: 21,548,673
  - FP16 parameters: 21,548,673
  - Memory reduction: ~50%
  - Works on: CPU and GPU
  - Note: May have numerical stability issues

--------------------------------------------------------------------------------
QUANTIZED MODEL (FP16)
--------------------------------------------------------------------------------

================================================================================
Evaluating: Taylor_Pruned_FT (FP16 Quantized)
================================================================================

1. Measuring accuracy...
  Evaluated 150/157 batches

2. Measuring inference time...
Average inference time per input: 0.5107 ms

3. Measuring model size...
Model size: 41.30 MB

4. Counting parameters...
Total parameters: 21,548,673
Trainable parameters: 21,548,673

Saved quantized checkpoint: C:\source\checkpoints\TS7\ResNet50_ImageNet_FTAP_TAY_best_FP16.pth

Results saved to: c:\source\repos\cleanai-v5\test_scenarios\TS7_ResNet50_ImageNet\TS7_Quantization_Results.json

========================================================================================================================
QUANTIZATION COMPARISON SUMMARY
========================================================================================================================
Model                          Type       Acc (%)    Time (ms)    Size (MB)    Params (M)   RAM (MB)
------------------------------------------------------------------------------------------------------------------------
Original_FT                    FP32       78.18      1.0955       97.80        25.56        1319.7
Original_FT (FP16)             FP16       78.16      0.5375       48.95        25.56        1495.6
  └─ Improvement                          -0.02      2.04        x -49.9       %
------------------------------------------------------------------------------------------------------------------------
Coverage_Pruned_FT             FP32       71.04      1.2868       82.88        21.65        1591.3
Coverage_Pruned_FT (FP16)      FP16       71.14      0.7648       41.49        21.65        1613.0
  └─ Improvement                          0.10       1.68        x -49.9       %
------------------------------------------------------------------------------------------------------------------------
Wanda_Pruned_FT                FP32       67.08      1.1014       82.22        21.48        1635.1
Wanda_Pruned_FT (FP16)         FP16       67.18      0.6041       41.16        21.48        1636.8
  └─ Improvement                          0.10       1.82        x -49.9       %
------------------------------------------------------------------------------------------------------------------------
Magnitude_Pruned_FT            FP32       69.44      0.8469       81.55        21.30        1651.9
Magnitude_Pruned_FT (FP16)     FP16       69.54      0.4118       40.83        21.30        1644.0
  └─ Improvement                          0.10       2.06        x -49.9       %
------------------------------------------------------------------------------------------------------------------------
Taylor_Pruned_FT               FP32       75.74      1.0226       82.49        21.55        1675.3
Taylor_Pruned_FT (FP16)        FP16       75.76      0.5107       41.30        21.55        1659.4
  └─ Improvement                          0.02       2.00        x -49.9       %
------------------------------------------------------------------------------------------------------------------------
========================================================================================================================

KEY METRICS:
--------------------------------------------------------------------------------

Original_FT:
  Size Reduction:     49.95%
  Speedup:            2.04x
  Accuracy Drop:      0.02%

Coverage_Pruned_FT:
  Size Reduction:     49.94%
  Speedup:            1.68x
  Accuracy Drop:      -0.10%

Wanda_Pruned_FT:
  Size Reduction:     49.94%
  Speedup:            1.82x
  Accuracy Drop:      -0.10%

Magnitude_Pruned_FT:
  Size Reduction:     49.94%
  Speedup:            2.06x
  Accuracy Drop:      -0.10%

Taylor_Pruned_FT:
  Size Reduction:     49.94%
  Speedup:            2.00x
  Accuracy Drop:      -0.02%

================================================================================

################################################################################
# TS7 Quantization Analysis Complete!
################################################################################





*********************************




Mükemmel soru! İşte **FP16** vs **Dynamic INT8** performans farkının nedenleri:

## 🎯 FP16 Quantization (GPU)

### ✅ Neden Çok İyi Çalışıyor?

1. **Tam Model Dönüşümü**
```python
model.half()  # TÜM parametreler + activations → FP16
# Weights: FP32 (32-bit) → FP16 (16-bit) = %50 azalma
# Activations: FP32 → FP16 = Memory bandwidth %50 azalma
```

2. **GPU Hardware Desteği**
   - NVIDIA Tensor Cores FP16'yı **native** destekliyor
   - RTX serisi: FP16 işlemler FP32'den **2-3x daha hızlı**
   - Memory bandwidth yarıya iniyor → GPU-CPU transfer hızlanıyor

3. **Gerçek Boyut Azalması**
```
Original: 97.79 MB (FP32)
FP16:     48.90 MB (FP16) → ~50% azalma ✅
```

---

## ⚠️ Dynamic INT8 Quantization (CPU)

### ❌ Neden Zayıf Kalıyor?

1. **Sadece Weights Quantized**
```python
# Dynamic INT8 (şu anki)
Weights: FP32 → INT8 (8-bit)      ✅
Activations: FP32 → FP32 (32-bit)  ❌ Değişmiyor!

# Checkpoint içeriği:
- INT8 weights         (~25 MB)
- FP32 metadata        (~60 MB)  ← Bu kaydediliyor!
- optimizer state      (varsa FP32)
- scheduler state      (varsa FP32)
```

**Sonuç**: 97.79 MB → 91.93 MB (sadece ~6% azalma)

2. **Dequantization Overhead**
```python
# Her inference'da:
INT8 weights → FP32'ye dönüştür (dequantize)
FP32 activations ile hesapla
Sonuç: Ekstra conversion maliyeti!
```

3. **ResNet50 = Çoğunlukla Conv Layers**
```python
# Dynamic INT8 hangi layerları hızlandırır?
✅ Linear layers  (fully connected)
✅ LSTM/GRU       (recurrent)
❌ Conv2D layers  (convolutional) ← ResNet'in %90'ı bu!

# ResNet50 yapısı:
Conv layers: ~%90
Linear (fc): ~%10  ← Sadece son layer!
```

---

## 📊 Karşılaştırma

| Özellik | FP16 (GPU) | Dynamic INT8 (CPU) |
|---------|------------|-------------------|
| **Weights** | 16-bit | 8-bit |
| **Activations** | 16-bit | 32-bit ❌ |
| **Boyut Azalması** | ~50% ✅ | ~5-10% ❌ |
| **Hızlanma** | 2-3x (GPU) ✅ | Yok/Negatif ❌ |
| **Hardware** | GPU Tensor Cores | CPU FBGEMM |
| **Model Tipi** | Tüm modeller | Linear/LSTM odaklı |
| **ResNet İçin** | Mükemmel ✅ | Zayıf ❌ |

---

## 💡 Çözüm Önerileri

### 1. **Static INT8 (İdeal ama Windows'ta sorunlu)**
```python
# Hem weights hem activations INT8
Weights: INT8
Activations: INT8  ← Bu önemli!
Boyut: ~75% azalma (teorik)
Hız: 2-4x (CPU'da)

# Sorun:
PyTorch Windows build → ResNet static quantization desteklemiyor
'quantized::conv2d.new' hatası
```

### 2. **Quantization-Aware Training (QAT)**
```python
# En iyi accuracy için
Model baştan INT8 için train edilir
Fake-quantization nodes eklenir
→ Daha iyi accuracy, gerçek INT8 benefits
```

### 3. **ONNX Runtime + Quantization** ⭐
```python
# PyTorch → ONNX → Quantize
import onnxruntime as ort

#장점:
- Windows'ta static INT8 çalışır
- Daha agresif optimizations
- Cross-platform deployment
```

### 4. **FP16 Kullanmaya Devam Et** (Önerim) ✅
```python
# Mevcut durumda en iyi seçenek:
✅ GPU varsa: FP16 quantization (şu anki)
   - %50 boyut azalması
   - 2-3x hızlanma
   - Stabil ve kolay

✅ CPU deployment: Model pruning + FP16
   - Pruning: %15-20 param azaltma
   - FP16: %50 boyut azalması
   - Toplam: ~60% boyut azalması
```

---

## 🎓 Özet

**Sorunun cevabı:**

1. **FP16**: GPU'da native destekleniyor, tüm model dönüşüyor → Büyük kazanç
2. **Dynamic INT8**: Sadece weights quantized, Conv layers hızlanmıyor → Minimal kazanç
3. **Static INT8**: Teoride iyi ama PyTorch Windows'ta ResNet için broken
4. **Çözüm**: GPU deployment için FP16 kullan, CPU için pruning + dynamic INT8 kombinasyonu

Daha fazla compression istiyorsanız ONNX Runtime ile static INT8 deneyebiliriz, isterseniz hazırlayayım? 🚀