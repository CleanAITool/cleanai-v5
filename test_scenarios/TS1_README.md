# Test Scenario TS1: ResNet-18 CIFAR-10 Pruning Analysis

Bu test senaryosu, ResNet-18 modelinin CIFAR-10 dataseti üzerinde iki farklı pruning yöntemi (Neuron Coverage ve WANDA) ile budanmasını ve sonuçların karşılaştırılmasını içerir.

## 📋 Senaryo Bilgileri

- **Test Senaryosu No**: TS1
- **Model**: ResNet-18
- **Dataset**: CIFAR-10
- **Pruning Oranı**: 20% (0.2)
- **Pruning Yöntemleri**: 
  - Neuron Coverage
  - WANDA (Weight AND Activation)

## 📁 Klasör Yapısı

```
C:\source\
├── downloaded_models/                    # Pretrained modeller
├── downloaded_datasets/                  # CIFAR-10 dataset
├── checkpoints\
│   ├── TS1/                             # Script 1 checkpointleri
│   │   ├── ResNet18_CIFAR10_pretrained.pth
│   │   ├── ResNet18_CIFAR10_FT_epoch5.pth
│   │   ├── ResNet18_CIFAR10_FT_epoch10.pth
│   │   └── ResNet18_CIFAR10_FT_final.pth
│   │
│   ├── TS1_Coverage_ResNet18_CIFAR10/   # Script 2 checkpointleri
│   │   ├── ResNet18_CIFAR10_pruned_NC.pth
│   │   ├── ResNet18_CIFAR10_FTAP_NC_epoch5.pth
│   │   ├── ResNet18_CIFAR10_FTAP_NC_epoch10.pth
│   │   ├── ResNet18_CIFAR10_FTAP_NC_final.pth
│   │   └── reports/
│   │       └── TS1_Coverage_ResNet18_CIFAR10.pdf
│   │
│   └── TS1_Wanda_ResNet18_CIFAR10/      # Script 3 checkpointleri
│       ├── ResNet18_CIFAR10_pruned_W.pth
│       ├── ResNet18_CIFAR10_FTAP_W_epoch5.pth
│       ├── ResNet18_CIFAR10_FTAP_W_epoch10.pth
│       ├── ResNet18_CIFAR10_FTAP_W_final.pth
│       └── reports/
│           └── TS1_Wanda_ResNet18_CIFAR10.pdf
```

## 🚀 Çalıştırma Sırası

### Script 1: Model Hazırlama ve Fine-Tuning
```bash
python test_scenarios/TS1_01_prepare_model.py
```

**Amaç**: Pretrained ResNet-18'i indir, CIFAR-10'a uyarla ve fine-tune et.

**Çıktılar**:
- Pretrained model checkpoint
- Her 5 epochta bir checkpoint (FT_epoch5, FT_epoch10, ...)
- Final fine-tuned model
- Before/After accuracy karşılaştırma tablosu

**Parametreler**:
- Epochs: 20
- Batch Size: 128
- Learning Rate: 0.001
- Optimizer: Adam

---

### Script 2: Neuron Coverage Pruning
```bash
python test_scenarios/TS1_02_coverage_pruning.py
```

**Amaç**: Fine-tuned modele Neuron Coverage yöntemiyle pruning uygula ve fine-tune et.

**Çıktılar**:
- Pruned model checkpoint (NC)
- Her 5 epochta bir checkpoint (FTAP_NC_epoch5, ...)
- Final fine-tuned pruned model
- Kapsamlı karşılaştırma tablosu
- PDF rapor

**Parametreler**:
- Pruning Ratio: 20%
- Coverage Metric: normalized_mean
- Global Pruning: True
- Iterative Steps: 5
- Fine-Tuning Epochs: 30

**Karşılaştırma Tablosu İçeriği**:
- Accuracy (%)
- Size (MB)
- Parameters (M)
- FLOPs (G)
- Average Inference Time (ms)

---

### Script 3: WANDA Pruning
```bash
python test_scenarios/TS1_03_wanda_pruning.py
```

**Amaç**: Fine-tuned modele WANDA yöntemiyle pruning uygula ve fine-tune et.

**Çıktılar**:
- Pruned model checkpoint (W)
- Her 5 epochta bir checkpoint (FTAP_W_epoch5, ...)
- Final fine-tuned pruned model
- Kapsamlı karşılaştırma tablosu
- PDF rapor

**Parametreler**:
- Pruning Ratio: 20%
- Method: WANDA (Weight × Activation)
- Global Pruning: True
- Iterative Steps: 5
- Calibration Batches: 50
- Fine-Tuning Epochs: 30

**Karşılaştırma Tablosu İçeriği**:
- Accuracy (%)
- Size (MB)
- Parameters (M)
- FLOPs (G)
- Average Inference Time (ms)

---

## 📊 Beklenen Sonuçlar

### Script 1: Model Preparation
```
FINE-TUNING RESULTS - COMPARISON TABLE
================================================================================
Metric                         Before Fine-Tuning        After Fine-Tuning
--------------------------------------------------------------------------------
Accuracy (%)                                ~15.00                    ~92.00
Loss                                        ~2.3000                   ~0.2500
Total Parameters                        11,173,962                11,173,962
Trainable Parameters                    11,173,962                11,173,962
--------------------------------------------------------------------------------
✓ Accuracy Improvement: +77.00%
```

### Script 2: Coverage Pruning
```
NEURON COVERAGE PRUNING - COMPREHENSIVE COMPARISON TABLE
====================================================================================================
Metric                         Original (FT)          After Pruning       After Pruning+FT
----------------------------------------------------------------------------------------------------
Accuracy (%)                           92.00                  90.20                  91.70
Size (MB)                              42.60                  34.08                  34.08
Parameters (M)                         11.17                   8.94                   8.94
FLOPs (G)                               0.56                   0.45                   0.45
Avg Inference Time (ms)                 2.45                   2.15                   2.15
----------------------------------------------------------------------------------------------------

Summary
  Parameter Reduction                                     20.00%
  Size Reduction                                          20.00%
  Speedup                                                   1.14x
  Accuracy Recovery (FT)                                   +1.50%
  Final Accuracy Drop                                      -0.30%
```

### Script 3: WANDA Pruning
```
WANDA PRUNING - COMPREHENSIVE COMPARISON TABLE
====================================================================================================
Metric                         Original (FT)          After Pruning       After Pruning+FT
----------------------------------------------------------------------------------------------------
Accuracy (%)                           92.00                  90.80                  91.85
Size (MB)                              42.60                  34.08                  34.08
Parameters (M)                         11.17                   8.94                   8.94
FLOPs (G)                               0.56                   0.45                   0.45
Avg Inference Time (ms)                 2.45                   2.10                   2.10
----------------------------------------------------------------------------------------------------

Summary
  Parameter Reduction                                     20.00%
  Size Reduction                                          20.00%
  Speedup                                                   1.17x
  Accuracy Recovery (FT)                                   +1.05%
  Final Accuracy Drop                                      -0.15%
```

## ⚙️ Gereksinimler

```bash
pip install -r requirements.txt
```

**Minimum Gereksinimler**:
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU için)
- 8GB+ RAM
- 10GB+ Disk Space

## 🔍 İsimlendirme Kuralları

### Checkpoint Formatları

1. **Fine-Tuning Checkpoint**:
   - Format: `{Model}_{Dataset}_FT_epoch{N}.pth`
   - Örnek: `ResNet18_CIFAR10_FT_epoch10.pth`

2. **Pruning Sonrası (Fine-Tuning Öncesi)**:
   - Format: `{Model}_{Dataset}_pruned_{Method}.pth`
   - Örnek: `ResNet18_CIFAR10_pruned_NC.pth` (Neuron Coverage)
   - Örnek: `ResNet18_CIFAR10_pruned_W.pth` (WANDA)

3. **Pruning + Fine-Tuning Sonrası**:
   - Format: `{Model}_{Dataset}_FTAP_{Method}_epoch{N}.pth`
   - Örnek: `ResNet18_CIFAR10_FTAP_NC_epoch15.pth`
   - Örnek: `ResNet18_CIFAR10_FTAP_W_epoch20.pth`

### Method Kısaltmaları

- **NC**: Neuron Coverage
- **W**: WANDA
- **FT**: Fine-Tuning
- **FTAP**: Fine-Tuning After Pruning

## 📝 Notlar

1. **GPU Kullanımı**: Scriptler otomatik olarak CUDA varsa GPU kullanır.

2. **Checkpoint Kaydetme**: 
   - Her 5 epochta bir model kaydedilir
   - Final model her zaman kaydedilir

3. **Memory Yönetimi**:
   - Batch size GPU memory'e göre ayarlanabilir
   - Coverage analysis için `max_batches` parametresi kullanılır

4. **Reproducibility**:
   - Random seed scriptlerde set edilmemiştir
   - İstenen sonuçlar için seed eklenebilir

5. **Fine-Tuning Süreleri**:
   - Script 1: ~20-30 dakika
   - Script 2: ~40-60 dakika
   - Script 3: ~40-60 dakika

## 🐛 Sorun Giderme

### "Fine-tuned model not found" Hatası
```bash
# Önce Script 1'i çalıştırın
python test_scenarios/TS1_01_prepare_model.py
```

### GPU Memory Hatası
```python
# Batch size'ı küçültün
BATCH_SIZE = 64  # veya 32
```

### Dataset İndirme Hatası
```python
# Manuel indirme için
datasets.CIFAR10(root=str(DATASET_DIR), train=True, download=True)
```

## 📚 Referanslar

- **WANDA Paper**: ["A Simple and Effective Pruning Approach for Large Language Models"](https://arxiv.org/abs/2306.11695)
- **Torch-Pruning**: [GitHub](https://github.com/VainF/Torch-Pruning)
- **CleanAI**: [README.md](../README.md)

## 📧 İletişim

Sorular ve öneriler için issue açabilirsiniz.

---

**Test Scenario TS1** - ResNet-18 CIFAR-10 Pruning Analysis  
*CleanAI v5 Framework*
