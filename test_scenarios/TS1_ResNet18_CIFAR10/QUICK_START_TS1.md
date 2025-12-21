# TS1 Test Senaryosu - Hızlı Başlangıç

## 🚀 Tek Komutla Çalıştır

```bash
# Tüm test senaryosunu çalıştır (3 script sırayla)
cd test_scenarios
python TS1_run_all.py
```

## 📋 Adım Adım Çalıştırma

### Adım 1: Model Hazırlama (20-30 dakika)
```bash
python TS1_01_prepare_model.py
```
✅ ResNet-18 indirilir ve CIFAR-10'a uyarlanır  
✅ 20 epoch fine-tuning yapılır  
✅ Her 5 epochta checkpoint kaydedilir  

### Adım 2: Coverage Pruning (40-60 dakika)
```bash
python TS1_02_coverage_pruning.py
```
✅ %20 pruning uygulanır (Neuron Coverage)  
✅ 30 epoch fine-tuning yapılır  
✅ PDF rapor oluşturulur  

### Adım 3: WANDA Pruning (40-60 dakika)
```bash
python TS1_03_wanda_pruning.py
```
✅ %20 pruning uygulanır (WANDA)  
✅ 30 epoch fine-tuning yapılır  
✅ PDF rapor oluşturulur  

### Adım 4: Sonuçları Karşılaştır (2-3 dakika)
```bash
python TS1_compare_results.py
```
✅ Tüm modeller karşılaştırılır  
✅ Detaylı tablo gösterilir  

## 📊 Beklenen Çıktılar

### Checkpoints
```
C:\source\checkpoints\
├── TS1\
│   ├── ResNet18_CIFAR10_pretrained.pth
│   ├── ResNet18_CIFAR10_FT_epoch5.pth
│   ├── ResNet18_CIFAR10_FT_epoch10.pth
│   ├── ResNet18_CIFAR10_FT_epoch15.pth
│   ├── ResNet18_CIFAR10_FT_epoch20.pth
│   └── ResNet18_CIFAR10_FT_final.pth
│
├── TS1_Coverage_ResNet18_CIFAR10\
│   ├── ResNet18_CIFAR10_pruned_NC.pth
│   ├── ResNet18_CIFAR10_FTAP_NC_epoch5.pth
│   ├── ...
│   ├── ResNet18_CIFAR10_FTAP_NC_final.pth
│   └── reports\
│       └── TS1_Coverage_ResNet18_CIFAR10.pdf
│
└── TS1_Wanda_ResNet18_CIFAR10\
    ├── ResNet18_CIFAR10_pruned_W.pth
    ├── ResNet18_CIFAR10_FTAP_W_epoch5.pth
    ├── ...
    ├── ResNet18_CIFAR10_FTAP_W_final.pth
    └── reports\
        └── TS1_Wanda_ResNet18_CIFAR10.pdf
```

### Karşılaştırma Tablosu
```
==================================================================================================================
Metric                    Original (FT)    Coverage Pruned   Coverage Final   WANDA Pruned     WANDA Final
------------------------------------------------------------------------------------------------------------------
Accuracy (%)                     92.00            90.20            91.70           90.80            91.85
Parameters (M)                   11.17             8.94             8.94            8.94             8.94
Size (MB)                        42.60            34.08            34.08           34.08            34.08
Avg Inference Time (ms)           2.45             2.15             2.15            2.10             2.10
==================================================================================================================
```

## ⚙️ Özelleştirme

### Pruning Oranını Değiştir
```python
# TS1_02_coverage_pruning.py veya TS1_03_wanda_pruning.py içinde
PRUNING_RATIO = 0.3  # %30 için
```

### Fine-Tuning Epoch Sayısını Değiştir
```python
# Her script içinde
FINE_TUNE_EPOCHS = 40  # 40 epoch için
```

### Batch Size Değiştir (GPU memory için)
```python
BATCH_SIZE = 64  # Küçük GPU için
```

## 🐛 Sorun Giderme

### "CUDA out of memory"
```python
BATCH_SIZE = 64  # veya 32
```

### "Fine-tuned model not found"
```bash
# Önce Script 1'i çalıştır
python TS1_01_prepare_model.py
```

### Dataset indirilemiyor
```bash
# Manuel indirme
# CIFAR-10 otomatik indirilir, internet bağlantısını kontrol edin
```

## 📚 Detaylı Dokümantasyon

Daha fazla bilgi için:
- [TS1_README.md](TS1_README.md) - Tam dokümantasyon
- [README.md](README.md) - Test senaryoları genel bakış

## 💡 İpuçları

1. **GPU Kullanımı**: Otomatik olarak CUDA varsa GPU kullanılır
2. **Checkpoint Yönetimi**: Her 5 epochta otomatik kaydedilir
3. **Hızlı Test**: `MAX_BATCHES = 10` ile hızlı test yapılabilir
4. **Reproducibility**: Random seed ekleyerek aynı sonuçları elde edebilirsiniz

## ⏱️ Tahmini Süreler

- Script 1 (Hazırlık): ~25 dakika
- Script 2 (Coverage): ~50 dakika  
- Script 3 (WANDA): ~50 dakika
- **Toplam**: ~2 saat (GPU ile)

CPU ile 3-4x daha uzun sürebilir.

---

**Not**: İlk çalıştırmada CIFAR-10 dataset otomatik indirilecektir (~170 MB).
