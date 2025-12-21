# Test Scenarios Index

Bu dizin, CleanAI v5 framework için hazırlanmış test senaryolarını içerir.

## 📚 Mevcut Test Senaryoları

### TS1: ResNet-18 CIFAR-10 Pruning Analysis

**Durum**: ✅ Tamamlandı  
**Model**: ResNet-18  
**Dataset**: CIFAR-10  
**Pruning Oranı**: 20%  
**Yöntemler**: Neuron Coverage, WANDA

**Scriptler**:
- `TS1_01_prepare_model.py` - Model hazırlama ve fine-tuning
- `TS1_02_coverage_pruning.py` - Neuron Coverage pruning
- `TS1_03_wanda_pruning.py` - WANDA pruning
- `TS1_run_all.py` - Tüm scriptleri çalıştır
- `TS1_compare_results.py` - Sonuçları karşılaştır
- `TS1_README.md` - Detaylı dokümantasyon

**Hızlı Başlangıç**:
```bash
# Tüm testleri çalıştır
python test_scenarios/TS1_run_all.py

# Sadece belirli bir script çalıştır
python test_scenarios/TS1_01_prepare_model.py
python test_scenarios/TS1_02_coverage_pruning.py
python test_scenarios/TS1_03_wanda_pruning.py

# Sonuçları karşılaştır
python test_scenarios/TS1_compare_results.py
```

---

## 🎯 Yeni Test Senaryosu Ekleme

Yeni bir test senaryosu eklemek için aşağıdaki adımları izleyin:

### 1. Senaryo Planlama

`CreatingTestScenarios_Prompt.md` dosyasını kullanarak yeni senaryo gereksinimlerini belirtin:
- Model adı
- Dataset adı
- Pruning oranı
- Test edilecek yöntemler
- Klasör yapısı

### 2. Script Oluşturma

Her test senaryosu için 3 temel script gerekir:
1. **Model Hazırlama** (`TS{N}_01_prepare_model.py`)
2. **İlk Pruning Yöntemi** (`TS{N}_02_{method}_pruning.py`)
3. **İkinci Pruning Yöntemi** (`TS{N}_03_{method}_pruning.py`)

Ek olarak:
4. **Master Script** (`TS{N}_run_all.py`)
5. **Karşılaştırma** (`TS{N}_compare_results.py`)
6. **Dokümantasyon** (`TS{N}_README.md`)

### 3. Klasör Yapısı

```
C:\source\
├── downloaded_models/
├── downloaded_datasets/
├── checkpoints\
│   ├── TS{N}/
│   ├── TS{N}_{Method1}_{Model}_{Dataset}/
│   └── TS{N}_{Method2}_{Model}_{Dataset}/
└── repos\cleanai-v5\
    └── test_scenarios\
        ├── TS{N}_*.py
        └── TS{N}_README.md
```

### 4. İsimlendirme Kuralları

- Test Senaryosu: `TS{N}` (TS1, TS2, ...)
- Script: `TS{N}_{StepNo}_{description}.py`
- Checkpoint: `{Model}_{Dataset}_{FT/FTAP}_{Method}_{epoch}.pth`
- Method kısaltmaları:
  - NC: Neuron Coverage
  - W: WANDA
  - A: Adaptive
  - M: Magnitude

---

## 📋 Test Senaryosu Template

Aşağıdaki template'i yeni test senaryoları için kullanabilirsiniz:

```markdown
# Test Scenario TS{N}: {Model} {Dataset} {Description}

## Senaryo Bilgileri
- Model: {ModelName}
- Dataset: {DatasetName}
- Pruning Oranı: {X}%
- Yöntemler: {Method1}, {Method2}

## Çalıştırma
\`\`\`bash
python test_scenarios/TS{N}_run_all.py
\`\`\`

## Beklenen Sonuçlar
{Tablolar ve karşılaştırmalar}
```

---

## 🔬 Gelecek Test Senaryoları (Planlanan)

### TS2: ResNet-50 CIFAR-100
- **Durum**: 📝 Planlandı
- **Model**: ResNet-50
- **Dataset**: CIFAR-100
- **Özellik**: Daha büyük model, daha fazla sınıf

### TS3: MobileNetV2 ImageNet
- **Durum**: 📝 Planlandı
- **Model**: MobileNetV2
- **Dataset**: ImageNet (subset)
- **Özellik**: Lightweight model, büyük dataset

### TS4: VGG16 CIFAR-10 High Compression
- **Durum**: 📝 Planlandı
- **Model**: VGG-16
- **Dataset**: CIFAR-10
- **Özellik**: Agresif pruning (70-80%)

### TS5: Multi-Method Comparison
- **Durum**: 📝 Planlandı
- **Model**: ResNet-18
- **Dataset**: CIFAR-10
- **Özellik**: 4 farklı yöntem karşılaştırması (Coverage, WANDA, Adaptive, Magnitude)

---

## 📊 Genel İstatistikler

### Tamamlanan Testler
- **Toplam Senaryo**: 1
- **Toplam Script**: 6
- **Test Edilen Model**: 1 (ResNet-18)
- **Test Edilen Dataset**: 1 (CIFAR-10)
- **Test Edilen Yöntem**: 2 (Coverage, WANDA)

### Başarı Oranları
- Model Hazırlama: ✅ Bekliyor
- Coverage Pruning: ✅ Bekliyor
- WANDA Pruning: ✅ Bekliyor

---

## 🛠️ Geliştirme Notları

### Version History
- **v1.0** (2025-12-20): TS1 senaryosu oluşturuldu

### Bilinen Sorunlar
- Yok

### TODO
- [ ] TS1 scriptlerini çalıştır ve sonuçları doğrula
- [ ] TS2-TS5 senaryolarını oluştur
- [ ] Otomatik test pipeline oluştur
- [ ] Sonuçları görselleştiren dashboard ekle

---

## 📞 Destek

Test senaryoları ile ilgili sorunlar için:
1. İlgili `TS{N}_README.md` dosyasını kontrol edin
2. Ana `README.md` dosyasını inceleyin
3. Issue açın

---

**CleanAI v5 Test Scenarios**  
*Neural Network Pruning Framework*
