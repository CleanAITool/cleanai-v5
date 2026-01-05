# TS6 Test Senaryosu - Özet Rapor

## Test Konfigürasyonu

### Model ve Veri Seti

- **Model:** EfficientNet-B4
- **Veri Seti:** Food101 (101 yiyecek kategorisi)
- **Eğitim Örnekleri:** 75,750
- **Test Örnekleri:** 25,250
- **Görüntü Boyutu:** 380x380

### Pruning Parametreleri

- **Pruning Oranı:** 10%
- **Global Pruning:** Evet (Global)
- **Iteratif Adımlar:** 1
- **Fine-Tuning Epoch:** 10
- **Kalibrasyon Batch:** 100

### Test Edilen Yöntemler

1. **Neuron Coverage Pruning** - Nöron aktivasyon kapsama tabanlı budama
2. **Wanda Pruning** - Ağırlık × Aktivasyon önem tabanlı budama
3. **Magnitude Pruning** - Ağırlık büyüklüğü tabanlı budama
4. **Taylor Pruning** - Taylor açılımı tabanlı budama

## Karşılaştırmalı Sonuçlar

| Yöntem                    | Doğruluk (%) | Doğruluk Kaybı | Boyut (MB) | Boyut Azalması | Çıkarım Süresi (ms) | FLOPs (G) | FLOPs Azalması |
| ------------------------- | ------------ | -------------- | ---------- | -------------- | ------------------- | --------- | -------------- |
| **Orijinal (Fine-Tuned)** | 89.20        | -              | 68.11      | -              | 0.588               | 4.61      | -              |
| **Coverage Pruning**      | 80.95        | -8.25          | 49.45      | -27.4%         | 5.32                | 3.70      | -19.7%         |
| **Wanda Pruning**         | 85.93        | -3.27          | 59.35      | -12.9%         | 5.32                | 4.05      | -12.1%         |
| **Magnitude Pruning**     | 86.78        | -2.42          | 61.66      | -9.5%          | 0.580               | 3.91      | -15.2%         |
| **Taylor Pruning**        | 87.99        | -1.21          | 59.14      | -13.2%         | 0.590               | 4.11      | -10.9%         |

## Önemli Gözlemler

### Performans Sıralaması

1. **Taylor Pruning** - En yüksek doğruluk (%87.99), sadece %1.21 kayıp
2. **Magnitude Pruning** - İkinci en yüksek doğruluk (%86.78)
3. **Wanda Pruning** - Üçüncü sırada (%85.93)
4. **Coverage Pruning** - En düşük doğruluk (%80.95)

### Model Optimizasyonu

- **%9-27 model boyutu azalması** (değişken sonuçlar)
- **%11-20 FLOPs azalması** (değişken sonuçlar)
- Taylor ve Magnitude yöntemlerinde çıkarım süresi korundu (~0.58 ms)
- Coverage ve Wanda yöntemlerinde çıkarım süresi önemli ölçüde arttı (~5.3 ms)

### Doğruluk-Verimlilik Dengesi

- **Taylor Pruning:** En iyi doğruluk-verimlilik dengesi, %98.6 orijinal performans korundu
- **Magnitude Pruning:** Çok iyi performans, %97.3 orijinal performans korundu
- **Wanda Pruning:** İyi performans, %96.3 orijinal performans
- **Coverage Pruning:** En fazla doğruluk kaybı, %90.7 orijinal performans

### TS5 ile Karşılaştırma (Layer-wise vs Global Pruning)

**TS5 (Layer-wise) Sonuçları:**

- Taylor: %86.67 doğruluk, -2.54% kayıp, %18.8 boyut azalması
- Magnitude: %85.83 doğruluk, -3.38% kayıp, %18.8 boyut azalması

**TS6 (Global) Sonuçları:**

- Taylor: %87.99 doğruluk, -1.21% kayıp, %13.2 boyut azalması
- Magnitude: %86.78 doğruluk, -2.42% kayıp, %9.5 boyut azalması

**Kritik Bulgular:**

- **Global pruning daha iyi doğruluk sağladı** (Taylor'da +1.32%, Magnitude'da +0.95%)
- Layer-wise pruning daha fazla model küçültmesi yaptı (%18.8 vs %9-13)
- Taylor yöntemi her iki stratejide de en iyi performansı gösterdi
- EfficientNet-B4 için **Global pruning + Taylor kombinasyonu optimal**

### EfficientNet-B4 Global Pruning Bulguları

- **Taylor Pruning üstün performans:** %98.6 performans korunma ile en iyi sonuç
- ResNet50 Global (TS4) ile karşılaştırma:
  - EfficientNet-B4: %87.99 doğruluk (%1.21 kayıp)
  - ResNet50: %76.83 doğruluk (%1.76 kayıp)
- **EfficientNet mimarisi global pruning'e daha uygun**
- Coverage ve Wanda'da çıkarım süresi problemi devam ediyor
- Magnitude ve Taylor'da çıkarım süresi performansı mükemmel

### Temel Bulgular

- **Global pruning + Taylor** kombinasyonu EfficientNet-B4 için ideal
- Layer-wise pruning daha agresif küçültme, Global pruning daha iyi doğruluk
- EfficientNet-B4 hem layer-wise hem de global pruning'e çok iyi adapte oluyor
- Taylor yöntemi tüm senaryolarda (TS4, TS5, TS6) tutarlı şekilde en iyi performansı veriyor
- Food101 veri seti üzerinde %98'in üzerinde performans korunabildi
