# TS3 Test Senaryosu - Özet Rapor

## Test Konfigürasyonu

### Model ve Veri Seti

- **Model:** ResNet50
- **Veri Seti:** ImageNet
- **Validasyon Örnekleri:** 49,997
- **Kalibrasyon Örnekleri:** 25,600

### Pruning Parametreleri

- **Pruning Oranı:** 10%
- **Global Pruning:** Hayır (Layer-wise)
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
| **Orijinal (Fine-Tuned)** | 78.59        | -              | 97.70      | -              | 0.027               | 4.13      | -              |
| **Coverage Pruning**      | 72.13        | -6.46          | 79.67      | -18.4%         | 1.33                | 3.34      | -19.1%         |
| **Wanda Pruning**         | 74.50        | -4.09          | 79.67      | -18.4%         | 1.34                | 3.34      | -19.1%         |
| **Magnitude Pruning**     | 75.55        | -3.04          | 79.67      | -18.4%         | 0.014               | 3.34      | -19.1%         |
| **Taylor Pruning**        | 75.54        | -3.05          | 79.67      | -18.4%         | 0.018               | 3.34      | -19.1%         |

## Önemli Gözlemler

### Performans Sıralaması

1. **Magnitude Pruning** - En yüksek doğruluk (%75.55)
2. **Taylor Pruning** - İkinci en yüksek doğruluk (%75.54)
3. **Wanda Pruning** - Üçüncü sırada (%74.50)
4. **Coverage Pruning** - En düşük doğruluk (%72.13)

### Model Optimizasyonu

- **%18.4 model boyutu azalması** (97.70 MB → 79.67 MB)
- **%19.1 FLOPs azalması** (4.13G → 3.34G)
- Magnitude ve Taylor yöntemlerinde çıkarım süresi azalırken, Coverage ve Wanda'da artmıştır

### Doğruluk-Verimlilik Dengesi

- **Magnitude Pruning:** En iyi doğruluk-verimlilik dengesi, %96.1 orijinal performans korundu
- **Taylor Pruning:** Magnitude'a çok yakın performans, %96.1 orijinal performans
- **Wanda Pruning:** İyi denge, %94.8 orijinal performans korundu
- **Coverage Pruning:** Daha fazla doğruluk kaybı, %91.8 orijinal performans

### TS2 ile Karşılaştırma

- TS3'te daha düşük pruning oranı (%10 vs %20) ve daha uzun fine-tuning (10 vs 5 epoch) kullanıldı
- TS3'te klasik yöntemler (Magnitude, Taylor) modern yöntemlerden (Wanda, Coverage) daha iyi performans gösterdi
- %10 pruning oranında Magnitude ve Taylor yöntemleri minimal doğruluk kaybı ile başarılı sonuçlar verdi
