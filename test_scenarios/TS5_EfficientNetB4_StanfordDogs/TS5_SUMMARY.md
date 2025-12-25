# TS5 Test Senaryosu - Özet Rapor

## Test Konfigürasyonu

### Model ve Veri Seti

- **Model:** EfficientNet-B4
- **Veri Seti:** Food101 (101 yiyecek kategorisi)
- **Eğitim Örnekleri:** 75,750
- **Test Örnekleri:** 25,250
- **Görüntü Boyutu:** 380x380

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
| **Orijinal (Fine-Tuned)** | 89.21        | -              | 68.11      | -              | 0.576               | 4.61      | -              |
| **Coverage Pruning**      | 82.69        | -6.52          | 55.35      | -18.8%         | 5.11                | 3.76      | -18.5%         |
| **Wanda Pruning**         | 83.30        | -5.91          | 55.35      | -18.8%         | 5.15                | 3.76      | -18.5%         |
| **Magnitude Pruning**     | 85.83        | -3.38          | 55.35      | -18.8%         | 0.560               | 3.76      | -18.5%         |
| **Taylor Pruning**        | 86.67        | -2.54          | 55.35      | -18.8%         | 0.556               | 3.76      | -18.5%         |

## Önemli Gözlemler

### Performans Sıralaması

1. **Taylor Pruning** - En yüksek doğruluk (%86.67), sadece %2.54 kayıp
2. **Magnitude Pruning** - İkinci en yüksek doğruluk (%85.83)
3. **Wanda Pruning** - Üçüncü sırada (%83.30)
4. **Coverage Pruning** - En düşük doğruluk (%82.69)

### Model Optimizasyonu

- **%18.8 model boyutu azalması** (68.11 MB → 55.35 MB)
- **%18.5 FLOPs azalması** (4.61G → 3.76G)
- Taylor ve Magnitude yöntemlerinde çıkarım süresi korundu (~0.56 ms)
- Coverage ve Wanda yöntemlerinde çıkarım süresi önemli ölçüde arttı (~5.1 ms)

### Doğruluk-Verimlilik Dengesi

- **Taylor Pruning:** En iyi doğruluk-verimlilik dengesi, %97.2 orijinal performans korundu
- **Magnitude Pruning:** Çok iyi performans, %96.2 orijinal performans korundu
- **Wanda Pruning:** İyi performans, %93.4 orijinal performans
- **Coverage Pruning:** En fazla doğruluk kaybı, %92.7 orijinal performans

### EfficientNet-B4 ile Özel Bulgular

- **Klasik yöntemler (Taylor, Magnitude) EfficientNet mimarisinde çok başarılı**
- ResNet50'ye göre daha yüksek doğruluk korunma oranı (%96-97 vs %75-76)
- EfficientNet'in kompakt yapısı pruning'e daha iyi adapte oluyor
- Coverage ve Wanda yöntemlerinde beklenmeyen çıkarım süresi artışı
- **Taylor yöntemi Food101 veri setinde mükemmel sonuçlar verdi**

### Model Karşılaştırması

- **EfficientNet-B4** (TS5): %86.67 doğruluk (Taylor), %97.2 performans korunma
- **ResNet50 Layer-wise** (TS3): %75.55 doğruluk (Magnitude), %96.1 performans korunma
- **ResNet50 Global** (TS4): %76.83 doğruluk (Taylor), %97.8 performans korunma
- EfficientNet-B4 hem daha yüksek başlangıç doğruluğu hem de daha iyi pruning sonuçları gösterdi

### Temel Bulgular

- Taylor ve Magnitude pruning EfficientNet-B4 için ideal seçimler
- EfficientNet mimarisi pruning işlemlerine ResNet'ten daha dirençli
- Layer-wise pruning stratejisi EfficientNet için uygun
- Food101 gibi özelleşmiş veri setlerinde yüksek performans korunabildi
