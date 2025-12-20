# CleanAI - Hızlı Başlangıç Kılavuzu

## 📁 Yeni Modüler Yapı

```
CleanAI_v5/
├── cleanai/                      # Ana paket
│   ├── importance/               # Importance metrikleri
│   │   ├── coverage.py          # Coverage-based
│   │   ├── wanda.py             # WANDA yöntemi
│   │   └── adaptive.py          # Adaptive coverage
│   ├── analyzers/                # Aktivasyon analizi
│   │   └── coverage_analyzer.py
│   ├── pruners/                  # Pruning algoritmaları
│   │   └── coverage_pruner.py
│   └── utils/                    # Yardımcı fonksiyonlar
│       ├── model_utils.py       # Model işlemleri
│       └── evaluation.py        # Değerlendirme
├── examples/                     # Örnek scriptler
│   ├── simple_pruning.py
│   └── wanda_comparison.py
└── main.py
```

## 🚀 Temel Kullanım

### 1. Basit Import

```python
# Eski yöntem (artık kullanılmıyor):
# from coverage_pruner import CoveragePruner
# from utils import evaluate_model

# Yeni modüler yöntem:
from cleanai import CoveragePruner, evaluate_model, count_parameters
```

### 2. Coverage-Based Pruning

```python
from cleanai import CoveragePruner
import torch

pruner = CoveragePruner(
    model=model,
    example_inputs=torch.randn(1, 3, 224, 224),
    test_loader=test_loader,
    pruning_ratio=0.3,
    importance_method='coverage',  # Coverage yöntemi
    global_pruning=True,
    device=device
)

pruned_model = pruner.prune()
```

### 3. WANDA Yöntemi

```python
pruner = CoveragePruner(
    model=model,
    example_inputs=example_inputs,
    test_loader=test_loader,
    pruning_ratio=0.3,
    importance_method='wanda',      # WANDA yöntemi
    max_batches=50,
    device=device
)

pruned_model = pruner.prune()
```

### 4. Adaptive Pruning

```python
pruner = CoveragePruner(
    model=model,
    example_inputs=example_inputs,
    test_loader=test_loader,
    pruning_ratio=0.5,
    importance_method='coverage',
    adaptive=True,                  # Adaptive mod
    iterative_steps=5,
    device=device
)

pruned_model = pruner.prune()
```

## 📊 Değerlendirme

### Model Karşılaştırma

```python
from cleanai import compare_models

results = compare_models(
    original_model=original_model,
    pruned_model=pruned_model,
    test_loader=test_loader,
    example_inputs=example_inputs,
    device=device
)

# Otomatik çıktı:
# - Parametre sayısı karşılaştırması
# - FLOPs karşılaştırması
# - Accuracy karşılaştırması
# - Inference time karşılaştırması
```

### Sadece Accuracy

```python
from cleanai import evaluate_model

accuracy = evaluate_model(model, test_loader, device)
print(f"Accuracy: {accuracy:.2f}%")
```

### Model İstatistikleri

```python
from cleanai import count_parameters, print_model_summary

# Parametre sayısı
params = count_parameters(model)
print(f"Total parameters: {params:,}")

# Detaylı özet
print_model_summary(model, (1, 3, 224, 224))
```

## 🎯 Mevcut Importance Yöntemleri

| Yöntem      | Açıklama                 | Hız       | Doğruluk |
| ----------- | ------------------------ | --------- | -------- |
| `coverage`  | Aktivasyon pattern'leri  | Hızlı     | İyi      |
| `wanda`     | Weight × Activation      | Hızlı     | Çok İyi  |
| `magnitude` | Sadece ağırlık büyüklüğü | Çok Hızlı | Orta     |
| `adaptive`  | Dinamik güncelleme       | Orta      | İyi      |

## 📝 Örnek Scriptler

### Simple Pruning Çalıştırma

```bash
cd examples
python simple_pruning.py
```

### WANDA Karşılaştırma

```bash
cd examples
python wanda_comparison.py
```

### Main Script

```bash
python main.py --model resnet18 --dataset cifar10 --pruning-ratio 0.3 --method wanda
```

## 🔧 Özel Importance Metriği Ekleme

```python
import torch
import torch_pruning as tp

class MyImportance(tp.importance.Importance):
    def __call__(self, group):
        # Özel importance hesaplama
        scores = compute_my_scores(group)
        return scores
```

## 💡 İpuçları

1. **Hızlı Test için**: `max_batches=10` kullanın
2. **Production için**: `max_batches=None` (tüm veri)
3. **Global pruning**: Tüm katmanlar arası optimize eder
4. **Iterative pruning**: Daha iyi sonuç, daha yavaş

## 🐛 Sorun Giderme

### Import Hatası

```python
# Eğer import hatası alırsanız:
import sys
sys.path.append('..')  # examples/ klasöründeyseniz
from cleanai import CoveragePruner
```

### CUDA Bellek Hatası

```python
# Batch size'ı düşürün:
pruner = CoveragePruner(..., max_batches=10)
```

## 📚 Daha Fazla Bilgi

- README.md: Tam dokümantasyon
- examples/: Örnek kullanımlar
- cleanai/: Kaynak kod ve docstring'ler
