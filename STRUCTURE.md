# CleanAI Proje Yapısı - Değişiklik Özeti

## 🔄 Eski vs Yeni Yapı

### ❌ Eski Yapı (Dağınık)

```
CleanAI_v5/
├── coverage_importance.py        # 3 sınıf tek dosyada
├── coverage_analyzer.py
├── coverage_pruner.py
├── utils.py                      # Tüm utility'ler
├── example_simple.py
├── example_wanda.py
└── main.py
```

### ✅ Yeni Yapı (Modüler)

```
CleanAI_v5/
├── cleanai/                      # Ana Python paketi
│   ├── __init__.py              # Tek yerden import
│   │
│   ├── importance/               # Importance metrikleri
│   │   ├── __init__.py
│   │   ├── coverage.py          # NeuronCoverageImportance
│   │   ├── wanda.py             # WandaImportance
│   │   └── adaptive.py          # AdaptiveNeuronCoverageImportance
│   │
│   ├── analyzers/                # Analiz araçları
│   │   ├── __init__.py
│   │   └── coverage_analyzer.py # CoverageAnalyzer, ActivationHook
│   │
│   ├── pruners/                  # Pruning algoritmaları
│   │   ├── __init__.py
│   │   └── coverage_pruner.py   # CoveragePruner
│   │
│   └── utils/                    # Yardımcı fonksiyonlar
│       ├── __init__.py
│       ├── model_utils.py       # Model analizi
│       └── evaluation.py        # Değerlendirme
│
├── examples/                     # Örnek scriptler
│   ├── __init__.py
│   ├── simple_pruning.py
│   └── wanda_comparison.py
│
├── main.py                       # Ana script
├── requirements.txt
├── README.md
└── QUICK_START.md
```

## 📦 Modül İçerikleri

### `cleanai/importance/`

- **coverage.py**: Coverage-based importance (aktivasyon pattern'leri)
- **wanda.py**: WANDA yöntemi (Weight × Activation)
- **adaptive.py**: Adaptive coverage (iterative pruning için)

### `cleanai/analyzers/`

- **coverage_analyzer.py**: Aktivasyon toplama ve coverage hesaplama

### `cleanai/pruners/`

- **coverage_pruner.py**: Yüksek seviye pruning interface

### `cleanai/utils/`

- **model_utils.py**: Parametre sayma, FLOPs, model kaydetme
- **evaluation.py**: Accuracy, inference time, karşılaştırma

## 🎯 Import Değişiklikleri

### Eski Import Yöntemi

```python
from coverage_importance import NeuronCoverageImportance, WandaImportance
from coverage_pruner import CoveragePruner
from coverage_analyzer import CoverageAnalyzer
from utils import evaluate_model, count_parameters
```

### Yeni Import Yöntemi

```python
# Tek satırda hepsi:
from cleanai import (
    NeuronCoverageImportance,
    WandaImportance,
    CoveragePruner,
    CoverageAnalyzer,
    evaluate_model,
    count_parameters
)

# veya spesifik modüllerden:
from cleanai.importance import WandaImportance
from cleanai.pruners import CoveragePruner
from cleanai.utils import evaluate_model
```

## 📊 Dosya Boyutları ve Satır Sayıları

| Eski Dosya             | Satır | →   | Yeni Modül             | Satır |
| ---------------------- | ----- | --- | ---------------------- | ----- |
| coverage_importance.py | 499   | →   | importance/coverage.py | ~230  |
|                        |       | →   | importance/wanda.py    | ~230  |
|                        |       | →   | importance/adaptive.py | ~70   |
| utils.py               | 403   | →   | utils/model_utils.py   | ~190  |
|                        |       | →   | utils/evaluation.py    | ~250  |

## ✨ Avantajlar

### 1. Daha İyi Organizasyon

- Her sınıf kendi dosyasında
- İlgili fonksiyonlar gruplandırılmış
- Dependency'ler daha açık

### 2. Kolay Bakım

- Bir modül değişince diğerleri etkilenmiyor
- Test yazmak daha kolay
- Kod tekrarı azaldı

### 3. Genişletilebilirlik

- Yeni importance metriği eklemek kolay
- Yeni analyzer eklemek kolay
- Mevcut kodu bozmadan ekleme

### 4. Profesyonel Yapı

- Python paket standartlarına uygun
- pip ile kurulabilir hale getirilebilir
- Dokümantasyon daha düzenli

### 5. Temiz Import'lar

```python
# Eski:
from coverage_importance import NeuronCoverageImportance, AdaptiveNeuronCoverageImportance, WandaImportance

# Yeni:
from cleanai import NeuronCoverageImportance, AdaptiveNeuronCoverageImportance, WandaImportance
```

## 🔧 Backward Compatibility

Eski dosyalar hala root dizinde duruyor (silmedik), böylece:

- Eski scriptler çalışmaya devam eder
- Yavaş yavaş yeni yapıya geçilebilir
- Test ve karşılaştırma yapılabilir

## 📝 Yapılacaklar Listesi (Tamamlandı)

- ✅ Klasör yapısı oluşturuldu
- ✅ Importance sınıfları ayrıldı (coverage, wanda, adaptive)
- ✅ Analyzer modülü taşındı
- ✅ Pruner modülü taşındı ve güncellendi
- ✅ Utils modülü organize edildi (model_utils, evaluation)
- ✅ Örnek scriptler examples/ klasörüne taşındı
- ✅ Tüm **init**.py dosyaları oluşturuldu
- ✅ main.py güncellendi
- ✅ README.md güncellendi
- ✅ QUICK_START.md oluşturuldu

## 🚀 Sonraki Adımlar (Opsiyonel)

1. **Unit testler ekle**: `tests/` klasörü oluştur
2. **CI/CD setup**: GitHub Actions
3. **Pip paketi yap**: `setup.py` ekle
4. **Dokümantasyon**: Sphinx ile API docs
5. **Benchmark suite**: Farklı modeller ve veri setleri
6. **Eski dosyaları temizle**: Root'taki eski .py dosyalarını sil

## 💡 Kullanım Örnekleri

### Yeni Yapıyla Hızlı Proje Başlatma

```python
# app.py
from cleanai import CoveragePruner, evaluate_model
import torch
from torchvision import models

def main():
    model = models.resnet18(pretrained=True)
    device = torch.device('cuda')

    pruner = CoveragePruner(
        model=model,
        example_inputs=torch.randn(1, 3, 224, 224),
        test_loader=test_loader,
        pruning_ratio=0.3,
        importance_method='wanda',
        device=device
    )

    pruned_model = pruner.prune()
    accuracy = evaluate_model(pruned_model, test_loader, device)
    print(f"Pruned model accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
```

## 📚 Kaynaklar

- **Main README**: Tam dokümantasyon
- **QUICK_START**: Hızlı başlangıç kılavuzu
- **examples/**: Çalışan örnekler
- **cleanai/**: Kaynak kodlar (docstring'lerle)
