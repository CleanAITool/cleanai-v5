# Willbedeleted - Silinecek Eski Dosyalar

Bu klasör, yeni modüler yapıya geçiş sırasında artık kullanılmayan eski dosyaları içerir.

## 📁 İçindekiler

### Eski Modül Dosyaları (Yeni Versiyonları cleanai/ içinde)

1. **coverage_importance.py**

   - Yeni konum: `cleanai/importance/coverage.py`, `wanda.py`, `adaptive.py`
   - 3 ayrı modüle bölündü

2. **coverage_analyzer.py**
   - Yeni konum: `cleanai/analyzers/coverage_analyzer.py`
3. **coverage_pruner.py**
   - Yeni konum: `cleanai/pruners/coverage_pruner.py`
4. **utils.py**
   - Yeni konum: `cleanai/utils/model_utils.py` ve `evaluation.py`
   - 2 modüle bölündü

### Diğer Gereksiz Dosyalar

5. **analyze_pruning_effect.py** - Kullanılmayan test scripti
6. **New Text Document.txt** - Boş text dosyası

## ⚠️ Önemli

Bu dosyalar **henüz silinmedi**, sadece taşındı:

- Eski scriptlerle uyumluluk testi için
- Backup amaçlı
- Kod karşılaştırması için

## 🗑️ Silme

Test ve doğrulama tamamlandıktan sonra güvenle silinebilir:

```bash
# Tüm klasörü silmek için:
Remove-Item -Recurse -Force willbedeleted
```

## ✅ Yeni Yapı

Artık şunları kullanın:

```python
# Eski (artık kullanılmıyor):
# from coverage_pruner import CoveragePruner
# from utils import evaluate_model

# Yeni (kullanın):
from cleanai import CoveragePruner, evaluate_model
```

---

**Oluşturulma Tarihi**: 20 Aralık 2025
**Durum**: Silinmeyi bekliyor
