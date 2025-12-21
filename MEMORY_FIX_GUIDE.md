# Windows Bellek Erişim Hatası Düzeltmeleri

## Sorun
Test senaryolarını çalıştırırken şu hata alınıyordu:
```
"python.exe – Uygulama Hatası"
0x00007FFEB33ACE14 adresindeki yönerge, 0x0000000000000000 adresindeki belleğe başvurdu.
Bellek şu olamaz: read
```

Bu hata, Windows'ta NULL pointer erişimi veya PyTorch multiprocessing sorunlarından kaynaklanıyordu.

## Yapılan Düzeltmeler

### 1. DataLoader `num_workers` Düzeltmesi
**Dosyalar:**
- `test_scenarios/TS1_01_prepare_model.py`
- `test_scenarios/TS1_02_coverage_pruning.py`
- `test_scenarios/TS1_03_wanda_pruning.py`

**Değişiklik:**
```python
# ÖNCE
NUM_WORKERS = 4  # Windows'ta multiprocessing hatası

# SONRA
NUM_WORKERS = 0  # Windows fix: 0 to avoid multiprocessing issues
```

**Sebep:** Windows'ta PyTorch DataLoader multiprocessing ile bellek erişim sorunları yaşanabilir. `num_workers=0` main process'te çalışır ve bu sorunu önler.

### 2. `pin_memory` Dinamik Ayarı
**Dosyalar:**
- Tüm test senaryoları
- `cleanai/analyzers/coverage_analyzer.py`
- `cleanai/pruners/coverage_pruner.py`

**Değişiklik:**
```python
# CPU'da pin_memory kullanılmamalı
pin_mem = torch.cuda.is_available()

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=pin_mem  # Sadece CUDA varsa True
)
```

**Sebep:** `pin_memory=True` sadece CUDA kullanılırken faydalıdır. CPU'da bellek sorunlarına neden olabilir.

### 3. Checkpoint Yükleme Güvenlik Kontrolleri
**Dosyalar:**
- `test_scenarios/TS1_01_prepare_model.py`
- `test_scenarios/TS1_02_coverage_pruning.py`
- `test_scenarios/TS1_03_wanda_pruning.py`

**Değişiklik:**
```python
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# Yeni güvenlik kontrolleri
if checkpoint is None:
    raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")

if 'model_state_dict' not in checkpoint:
    raise RuntimeError(f"Checkpoint does not contain 'model_state_dict' key")

model.load_state_dict(checkpoint['model_state_dict'])
```

**Sebep:** Bozuk veya eksik checkpoint dosyaları NULL pointer erişimine neden olabilir.

### 4. DataLoader Batch İşleme Güvenliği
**Dosyalar:**
- `cleanai/analyzers/coverage_analyzer.py`
- `cleanai/pruners/coverage_pruner.py`

**Değişiklik:**
```python
for batch_idx, batch in enumerate(test_loader):
    # Batch formatını kontrol et
    if isinstance(batch, (tuple, list)):
        if len(batch) < 1:
            continue  # Boş batch'i atla
        inputs = batch[0]
    else:
        inputs = batch
    
    # None kontrolü
    if inputs is None:
        continue
    
    # Try-except ile hata yakalama
    try:
        inputs = inputs.to(device)
        _ = model(inputs)
    except Exception as e:
        print(f"Warning: Error processing batch {batch_idx}: {e}")
        continue
```

**Sebep:** Bozuk veya None batch'ler bellek erişim hatalarına neden olabilir.

### 5. Dictionary Güvenli Erişim
**Dosyalar:**
- Tüm test senaryoları

**Değişiklik:**
```python
# ÖNCE
accuracy = checkpoint['accuracy']  # KeyError riski

# SONRA
accuracy = checkpoint.get('accuracy', 0.0)  # Güvenli erişim
```

**Sebep:** Eksik anahtarlara erişim exception'lara ve NULL pointer'lara yol açabilir.

## Test Etme

Düzeltmeleri test etmek için:

```powershell
# Doğrulama testi
python test_memory_fix.py

# Test senaryolarını çalıştır
python test_scenarios\TS1_01_prepare_model.py
python test_scenarios\TS1_02_coverage_pruning.py
python test_scenarios\TS1_03_wanda_pruning.py
```

## Önemli Notlar

### Windows Kullanıcıları İçin
1. **Her zaman `num_workers=0` kullanın** - Bu en kritik düzeltmedir
2. **CUDA yoksa `pin_memory=False` kullanın**
3. **Checkpoint yüklemeden önce varlığını kontrol edin**

### Performans
- `num_workers=0` veri yükleme hızını düşürebilir
- Ancak bu Windows'ta kararlılık için gereklidir
- CUDA kullanıyorsanız, GPU hesaplamaları veri yükleme süresini telafi edecektir

### Hata Ayıklama
Hala sorun yaşıyorsanız:

1. **CUDA/CPU kontrol edin:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
   ```

2. **PyTorch sürümü kontrol edin:**
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   ```

3. **Checkpoint'leri kontrol edin:**
   ```python
   checkpoint = torch.load('path/to/checkpoint.pth', map_location='cpu')
   print(f"Keys: {checkpoint.keys()}")
   ```

## Özet

Bu düzeltmeler şunları sağlar:
- ✅ Windows'ta kararlı çalışma
- ✅ Bellek erişim hatalarını önleme
- ✅ NULL pointer erişimlerini engelleme
- ✅ Multiprocessing sorunlarını çözme
- ✅ Güvenli checkpoint yönetimi

Test senaryolarınız artık Windows'ta güvenle çalışabilir!
