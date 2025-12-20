"""
Pruning Ratio Analizi - Neden %30 hedef → %51 gerçek?

Bu script pruning'in her katmandaki etkisini detaylı analiz eder.
"""

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def count_layer_parameters(module, name=""):
    """Her katmanın parametre sayısını hesapla"""
    params = 0
    details = []
    
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_params = m.weight.numel()
            if m.bias is not None:
                layer_params += m.bias.numel()
            params += layer_params
            full_name = f"{name}.{n}" if name else n
            details.append({
                'name': full_name,
                'type': 'Conv2d',
                'shape': f"{m.in_channels}→{m.out_channels}",
                'params': layer_params,
                'weight_shape': tuple(m.weight.shape)
            })
            
        elif isinstance(m, nn.BatchNorm2d):
            layer_params = 0
            if m.weight is not None:
                layer_params += m.weight.numel()
            if m.bias is not None:
                layer_params += m.bias.numel()
            params += layer_params
            full_name = f"{name}.{n}" if name else n
            details.append({
                'name': full_name,
                'type': 'BatchNorm2d',
                'shape': f"{m.num_features}",
                'params': layer_params,
            })
            
        elif isinstance(m, nn.Linear):
            layer_params = m.weight.numel()
            if m.bias is not None:
                layer_params += m.bias.numel()
            params += layer_params
            full_name = f"{name}.{n}" if name else n
            details.append({
                'name': full_name,
                'type': 'Linear',
                'shape': f"{m.in_features}→{m.out_features}",
                'params': layer_params,
                'weight_shape': tuple(m.weight.shape)
            })
    
    return params, details


def analyze_pruning_effect():
    """Pruning'in her katmandaki etkisini analiz et"""
    
    print("="*80)
    print("PRUNING EFFECT ANALYSIS - Why 30% target → 51% actual?")
    print("="*80)
    
    # Orijinal model
    model = SimpleCNN(num_classes=10)
    total_params, layer_details = count_layer_parameters(model)
    
    print("\n📊 ORIGINAL MODEL LAYER-BY-LAYER:")
    print("-"*80)
    
    cumulative_params = 0
    for detail in layer_details:
        cumulative_params += detail['params']
        percentage = (detail['params'] / total_params) * 100
        print(f"{detail['name']:30s} {detail['type']:15s} "
              f"{detail['shape']:20s} {detail['params']:>12,} params ({percentage:>5.1f}%)")
    
    print("-"*80)
    print(f"{'TOTAL':30s} {'':<15s} {'':<20s} {total_params:>12,} params (100.0%)")
    
    # Pruning simulation - Her katman %30 azalıyor
    print("\n\n🔪 SIMULATED PRUNING (30% per layer, uniform):")
    print("-"*80)
    
    pruning_ratio = 0.3
    
    simulated_params = {
        'features.0 (Conv2d)': {
            'original': 3 * 64 * 3 * 3,  # 3→64, 3x3 kernel
            'pruned': 3 * int(64 * (1-pruning_ratio)) * 3 * 3,
            'note': 'Input channels sabit (3), output 30% azalır'
        },
        'features.0 (BatchNorm)': {
            'original': 64 * 2,  # weight + bias
            'pruned': int(64 * (1-pruning_ratio)) * 2,
            'note': 'Conv output kanallarına bağlı'
        },
        'features.4 (Conv2d)': {
            'original': 64 * 128 * 3 * 3,
            'pruned': int(64 * (1-pruning_ratio)) * int(128 * (1-pruning_ratio)) * 3 * 3,
            'note': 'Hem input hem output 30% azalır (önceki Conv\'den)'
        },
        'features.4 (BatchNorm)': {
            'original': 128 * 2,
            'pruned': int(128 * (1-pruning_ratio)) * 2,
            'note': 'Conv output kanallarına bağlı'
        },
        'features.8 (Conv2d)': {
            'original': 128 * 256 * 3 * 3,
            'pruned': int(128 * (1-pruning_ratio)) * int(256 * (1-pruning_ratio)) * 3 * 3,
            'note': 'Hem input hem output 30% azalır'
        },
        'features.8 (BatchNorm)': {
            'original': 256 * 2,
            'pruned': int(256 * (1-pruning_ratio)) * 2,
            'note': 'Conv output kanallarına bağlı'
        },
        'classifier.0 (Linear)': {
            'original': (256 * 4 * 4) * 512,  # 2,097,152 params!
            'pruned': (int(256 * (1-pruning_ratio)) * 4 * 4) * int(512 * (1-pruning_ratio)),
            'note': '⚠️ ÇOK BÜYÜK ETKİ! Input size Conv3\'e bağlı'
        },
        'classifier.3 (Linear)': {
            'original': 512 * 10,
            'pruned': int(512 * (1-pruning_ratio)) * 10,  # Output korundu (final layer)
            'note': 'Input önceki Linear\'dan, output sabit (10 class)'
        }
    }
    
    total_original = 0
    total_pruned = 0
    
    print(f"{'Layer':35s} {'Original':>12s} {'Pruned':>12s} {'Reduction':>10s} {'Note':>30s}")
    print("-"*80)
    
    for layer_name, info in simulated_params.items():
        orig = info['original']
        pruned = info['pruned']
        reduction = ((orig - pruned) / orig) * 100
        
        total_original += orig
        total_pruned += pruned
        
        marker = " 🎯" if "LINEAR" in layer_name.upper() or "classifier" in layer_name else ""
        print(f"{layer_name + marker:35s} {orig:>12,} {pruned:>12,} {reduction:>9.1f}% "
              f"{info['note'][:40]:>30s}")
    
    print("-"*80)
    total_reduction = ((total_original - total_pruned) / total_original) * 100
    print(f"{'TOTAL':35s} {total_original:>12,} {total_pruned:>12,} {total_reduction:>9.1f}%")
    
    # Key insights
    print("\n\n💡 KEY INSIGHTS:")
    print("-"*80)
    print("1. ⚡ CASCADE EFFECT:")
    print("   Conv katmanlarında %30 pruning → Sonraki katmanların INPUT'u da azalır")
    print("   Örnek: Conv1 (64→45) budandı → Conv2'nin input 64→45 oldu")
    print("   Bu zincirleme bir reaksiyon yaratır!\n")
    
    print("2. 🎯 LINEAR LAYER DOMINANCE:")
    classifier_orig = simulated_params['classifier.0 (Linear)']['original']
    classifier_percentage = (classifier_orig / total_original) * 100
    print(f"   classifier.0 (Linear): {classifier_orig:,} params ({classifier_percentage:.1f}% of total!)")
    print("   Conv kanalları azalınca → Linear input size küçülür → DEV parameter azalması!")
    print("   256 channels → 179 channels olunca:")
    print(f"   Linear input: 256×4×4=4096 → 179×4×4=2864 (-%30)")
    print(f"   Ama parametreler: 4096×512 → 2864×358 = 2,097,152 → 1,025,312 (-%51!))\n")
    
    print("3. 🔗 DEPENDENCY PROPAGATION:")
    print("   Structured pruning = Tüm bağlı katmanlar birlikte budanır")
    print("   Conv output channel azalır → BN, ReLU, Next Conv INPUT hepsi azalır")
    print("   Bu yüzden 'local %30' global'de farklı bir orana dönüşür\n")
    
    print("4. ✅ BU NORMAL BIR DURUM:")
    print("   Torch-Pruning dependency graph'i doğru kullanıyor")
    print("   Model yapısal olarak küçültüldü (gerçek hız artışı var)")
    print("   Hedef %30 'uniform per layer' idi → Global etki farklı olabilir")
    
    print("\n" + "="*80)
    print("SONUÇ: %51 reduction DOĞRU ve BEKLENİYOR! ✓")
    print("="*80)
    print("\n💡 TIP: Eğer tam %30 global reduction istiyorsan:")
    print("   pruner = CoveragePruner(..., global_pruning=True, pruning_ratio=0.3)")
    print("   Bu, tüm modelde toplam %30 azalma hedefler.")
    print("="*80)


if __name__ == '__main__':
    analyze_pruning_effect()
