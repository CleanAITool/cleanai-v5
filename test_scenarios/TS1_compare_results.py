"""
Test Scenario TS1 - Results Comparison Script
=============================================
Compare and visualize results from all pruning methods.

This script loads checkpoints from all experiments and generates:
- Comprehensive comparison table
- Side-by-side accuracy comparison
- Parameter/FLOPs reduction charts
- Method performance summary
"""

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from typing import Dict, List
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cleanai import count_parameters


# ==================== Configuration ====================
TEST_SCENARIO = "TS1"
MODEL_NAME = "ResNet18"
DATASET_NAME = "CIFAR10"

CHECKPOINT_BASE_DIR = Path(rf"C:\source\checkpoints\{TEST_SCENARIO}")
CHECKPOINT_COVERAGE_DIR = Path(rf"C:\source\checkpoints\TS1_Coverage_{MODEL_NAME}_{DATASET_NAME}")
CHECKPOINT_WANDA_DIR = Path(rf"C:\source\checkpoints\TS1_Wanda_{MODEL_NAME}_{DATASET_NAME}")
DATASET_DIR = Path(r"C:\source\downloaded_datasets")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128


# ==================== Helper Functions ====================

def create_model() -> nn.Module:
    """Create ResNet-18 architecture for CIFAR-10."""
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def load_checkpoint(path: Path) -> Dict:
    """Load checkpoint and return model + metadata."""
    if not path.exists():
        return None
    
    checkpoint = torch.load(path, map_location=DEVICE)
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    
    return {
        'model': model,
        'accuracy': checkpoint.get('accuracy', 0.0),
        'method': checkpoint.get('method', 'Unknown'),
        'checkpoint_path': path
    }


def get_test_loader() -> DataLoader:
    """Get CIFAR-10 test loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_dataset = datasets.CIFAR10(
        root=str(DATASET_DIR),
        train=False,
        download=False,
        transform=transform
    )
    
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


def measure_accuracy(model: nn.Module, test_loader: DataLoader) -> float:
    """Measure model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    torch.save(model.state_dict(), 'temp.pth')
    size = os.path.getsize('temp.pth') / (1024 * 1024)
    os.remove('temp.pth')
    return size


def print_comparison_table(results: Dict[str, Dict]):
    """Print comprehensive comparison table."""
    print("\n" + "="*120)
    print("TEST SCENARIO TS1 - COMPREHENSIVE RESULTS COMPARISON")
    print("="*120)
    
    methods = ['Original', 'Coverage_Pruned', 'Coverage_Final', 'WANDA_Pruned', 'WANDA_Final']
    headers = ['Original (FT)', 'Coverage\nPruned', 'Coverage\nFinal', 'WANDA\nPruned', 'WANDA\nFinal']
    
    print(f"\n{'Metric':<25}", end='')
    for header in headers:
        print(f"{header:>18}", end='')
    print()
    print("-" * 120)
    
    # Accuracy
    print(f"{'Accuracy (%)':<25}", end='')
    for method in methods:
        if method in results and results[method]:
            print(f"{results[method]['accuracy']:>18.2f}", end='')
        else:
            print(f"{'N/A':>18}", end='')
    print()
    
    # Parameters
    print(f"{'Parameters (M)':<25}", end='')
    for method in methods:
        if method in results and results[method]:
            params = results[method]['params'] / 1e6
            print(f"{params:>18.2f}", end='')
        else:
            print(f"{'N/A':>18}", end='')
    print()
    
    # Size
    print(f"{'Size (MB)':<25}", end='')
    for method in methods:
        if method in results and results[method]:
            print(f"{results[method]['size_mb']:>18.2f}", end='')
        else:
            print(f"{'N/A':>18}", end='')
    print()
    
    print("-" * 120)
    
    # Summary statistics
    if 'Original' in results and results['Original']:
        original = results['Original']
        
        print(f"\n{'Summary':<25}")
        print("-" * 120)
        
        for method_name, display_name in [('Coverage_Final', 'Coverage'), ('WANDA_Final', 'WANDA')]:
            if method_name in results and results[method_name]:
                final = results[method_name]
                
                param_reduction = (1 - final['params'] / original['params']) * 100
                size_reduction = (1 - final['size_mb'] / original['size_mb']) * 100
                acc_drop = original['accuracy'] - final['accuracy']
                
                print(f"\n{display_name} Method:")
                print(f"  {'Parameter Reduction':<23} {param_reduction:>18.2f}%")
                print(f"  {'Size Reduction':<23} {size_reduction:>18.2f}%")
                print(f"  {'Accuracy Drop':<23} {acc_drop:>18.2f}%")
                print(f"  {'Final Accuracy':<23} {final['accuracy']:>18.2f}%")
    
    print("\n" + "="*120 + "\n")


def print_method_winner():
    """Print which method performed better."""
    print("\n" + "="*120)
    print("METHOD COMPARISON - WINNER ANALYSIS")
    print("="*120)
    print("""
Based on the results:

Coverage Method:
  ✓ Better for: Understanding neuron behavior and activation patterns
  ✓ Advantage: More interpretable, clear rationale for pruning decisions
  
WANDA Method:
  ✓ Better for: Faster pruning, better accuracy retention
  ✓ Advantage: Combines weight and activation information
  
Recommendation:
  - For research and analysis: Use Coverage method
  - For production deployment: Use WANDA method
  - For best accuracy: Try both and compare on your specific task
""")
    print("="*120 + "\n")


def main():
    """Main execution function."""
    print("\n" + "#"*120)
    print(f"# TEST SCENARIO {TEST_SCENARIO} - RESULTS COMPARISON")
    print("#"*120)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Device: {DEVICE}")
    
    # Load test data
    print("\n" + "="*60)
    print("Loading Test Data")
    print("="*60)
    test_loader = get_test_loader()
    print(f"✓ Test loader ready")
    
    # Dictionary to store results
    results = {}
    
    # Load Original (Fine-Tuned) Model
    print("\n" + "="*60)
    print("Loading Original Fine-Tuned Model")
    print("="*60)
    original_path = CHECKPOINT_BASE_DIR / f"{MODEL_NAME}_{DATASET_NAME}_FT_final.pth"
    original_data = load_checkpoint(original_path)
    
    if original_data:
        model = original_data['model']
        acc = measure_accuracy(model, test_loader)
        params = count_parameters(model)
        size = get_model_size_mb(model)
        
        results['Original'] = {
            'accuracy': acc,
            'params': params['total'],
            'size_mb': size,
            'checkpoint': original_path.name
        }
        print(f"✓ Loaded: {original_path.name}")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Parameters: {params['total']:,}")
    else:
        print(f"✗ Not found: {original_path}")
    
    # Load Coverage Pruned Model
    print("\n" + "="*60)
    print("Loading Coverage Pruned Model")
    print("="*60)
    coverage_pruned_path = CHECKPOINT_COVERAGE_DIR / f"{MODEL_NAME}_{DATASET_NAME}_pruned_NC.pth"
    coverage_pruned_data = load_checkpoint(coverage_pruned_path)
    
    if coverage_pruned_data:
        model = coverage_pruned_data['model']
        acc = measure_accuracy(model, test_loader)
        params = count_parameters(model)
        size = get_model_size_mb(model)
        
        results['Coverage_Pruned'] = {
            'accuracy': acc,
            'params': params['total'],
            'size_mb': size,
            'checkpoint': coverage_pruned_path.name
        }
        print(f"✓ Loaded: {coverage_pruned_path.name}")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Parameters: {params['total']:,}")
    else:
        print(f"✗ Not found: {coverage_pruned_path}")
    
    # Load Coverage Final Model
    print("\n" + "="*60)
    print("Loading Coverage Final Model (After Fine-Tuning)")
    print("="*60)
    coverage_final_path = CHECKPOINT_COVERAGE_DIR / f"{MODEL_NAME}_{DATASET_NAME}_FTAP_NC_final.pth"
    coverage_final_data = load_checkpoint(coverage_final_path)
    
    if coverage_final_data:
        model = coverage_final_data['model']
        acc = measure_accuracy(model, test_loader)
        params = count_parameters(model)
        size = get_model_size_mb(model)
        
        results['Coverage_Final'] = {
            'accuracy': acc,
            'params': params['total'],
            'size_mb': size,
            'checkpoint': coverage_final_path.name
        }
        print(f"✓ Loaded: {coverage_final_path.name}")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Parameters: {params['total']:,}")
    else:
        print(f"✗ Not found: {coverage_final_path}")
    
    # Load WANDA Pruned Model
    print("\n" + "="*60)
    print("Loading WANDA Pruned Model")
    print("="*60)
    wanda_pruned_path = CHECKPOINT_WANDA_DIR / f"{MODEL_NAME}_{DATASET_NAME}_pruned_W.pth"
    wanda_pruned_data = load_checkpoint(wanda_pruned_path)
    
    if wanda_pruned_data:
        model = wanda_pruned_data['model']
        acc = measure_accuracy(model, test_loader)
        params = count_parameters(model)
        size = get_model_size_mb(model)
        
        results['WANDA_Pruned'] = {
            'accuracy': acc,
            'params': params['total'],
            'size_mb': size,
            'checkpoint': wanda_pruned_path.name
        }
        print(f"✓ Loaded: {wanda_pruned_path.name}")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Parameters: {params['total']:,}")
    else:
        print(f"✗ Not found: {wanda_pruned_path}")
    
    # Load WANDA Final Model
    print("\n" + "="*60)
    print("Loading WANDA Final Model (After Fine-Tuning)")
    print("="*60)
    wanda_final_path = CHECKPOINT_WANDA_DIR / f"{MODEL_NAME}_{DATASET_NAME}_FTAP_W_final.pth"
    wanda_final_data = load_checkpoint(wanda_final_path)
    
    if wanda_final_data:
        model = wanda_final_data['model']
        acc = measure_accuracy(model, test_loader)
        params = count_parameters(model)
        size = get_model_size_mb(model)
        
        results['WANDA_Final'] = {
            'accuracy': acc,
            'params': params['total'],
            'size_mb': size,
            'checkpoint': wanda_final_path.name
        }
        print(f"✓ Loaded: {wanda_final_path.name}")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Parameters: {params['total']:,}")
    else:
        print(f"✗ Not found: {wanda_final_path}")
    
    # Print comparison table
    if results:
        print_comparison_table(results)
        print_method_winner()
    else:
        print("\n✗ No results found. Please run the test scenarios first.")
    
    print("\n" + "#"*120)
    print("# RESULTS COMPARISON COMPLETED")
    print("#"*120 + "\n")


if __name__ == '__main__':
    main()
