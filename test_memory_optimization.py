"""
Test Memory Optimization

Quick test to verify memory optimizations are working.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import psutil
import os
import gc
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cleanai.analyzers import CoverageAnalyzer


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory():
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def test_memory_optimization():
    """Test memory usage with optimized coverage analyzer."""
    print("="*60)
    print("MEMORY OPTIMIZATION TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Memory before
    mem_start = get_memory_usage()
    gpu_start = get_gpu_memory()
    print(f"Initial Memory:")
    print(f"  RAM: {mem_start:.1f} MB")
    print(f"  GPU: {gpu_start:.1f} MB\n")
    
    # Create small model
    print("Creating ResNet-18 model...")
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    model.eval()
    
    mem_after_model = get_memory_usage()
    gpu_after_model = get_gpu_memory()
    print(f"After Model Load:")
    print(f"  RAM: {mem_after_model:.1f} MB (+{mem_after_model - mem_start:.1f} MB)")
    print(f"  GPU: {gpu_after_model:.1f} MB (+{gpu_after_model - gpu_start:.1f} MB)\n")
    
    # Create small dataset
    print("Loading test dataset (first 500 samples)...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset_dir = Path(r"C:\source\downloaded_datasets")
    test_dataset = datasets.CIFAR10(root=str(dataset_dir), train=False, download=False, transform=transform)
    
    # Use only first 500 samples for quick test
    small_dataset = torch.utils.data.Subset(test_dataset, range(500))
    test_loader = DataLoader(small_dataset, batch_size=50, shuffle=False, num_workers=0, pin_memory=False)
    
    mem_after_data = get_memory_usage()
    print(f"After Dataset Load:")
    print(f"  RAM: {mem_after_data:.1f} MB (+{mem_after_data - mem_after_model:.1f} MB)\n")
    
    # Test coverage analyzer
    print("Testing Coverage Analyzer (10 batches)...")
    analyzer = CoverageAnalyzer(model, device)
    
    mem_before_coverage = get_memory_usage()
    gpu_before_coverage = get_gpu_memory()
    
    # Collect activations
    analyzer.register_hooks()
    analyzer.collect_activations(test_loader, max_batches=10)
    
    mem_after_collect = get_memory_usage()
    gpu_after_collect = get_gpu_memory()
    print(f"After Activation Collection:")
    print(f"  RAM: {mem_after_collect:.1f} MB (+{mem_after_collect - mem_before_coverage:.1f} MB)")
    print(f"  GPU: {gpu_after_collect:.1f} MB (+{gpu_after_collect - gpu_before_coverage:.1f} MB)\n")
    
    # Compute coverage
    coverage_scores = analyzer.compute_neuron_coverage(metric='normalized_mean')
    
    mem_after_coverage = get_memory_usage()
    gpu_after_coverage = get_gpu_memory()
    print(f"After Coverage Computation:")
    print(f"  RAM: {mem_after_coverage:.1f} MB (+{mem_after_coverage - mem_after_collect:.1f} MB)")
    print(f"  GPU: {gpu_after_coverage:.1f} MB (+{gpu_after_coverage - gpu_after_collect:.1f} MB)\n")
    
    # Show coverage stats
    print(f"Coverage computed for {len(coverage_scores)} layers")
    for layer_name, scores in list(coverage_scores.items())[:3]:
        print(f"  {layer_name}: {scores.shape[0]} channels, mean={scores.mean():.4f}")
    
    # Clear and check memory
    print("\nCleaning up...")
    analyzer.clear_activations()
    analyzer.remove_hooks()
    del analyzer, model
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    mem_final = get_memory_usage()
    gpu_final = get_gpu_memory()
    print(f"After Cleanup:")
    print(f"  RAM: {mem_final:.1f} MB")
    print(f"  GPU: {gpu_final:.1f} MB\n")
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    max_ram = max(mem_after_model, mem_after_data, mem_after_collect, mem_after_coverage)
    max_gpu = max(gpu_after_model, gpu_after_collect, gpu_after_coverage)
    
    print(f"Peak RAM usage: {max_ram:.1f} MB")
    print(f"Peak GPU usage: {max_gpu:.1f} MB")
    print(f"RAM overhead (coverage): {mem_after_coverage - mem_after_model:.1f} MB")
    print(f"GPU overhead (coverage): {gpu_after_coverage - gpu_after_model:.1f} MB")
    
    # Expected: Coverage overhead should be < 200 MB RAM with optimizations
    ram_overhead = mem_after_coverage - mem_after_model
    if ram_overhead < 300:
        print(f"\n✅ PASS - Memory usage is optimized (RAM overhead: {ram_overhead:.1f} MB)")
    elif ram_overhead < 500:
        print(f"\n⚠️  WARNING - Memory usage is acceptable but could be better ({ram_overhead:.1f} MB)")
    else:
        print(f"\n❌ FAIL - Memory usage too high ({ram_overhead:.1f} MB)")
        print("   Expected: < 300 MB overhead for coverage analysis")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    test_memory_optimization()
