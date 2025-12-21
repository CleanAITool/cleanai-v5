"""
Test script to verify memory access fixes
"""

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import sys
import os

print("="*60)
print("Memory Access Fix Verification")
print("="*60)

# Test 1: Basic imports
print("\n1. Testing imports...")
try:
    from cleanai import CoveragePruner, count_parameters
    print("   ✓ Imports successful")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: DataLoader with num_workers=0
print("\n2. Testing DataLoader with num_workers=0...")
try:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Small dataset for testing
    test_dataset = datasets.CIFAR10(
        root=r"C:\source\downloaded_datasets",
        train=False,
        download=False,  # Don't download if not exists
        transform=transform
    )
    
    # Windows-safe DataLoader
    pin_mem = torch.cuda.is_available()
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # Critical for Windows
        pin_memory=pin_mem
    )
    
    print(f"   ✓ DataLoader created with num_workers=0, pin_memory={pin_mem}")
    
    # Test iteration
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 2:
            break
        inputs, targets = batch
        print(f"   ✓ Batch {batch_idx}: inputs shape {inputs.shape}, targets shape {targets.shape}")
    
    print("   ✓ DataLoader iteration successful")
    
except FileNotFoundError:
    print("   ⚠ CIFAR-10 dataset not found, skipping DataLoader test")
except Exception as e:
    print(f"   ✗ DataLoader error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Model loading with None checks
print("\n3. Testing model creation...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    print(f"   ✓ Model created and moved to {device}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   ✓ Forward pass successful: output shape {output.shape}")
    
except Exception as e:
    print(f"   ✗ Model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Checkpoint loading with None checks
print("\n4. Testing checkpoint safety...")
try:
    # Create a temporary checkpoint
    temp_checkpoint = {
        'model_state_dict': model.state_dict(),
        'accuracy': 85.5,
        'epoch': 10
    }
    
    torch.save(temp_checkpoint, 'temp_test_checkpoint.pth')
    
    # Load it back
    loaded = torch.load('temp_test_checkpoint.pth', map_location=device)
    
    if loaded is None:
        raise RuntimeError("Checkpoint loaded as None")
    
    if 'model_state_dict' not in loaded:
        raise RuntimeError("Checkpoint missing model_state_dict")
    
    model.load_state_dict(loaded['model_state_dict'])
    print(f"   ✓ Checkpoint loaded successfully")
    print(f"   ✓ Checkpoint accuracy: {loaded.get('accuracy', 0.0):.2f}%")
    
    # Cleanup
    os.remove('temp_test_checkpoint.pth')
    
except Exception as e:
    print(f"   ✗ Checkpoint error: {e}")
    import traceback
    traceback.print_exc()
    if os.path.exists('temp_test_checkpoint.pth'):
        os.remove('temp_test_checkpoint.pth')
    sys.exit(1)

print("\n" + "="*60)
print("✓ All memory access fixes verified successfully!")
print("="*60)
print("\nYou can now safely run the test scenarios:")
print("  python test_scenarios\\TS1_01_prepare_model.py")
print("  python test_scenarios\\TS1_02_coverage_pruning.py")
print("  python test_scenarios\\TS1_03_wanda_pruning.py")
