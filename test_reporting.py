# -*- coding: utf-8 -*-
"""
CleanAI Reporting Test with ResNet-18
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("CleanAI - ResNet-18 Pruning & Reporting Test")
print("=" * 80)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# 1. Load ResNet-18
print("\n1. Loading ResNet-18 model...")
model = torchvision.models.resnet18(weights='DEFAULT')
model.eval()

# Modify for CIFAR-10 (10 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model = model.to(device)

print("   [OK] Model loaded")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 2. Prepare dataset
print("\n2. Preparing CIFAR-10 dataset...")
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Use small subset for fast testing
test_subset = torch.utils.data.Subset(test_dataset, range(500))
test_loader = torch.utils.data.DataLoader(
    test_subset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print(f"   [OK] Dataset ready (500 samples)")

# 3. Import CleanAI
print("\n3. Importing CleanAI...")
from cleanai import CoveragePruner, generate_pruning_report

print("   [OK] CleanAI imported successfully")

# 4. Setup pruner
print("\n4. Setting up pruner...")
example_inputs = torch.randn(1, 3, 224, 224).to(device)
original_model = torchvision.models.resnet18(weights='DEFAULT')
original_model.fc = nn.Linear(num_features, 10)
original_model = original_model.to(device)
original_model.eval()

pruner = CoveragePruner(
    model=model,
    example_inputs=example_inputs,
    test_loader=test_loader,
    importance_method='coverage',
    global_pruning=False,
    device=device,
    pruning_ratio=0.3,
    max_batches=10  # Use 10 batches for fast coverage analysis
)

print("   [OK] Pruner initialized")

# 5. Perform pruning (coverage is automatically analyzed)
print("\n5. Pruning model (30% pruning ratio)...")
print("   (Coverage analysis included automatically)")

pruning_info = pruner.prune()
pruned_model = pruner.model  # Get the pruned model

print(f"   [OK] Pruning complete")
print(f"   Before: {sum(p.numel() for p in original_model.parameters()):,} parameters")
print(f"   After:  {sum(p.numel() for p in pruned_model.parameters()):,} parameters")

# 6. Generate report using API
print("\n6. Generating comprehensive PDF report...")
print("   This may take 1-2 minutes...")

# Get coverage analyzer if available
coverage_analyzer = None
if hasattr(pruner.importance, 'coverage_analyzer'):
    coverage_analyzer = pruner.importance.coverage_analyzer

report_path = generate_pruning_report(
    model_before=original_model,
    model_after=pruned_model,
    model_name="ResNet-18",
    dataloader=test_loader,
    device=device,
    report_name="resnet18_test_report",
    coverage_analyzer=coverage_analyzer,
    pruning_method="coverage",
    pruning_ratio=0.3,
    iterative_steps=1,
    dataset_name="CIFAR-10",
    output_dir="reports"
)

print("\n" + "=" * 80)
print("[SUCCESS] Test completed successfully!")
print("=" * 80)
print(f"\n[REPORT] Report saved to: {report_path}")
print("\nReport includes:")
print("  [OK] Executive Summary")
print("  [OK] Model & Experiment Information")
print("  [OK] Coverage Analysis with Charts")
print("  [OK] Pruning Decision Explanations")
print("  [OK] Post-Pruning Structure")
print("  [OK] Performance Comparison")
print("  [OK] Risk Analysis & Recommendations")
print("\nOpen the PDF to view detailed analysis!")
print("=" * 80)
