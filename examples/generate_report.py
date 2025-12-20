"""
Example: Generate Pruning Report

This example demonstrates how to generate a comprehensive PDF report
after pruning a neural network.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from cleanai import CoveragePruner, generate_pruning_report


def main():
    """Main function."""
    print("=" * 80)
    print("CleanAI - Pruning Report Generation Example")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load a simple model
    print("\n1. Loading Model...")
    model = torchvision.models.resnet18(weights='DEFAULT')
    model.eval()
    
    # Modify the final layer for CIFAR-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model = model.to(device)
    
    print(f"   Model: ResNet-18")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare data
    print("\n2. Preparing Dataset...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Use a small subset for demo
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Use smaller subset for faster demo
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    
    print(f"   Test samples: {len(test_subset)}")
    
    # Clone model before pruning
    model_before = model
    
    # Prune the model
    print("\n3. Pruning Model...")
    pruner = CoveragePruner(
        model=model,
        example_inputs=torch.randn(1, 3, 224, 224).to(device),
        importance_method='coverage',  # Use coverage-based pruning
        global_pruning=False
    )
    
    # Analyze coverage
    print("   - Analyzing neuron coverage...")
    pruner.analyze_coverage(
        dataloader=test_loader,
        device=device,
        num_samples=500  # Use subset for faster analysis
    )
    
    # Perform pruning
    pruning_ratio = 0.3  # Prune 30% of parameters
    print(f"   - Pruning {pruning_ratio*100:.0f}% of parameters...")
    
    pruned_model = pruner.prune(
        pruning_ratio=pruning_ratio,
        iterative_steps=1
    )
    
    model_after = pruned_model.to(device)
    
    print("\n   Pruning complete!")
    print(f"   Before: {sum(p.numel() for p in model_before.parameters()):,} parameters")
    print(f"   After:  {sum(p.numel() for p in model_after.parameters()):,} parameters")
    
    # Generate comprehensive report
    print("\n4. Generating Comprehensive PDF Report...")
    print("   This may take a minute...")
    
    report_path = generate_pruning_report(
        model_before=model_before,
        model_after=model_after,
        model_name="ResNet-18",
        dataloader=test_loader,
        device=device,
        report_name="resnet18_coverage_pruning_report",  # Custom report name
        coverage_analyzer=pruner.coverage_analyzer,
        pruning_method="coverage",
        pruning_ratio=pruning_ratio,
        iterative_steps=1,
        dataset_name="CIFAR-10",
        output_dir="reports"
    )
    
    print("\n" + "=" * 80)
    print(f"✅ Report successfully generated!")
    print(f"📄 Report location: {report_path}")
    print("=" * 80)
    
    print("\nReport Contents:")
    print("  1. Cover Page & Executive Summary")
    print("  2. Model & Experiment Information")
    print("  3. Pre-Pruning Coverage Analysis")
    print("  4. Pruning Decision Mechanism (Explainability)")
    print("  5. Post-Pruning Model Structure")
    print("  6. Performance Comparison (with charts)")
    print("  7. Risk Analysis & Reliability Assessment")
    
    print("\nYou can now open the PDF report to view detailed analysis!")


if __name__ == '__main__':
    main()
