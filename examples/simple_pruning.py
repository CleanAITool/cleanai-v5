"""
Example Script: Simple Coverage-Based Pruning Demo

This script demonstrates a minimal example of using the coverage-based
pruning framework on a simple model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append('..')

from cleanai import CoveragePruner, print_model_summary, count_parameters


# Define a simple CNN model
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


def main():
    """Simple example of coverage-based pruning."""
    
    print("="*60)
    print("Coverage-Based Pruning - Simple Example")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create a simple model
    print("\n1. Creating model...")
    model = SimpleCNN(num_classes=10)
    model.to(device)
    
    # Create dummy test data (100 samples)
    print("\n2. Creating dummy test data...")
    test_inputs = torch.randn(100, 3, 32, 32)
    test_targets = torch.randint(0, 10, (100,))
    test_dataset = TensorDataset(test_inputs, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Example input for graph tracing
    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    
    # Print original model
    print("\n3. Original Model:")
    print_model_summary(model, example_inputs)
    original_params = count_parameters(model)['total']
    
    # Ignore the final classifier
    ignored_layers = [model.classifier[-1]]
    
    # Initialize pruner
    print("\n4. Initializing Coverage-Based Pruner...")
    print("   - Pruning ratio: 30%")
    print("   - Coverage metric: normalized_mean")
    print("   - Global pruning: False (uniform per-layer)")
    
    pruner = CoveragePruner(
        model=model,
        example_inputs=example_inputs,
        test_loader=test_loader,
        pruning_ratio=0.1,  # Remove 30% of channels
        coverage_metric='normalized_mean',
        global_pruning=False,
        iterative_steps=1,
        ignored_layers=ignored_layers,
        device=device,
        verbose=True
    )
    
    # Perform pruning
    print("\n5. Performing Pruning...") 
    results = pruner.prune()
    
    # Get pruned model
    pruned_model = pruner.get_model()
    
    # Print pruned model
    print("\n6. Pruned Model:")
    print_model_summary(pruned_model, example_inputs)
    pruned_params = count_parameters(pruned_model)['total']
    
    # Summary
    print("\n" + "="*60)
    print("Pruning Summary")
    print("="*60)
    print(f"Original parameters:  {original_params:,}")
    print(f"Pruned parameters:    {pruned_params:,}")
    print(f"Reduction:            {(1 - pruned_params/original_params)*100:.2f}%")
    print("\n✓ Pruning completed successfully!")
    print("\nNext steps:")
    print("  1. Fine-tune the pruned model on your training data")
    print("  2. Evaluate on validation set")
    print("  3. Compare with other pruning methods")
    print("="*60)


if __name__ == '__main__':
    main()
