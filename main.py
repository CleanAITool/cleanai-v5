"""
Main Script for Neuron Coverage-Based Pruning

This script demonstrates how to use the coverage-based pruning framework
to prune neural networks based on neuron activation patterns from test data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from typing import Optional
import argparse
from pathlib import Path

from cleanai import (
    CoveragePruner,
    print_model_summary,
    compare_models,
    save_model,
    evaluate_model,
    count_parameters
)


def get_dataloaders(
    dataset_name: str = 'cifar10',
    data_path: str = './data',
    batch_size: int = 128,
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader]:
    """
    Get train and test dataloaders for specified dataset.
    
    Args:
        dataset_name: Name of dataset ('cifar10', 'cifar100', 'mnist')
        data_path: Path to store/load dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test
        )
        
    elif dataset_name.lower() == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        train_dataset = datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform_test
        )
        
    elif dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root=data_path, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_path, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def get_model(
    model_name: str = 'resnet18',
    num_classes: int = 10,
    pretrained: bool = False
) -> nn.Module:
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        # Modify for CIFAR (smaller input)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def main(args: argparse.Namespace) -> None:
    """
    Main pruning pipeline.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders
    print(f"\nLoading dataset: {args.dataset}")
    train_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get model
    print(f"\nLoading model: {args.model}")
    num_classes = 100 if args.dataset.lower() == 'cifar100' else 10
    model = get_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    
    # Print original model summary
    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    print("\n" + "#"*60)
    print("ORIGINAL MODEL")
    print("#"*60)
    print_model_summary(model, example_inputs)
    
    # Evaluate original model
    criterion = nn.CrossEntropyLoss()
    print("\nEvaluating original model...")
    original_metrics = evaluate_model(model, test_loader, criterion, device)
    print(f"Original Accuracy: {original_metrics['accuracy']:.4f}")
    
    # Create a copy of original model for comparison
    import copy
    original_model = copy.deepcopy(model)
    
    # Prepare ignored layers (typically the final classifier)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == num_classes:
            ignored_layers.append(m)
    
    # Initialize coverage-based pruner
    print("\n" + "#"*60)
    print("INITIALIZING COVERAGE-BASED PRUNER")
    print("#"*60)
    
    pruner = CoveragePruner(
        model=model,
        example_inputs=example_inputs,
        test_loader=test_loader,
        pruning_ratio=args.pruning_ratio,
        coverage_metric=args.coverage_metric,
        global_pruning=args.global_pruning,
        iterative_steps=args.iterative_steps,
        max_batches=args.max_batches,
        ignored_layers=ignored_layers,
        round_to=args.round_to,
        device=device,
        verbose=True,
        adaptive=args.adaptive
    )
    
    # Perform pruning
    print("\n" + "#"*60)
    print("STARTING PRUNING PROCESS")
    print("#"*60)
    
    pruning_results = pruner.prune()
    
    # Get pruned model
    pruned_model = pruner.get_model()
    
    # Print pruned model summary
    print("\n" + "#"*60)
    print("PRUNED MODEL")
    print("#"*60)
    print_model_summary(pruned_model, example_inputs)
    
    # Compare models
    comparison = compare_models(
        original_model=original_model,
        pruned_model=pruned_model,
        example_inputs=example_inputs,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # Save pruned model
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'model_name': args.model,
            'dataset': args.dataset,
            'pruning_ratio': args.pruning_ratio,
            'coverage_metric': args.coverage_metric,
            'global_pruning': args.global_pruning,
            'iterative_steps': args.iterative_steps,
            'original_params': count_parameters(original_model)['total'],
            'pruned_params': count_parameters(pruned_model)['total'],
            'original_accuracy': original_metrics.get('accuracy', 0),
            'comparison': comparison
        }
        
        save_model(pruned_model, args.save_path, metadata)
    
    print("\n" + "#"*60)
    print("PRUNING COMPLETE!")
    print("#"*60)
    
    # Print final summary
    print("\nFinal Results:")
    print(f"  Parameter Reduction: {comparison['parameters']['reduction']:.2%}")
    print(f"  FLOPs Reduction: {comparison['flops']['reduction']:.2%}")
    print(f"  Speedup: {comparison['inference_time']['speedup']:.2f}x")
    if 'accuracy' in comparison:
        print(f"  Accuracy Drop: {comparison['accuracy']['drop']*100:.2f}%")
    
    print("\nNote: Fine-tuning the pruned model is recommended to recover accuracy.")
    print("You can use standard training procedures with the pruned model.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neuron Coverage-Based Model Pruning')
    
    # Model and dataset
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'vgg16', 'mobilenet_v2'],
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist'],
                        help='Dataset to use')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model')
    
    # Pruning parameters
    parser.add_argument('--pruning-ratio', type=float, default=0.3,
                        help='Pruning ratio (0.0-1.0), e.g., 0.3 = remove 30%% of channels')
    parser.add_argument('--coverage-metric', type=str, default='normalized_mean',
                        choices=['normalized_mean', 'frequency', 'mean_absolute', 'combined'],
                        help='Coverage metric for importance estimation')
    parser.add_argument('--global-pruning', action='store_true',
                        help='Use global pruning (vs uniform per-layer)')
    parser.add_argument('--iterative-steps', type=int, default=1,
                        help='Number of iterative pruning steps')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive coverage (recompute during iterative pruning)')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of test batches for coverage analysis')
    parser.add_argument('--round-to', type=int, default=None,
                        help='Round channels to nearest multiple (e.g., 8)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Other
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--save-path', type=str, default='./pruned_model.pth',
                        help='Path to save pruned model')
    
    args = parser.parse_args()
    
    main(args)
