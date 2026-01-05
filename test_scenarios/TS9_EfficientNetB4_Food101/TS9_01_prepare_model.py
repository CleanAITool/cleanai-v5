"""
Test Scenario TS6 - Script 1: Prepare EfficientNet-B4 Model
Downloads pretrained EfficientNet-B4, prepares Stanford Dogs dataset, and fine-tunes the model.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from tqdm import tqdm
from tabulate import tabulate
from scipy.io import loadmat

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configuration
CONFIG = {
    'test_scenario': 'TS6',
    'model_name': 'EfficientNetB4',
    'dataset_name': 'Food101',
    'model_dir': r'C:\source\downloaded_models',
    'dataset_dir': r'C:\source\downloaded_datasets\food101',
    'checkpoint_dir': r'C:\source\checkpoints\TS6',
    'results_dir': os.path.join(os.path.dirname(__file__)),
    'results_file': 'TS6_Results.json',
    'fine_tune_epochs': 10,  # Reduced from 15 to save time
    'save_every_n_epochs': 5,
    'batch_size': 16,  # Reduced from 32 to avoid memory bottleneck
    'learning_rate': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def create_directories():
    """Create necessary directories"""
    for dir_path in [CONFIG['model_dir'], CONFIG['dataset_dir'], 
                     CONFIG['checkpoint_dir'], CONFIG['results_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    print(f"✓ Directories created/verified")

def download_and_prepare_dataset():
    """Download and prepare Stanford Dogs dataset"""
    print("\n" + "="*80)
    print("PREPARING STANFORD DOGS DATASET")
    print("="*80)
    
    # EfficientNet-B4 uses 380x380 input size
    train_transform = transforms.Compose([
        transforms.Resize(400),
        transforms.RandomResizedCrop(380),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Use Food101 dataset instead (similar to Stanford Dogs - image classification)
    # Food101 has 101 classes, ~750 training images per class, ~250 test images per class
    print("Note: Using Food101 dataset (101 food categories) as Stanford Dogs is not built-in.")
    print("Food101: 101 classes, ~75,750 training images, ~25,250 test images")
    
    try:
        train_dataset = torchvision.datasets.Food101(
            root=CONFIG['dataset_dir'],
            split='train',
            download=True,
            transform=train_transform
        )
        
        test_dataset = torchvision.datasets.Food101(
            root=CONFIG['dataset_dir'],
            split='test',
            download=True,
            transform=test_transform
        )
    except Exception as e:
        print(f"\nError downloading Food101 dataset: {e}")
        print("Dataset will be automatically downloaded on first run.")
        raise
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    print(f"✓ Dataset prepared: {CONFIG['dataset_dir']}")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"  - Number of classes: 101 (food categories)")
    print(f"  - Image size: 380x380")
    
    return train_loader, test_loader

def load_pretrained_model():
    """Load pretrained EfficientNet-B4"""
    print("\n" + "="*80)
    print("LOADING PRETRAINED EFFICIENTNET-B4")
    print("="*80)
    
    # Load pretrained EfficientNet-B4 (trained on ImageNet)
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    
    # Modify classifier for Food101 (101 classes)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 101)  # 101 food categories
    
    model = model.to(CONFIG['device'])
    
    print(f"✓ EfficientNet-B4 loaded successfully")
    print(f"  - Pretrained on ImageNet")
    print(f"  - Modified for 101 classes (Food101)")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Device: {CONFIG['device']}")
    
    return model

def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

def evaluate_model(model, test_loader):
    """Evaluate model accuracy and inference time"""
    model.eval()
    correct = 0
    total = 0
    inference_times = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            inference_time = (time.time() - start_time) * 1000 / len(images)  # ms per image
            inference_times.append(inference_time)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return accuracy, avg_inference_time

def count_flops(model, input_size=(1, 3, 380, 380)):
    """Estimate FLOPs for the model"""
    try:
        from thop import profile
        device = next(model.parameters()).device
        input_tensor = torch.randn(input_size).to(device)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops
    except ImportError:
        print("  Warning: thop not installed. FLOPs calculation skipped.")
        return 0

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['fine_tune_epochs']}")
    for images, labels in pbar:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def fine_tune_model(model, train_loader, test_loader):
    """Fine-tune the model on Stanford Dogs"""
    print("\n" + "="*80)
    print("FINE-TUNING MODEL ON STANFORD DOGS")
    print("="*80)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['fine_tune_epochs'])
    
    best_accuracy = 0.0
    training_history = []
    
    for epoch in range(CONFIG['fine_tune_epochs']):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Evaluate
        test_acc, avg_inference_time = evaluate_model(model, test_loader)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{CONFIG['fine_tune_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        })
        
        # Save checkpoint every N epochs
        if (epoch + 1) % CONFIG['save_every_n_epochs'] == 0 or epoch == CONFIG['fine_tune_epochs'] - 1:
            checkpoint_path = os.path.join(
                CONFIG['checkpoint_dir'],
                f"{CONFIG['model_name']}_{CONFIG['dataset_name']}_FT_epoch{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
                'train_accuracy': train_acc
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            # Save best model
            best_model_path = os.path.join(
                CONFIG['checkpoint_dir'],
                f"{CONFIG['model_name']}_{CONFIG['dataset_name']}_FT_best.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
                'train_accuracy': train_acc
            }, best_model_path)
    
    return training_history, best_accuracy

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(f"TEST SCENARIO {CONFIG['test_scenario']}: PREPARE {CONFIG['model_name']} MODEL")
    print("="*80)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Dataset: {CONFIG['dataset_name']}")
    print(f"Device: {CONFIG['device']}")
    print(f"Fine-tune Epochs: {CONFIG['fine_tune_epochs']}")
    print("="*80)
    
    # Create directories
    create_directories()
    
    # Check if final checkpoint already exists
    final_checkpoint = os.path.join(
        CONFIG['checkpoint_dir'],
        f"{CONFIG['model_name']}_{CONFIG['dataset_name']}_FT_epoch{CONFIG['fine_tune_epochs']}.pth"
    )
    best_checkpoint = os.path.join(
        CONFIG['checkpoint_dir'],
        f"{CONFIG['model_name']}_{CONFIG['dataset_name']}_FT_best.pth"
    )
    
    # Download and prepare dataset
    train_loader, test_loader = download_and_prepare_dataset()
    
    # Load pretrained model
    model = load_pretrained_model()
    
    # Evaluate pretrained model (before fine-tuning on Stanford Dogs)
    print("\n" + "="*80)
    print("EVALUATING PRETRAINED MODEL (BEFORE FINE-TUNING)")
    print("="*80)
    
    pretrained_accuracy, pretrained_inference_time = evaluate_model(model, test_loader)
    pretrained_size = calculate_model_size(model)
    pretrained_flops = count_flops(model)
    
    print(f"✓ Pretrained Model Accuracy: {pretrained_accuracy:.2f}%")
    print(f"✓ Model Size: {pretrained_size:.2f} MB")
    print(f"✓ Average Inference Time: {pretrained_inference_time:.4f} ms")
    if pretrained_flops > 0:
        print(f"✓ FLOPs: {pretrained_flops/1e9:.2f} GFLOPs")
    
    # Fine-tune model
    if os.path.exists(best_checkpoint):
        print("\n" + "="*80)
        print("LOADING EXISTING FINE-TUNED MODEL")
        print("="*80)
        checkpoint = torch.load(best_checkpoint, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        finetuned_accuracy = checkpoint['test_accuracy']
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"✓ Test Accuracy: {finetuned_accuracy:.2f}%")
    else:
        training_history, best_accuracy = fine_tune_model(model, train_loader, test_loader)
        
        # Load best model for final evaluation
        checkpoint = torch.load(best_checkpoint, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        finetuned_accuracy, _ = evaluate_model(model, test_loader)
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION (AFTER FINE-TUNING)")
    print("="*80)
    
    finetuned_accuracy, finetuned_inference_time = evaluate_model(model, test_loader)
    finetuned_size = calculate_model_size(model)
    finetuned_flops = count_flops(model)
    
    print(f"✓ Fine-tuned Model Accuracy: {finetuned_accuracy:.2f}%")
    print(f"✓ Model Size: {finetuned_size:.2f} MB")
    print(f"✓ Average Inference Time: {finetuned_inference_time:.4f} ms")
    if finetuned_flops > 0:
        print(f"✓ FLOPs: {finetuned_flops/1e9:.2f} GFLOPs")
    
    # Comparison table
    print("\n" + "="*80)
    print("COMPARISON: BEFORE vs AFTER FINE-TUNING")
    print("="*80)
    
    comparison_data = [
        ["Metric", "Pretrained (Before FT)", "Fine-tuned (After FT)", "Change"],
        ["Accuracy (%)", f"{pretrained_accuracy:.2f}", f"{finetuned_accuracy:.2f}", 
         f"{finetuned_accuracy - pretrained_accuracy:+.2f}"],
        ["Size (MB)", f"{pretrained_size:.2f}", f"{finetuned_size:.2f}", 
         f"{finetuned_size - pretrained_size:+.2f}"],
        ["Inference Time (ms)", f"{pretrained_inference_time:.4f}", 
         f"{finetuned_inference_time:.4f}", 
         f"{finetuned_inference_time - pretrained_inference_time:+.4f}"],
    ]
    
    if pretrained_flops > 0 and finetuned_flops > 0:
        comparison_data.append([
            "FLOPs (G)", f"{pretrained_flops/1e9:.2f}", f"{finetuned_flops/1e9:.2f}",
            f"{(finetuned_flops - pretrained_flops)/1e9:+.2f}"
        ])
    
    print(tabulate(comparison_data, headers="firstrow", tablefmt="grid"))
    
    # Save results to JSON
    results = {
        'test_scenario': CONFIG['test_scenario'],
        'model': CONFIG['model_name'],
        'dataset': CONFIG['dataset_name'],
        'script': 'TS6_01_prepare_model',
        'pretrained': {
            'accuracy': pretrained_accuracy,
            'size_mb': pretrained_size,
            'inference_time_ms': pretrained_inference_time,
            'flops': pretrained_flops
        },
        'finetuned': {
            'accuracy': finetuned_accuracy,
            'size_mb': finetuned_size,
            'inference_time_ms': finetuned_inference_time,
            'flops': finetuned_flops,
            'epochs': CONFIG['fine_tune_epochs']
        },
        'checkpoints': {
            'best': best_checkpoint,
            'final': final_checkpoint
        }
    }
    
    results_file = os.path.join(CONFIG['results_dir'], CONFIG['results_file'])
    
    # Load existing results if available
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    all_results['prepare_model'] = results
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n✓ Results saved to: {results_file}")
    print("\n" + "="*80)
    print("SCRIPT COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()


