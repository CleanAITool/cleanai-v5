"""
Test Scenario TS9: FP16 Quantization Analysis
=============================================

This script evaluates FP16 quantization on all TS9 pruned models:
- Original Fine-tuned Model
- Coverage Pruning + Fine-tuned
- Wanda Pruning + Fine-tuned  
- Magnitude Pruning + Fine-tuned
- Taylor Pruning + Fine-tuned

For each model, measures:
- Accuracy
- Average inference time per input
- Model size (checkpoint file size)
- Parameter count
- RAM usage (GPU/CPU memory)

Then applies CleanAI FP16 quantization and re-measures all metrics.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import psutil
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cleanai import ModelQuantizer, count_parameters
from cleanai.utils.evaluation import evaluate_model


# ===================== CONFIGURATION =====================
CONFIG = {
    'checkpoint_dir': r'C:\source\checkpoints\TS9',
    'dataset_dir': r'C:\source\downloaded_datasets\food101',
    'results_file': 'TS9_Quantization_Results.json',
    'batch_size': 16,  # EfficientNetB4 uses larger images, smaller batch
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'eval_samples': 5000,  # Use 5000 samples for faster evaluation
}

# Model checkpoints to evaluate
MODELS_TO_EVALUATE = {
    'Original_FT': {
        'checkpoint': 'EfficientNetB4_Food101_FT_best.pth',
        'description': 'Original Fine-tuned EfficientNetB4',
        'is_pruned': False,
    },
    'Coverage_Pruned_FT': {
        'checkpoint': 'EfficientNetB4_Food101_FTAP_NC_best.pth',
        'description': 'Coverage Pruning + Fine-tuned',
        'is_pruned': True,
    },
    'Wanda_Pruned_FT': {
        'checkpoint': 'EfficientNetB4_Food101_FTAP_W_best.pth',
        'description': 'Wanda Pruning + Fine-tuned',
        'is_pruned': True,
    },
    'Magnitude_Pruned_FT': {
        'checkpoint': 'EfficientNetB4_Food101_FTAP_MAG_best.pth',
        'description': 'Magnitude Pruning + Fine-tuned',
        'is_pruned': True,
    },
    'Taylor_Pruned_FT': {
        'checkpoint': 'EfficientNetB4_Food101_FTAP_TAY_best.pth',
        'description': 'Taylor Pruning + Fine-tuned',
        'is_pruned': True,
    },
}


# ===================== DATA LOADING =====================
def get_food101_loader(batch_size=16, num_workers=4, num_samples=None):
    """Load Food101 test dataset."""
    print(f"\nLoading Food101 test dataset...")
    print(f"Dataset directory: {CONFIG['dataset_dir']}")
    
    # Food101/EfficientNetB4 uses 380x380 images
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transform = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Use Food101 test split
    test_dataset = datasets.Food101(
        root=CONFIG['dataset_dir'],
        split='test',
        download=False,
        transform=transform
    )
    
    # Use subset if specified
    if num_samples and num_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:num_samples].tolist()
        test_dataset = Subset(test_dataset, indices)
        print(f"Using {num_samples} samples from test set")
    else:
        print(f"Using full test set ({len(test_dataset)} samples)")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader


# ===================== MODEL LOADING =====================
def load_model_from_checkpoint(checkpoint_path, is_pruned=False, device='cuda'):
    """Load model from checkpoint."""
    print(f"\nLoading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create base model
    if is_pruned:
        # For pruned models, prioritize loading the full model object
        if 'model' in checkpoint:
            model = checkpoint['model']
            print("Loaded pruned model architecture from checkpoint['model']")
        else:
            # If no 'model' key, we cannot reconstruct the pruned architecture
            # Skip this model as we cannot load it properly
            raise RuntimeError(
                f"Cannot load pruned model - checkpoint missing 'model' key.\n"
                f"Pruned models have modified architecture that cannot be loaded into base EfficientNetB4.\\n"
                f"Please re-run the pruning script to save the full model with 'model' key."
            )
    else:
        # For non-pruned models, create base EfficientNetB4 and load state dict
        from torchvision.models import efficientnet_b4
        model = efficientnet_b4(weights=None)
        
        # Modify classifier for Food101 (101 classes)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 101)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out FLOPs counting keys
        state_dict = {k: v for k, v in state_dict.items() 
                     if not k.endswith('total_ops') and not k.endswith('total_params')}
        
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model


# ===================== METRICS COLLECTION =====================
def get_model_size_mb(model, temp_path='temp_model.pth'):
    """Get model size in MB by saving to disk."""
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_mb


def get_memory_usage_mb():
    """Get current RAM usage in MB."""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)
    
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        return {
            'ram_mb': ram_mb,
            'gpu_allocated_mb': gpu_mb,
            'gpu_reserved_mb': gpu_reserved_mb,
        }
    else:
        return {
            'ram_mb': ram_mb,
            'gpu_allocated_mb': 0,
            'gpu_reserved_mb': 0,
        }


def measure_inference_time(model, data_loader, device='cuda', num_batches=50):
    """Measure average inference time per input."""
    model.eval()
    total_time = 0
    total_samples = 0
    
    # Get model dtype for FP16/BF16 models
    model_dtype = next(model.parameters()).dtype
    
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            images = images.to(device)
            
            # Match input dtype to model dtype
            if model_dtype in [torch.float16, torch.bfloat16]:
                images = images.to(model_dtype)
            
            batch_size = images.size(0)
            
            # Warmup for first batch
            if i == 0:
                _ = model(images)
                if device == 'cuda':
                    torch.cuda.synchronize()
                continue
            
            # Measure time
            start_time = time.time()
            _ = model(images)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            total_time += elapsed
            total_samples += batch_size
    
    avg_time_per_input_ms = (total_time / total_samples) * 1000
    return avg_time_per_input_ms


def evaluate_model_metrics(model, data_loader, model_name, device='cuda'):
    """Evaluate all metrics for a model."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")
    
    # Memory before
    mem_before = get_memory_usage_mb()
    
    # 1. Accuracy
    print("\n1. Measuring accuracy...")
    accuracy = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        verbose=True
    )
    
    # 2. Inference time
    print("\n2. Measuring inference time...")
    inference_time_ms = measure_inference_time(
        model=model,
        data_loader=data_loader,
        device=device,
        num_batches=50
    )
    print(f"Average inference time per input: {inference_time_ms:.4f} ms")
    
    # 3. Model size
    print("\n3. Measuring model size...")
    model_size_mb = get_model_size_mb(model)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # 4. Parameter count
    print("\n4. Counting parameters...")
    param_info = count_parameters(model)
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    
    # 5. Memory after
    mem_after = get_memory_usage_mb()
    
    metrics = {
        'accuracy': accuracy,
        'inference_time_ms': inference_time_ms,
        'model_size_mb': model_size_mb,
        'total_parameters': param_info['total'],
        'trainable_parameters': param_info['trainable'],
        'memory_before': mem_before,
        'memory_after': mem_after,
    }
    
    return metrics


# ===================== QUANTIZATION =====================
def apply_fp16_quantization(model, device='cuda'):
    """Apply FP16 quantization using CleanAI."""
    print("\nApplying FP16 quantization...")
    
    quantizer = ModelQuantizer(
        model=model,
        method='fp16',
        device=device,
        verbose=True
    )
    
    quantized_model = quantizer.quantize()
    return quantized_model


# ===================== MAIN COMPARISON =====================
def run_quantization_comparison():
    """Run complete quantization comparison."""
    print("\n" + "="*80)
    print("TS9 FP16 QUANTIZATION ANALYSIS")
    print("="*80)
    
    # Check dataset
    if not os.path.exists(CONFIG['dataset_dir']):
        print(f"\nERROR: Food101 dataset not found at: {CONFIG['dataset_dir']}")
        print("Please download Food101 dataset first.")
        return
    
    # Load data
    data_loader = get_food101_loader(
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        num_samples=CONFIG['eval_samples']
    )
    
    # Results storage
    results = {
        'config': CONFIG.copy(),
        'models': {},
    }
    
    device = CONFIG['device']
    print(f"\nUsing device: {device}")
    
    # Evaluate each model
    for model_key, model_info in MODELS_TO_EVALUATE.items():
        checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], model_info['checkpoint'])
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"\nWARNING: Checkpoint not found: {checkpoint_path}")
            print(f"Skipping {model_key}...")
            continue
        
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_key}")
        print(f"# {model_info['description']}")
        print(f"{'#'*80}")
        
        # Load original model
        try:
            model = load_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                is_pruned=model_info['is_pruned'],
                device=device
            )
        except Exception as e:
            print(f"\nERROR loading model: {e}")
            print(f"Skipping {model_key}...")
            continue
        
        # Evaluate original model
        print("\n" + "-"*80)
        print("ORIGINAL MODEL (FP32)")
        print("-"*80)
        original_metrics = evaluate_model_metrics(
            model=model,
            data_loader=data_loader,
            model_name=f"{model_key} (Original)",
            device=device
        )
        
        # Apply FP16 quantization
        print("\n" + "-"*80)
        print("APPLYING FP16 QUANTIZATION")
        print("-"*80)
        quantized_model = apply_fp16_quantization(model, device=device)
        
        # Evaluate quantized model
        print("\n" + "-"*80)
        print("QUANTIZED MODEL (FP16)")
        print("-"*80)
        quantized_metrics = evaluate_model_metrics(
            model=quantized_model,
            data_loader=data_loader,
            model_name=f"{model_key} (FP16 Quantized)",
            device=device
        )
        
        # Store results
        results['models'][model_key] = {
            'description': model_info['description'],
            'checkpoint': model_info['checkpoint'],
            'is_pruned': model_info['is_pruned'],
            'original': original_metrics,
            'fp16_quantized': quantized_metrics,
            'improvements': {
                'size_reduction_percent': ((original_metrics['model_size_mb'] - quantized_metrics['model_size_mb']) 
                                          / original_metrics['model_size_mb'] * 100),
                'speedup': original_metrics['inference_time_ms'] / quantized_metrics['inference_time_ms'],
                'accuracy_drop': original_metrics['accuracy'] - quantized_metrics['accuracy'],
            }
        }
        
        # Save checkpoint (optional)
        quantized_checkpoint_name = model_info['checkpoint'].replace('.pth', '_FP16.pth')
        quantized_checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], quantized_checkpoint_name)
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'original_checkpoint': model_info['checkpoint'],
            'quantization_method': 'fp16',
        }, quantized_checkpoint_path)
        print(f"\nSaved quantized checkpoint: {quantized_checkpoint_path}")
        
        # Cleanup
        del model, quantized_model
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Save results
    results_path = os.path.join(
        os.path.dirname(__file__),
        CONFIG['results_file']
    )
    with open(results_path, 'w') as f:
        json.dump(results, indent=4, fp=f)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary table
    print_summary_table(results)
    
    return results


# ===================== SUMMARY TABLE =====================
def print_summary_table(results):
    """Print comprehensive summary table."""
    print("\n" + "="*120)
    print("QUANTIZATION COMPARISON SUMMARY")
    print("="*120)
    
    # Header
    header = (
        f"{'Model':<30} {'Type':<10} {'Acc (%)':<10} {'Time (ms)':<12} "
        f"{'Size (MB)':<12} {'Params (M)':<12} {'RAM (MB)':<12}"
    )
    print(header)
    print("-" * 120)
    
    for model_key, model_data in results['models'].items():
        # Original model row
        orig = model_data['original']
        print(
            f"{model_key:<30} {'FP32':<10} "
            f"{orig['accuracy']:<10.2f} {orig['inference_time_ms']:<12.4f} "
            f"{orig['model_size_mb']:<12.2f} {orig['total_parameters']/1e6:<12.2f} "
            f"{orig['memory_after']['ram_mb']:<12.1f}"
        )
        
        # Quantized model row
        quant = model_data['fp16_quantized']
        imp = model_data['improvements']
        print(
            f"{model_key + ' (FP16)':<30} {'FP16':<10} "
            f"{quant['accuracy']:<10.2f} {quant['inference_time_ms']:<12.4f} "
            f"{quant['model_size_mb']:<12.2f} {quant['total_parameters']/1e6:<12.2f} "
            f"{quant['memory_after']['ram_mb']:<12.1f}"
        )
        
        # Improvement row
        print(
            f"{'  └─ Improvement':<30} {'':<10} "
            f"{-imp['accuracy_drop']:<10.2f} {imp['speedup']:<12.2f}x "
            f"{-imp['size_reduction_percent']:<12.1f}% {'':<12} {'':<12}"
        )
        print("-" * 120)
    
    print("="*120)
    
    # Additional statistics
    print("\nKEY METRICS:")
    print("-" * 80)
    
    for model_key, model_data in results['models'].items():
        imp = model_data['improvements']
        print(f"\n{model_key}:")
        print(f"  Size Reduction:     {imp['size_reduction_percent']:.2f}%")
        print(f"  Speedup:            {imp['speedup']:.2f}x")
        print(f"  Accuracy Drop:      {imp['accuracy_drop']:.2f}%")
    
    print("\n" + "="*80)


# ===================== ENTRY POINT =====================
if __name__ == '__main__':
    results = run_quantization_comparison()
    
    print("\n" + "#"*80)
    print("# TS9 Quantization Analysis Complete!")
    print("#"*80 + "\n")
