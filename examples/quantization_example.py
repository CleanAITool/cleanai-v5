"""
Example: Using CleanAI ModelQuantizer API

Demonstrates how to use the ModelQuantizer class for:
- INT8 Dynamic Quantization
- INT8 Static Quantization
- FP16 Half Precision
- BFloat16 Precision
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Import from cleanai
from cleanai import ModelQuantizer, quantize_model, count_parameters


# ==================== Example 1: INT8 Dynamic Quantization ====================
def example_int8_dynamic():
    """INT8 dynamic quantization example."""
    print("\n" + "="*80)
    print("EXAMPLE 1: INT8 DYNAMIC QUANTIZATION")
    print("="*80)
    
    # Load a pretrained model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Method 1: Using ModelQuantizer class
    quantizer = ModelQuantizer(
        model=model,
        method='dynamic',
        dtype='int8',  # Can also use 'uint8'
        verbose=True
    )
    
    quantized_model = quantizer.quantize()
    
    # Method 2: Using quick helper function
    # quantized_model = quantize_model(model, method='dynamic', dtype='int8')
    
    return quantized_model


# ==================== Example 2: INT8 Static Quantization ====================
def example_int8_static():
    """INT8 static quantization with calibration."""
    print("\n" + "="*80)
    print("EXAMPLE 2: INT8 STATIC QUANTIZATION")
    print("="*80)
    
    # Load model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Prepare calibration data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Use a small subset for calibration (100-1000 samples is enough)
    calibration_dataset = datasets.FakeData(
        size=500,
        image_size=(3, 224, 224),
        num_classes=1000,
        transform=transform
    )
    
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Apply static quantization
    quantizer = ModelQuantizer(
        model=model,
        method='static',
        dtype='int8',
        calibration_loader=calibration_loader,
        calibration_batches=50,
        qconfig='fbgemm',  # 'fbgemm' for x86, 'qnnpack' for ARM
        verbose=True
    )
    
    quantized_model = quantizer.quantize()
    
    return quantized_model


# ==================== Example 3: FP16 Half Precision ====================
def example_fp16():
    """FP16 half precision conversion."""
    print("\n" + "="*80)
    print("EXAMPLE 3: FP16 HALF PRECISION")
    print("="*80)
    
    # Load model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Apply FP16 conversion
    quantizer = ModelQuantizer(
        model=model,
        method='fp16',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )
    
    fp16_model = quantizer.quantize()
    
    # Or use quick helper
    # fp16_model = quantize_model(model, method='fp16')
    
    return fp16_model


# ==================== Example 4: BFloat16 Precision ====================
def example_bfloat16():
    """BFloat16 precision conversion."""
    print("\n" + "="*80)
    print("EXAMPLE 4: BFLOAT16 PRECISION")
    print("="*80)
    
    # Load model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Apply BFloat16 conversion
    quantizer = ModelQuantizer(
        model=model,
        method='bf16',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )
    
    bf16_model = quantizer.quantize()
    
    return bf16_model


# ==================== Example 5: Comparison of All Methods ====================
def example_comparison():
    """Compare all quantization methods."""
    print("\n" + "="*80)
    print("EXAMPLE 5: QUANTIZATION METHODS COMPARISON")
    print("="*80)
    
    # Load model
    model = models.resnet18(pretrained=True)
    original_params = count_parameters(model)
    
    import os
    
    # Measure original model
    torch.save(model.state_dict(), 'temp_original.pth')
    original_size_mb = os.path.getsize('temp_original.pth') / (1024 * 1024)
    os.remove('temp_original.pth')
    
    results = {
        'Original': {
            'size_mb': original_size_mb,
            'params': original_params['total'],
            'dtype': 'float32'
        }
    }
    
    # INT8 Dynamic
    print("\n1. Testing INT8 Dynamic...")
    int8_dynamic = quantize_model(model, method='dynamic', dtype='int8', verbose=False)
    torch.save(int8_dynamic.state_dict(), 'temp_int8_dynamic.pth')
    results['INT8 Dynamic'] = {
        'size_mb': os.path.getsize('temp_int8_dynamic.pth') / (1024 * 1024),
        'params': count_parameters(int8_dynamic)['total'],
        'dtype': 'int8'
    }
    os.remove('temp_int8_dynamic.pth')
    
    # FP16
    print("2. Testing FP16...")
    fp16 = quantize_model(model, method='fp16', verbose=False)
    torch.save(fp16.state_dict(), 'temp_fp16.pth')
    results['FP16'] = {
        'size_mb': os.path.getsize('temp_fp16.pth') / (1024 * 1024),
        'params': count_parameters(fp16)['total'],
        'dtype': 'float16'
    }
    os.remove('temp_fp16.pth')
    
    # BFloat16
    print("3. Testing BFloat16...")
    bf16 = quantize_model(model, method='bf16', verbose=False)
    torch.save(bf16.state_dict(), 'temp_bf16.pth')
    results['BFloat16'] = {
        'size_mb': os.path.getsize('temp_bf16.pth') / (1024 * 1024),
        'params': count_parameters(bf16)['total'],
        'dtype': 'bfloat16'
    }
    os.remove('temp_bf16.pth')
    
    # Print comparison table
    print("\n" + "="*80)
    print("QUANTIZATION COMPARISON")
    print("="*80)
    print(f"{'Method':<20} {'Size (MB)':<15} {'Reduction':<15} {'Data Type':<15}")
    print("-" * 80)
    
    for method, data in results.items():
        reduction = (1 - data['size_mb'] / original_size_mb) * 100 if method != 'Original' else 0
        print(f"{method:<20} {data['size_mb']:>12.2f}   {reduction:>10.1f}%   {data['dtype']:<15}")
    
    print("="*80 + "\n")


# ==================== Run Examples ====================
if __name__ == '__main__':
    print("\n" + "#"*80)
    print("# CleanAI ModelQuantizer Examples")
    print("#"*80)
    
    # Example 1: INT8 Dynamic
    int8_model = example_int8_dynamic()
    
    # Example 2: INT8 Static (commented - requires calibration data)
    # int8_static_model = example_int8_static()
    
    # Example 3: FP16
    fp16_model = example_fp16()
    
    # Example 4: BFloat16
    # bf16_model = example_bfloat16()
    
    # Example 5: Comparison
    example_comparison()
    
    print("\n" + "#"*80)
    print("# Examples completed!")
    print("#"*80 + "\n")
