"""
Model Quantization Utilities

Provides easy-to-use wrappers for PyTorch quantization methods including
dynamic quantization and static/post-training quantization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Set, Type, Dict, Any
import copy


class ModelQuantizer:
    """
    Universal quantization wrapper for PyTorch models.
    
    Supports:
    - Dynamic Quantization (int8: weights only, runtime activations)
    - Static Quantization (int8: weights + activations, requires calibration)
    - Half Precision (fp16: all parameters/activations to float16)
    - BFloat16 (bf16: all parameters/activations to bfloat16)
    
    Example:
        >>> # INT8 quantization
        >>> quantizer = ModelQuantizer(model, method='dynamic', dtype='int8')
        >>> quantized_model = quantizer.quantize()
        
        >>> # FP16 half precision
        >>> quantizer = ModelQuantizer(model, method='fp16')
        >>> fp16_model = quantizer.quantize()
        
        >>> # Static INT8 with calibration
        >>> quantizer = ModelQuantizer(model, method='static', dtype='int8', 
        ...                            calibration_loader=cal_loader)
        >>> quantized_model = quantizer.quantize()
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: str = 'dynamic',
        dtype: str = 'int8',
        calibration_loader: Optional[DataLoader] = None,
        calibration_batches: int = 100,
        qconfig: str = 'fbgemm',
        layer_types: Optional[Set[Type[nn.Module]]] = None,
        device: str = 'cpu',
        verbose: bool = True
    ):
        """
        Initialize ModelQuantizer.
        
        Args:
            model: PyTorch model to quantize
            method: Quantization method
                - 'dynamic': Dynamic int8 quantization
                - 'static': Static int8 quantization (requires calibration)
                - 'fp16': Half precision (float16)
                - 'bf16': BFloat16 precision
            dtype: Data type for int8 quantization
                - 'int8': Signed 8-bit integer (torch.qint8)
                - 'uint8': Unsigned 8-bit integer (torch.quint8)
            calibration_loader: DataLoader for calibration (required for static)
            calibration_batches: Number of batches to use for calibration
            qconfig: QConfig backend ('fbgemm' for x86, 'qnnpack' for ARM)
            layer_types: Layer types to quantize (None = all supported types)
            device: Device for calibration ('cpu' recommended for int8, 'cuda' ok for fp16)
            verbose: Print quantization progress
        """
        self.original_model = model
        self.method = method.lower()
        self.dtype_str = dtype.lower()
        self.calibration_loader = calibration_loader
        self.calibration_batches = calibration_batches
        self.qconfig = qconfig
        self.layer_types = layer_types
        self.device = device
        self.verbose = verbose
        
        # Convert dtype string to torch dtype
        self._setup_dtype()
        
        # Move model to appropriate device
        if self.method in ['fp16', 'bf16']:
            self.model = model.to(device)  # FP16/BF16 can use GPU
        else:
            self.model = model.cpu()  # INT8 quantization requires CPU
        
        # Original model info
        self.original_params = self._count_parameters(model)
        
        # Validate inputs
        self._validate_inputs()
    
    def _setup_dtype(self):
        """Setup data type based on method and dtype string."""
        if self.method in ['dynamic', 'static']:
            # INT8 quantization
            if self.dtype_str == 'int8':
                self.dtype = torch.qint8
            elif self.dtype_str == 'uint8':
                self.dtype = torch.quint8
            else:
                raise ValueError(f"For int8 quantization, dtype must be 'int8' or 'uint8', got '{self.dtype_str}'")
        elif self.method == 'fp16':
            self.dtype = torch.float16
        elif self.method == 'bf16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = None
    
    def _validate_inputs(self):
        """Validate quantization parameters."""
        valid_methods = ['dynamic', 'static', 'fp16', 'bf16']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{self.method}'")
        
        if self.method == 'static' and self.calibration_loader is None:
            raise ValueError("Static quantization requires calibration_loader")
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total parameters in model."""
        return sum(p.numel() for p in model.parameters())
    
    def _print(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(msg)
    
    def quantize_dynamic(self) -> nn.Module:
        """
        Apply dynamic quantization.
        
        - Weights are quantized ahead of time (int8)
        - Activations are quantized dynamically at runtime
        - Best for Linear/LSTM layers
        - No calibration required
        
        Returns:
            Quantized model
        """
        self._print("\n" + "="*60)
        self._print("APPLYING DYNAMIC QUANTIZATION")
        self._print("="*60)
        self._print(f"Quantization dtype: {self.dtype}")
        
        # Set model to eval mode
        self.model.eval()
        
        # Apply dynamic quantization
        if self.layer_types is None:
            # Quantize all supported layers
            self._print("Target layers: All supported (Linear, LSTM, GRU, etc.)")
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                dtype=self.dtype
            )
        else:
            # Quantize specific layer types
            self._print(f"Target layers: {', '.join([t.__name__ for t in self.layer_types])}")
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                qconfig_spec=self.layer_types,
                dtype=self.dtype
            )
        
        quantized_params = self._count_parameters(quantized_model)
        
        self._print(f"\n✓ Dynamic quantization completed")
        self._print(f"  - Original parameters: {self.original_params:,}")
        self._print(f"  - Quantized parameters: {quantized_params:,}")
        self._print(f"  - Note: Model runs on CPU only")
        
        return quantized_model
    
    def quantize_static(self) -> nn.Module:
        """
        Apply static/post-training quantization.
        
        - Weights are quantized (int8)
        - Activations are quantized using calibrated scale/zero-point
        - Best for Conv2d layers and full model quantization
        - Requires calibration data
        
        Returns:
            Quantized model
        """
        self._print("\n" + "="*60)
        self._print("APPLYING STATIC QUANTIZATION")
        self._print("="*60)
        self._print(f"Quantization dtype: {self.dtype}")
        self._print(f"QConfig backend: {self.qconfig}")
        self._print(f"Calibration batches: {self.calibration_batches}")
        
        # Set quantization backend
        torch.backends.quantized.engine = self.qconfig
        
        # Create a copy to avoid modifying original
        model = copy.deepcopy(self.model)
        model.eval()
        
        # Set quantization config
        model.qconfig = torch.quantization.get_default_qconfig(self.qconfig)
        
        # Fuse modules if possible (Conv+BN+ReLU, etc.)
        self._print("\nFusing modules...")
        try:
            torch.quantization.fuse_modules(model, inplace=True)
        except Exception as e:
            self._print(f"  Note: Module fusion skipped ({str(e)})")
        
        # Prepare model for quantization
        self._print("Preparing model for quantization...")
        torch.quantization.prepare(model, inplace=True)
        
        # Calibration: Run forward passes to collect activation statistics
        self._print(f"Running calibration with {self.calibration_batches} batches...")
        model.to(self.device)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.calibration_loader):
                if batch_idx >= self.calibration_batches:
                    break
                
                # Handle different batch formats
                if isinstance(batch_data, (tuple, list)):
                    inputs = batch_data[0]
                else:
                    inputs = batch_data
                
                inputs = inputs.to(self.device)
                model(inputs)
                
                if (batch_idx + 1) % 20 == 0:
                    self._print(f"  Calibration progress: {batch_idx + 1}/{self.calibration_batches}")
        
        # Convert to quantized model
        self._print("\nConverting to quantized model...")
        model.cpu()
        torch.quantization.convert(model, inplace=True)
        
        # Ensure model is in eval mode
        model.eval()
        
        quantized_params = self._count_parameters(model)
        
        self._print(f"\n✓ Static quantization completed")
        self._print(f"  - Original parameters: {self.original_params:,}")
        self._print(f"  - Quantized parameters: {quantized_params:,}")
        self._print(f"  - Calibration samples: {batch_idx + 1 * self.calibration_loader.batch_size}")
        self._print(f"  - Backend: {torch.backends.quantized.engine}")
        self._print(f"  - Note: Model runs on CPU only")
        
        return model
    
    def quantize_fp16(self) -> nn.Module:
        """
        Convert model to half precision (FP16).
        
        - All parameters and activations → float16
        - Works on both CPU and GPU
        - ~50% memory reduction
        - Faster inference on modern GPUs
        - May have numerical instability issues
        
        Returns:
          if self.method == 'fp16':
            return self.quantize_fp16()
        elif self.method == 'bf16':
            return self.quantize_bfloat16()
        el  FP16 model
        """
        self._print("\n" + "="*60)
        self._print("CONVERTING TO HALF PRECISION (FP16)")
        self._print("="*60)
        
        # Convert model to half precision
        fp16_model = self.model.half()
        
        quantized_params = self._count_parameters(fp16_model)
        
        self._print(f"\n✓ FP16 conversion completed")
        self._print(f"  - Original parameters: {self.original_params:,}")
        self._print(f"  - FP16 parameters: {quantized_params:,}")
        self._print(f"  - Memory reduction: ~50%")
        self._print(f"  - Works on: CPU and GPU")
        self._print(f"  - Note: May have numerical stability issues")
        
        return fp16_model
    
    def quantize_bfloat16(self) -> nn.Module:
        """
        Convert model to BFloat16 precision.
        
        - All parameters and activations → bfloat16
        - Works on both CPU and GPU (GPU preferred)
        - ~50% memory reduction
        - Better numerical stability than FP16
        - Requires newer hardware (Ampere+ GPUs, recent CPUs)
        
        Returns:
            BFloat16 model
        """
        self._print("\n" + "="*60)
        self._print("CONVERTING TO BFLOAT16 PRECISION")
        self._print("="*60)
        
        # Convert model to bfloat16
        bf16_model = self.model.to(torch.bfloat16)
        
        quantized_params = self._count_parameters(bf16_model)
        
        self._print(f"\n✓ BFloat16 conversion completed")
        self._print(f"  - Original parameters: {self.original_params:,}")
        self._print(f"  - BFloat16 parameters: {quantized_params:,}")
        self._print(f"  - Memory reduction: ~50%")
        self._print(f"  - Works on: CPU and GPU (GPU preferred)")
        self._print(f"  - Note: Better stability than FP16, requires newer hardware")
        
        return bf16_model
    
    def quantize(self) -> nn.Module:
        """
        Apply quantization based on specified method.
        
        Returns:
            Quantized model
        """
        if self.method == 'dynamic':
            return self.quantize_dynamic()
        elif self.method == 'static':
            return self.quantize_static()
        elif self.method == 'fp16':
            return self.quantize_fp16()
        elif self.method == 'bf16':
            return self.quantize_bfloat16()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_quantization_info(self, quantized_model: nn.Module) -> Dict[str, Any]:
        """
        Get information about quantized model.
        
        Args:
            quantized_model: Quantized model
            
        Returns:
            Dictionary with quantization information
        """
        info = {
            'method': self.method,
            'dtype': str(self.dtype),
            'original_parameters': self.original_params,
            'quantized_parameters': self._count_parameters(quantized_model),
            'qconfig': self.qconfig if self.method == 'static' else None,
            'calibration_batches': self.calibration_batches if self.method == 'static' else None
        }
        
        return info


def quantize_model(
    model: nn.Module,
    method: str = 'dynamic',
    dtype: str = 'int8',
    calibration_loader: Optional[DataLoader] = None,
    **kwargs
) -> nn.Module:
    """
    Quick helper function to quantize a model.
    
    Args:
        model: PyTorch model to quantize
        method: Quantization method
            - 'dynamic': Dynamic int8 quantization
            - 'static': Static int8 quantization
            - 'fp16': Half precision (float16)
            - 'bf16': BFloat16 precision
        dtype: Data type for int8 quantization ('int8' or 'uint8')
        calibration_loader: DataLoader for calibration (required for static)
        **kwargs: Additional arguments for ModelQuantizer
        
    Returns:
        Quantized model
        
    Example:
        >>> # INT8 dynamic quantization
        >>> quantized_model = quantize_model(model, method='dynamic', dtype='int8')
        
        >>> # FP16 half precision
        >>> fp16_model = quantize_model(model, method='fp16')
        
        >>> # INT8 static quantization
        >>> quantized_model = quantize_model(
        ...     model, 
        ...     method='static',
        ...     dtype='int8',
        ...     calibration_loader=calibration_loader
        ... )
    """
    quantizer = ModelQuantizer(
        model=model,
        method=method,
        dtype=dtype,
        calibration_loader=calibration_loader,
        **kwargs
    )
    
    return quantizer.quantize()
