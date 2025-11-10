"""
quant_cy_npu - NPU Quantization Operators for 910B

This package provides high-performance quantization operators optimized for 
Ascend 910B NPU, including HiF8, HiF4, MXFP4, MXFP8, and NVF4 quantization.
"""

__version__ = "1.0.0"
__author__ = "Megatron-LM Team"

# Import core modules
from .base.QType import QType
from .base.QTensor import QTensor
from .base.QFunc.quant_basic import quant_dequant_float

# Import quantization functions
try:
    from .base.QFunc.hif8 import hif8_quantize, hif8_dequantize
    from .base.QFunc.hifx import hifx_quantize, hifx_dequantize
except ImportError:
    # Fallback to basic implementations if NPU ops not available
    hif8_quantize = None
    hif8_dequantize = None
    hifx_quantize = None
    hifx_dequantize = None

# Import NPU operators if available
try:
    import quant_cy_npu_npu_ops
    NPU_OPS_AVAILABLE = True
except ImportError:
    NPU_OPS_AVAILABLE = False
    quant_cy_npu_npu_ops = None

__all__ = [
    'QType',
    'QTensor', 
    'quant_dequant_float',
    'hif8_quantize',
    'hif8_dequantize',
    'hifx_quantize',
    'hifx_dequantize',
    'NPU_OPS_AVAILABLE',
]

def get_npu_ops_status():
    """Get the status of NPU operators availability"""
    return {
        'available': NPU_OPS_AVAILABLE,
        'version': __version__,
        'supported_formats': [
            'hif8', 'hifx4_v12', 'hifx3_v12', 'hifx2_v12',
            'mxfp4', 'mxfp8e4m3', 'mxfp8e5m2',
            'nvf4'
        ]
    }

def print_status():
    """Print the current status of the quantization package"""
    status = get_npu_ops_status()
    print(f"quant_cy_npu v{status['version']}")
    print(f"NPU Operators Available: {'✅' if status['available'] else '❌'}")
    print(f"Supported Formats: {', '.join(status['supported_formats'])}")
    
    if not status['available']:
        print("\n⚠️  NPU operators not available. Using Python fallback implementations.")
        print("   To enable NPU acceleration, run: python setup.py build_ext --inplace")
