#!/bin/bash

# Build script for quant_cy_npu on 910B environment
# This script compiles the NPU quantization operators

set -e

echo "🔧 Building quant_cy_npu for 910B environment..."

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Error: setup.py not found. Please run this script from the quant_cy_npu directory."
    exit 1
fi

# Check environment
echo "🔍 Checking environment..."

# Check if torch_npu is available
python3 -c "import torch_npu; print('✅ torch_npu available')" || {
    echo "❌ torch_npu not available. Please install torch_npu first."
    exit 1
}

# Check if NPU devices are available
python3 -c "import torch_npu; print(f'✅ NPU devices: {torch_npu.npu.device_count()}')" || {
    echo "❌ No NPU devices available."
    exit 1
}

# Check if Ascend toolkit is installed
if [ ! -d "/usr/local/Ascend/ascend-toolkit" ]; then
    echo "❌ Ascend toolkit not found at /usr/local/Ascend/ascend-toolkit"
    echo "   Please install Ascend-CANN-toolkit for 910B"
    exit 1
fi

echo "✅ Environment check passed"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/
find . -name "*.so" -delete
find . -name "*.pyd" -delete

# Set environment variables for 910B
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
export PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH

# Build the extension
echo "🔨 Building NPU quantization operators..."
python3 setup.py build_ext --inplace

# Check if build was successful
if [ -f "quant_cy_npu_npu_ops*.so" ] || [ -f "quant_cy_npu_npu_ops*.pyd" ]; then
    echo "✅ Build successful!"
    
    # Test the installation
    echo "🧪 Testing installation..."
    python3 -c "
import sys
sys.path.insert(0, '.')
try:
    import quant_cy_npu
    quant_cy_npu.print_status()
    print('✅ Installation test passed')
except Exception as e:
    print(f'❌ Installation test failed: {e}')
    sys.exit(1)
"
    
    echo ""
    echo "🎉 quant_cy_npu successfully built and installed!"
    echo "   You can now use NPU-accelerated quantization operators."
    echo ""
    echo "📖 Usage example:"
    echo "   import quant_cy_npu"
    echo "   from quant_cy_npu import QType, quant_dequant_float"
    echo "   import torch"
    echo "   import torch_npu"
    echo ""
    echo "   # Create test tensor on NPU"
    echo "   x = torch.randn(1024, 1024).npu()"
    echo "   qtype = QType('hif8')"
    echo "   y = quant_dequant_float(x, qtype)"
    echo ""
    
else
    echo "❌ Build failed. Check the error messages above."
    exit 1
fi
