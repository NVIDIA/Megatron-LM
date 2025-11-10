#!/usr/bin/env python3
"""
Quick test script to verify quant_cy_npu structure without compilation
"""

import sys
import os

def test_imports():
    """Test if we can import the basic modules"""
    print("🔍 Testing basic imports...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Test QType import
        from quant_cy_npu.base.QType import QType
        print("✅ QType imported successfully")
        
        # Test QType creation
        qtype = QType('hif8')
        print(f"✅ QType created: {qtype}")
        
        # Test other formats
        formats = ['hifx4_v12', 'mxfp4', 'mxfp8e4m3', 'mxfp8e5m2', 'nvf4']
        for fmt in formats:
            try:
                qtype = QType(fmt)
                print(f"✅ QType({fmt}): {qtype}")
            except Exception as e:
                print(f"⚠️  QType({fmt}) failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_quantization_basic():
    """Test basic quantization without NPU ops"""
    print("\n🔍 Testing basic quantization...")
    
    try:
        import torch
        import numpy as np
        from quant_cy_npu.base.QType import QType
        from quant_cy_npu.base.QFunc.quant_basic import quant_dequant_float
        
        # Create test tensor
        x = torch.randn(32, 32, dtype=torch.float32)
        print(f"✅ Created test tensor: {x.shape}")
        
        # Test HiF8 quantization
        qtype = QType('hif8')
        y = quant_dequant_float(x, qtype, force_py=True)
        
        error = torch.norm(x - y).item()
        print(f"✅ HiF8 quantization error: {error:.6f}")
        
        # Test other formats
        test_formats = ['hifx4_v12', 'mxfp4', 'mxfp8e4m3']
        for fmt in test_formats:
            try:
                qtype = QType(fmt)
                y = quant_dequant_float(x, qtype, force_py=True)
                error = torch.norm(x - y).item()
                print(f"✅ {fmt} quantization error: {error:.6f}")
            except Exception as e:
                print(f"⚠️  {fmt} quantization failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic quantization test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        'setup.py',
        'build.sh',
        'README.md',
        'quant_cy_npu/__init__.py',
        'quant_cy_npu/base/__init__.py',
        'quant_cy_npu/base/QType.py',
        'quant_cy_npu/base/QTensor.py',
        'quant_cy_npu/base/QFunc/__init__.py',
        'quant_cy_npu/base/QFunc/quant_basic.py',
        'quant_cy_npu/base/cusrc/npu_quant.cpp',
        'quant_cy_npu/base/cusrc/hif8_quant_op.h',
        'quant_cy_npu/base/cusrc/tensorutils.h',
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🧪 quant_cy_npu Quick Test")
    print("=" * 40)
    
    # Test file structure
    if not test_file_structure():
        print("\n❌ File structure test failed.")
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed.")
        return False
    
    # Test basic quantization
    if not test_quantization_basic():
        print("\n❌ Basic quantization test failed.")
        return False
    
    print("\n🎉 All quick tests passed!")
    print("✅ quant_cy_npu structure is correct")
    print("📝 Next step: Run ./build.sh to compile NPU operators")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
