#!/usr/bin/env python3
"""
Simple test to isolate import issues
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing individual imports...")

try:
    print("1. Testing QType import...")
    from quant_cy_npu.base.QType import QType
    print("✅ QType imported")
    
    print("2. Testing QType creation...")
    qtype = QType('hif8')
    print(f"✅ QType created: {qtype}")
    
    print("3. Testing quant_basic import...")
    from quant_cy_npu.base.QFunc.quant_basic import quant_py
    print("✅ quant_py imported")
    
    print("4. Testing quant_dequant_float import...")
    from quant_cy_npu.base.QFunc.quant_basic import quant_dequant_float
    print("✅ quant_dequant_float imported")
    
    print("5. Testing main module import...")
    import quant_cy_npu
    print("✅ quant_cy_npu imported")
    
    print("\n🎉 All imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
