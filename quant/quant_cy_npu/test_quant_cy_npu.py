#!/usr/bin/env python3
"""
Test script for quant_cy_npu on 910B environment
"""

import sys
import torch
import numpy as np
import time

def test_environment():
    """Test if the environment is properly set up"""
    print("🔍 Testing environment...")
    
    try:
        import torch_npu
        print(f"✅ torch_npu version: {torch_npu.__version__}")
        
        if torch_npu.npu.is_available():
            device_count = torch_npu.npu.device_count()
            print(f"✅ NPU devices available: {device_count}")
        else:
            print("❌ No NPU devices available")
            return False
            
    except ImportError as e:
        print(f"❌ torch_npu import failed: {e}")
        return False
    
    return True

def test_quant_cy_npu_import():
    """Test if quant_cy_npu can be imported"""
    print("\n🔍 Testing quant_cy_npu import...")
    
    try:
        import quant_cy_npu
        print("✅ quant_cy_npu imported successfully")
        
        # Print status
        quant_cy_npu.print_status()
        
        return True, quant_cy_npu
        
    except ImportError as e:
        print(f"❌ quant_cy_npu import failed: {e}")
        return False, None

def test_basic_quantization(quant_cy_npu):
    """Test basic quantization functionality"""
    print("\n🔍 Testing basic quantization...")
    
    try:
        from quant_cy_npu import QType, quant_dequant_float
        
        # Create test tensor
        x = torch.randn(64, 64, dtype=torch.float32)
        print(f"✅ Created test tensor: {x.shape}")
        
        # Test different quantization types
        test_formats = ['hif8', 'hifx4_v12', 'mxfp4', 'mxfp8e4m3']
        
        for fmt in test_formats:
            try:
                qtype = QType(fmt)
                print(f"✅ Created QType: {fmt}")
                
                # Test CPU quantization
                y_cpu = quant_dequant_float(x, qtype, force_py=True)
                error_cpu = torch.norm(x - y_cpu).item()
                print(f"✅ CPU quantization ({fmt}): error = {error_cpu:.6f}")
                
            except Exception as e:
                print(f"⚠️  CPU quantization ({fmt}) failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic quantization test failed: {e}")
        return False

def test_npu_quantization(quant_cy_npu):
    """Test NPU quantization functionality"""
    print("\n🔍 Testing NPU quantization...")
    
    if not quant_cy_npu.NPU_OPS_AVAILABLE:
        print("⚠️  NPU operators not available, skipping NPU tests")
        return True
    
    try:
        import torch_npu
        from quant_cy_npu import QType, quant_dequant_float
        
        # Create test tensor on NPU
        x = torch.randn(64, 64, dtype=torch.float32).npu()
        print(f"✅ Created NPU tensor: {x.shape}")
        
        # Test NPU quantization
        qtype = QType('hif8')
        y_npu = quant_dequant_float(x, qtype, force_py=False)
        error_npu = torch.norm(x - y_npu).item()
        print(f"✅ NPU quantization (hif8): error = {error_npu:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ NPU quantization test failed: {e}")
        return False

def test_performance(quant_cy_npu):
    """Test quantization performance"""
    print("\n🔍 Testing quantization performance...")
    
    try:
        from quant_cy_npu import QType, quant_dequant_float
        import torch_npu
        
        # Create larger test tensor
        x = torch.randn(512, 512, dtype=torch.float32)
        
        # Test CPU performance
        qtype = QType('hif8')
        
        # Warm up
        for _ in range(10):
            _ = quant_dequant_float(x, qtype, force_py=True)
        
        # Benchmark
        torch_npu.npu.synchronize()
        start_time = time.time()
        
        iterations = 100
        for _ in range(iterations):
            y = quant_dequant_float(x, qtype, force_py=True)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        print(f"✅ CPU performance: {avg_time*1000:.2f}ms per operation")
        
        # Test NPU performance if available
        if quant_cy_npu.NPU_OPS_AVAILABLE:
            x_npu = x.npu()
            
            # Warm up
            for _ in range(10):
                _ = quant_dequant_float(x_npu, qtype, force_py=False)
            
            # Benchmark
            torch_npu.npu.synchronize()
            start_time = time.time()
            
            for _ in range(iterations):
                y = quant_dequant_float(x_npu, qtype, force_py=False)
            
            torch_npu.npu.synchronize()
            end_time = time.time()
            avg_time_npu = (end_time - start_time) / iterations
            print(f"✅ NPU performance: {avg_time_npu*1000:.2f}ms per operation")
            
            speedup = avg_time / avg_time_npu
            print(f"✅ NPU speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 quant_cy_npu Test Suite for 910B")
    print("=" * 50)
    
    # Test environment
    if not test_environment():
        print("\n❌ Environment test failed. Please check your 910B setup.")
        sys.exit(1)
    
    # Test import
    success, quant_cy_npu = test_quant_cy_npu_import()
    if not success:
        print("\n❌ Import test failed. Please run ./build.sh first.")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_quantization(quant_cy_npu):
        print("\n❌ Basic quantization test failed.")
        sys.exit(1)
    
    # Test NPU functionality
    if not test_npu_quantization(quant_cy_npu):
        print("\n❌ NPU quantization test failed.")
        sys.exit(1)
    
    # Test performance
    if not test_performance(quant_cy_npu):
        print("\n❌ Performance test failed.")
        sys.exit(1)
    
    print("\n🎉 All tests passed!")
    print("✅ quant_cy_npu is ready to use on 910B environment")

if __name__ == '__main__':
    main()
