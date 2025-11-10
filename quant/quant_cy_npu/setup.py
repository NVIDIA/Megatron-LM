#!/usr/bin/env python3
"""
Setup script for quant_cy_npu - NPU quantization operators for 910B
"""

import os
import sys
import torch
import torch_npu
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup, find_packages

# Check if we're on 910B environment
def check_910b_environment():
    """Check if we're running on 910B NPU environment"""
    try:
        import torch_npu
        if hasattr(torch_npu, 'npu'):
            print("✅ torch_npu detected")
        else:
            print("❌ torch_npu not properly installed")
            return False
    except ImportError:
        print("❌ torch_npu not found")
        return False
    
    # Check NPU device availability
    try:
        if torch_npu.npu.is_available():
            device_count = torch_npu.npu.device_count()
            print(f"✅ NPU devices available: {device_count}")
            return True
        else:
            print("❌ No NPU devices available")
            return False
    except Exception as e:
        print(f"❌ Error checking NPU: {e}")
        return False

# Get NPU compilation flags
def get_npu_compile_args():
    """Get NPU-specific compilation arguments for 910B"""
    args = [
        '-std=c++17',
        '-O2',
        '-fPIC',
        '-DASCEND_910B',  # 910B specific flag
        '-D__DAV_C220_VEC__',  # Vector instruction support
        '-DASCEND_IS_AIV',  # AI Vector support
    ]
    
    # Add NPU include paths
    npu_include_paths = [
        '/usr/local/Ascend/ascend-toolkit/latest/include',
        '/usr/local/Ascend/ascend-toolkit/latest/include/aclnn',
        '/usr/local/Ascend/ascend-toolkit/latest/include/aclnn_base',
        '/usr/local/Ascend/ascend-toolkit/latest/include/aclnn_common',
        '/usr/local/Ascend/ascend-toolkit/latest/include/aclnn_math',
        '/usr/local/Ascend/ascend-toolkit/latest/include/aclnn_vector',
    ]
    
    for path in npu_include_paths:
        if os.path.exists(path):
            args.append(f'-I{path}')
    
    return args

def get_npu_link_args():
    """Get NPU-specific linking arguments for 910B"""
    args = [
        '-L/usr/local/Ascend/ascend-toolkit/latest/lib64',
        '-lascendcl',
        '-laclnn',
        '-laclnn_base',
        '-laclnn_common', 
        '-laclnn_math',
        '-laclnn_vector',
    ]
    return args

def main():
    print("🔧 Setting up quant_cy_npu for 910B environment...")
    
    # Check environment
    if not check_910b_environment():
        print("❌ Environment check failed. Please ensure you're running on 910B with proper torch_npu installation.")
        sys.exit(1)
    
    # Define the extension
    ext_modules = []
    
    # NPU Quantization Extension
    npu_quant_sources = [
        'quant_cy_npu/base/cusrc/npu_quant.cpp',
        'quant_cy_npu/base/cusrc/npu_quantop_base.cpp',
    ]
    
    # Check if source files exist
    missing_files = []
    for source in npu_quant_sources:
        if not os.path.exists(source):
            missing_files.append(source)
    
    if missing_files:
        print(f"❌ Missing source files: {missing_files}")
        sys.exit(1)
    
    # Create NPU extension
    npu_extension = CUDAExtension(
        name='quant_cy_npu_npu_ops',
        sources=npu_quant_sources,
        include_dirs=[
            'quant_cy_npu/base/cusrc',
            '/usr/local/Ascend/ascend-toolkit/latest/include',
        ],
        extra_compile_args=get_npu_compile_args(),
        extra_link_args=get_npu_link_args(),
        define_macros=[
            ('ASCEND_910B', '1'),
            ('__DAV_C220_VEC__', '1'),
            ('ASCEND_IS_AIV', '1'),
        ]
    )
    
    ext_modules.append(npu_extension)
    
    # Setup configuration
    setup(
        name='quant_cy_npu',
        version='1.0.0',
        description='NPU Quantization Operators for 910B',
        author='Megatron-LM Team',
        packages=find_packages(),
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
        python_requires='>=3.7',
        install_requires=[
            'torch>=1.8.0',
            'torch_npu>=1.0.0',
            'numpy>=1.19.0',
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ],
    )

if __name__ == '__main__':
    main()
