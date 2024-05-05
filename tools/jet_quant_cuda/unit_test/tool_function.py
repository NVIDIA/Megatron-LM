import torch
from importlib import import_module
import os
import sys
import subprocess

def analysis_diff(origin_tensor, quantized_tensor):
    diff = origin_tensor - quantized_tensor
    abs_error_norm = torch.norm(diff)
    origin_norm = torch.norm(origin_tensor)
    rela_error_norm = abs_error_norm / origin_norm
    return abs_error_norm, rela_error_norm

def build_and_import_module(pkg_path, module_name):
    setup_path = os.path.join(pkg_path, 'setup.py')
    build_base_path = os.path.join(pkg_path, 'build')
    build_cmd = [sys.executable, setup_path, 'build', '--build-base', build_base_path]
    subprocess.check_call(build_cmd)

    build_lib_path = find_build_lib(os.path.join(pkg_path, 'build'))
    if build_lib_path:
        sys.path.append(build_lib_path)
    else:
        raise ImportError("Could not find built library path")
    print('build_lib_path', build_lib_path)
    
    module = import_module(module_name)
    
    return module

def find_build_lib(base_path):
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if d.startswith("lib."):
                return os.path.join(root, d)
    return None