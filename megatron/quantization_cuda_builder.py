import subprocess
import sys
import os
from importlib import import_module
import torch

def build_module(pkg_path):
    setup_path = os.path.join(pkg_path, 'setup.py')
    build_base_path = os.path.join(pkg_path, 'build')
    build_cmd = [sys.executable, setup_path, 'build', '--build-base', build_base_path]
    subprocess.check_call(build_cmd)

def find_module(pkg_path, module_name):
    build_lib_path = find_build_lib(os.path.join(pkg_path, 'build'))
    if build_lib_path:
        sys.path.append(build_lib_path)
    else:
        return None
    
    module = import_module(module_name)
    
    return module

def find_build_lib(base_path):
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if d.startswith("lib."):
                return os.path.join(root, d)
    return None

if __name__ == '__main__':

    pkg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../','tools/jet_quant_cuda')
    # print('pkg path:', pkg_path)

    build_module(pkg_path)

    quantization_cuda = find_module(pkg_path, 'quantization_cuda')