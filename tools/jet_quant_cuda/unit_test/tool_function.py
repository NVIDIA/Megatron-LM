# Copyright 2023-2024 Bytedance Ltd. and/or its affiliates 


# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

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