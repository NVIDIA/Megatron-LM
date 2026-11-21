# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record effective CUDA driver and numeric library identities inside an arm."""

import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
import transformer_engine

driver = ctypes.CDLL('libcuda.so.1')
driver_version = ctypes.c_int()
driver.cuDriverGetVersion(ctypes.byref(driver_version))
init_status = driver.cuInit(0)
result = {
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'transformer_engine': transformer_engine.__version__,
    'driver_api_version': driver_version.value,
    'cuInit_status': init_status,
    'nccl': torch.cuda.nccl.version(),
    'libraries': {},
    'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
    'float32_matmul_precision': torch.get_float32_matmul_precision(),
    'te_module_path': transformer_engine.__file__,
    'gpus': [
        {
            'name': torch.cuda.get_device_name(index),
            'sm_count': torch.cuda.get_device_properties(index).multi_processor_count,
            'capability': list(torch.cuda.get_device_capability(index)),
        }
        for index in range(torch.cuda.device_count())
    ],
    'environment': {
        key: os.environ.get(key)
        for key in [
            'LD_LIBRARY_PATH',
            'CUDA_DEVICE_MAX_CONNECTIONS',
            'CUBLAS_WORKSPACE_CONFIG',
            'NCCL_ALGO',
            'NCCL_PROTO',
            'NCCL_NVLS_ENABLE',
            'TORCH_DIST_USE_NCCL2',
            'NVIDIA_DISABLE_REQUIRE',
            'NVIDIA_REQUIRE_CUDA',
            'CUDA_VISIBLE_DEVICES',
            'NVTE_BUILD_NUM_PHILOX_ROUNDS',
            'PYTHONHASHSEED',
            'NVIDIA_TF32_OVERRIDE',
            'NCCL_TOPO_FILE',
            'NCCL_GRAPH_FILE',
            'NCCL_MIN_NCHANNELS',
            'NCCL_MAX_NCHANNELS',
            'NCCL_P2P_DISABLE',
            'NCCL_P2P_CUMEM_ENABLE',
            'NCCL_NET',
            'NCCL_TUNER_PLUGIN',
        ]
    },
}
for name, function in [
    ('libcublasLt.so.13', 'cublasLtGetVersion'),
    ('libnccl.so.2', 'ncclGetVersion'),
]:
    lib = ctypes.CDLL(name)
    if function == 'cublasLtGetVersion':
        lib.cublasLtGetVersion.restype = ctypes.c_size_t
        result['libraries'][name] = lib.cublasLtGetVersion()
    else:
        value = ctypes.c_int()
        result['libraries'][name] = [lib.ncclGetVersion(ctypes.byref(value)), value.value]
result['loaded_libraries'] = sorted(
    {
        line.split()[-1]
        for line in Path('/proc/self/maps').read_text().splitlines()
        if any(
            token in line
            for token in [
                'libcuda.so',
                'libnvidia-ptxjitcompiler',
                'libcublas',
                'libnccl',
                'libcudnn',
            ]
        )
    }
)
result['library_sha256'] = {}
for name in result['loaded_libraries']:
    if any(
        token in name for token in ['libcuda.so', 'libnccl.so', 'libcublas.so', 'libcublasLt.so']
    ):
        with open(name, 'rb') as source:
            result['library_sha256'][name] = hashlib.file_digest(source, 'sha256').hexdigest()
print(json.dumps(result, indent=2))
