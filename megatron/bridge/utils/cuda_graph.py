# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/utils/cuda_graph.py
# Source license: Apache-2.0.

from typing import Any

from megatron.core.transformer.enums import CudaGraphScope

try:
    from megatron.core.transformer.enums import CudaGraphModule
except ImportError:
    CudaGraphModule = None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value or value == "full":
            return []
        return value.split(",")
    if isinstance(value, list):
        return value
    return [value]


def _member_name(value: Any) -> str:
    if isinstance(value, str):
        return value.rsplit(".", 1)[-1]
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", 1)[-1]


def _member_name_list(value: Any) -> list[str]:
    return [_member_name(item) for item in _as_list(value)]


def _supports_cuda_graph_modules(config: Any) -> bool:
    return CudaGraphModule is not None and hasattr(config, "cuda_graph_modules")


def _module_value(name: str):
    if CudaGraphModule is not None:
        return CudaGraphModule[name]
    return CudaGraphScope[name]


def set_cuda_graph_modules(config: Any, modules: Any) -> None:
    """Set per-layer CUDA graph modules using the current MCore API when available."""

    module_names = _member_name_list(modules)
    if _supports_cuda_graph_modules(config):
        config.cuda_graph_modules = [_module_value(name) for name in module_names]
        if hasattr(config, "cuda_graph_scope"):
            config.cuda_graph_scope = None
    else:
        config.cuda_graph_scope = [CudaGraphScope[name] for name in module_names]


def clear_cuda_graph_modules(config: Any) -> None:
    """Clear per-layer CUDA graph modules using the active MCore API."""

    set_cuda_graph_modules(config, [])
