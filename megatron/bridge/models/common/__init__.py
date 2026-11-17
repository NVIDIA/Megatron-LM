# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/common/__init__.py
# Source license: Apache-2.0.

from megatron.bridge.models.common.base import (
    BuildConfigT,
    MLAModelConfigMixin,
    ModelBuilder,
    ModelConfig,
    ModelConfigOverrideMixin,
    ModelT,
    Serializable,
    compose_hooks,
    deserialize_model_config,
)
from megatron.bridge.models.common.unimodal import (
    build_virtual_pipeline_stages,
    unimodal_build_distributed_models,
)

__all__ = [
    "BuildConfigT",
    "MLAModelConfigMixin",
    "ModelBuilder",
    "ModelConfig",
    "ModelConfigOverrideMixin",
    "ModelT",
    "Serializable",
    "compose_hooks",
    "deserialize_model_config",
    "build_virtual_pipeline_stages",
    "unimodal_build_distributed_models",
]
