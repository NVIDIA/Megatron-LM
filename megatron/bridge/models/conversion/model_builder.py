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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/model_provider.py
# Source license: Apache-2.0.

"""Builder integration that preserves model-specific FP32 parameters."""

from functools import partial
from typing import Any, Callable

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.training.models.base import ModelConfig


def _apply_mixed_precision_wrapper(
    model: list[MegatronModule],
    model_config: Any,
    mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule],
) -> list[MegatronModule]:
    """Wrap a model while preserving parameters explicitly marked for FP32."""
    keep_in_fp32: list[tuple[torch.nn.Module, str, torch.Tensor]] = []
    for model_module in model:
        for submodule in model_module.modules():
            # Preserve the existing MCore expert-bias contract.
            if hasattr(submodule, "_maintain_float32_expert_bias"):
                expert_bias = getattr(submodule, "expert_bias", None)
                if expert_bias is not None:
                    keep_in_fp32.append((submodule, "expert_bias", expert_bias.data.clone()))

            # Model-specific modules can mark direct parameters that must not be
            # truncated by Float16Module's recursive half()/bfloat16() cast.
            parameter_names = vars(submodule).get("_keep_in_float32_parameter_names", ())
            if not isinstance(parameter_names, (list, tuple)):
                raise TypeError("_keep_in_float32_parameter_names must be a list or tuple")
            for parameter_name in parameter_names:
                parameter = getattr(submodule, parameter_name, None)
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError(
                        f"{type(submodule).__name__}.{parameter_name} must be a Parameter to remain in FP32"
                    )
                keep_in_fp32.append((submodule, parameter_name, parameter.data.clone()))

    wrapped_model = [mixed_precision_wrapper(model_config, model_module) for model_module in model]

    for submodule, parameter_name, fp32_data in keep_in_fp32:
        getattr(submodule, parameter_name).data = fp32_data

    return wrapped_model


def _wrap_keeping_float32_tensors(
    mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule],
    config: Any,
    module: MegatronModule,
) -> MegatronModule:
    """Wrap one model stage through ``_apply_mixed_precision_wrapper``."""
    return _apply_mixed_precision_wrapper([module], config, mixed_precision_wrapper)[0]


def build_distributed_models(
    model_config: ModelConfig, pg_collection: ProcessGroupCollection, **kwargs: Any
) -> list[MegatronModule]:
    """Build a builder-backed config through its ``ModelBuilder``, keeping FP32-marked tensors exact.

    Megatron Core's builders apply ``mixed_precision_wrapper`` directly, so ``Float16Module`` rounds
    router ``expert_bias`` values loaded before the wrap (Hugging Face weights) and casts
    parameters listed in ``_keep_in_float32_parameter_names`` to the model dtype. The wrapper is
    applied through ``_apply_mixed_precision_wrapper`` instead, which restores both after the cast.

    Args:
        model_config: Builder-backed model config.
        pg_collection: Model communication process groups.
        **kwargs: Arguments for ``ModelBuilder.build_distributed_models``. Its
            ``mixed_precision_wrapper`` is ``Float16Module`` when omitted; ``None`` skips wrapping.

    Returns:
        Model stages created by the configured builder.
    """
    mixed_precision_wrapper = kwargs.get("mixed_precision_wrapper", Float16Module)
    if mixed_precision_wrapper is not None:
        kwargs["mixed_precision_wrapper"] = partial(
            _wrap_keeping_float32_tensors, mixed_precision_wrapper
        )
    builder = model_config.get_builder_cls()(model_config)
    return builder.build_distributed_models(pg_collection=pg_collection, **kwargs)
