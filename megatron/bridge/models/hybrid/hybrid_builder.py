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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/hybrid/hybrid_builder.py
# Source license: Apache-2.0.

import copy
import inspect
from dataclasses import dataclass
from typing import ClassVar

import torch

from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_inference_stack_spec as default_hybrid_inference_stack_spec,
)
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec as default_hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.post_training.modelopt.hybrid.model_specs import get_hybrid_stack_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnBackend
from megatron.training.models.hybrid import HybridModelBuilder as MCoreHybridModelBuilder
from megatron.training.models.hybrid import HybridModelConfig as MCoreHybridModelConfig
from megatron.training.models.hybrid import resolve_hybrid_stack_spec

DEFAULT_MAMBA_CHUNK_SIZE = 128


@dataclass(kw_only=True)
class HybridModelConfig(MCoreHybridModelConfig):
    """Bridge Hybrid config with transitional output-logit dtype support."""

    builder: ClassVar[str] = "megatron.bridge.models.hybrid.HybridModelBuilder"
    logit_dtype: torch.dtype | None = None


class HybridModelBuilder(MCoreHybridModelBuilder):
    """Bridge Hybrid builder that prevents silent fallback on older MCore."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> HybridModel:
        """Build a Hybrid stage after validating MCore logit-dtype support."""
        logit_dtype_kwarg(HybridModel, self._model_config.logit_dtype)
        return super().build_model(pg_collection, pre_process, post_process, vp_stage)


def transformer_engine_hybrid_stack_spec() -> ModuleSpec:
    """Return the default Hybrid stack spec with Transformer Engine layers.

    This is a named function (not a lambda) to allow proper serialization
    and reconstruction from checkpoints. Named functions can be imported
    via their module path, unlike lambdas.

    Returns:
        Default Hybrid stack specification from megatron.core.
    """
    return default_hybrid_stack_spec


def modelopt_hybrid_stack_spec(config: "HybridModelConfig | None" = None) -> ModuleSpec:
    """Hybrid stack specification for quantization with ModelOpt.

    Uses Norm instead of TENorm and ColumnParallelLinear/RowParallelLinear
    instead of TE layers to enable proper quantizer insertion by ModelOpt.

    Args:
        config: Optional Hybrid configuration object.

    Returns:
        Module specification for quantization-ready Hybrid stack.
    """
    return get_hybrid_stack_modelopt_spec(local_core_attention=False, remap_te_layernorm=True)


def get_default_hybrid_stack_spec(config: "HybridModelConfig") -> ModuleSpec:
    """Determine the most appropriate Hybrid stack specification based on configuration.

    Args:
        config: Hybrid configuration object.

    Returns:
        Appropriate module specification based on config.
    """
    transformer = getattr(config, "transformer", config)
    if transformer.transformer_impl == "inference_optimized":
        return default_hybrid_inference_stack_spec
    if config.restore_modelopt_state:
        return modelopt_hybrid_stack_spec(config)
    return transformer_engine_hybrid_stack_spec()


def configure_mamba_chunk_size(stack_spec: ModuleSpec, chunk_size: int) -> ModuleSpec:
    """Return a stack spec whose Mamba mixer uses the requested scan chunk size.

    Args:
        stack_spec: Hybrid stack specification.
        chunk_size: Chunk size of the Mamba scan.

    Returns:
        ``stack_spec`` for the default chunk size, otherwise a configured copy.
    """
    if chunk_size == DEFAULT_MAMBA_CHUNK_SIZE:
        return stack_spec

    configured_spec = copy.deepcopy(stack_spec)
    mixer_spec = configured_spec.submodules.mamba_layer.submodules.mixer
    mixer_spec.params = {**mixer_spec.params, "chunk_size": chunk_size}
    return configured_spec


def use_local_core_attention(stack_spec: ModuleSpec) -> ModuleSpec:
    """Return a copy of a stack spec whose attention layers use Megatron Core's local attention.

    Args:
        stack_spec: Hybrid stack specification.

    Returns:
        A copy of ``stack_spec`` with ``DotProductAttention`` as its core attention.
    """
    from megatron.core.transformer.dot_product_attention import DotProductAttention

    local_spec = copy.deepcopy(stack_spec)
    attention_layer = getattr(local_spec.submodules, "attention_layer", None)
    if attention_layer is not None:
        attention_layer.submodules.self_attention.submodules.core_attention = DotProductAttention
    return local_spec


class BridgeHybridModelBuilder(HybridModelBuilder):
    """Hybrid builder that resolves the stack spec the way ``HybridModelProvider`` does.

    The configured stack spec may be a ``ModuleSpec`` or a factory that takes no
    arguments or the model config. When it is unset, Bridge's default selection
    applies, which also remaps fused Transformer Engine layer norms for ModelOpt
    checkpoints. The local attention backend and a non-default Mamba chunk size
    are applied to a copy of the resolved spec.
    """

    def _resolve_hybrid_stack_spec(self) -> ModuleSpec:
        """Resolve the stack spec for the configured model.

        Returns:
            Stack specification used to build every stage.
        """
        model_config = self._model_config
        transformer = model_config.transformer
        stack_spec = model_config.hybrid_stack_spec
        if stack_spec is None:
            # Bridge's default selection has no wide-residual stacks; Megatron Core's does.
            if transformer.wide_residual is not None:
                resolved_spec = resolve_hybrid_stack_spec(model_config)
            else:
                resolved_spec = get_default_hybrid_stack_spec(model_config)
        elif isinstance(stack_spec, ModuleSpec):
            resolved_spec = stack_spec
        elif inspect.signature(stack_spec).parameters:
            resolved_spec = stack_spec(model_config)
        else:
            resolved_spec = stack_spec()

        if transformer.attention_backend in {AttnBackend.local, "local"}:
            resolved_spec = use_local_core_attention(resolved_spec)
        return configure_mamba_chunk_size(resolved_spec, model_config.mamba_chunk_size)

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> HybridModel:
        """Build a Hybrid stage from the resolved stack spec.

        Args:
            pg_collection: Process groups used to construct the model.
            pre_process: Whether to include input processing on this stage.
            post_process: Whether to include output processing on this stage.
            vp_stage: Virtual pipeline stage to construct.

        Returns:
            The constructed Hybrid model stage.
        """
        model_config = self._model_config
        configured_spec = model_config.hybrid_stack_spec
        model_config.hybrid_stack_spec = self._resolve_hybrid_stack_spec()
        try:
            return super().build_model(pg_collection, pre_process, post_process, vp_stage)
        finally:
            model_config.hybrid_stack_spec = configured_spec


__all__ = [
    "DEFAULT_MAMBA_CHUNK_SIZE",
    "BridgeHybridModelBuilder",
    "HybridModelBuilder",
    "HybridModelConfig",
    "configure_mamba_chunk_size",
    "get_default_hybrid_stack_spec",
    "modelopt_hybrid_stack_spec",
    "transformer_engine_hybrid_stack_spec",
    "use_local_core_attention",
]
