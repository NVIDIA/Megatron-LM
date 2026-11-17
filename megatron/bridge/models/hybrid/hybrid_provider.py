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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/hybrid/hybrid_provider.py
# Source license: Apache-2.0.

import copy
import inspect
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Self

import torch

from megatron.bridge.models.hybrid.hybrid_builder import (
    DEFAULT_MAMBA_CHUNK_SIZE,
    configure_mamba_chunk_size,
    use_local_core_attention,
)
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils import fusions
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_inference_stack_spec as default_hybrid_inference_stack_spec,
)
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec as default_hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel as MCoreHybridModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.post_training.modelopt.hybrid.model_specs import get_hybrid_stack_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_hybrid_layer_allocation import (
    Symbols,
    get_hybrid_total_layer_count,
    parse_hybrid_pattern,
)
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnBackend

logger = logging.getLogger(__name__)


def modelopt_hybrid_stack_spec(config: "HybridModelProvider | None" = None) -> ModuleSpec:
    """Hybrid stack specification for quantization with ModelOpt.

    Uses Norm instead of TENorm and ColumnParallelLinear/RowParallelLinear
    instead of TE layers to enable proper quantizer insertion by ModelOpt.

    Args:
        config: Optional Hybrid configuration object.

    Returns:
        Module specification for quantization-ready Hybrid stack.
    """
    return get_hybrid_stack_modelopt_spec(local_core_attention=False, remap_te_layernorm=True)


def transformer_engine_hybrid_stack_spec() -> ModuleSpec:
    """Return the default Hybrid stack spec with Transformer Engine layers.

    This is a named function (not a lambda) to allow proper serialization
    and reconstruction from checkpoints. Named functions can be imported
    via their module path, unlike lambdas.

    Returns:
        Default Hybrid stack specification from megatron.core.
    """
    return default_hybrid_stack_spec


def get_default_hybrid_stack_spec(config: "HybridModelProvider") -> ModuleSpec:
    """Determine the most appropriate Hybrid stack specification based on configuration.

    Args:
        config: Hybrid configuration object.

    Returns:
        Appropriate module specification based on config.
    """
    if getattr(config, "transformer_impl", None) == "inference_optimized":
        return default_hybrid_inference_stack_spec
    if getattr(config, "restore_modelopt_state", False):
        return modelopt_hybrid_stack_spec(config)
    return transformer_engine_hybrid_stack_spec()


@dataclass
class HybridModelProvider(TransformerConfig, ModelProviderMixin[MCoreHybridModel]):
    """Configuration and provider for Megatron Core Hybrid models.

    This class extends TransformerConfig with Hybrid-specific parameters and
    provides a method to instantiate configured Hybrid models.
    """

    # Model configuration
    fp16_lm_cross_entropy: bool = False
    logit_dtype: torch.dtype | None = None
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    fp16: bool = False
    bf16: bool = True
    num_layers: int | None = None
    mamba_num_groups: int = 8
    mamba_chunk_size: int = DEFAULT_MAMBA_CHUNK_SIZE
    num_attention_heads: int = 1
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str | None = None
    hybrid_layer_pattern: str | None = None
    seq_length: int = 8192
    # HybridModel with no attention has no need for position embeddings, so none is default.
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none"
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: float | None = None
    scatter_embedding_sequence_parallel: bool = True
    apply_rope_fusion: bool = True
    make_vocab_size_divisible_by: int = 128
    gated_linear_unit: bool = False
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    attention_backend: AttnBackend = AttnBackend.flash
    deallocate_pipeline_outputs: bool = True
    bias_dropout_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    gradient_accumulation_fusion: bool = field(
        default_factory=fusions.can_enable_gradient_accumulation_fusion
    )
    hybrid_stack_spec: (
        ModuleSpec | Callable[[], ModuleSpec] | Callable[["HybridModelProvider"], ModuleSpec] | None
    ) = None
    vocab_size: int | None = None
    should_pad_vocab: bool = False
    hf_model_id: str | None = None
    """Optional HuggingFace model identifier associated with this provider."""

    hf_model_revision: str | None = None
    """Optional immutable HuggingFace revision used to construct this provider."""

    hf_model_text_only: bool = False
    """Whether the source HF checkpoint was projected to its language model."""

    _pg_collection: ProcessGroupCollection | None = None

    # MTP
    mtp_num_layers: int | None = 0
    mtp_hybrid_override_pattern: str | None = None
    keep_mtp_spec_in_bf16: bool = False

    # If True, restore modelopt_state that contains quantization, sparsity, and speculative decoding state.
    restore_modelopt_state: bool = False

    def finalize(self) -> None:
        """Finalize the Hybrid model provider.

        Calculates the number of layers from ``hybrid_layer_pattern`` and executes
        the deferred MCore post-init logic.
        """
        if self.mamba_chunk_size < 1:
            raise ValueError("mamba_chunk_size must be at least 1.")

        # Check if hybrid_override_pattern is specified and throw deprecation warning.
        used_hybrid_override_pattern = False
        if self.hybrid_override_pattern is not None:
            assert self.hybrid_layer_pattern is None, (
                "hybrid_override_pattern and hybrid_layer_pattern cannot both be specified. "
                "hybrid_override_pattern is deprecated; use hybrid_layer_pattern instead."
            )
            if get_rank_safe() == 0:
                warnings.warn(
                    "hybrid_override_pattern is deprecated. Use hybrid_layer_pattern instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.hybrid_layer_pattern = self.hybrid_override_pattern
            self.hybrid_override_pattern = None
            used_hybrid_override_pattern = True

        # Combine hybrid_layer_pattern (main decoder) with mtp_hybrid_override_pattern
        # into a single unified pattern that MCore HybridModel can parse.
        if self.hybrid_layer_pattern is not None and self.mtp_hybrid_override_pattern:
            sep = Symbols.MTP_SEPARATOR
            main_pattern = self.hybrid_layer_pattern.split(sep)[0]
            # When mtp_use_repeated_layer=True, the shared MTP layer always exists
            # in the model and mtp_num_layers controls forward pass repetitions.
            if self.mtp_use_repeated_layer:
                num_pattern_copies = max(1, self.mtp_num_layers or 0)
            else:
                num_pattern_copies = self.mtp_num_layers or 0
            self.hybrid_layer_pattern = (
                main_pattern
                + sep
                + sep.join([self.mtp_hybrid_override_pattern] * num_pattern_copies)
            )

            # Validate mtp_num_layers against the constructed pattern.
            if sep in self.hybrid_layer_pattern:
                parsed = parse_hybrid_pattern(self.hybrid_layer_pattern)
                if parsed.mtp_pattern and parsed.mtp_num_depths > 0:
                    inferred_mtp_num_layers = parsed.mtp_num_depths
                    if self.mtp_num_layers is None:
                        self.mtp_num_layers = inferred_mtp_num_layers
                    elif self.mtp_use_repeated_layer:
                        # With repeated layers, pattern count reflects architecture
                        # while mtp_num_layers controls forward pass repetitions.
                        pass
                    elif self.mtp_num_layers != inferred_mtp_num_layers:
                        logger.warning(
                            "mtp_num_layers (%s) conflicts with MTP depth count (%s) in pattern '%s'. "
                            "Using the inferred value (%s).",
                            self.mtp_num_layers,
                            inferred_mtp_num_layers,
                            self.hybrid_layer_pattern,
                            inferred_mtp_num_layers,
                        )
                        self.mtp_num_layers = inferred_mtp_num_layers

        # Check if hybrid_layer_pattern is specified and derive num_layers from pattern.
        if self.hybrid_layer_pattern is not None:
            num_layers_in_pattern = get_hybrid_total_layer_count(self.hybrid_layer_pattern)
            if self.num_layers is not None:
                if used_hybrid_override_pattern:
                    assert self.num_layers == num_layers_in_pattern, (
                        f"num_layers ({self.num_layers}) does not match the number of layers "
                        f"derived from hybrid_override_pattern ({num_layers_in_pattern}). "
                        f"Please correct num_layers or the pattern."
                    )
                else:
                    assert self.num_layers == num_layers_in_pattern, (
                        f"num_layers ({self.num_layers}) does not match the number of layers "
                        f"derived from hybrid_layer_pattern ({num_layers_in_pattern}). "
                        f"Please correct num_layers or the pattern."
                    )
            self.num_layers = num_layers_in_pattern

        super().finalize()

    def _resolve_hybrid_stack_spec(self) -> ModuleSpec:
        """Resolve the configured Hybrid stack spec."""
        hybrid_stack_spec = self.hybrid_stack_spec
        if hybrid_stack_spec is None:
            resolved_spec = get_default_hybrid_stack_spec(self)
        elif not isinstance(hybrid_stack_spec, ModuleSpec):
            if len(inspect.signature(hybrid_stack_spec).parameters) > 0:
                resolved_spec = hybrid_stack_spec(self)
            else:
                resolved_spec = hybrid_stack_spec()
        else:
            resolved_spec = hybrid_stack_spec

        if self.attention_backend in {AttnBackend.local, "local"}:
            resolved_spec = use_local_core_attention(resolved_spec)

        return configure_mamba_chunk_size(resolved_spec, self.mamba_chunk_size)

    def _copy_config_without_runtime_process_groups(self, *, deep: bool) -> Self:
        """Copy this config without the runtime-only process-group collection."""
        model_config = copy.copy(self)
        model_config._pg_collection = None
        return copy.deepcopy(model_config) if deep else model_config

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreHybridModel:
        """Configure and instantiate a Megatron Core Hybrid model based on this configuration.

        Args:
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage.
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage.
            vp_stage: Virtual pipeline stage.

        Returns:
            Configured Megatron Core Hybrid model instance.
        """
        hybrid_stack_spec = self._resolve_hybrid_stack_spec()

        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        vp_size = self.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(
                vp_stage=vp_stage, vp_size=vp_size
            ) and is_pp_first_stage(self._pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(
                vp_stage=vp_stage, vp_size=vp_size
            ) and is_pp_last_stage(self._pg_collection.pp)
        self._vp_stage = vp_stage

        # MCore creates independent per-layer configs by deep-copying this config. Detach
        # runtime-only process groups during construction and pass them through the dedicated
        # argument instead; torch.distributed.ProcessGroup cannot be deep-copied. The model must
        # retain this provider as its root config because Bridge installs DDP schedule callbacks
        # on the provider after wrapping the model.
        pg_collection = self._pg_collection
        self._pg_collection = None
        try:
            return MCoreHybridModel(
                config=self,
                hybrid_stack_spec=hybrid_stack_spec,
                vocab_size=padded_vocab_size,
                max_sequence_length=self.seq_length,
                hybrid_layer_pattern=self.hybrid_layer_pattern,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                **logit_dtype_kwarg(MCoreHybridModel, self.logit_dtype),
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                pre_process=pre_process,
                post_process=post_process,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
                scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
            )
        finally:
            self._pg_collection = pg_collection
