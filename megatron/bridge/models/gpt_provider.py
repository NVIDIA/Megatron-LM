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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/gpt_provider.py
# Source license: Apache-2.0.

import contextlib
import inspect
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Literal, Optional, Union

import torch

from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils import fusions
from megatron.bridge.utils.cuda_graph import validate_cuda_graph_configuration
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.dot_product_attention import (
    DotProductAttention as MCoreDotProductAttention,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import get_pg_rank

logger = logging.getLogger(__name__)


def transformer_engine_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Create a Transformer Engine layer specification based on the provided config."""
    spec_params = inspect.signature(get_gpt_layer_with_transformer_engine_spec).parameters
    if "use_te_op_fuser" in spec_params:
        kwargs = {"use_te_op_fuser": config.use_transformer_engine_op_fuser}
    else:
        kwargs = {}
    if "use_grouped_gemm_for_dense_mlp" in spec_params:
        kwargs["use_grouped_gemm_for_dense_mlp"] = config.dense_grouped_gemm
    return get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        fp8=bool(config.num_moe_experts and (config.fp8 is not None)),
        **kwargs,
    )


def transformer_engine_full_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Create a full Transformer Engine layer specification with autocast support.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for full TE layers
    """
    from megatron.bridge.models.gpt_full_te_layer_autocast_spec import (
        get_gpt_full_te_layer_autocast_spec,
    )

    return get_gpt_full_te_layer_autocast_spec(transformer_config=config)


def local_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Create a local layer specification without Transformer Engine.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for local implementation layers
    """
    return get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        normalization=config.normalization,
    )


def modelopt_transformer_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Layer specification for quantization with ModelOpt."""
    # arbitrary attention mask is used for speculative decoding training
    # When context parallel > 1, only causal mask type is supported
    from megatron.core import parallel_state

    use_arbitrary_attention_mask = (
        config.use_arbitrary_attention_mask
        if config.use_arbitrary_attention_mask is not None
        else parallel_state.get_context_parallel_world_size() == 1
    )
    return get_gpt_modelopt_spec(
        config=config,
        local_core_attention=False,
        remap_te_layernorm=True,
        real_quant_cfg="None",
        use_arbitrary_attention_mask=use_arbitrary_attention_mask,
    )


def default_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Determine the most appropriate layer specification based on availability."""
    if config.use_transformer_engine_full_layer_spec:
        return transformer_engine_full_layer_spec(config)
    else:
        return transformer_engine_layer_spec(config)


@dataclass
class GPTModelProvider(TransformerConfig, ModelProviderMixin[MCoreGPTModel]):
    """Configuration and provider for Megatron Core GPT models.

    This class extends TransformerConfig with GPT-specific parameters and
    provides a method to instantiate configured GPT models.
    """

    # Model configuration
    fp16_lm_cross_entropy: bool = False
    logit_dtype: torch.dtype | None = None
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope", "yarn"] = "learned_absolute"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    rope_scaling: bool = False
    rope_scaling_factor: float = 1.0
    rotary_scaling_factor: Optional[float] = None
    seq_len_interpolation_factor: Optional[float] = None

    # YaRN parameters (yarn_*) are declared on the base TransformerConfig.

    seq_length: int = 1024
    attention_softmax_in_fp32: bool = False
    deallocate_pipeline_outputs: bool = True
    scatter_embedding_sequence_parallel: bool = True
    tp_only_amax_red: bool = False
    tp_comm_overlap_cfg: Optional[Union[str, dict[str, Any]]] = None
    """Config file when tp_comm_overlap is enabled."""

    use_transformer_engine_full_layer_spec: bool = False
    use_transformer_engine_op_fuser: bool = False
    dense_grouped_gemm: bool = False
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = (
        default_layer_spec
    )

    mtp_layer_spec_transform: Optional[Callable[["GPTModelProvider", ModuleSpec], ModuleSpec]] = (
        None
    )
    """Optional fix-up applied to an MTP layer spec re-derived straight from MCore.

    A standalone MTP pipeline stage owns no decoder layers, so ``mtp_block_spec`` cannot
    reuse ``transformer_layer_spec``'s output and calls ``get_gpt_decoder_layer_specs``
    instead. Any model whose layer spec is not plain MCore therefore loses its
    customisation on exactly that stage. Set this to re-apply it.
    """

    hf_model_id: str | None = None
    """Optional HuggingFace model identifier associated with this provider."""

    hf_model_revision: str | None = None
    """Optional immutable HuggingFace revision used to construct this provider."""

    # This represents the unpadded vocab size
    # The padded vocab size is automatically calculated in the provide() method.
    vocab_size: Optional[int] = None
    # Set if the tokenizer provides the vocab size. In this case, the vocab size will be padded
    # Controls whether vocab size should be padded for tensor parallelism
    should_pad_vocab: bool = False

    # MoE / FP8
    num_moe_experts: Optional[int] = None
    moe_grouped_gemm: bool = False
    qk_layernorm: bool = False
    fp8: Optional[str] = None
    normalization: str = "LayerNorm"

    # Multi-token prediction
    mtp_enabled: bool = False

    # Additional parameters that might be needed
    init_model_with_meta_device: bool = False
    use_te_rng_tracker: bool = False
    virtual_pipeline_model_parallel_size: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # Fusions
    masked_softmax_fusion: bool = True
    cross_entropy_loss_fusion: bool = True  # Generally beneficial, no specific dependencies
    gradient_accumulation_fusion: bool = field(
        default_factory=fusions.can_enable_gradient_accumulation_fusion
    )

    # If True, restore the modelopt_state that contains quantization, sparsity, speculative decoding transformation state.
    restore_modelopt_state: bool = False

    # Whether to use AttnMaskType.arbitrary in the ModelOpt spec.
    # If None, it will be determined by the default behavior (arbitrary only when context_parallel==1).
    # Set to False when using packed/remove-padding (THD) data format.
    use_arbitrary_attention_mask: Optional[bool] = None

    _pg_collection: Optional[ProcessGroupCollection] = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Configure and instantiate a Megatron Core GPT model based on this configuration.

        Args:
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage
            vp_stage: Virtual pipeline stage

        Returns:
            MCoreGPTModel: Configured Megatron Core GPT model instance
        """
        # Validate fusion configurations
        if not fusions.validate_rope_fusion_compatibility(self):
            self.apply_rope_fusion = False

        validate_cuda_graph_configuration(self)
        if self.cuda_graph_impl != "none":
            assert getattr(self, "use_te_rng_tracker", False), (
                "Transformer engine's RNG tracker is required for cudagraphs, it can be "
                "enabled with use_te_rng_tracker=True'."
            )

        vp_size = self.virtual_pipeline_model_parallel_size
        is_pipeline_asymmetric = getattr(
            self, "account_for_embedding_in_pipeline_split", False
        ) or getattr(self, "account_for_loss_in_pipeline_split", False)
        is_pipeline_asymmetric |= (
            getattr(self, "num_layers_in_first_pipeline_stage", None)
            or getattr(self, "num_layers_in_last_pipeline_stage", None)
        ) is not None
        is_flexible_pp_layout = is_pipeline_asymmetric or (
            getattr(self, "pipeline_model_parallel_layout", None) is not None
        )
        if vp_size and not is_flexible_pp_layout:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(
                self, **_callable_spec_kwargs(transformer_layer_spec, self, vp_stage)
            )

        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        # Initialize model as meta data instead of allocating data on a device
        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(torch.device, device="meta")

        if self.attention_backend == AttnBackend.local:
            if hasattr(transformer_layer_spec, "submodules"):
                transformer_layer_spec.submodules.self_attention.submodules.core_attention = (
                    MCoreDotProductAttention
                )
        # Determine pre/post flags if not provided using vp + pp stage
        if pre_process is None:
            pre_process = is_vp_first_stage(
                vp_stage=vp_stage, vp_size=vp_size
            ) and is_pp_first_stage(self._pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(
                vp_stage=vp_stage, vp_size=vp_size
            ) and is_pp_last_stage(self._pg_collection.pp)
        # Expose vp stage on config for downstream modules (e.g., TE layers)
        # so they can compute correct offsets without legacy globals.
        self._vp_stage = vp_stage
        with model_init_device_context():
            model = MCoreGPTModel(
                self,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=padded_vocab_size,
                max_sequence_length=self.seq_length,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                **logit_dtype_kwarg(MCoreGPTModel, self.logit_dtype),
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                rope_scaling=self.rope_scaling,
                rope_scaling_factor=self.rope_scaling_factor,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                pre_process=pre_process,
                post_process=post_process,
                scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
                pg_collection=self._pg_collection,
                vp_stage=vp_stage,
                mtp_block_spec=mtp_block_spec(self, vp_stage=vp_stage),
            )

        # If using full TE layer, need to set TP, CP group since the module call
        # is not routed through megatron core, which normally handles passing the
        # TP, CP group to the TE modules.
        # Deep iterate but skip self to avoid infinite recursion.
        if self.use_transformer_engine_full_layer_spec:
            # Copied from:
            # https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py
            if self._pg_collection.tp.size() > 1:
                for index, child in enumerate(model.modules()):
                    if index == 0:
                        continue
                    if hasattr(child, "set_tensor_parallel_group"):
                        tp_group = self._pg_collection.tp
                        child.set_tensor_parallel_group(tp_group)

            if self._pg_collection.cp.size() > 1:
                cp_stream = torch.cuda.Stream()
                for index, child in enumerate(model.modules()):
                    if index == 0:
                        continue
                    if hasattr(child, "set_context_parallel_group"):
                        cp_group = self._pg_collection.cp
                        cp_global_ranks = torch.distributed.get_process_group_ranks(cp_group)
                        child.set_context_parallel_group(cp_group, cp_global_ranks, cp_stream)

        return model


def _callable_spec_kwargs(
    spec_fn: Callable,
    config: "GPTModelProvider",
    vp_stage: Optional[int],
    pp_rank: Optional[int] = None,
) -> dict[str, Any]:
    """Build the keyword arguments a callable transformer layer spec declares.

    An explicit ``pp_rank`` takes precedence over the rank of the provider's own groups.
    """
    params = inspect.signature(spec_fn).parameters
    kwargs: dict[str, Any] = {}
    if "vp_stage" in params:
        kwargs["vp_stage"] = vp_stage
    if "pp_rank" in params:
        if pp_rank is None:
            # Resolve the pipeline rank from the provider's own groups so that block spec
            # builders do not fall back to the MPU globals, which decentralized runs never set.
            pg_collection = getattr(config, "_pg_collection", None)
            if pg_collection is not None:
                pp_rank = get_pg_rank(pg_collection.pp)
        if pp_rank is not None:
            kwargs["pp_rank"] = pp_rank
    return kwargs


def mtp_block_spec(
    config: "GPTModelProvider", vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> Optional[ModuleSpec]:
    """Pass in the MTP block spec if model has MTP layers.

    Args:
        config: GPT configuration object
        vp_stage: Virtual pipeline stage to build the spec for.
        pp_rank: Pipeline rank to build the spec for. Callers that build specs without the MPU
            globals, such as MegatronMIMO, must pass it; otherwise MCore falls back to them.

    Returns:
        ModuleSpec: The MTP module specification
    """
    if getattr(config, "mtp_num_layers", None):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

        if isinstance(config.transformer_layer_spec, Callable):
            spec = config.transformer_layer_spec(
                config,
                **_callable_spec_kwargs(config.transformer_layer_spec, config, vp_stage, pp_rank),
            )
        else:
            spec = config.transformer_layer_spec
        if hasattr(spec, "layer_specs") and len(spec.layer_specs) == 0:
            # Get the decoder layer spec explicitly if no decoder layer in the last stage,
            # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
            # Re-derive all decoder layer specs and use the last one to get the correct
            # layer type (dense vs MoE) for the MTP transformer layer.
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs

            decoder_layer_specs = get_gpt_decoder_layer_specs(
                config,
                use_transformer_engine=True,
                normalization=config.normalization,
                qk_l2_norm=config.qk_l2_norm,
            )
            spec = decoder_layer_specs[-1]
            # This spec came from MCore directly, so anything transformer_layer_spec would
            # have changed is absent from it. Give the model a chance to re-apply it.
            transform = getattr(config, "mtp_layer_spec_transform", None)
            if transform is not None:
                spec = transform(config, spec)
        return get_gpt_mtp_block_spec(
            config, spec, use_transformer_engine=True, vp_stage=vp_stage, pp_rank=pp_rank
        )
    else:
        return None
