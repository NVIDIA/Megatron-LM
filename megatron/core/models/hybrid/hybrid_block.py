# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from megatron.core.context_parallel import ContextParallelLayoutManager, CPLayout, THDCPLayoutPlan
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TENorm
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    get_layer_type_list_from_layer_config_list,
    validate_segment_layers,
)
from megatron.core.models.hybrid.layers import utils as layer_utils
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.recompute import checkpointed_forward
from megatron.core.ssm.context_parallel.chunkwise import build_packed_sequence_cp_metadata
from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import annotate_first_last_layer
from megatron.core.transformer.hyper_connection import (
    HyperConnectionModule,
    learned_output_contract,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor


@dataclass
class HybridStackSubmodules:
    """
    A class for the module specs for the HybridStack.
    """

    mamba_layer: Union[ModuleSpec, type] = IdentityOp
    gdn_layer: Union[ModuleSpec, type] = IdentityOp
    attention_layer: Union[ModuleSpec, type] = IdentityOp
    dsa_layer: Union[ModuleSpec, type] = IdentityOp
    mla_layer: Union[ModuleSpec, type] = IdentityOp
    mlp_layer: Union[ModuleSpec, type] = IdentityOp
    moe_layer: Union[ModuleSpec, type] = IdentityOp
    mtp_block_spec: Optional[ModuleSpec] = None


class HybridStack(MegatronModule):
    """
    Constructor for the HybridStack class.

    Args:
        config (TransformerConfig): the model configuration
        submodules (HybridStackSubmodules): the submodules for the stack
        pre_process (bool, optional): whether to include an embedding layer.
            Defaults to True.
        layer_type_list (list[str], optional): This argument exists for backwards-compatibility
            reasons, allowing callers to construct ``HybridStack`` directly with layer symbols.
            It is immediately converted to independent per-layer configs.
        layer_config_list (Sequence[TransformerConfig], optional): per-layer configs for this
            pipeline segment. When provided by HybridModel, pipeline stage selection has already
            been done via '|' separators in the pattern. Exactly one of ``layer_type_list`` or
            ``layer_config_list`` must be provided.
        pp_layer_offset (int, optional): the global layer offset for this pipeline
            segment. Defaults to 0.
        post_layer_norm (bool, optional): whether to include a final layer norm.
            Defaults to True.
        post_process (bool, optional): whether to include an output layer.
            Defaults to True.
        device (optional): the device to use. Defaults to None.
        dtype (optional): the data type to use. Defaults to None.
        pg_collection (ProcessGroupCollection): the required model communication
            process groups to use.
        is_mtp_layer (bool, optional): whether this is an MTP layer. Defaults to False.
        boundary_layout (CPLayout, optional): CP layout at the stack boundary.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: HybridStackSubmodules,
        pre_process: bool = True,
        layer_type_list: list[str] | None = None,
        pp_layer_offset: int = 0,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
        layer_config_list: Sequence[TransformerConfig] | None = None,
        boundary_layout: CPLayout | None = None,
    ) -> None:
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        if (layer_type_list is None) == (layer_config_list is None):
            raise ValueError("Exactly one of layer_type_list or layer_config_list must be provided")
        if layer_type_list is not None:
            if any(
                not isinstance(layer_symbol, str) or len(layer_symbol) != 1
                for layer_symbol in layer_type_list
            ):
                raise ValueError("Each entry in layer_type_list must be a single layer symbol")
            segment = ''.join(layer_type_list)
            warnings.warn(
                "DEPRECATED(layer_type_list): please use `layer_config_list` instead",
                DeprecationWarning,
                stacklevel=2,
            )
            layer_config_list = validate_segment_layers(segment, config)

        for layer_config in layer_config_list:
            layer_utils.validate_tp_comm_overlap(
                layer_config,
                layer_utils.get_layer_symbol_from_config(layer_config),
                has_mtp=is_mtp_layer,
            )

        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_layer_norm = post_layer_norm
        self.post_process = post_process
        self.is_mtp_layer = is_mtp_layer
        boundary_layout = (
            self.config.linear_cp_layout if boundary_layout is None else boundary_layout
        )

        assert pg_collection is not None, "pg_collection must be provided for HybridStack"

        self.pp_group = pg_collection.pp
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.tp_cp_group = pg_collection.tp_cp

        # Required for pipeline parallel schedules
        self.input_tensor = None
        self.pg_collection = pg_collection

        self._mhc_block_end_plan: Optional[List[bool]] = None

        self.layer_config_list = layer_config_list
        self._has_linear_layer_with_chunkwise_cp = self.cp_group.size() > 1 and any(
            type(layer_config) is layer_utils.MambaLayerConfig
            and layer_config.linear_cp_mode == "chunkwise"
            for layer_config in self.layer_config_list
        )
        self._cp_layout_manager = None
        if self.cp_group.size() > 1:
            layer_layouts = tuple(
                (
                    layer_config.attention_cp_layout
                    if type(layer_config) in layer_utils.Symbols.ATTENTION_LAYER_CONFIGS
                    else layer_config.linear_cp_layout
                )
                for layer_config in self.layer_config_list
            )
            self._cp_layout_manager = ContextParallelLayoutManager(
                layer_layouts=layer_layouts,
                boundary_layout=boundary_layout,
                sequence_parallel=self.config.sequence_parallel,
                cp_group=self.cp_group,
                tp_group=self.tp_group,
                tp_cp_group=self.tp_cp_group,
            )
        if getattr(self.config, "mla_down_proj_fusion", False):
            submodules = self._fuse_mla_down_proj(submodules)

        # Build layers from the pre-selected segment
        self.layers = nn.ModuleList()
        for i, layer_config in enumerate(self.layer_config_list):
            layer_number = i + 1 + pp_layer_offset
            if layer_config.fp8:
                quant_init_context = get_fp8_context(
                    layer_config, i + pp_layer_offset, is_init=True
                )
            elif layer_config.fp4:
                quant_init_context = get_fp4_context(
                    layer_config, i + pp_layer_offset, is_init=True
                )
            else:
                quant_init_context = nullcontext()
            with quant_init_context:
                if type(layer_config) is layer_utils.MambaLayerConfig:
                    layer = build_module(
                        submodules.mamba_layer,
                        config=layer_config,
                        layer_number=layer_number,
                        pp_layer_offset=pp_layer_offset,
                        pg_collection=pg_collection,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif type(layer_config) is layer_utils.AttentionLayerConfig:
                    layer = build_module(
                        submodules.attention_layer,
                        config=layer_config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif type(layer_config) is layer_utils.DSALayerConfig:
                    layer = build_module(
                        submodules.dsa_layer,
                        config=layer_config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif type(layer_config) is layer_utils.MLALayerConfig:
                    layer = build_module(
                        submodules.mla_layer,
                        config=layer_config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif type(layer_config) is layer_utils.MLPLayerConfig:
                    layer = build_module(
                        submodules.mlp_layer,
                        config=layer_config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif type(layer_config) is layer_utils.MoELayerConfig:
                    layer = build_module(
                        submodules.moe_layer,
                        config=layer_config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif type(layer_config) is layer_utils.GDNLayerConfig:
                    gdn_layer_spec = submodules.gdn_layer
                    if layer_config.experimental_attention_variant == "gdn2":
                        # 'G' layers build the GDN2 variant when the gdn2 experimental
                        # attention variant is selected.
                        from megatron.core.ssm.gated_delta_net import GatedDeltaNet2

                        gdn_layer_spec = copy.deepcopy(gdn_layer_spec)
                        gdn_layer_spec.submodules.self_attention.module = GatedDeltaNet2
                    layer = build_module(
                        gdn_layer_spec,
                        config=layer_config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        # Set to False as we do not want to change offset.
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                else:
                    raise ValueError(
                        f"Unexpected hybrid layer config type: {type(layer_config).__name__}"
                    )

            if self.config.enable_mhc_connections:
                layer = HyperConnectionHybridLayer(config=layer_config, layer=layer)
            self.layers.append(layer)

        if self.config.cuda_graph_impl == "local":
            annotate_first_last_layer(self.layers)

        # Required for activation recomputation
        self.num_layers_per_pipeline_rank = len(self.layers)

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_norm = TENorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

        if self.config.enable_mhc_connections and self.post_process and not self.is_mtp_layer:
            hc_mult = self.config.mhc_num_residual_streams
            hc_dim = self.config.hidden_size * hc_mult
            self.hc_head_fn = mark_keep_in_fp32(nn.Parameter(torch.randn(hc_mult, hc_dim)))
            self.hc_head_base = mark_keep_in_fp32(nn.Parameter(torch.zeros(hc_mult)))
            self.hc_head_scale = mark_keep_in_fp32(nn.Parameter(torch.ones(1)))
            nn.init.xavier_uniform_(self.hc_head_fn)
            if self.config.sequence_parallel:
                setattr(self.hc_head_fn, 'sequence_parallel', True)
                setattr(self.hc_head_base, 'sequence_parallel', True)
                setattr(self.hc_head_scale, 'sequence_parallel', True)

    @property
    def layer_type_list(self) -> list[str]:
        """Return layer symbols derived from the per-layer configs.

        This property exists for backwards-compatibility reasons so callers that read
        ``HybridStack.layer_type_list`` continue to work. ``layer_config_list`` remains
        the source of truth.
        """
        return get_layer_type_list_from_layer_config_list(self.layer_config_list)

    def _fuse_mla_down_proj(self, submodules: HybridStackSubmodules) -> HybridStackSubmodules:
        # Avoid modifying the original object so users don't get surprised about their `submodules`
        # being modified underneath them.
        submodules = copy.deepcopy(submodules)
        mla_spec = submodules.mla_layer
        # We always fuse the input layernorm because Hybrid always uses TransformerEngine.
        mla_spec.submodules.input_layernorm = IdentityOp
        mla_spec.submodules.self_attention.module = FusedMLASelfAttention
        mla_spec.submodules.self_attention.submodules.linear_qkv_down_proj = (
            TELayerNormColumnParallelLinear
        )
        mla_spec.submodules.self_attention.submodules.linear_q_down_proj = None
        mla_spec.submodules.self_attention.submodules.linear_kv_down_proj = None
        mla_spec.submodules.sharded_state_dict_keys_map = {
            "self_attention.linear_q_down_proj.layer_norm_": "input_layernorm.",
            "self_attention.linear_kv_down_proj.layer_norm_": "input_layernorm.",
            "self_attention.linear_qkv_down_proj.layer_norm_": "input_layernorm.",
        }
        return submodules

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def mamba_state_shapes_per_request(self) -> Optional[Tuple[Tuple[int], Tuple[int]]]:
        """
        Returns the recurrent mixer's conv and SSM state shapes per input sequence
        if this block contains Mamba or GDN layers (this may not be the case with PP > 1).
        """
        for layer_config, layer in zip(self.layer_config_list, self.layers, strict=True):
            if type(layer_config) is layer_utils.MambaLayerConfig:
                return layer.mamba_state_shapes_per_request()
            if type(layer_config) is layer_utils.GDNLayerConfig:
                if hasattr(layer, 'mamba_state_shapes_per_request'):
                    state_shapes = layer.mamba_state_shapes_per_request()
                    if state_shapes is not None:
                        return state_shapes
                return layer.self_attention.mamba_state_shapes_per_request()
        return None

    def _compute_mhc_block_end_plan(self) -> List[bool]:
        """Compute deterministic per-layer mHC recompute block boundaries."""
        num_layers = len(self.layers)
        block_ends: List[bool] = [False] * num_layers
        if num_layers == 0:
            return block_ends

        layers_per_block = self.config.mhc_recompute_layer_num
        for layer_idx in range(num_layers):
            is_last_in_stack = layer_idx == num_layers - 1
            block_ends[layer_idx] = is_last_in_stack or (
                layers_per_block is not None and (layer_idx + 1) % layers_per_block == 0
            )
        return block_ends

    def _build_mhc_recompute_layer_plan(
        self, use_mhc_recompute: bool
    ) -> Tuple[List[Optional[CheckpointWithoutOutputManager]], List[bool]]:
        """Build single-use recompute managers for this forward pass."""
        num_layers = len(self.layers)
        if not use_mhc_recompute or num_layers == 0:
            return [None] * num_layers, [False] * num_layers

        if self._mhc_block_end_plan is None:
            self._mhc_block_end_plan = self._compute_mhc_block_end_plan()
        block_ends = self._mhc_block_end_plan

        layer_managers: List[Optional[CheckpointWithoutOutputManager]] = [None] * num_layers
        manager = CheckpointWithoutOutputManager()
        for layer_idx in range(num_layers):
            layer_managers[layer_idx] = manager
            if block_ends[layer_idx] and layer_idx != num_layers - 1:
                manager = CheckpointWithoutOutputManager()
        return layer_managers, block_ends

    @staticmethod
    def _finalize_mhc_recompute_layer(
        manager: Optional[CheckpointWithoutOutputManager], hidden_states: Tensor, is_block_end: bool
    ) -> None:
        """Finalize the current mHC recompute block when its last layer finishes."""
        if manager is not None and is_block_end:
            manager.discard_all_outputs_and_register_unified_recompute(hidden_states)

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask=None,
        packed_seq_params_by_layout: dict[CPLayout, PackedSeqParams | None] | None = None,
        cp_layout_plan: THDCPLayoutPlan | None = None,
    ):
        """
        Forward function of the HybridStack class.

        It either returns the Loss values if labels are given or the
            final hidden units

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): the input tensor.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): the attention mask.
            inference_context (BaseInferenceContext): the inference parameters.
            rotary_pos_emb (Tensor, optional): the rotary positional embeddings.
                Defaults to None.
        Returns:
            Tensor: the output tensor.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if self._has_linear_layer_with_chunkwise_cp and padding_mask is not None:
            raise NotImplementedError(
                "Hybrid chunkwise context parallelism does not support padding masks."
            )

        cp_layout_state = None
        if self._cp_layout_manager is not None:
            cp_layout_state = self._cp_layout_manager.build_forward_state(
                packed_seq_params,
                packed_seq_params_by_layout=packed_seq_params_by_layout,
                thd_plan=cp_layout_plan,
            )

        packed_sequence_cp_metadata = None
        if self._has_linear_layer_with_chunkwise_cp and packed_seq_params is not None:
            if packed_seq_params.seq_idx is None:
                raise ValueError("Packed chunkwise CP requires packed_seq_params.seq_idx")
            packed_sequence_cp_metadata = build_packed_sequence_cp_metadata(
                packed_seq_params.seq_idx,
                cp_rank=self.cp_group.rank(),
                cp_size=self.cp_group.size(),
            )

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if self.config.enable_mhc_connections and self.pre_process and not self.is_mtp_layer:
            hidden_states = HyperConnectionModule.input_expand(
                hidden_states, self.config.mhc_num_residual_streams
            )

        if inference_context and inference_context.is_static_batching():
            # NOTE(bnorick): match BaseInferenceContext attributes for
            # mamba_ssm.utils.generation.BaseInferenceContext,
            # this hack supports eval
            inference_context.max_seqlen = inference_context.max_sequence_length
            inference_context.seqlen_offset = inference_context.sequence_len_offset

        if (
            (self.config.cuda_graph_impl == "local" or self.config.flash_decode)
            and inference_context
            and inference_context.is_static_batching()
            and InferenceMode.is_active()
        ):
            current_batch_size = hidden_states.shape[1]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device='cuda',
            )
        else:
            sequence_len_offset = None

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        use_fp4_context = self.config.fp4 is not None
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        if use_inner_fp8_context:

            def get_inner_quant_context(config, layer_number):
                return get_fp8_context(config, layer_number)

        elif use_fp4_context:

            def get_inner_quant_context(config, layer_number):
                return get_fp4_context(config, layer_number)

        else:

            def get_inner_quant_context(config, layer_number):
                return nullcontext()

        use_mhc_recompute = (
            self.training
            and self.config.enable_mhc_connections
            and self.config.recompute_granularity == 'selective'
            and "mhc" in self.config.recompute_modules
        )
        mhc_layer_managers, mhc_block_ends = self._build_mhc_recompute_layer_plan(use_mhc_recompute)

        with outer_fp8_context:
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = checkpointed_forward(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=None,
                    context_mask=None,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=None,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                    use_inner_quantization_context=(use_inner_fp8_context or use_fp4_context),
                    cp_layout_state=cp_layout_state,
                    packed_sequence_cp_metadata=packed_sequence_cp_metadata,
                )
            else:
                for layer_idx, (layer_config, layer) in enumerate(
                    zip(self.layer_config_list, self.layers, strict=True)
                ):
                    layer_packed_seq_params = packed_seq_params
                    if cp_layout_state is not None:
                        hidden_states, layer_packed_seq_params = cp_layout_state.prepare_layer(
                            layer_idx, hidden_states
                        )
                    # Layers have 1-indexed layer numbers attribute.
                    inner_quant_context = get_inner_quant_context(
                        layer_config, layer.layer_number - 1
                    )
                    mhc_manager = mhc_layer_managers[layer_idx]
                    if mhc_manager is not None:
                        mhc_manager.is_last_layer_in_recompute_block = mhc_block_ends[layer_idx]
                    layer_cp_metadata = (
                        packed_sequence_cp_metadata
                        if type(layer_config) is layer_utils.MambaLayerConfig
                        and layer_config.linear_cp_mode == "chunkwise"
                        else None
                    )

                    with inner_quant_context:
                        if isinstance(layer, (TransformerLayer, HyperConnectionHybridLayer)):
                            layer_kwargs = dict(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                inference_context=inference_context,
                                rotary_pos_emb=rotary_pos_emb,
                                sequence_len_offset=sequence_len_offset,
                                packed_seq_params=layer_packed_seq_params,
                                padding_mask=padding_mask,
                            )
                            if layer_cp_metadata is not None:
                                layer_kwargs["packed_sequence_cp_metadata"] = layer_cp_metadata
                            if mhc_manager is not None and isinstance(
                                layer, HyperConnectionHybridLayer
                            ):
                                layer_kwargs["mhc_recompute_manager"] = mhc_manager
                            hidden_states, _ = layer(**layer_kwargs)
                        elif layer_cp_metadata is not None:
                            hidden_states = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                inference_context=inference_context,
                                packed_seq_params=layer_packed_seq_params,
                                packed_sequence_cp_metadata=layer_cp_metadata,
                            )
                        else:  # MambaLayer, Expert, or MLP
                            hidden_states = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                inference_context=inference_context,
                                packed_seq_params=layer_packed_seq_params,
                            )

                    # The attention layer (currently a simplified transformer layer)
                    # outputs a tuple of (hidden_states, context). Context is intended
                    # for cross-attention, and is not needed in our model.
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    if cp_layout_state is not None:
                        hidden_states = cp_layout_state.finalize_layer(layer_idx, hidden_states)

                    self._finalize_mhc_recompute_layer(
                        manager=mhc_manager,
                        hidden_states=hidden_states,
                        is_block_end=mhc_block_ends[layer_idx],
                    )

        mhc_multistream = None
        if self.config.enable_mhc_connections and self.post_process and not self.is_mtp_layer:
            if (self.config.mtp_num_layers or 0) > 0:
                mhc_multistream = hidden_states
            hidden_states = learned_output_contract(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
                self.config.mhc_num_residual_streams,
                self.config.layernorm_epsilon,
            )

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_norm(hidden_states)

        # Ensure that the tensor passed between pipeline parallel stages is
        # viewless. See related notes in TransformerBlock and TransformerLayer
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        if mhc_multistream is not None:
            return hidden_states, mhc_multistream
        return hidden_states

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Optional[tuple] = None,
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """
        Returns a sharded state dictionary for the current object.

        This function constructs a sharded state dictionary by iterating over the layers
        in the current object, computing the sharded state dictionary for each layer,
        and combining the results into a single dictionary.

        Parameters:
            prefix (str): The prefix to use for the state dictionary keys.
            sharded_offsets (tuple): The sharded offsets to use for the state dictionary.
            metadata (dict): Additional metadata to use when computing the sharded state dictionary.

        Returns:
            dict: The sharded state dictionary for the current object.
        """

        sharded_offsets = sharded_offsets or ()
        sharded_state_dict = {}
        layer_prefix = f'{prefix}layers.'

        for local_layer_idx, layer in enumerate(self.layers):

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = (
                f'{layer_prefix}{local_layer_idx}.'  # module list index in HybridStack
            )

            sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )

            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module,
                        f'{prefix}{name}.',
                        sharded_offsets,
                        metadata,
                        tp_group=self.tp_group,
                    )
                )

        local_state_dict: dict = {}
        self._save_to_state_dict(local_state_dict, '', keep_vars=True)
        if local_state_dict:
            metadata = ensure_metadata_has_dp_cp_group(metadata)
            sharded_state_dict.update(
                make_sharded_tensors_for_checkpoint(
                    local_state_dict,
                    prefix,
                    sharded_offsets=sharded_offsets,
                    tp_group=self.tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            )

        return sharded_state_dict


# Backward-compatible aliases
MambaStackSubmodules = HybridStackSubmodules
MambaStack = HybridStack
