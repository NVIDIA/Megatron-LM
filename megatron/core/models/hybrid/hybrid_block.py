# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import CheckpointManager
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.hyper_connection import (
    HyperConnectionModule,
    learned_output_contract,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
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
    csa_layer: Union[ModuleSpec, type] = IdentityOp
    hca_layer: Union[ModuleSpec, type] = IdentityOp
    window_layer: Union[ModuleSpec, type] = IdentityOp
    mlp_layer: Union[ModuleSpec, type] = IdentityOp
    moe_layer: Union[ModuleSpec, type] = IdentityOp
    mtp_block_spec: Optional[ModuleSpec] = None


class HyperConnectionHybridLayer(GraphableMegatronModule):
    """Layer-boundary mHC wrapper for HybridStack layers.

    Hybrid layers already own their local residual paths. For this initial
    integration we treat each hybrid layer as a single function by aggregating
    n streams to the layer input, running the existing layer, and feeding only
    the layer delta back through mHC expansion. The expansion path intentionally
    uses zero additional dropout because the wrapped hybrid layer has already
    applied its local dropout/residual update before the delta is computed.

    Checkpoint compatibility: this is a *wrapper* (the inner layer is held as
    `self.inner_layer`), so wrapped-layer state_dict keys are nested under
    `inner_layer.` (e.g. `layers.0.inner_layer.input_layernorm.weight` instead
    of `layers.0.input_layernorm.weight`). HybridStack checkpoints saved with
    `enable_hyper_connections=False` cannot be loaded into a model with
    `enable_hyper_connections=True` (and vice versa) without a key-mapping
    migration. Note: this differs from `HyperConnectionTransformerLayer`,
    which subclasses `TransformerLayer` and only adds new sibling fields,
    keeping all base keys stable.

    CUDA graphs: this wrapper subclasses ``GraphableMegatronModule`` so that, with
    ``cuda_graph_impl="transformer_engine"``, the whole wrapper forward (mHC
    aggregate + inner layer + n-stream BDA) is captured as one per-layer graph —
    mirroring ``HyperConnectionTransformerLayer`` on the GPT path. Without this,
    the TE graph discovery (``_layer_is_graphable``) only inspects the top-level
    layer type and silently skips every wrapped layer, so an mHC-enabled
    HybridStack would run entirely eager. The inner layer's own ``__call__`` graph
    routing is bypassed during capture (see ``_call_inner_layer``) to avoid nested
    capture; the inner layer's params are still covered by the wrapper graph's
    manual hooks because ``_get_submodules_under_cudagraphs`` returns ``[self]``.
    """

    def __init__(self, config: TransformerConfig, layer: MegatronModule) -> None:
        super().__init__(config=config)
        self.inner_layer = layer
        self.layer_number = layer.layer_number
        self.hyper_connection = HyperConnectionModule(config=config, layer_number=self.layer_number)
        if config.params_dtype is not None:
            self.hyper_connection.to(dtype=config.params_dtype)
        if hasattr(layer, 'tp_group'):
            self.tp_group = layer.tp_group

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Override to produce n-stream hidden_states of shape [s, b, n*C].

        CUDA graph capture allocates static buffers sized by this method. The base
        returns [s, b, C], but mHC layers carry n-stream hidden states [s, b, n*C].
        Mirrors ``HyperConnectionTransformerLayer.get_layer_static_inputs``.
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        hs = static_inputs["hidden_states"]
        n = self.config.num_residual_streams
        static_inputs["hidden_states"] = torch.ones(
            (hs.shape[0], hs.shape[1], n * self.config.hidden_size),
            dtype=hs.dtype,
            requires_grad=hs.requires_grad,
            device=hs.device,
        )
        return static_inputs

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """Capture the whole wrapper forward as one graph.

        The wrapper forward returns ``(hidden_states, context)``; ``context`` is
        ``None`` for the hybrid layer types that participate in graphing (no
        cross-attention), so it is dropped from the captured outputs — a tuple
        containing ``None`` cannot be a CUDA-graph output. Mirrors the
        ``context is not None`` handling in ``TransformerLayer._te_cuda_graph_capture``.
        """
        hidden_states, context = self.forward(*args, **kwargs)
        cuda_graph_outputs = [hidden_states]
        if context is not None:
            cuda_graph_outputs.append(context)
        return tuple(cuda_graph_outputs)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """Replay the captured wrapper graph and restore the (hidden_states, context) contract.

        The whole wrapper forward is captured as one graph whose only output is the
        layer's n-stream hidden_states (``context`` is ``None`` for the graphed
        hybrid layer types, so it was dropped at capture). The HybridStack forward
        loop unpacks ``hidden_states, _ = layer(...)``, so re-append ``None`` here.
        """
        cuda_graph_output = list(super()._te_cuda_graph_replay(*args, **kwargs))
        return cuda_graph_output[0], None

    def mamba_state_shapes_per_request(self) -> Optional[Tuple[Tuple[int], Tuple[int]]]:
        """Delegate Mamba inference state shape requests to the wrapped layer."""
        if not hasattr(self.inner_layer, 'mamba_state_shapes_per_request'):
            return None
        return self.inner_layer.mamba_state_shapes_per_request()

    def _call_inner_layer(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext],
        rotary_pos_emb: Optional[Tensor],
        sequence_len_offset: Optional[Tensor],
        packed_seq_params: Optional[PackedSeqParams],
        padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # When this wrapper is itself being CUDA-graph captured, the inner layer
        # must run as a plain forward: routing through its ``__call__`` would
        # trigger nested TE graph capture (the inner layer is also a
        # GraphableMegatronModule). During eager steps we keep ``__call__`` so the
        # inner layer's forward pre-hooks (e.g. param all-gather) fire normally;
        # under graph replay these are driven by the wrapper's manual hooks.
        from megatron.core.transformer.cuda_graphs import is_graph_capturing

        inner = self.inner_layer.forward if is_graph_capturing() else self.inner_layer

        if isinstance(self.inner_layer, TransformerLayer):
            output = inner(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
                _called_from_hybrid_mhc_wrapper=True,
            )
        else:
            # Non-transformer layers (e.g. MambaLayer; GatedDeltaNet which does
            # accept `sequence_len_offset` is currently always wrapped inside a
            # TransformerLayer spec, so it takes the branch above) do not accept
            # rotary_pos_emb / sequence_len_offset / padding_mask — pass only
            # the common arguments. New layer types that consume any of these
            # must add explicit handling here.
            output = inner(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )

        if isinstance(output, tuple):
            context = output[1] if len(output) > 1 else None
            return output[0], context
        return output, None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        sequence_len_offset: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[Tensor] = None,
        mhc_recompute_manager=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the wrapped hybrid layer through one layer-boundary mHC update.

        ``attention_mask`` defaults to ``None`` so that CUDA-graph capture, which
        calls this forward with only the static ``hidden_states`` input, does not
        fail on a missing positional argument (causal masking is inferred by the
        attention backend when the mask is ``None``).
        """
        residual = hidden_states
        aggregated, h_res, h_post = self.hyper_connection(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager
        )
        layer_output, context = self._call_inner_layer(
            aggregated,
            attention_mask,
            inference_context,
            rotary_pos_emb,
            sequence_len_offset,
            packed_seq_params,
            padding_mask,
        )
        # The inner hybrid layer already applied its own local residual/dropout, so
        # it returns `aggregated + f(aggregated)`. We feed only the function
        # delta `f(aggregated)` into the n-stream BDA so it does not double-count
        # the residual that mHC owns. The temporary [s, b, C] tensor here is the
        # simplest correct form; a future optimization could fuse the subtraction
        # into `fused_h_res_h_post_bda` to avoid the allocation.
        # Sanity check: this contract requires the inner layer to preserve shape;
        # any mismatch indicates a future layer type is breaking the residual
        # assumption and would silently corrupt the n-stream state.
        if layer_output.shape != aggregated.shape:
            raise RuntimeError(
                "HyperConnectionHybridLayer requires inner layers to preserve "
                f"hidden-state shape. Got {tuple(layer_output.shape)} from inner layer "
                f"vs {tuple(aggregated.shape)} input; layer must add its own residual."
            )
        # `fp32_residual_connection=True` may cause some inner layers (e.g.,
        # MambaLayer) to return `layer_output` in fp32 while `aggregated` is in
        # compute dtype; explicitly upcast `aggregated` so the subtraction stays
        # in fp32 instead of relying on PyTorch's implicit promotion.
        if self.config.fp32_residual_connection and aggregated.dtype != layer_output.dtype:
            aggregated = aggregated.to(layer_output.dtype)
        layer_delta = layer_output - aggregated
        # `dropout_prob=0.0` already disables dropout regardless of training mode;
        # `training=self.training` is more semantically accurate than hard-coding
        # False during a training-mode forward.
        is_last_in_recompute_block = bool(
            mhc_recompute_manager is not None
            and getattr(mhc_recompute_manager, "is_last_layer_in_recompute_block", False)
        )
        mhc_bda_manager = None if is_last_in_recompute_block else mhc_recompute_manager

        hidden_states = self.hyper_connection.fused_h_res_h_post_bda(
            h_res,
            residual,
            h_post,
            (layer_delta, None),
            dropout_prob=0.0,
            training=self.training,
            fused=False,
            manager=mhc_bda_manager,
        )
        # In `HyperConnectionTransformerLayer` the n-stream output stays in compute
        # dtype because the post-attention `x` is in compute dtype. In the hybrid
        # wrapper, `layer_delta` may be fp32 (when `fp32_residual_connection=True`
        # or an inner layer upcasts), so `fused_h_res_h_post_bda`'s `output.to(x.dtype)`
        # would leave the result in fp32 and silently propagate fp32 n-stream
        # hidden states to every subsequent layer (~2x activation memory). Restore
        # the compute-dtype contract here.
        if (
            self.config.fp32_residual_connection
            and self.config.params_dtype is not None
            and hidden_states.dtype != self.config.params_dtype
        ):
            hidden_states = hidden_states.to(self.config.params_dtype)
        return hidden_states, context


class HybridStack(MegatronModule):
    """
    Constructor for the HybridStack class.

    Args:
        config (TransformerConfig): the model configuration
        submodules (HybridStackSubmodules): the submodules for the stack
        pre_process (bool, optional): whether to include an embedding layer.
            Defaults to True.
        layer_type_list (list, optional): pre-computed list of layer type symbols for
            this pipeline segment. When provided (by HybridModel), pipeline stage
            selection has already been done via '|' separators in the pattern.
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
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: HybridStackSubmodules,
        pre_process: bool = True,
        layer_type_list: Optional[list[str]] = None,
        pp_layer_offset: int = 0,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_layer_norm = post_layer_norm
        self.post_process = post_process
        self.is_mtp_layer = is_mtp_layer

        assert pg_collection is not None, "pg_collection must be provided for HybridStack"

        self.pp_group = pg_collection.pp
        self.tp_group = pg_collection.tp

        # Required for pipeline parallel schedules
        self.input_tensor = None
        self.pg_collection = pg_collection

        # Lazily populated mHC recompute layout cache (deterministic from config
        # and num_layers); see `_build_mhc_recompute_layer_plan`.
        self._mhc_block_end_plan: Optional[List[bool]] = None

        assert layer_type_list is not None, (
            "layer_type_list must be provided. It should be pre-computed from "
            "--hybrid-layer-pattern by HybridModel."
        )
        self.layer_type_list = layer_type_list

        # Build layers from the pre-selected segment
        self.layers = nn.ModuleList()
        for i, layer_type in enumerate(self.layer_type_list):
            layer_number = i + 1 + pp_layer_offset
            if self.config.fp8:
                quant_init_context = get_fp8_context(self.config, i + pp_layer_offset, is_init=True)
            elif self.config.fp4:
                quant_init_context = get_fp4_context(self.config, i + pp_layer_offset, is_init=True)
            else:
                quant_init_context = nullcontext()
            with quant_init_context:
                if layer_type == LayerSymbols.MAMBA:
                    layer = build_module(
                        submodules.mamba_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pp_layer_offset=pp_layer_offset,
                        pg_collection=pg_collection,
                    )
                elif layer_type == LayerSymbols.ATTENTION:
                    layer = build_module(
                        submodules.attention_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif layer_type == LayerSymbols.DS_ATTENTION:
                    layer = build_module(
                        submodules.dsa_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif layer_type == LayerSymbols.CSA:
                    # DSv4 Compressed Sparse Attention (compress_ratio fixed by the spec).
                    layer = build_module(
                        submodules.csa_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif layer_type == LayerSymbols.HCA:
                    # DSv4 Heavily Compressed Attention (compress_ratio fixed by the spec).
                    layer = build_module(
                        submodules.hca_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif layer_type == LayerSymbols.WINDOW:
                    # DSv4 sliding-window-only attention (compress_ratio=0 fixed by the spec;
                    # no compressor / no top-k indexer — attends only within csa_window_size).
                    layer = build_module(
                        submodules.window_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif layer_type == LayerSymbols.MLP:
                    layer = build_module(
                        submodules.mlp_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                    )
                elif layer_type == LayerSymbols.MOE:
                    layer = build_module(
                        submodules.moe_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                    )
                elif layer_type == LayerSymbols.GDN:
                    layer = build_module(
                        submodules.gdn_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        # Set to False as we do not want to change offset.
                        add_layer_offset=False,
                    )
                else:
                    raise ValueError("unexpected layer_type")
            if self.config.enable_hyper_connections:
                layer = HyperConnectionHybridLayer(config=self.config, layer=layer)
            self.layers.append(layer)

        # Required for activation recomputation
        self.num_layers_per_pipeline_rank = len(self.layers)

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_norm = TENorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

        # Skip hc_head_* params inside the nested MTP HybridStack — `forward()`
        # no longer calls `learned_output_contract` there (MTP owns that), so these
        # params would be orphaned and break DDP's per-param grad-ready accounting
        # with a `len(per_param_grad_ready_counts) != len(params)` AssertionError.
        if self.config.enable_hyper_connections and self.post_process and not self.is_mtp_layer:
            hc_mult = self.config.num_residual_streams
            hc_dim = self.config.hidden_size * hc_mult
            self.hc_head_fn = nn.Parameter(torch.randn(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.zeros(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.ones(1))
            nn.init.xavier_uniform_(self.hc_head_fn)
            if self.config.sequence_parallel:
                setattr(self.hc_head_fn, 'sequence_parallel', True)
                setattr(self.hc_head_base, 'sequence_parallel', True)
                setattr(self.hc_head_scale, 'sequence_parallel', True)

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
        Returns the Mamba conv and ssm states shapes per input sequence
        if this block contains Mamba layers (this may not be the case with PP > 1).
        """
        for layer_type, layer in zip(self.layer_type_list, self.layers):
            if layer_type == LayerSymbols.MAMBA:
                return layer.mamba_state_shapes_per_request()
        return None

    def _compute_mhc_block_end_plan(self) -> List[bool]:
        """Compute per-layer block-end markers (deterministic from config)."""
        num_layers = len(self.layers)
        is_recompute_block_end: List[bool] = [False] * num_layers
        if num_layers == 0:
            return is_recompute_block_end
        mhc_recompute_layer_num = self.config.mhc_recompute_layer_num
        for l_no in range(num_layers):
            is_last_in_stack = l_no == num_layers - 1
            is_last_in_recompute_block = is_last_in_stack
            if mhc_recompute_layer_num is not None:
                is_last_in_recompute_block = is_last_in_stack or (
                    (l_no + 1) % mhc_recompute_layer_num == 0
                )
            is_recompute_block_end[l_no] = is_last_in_recompute_block
        return is_recompute_block_end

    def _build_mhc_recompute_layer_plan(
        self, use_mhc_recompute: bool
    ) -> Tuple[List[Optional[CheckpointManager]], List[bool]]:
        """Pre-build per-layer MHC recompute managers and block-end markers.

        The block-end plan is deterministic from config and cached on the
        instance; only the per-block ``CheckpointManager`` instances are
        allocated fresh per forward pass (managers are single-use). Mirrors
        the caching scheme used by ``TransformerBlock``.
        """
        num_layers = len(self.layers)
        if not use_mhc_recompute or num_layers == 0:
            return [None] * num_layers, [False] * num_layers

        if self._mhc_block_end_plan is None:
            self._mhc_block_end_plan = self._compute_mhc_block_end_plan()
        is_recompute_block_end = self._mhc_block_end_plan

        layer_managers: List[Optional[CheckpointManager]] = [None] * num_layers
        mhc_manager = CheckpointManager()
        for l_no in range(num_layers):
            layer_managers[l_no] = mhc_manager
            if is_recompute_block_end[l_no] and l_no != num_layers - 1:
                mhc_manager = CheckpointManager()
        return layer_managers, is_recompute_block_end

    @staticmethod
    def _finalize_mhc_recompute_layer(
        mhc_manager: Optional[CheckpointManager],
        hidden_states: Tensor,
        is_last_in_recompute_block: bool,
    ) -> None:
        """Finalize MHC recompute state for the current layer when a block ends."""
        if mhc_manager is not None and is_last_in_recompute_block:
            mhc_manager.discard_all_outputs_and_register_unified_recompute(hidden_states)

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
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
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
            Tensor in the common case. A 2-tuple ``(hidden_states, mhc_multistream)`` ONLY when
            ``enable_hyper_connections and post_process and mtp_num_layers > 0 and not
            is_mtp_layer`` — the extra element is the pre-contraction multi-stream tensor that
            MTP's ``_concat_embeddings`` consumes. Callers (e.g. ``HybridModel.forward``) must
            handle both; pipeline send/recv only ever transfers the contracted ``hidden_states``.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        # Skip input_expand inside MTP nested HybridStack: when mHC + MTP, the outer
        # decoder hands in already-multi-stream hidden_states via mhc_multistream
        # (see multi_token_prediction.py _concat_embeddings), so expanding again would
        # produce [s, b, n*(n*h)] instead of [s, b, n*h] and break HC mapping_proj.
        if self.config.enable_hyper_connections and self.pre_process and not self.is_mtp_layer:
            hidden_states = HyperConnectionModule.input_expand(
                hidden_states, self.config.num_residual_streams
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
            and not self.training
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
            and self.config.enable_hyper_connections
            and self.config.recompute_granularity == 'selective'
            and "mhc" in self.config.recompute_modules
        )
        mhc_layer_managers, mhc_is_last_in_recompute_block = self._build_mhc_recompute_layer_plan(
            use_mhc_recompute
        )

        with outer_fp8_context:
            for l_no, layer in enumerate(self.layers):
                # Layers have 1-indexed layer numbers attribute.
                inner_quant_context = get_inner_quant_context(self.config, layer.layer_number - 1)
                mhc_manager = mhc_layer_managers[l_no]
                if mhc_manager is not None:
                    mhc_manager.is_last_layer_in_recompute_block = mhc_is_last_in_recompute_block[
                        l_no
                    ]

                with inner_quant_context:
                    if isinstance(layer, (TransformerLayer, HyperConnectionHybridLayer)):
                        layer_kwargs = dict(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            rotary_pos_emb=rotary_pos_emb,
                            sequence_len_offset=sequence_len_offset,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                        )
                        if mhc_manager is not None and isinstance(
                            layer, HyperConnectionHybridLayer
                        ):
                            layer_kwargs["mhc_recompute_manager"] = mhc_manager
                        hidden_states, _ = layer(**layer_kwargs)
                    else:  # MambaLayer, Expert, or MLP
                        hidden_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                        )

                # The attention layer (currently a simplified transformer layer)
                # outputs a tuple of (hidden_states, context). Context is intended
                # for cross-attention, and is not needed in our model.
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

                self._finalize_mhc_recompute_layer(
                    mhc_manager=mhc_manager,
                    hidden_states=hidden_states,
                    is_last_in_recompute_block=mhc_is_last_in_recompute_block[l_no],
                )

        # When mHC + MTP, save the pre-contraction multi-stream tensor for MTP input.
        # MTP's _concat_embeddings mHC branch expects [s, b, n*h] (multi-stream), while
        # the contracted hidden_states is [s, b, h]. Mirrors transformer_block.py:948-988.
        # Only the OUTER decoder stack does this; nested MTP stacks (is_mtp_layer=True)
        # must keep returning a single Tensor so MTP's _postprocess receives the right
        # type for learned_output_contract.
        # On the final stage of a (non-MTP) stack with mHC active, capture the pre-contraction
        # multi-stream tensor for MTP's `_concat_embeddings` (only meaningful when MTP layers
        # exist, i.e. mtp_num_layers > 0), THEN contract the streams. Combining capture and
        # contraction avoids repeating the condition. Nested MTP HybridStacks (is_mtp_layer=True)
        # must NOT contract here — MTP's own `_postprocess` calls learned_output_contract +
        # final_layernorm itself, so doing it here would double-collapse the multi-stream tensor.
        mhc_multistream = None
        if self.config.enable_hyper_connections and self.post_process and not self.is_mtp_layer:
            if (self.config.mtp_num_layers or 0) > 0:
                mhc_multistream = hidden_states
            hidden_states = learned_output_contract(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
                self.config.num_residual_streams,
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
                    sharded_offsets=sharded_offsets or (),
                    tp_group=self.tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            )

        return sharded_state_dict


# Backward-compatible aliases
MambaStackSubmodules = HybridStackSubmodules
MambaStack = HybridStack
