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
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.packed_seq_params import PackedSeqParams, has_packed_seq_params_cuda_graph_kwargs
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.recompute import checkpointed_forward
from megatron.core.tensor_parallel.random import CheckpointManager
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import annotate_first_last_layer
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.hyper_connection import (
    HyperConnectionModule,
    learned_output_contract,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import (
    GraphableMegatronModule,
    MegatronModule,
    convert_module_to_dtype_except_fp32_marked,
    mark_keep_in_fp32,
)
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

    Hybrid layers already own their local residual paths. Each wrapped layer is
    treated as one function by aggregating n streams to its input, running the
    existing layer, and feeding only the layer delta back through mHC expansion.

    This wrapper nests the inner layer under inner_layer. Checkpoints cannot
    switch between mHC-enabled and ordinary HybridStacks without key migration.
    """

    def __init__(self, config: TransformerConfig, layer: MegatronModule) -> None:
        super().__init__(config=config)
        self.inner_layer = layer
        self.layer_number = layer.layer_number
        self.offload_module_in_cuda_graph = getattr(
            layer, "offload_module_in_cuda_graph", False
        )
        self.hyper_connection = HyperConnectionModule(config=config, layer_number=self.layer_number)
        if config.params_dtype is not None:
            convert_module_to_dtype_except_fp32_marked(
                self.hyper_connection, config.params_dtype
            )
        if hasattr(layer, 'tp_group'):
            self.tp_group = layer.tp_group

    def create_mcore_cudagraph_manager(self, config):
        """Leave local CUDA graph routing on the already-constructed inner layer.

        This wrapper adds a TE per-layer graph boundary. Installing a second
        local manager here would capture the wrapper regardless of the inner
        layer's local graph scope and could nest the inner manager.
        """
        return None

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Return the inner layer's static inputs with n-stream hidden width."""
        if hasattr(self.inner_layer, "get_layer_static_inputs"):
            static_inputs = self.inner_layer.get_layer_static_inputs(
                seq_length, micro_batch_size
            )
        else:
            static_inputs = super().get_layer_static_inputs(
                seq_length, micro_batch_size
            )
        hidden_states = static_inputs["hidden_states"]
        static_inputs["hidden_states"] = torch.ones(
            (
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.config.num_residual_streams * self.config.hidden_size,
            ),
            dtype=hidden_states.dtype,
            requires_grad=hidden_states.requires_grad,
            device=hidden_states.device,
        )
        return static_inputs

    def _set_te_cuda_graph_packed_seq_params_static_metadata(
        self, static_metadata, tensor_kwarg_names=None
    ):
        """Delegate the #5672 packed-sequence graph contract to the inner layer."""
        return self.inner_layer._set_te_cuda_graph_packed_seq_params_static_metadata(
            static_metadata, tensor_kwarg_names
        )

    def _get_te_cuda_graph_packed_seq_params_static_metadata(self):
        return self.inner_layer._get_te_cuda_graph_packed_seq_params_static_metadata()

    def _validate_te_cuda_graph_packed_seq_params_static_metadata(
        self, static_metadata
    ):
        return self.inner_layer._validate_te_cuda_graph_packed_seq_params_static_metadata(
            static_metadata
        )

    def _get_te_cuda_graph_packed_seq_params_tensor_kwarg_names(self):
        return self.inner_layer._get_te_cuda_graph_packed_seq_params_tensor_kwarg_names()

    def _validate_te_cuda_graph_packed_seq_params_tensor_kwargs(self, tensor_kwargs):
        return self.inner_layer._validate_te_cuda_graph_packed_seq_params_tensor_kwargs(
            tensor_kwargs
        )

    def _rebuild_te_cuda_graph_packed_seq_params(self, kwargs):
        if hasattr(self.inner_layer, "_rebuild_te_cuda_graph_packed_seq_params"):
            return self.inner_layer._rebuild_te_cuda_graph_packed_seq_params(kwargs)
        assert not has_packed_seq_params_cuda_graph_kwargs(kwargs), (
            "Wrapped non-Transformer layers do not support flattened "
            "PackedSeqParams CUDA graph inputs."
        )
        return None

    def _flatten_te_cuda_graph_packed_seq_params(self, kwargs):
        if hasattr(self.inner_layer, "_flatten_te_cuda_graph_packed_seq_params"):
            if (
                self.inner_layer._get_te_cuda_graph_packed_seq_params_static_metadata()
                is None
            ):
                assert not has_packed_seq_params_cuda_graph_kwargs(kwargs), (
                    "Wrapped Transformer layers captured without packed-sequence "
                    "metadata cannot receive flattened PackedSeqParams CUDA graph inputs."
                )
                kwargs.pop("packed_seq_params", None)
                return None
            return self.inner_layer._flatten_te_cuda_graph_packed_seq_params(kwargs)
        assert kwargs.get("packed_seq_params") is None, (
            "Wrapped non-Transformer layers do not support PackedSeqParams "
            "CUDA graph replay."
        )
        kwargs.pop("packed_seq_params", None)
        return None

    def __call__(self, *args, **kwargs):
        """Keep the non-Tensor mHC recompute manager outside TE graph inputs."""
        self._mhc_recompute_manager = kwargs.pop("mhc_recompute_manager", None)
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._mhc_recompute_manager = None

    def _inner_is_moe(self) -> bool:
        from megatron.core.transformer.moe.moe_layer import MoELayer

        return isinstance(self.inner_layer, TransformerLayer) and isinstance(
            getattr(self.inner_layer, 'mlp', None), MoELayer
        )

    def _inner_is_partial_moe_capture(self) -> bool:
        return (
            self._inner_is_moe()
            and bool(self.config.cuda_graph_modules)
            and CudaGraphModule.moe_router in self.config.cuda_graph_modules
        )

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """Capture the whole wrapper, or only its graph-safe partial-MoE prefix."""
        sample_kwarg_names = frozenset(kwargs)
        captured_sample_kwarg_names = getattr(
            self, "_te_cuda_graph_sample_kwarg_names", None
        )
        assert (
            captured_sample_kwarg_names is None
            or captured_sample_kwarg_names == sample_kwarg_names
        ), (
            "HyperConnectionHybridLayer TE CUDA graph captures must use a stable "
            "keyword-input signature."
        )
        self._te_cuda_graph_sample_kwarg_names = sample_kwarg_names
        self._rebuild_te_cuda_graph_packed_seq_params(kwargs)
        if self._inner_is_partial_moe_capture():
            hidden_states = args[0] if args else kwargs["hidden_states"]
            aggregated, h_res, h_post, residual = self.hyper_connection(
                hidden_states, return_residual=True
            )
            inner_kwargs = dict(kwargs)
            inner_kwargs.pop("hidden_states", None)
            inner_outputs = list(
                self.inner_layer._te_cuda_graph_capture(aggregated, **inner_kwargs)
            )
            return tuple(inner_outputs) + (h_post, h_res, residual)

        record_offload_events = (
            isinstance(self.inner_layer, TransformerLayer)
            and getattr(self.inner_layer, "offload_module_in_cuda_graph", False)
        )
        if record_offload_events:
            if args:
                hidden_states = self.inner_layer.off_interface.backward_record(args[0])
                args = (hidden_states,) + args[1:]
            else:
                hidden_states = self.inner_layer.off_interface.backward_record(
                    kwargs.pop("hidden_states")
                )
                kwargs["hidden_states"] = hidden_states
        hidden_states, context = self.forward(*args, **kwargs)
        outputs = [hidden_states]
        if context is not None:
            outputs.append(context)
        if record_offload_events:
            self.inner_layer.off_interface.forward_record()
        return tuple(outputs)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """Replay a whole-wrapper graph or resume an eager partial-MoE tail."""
        self._flatten_te_cuda_graph_packed_seq_params(kwargs)
        sample_kwarg_names = getattr(
            self, "_te_cuda_graph_sample_kwarg_names", None
        )
        assert sample_kwarg_names is not None, (
            "HyperConnectionHybridLayer TE CUDA graph replay requires the keyword "
            "signature recorded during capture."
        )
        for key in tuple(kwargs):
            if key != "hidden_states" and key not in sample_kwarg_names:
                kwargs.pop(key)
        delayed_offload = (
            isinstance(self.inner_layer, TransformerLayer)
            and getattr(self.inner_layer.config, "delay_offload_until_cuda_graph", False)
        )
        if delayed_offload:
            self.inner_layer.off_interface.enter_replay()
        try:
            outputs = list(super()._te_cuda_graph_replay(*args, **kwargs))
            if delayed_offload:
                self.inner_layer.off_interface.flush_delayed_groups()

            if self._inner_is_partial_moe_capture():
                residual = outputs.pop()
                h_res = outputs.pop()
                h_post = outputs.pop()
                _, mlp_output_with_bias = (
                    self.inner_layer.resume_moe_experts_after_partial_cudagraph(
                        outputs
                    )
                )
                hidden_states = self.hyper_connection.fused_h_res_h_post_bda(
                    h_res,
                    residual,
                    h_post,
                    mlp_output_with_bias,
                    dropout_prob=self.inner_layer.hidden_dropout,
                    training=self.training,
                    fused=self.inner_layer.config.bias_dropout_fusion,
                    manager=None,
                )
                if (
                    self.config.fp32_residual_connection
                    and self.config.params_dtype is not None
                    and hidden_states.dtype != self.config.params_dtype
                ):
                    hidden_states = hidden_states.to(self.config.params_dtype)
                return hidden_states, None

            return outputs[0], None
        finally:
            if delayed_offload:
                self.inner_layer.off_interface.exit_replay()

    def _get_te_cuda_graph_replay_args(self, *args, **kwargs):
        """Use the wrapped TransformerLayer's replay-argument normalization."""
        if not isinstance(getattr(self, "inner_layer", None), TransformerLayer):
            return super()._get_te_cuda_graph_replay_args(*args, **kwargs)

        missing = object()
        previous_microbatch = getattr(self.inner_layer, "current_microbatch", missing)
        self.inner_layer.current_microbatch = getattr(self, "current_microbatch", 0)
        try:
            return self.inner_layer._get_te_cuda_graph_replay_args(*args, **kwargs)
        finally:
            if previous_microbatch is missing:
                del self.inner_layer.current_microbatch
            else:
                self.inner_layer.current_microbatch = previous_microbatch

    def _get_submodules_under_cudagraphs(self):
        if self._inner_is_partial_moe_capture():
            return [
                self.hyper_connection
            ] + self.inner_layer._get_submodules_under_cudagraphs()
        return super()._get_submodules_under_cudagraphs()

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
        input_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
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
                input_ids=input_ids,
                _called_from_hybrid_mhc_wrapper=True,
            )
        else:
            # Mamba-like layers only consume the common HybridStack arguments.
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

    def _call_inner_transformer_layer_without_local_bda(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext],
        rotary_pos_emb: Optional[Tensor],
        sequence_len_offset: Optional[Tensor],
        packed_seq_params: Optional[PackedSeqParams],
        padding_mask: Optional[Tensor],
        input_ids: Optional[Tensor] = None,
    ) -> Optional[Tuple[Tuple[Tensor, Optional[Tensor]], Optional[Tensor], float, bool]]:
        """Return a raw branch output for split Hybrid TransformerLayer instances.

        Hybrid layers are normally attention-only or MLP/MoE-only. For those
        layers, bypass the inner layer's local residual/BDA and let the mHC BDA
        own that operation directly.
        """
        if not isinstance(self.inner_layer, TransformerLayer):
            return None

        layer = self.inner_layer
        if InferenceMode.is_active() and layer.config.inference_fuse_tp_communication:
            return None

        has_attention = not isinstance(layer.self_attention, IdentityOp)
        has_cross_attention = not isinstance(layer.cross_attention, IdentityOp)
        has_mlp = not isinstance(layer.mlp, IdentityOp)

        if has_cross_attention or has_attention == has_mlp:
            return None

        if has_attention:
            output_with_bias, attn_norm_manager, residual = (
                layer._forward_self_attention_output_with_bias(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )
            )
            output_with_bias = layer._group_offload_output_with_bias(
                output_with_bias, attn_norm_manager, forced_released_tensors=[residual]
            )
            return output_with_bias, None, layer.hidden_dropout, layer.config.bias_dropout_fusion

        output_with_bias, residual = layer._forward_mlp_output_with_bias(
            hidden_states,
            inference_context=inference_context,
            padding_mask=padding_mask,
            packed_seq_params=packed_seq_params,
            input_ids=input_ids,
        )
        if layer.mlp_norm_manager is not None:
            output_with_bias = layer._group_offload_output_with_bias(
                output_with_bias, layer.mlp_norm_manager, forced_released_tensors=[residual]
            )
            layer.mlp_norm_manager = None
        return output_with_bias, None, layer.hidden_dropout, layer.config.bias_dropout_fusion

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
        input_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the wrapped hybrid layer through one layer-boundary mHC update."""
        if mhc_recompute_manager is None:
            mhc_recompute_manager = getattr(self, '_mhc_recompute_manager', None)
        aggregated, h_res, h_post, residual = self.hyper_connection(
            hidden_states,
            mhc_recompute_manager=mhc_recompute_manager,
            return_residual=True,
        )
        fast_path_result = self._call_inner_transformer_layer_without_local_bda(
            aggregated,
            attention_mask,
            inference_context,
            rotary_pos_emb,
            sequence_len_offset,
            packed_seq_params,
            padding_mask,
            input_ids,
        )

        if fast_path_result is None:
            layer_output, context = self._call_inner_layer(
                aggregated,
                attention_mask,
                inference_context,
                rotary_pos_emb,
                sequence_len_offset,
                packed_seq_params,
                padding_mask,
                input_ids,
            )
            if self.config.fp32_residual_connection and aggregated.dtype != layer_output.dtype:
                aggregated = aggregated.to(layer_output.dtype)
            layer_output_with_bias = (layer_output - aggregated, None)
            dropout_prob = 0.0
            bias_dropout_fusion = False
        else:
            layer_output_with_bias, context, dropout_prob, bias_dropout_fusion = fast_path_result

        layer_output = layer_output_with_bias[0]
        if layer_output.shape != aggregated.shape:
            raise RuntimeError(
                "HyperConnectionHybridLayer requires wrapped branches to preserve "
                f"hidden-state shape. Got {tuple(layer_output.shape)} from wrapped branch "
                f"vs {tuple(aggregated.shape)} input."
            )

        is_last_in_recompute_block = bool(
            mhc_recompute_manager is not None
            and getattr(mhc_recompute_manager, "is_last_layer_in_recompute_block", False)
        )
        mhc_bda_manager = None if is_last_in_recompute_block else mhc_recompute_manager
        hidden_states = self.hyper_connection.fused_h_res_h_post_bda(
            h_res,
            residual,
            h_post,
            layer_output_with_bias,
            dropout_prob=dropout_prob,
            training=self.training,
            fused=bias_dropout_fusion,
            manager=mhc_bda_manager,
        )
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
        mtp_layer_number (int, optional): enclosing MTP depth for nested MoE metrics.
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
        mtp_layer_number: Optional[int] = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_layer_norm = post_layer_norm
        self.post_process = post_process
        self.is_mtp_layer = is_mtp_layer
        self.mtp_layer_number = mtp_layer_number

        assert pg_collection is not None, "pg_collection must be provided for HybridStack"

        self.pp_group = pg_collection.pp
        self.tp_group = pg_collection.tp

        # Required for pipeline parallel schedules
        self.input_tensor = None
        self.pg_collection = pg_collection

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
                        name=(name + f".layers.{i}") if name is not None else None,
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
                        name=(name + f".layers.{i}") if name is not None else None,
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
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.CSA:
                    layer = build_module(
                        submodules.csa_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.HCA:
                    layer = build_module(
                        submodules.hca_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.WINDOW:
                    layer = build_module(
                        submodules.window_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.MLP:
                    layer = build_module(
                        submodules.mlp_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.MOE:
                    layer = build_module(
                        submodules.moe_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.GDN:
                    layer = build_module(
                        submodules.gdn_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        # Set to False as we do not want to change offset.
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                else:
                    raise ValueError("unexpected layer_type")
            if self.is_mtp_layer and self.mtp_layer_number is not None:
                self._set_mtp_layer_number_for_moe_metrics(
                    layer, self.mtp_layer_number
                )
            if self.config.enable_hyper_connections:
                layer = HyperConnectionHybridLayer(config=self.config, layer=layer)
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

        if (
            self.config.enable_hyper_connections
            and self.post_process
            and not self.is_mtp_layer
        ):
            hc_mult = self.config.num_residual_streams
            hc_dim = self.config.hidden_size * hc_mult
            self.hc_head_fn = mark_keep_in_fp32(nn.Parameter(torch.randn(hc_mult, hc_dim)))
            self.hc_head_base = mark_keep_in_fp32(nn.Parameter(torch.zeros(hc_mult)))
            self.hc_head_scale = mark_keep_in_fp32(nn.Parameter(torch.ones(1)))
            nn.init.xavier_uniform_(self.hc_head_fn)
            if self.config.sequence_parallel:
                setattr(self.hc_head_fn, 'sequence_parallel', True)
                setattr(self.hc_head_base, 'sequence_parallel', True)
                setattr(self.hc_head_scale, 'sequence_parallel', True)

    @staticmethod
    def _set_mtp_layer_number_for_moe_metrics(
        layer: torch.nn.Module, mtp_layer_number: int
    ) -> None:
        """Propagate the enclosing MTP depth to nested MTP MoE routers."""
        for module in layer.modules():
            router = getattr(module, "router", None)
            if router is not None and getattr(router, "is_mtp_layer", False):
                router.mtp_layer_number = mtp_layer_number

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
    ) -> Tuple[List[Optional[CheckpointManager]], List[bool]]:
        """Build single-use recompute managers for this forward pass."""
        num_layers = len(self.layers)
        if not use_mhc_recompute or num_layers == 0:
            return [None] * num_layers, [False] * num_layers

        if self._mhc_block_end_plan is None:
            self._mhc_block_end_plan = self._compute_mhc_block_end_plan()
        block_ends = self._mhc_block_end_plan

        layer_managers: List[Optional[CheckpointManager]] = [None] * num_layers
        manager = CheckpointManager()
        for layer_idx in range(num_layers):
            layer_managers[layer_idx] = manager
            if block_ends[layer_idx] and layer_idx != num_layers - 1:
                manager = CheckpointManager()
        return layer_managers, block_ends

    @staticmethod
    def _finalize_mhc_recompute_layer(
        manager: Optional[CheckpointManager], hidden_states: Tensor, is_block_end: bool
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
        input_ids: Optional[Tensor] = None,
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
            input_ids (Tensor, optional): Token IDs forwarded to hash-routed
                TransformerLayer instances. Defaults to None.
        Returns:
            Tensor: the output tensor.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if (
            self.config.enable_hyper_connections
            and self.pre_process
            and not self.is_mtp_layer
        ):
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
            and self.config.enable_hyper_connections
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
                    input_ids=input_ids,
                    use_inner_quantization_context=(use_inner_fp8_context or use_fp4_context),
                )
            else:
                for layer_idx, layer in enumerate(self.layers):
                    # Layers have 1-indexed layer numbers attribute.
                    inner_quant_context = get_inner_quant_context(
                        self.config, layer.layer_number - 1
                    )
                    mhc_manager = mhc_layer_managers[layer_idx]
                    if mhc_manager is not None:
                        mhc_manager.is_last_layer_in_recompute_block = mhc_block_ends[layer_idx]

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
                            if input_ids is not None:
                                layer_kwargs['input_ids'] = input_ids
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
                        manager=mhc_manager,
                        hidden_states=hidden_states,
                        is_block_end=mhc_block_ends[layer_idx],
                    )

        mhc_multistream = None
        if (
            self.config.enable_hyper_connections
            and self.post_process
            and not self.is_mtp_layer
        ):
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
                    sharded_offsets=sharded_offsets,
                    tp_group=self.tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            )

        return sharded_state_dict


# Backward-compatible aliases
MambaStackSubmodules = HybridStackSubmodules
MambaStack = HybridStack
