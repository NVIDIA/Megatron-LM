# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron Module."""

from copy import copy as shallow_copy
from functools import partial
from typing import Optional, Tuple

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)

_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)
_TE_CUDA_GRAPH_ROUTE_MICROBATCH_ATTR = "_te_cuda_graph_route_microbatch_id"
_TE_CUDA_GRAPH_ROUTE_SLOT_ATTR = "_te_cuda_graph_route_slot"


def param_is_not_shared(param):  # pylint: disable=missing-function-docstring
    return not hasattr(param, 'shared') or not param.shared


class MegatronModule(torch.nn.Module):
    """Base Megatron module inhertied by all Models.

    Megatron specific extensions of torch Module with support
    for pipelining

    Args:
        config (TransformerConfig): Transformer config
    """

    # def __init__(self, config: TransformerConfig, share_word_embeddings=True):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    @staticmethod
    def _validate_route_metadata_pair(layout_i32, route_i64, *, name):
        """Validate one packed graph-route metadata owner pair."""
        if not isinstance(layout_i32, torch.Tensor) or not isinstance(route_i64, torch.Tensor):
            raise TypeError(f"{name} must contain two torch.Tensor owners")
        if layout_i32.dtype != torch.int32 or route_i64.dtype != torch.int64:
            raise TypeError(
                f"{name} must be (int32 layout, int64 route), got "
                f"({layout_i32.dtype}, {route_i64.dtype})"
            )
        if layout_i32.dim() != 1 or route_i64.dim() != 1:
            raise ValueError(
                f"{name} owners must be flat, got shapes "
                f"{tuple(layout_i32.shape)} and {tuple(route_i64.shape)}"
            )
        if not layout_i32.is_contiguous() or not route_i64.is_contiguous():
            raise ValueError(f"{name} owners must be contiguous")
        if layout_i32.device != route_i64.device:
            raise ValueError(
                f"{name} owners must be on one device, got "
                f"{layout_i32.device} and {route_i64.device}"
            )
        if layout_i32.numel() == 0 or route_i64.numel() == 0:
            raise ValueError(f"{name} owners must be non-empty")
        layout_range = (
            layout_i32.data_ptr(),
            layout_i32.data_ptr() + layout_i32.numel() * layout_i32.element_size(),
        )
        route_range = (
            route_i64.data_ptr(),
            route_i64.data_ptr() + route_i64.numel() * route_i64.element_size(),
        )
        if max(layout_range[0], route_range[0]) < min(layout_range[1], route_range[1]):
            raise ValueError(f"{name} owners must not overlap in storage")

    def set_te_cuda_graph_route_metadata_arenas(self, arenas, *, logical_route_numel, cp_rank=None):
        """Retain one fixed-address DSA route-metadata pair per graph slot."""
        arenas = tuple(tuple(pair) for pair in arenas)
        if not arenas:
            raise ValueError("TE CUDA Graph route-metadata arenas must contain at least one slot")
        if not isinstance(logical_route_numel, int) or logical_route_numel <= 0:
            raise ValueError("TE CUDA Graph logical route length must be a positive integer")
        cp_size = self.config.context_parallel_size
        cp_group = getattr(getattr(self, "pg_collection", None), "cp", None)
        if cp_group is None or cp_group.size() != cp_size:
            raise RuntimeError(
                "TE CUDA Graph route-metadata arena requires the owning stack's explicit CP "
                f"group to have size {cp_size}"
            )
        owner_cp_rank = cp_group.rank()
        if cp_rank is None:
            cp_rank = owner_cp_rank
        elif cp_rank != owner_cp_rank:
            raise ValueError(
                "TE CUDA Graph route-metadata arena rank must match the owning stack's CP "
                f"group: owner={owner_cp_rank}, requested={cp_rank}"
            )
        if not isinstance(cp_rank, int) or isinstance(cp_rank, bool) or not 0 <= cp_rank < cp_size:
            raise ValueError(
                f"TE CUDA Graph CP-local rank must be in [0, {cp_size}), got {cp_rank}"
            )

        reference_layout_schema = None
        schemas = []
        ptrs = []
        seen_ptrs = set()
        seen_ranges = []
        seen_route_lengths = set()
        for slot, pair in enumerate(arenas):
            if len(pair) != 2:
                raise ValueError(
                    f"TE CUDA Graph route-metadata slot {slot} must contain exactly two owners"
                )
            layout_i32, route_i64 = pair
            self._validate_route_metadata_pair(
                layout_i32, route_i64, name=f"TE CUDA Graph route-metadata slot {slot}"
            )
            layout_schema = (tuple(layout_i32.shape), layout_i32.device, route_i64.device)
            if reference_layout_schema is None:
                reference_layout_schema = layout_schema
            elif layout_schema != reference_layout_schema:
                raise ValueError(
                    "TE CUDA Graph route-metadata slots must have one fixed layout schema; "
                    f"slot 0 has {reference_layout_schema}, slot {slot} has {layout_schema}"
                )
            if route_i64.numel() <= logical_route_numel:
                raise ValueError(
                    f"TE CUDA Graph route-metadata slot {slot} must have a positive signature "
                    f"suffix after {logical_route_numel} logical entries"
                )
            pair_ptrs = (layout_i32.data_ptr(), route_i64.data_ptr())
            if any(ptr in seen_ptrs for ptr in pair_ptrs):
                raise ValueError(
                    "TE CUDA Graph route-metadata owners must not alias across live slots"
                )
            pair_ranges = (
                (
                    layout_i32.data_ptr(),
                    layout_i32.data_ptr() + layout_i32.numel() * layout_i32.element_size(),
                ),
                (
                    route_i64.data_ptr(),
                    route_i64.data_ptr() + route_i64.numel() * route_i64.element_size(),
                ),
            )
            if any(
                max(current[0], previous[0]) < min(current[1], previous[1])
                for current in pair_ranges
                for previous in seen_ranges
            ):
                raise ValueError(
                    "TE CUDA Graph route-metadata owners must not overlap across live slots"
                )
            if route_i64.numel() in seen_route_lengths:
                raise ValueError(
                    "TE CUDA Graph route-metadata route lengths must be unique across live slots"
                )
            seen_route_lengths.add(route_i64.numel())
            seen_ptrs.update(pair_ptrs)
            seen_ranges.extend(pair_ranges)
            ptrs.append(pair_ptrs)
            schemas.append(
                (
                    tuple(layout_i32.shape),
                    tuple(route_i64.shape),
                    layout_i32.device,
                    route_i64.device,
                )
            )

        self._te_cuda_graph_route_metadata_arenas = arenas
        self._te_cuda_graph_route_metadata_arena_ptrs = tuple(ptrs)
        self._te_cuda_graph_route_metadata_arena_schemas = tuple(schemas)
        self._te_cuda_graph_logical_route_numel = logical_route_numel
        self._te_cuda_graph_cp_rank = cp_rank

    def get_te_cuda_graph_route_metadata_arena(self, microbatch_idx=None):
        """Return and validate the fixed-address pair selected by graph-slot modulo."""
        arenas = getattr(self, "_te_cuda_graph_route_metadata_arenas", ())
        arena_ptrs = getattr(self, "_te_cuda_graph_route_metadata_arena_ptrs", ())
        if not arenas:
            raise RuntimeError("TE CUDA Graph route-metadata arenas have not been attached")
        if len(arenas) != len(arena_ptrs):
            raise RuntimeError("TE CUDA Graph route-metadata arena slot bookkeeping is malformed")

        if microbatch_idx is None:
            microbatch_idx = getattr(self, "current_microbatch", 0)
        if (
            not isinstance(microbatch_idx, int)
            or isinstance(microbatch_idx, bool)
            or microbatch_idx < 0
        ):
            raise ValueError(
                f"TE CUDA Graph route-metadata microbatch index must be non-negative, "
                f"got {microbatch_idx!r}"
            )
        slot = microbatch_idx % len(arenas)
        layout_i32, route_i64 = arenas[slot]
        expected_layout_ptr, expected_route_ptr = arena_ptrs[slot]
        actual_ptrs = (layout_i32.data_ptr(), route_i64.data_ptr())
        if actual_ptrs != (expected_layout_ptr, expected_route_ptr):
            raise RuntimeError(
                f"TE CUDA Graph route-metadata slot {slot} changed address: expected "
                f"({expected_layout_ptr}, {expected_route_ptr}), got {actual_ptrs}"
            )
        self._validate_route_metadata_pair(
            layout_i32, route_i64, name=f"TE CUDA Graph route-metadata slot {slot}"
        )
        actual_schema = (
            tuple(layout_i32.shape),
            tuple(route_i64.shape),
            layout_i32.device,
            route_i64.device,
        )
        schemas = getattr(self, "_te_cuda_graph_route_metadata_arena_schemas", ())
        if len(schemas) != len(arenas) or actual_schema != schemas[slot]:
            raise RuntimeError(
                f"TE CUDA Graph route-metadata slot {slot} changed schema: got {actual_schema}"
            )
        return layout_i32, route_i64

    def clear_te_cuda_graph_route_metadata_arenas(self):
        """Release fixed-address route owners after the corresponding graphs are reset."""
        self._te_cuda_graph_route_metadata_arenas = ()
        self._te_cuda_graph_route_metadata_arena_ptrs = ()
        self._te_cuda_graph_route_metadata_arena_schemas = ()
        self._te_cuda_graph_logical_route_numel = None
        self._te_cuda_graph_cp_rank = None

    def _stage_te_cuda_graph_route_metadata(self, packed_seq_params):
        """Copy two eager owners into this chunk's selected graph arena exactly once."""
        if not getattr(self, "_te_cuda_graph_route_metadata_arenas", ()):
            return packed_seq_params
        if not getattr(self.config, "dsa_cp_balance_indexer_graph_dynamic_packs", False):
            raise RuntimeError(
                "TE CUDA Graph route-metadata arenas are attached while the graph-dynamic "
                "balanced-CP route mode is disabled"
            )
        if packed_seq_params is None:
            raise RuntimeError(
                "TE CUDA Graph replay with a route-metadata arena requires PackedSeqParams"
            )

        from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

        source_layout, source_route = cp_balanced_indexer.get_graph_dynamic_plan_buffers(
            packed_seq_params
        )
        source_plan = cp_balanced_indexer.get_graph_dynamic_plan(packed_seq_params)
        expected_cp_rank = getattr(self, "_te_cuda_graph_cp_rank", None)
        expected_cp_size = self.config.context_parallel_size
        expected_l_local = self.config.max_seqlen_per_dp_cp_rank
        expected_contract = (expected_cp_size, expected_cp_rank, expected_l_local, 0)
        actual_contract = (
            None
            if source_plan is None
            else (
                source_plan.get("cp_size"),
                source_plan.get("cp_rank"),
                source_plan.get("l_local"),
                source_plan.get("route_padding", 0),
            )
        )
        if actual_contract != expected_contract:
            raise ValueError(
                "Runtime graph-route metadata must be an exact rank-local source plan: "
                f"expected=(cp={expected_cp_size}, rank={expected_cp_rank}, "
                f"L={expected_l_local}, padding=0), got={actual_contract}"
            )
        self._validate_route_metadata_pair(
            source_layout, source_route, name="runtime graph-route metadata"
        )
        arena_layout, arena_route = self.get_te_cuda_graph_route_metadata_arena()
        logical_route_numel = getattr(self, "_te_cuda_graph_logical_route_numel", None)
        if not isinstance(logical_route_numel, int) or logical_route_numel <= 0:
            raise RuntimeError("TE CUDA Graph logical route length bookkeeping is malformed")
        if (
            source_layout.shape != arena_layout.shape
            or source_route.numel() != logical_route_numel
            or arena_route.numel() <= logical_route_numel
            or source_layout.device != arena_layout.device
            or source_route.device != arena_route.device
        ):
            raise ValueError(
                "Runtime graph-route metadata does not match the captured arena schema: "
                f"source=({tuple(source_layout.shape)}, {tuple(source_route.shape)}, "
                f"{source_layout.device}, {source_route.device}), "
                f"arena=({tuple(arena_layout.shape)}, {tuple(arena_route.shape)}, "
                f"{arena_layout.device}, {arena_route.device})"
            )

        # These are the only eager-to-static metadata copies for the whole stack forward.
        arena_layout.copy_(source_layout)
        arena_route.narrow(0, 0, logical_route_numel).copy_(source_route)

        staged_params = shallow_copy(packed_seq_params)
        microbatch_idx = getattr(self, "current_microbatch", 0)
        if (
            not isinstance(microbatch_idx, int)
            or isinstance(microbatch_idx, bool)
            or microbatch_idx < 0
        ):
            raise ValueError(
                "TE CUDA Graph route-metadata microbatch index must be a non-negative integer, "
                f"got {microbatch_idx!r}"
            )
        setattr(staged_params, _TE_CUDA_GRAPH_ROUTE_MICROBATCH_ATTR, microbatch_idx)
        setattr(
            staged_params,
            _TE_CUDA_GRAPH_ROUTE_SLOT_ATTR,
            microbatch_idx % len(self._te_cuda_graph_route_metadata_arenas),
        )
        cp_balanced_indexer.attach_graph_dynamic_plan_buffers(
            staged_params,
            arena_layout,
            arena_route,
            self.config.context_parallel_size,
            self.config.max_seqlen_per_dp_cp_rank,
            route_padding=arena_route.numel() - logical_route_numel,
            cp_rank=expected_cp_rank,
        )
        staged_layout, staged_route = cp_balanced_indexer.get_graph_dynamic_plan_buffers(
            staged_params
        )
        if (
            staged_layout.data_ptr() != arena_layout.data_ptr()
            or staged_route.data_ptr() != arena_route.data_ptr()
        ):
            raise RuntimeError(
                "Attaching graph-route metadata did not preserve the captured arena owners"
            )
        original_layout, original_route = cp_balanced_indexer.get_graph_dynamic_plan_buffers(
            packed_seq_params
        )
        if (
            original_layout.data_ptr() != source_layout.data_ptr()
            or original_route.data_ptr() != source_route.data_ptr()
        ):
            raise RuntimeError("Staging graph-route metadata mutated the caller's PackedSeqParams")
        return staged_params

    def state_dict_for_save_checkpoint(self, prefix: str = '', keep_vars: bool = False):
        """Override state dict for saving checkpoints Use this function to override the
        state dict for saving checkpoints.

        Args:
            prefix (str, optional): _description_. Defaults to ''.
            keep_vars (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        return self.state_dict(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Default implementation for sharded state dict for distributed checkpointing.

        General definition of sharded_state_dict simply calls `sharded_state_dict_default`
        (which call sharded_state_dict method if possible or a default implementation otherwise)
        recursively on all submodules.

        Args:
            prefix (str): prefix for the state dict keys
            sharded_offsets (Tuple[Tuple[int, int, int]], optional): sharding already
                applied (e.g. PP related) by sup-modules. Passed along to ShardedTensor
            metadata (dict, optional): metadata passed recursively to sharded_state_dict methods

        Returns:
            dict: dictionary of state dict keys mapped to ShardedTensors
        """
        sharded_state_dict = {}
        # Save parameters
        self._save_to_state_dict(sharded_state_dict, '', keep_vars=True)
        if not hasattr(self, 'tp_group'):
            # some model interface hasn't updated for m4, fallback needed
            tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            tp_group = self.tp_group
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            sharded_offsets=sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=metadata['dp_cp_group'],
        )
        # Recurse into submodules
        for name, module in self.named_children():
            sharded_state_dict.update(
                sharded_state_dict_default(
                    module, f'{prefix}{name}.', sharded_offsets, metadata, tp_group=tp_group
                )
            )
        return sharded_state_dict

    def set_is_first_microbatch(self):
        """Sets the is_first_microbatch flag if it exists and config.fp8==True.
        When this flag is set, TE modules will update their fp8 parameter cache.
        If kitchen is being used, kitchen controls quantization level.
        A quant_recipe (e.g. from --te-precision-config-file) also enables the flag.
        """
        if (
            self.config.fp8 is not None
            or self.config.fp4 is not None
            or getattr(self.config, 'use_kitchen', False)
            or getattr(self.config, 'quant_recipe', None) is not None
        ):
            if not hasattr(self, "modules_with_is_first_microbatch"):
                self.modules_with_is_first_microbatch = []
                for m in self.modules():
                    if hasattr(m, "is_first_microbatch"):
                        self.modules_with_is_first_microbatch.append(m)
            for m in self.modules_with_is_first_microbatch:
                m.is_first_microbatch = True

    def set_symmetric_ar(self, set_to: Optional[str] = None) -> None:
        """
        Set symmetric all-reduce functionality across all eligible modules.

        This method traverses the model's module hierarchy to find all modules
        with the 'symmetric_ar_type' attribute, caches them, and then sets their
        '_symmetric_ar_cache' attribute to the specified value to enable or
        disable symmetric all-reduce operations.

        Args:
            set_to (Any, optional): Value to set for the 'symmetric_ar_type' to.
            Allowed choices ['two_shot', "one_shot", "multimem_all_reduce", None]
        """
        assert set_to in ['two_shot', "one_shot", "multimem_all_reduce", None]

        # Recursive function to find all modules with our target attributes
        def create_ar_cache(module):
            # Check if this module has any of our target attributes
            if hasattr(module, "symmetric_ar_type"):
                self._symmetric_ar_cache.append(module)

            # Check all children modules recursively
            for child in module._modules.values():
                if child is not None:
                    create_ar_cache(child)

        if not hasattr(self, "_symmetric_ar_cache"):
            self._symmetric_ar_cache = []
            create_ar_cache(self)

        for module in self._symmetric_ar_cache:
            module._symmetric_ar_cache = set_to


class GraphableMegatronModule(MegatronModule):
    """Megatron module that can be used to capture and replay CUDA graphs.
    Now only TransformerLayer and MambaLayer are graphable.

    Args:
        config (TransformerConfig): Transformer config
    """

    def __init__(self, config: TransformerConfig, vp_stage: Optional[int] = None):
        super().__init__(config)

        assert isinstance(config, TransformerConfig), "config must be a TransformerConfig"

        # Enable cuda graphs.
        if config.cuda_graph_impl == "local":
            if hasattr(self, "create_mcore_cudagraph_manager"):
                self.create_mcore_cudagraph_manager(config)
            else:
                from megatron.core.transformer.cuda_graphs import CudaGraphManager

                self.cudagraph_manager = CudaGraphManager(config)
        elif config.cuda_graph_impl == "transformer_engine":
            # List to store CUDA graphs. A list of `N` CUDA graphs for this layer where N is
            # the number of microbatches. Multiple CUDA graphs per layer is required to support
            # pipelining which requires running FWD graph of multiple microbatches before BWD
            # graph. To enable CUDA graph, this list should be populated in the model training
            # script with the graphs returned by make_graphed_callables API before the first
            # training step.
            self.cuda_graphs = []
            # Positional hidden-state inputs used as TE's fixed CUDA Graph input
            # surfaces, indexed exactly like ``cuda_graphs``.  Most layers do not
            # need to retain these handles.  They are exposed for eager producers
            # whose recompute must restore bytes at the address captured by a
            # downstream graph (for example, mHC aggregation feeding attention).
            self._te_cuda_graph_static_hidden_inputs = ()
            self._te_cuda_graph_static_hidden_input_ptrs = ()
            # List to store forward pre-hooks. Forward pre-hooks are not captured into CUDA
            # graphs. Those hooks and args are collected in this list and should be manually
            # triggered before CUDA Graph running. This is required to ensure the correct param
            # all-gather overlap with forward compute.
            self.cuda_graph_manual_hooks = []
            # _CudaGraphBackwardDWWrapper object used to manage the wgrad backward computation.
            # The `backward_dw` func api is the same as `TransformerLayerNode.backward_dw` and
            # calls wgrad computation in attention module (contains attn and shared expert)
            # according to CUDA graph scope.
            self.cuda_graph_backward_dw_wrapper = None

    def _set_te_cuda_graph_route_replay_state(self, packed_seq_params):
        """Pin replay to the graph slot retained by a checkpointed packed invocation."""
        microbatch_idx = getattr(packed_seq_params, _TE_CUDA_GRAPH_ROUTE_MICROBATCH_ATTR, None)
        slot = getattr(packed_seq_params, _TE_CUDA_GRAPH_ROUTE_SLOT_ATTR, None)
        if microbatch_idx is None and slot is None:
            self._te_cuda_graph_route_replay_state = None
            return
        if (
            not getattr(self.config, "dsa_cp_balance_indexer_graph_dynamic_packs", False)
            or not isinstance(microbatch_idx, int)
            or isinstance(microbatch_idx, bool)
            or microbatch_idx < 0
            or not self.cuda_graphs
            or slot != microbatch_idx % len(self.cuda_graphs)
        ):
            raise RuntimeError(
                "TE CUDA Graph replay received malformed staged route-slot metadata: "
                f"microbatch={microbatch_idx!r}, slot={slot!r}"
            )
        self._te_cuda_graph_route_replay_state = (microbatch_idx, slot)

    def set_te_cuda_graph_static_hidden_inputs(self, inputs):
        """Retain TE's fixed hidden-state input surface for each graph slot.

        ``TECudaGraphHelper`` calls this only after ``make_graphed_callables``
        returns because TE may rebind sample inputs while optimizing graph-buffer
        reuse.  Different slots are allowed to alias when their schedule
        lifetimes do not overlap.
        """
        inputs = tuple(inputs)
        if len(inputs) != len(self.cuda_graphs):
            raise ValueError(
                "TE CUDA Graph static-input count must match graph count: "
                f"got {len(inputs)} inputs and {len(self.cuda_graphs)} graphs"
            )
        if not all(isinstance(tensor, torch.Tensor) and tensor.is_cuda for tensor in inputs):
            raise TypeError("TE CUDA Graph static hidden inputs must be CUDA tensors")

        self._te_cuda_graph_static_hidden_inputs = inputs
        self._te_cuda_graph_static_hidden_input_ptrs = tuple(tensor.data_ptr() for tensor in inputs)

    def get_te_cuda_graph_static_hidden_input(self, microbatch_idx=None):
        """Return the fixed hidden-state input for a TE CUDA Graph slot."""
        if not self._te_cuda_graph_static_hidden_inputs:
            raise RuntimeError("TE CUDA Graph static hidden inputs have not been attached")

        if microbatch_idx is None:
            microbatch_idx = getattr(self, 'current_microbatch', 0)
        graph_index = microbatch_idx % len(self._te_cuda_graph_static_hidden_inputs)
        tensor = self._te_cuda_graph_static_hidden_inputs[graph_index]
        expected_ptr = self._te_cuda_graph_static_hidden_input_ptrs[graph_index]
        if tensor.data_ptr() != expected_ptr:
            raise RuntimeError(
                f"TE CUDA Graph static hidden input {graph_index} changed address: "
                f"expected {expected_ptr}, got {tensor.data_ptr()}"
            )
        return tensor

    def clear_te_cuda_graph_static_hidden_inputs(self):
        """Release retained TE static-input handles when graphs are deleted."""
        self._te_cuda_graph_static_hidden_inputs = ()
        self._te_cuda_graph_static_hidden_input_ptrs = ()

    def init_backward_dw_wrapper(self):
        """Initialize the backward_dw_wrapper."""
        from megatron.core.models.gpt.fine_grained_callables import _BackwardDWWrapper

        config = getattr(self, 'config', None)
        assert config is not None, (
            "TransformerLayer must be initialized before calling " "`init_backward_dw_wrapper`."
        )
        self.backward_dw_wrapper = _BackwardDWWrapper(self)

    def set_te_cuda_graph_backward_dw_wrapper(self):
        """Replace the backward_dw callable with dw cuda graph."""
        assert (
            self.backward_dw_wrapper is not None
        ), "`backward_dw_wrapper` must be set when cuda graphs are enabled for ep overlap."
        self.backward_dw_wrapper.set_graphed_backward_dw_callable(
            partial(self._te_cuda_graph_backward_dw_graph, self.current_microbatch)
        )

    def _te_cuda_graph_backward_dw_graph(self, microbatch_idx):
        """
        CUDA Graph backward weight gradient computation for current layer.
        """
        cg_index = microbatch_idx % len(self.cuda_graphs)
        if not hasattr(self.cuda_graphs[cg_index], 'backward_dw'):
            return
        self.cuda_graphs[cg_index].backward_dw()

    def _is_thd_cuda_graph(self):
        """Check if THD format with CUDA Graph is being used."""
        return (
            getattr(self.config, 'sequence_packing_scheduler', None) is not None
            and self.config.cuda_graph_impl != "none"
        )

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """
        Get the static inputs for the layer.
        We assume that the module has one hidden_states input, whose shape is inferred
        from the seq_length, micro_batch_size, and parallel config.
        Override this method if the module has other inputs.

        For THD + CUDA Graph, hidden_states uses the padded max sequence length with
        micro_batch_size=1 (packed sequence format).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the static inputs for the layer.
        """
        # Calculate data shape related values.
        context_parallel_size = self.config.context_parallel_size
        sequence_parallel = self.config.sequence_parallel
        tensor_model_parallel_size = self.config.tensor_model_parallel_size

        if self._is_thd_cuda_graph():
            # THD + CUDA Graph: pre-padded packed-sequence buffer, batch dim = 1.
            assert (
                self.config.max_seqlen_per_dp_cp_rank is not None
            ), "max_seqlen_per_dp_cp_rank must be set when using THD format with CUDA Graph."
            slen_full = self.config.max_seqlen_per_dp_cp_rank
            batch = 1
        else:
            # SBHD path: per-rank seq is split by CP and (optionally) by TP under SP.
            slen_full = seq_length // context_parallel_size
            batch = micro_batch_size
        slen_per_cptp = slen_full // tensor_model_parallel_size if sequence_parallel else slen_full

        # Static input dtype must match the runtime activation dtype that flows
        # through the captured graph.
        if self.config.bf16:
            dtype = torch.bfloat16
        elif self.config.fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32

        return {
            "hidden_states": torch.ones(
                (slen_per_cptp, batch, self.config.hidden_size),
                dtype=dtype,
                requires_grad=True,
                device=torch.cuda.current_device(),
            )
        }

    def setup_manual_hooks(self, make_hook_func):
        """
        Set CUDA Graph manual hooks for the submodules that contain direct parameters and are
        covered by cudagraphs.
        """
        self.cuda_graph_manual_hooks = []

        # Select the modules who contain direct parameters and are covered by cudagraphs.
        # Add these modules to the `cuda_graph_manual_hooks` because their hooks will not
        # be automatically triggered when they go through the CUDA Graph path.
        param_modules = {}
        for submodule in self._get_submodules_under_cudagraphs():
            for module in submodule.modules():
                if next(module.parameters(recurse=False), None) is not None:
                    # Module contains direct parameters.
                    param_modules[id(module)] = module
        for module in param_modules.values():
            self.cuda_graph_manual_hooks.append((make_hook_func(), (module,)))

    def _get_submodules_under_cudagraphs(self):
        """
        Get the submodules that are covered by cudagraphs. Return a list that only contains the
        module itself if the whole layer is covered by cudagraphs.
        """
        return [self]

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """
        CUDA Graph capture for this layer using TE interface.
        Normally it's just a forward pass if we're capturing the entire layer.
        """
        return self.forward(*args, **kwargs)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """
        CUDA graph replay for this layer and microbatch `self.current_microbatch` using TE
        interface. TransformerEngine versions>=1.10 allow keyword arguments with CUDA graph.
        However, CUDA graph accepts only Tensor inputs.
        Hence, check if the arguments are all tensors.
        """
        try:
            for arg in args:
                assert isinstance(arg, torch.Tensor), "CUDA graph accepts only Tensor inputs."
            for _, value in kwargs.items():
                assert value is None or isinstance(
                    value, torch.Tensor
                ), "CUDA graph accepts only Tensor inputs."

            replay_state = getattr(self, "_te_cuda_graph_route_replay_state", None)
            microbatch_idx = (
                replay_state[0]
                if replay_state is not None
                else getattr(self, 'current_microbatch', 0)
            )
            cg_index = microbatch_idx % len(self.cuda_graphs)
            cudagraph_args, cudagraph_kwargs = self._get_te_cuda_graph_replay_args(*args, **kwargs)
            cudagraph_kwargs['is_first_microbatch'] = microbatch_idx == 0

            for hook, hook_args in self.cuda_graph_manual_hooks:
                hook(*hook_args)
            return self.cuda_graphs[cg_index](*cudagraph_args, **cudagraph_kwargs)
        finally:
            self._te_cuda_graph_route_replay_state = None

    def _get_te_cuda_graph_replay_args(self, *args, **kwargs):
        """Helper function to get tensor arguments for TE CUDA graph."""
        if len(args) == 0:
            assert 'hidden_states' in kwargs, "hidden_states is required."
            hidden_states = kwargs.pop('hidden_states')
            cudagraph_args = (hidden_states,)
        else:
            assert (
                'hidden_states' not in kwargs
            ), "hidden_states should only be passed as either a positional or keyword argument."
            cudagraph_args = tuple(args)

        cudagraph_kwargs = kwargs.copy()
        cudagraph_kwargs['is_first_microbatch'] = getattr(self, 'current_microbatch', 0) == 0
        if self.config.fine_grained_activation_offloading and getattr(
            self, 'offload_module_in_cuda_graph', False
        ):
            from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
                FineGrainedActivationOffloadingInterface as off_interface,
            )

            # TE captures/replays the module on its own graph stream. Passing the
            # offload stream/event in lets TE order graph compute with D2H/H2D
            # transfers managed by the fine-grained offload manager.
            cudagraph_kwargs['cuda_graph_stream'] = off_interface.cuda_graph_stream()
            cudagraph_kwargs['cuda_graph_event'] = off_interface.cuda_graph_event()
        return cudagraph_args, cudagraph_kwargs

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        return hasattr(self, 'cudagraph_manager')

    def _should_call_te_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the TE cudagraph path.
        """
        from megatron.core.transformer.cuda_graphs import is_graph_capturing

        return (
            self.config.cuda_graph_impl == "transformer_engine"
            and self.training
            and (is_graph_capturing() or self.cuda_graphs)
        )

    def __call__(self, *args, **kwargs):
        if self._should_call_local_cudagraph(*args, **kwargs):
            return self.cudagraph_manager(self, args, kwargs)
        elif self._should_call_te_cudagraph(*args, **kwargs):
            if not self.cuda_graphs:
                # Do CUDA Graphs capture.
                cuda_graph_func = self._te_cuda_graph_capture
            else:
                # Do CUDA Graphs replay.
                cuda_graph_func = self._te_cuda_graph_replay
            return cuda_graph_func(*args, **kwargs)
        return super().__call__(*args, **kwargs)


def conversion_helper(val, conversion):
    """Recursively applies a conversion function to values in nested data structures.

    Args:
        val: A single value or a nested structure (tuple/list) of values to convert
        conversion (callable): A function that performs the desired conversion on a single value

    Returns:
        The converted value, maintaining the same nested structure as the input.
        If input is a single value, returns the converted value.
        If input is a tuple/list, returns a tuple/list with all elements converted.
    """
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Converts floating-point values from fp32 to fp16.

    Args:
        val: The value to convert. Can be a single number, a tuple, or a list.
        float16_convertor: A function that converts a single fp32 value to fp16
    """

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Converts floating-point values from fp16 to fp32.

    Args:
        val: The value to convert. Can be a single number, a tuple, or a list.
    """

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


def mark_keep_in_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Mark a parameter or buffer so that ``Float16Module`` keeps it in FP32.

    Some parameters must stay in FP32 even when the rest of the model is converted to
    FP16/BF16 (e.g. the ``ape`` and ``attn_sink`` parameters of DeepSeek V4 sparse
    attention, which are FP32 in the reference checkpoint).

    Args:
        tensor: The parameter or buffer to mark.

    Returns:
        The same tensor, for call-site convenience.
    """
    tensor.keep_in_fp32 = True
    return tensor


def convert_module_to_dtype_except_fp32_marked(
    module: torch.nn.Module, dtype: torch.dtype
) -> torch.nn.Module:
    """Cast floating-point parameters and buffers of ``module`` to ``dtype``.

    Tensors marked with :func:`mark_keep_in_fp32` are left untouched.

    Args:
        module: The module to convert in place.
        dtype: The target floating-point dtype (``torch.half`` or ``torch.bfloat16``).

    Returns:
        The converted module.
    """
    return module._apply(
        lambda t: (
            t.to(dtype) if t.is_floating_point() and not getattr(t, 'keep_in_fp32', False) else t
        )
    )


class Float16Module(MegatronModule):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(Float16Module, self).__init__(config)
        self.config = config
        self.fp16 = config.fp16
        self.bf16 = config.bf16
        self.vp_size = config.virtual_pipeline_model_parallel_size
        self.vp_stage = getattr(module, 'vp_stage', None)
        self.pg_collection = getattr(module, 'pg_collection', None)

        if self.fp16:
            self.add_module(
                'module', convert_module_to_dtype_except_fp32_marked(module, torch.half)
            )

            def float16_convertor(val):
                return val.half()

        elif self.bf16:
            self.add_module(
                'module', convert_module_to_dtype_except_fp32_marked(module, torch.bfloat16)
            )

            def float16_convertor(val):
                return val.bfloat16()

        else:
            raise Exception('Either config.fp16 or config.bf16 should be True.')

        self.float16_convertor = float16_convertor

    def set_input_tensor(self, input_tensor):  # pylint: disable=missing-function-docstring
        return self.module.set_input_tensor(input_tensor)

    def forward(self, *inputs, fp32_output=True, **kwargs):
        """
        Execute the wrapped module in model precision and optionally upcast outputs to fp32.

        On the first pipeline stage, positional/keyword tensor inputs are converted to the
        module precision (fp16 or bf16) before invoking the wrapped module. The wrapped module
        is called with the provided inputs and keyword arguments. On the last pipeline stage
        only, outputs are upcast to fp32 if ``fp32_output`` is True; otherwise, outputs are
        returned in the model precision (fp16/bf16).

        Args:
            *inputs: Positional inputs forwarded to the wrapped module (converted to fp16/bf16 on
                the pipeline first stage).
            fp32_output (bool, keyword-only): If True (default), upcast outputs to fp32 on the
                pipeline last stage. Has no effect on non-last stages. Set to False to keep outputs
                in model precision when downstream consumers expect half precision or to avoid
                extra casts.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            The wrapped module's outputs, potentially upcast to fp32 depending on pipeline stage
            and ``fp32_output``.
        """
        from megatron.core.pipeline_parallel.utils import (
            is_pp_first_stage,
            is_pp_last_stage,
            is_vp_first_stage,
            is_vp_last_stage,
        )

        if self.pg_collection is None:
            pp_group = parallel_state.get_pipeline_model_parallel_group()
        else:
            pp_group = self.pg_collection.pp
        if is_vp_first_stage(self.vp_stage, self.vp_size) and is_pp_first_stage(pp_group):
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if (
            is_vp_last_stage(self.vp_stage, self.vp_size)
            and is_pp_last_stage(pp_group)
            and fp32_output is True
        ):
            outputs = float16_to_fp32(outputs)
        return outputs

    def state_dict(
        self, destination=None, prefix='', keep_vars=False
    ):  # pylint: disable=missing-function-docstring
        return self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Retrieve state_dict from the module being wrapped."""
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(self, prefix='', *args, **kwargs):
        """Retrieve sharded_state_dict from the module being wrapped."""
        return self.module.sharded_state_dict(prefix, *args, **kwargs)

    def load_state_dict(
        self, state_dict, strict=True
    ):  # pylint: disable=missing-function-docstring
        self.module.load_state_dict(state_dict, strict=strict)
