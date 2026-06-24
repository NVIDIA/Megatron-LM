# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedStateDict
from megatron.core.distributed.param_and_grad_buffer import group_params_for_buffers
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, get_pg_size, log_single_rank

from ..fp8_utils import (
    _stage_param_to_bf16,
    copy_back_gathered_bf16_into_fp8_param,
    is_float8tensor,
    post_all_gather_processing,
)
from .clip_grads import count_zeros_fp32, get_grad_norm_fp32
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
    _get_param_grad_norm_group,
    _validate_grad_norm_group,
)
from .optimizer_config import OptimizerConfig
from .param_layout import (
    FullParamLayout,
    PerBufferParamLayout,
    bucket_end_divisor,
    pad_param_start,
    pad_to_divisor,
)

logger = logging.getLogger(__name__)


def is_managed_by_layer_wise_optimizer(param: torch.nn.Parameter) -> bool:
    """Whether a parameter is managed by :class:`LayerWiseDistributedOptimizer`.

    Returns True for the 2D matrix-like weight parameters that Muon orthogonalizes
    via Newton-Schulz, and False for embeddings, biases, LayerNorm weights, and
    any other non-matrix parameter (which are handled by Adam through a separate
    :class:`DistributedOptimizer`).

    Mirrors the routing rule applied by ``_get_param_groups`` /
    ``default_param_overrides`` for Muon.
    """
    if not param.dim() == 2:
        return False
    if getattr(param, 'is_embedding_or_output_parameter', False):
        return False
    return True


def _bucket_is_managed_by_layer_wise_optimizer(bucket, default_for_untagged: bool = True) -> bool:
    """Whether a DDP bucket belongs to a LayerWise-managed buffer.

    Buckets are built from params that share a :class:`BufferKey`, so checking
    the first param's tag is sufficient. ``default_for_untagged`` controls the
    legacy (no-tagging) case: callers asking "is this mine?" from the LayerWise
    side pass ``True`` (legacy LayerWise owns everything); callers asking from
    the DistOpt side pass ``False`` (legacy DistOpt also owns everything, so
    untagged buckets are *not* LayerWise-managed).
    """
    if not bucket.params_list:
        return False
    param = bucket.params_list[0]
    if not hasattr(param, 'is_managed_by_layer_wise_optimizer'):
        return default_for_untagged
    return param.is_managed_by_layer_wise_optimizer


def _param_sort_key(numel: int, identity: tuple) -> tuple:
    """Rank-independent total-order key for ping-pong ownership: ``(numel, *canonical-identity)``.

    ``numel`` alone is not a total order (stable sort tie-breaks by rank-local insertion order),
    so equal-numel params would get different owners across ranks; the canonical identity
    ``(chunk_idx, buffer_idx, global_start_index)`` makes it total and identical on every rank.
    """
    return (numel,) + tuple(identity)


def tag_params_for_buffer_routing(model_chunks) -> None:
    """Tag every requires-grad param with ``is_managed_by_layer_wise_optimizer``.

    Run this once on the un-DDP-wrapped model chunks before
    :class:`DistributedDataParallel` constructs its grad/param buffers — the
    grouping function ``group_params_for_buffers`` reads this attribute to
    decide which buffer each param lands in (LayerWise shard-aligned buffer vs
    DistOpt-style byte-level buffer).
    """
    for model_chunk in model_chunks:
        for param in model_chunk.parameters():
            if not param.requires_grad:
                continue
            param.is_managed_by_layer_wise_optimizer = is_managed_by_layer_wise_optimizer(param)


class LayerWiseDistributedOptimizer(ChainedOptimizer):
    """Layer-wise distributed optimizer for Megatron-core models.

    Experimental distributed optimizer wrapper that distributes weight to DP ranks by layer.
    Implemented as ChainedOptimizer to support multiple optimizers (e.g. muon + adamW)
    When using, keep all megatron distributed-optimizer related options OFF.

    How LayerWiseDistributedOptimizer work:
    1. weights are splited into lists and each rank only keep its shard in its optimizer
    2. Megatron DDP handle allreduce grad, note that each rank have full model and grad
    3. optimizer is already modified so only param belong to this DP rank is updated
    4. grad_norm and zero counting will reduce metrics globally in step function
    5. Do regular update with chained optimizers, modified optimizer only update shard
    6. allgather updated params to every rank
    """

    @staticmethod
    def _shard_divisor(data_parallel_world_size: int, ddp_config) -> int:
        """Per-shard alignment divisor.

        Guarantees that ``dp_size * shard_size`` satisfies bucket-end alignment
        and that every shard start is 64-element aligned (required by
        :func:`pad_param_start`).
        """
        dp_size = data_parallel_world_size
        bucket_divisor = bucket_end_divisor(dp_size, ddp_config.pad_buckets_for_high_nccl_busbw)
        return math.lcm(64, bucket_divisor // dp_size)

    @staticmethod
    def _compute_per_buffer_param_layout(
        params: List[torch.nn.Parameter],
        bucket_size: Optional[int],
        data_parallel_world_size: int,
        ddp_config,
        param_indices: Optional[List[int]] = None,
    ) -> 'PerBufferParamLayout':
        """Compute parameter layout with shard-aligned buckets via LPT bin-packing.

        Assigns parameters to ``dp_size`` shards within each bucket so that no
        parameter is split across a shard boundary, while keeping each bucket
        confined to a contiguous range in backprop order.

        **Algorithm** (operates in reverse model / backprop order):

        1. Walk parameters in backprop order, accumulating them into a chunk.
           A shared (tied) embedding triggers an immediate finalisation
           followed by an isolated bucket for that embedding alone.
        2. When the chunk's total numel reaches ``bucket_size`` (or all
           params have been consumed), bin-pack the chunk into ``dp_size``
           shards via greedy LPT — sort by numel descending and assign each
           param to the shard with the smallest current load.
        3. Pad each shard to ``max(shard_cursors)`` aligned to
           :meth:`_shard_divisor`, then emit the bucket.

        Each bucket therefore spans a contiguous backprop range so that
        ``overlap_grad_reduce`` can dispatch the bucket's reduce-scatter as
        soon as the bucket's backward segment finishes — preserving the
        original DDP overlap semantics.  LPT bin-packing keeps shards close
        to balanced; for uniform transformer blocks where ``params_per_layer
        * num_layers`` is a multiple of ``dp_size`` the packing is perfect.

        Args:
            params: Parameters in model-definition (forward) order.
            bucket_size: Approximate elements per bucket (``None`` → single bucket).
            data_parallel_world_size: Size of the data-parallel group.
            ddp_config: :class:`DistributedDataParallelConfig`.
            param_indices: Optional per-param dtype indices (passed through).

        Returns:
            :class:`PerBufferParamLayout` with shard-aligned buckets.
        """
        dp_size = data_parallel_world_size
        shard_divisor = LayerWiseDistributedOptimizer._shard_divisor(dp_size, ddp_config)

        total_param_numel = sum(p.data.nelement() for p in params)

        param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = {}
        bucket_indices: List[Tuple[int, int]] = []
        per_bucket_numel_unpadded: List[int] = []
        buffer_cursor = 0
        bucket_id = 0
        shard_imbalance_padding_numel = 0

        def _emit_bucket(
            chunk_params: List[torch.nn.Parameter], shared_embedding: bool = False
        ) -> None:
            """Bin-pack *chunk_params* into ``dp_size`` shards and emit a bucket.

            With ``shared_embedding=True``, the chunk must contain a single
            parameter; it goes into shard 0 with same-size padding in
            shards 1..dp_size-1 so the embedding fits entirely within one
            shard (needed for the cross-stage tied-embedding all-reduce).
            """
            nonlocal buffer_cursor, bucket_id, shard_imbalance_padding_numel
            if not chunk_params:
                return

            shard_assignments: List[List[Tuple[Optional[torch.nn.Parameter], int]]] = [
                [] for _ in range(dp_size)
            ]
            shard_cursors = [0] * dp_size

            if shared_embedding:
                assert len(chunk_params) == 1
                param = chunk_params[0]
                numel = param.data.nelement()
                shard_assignments[0].append((param, numel))
                shard_cursors[0] = numel
                for shard_id in range(1, dp_size):
                    shard_assignments[shard_id].append((None, numel))
                    shard_cursors[shard_id] = numel
            else:
                # Greedy LPT: largest first, assign to the least-loaded shard.
                # The within-shard order is sorted-by-numel, not backprop —
                # that is fine because all params in the chunk share the same
                # bucket_id, so DDP's backprop-order iteration still sees
                # monotonic bucket_ids across the chunk boundary.
                for param in sorted(chunk_params, key=lambda p: -p.data.nelement()):
                    numel = param.data.nelement()
                    min_shard = min(range(dp_size), key=lambda s: shard_cursors[s])
                    placement = pad_param_start(shard_cursors[min_shard])
                    shard_assignments[min_shard].append((param, numel))
                    shard_cursors[min_shard] = placement + numel

            padded_shard_size = pad_to_divisor(max(shard_cursors), shard_divisor)
            bucket_start_index = buffer_cursor
            for shard_id in range(dp_size):
                shard_start_index = bucket_start_index + shard_id * padded_shard_size
                cursor = shard_start_index
                for p, numel in shard_assignments[shard_id]:
                    cursor = pad_param_start(cursor)
                    if p is not None:
                        param_index_map[p] = (cursor, cursor + numel, bucket_id)
                    cursor += numel
                shard_imbalance_padding_numel += padded_shard_size - shard_cursors[shard_id]
            bucket_end_index = bucket_start_index + dp_size * padded_shard_size
            bucket_indices.append((bucket_start_index, bucket_end_index))
            per_bucket_numel_unpadded.append(sum(p.data.nelement() for p in chunk_params))
            buffer_cursor = bucket_end_index
            bucket_id += 1

        # Each chunk spans a contiguous backprop range. Bucket ids therefore
        # increase monotonically when ``_ParamAndGradBuffer.__init__`` iterates
        # params in backprop order, satisfying its ``bucket_id == cur + 1``
        # invariant.
        #
        # Padding floor: the on-buffer bucket size is ``dp_size *
        # max_shard_cursor``, which is at least ``dp_size * chunk_max_param``
        # because some shard must hold that param whole. If a single param
        # dominates the chunk, finalising on ``chunk_numel >= bucket_size``
        # alone would emit a bucket with most of its shards near-empty
        # padding. Instead extend the chunk so its raw numel approaches the
        # padded buffer size, capping per-bucket overhead at ``1 /
        # PADDING_FLOOR - 1`` (~11% at 0.9). Falls back to ``bucket_size``
        # when no single param dominates.
        PADDING_FLOOR = 0.9
        chunk_params: List[torch.nn.Parameter] = []
        chunk_numel = 0
        chunk_max_param = 0
        for param in reversed(params):
            param_numel = param.data.nelement()
            if getattr(param, 'shared_embedding', False):
                # Finalize any in-progress chunk so the shared-embedding
                # bucket comes after it in backprop order.
                _emit_bucket(chunk_params)
                chunk_params = []
                chunk_numel = 0
                chunk_max_param = 0
                _emit_bucket([param], shared_embedding=True)
                continue
            chunk_params.append(param)
            chunk_numel += param_numel
            chunk_max_param = max(chunk_max_param, param_numel)
            if bucket_size is not None:
                threshold = max(bucket_size, int(dp_size * chunk_max_param * PADDING_FLOOR))
                if chunk_numel >= threshold:
                    _emit_bucket(chunk_params)
                    chunk_params = []
                    chunk_numel = 0
                    chunk_max_param = 0
        _emit_bucket(chunk_params)

        total_buffer_numel = bucket_indices[-1][1] if bucket_indices else 0
        total_padding = total_buffer_numel - total_param_numel
        log_single_rank(
            logger,
            logging.INFO,
            f"Layerwise param layout: {len(params)} params, "
            f"{len(bucket_indices)} buckets, "
            f"dp_size={dp_size}, "
            f"total_param_numel={total_param_numel}, "
            f"total_buffer_numel={total_buffer_numel}, "
            f"total_padding={total_padding} "
            f"(shard_imbalance={shard_imbalance_padding_numel}), "
            f"overhead={total_padding / max(total_param_numel, 1) * 100:.1f}%",
        )

        return PerBufferParamLayout(
            param_index_map=param_index_map,
            bucket_indices=bucket_indices,
            per_bucket_numel_unpadded=per_bucket_numel_unpadded,
            param_indices=param_indices if param_indices is not None else [],
        )

    @staticmethod
    def compute_full_param_layout(
        params: List[torch.nn.Parameter],
        bucket_size: Optional[int],
        data_parallel_world_size: int,
        ddp_config,
        expert_data_parallel_world_size: Optional[int] = None,
    ) -> 'FullParamLayout':
        """Compute parameter layouts for all buffer groups.

        Groups parameters by :class:`BufferKey` via :func:`group_params_for_buffers`
        and produces a layerwise shard-aligned size-matching layout per buffer.
        Every parameter stays within a single shard so the local optimizer step
        (e.g. Newton-Schulz iteration for Muon) can run on whole tensors.

        Args:
            params: All parameters to lay out.
            bucket_size: Approximate elements per bucket (``None`` → single bucket).
            data_parallel_world_size: DP group size for dense parameters.
            ddp_config: :class:`DistributedDataParallelConfig`.
            expert_data_parallel_world_size: Expert DP group size (defaults to
                ``data_parallel_world_size``).

        Returns:
            :class:`FullParamLayout` with a :class:`PerBufferParamLayout` per buffer group.
        """
        # Avoid a circular import: DistributedOptimizer imports LayerWise indirectly.
        from ..distributed.param_and_grad_buffer import _compute_default_per_buffer_param_layout
        from .distrib_optimizer import DistributedOptimizer

        # Decoupled layout (use_layer_wise_param_layout=False): LayerWise (Muon) buffers use a
        # compact no-padding DDP layout (and locally disable DistributedOptimizer semantics in
        # DDP), so they must NOT receive the shard-aligned ``dp_size * max(shard_load)`` padded
        # layout here. Non-LayerWise buffers keep DistOpt's byte-level layout regardless.
        decouple_ddp_layout = not getattr(ddp_config, 'use_layer_wise_param_layout', True)

        # fp8 Muon grads key to uint8 (own buffer); partition_buckets later merges the non-fp8
        # bucket groups into the fp8 group to aggregate communication.
        buffer_groups = group_params_for_buffers(
            params,
            ddp_config.grad_reduce_in_fp32,
            merge_layerwise_fp8_grads=not getattr(ddp_config, 'use_layer_wise_param_layout', True),
        )
        layouts = {}
        for buffer_key, (group_params, param_indices) in buffer_groups.items():
            if buffer_key.is_expert_parallel:
                dp_world_size = (
                    expert_data_parallel_world_size
                    if expert_data_parallel_world_size is not None
                    else data_parallel_world_size
                )
            else:
                dp_world_size = data_parallel_world_size

            # Dispatch per buffer: LayerWise (Muon) params get the shard-aligned
            # layout; non-LayerWise params (e.g. Adam-managed embeddings, biases)
            # get DistOpt's byte-level layout.
            if buffer_key.is_managed_by_layer_wise_optimizer and decouple_ddp_layout:
                # Decouple path (incl. FP8 param-gather): compact no-padding layout (DDP treats this
                # buffer as non-DistOpt). Attach param_indices so DDP's consistency check passes.
                per_buffer_layout = _compute_default_per_buffer_param_layout(
                    group_params, bucket_size
                )
                per_buffer_layout.param_indices = param_indices
                layouts[buffer_key] = per_buffer_layout
                continue
            if buffer_key.is_managed_by_layer_wise_optimizer:
                compute_per_buffer_layout = (
                    LayerWiseDistributedOptimizer._compute_per_buffer_param_layout
                )
            else:
                compute_per_buffer_layout = DistributedOptimizer._compute_per_buffer_param_layout
            layouts[buffer_key] = compute_per_buffer_layout(
                group_params, bucket_size, dp_world_size, ddp_config, param_indices
            )
        return FullParamLayout(layouts=layouts)

    def __init__(
        self,
        optimizers: List[MegatronOptimizer],
        config: OptimizerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        init_state_fn_list: Optional[List[Callable]] = None,
        model_chunks: Optional[List] = None,
    ) -> None:
        """
        Initialize LayerWiseDistributedOptimizer.

        Args:
            optimizers: List of MegatronOptimizers.
            config: OptimizerConfig.
            pg_collection: ProcessGroupCollection.
            init_state_fn_list: List of init state functions.
            model_chunks: DDP-wrapped model chunks.
        """

        self.pg_collection = pg_collection

        # Decoupled layout (use_layer_wise_param_layout=False): LayerWise (Muon) buffers use a
        # compact no-padding DDP layout that does NOT encode whole-param shard ownership, and DDP
        # treats them as non-DistOpt (all-reduce gradients). Ownership must therefore use the
        # legacy whole-param ping-pong assignment and param sync must use ``allgather_params``,
        # even though the model chunks carry a (compact) full_param_layout for DDP's benefit.
        self.decouple_ddp_layout = not getattr(config, 'use_layer_wise_param_layout', True)

        full_param_layouts = None
        if model_chunks is not None and not self.decouple_ddp_layout:
            full_param_layouts = [
                chunk.full_param_layout
                for chunk in model_chunks
                if hasattr(chunk, 'full_param_layout') and chunk.full_param_layout is not None
            ] or None
        # Decouple path keeps whole-matrix ping-pong ownership (Newton-Schulz runs on whole
        # matrices on one rank; param sync via ``allgather_params``). ``model_chunks`` lets the
        # ping-pong fallback break equal-numel ties by a rank-independent identity (see below).
        self.shard_params(optimizers, full_param_layouts, model_chunks)

        # Engage FP8 param sync automatically when the decouple-managed params are actually
        # quantized (fp8_param_gather on + TE Float8/MXFP8 weights). Off -> plain bf16 path.
        self.use_fp8_param_sync = False
        if self.decouple_ddp_layout:
            for params_list in (self.dp_cp_params_list, self.expt_dp_params_list):
                if not params_list:
                    continue
                for per_rank in params_list:
                    if any(is_float8tensor(p) for p in per_rank):
                        self.use_fp8_param_sync = True
                        break
                if self.use_fp8_param_sync:
                    break

        # When a full_param_layout is available (and we are not decoupling),
        # ddp_config.use_distributed_optimizer is True and model params are views into the
        # DDP param buffer.  After the optimizer step copies updated fp32 main params → bf16
        # model params, the buffer is already up-to-date in-place.  We can use DDP's
        # buffer-based all-gather (start_param_sync) instead of the flatten/unflatten
        # allgather_params path.  In the decouple path (incl. FP8 param-gather), Muon buffers are
        # non-DistOpt and own whole params via ping-pong, so we use the legacy allgather_params.
        self.use_buffer_param_sync = full_param_layouts is not None

        # Set up overlap param gather using DDP bucket infrastructure.
        self.overlap_param_gather = config.overlap_param_gather
        if self.overlap_param_gather and not self.use_buffer_param_sync:
            # Legacy path: set up per-bucket param lists for variable-size all-gather.
            # When use_buffer_param_sync is True, the standard distributed optimizer
            # all-gather path is used and this setup is not needed.
            assert (
                model_chunks is not None
            ), "model_chunks must be provided if overlap_param_gather is True"
            self.set_bucket_layerwise_params_list(model_chunks)

        if init_state_fn_list:
            assert len(init_state_fn_list) == len(
                optimizers
            ), "init_state_fn_list must be the same length as optimizers if provided"

        # Wrap base torch optimizers with Float16 for bf16 training.
        # Callers pass base optimizers; wrapping happens here *after*
        # shard_params so master weights are only created for the local shard.
        if config.bf16:
            for i in range(len(optimizers)):
                opt = optimizers[i]
                if isinstance(opt, (Float16OptimizerWithFloat16Params, FP32Optimizer)):
                    raise TypeError(
                        'LayerWiseDistributedOptimizer expects base torch optimizers, '
                        f'got {type(opt).__name__}. Do not pre-wrap with Megatron optimizers.'
                    )
                optimizers[i] = Float16OptimizerWithFloat16Params(
                    opt, config, None, init_state_fn_list[i] if init_state_fn_list else None
                )
                # Non-DistOpt LayerWise child has no byte-shard param buffer, so tag it to route
                # step_with_ready_grads to copy fp32 master straight into model ``param.data`` even
                # under reuse_grad_buf (which would otherwise call the unsupported
                # ``_copy_main_params_to_param_buffer``); the gather staging reuses the grad buffer.
                optimizers[i]._layer_wise_non_distopt_child = True

        super().__init__(optimizers)

        # Assign self.model_chunks AFTER super().__init__: ChainedOptimizer.__init__
        # resets self.model_chunks to [] and then repopulates only from chained
        # children that have a model_chunks attribute (DistOpt does, Float16-wrapped
        # raw torch optimizers do not). Set it here so LayerWise.step's
        # ``for model_chunk in self.model_chunks`` actually iterates.
        self.model_chunks = model_chunks if model_chunks is not None else []

        # TODO(kunlun, deyuf): potential future perf optimization
        # since allreduce is unchanged and handled by megatron DDP, they're already in
        # contiguous gbuf. So instead of shard param by layer randomly, we can shard by
        # buf range but keep some "extras" to keep boundary weight not sharded.
        # This way each rank do some duplicated work but allgather_v is no longer needed
        # All current distopt optimization can also be potentially applied

    def shard_params(self, optimizers, full_param_layouts=None, model_chunks=None):
        """Shard params across ranks according to the computed param layout.

        Each param's shard assignment is derived from the :class:`FullParamLayout`
        stored on the DDP model chunks.  Within each bucket the buffer is divided
        into ``dp_size`` equal shards; a param's shard index is determined by its
        position in the buffer.

        Falls back to the legacy ping-pong-by-numel strategy when no layout is
        available (e.g. ``dp_size == 1`` or no DDP wrapper).

        Args:
            optimizers: Optimizers whose param groups will be narrowed to
                the local rank's shard.
            full_param_layouts: List of :class:`FullParamLayout` (one per model
                chunk).  ``None`` triggers the legacy fallback.
        """
        # Simplify when dp_cp group size is 1.
        dp_cp_size = get_pg_size(self.pg_collection.dp_cp)
        if dp_cp_size == 1:
            self.dp_cp_params_list = None
            self.expt_dp_params_list = None
            return

        expt_dp_size = get_pg_size(self.pg_collection.expt_dp)

        if full_param_layouts is not None:
            self._shard_params_from_layout(optimizers, full_param_layouts, dp_cp_size, expt_dp_size)
        else:
            self._shard_params_ping_pong(optimizers, dp_cp_size, expt_dp_size, model_chunks)

    def _shard_params_from_layout(self, optimizers, full_param_layouts, dp_cp_size, expt_dp_size):
        """Derive shard assignments from the param layout."""
        dp_cp_rank = get_pg_rank(self.pg_collection.dp_cp)
        expt_dp_rank = get_pg_rank(self.pg_collection.expt_dp)

        self.dp_cp_params_list = [[] for _ in range(dp_cp_size)]
        self.expt_dp_params_list = [[] for _ in range(expt_dp_size)]

        # Map each param to its shard index.
        param_to_shard: Dict[torch.nn.Parameter, int] = {}
        for full_layout in full_param_layouts:
            for buffer_key, layout in full_layout.layouts.items():
                # Non-LayerWise buffers (e.g. Adam-managed embeddings, biases,
                # layernorms with a DistOpt byte-level layout) are managed by a
                # separate DistributedOptimizer; LayerWise does not own them.
                if not buffer_key.is_managed_by_layer_wise_optimizer:
                    continue
                dp_size = expt_dp_size if buffer_key.is_expert_parallel else dp_cp_size
                for param, (
                    param_start_index,
                    param_end_index,
                    bucket_id,
                ) in layout.param_index_map.items():
                    bucket_start_index, bucket_end_index = layout.bucket_indices[bucket_id]
                    shard_size = (bucket_end_index - bucket_start_index) // dp_size
                    shard_id = (param_start_index - bucket_start_index) // shard_size
                    shard_end_index = bucket_start_index + (shard_id + 1) * shard_size
                    assert param_end_index <= shard_end_index, (
                        f"Param (shape={tuple(param.shape)}, numel={param.numel()}) at "
                        f"({param_start_index}, {param_end_index}) crosses shard boundary "
                        f"in bucket ({bucket_start_index}, {bucket_end_index}) with "
                        f"shard_size={shard_size}, shard_id={shard_id}, "
                        f"shard_end_index={shard_end_index}. The layout must keep every "
                        f"param fully within one shard."
                    )
                    param_to_shard[param] = shard_id

        # Collect all param groups and assign params to per-rank lists.
        param_groups = []
        for optimizer in optimizers:
            param_groups += optimizer.param_groups
        param_groups_this_rank = [[] for _ in param_groups]

        for group_index, group in enumerate(param_groups):
            is_expert = group.get("is_expert_parallel", False)
            local_rank = expt_dp_rank if is_expert else dp_cp_rank
            params_list = self.expt_dp_params_list if is_expert else self.dp_cp_params_list

            for param in group["params"]:
                assert param in param_to_shard, (
                    f"Optimizer param (shape={tuple(param.shape)}, numel={param.numel()}) "
                    f"not found in any param layout. Ensure all optimizer params are "
                    f"included in the full_param_layout passed to DDP."
                )
                shard_id = param_to_shard[param]
                params_list[shard_id].append(param)
                if shard_id == local_rank:
                    param_groups_this_rank[group_index].append(param)

        # Now we modify the group to only handle local params.
        for group, local_params in zip(param_groups, param_groups_this_rank):
            group["params"] = local_params

        # Simplify when expt_dp group size is 1 or expert parallel is off.
        if expt_dp_size == 1 or len(self.expt_dp_params_list[0]) == 0:
            self.expt_dp_params_list = None

    def _build_param_sort_keys(self, model_chunks):
        """Build ``{param: (chunk_idx, buffer_idx, global_start_index)}`` — a rank-independent key
        for every requires-grad param.

        Both the chunk/buffer enumeration order and the ``param_index_map`` offsets come purely
        from model construction (identical across DP ranks), so the key is the same on every rank.
        Used to break equal-numel ties in ``_shard_params_ping_pong``. Returns ``None`` if no layout
        info is available, so the caller falls back to legacy numel-only ordering.
        """
        if model_chunks is None:
            return None
        identity: Dict[torch.nn.Parameter, tuple] = {}
        for chunk_idx, chunk in enumerate(model_chunks):
            buffers = list(getattr(chunk, 'buffers', [])) + list(
                getattr(chunk, 'expert_parallel_buffers', [])
            )
            for buffer_idx, buffer in enumerate(buffers):
                param_index_map = getattr(buffer, 'param_index_map', None)
                if param_index_map is None:
                    continue
                for param, (global_start, _global_end, _bucket_id) in param_index_map.items():
                    identity[param] = (chunk_idx, buffer_idx, global_start)
        return identity or None

    def _shard_params_ping_pong(self, optimizers, dp_cp_size, expt_dp_size, model_chunks=None):
        """Legacy ping-pong shard assignment (no layout available).

        Legacy: this method is a fallback for when no ``full_param_layout``
        is provided.  Once all call sites supply a layout, this can be removed
        in favor of :meth:`_shard_params_from_layout`.

        Parameters are sorted by a rank-independent TOTAL order and assigned ping-pong style. E.g.
        4 ranks, 10 params p0-p9 -> [[p0, p7, p8], [p1, p6, p9], [p2, p5], [p3, p4]].

        CRITICAL: the sort key MUST be identical across DP ranks. ``numel`` alone is not (stable
        sort tie-breaks equal-numel params by insertion order), which would give different owners
        per rank -> params double-owned or zero-owned on the first step. So we tie-break by the
        canonical identity ``(chunk_idx, buffer_idx, global_start_index)``.
        """
        dp_cp_idx, expt_dp_idx = 0, 0
        # Create ping-pong style loop so memory is more balanced.
        dp_cp_loop = list(range(dp_cp_size)) + list(range(dp_cp_size))[::-1]
        expt_dp_loop = list(range(expt_dp_size)) + list(range(expt_dp_size))[::-1]
        self.dp_cp_params_list = [[] for _ in range(dp_cp_size)]
        self.expt_dp_params_list = [[] for _ in range(expt_dp_size)]
        # Get all param groups.
        param_groups = []
        for optimizer in optimizers:
            param_groups += optimizer.param_groups

        # Sort param in all groups by a rank-independent TOTAL order, then assign to each rank.
        identity = self._build_param_sort_keys(model_chunks)
        param_list = []
        for group_index, group in enumerate(param_groups):
            for p in group["params"]:
                param_list.append((p, group_index))
        if identity is not None:
            # Total order: (numel, canonical-global-identity). Identical on every DP rank.
            missing = [p for (p, _) in param_list if p not in identity]
            assert not missing, (
                "ping-pong ownership requires a canonical identity for every Muon param, "
                f"but {len(missing)} param(s) were not found in any model-chunk buffer's "
                "param_index_map. Cannot guarantee identical ownership across ranks (the "
                "allgather_params gather assumes every rank agrees on each param's single owner)."
            )
            param_list.sort(key=lambda x: _param_sort_key(x[0].numel(), identity[x[0]]))
        else:
            # No layout info: keep the legacy numel-only ordering.
            param_list.sort(key=lambda x: x[0].numel())
        param_groups_this_rank = [[] for g in param_groups]

        # Assign params to rank in ping-pong style loop.
        for p, group_index in param_list:
            if param_groups[group_index].get("is_expert_parallel", False):
                if expt_dp_loop[expt_dp_idx] == get_pg_rank(self.pg_collection.expt_dp):
                    param_groups_this_rank[group_index].append(p)
                self.expt_dp_params_list[expt_dp_loop[expt_dp_idx]].append(p)
                expt_dp_idx = (expt_dp_idx + 1) % len(expt_dp_loop)
            else:
                if dp_cp_loop[dp_cp_idx] == get_pg_rank(self.pg_collection.dp_cp):
                    param_groups_this_rank[group_index].append(p)
                self.dp_cp_params_list[dp_cp_loop[dp_cp_idx]].append(p)
                dp_cp_idx = (dp_cp_idx + 1) % len(dp_cp_loop)

        # Now we modify the group to only handle local params.
        for groups, params in zip(param_groups, param_groups_this_rank):
            groups["params"] = params

        # Simplify when expt_dp group size is 1 or expert parallel is off.
        if expt_dp_size == 1 or len(self.expt_dp_params_list[0]) == 0:
            self.expt_dp_params_list = None

    def set_bucket_layerwise_params_list(self, model_chunks):
        """Map sharded params to DDP buckets for async all-gather.

        Legacy: only used by the variable-size all-gather path
        (``use_buffer_param_sync=False``).  Once all call sites supply a
        ``full_param_layout``, this can be removed — the standard distributed
        optimizer buffer all-gather handles param sync without per-bucket
        param lists.

        For each bucket in each model chunk's bucket groups, build per-rank param lists
        by cross-referencing the layer-wise sharded param lists with the bucket's params.

        Args:
            model_chunks: DDP-wrapped model chunks with bucket_groups.
        """
        for model_chunk in model_chunks:
            for group in model_chunk.bucket_groups:
                for bucket in group.buckets:
                    if not _bucket_is_managed_by_layer_wise_optimizer(bucket):
                        continue
                    if self.dp_cp_params_list is not None:
                        bucket_params_list = [
                            [] for _ in range(get_pg_size(self.pg_collection.dp_cp))
                        ]
                        for bucket_list, full_params_list in zip(
                            bucket_params_list, self.dp_cp_params_list
                        ):
                            for param in full_params_list:
                                if param in bucket.params:
                                    bucket_list.append(param)
                    else:
                        # dp_cp_size == 1: single rank owns all params; init the structure anyway
                        # (mirrors the expert block; shard_params sets dp_cp_params_list=None here).
                        bucket_params_list = [list(bucket.params_list)]
                    bucket.set_layerwise_params_list(bucket_params_list)
            # Do the same for expert parallel bucket groups.
            for group in model_chunk.expert_parallel_bucket_groups:
                for bucket in group.buckets:
                    if not _bucket_is_managed_by_layer_wise_optimizer(bucket):
                        continue
                    if self.expt_dp_params_list is not None:
                        bucket_params_list = [
                            [] for _ in range(get_pg_size(self.pg_collection.expt_dp))
                        ]
                        for bucket_list, full_params_list in zip(
                            bucket_params_list, self.expt_dp_params_list
                        ):
                            for param in full_params_list:
                                if param in bucket.params:
                                    bucket_list.append(param)
                    else:
                        # expt_dp_size == 1: single rank owns all params, no
                        # all-gather needed but data structures must be initialized.
                        bucket_params_list = [list(bucket.params_list)]
                    bucket.set_layerwise_params_list(bucket_params_list)

    @torch.no_grad()
    def allgather_params(self) -> None:
        """All-gather updated params from all ranks.

        Legacy: only used when ``use_buffer_param_sync=False``.  Once all
        call sites supply a ``full_param_layout``, this can be removed — the
        standard distributed optimizer buffer all-gather (via
        ``start_param_sync``) replaces this flatten/unflatten path.

        Two transport variants share the same uneven (all-gather-v) shape:

        * **bf16** (``use_fp8_param_sync=False``): all-gather owned bf16 ``param.data``, copy_ into
          non-owned params.
        * **fp8** (``use_fp8_param_sync=True``): stage owned fp32 master->bf16, all-gather bf16,
          requantize into EVERY rank's ``param.data`` (owned included) so all hold
          ``Q(bf16(master))`` (== OFF/Adam). Then ``post_all_gather_processing`` rebuilds fp8
          columnwise/transpose (blockwise/Float8; mxfp8 noop since copy-back already forced it).
        """

        # FP8-aware variant: stage bf16, uneven all-gather bf16, requantize per rank.
        def _allgather_helper_fp8(params_list, group):
            # TODO(perf, blockwise-only): blockwise could gather the owner's fp8 rowwise data
            # (~2x less comm) instead of bf16; mxfp8 must stay on bf16. See the matching TODO in
            # ``_ParamAndGradBucketGroup.start_param_sync`` for the full rationale.
            rank = get_pg_rank(group)
            dp_size = get_pg_size(group)
            # Device from any non-empty owned list (rank 0 may own zero params in the layout).
            device = next((params[0].device for params in params_list if len(params) > 0), None)
            if device is None:
                # No rank owns any param in this buffer -> nothing to gather.
                return

            # Stage fp32 master->bf16 (high-precision source), not lossy dequant(fp8).
            owned = params_list[rank]
            src = (
                _flatten_dense_tensors([_stage_param_to_bf16(p) for p in owned])
                if len(owned) > 0
                else torch.empty(0, device=device, dtype=torch.bfloat16)
            )
            flat_sizes = [sum(p.numel() for p in params) for params in params_list]
            if max(flat_sizes) == 0:
                return

            gather_list = []
            for i in range(dp_size):
                if i == rank:
                    gather_list.append(src)
                else:
                    gather_list.append(
                        torch.empty(flat_sizes[i], device=device, dtype=torch.bfloat16)
                    )

            torch.distributed.all_gather(gather_list, src, group=group)

            # Requantize the gathered bf16 into EVERY rank's params (owned included) so all ranks
            # hold Q(bf16(master)), matching OFF/Adam. Unflatten by param shape (logical numel).
            for idx, params in enumerate(params_list):
                if len(params) == 0:
                    continue
                templates = [
                    torch.empty(p.shape, device="meta", dtype=torch.bfloat16) for p in params
                ]
                updated_params = _unflatten_dense_tensors(gather_list[idx], templates)
                for updated_bf16, model_p in zip(updated_params, params):
                    copy_back_gathered_bf16_into_fp8_param(model_p, updated_bf16)

            # Rebuild fp8 columnwise/transpose after the gather (mirrors the overlap / DistOpt
            # paths; blockwise/Float8 build it, mxfp8 is a noop). Else it'd be deferred to forward.
            fp8_params = [p for params in params_list for p in params if is_float8tensor(p)]
            if fp8_params:
                post_all_gather_processing(fp8_params)

        # helper function to flatten local params, all-gather,
        # unflatten and copy to model params
        def _allgather_helper(params_list, group):
            device = params_list[0][0].device
            dtype = params_list[0][0].dtype
            rank = get_pg_rank(group)
            dp_size = get_pg_size(group)
            # Flatten this rank's params.
            src = (
                _flatten_dense_tensors(params_list[rank])
                if len(params_list[rank]) > 0
                else torch.empty(0, device=device, dtype=dtype)
            )
            flat_sizes = [sum(p.numel() for p in params) for params in params_list]
            if max(flat_sizes) == 0:
                return

            # Allocate per-rank receive buffers with actual sizes (no padding).
            # PyTorch's NCCL backend handles uneven sizes in all_gather via
            # grouped send/recv internally. Reuse src for local rank's slot.
            gather_list = []
            for i in range(dp_size):
                if i == rank:
                    gather_list.append(src)
                else:
                    gather_list.append(torch.empty(flat_sizes[i], device=device, dtype=dtype))

            torch.distributed.all_gather(gather_list, src, group=group)

            # Unflatten and copy gathered params for each rank.
            for idx, params in enumerate(params_list):
                if len(params) == 0 or idx == rank:
                    continue
                updated_params = _unflatten_dense_tensors(gather_list[idx], params)
                for updated_p, model_p in zip(updated_params, params):
                    model_p.data.copy_(updated_p)

        if self.pg_collection is None:
            return
        helper = _allgather_helper_fp8 if self.use_fp8_param_sync else _allgather_helper
        if self.dp_cp_params_list:
            helper(self.dp_cp_params_list, self.pg_collection.dp_cp)
        if self.expt_dp_params_list:
            helper(self.expt_dp_params_list, self.pg_collection.expt_dp)

    @torch.no_grad()
    def broadcast_params(self):
        """All rank broadcast updated local params."""
        # Broadcast linear layer weights to all other ranks. Kept as reference test.
        if self.dp_cp_params_list is None:
            return
        for i, params in enumerate(self.dp_cp_params_list):
            src_global_rank = torch.distributed.get_global_rank(self.pg_collection.dp_cp, i)
            for p in params:
                torch.distributed.broadcast(p, src_global_rank, self.pg_collection.dp_cp)
        if self.expt_dp_params_list is None:
            return
        for i, params in enumerate(self.expt_dp_params_list):
            src_global_rank = torch.distributed.get_global_rank(self.pg_collection.expt_dp, i)
            for p in params:
                torch.distributed.broadcast(p, src_global_rank, self.pg_collection.expt_dp)

    @torch.no_grad()
    def get_grad_norm(self):
        # similar to dist opt, always aggregate globally
        grads_for_norm = []
        for optimizer in self.chained_optimizers:
            grads_for_norm += optimizer.get_grads_for_grad_norm()
        grad_norm = get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=None)
        return grad_norm

    def has_grad_norm_group(self, grad_norm_group: str) -> bool:
        """Whether any global rank owns params for a registered grad-norm group.

        Overrides ChainedOptimizer to use a single global all-reduce (group=None),
        matching the scope of get_grad_norm and _get_grad_norm_for_group which also
        reduce globally. All LayerWise grad-stats reductions are global (identical to
        DistributedOptimizer's pattern), so the existence check must be too — using
        a per-sub-optimizer group here would create a collective mismatch.
        """
        _validate_grad_norm_group(grad_norm_group)
        if getattr(self, '_has_grad_norm_group_cache', None) is None:
            self._has_grad_norm_group_cache = {}
        cache = self._has_grad_norm_group_cache
        if grad_norm_group not in cache:
            local = False
            for optimizer in self.chained_optimizers:
                for param in optimizer.get_parameters():
                    param_grad_norm_group = _get_param_grad_norm_group(param)
                    if param_grad_norm_group is None:
                        continue
                    _validate_grad_norm_group(param_grad_norm_group)
                    local = local or param_grad_norm_group == grad_norm_group
            flag = torch.tensor([1 if local else 0], dtype=torch.int, device='cuda')
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=None)
            cache[grad_norm_group] = bool(flag.item() > 0)
        return cache[grad_norm_group]

    @torch.no_grad()
    def _get_grad_norm_for_group(self, grad_norm_group: str):
        # similar to dist opt, always aggregate globally
        grads_for_norm = []
        for optimizer in self.chained_optimizers:
            grads_for_norm += optimizer.get_grads_for_grad_norm(grad_norm_group)
        grad_norm = get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=None)
        return grad_norm

    @torch.no_grad()
    def count_zeros(self):
        params = []
        for optimizer in self.chained_optimizers:
            params += optimizer.get_parameters()
        return count_zeros_fp32(
            params,
            grad_stats_parallel_group=None,
            use_decoupled_grad=self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8,
        )

    def start_param_sync_for_bucket_group_subset(self) -> None:
        """Trigger ``start_param_sync`` on LayerWise-managed bucket groups only.

        Walks each model chunk's dense + expert-parallel bucket groups and
        skips any group not managed by LayerWise, so a sibling
        :class:`DistributedOptimizer`'s own ``start_param_sync`` call does not
        double-sync the same buckets. Uses
        :meth:`DistributedDataParallel._start_bucket_group_param_sync` so FP8
        post-all-gather processing (and MXFP8 copy) still runs.
        """
        for model_chunk in self.model_chunks:
            for bucket_group in (
                model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups
            ):
                if bucket_group.buckets and _bucket_is_managed_by_layer_wise_optimizer(
                    bucket_group.buckets[0]
                ):
                    model_chunk._start_bucket_group_param_sync(bucket_group, force_sync=False)

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step then all-gather LayerWise-managed param buffers.

        Placed on ``step_with_ready_grads`` (not ``step``) so the param sync also
        runs when this optimizer is a child of an outer ``ChainedOptimizer``,
        which calls ``step_with_ready_grads`` directly on each child and bypasses
        ``step``.
        """
        success = super().step_with_ready_grads()

        # All-gather updated params. If overlap_param_gather is True, the all-gather
        # is deferred to the forward pre-hooks via DDP bucket infrastructure.
        if not self.overlap_param_gather:
            if self.use_buffer_param_sync:
                # Model params are views into the DDP param buffer
                # (ddp_config.use_distributed_optimizer=True). The optimizer step
                # already copied updated fp32 main params → bf16 model params (=
                # buffer views), so the buffer is up-to-date. Trigger the standard
                # buffer all-gather, but only for LayerWise-managed bucket groups
                # so a sibling DistributedOptimizer's own ``start_param_sync`` call
                # is not duplicated for the same buckets.
                self.start_param_sync_for_bucket_group_subset()
            else:
                self.allgather_params()

        return success

    # TODO(deyuf): need to improve dist checkpointing design to properly handle this
    # fp32_from_fp16_params is list, each sub list could be empty if group is empty
    # this breaks dist checkpointing assumption since extract_sharded_base drop list structure
    # for now, we convert it to dict with index as key and convert back in load_state_dict
    def load_state_dict(self, state_dict):
        if len(self.chained_optimizers) == 1:
            wrapped_state_dict = {1: state_dict}
        else:
            wrapped_state_dict = state_dict
        for sd in wrapped_state_dict.values():
            if 'fp32_from_fp16_params' in sd and isinstance(sd['fp32_from_fp16_params'], dict):
                logger.info('[layerwise] converting fp32_from_fp16_params from dict to list')
                sd['fp32_from_fp16_params'] = [
                    v for k, v in sorted(sd['fp32_from_fp16_params'].items())
                ]
        super().load_state_dict(state_dict)

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False, **kwargs
    ):
        """
        Sharded state dict for torch_dist format checkpointing.
        For fixed DP usage only, set replica_id to 0 for all ShardedTensor.
        """
        sharded_state_dict = super().sharded_state_dict(
            model_sharded_state_dict, is_loading, **kwargs
        )

        # for fixed DP usage only
        for sh_base in nested_values(sharded_state_dict):
            if hasattr(sh_base, 'replica_id'):
                assert (
                    isinstance(sh_base.replica_id, int) or len(sh_base.replica_id) == 3
                ), f'Expected replica_id as int or (PP, TP, DP), got: {sh_base}'
                sh_base.replica_id = (
                    0 if isinstance(sh_base.replica_id, int) else (*sh_base.replica_id[:2], 0)
                )

        # later code assume list but chained optimizer fallback to non-list if there's only one
        if len(self.chained_optimizers) == 1:
            wrapped_sharded_state_dict = {1: sharded_state_dict}
        else:
            wrapped_sharded_state_dict = sharded_state_dict

        # Adjust dict rank 0 output correct global metadata into common_dict
        for sd in wrapped_sharded_state_dict.values():
            # wrap empty containers into LocalNonpersistentObject so it won't be saved/loaded
            # params is already wrapped, we only need to handle fp32_from_fp16_params and state
            # more details in load_state_dict comment
            if 'fp32_from_fp16_params' in sd:
                sd['fp32_from_fp16_params'][:] = [
                    group if group else LocalNonpersistentObject(group)
                    for group in sd['fp32_from_fp16_params']
                ]
                sd['fp32_from_fp16_params'] = {
                    i: v for i, v in enumerate(sd['fp32_from_fp16_params'])
                }
            # state is a single dict and will be empty if optimizer is fully empty
            if not sd['optimizer']['state']:
                sd['optimizer']['state'] = LocalNonpersistentObject(sd['optimizer']['state'])
            # group keys(e.g. 'step') might be missing or not updated
            for i, group in enumerate(sd['optimizer']['param_groups']):
                # keep local param tensor so we only gather metadata
                local_params = group.pop('params')
                # save whether this group is empty, so we can use non-empty rank for metadata
                group['params'] = bool(local_params.unwrap())
                all_rank_groups = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(all_rank_groups, group)
                # find first non-empty group if it exists
                nonempty_rank_group = next((g for g in all_rank_groups if g['params']), group)
                nonempty_rank_group['params'] = local_params
                sd['optimizer']['param_groups'][i] = nonempty_rank_group
        return sharded_state_dict

    def save_state_dict_to_file(self, filename: str) -> None:
        """Save the parameter state of the optimizer. For torch format only.
        Args:
            filename: The filename to save the parameter state.
        """
        torch.save(super().state_dict(), filename)

    def load_state_dict_from_file(self, filename: str) -> None:
        """Load the parameter state of the optimizer. For torch format only."""
        super().load_state_dict(torch.load(filename))
