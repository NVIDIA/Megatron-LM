# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedStateDict
from megatron.core.distributed.param_and_grad_buffer import group_params_for_buffers
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, get_pg_size, log_single_rank

from .clip_grads import count_zeros_fp32, get_grad_norm_fp32
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig
from .param_layout import (
    FullParamLayout,
    PerBufferParamLayout,
    pad_bucket_end,
    pad_param_start,
    pad_to_divisor,
)

logger = logging.getLogger(__name__)


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
    def tag_params(model_chunks, eopt_name, config_overrides=None):
        """Tag each trainable parameter with ``use_layerwise_distributed_optimizer``.

        Parameters managed by the emerging optimizer (e.g. Muon for 2D weights) are
        tagged ``True``; parameters routed to a fallback optimizer (e.g. Adam for
        biases, norms, embeddings) are tagged ``False``.  The tag is read by
        :func:`group_params_for_buffers` to separate them into distinct buffers.

        Must be called before DDP construction so that buffer grouping reflects
        the optimizer routing.

        Args:
            model_chunks: Model chunks whose parameters will be tagged.
            eopt_name: Name of the emerging optimizer (e.g. ``'muon'``).
            config_overrides: Optimizer config overrides (same dict passed to
                :func:`_get_megatron_emerging_optimizer`).
        """
        from .emerging_optimizers import _EMERGING_OPTIMIZERS

        all_overrides = dict(config_overrides or {})
        if eopt_name in _EMERGING_OPTIMIZERS:
            all_overrides.update(_EMERGING_OPTIMIZERS[eopt_name].default_param_overrides)

        for model_chunk in model_chunks:
            for name, param in model_chunk.named_parameters():
                if not param.requires_grad:
                    continue
                opt_name = eopt_name
                for param_key, override in all_overrides.items():
                    if param_key.matches(param, name) and 'optimizer' in override:
                        opt_name = override['optimizer']
                        break
                param.use_layerwise_distributed_optimizer = opt_name in _EMERGING_OPTIMIZERS

    @staticmethod
    def _shard_divisor(data_parallel_world_size: int, ddp_config) -> int:
        """Per-shard alignment divisor.

        Guarantees that ``dp_size * shard_size`` satisfies bucket-end alignment
        and that every shard start is 64-element aligned (required by
        :func:`pad_param_start`).
        """
        dp_size = data_parallel_world_size
        if ddp_config.pad_buckets_for_high_nccl_busbw:
            bucket_divisor = math.lcm(dp_size, 128, 2**16)
        else:
            bucket_divisor = math.lcm(dp_size, 128)
        return math.lcm(64, bucket_divisor // dp_size)

    @staticmethod
    def _compute_per_buffer_param_layout(
        params: List[torch.nn.Parameter],
        bucket_size: Optional[int],
        data_parallel_world_size: int,
        ddp_config,
        param_indices: Optional[List[int]] = None,
    ) -> 'PerBufferParamLayout':
        """Compute parameter layout with shard-aligned buckets via size-matching.

        Assigns parameters to ``dp_size`` equal-sized shards within each bucket
        so that no parameter is ever split across a shard boundary.

        **Algorithm** (operates in reverse model / backprop order):

        1. Separate shared-embedding parameters (isolated buckets, emitted first).
        2. Pool the remaining parameters in backprop order, indexed by numel.
        3. Pop the next unassigned parameter (size *S*) and assign it to shard 0.
        4. For shards 1 … ``dp_size - 1``, assign the next unassigned parameter
           of size *S* (also in backprop order).  If none is available, insert
           padding of size *S*.  Every shard grows by exactly *S* elements, so
           all shards stay the same size.
        5. When the bucket total reaches *bucket_size*, finalise the bucket
           (pad shard size to :meth:`_shard_divisor`) and start a new one.
        6. Repeat from 3 until all parameters are assigned.

        Because repeated layers produce many parameters of the same shape,
        size-matching naturally keeps whole parameters together without any
        name-parsing heuristic.  For a model with *N* identical transformer
        layers, padding overhead is zero.

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
        shard_div = LayerWiseDistributedOptimizer._shard_divisor(dp_size, ddp_config)

        # -- 0. Separate shared-embedding params. -------------------------
        shared_emb_params: List[torch.nn.Parameter] = []
        regular_params: List[torch.nn.Parameter] = []
        total_param_numel = 0
        for p in params:
            total_param_numel += p.data.nelement()
            if getattr(p, 'shared_embedding', False):
                shared_emb_params.append(p)
            else:
                regular_params.append(p)

        # -- 1. Build backprop-order pool & per-size index. ---------------
        pool = list(reversed(regular_params))
        assigned: set[int] = set()  # id(param) of assigned params

        size_groups: Dict[int, List[torch.nn.Parameter]] = defaultdict(list)
        for param in pool:
            size_groups[param.data.nelement()].append(param)
        size_cursors: Dict[int, int] = defaultdict(int)

        overall_cursor = 0

        def _next_unassigned() -> Optional[torch.nn.Parameter]:
            nonlocal overall_cursor
            while overall_cursor < len(pool):
                if id(pool[overall_cursor]) not in assigned:
                    return pool[overall_cursor]
                overall_cursor += 1
            return None

        def _next_with_size(S: int) -> Optional[torch.nn.Parameter]:
            """Next unassigned param of size *S* in backprop order."""
            group = size_groups[S]
            c = size_cursors[S]
            while c < len(group):
                if id(group[c]) not in assigned:
                    size_cursors[S] = c
                    return group[c]
                c += 1
            size_cursors[S] = c
            return None

        # -- 2. Output accumulators. --------------------------------------
        param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = {}
        bucket_indices: List[Tuple[int, int]] = []
        per_bucket_numel_unpadded: List[int] = []
        buf_cursor = 0  # write position in the contiguous buffer
        bucket_id = 0

        # -- 3. Emit isolated buckets for shared-embedding params. --------
        for param in reversed(shared_emb_params):
            pos = pad_param_start(buf_cursor)
            numel = param.data.nelement()
            param_index_map[param] = (pos, pos + numel, bucket_id)
            unpadded = (pos + numel) - buf_cursor
            bucket_end = pad_bucket_end(
                pos + numel, dp_size, ddp_config.pad_buckets_for_high_nccl_busbw
            )
            bucket_indices.append((buf_cursor, bucket_end))
            per_bucket_numel_unpadded.append(unpadded)
            buf_cursor = bucket_end
            bucket_id += 1

        # -- 4. Size-matching loop for regular params. --------------------
        #   Per-shard state: list of (param | None, numel) assignments.
        shard_assignments: List[List[Tuple[Optional[torch.nn.Parameter], int]]] = [
            [] for _ in range(dp_size)
        ]
        shard_pos = 0  # position within each shard (identical for all shards)
        bucket_unpadded = 0
        size_match_padding_numel = 0  # elements used for empty-shard-slot padding

        def _finalize_bucket() -> None:
            nonlocal buf_cursor, bucket_id, shard_assignments, shard_pos, bucket_unpadded
            if shard_pos == 0:
                return
            padded_shard_size = pad_to_divisor(shard_pos, shard_div)
            bucket_start = buf_cursor

            for shard_idx in range(dp_size):
                shard_start = bucket_start + shard_idx * padded_shard_size
                pos = shard_start
                for param, numel in shard_assignments[shard_idx]:
                    pos = pad_param_start(pos)
                    if param is not None:
                        param_index_map[param] = (pos, pos + numel, bucket_id)
                    pos += numel

            bucket_end = bucket_start + dp_size * padded_shard_size
            bucket_indices.append((bucket_start, bucket_end))
            per_bucket_numel_unpadded.append(bucket_unpadded)
            buf_cursor = bucket_end
            bucket_id += 1

            shard_assignments = [[] for _ in range(dp_size)]
            shard_pos = 0
            bucket_unpadded = 0

        while True:
            param = _next_unassigned()
            if param is None:
                break

            S = param.data.nelement()
            assigned.add(id(param))
            shard_assignments[0].append((param, S))
            bucket_unpadded += S

            for shard_idx in range(1, dp_size):
                match = _next_with_size(S)
                if match is not None:
                    assigned.add(id(match))
                    shard_assignments[shard_idx].append((match, S))
                    bucket_unpadded += S
                else:
                    shard_assignments[shard_idx].append((None, S))
                    size_match_padding_numel += S

            shard_pos = pad_param_start(shard_pos) + S

            if bucket_size is not None:
                bucket_total = dp_size * pad_to_divisor(shard_pos, shard_div)
                if bucket_total >= bucket_size:
                    _finalize_bucket()

        _finalize_bucket()

        # -- 5. Log padding overhead. ------------------------------------
        total_buffer_numel = bucket_indices[-1][1] if bucket_indices else 0
        total_padding = total_buffer_numel - total_param_numel
        alignment_and_shard_end_padding = total_padding - size_match_padding_numel
        log_single_rank(
            logger,
            logging.INFO,
            f"Layerwise param layout: {len(params)} params, "
            f"{len(bucket_indices)} buckets, "
            f"dp_size={dp_size}, "
            f"total_param_numel={total_param_numel}, "
            f"total_buffer_numel={total_buffer_numel}, "
            f"total_padding={total_padding} "
            f"(size_match={size_match_padding_numel}, "
            f"alignment+shard_end={alignment_and_shard_end_padding}), "
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
        """Compute parameter layouts for all buffer groups with shard-aligned buckets.

        Groups parameters by :class:`BufferKey` via :func:`group_params_for_buffers`,
        then delegates to :meth:`_compute_per_buffer_param_layout` (shard-aligned) for
        buffers with ``use_layerwise_distributed_optimizer=True``, or to
        :meth:`DistributedOptimizer._compute_per_buffer_param_layout` (standard) for
        the remaining buffers.

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
        from .distrib_optimizer import DistributedOptimizer

        buffer_groups = group_params_for_buffers(params, ddp_config.grad_reduce_in_fp32)
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

            if buffer_key.use_layerwise_distributed_optimizer:
                layout = LayerWiseDistributedOptimizer._compute_per_buffer_param_layout(
                    group_params, bucket_size, dp_world_size, ddp_config, param_indices
                )
            else:
                layout = DistributedOptimizer._compute_per_buffer_param_layout(
                    group_params, bucket_size, dp_world_size, ddp_config, param_indices
                )
            layouts[buffer_key] = layout
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

        full_param_layouts = None
        if model_chunks is not None:
            full_param_layouts = [
                chunk.full_param_layout
                for chunk in model_chunks
                if hasattr(chunk, 'full_param_layout') and chunk.full_param_layout is not None
            ] or None
        self.shard_params(optimizers, full_param_layouts)

        # When a full_param_layout is available, ddp_config.use_distributed_optimizer
        # is True and model params are views into the DDP param buffer.  After the
        # optimizer step copies updated fp32 main params → bf16 model params, the
        # buffer is already up-to-date in-place.  We can use DDP's buffer-based
        # all-gather (start_param_sync) instead of the flatten/unflatten allgather_params
        # path.
        self.use_buffer_param_sync = full_param_layouts is not None
        self.model_chunks = model_chunks

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

        super().__init__(optimizers)

        # TODO(kunlun, deyuf): potential future perf optimization
        # since allreduce is unchanged and handled by megatron DDP, they're already in
        # contiguous gbuf. So instead of shard param by layer randomly, we can shard by
        # buf range but keep some "extras" to keep boundary weight not sharded.
        # This way each rank do some duplicated work but allgather_v is no longer needed
        # All current distopt optimization can also be potentially applied

    def shard_params(self, optimizers, full_param_layouts=None):
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
            self._shard_params_ping_pong(optimizers, dp_cp_size, expt_dp_size)

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
                dp_size = expt_dp_size if buffer_key.is_expert_parallel else dp_cp_size
                for param, (start, end, bucket_id) in layout.param_index_map.items():
                    bstart, bend = layout.bucket_indices[bucket_id]
                    shard_size = (bend - bstart) // dp_size
                    shard_idx = (start - bstart) // shard_size
                    param_to_shard[param] = shard_idx

        # Collect all param groups and assign params to per-rank lists.
        param_groups = []
        for optimizer in optimizers:
            param_groups += optimizer.param_groups
        param_groups_this_rank = [[] for _ in param_groups]

        for group_index, group in enumerate(param_groups):
            is_expert = group.get("is_expert_parallel", False)
            local_rank = expt_dp_rank if is_expert else dp_cp_rank
            params_list = self.expt_dp_params_list if is_expert else self.dp_cp_params_list

            for p in group["params"]:
                assert p in param_to_shard, (
                    f"Optimizer param (shape={tuple(p.shape)}, numel={p.numel()}) not found "
                    f"in any param layout. Ensure all optimizer params are included in the "
                    f"full_param_layout passed to DDP."
                )
                shard_idx = param_to_shard[p]
                params_list[shard_idx].append(p)
                if shard_idx == local_rank:
                    param_groups_this_rank[group_index].append(p)

        # Now we modify the group to only handle local params.
        for group, local_params in zip(param_groups, param_groups_this_rank):
            group["params"] = local_params

        # Simplify when expt_dp group size is 1 or expert parallel is off.
        if expt_dp_size == 1 or len(self.expt_dp_params_list[0]) == 0:
            self.expt_dp_params_list = None

    def _shard_params_ping_pong(self, optimizers, dp_cp_size, expt_dp_size):
        """Legacy ping-pong-by-numel shard assignment (no layout available).

        Legacy: this method is a fallback for when no ``full_param_layout``
        is provided.  Once all call sites supply a layout, this can be removed
        in favor of :meth:`_shard_params_from_layout`.

        List of parameters are sorted by numel and assigned to ranks in ping-pong style.
        Example of 4 ranks and 10 parameters p0-p9 after sorting, then dp_cp_params_list
        will be [[p0, p7, p8], [p1, p6, p9], [p2, p5], [p3, p4]].
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

        # Sort param in all groups by param numel and assign to each rank evenly.
        param_list = []
        for group_index, group in enumerate(param_groups):
            for p in group["params"]:
                param_list.append((p, group_index))
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
                    bucket_params_list = [[] for _ in range(get_pg_size(self.pg_collection.dp_cp))]
                    for bucket_list, full_params_list in zip(
                        bucket_params_list, self.dp_cp_params_list
                    ):
                        for param in full_params_list:
                            if param in bucket.params:
                                bucket_list.append(param)
                    bucket.set_layerwise_params_list(bucket_params_list)
            # Do the same for expert parallel bucket groups.
            for group in model_chunk.expert_parallel_bucket_groups:
                for bucket in group.buckets:
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
        """

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
        if self.dp_cp_params_list:
            _allgather_helper(self.dp_cp_params_list, self.pg_collection.dp_cp)
        if self.expt_dp_params_list:
            _allgather_helper(self.expt_dp_params_list, self.pg_collection.expt_dp)

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
            grads_for_norm += optimizer.get_main_grads_for_grad_norm()
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

    @torch.no_grad()
    def step(self):  # type: ignore[no-untyped-def]
        """step function for layer-wise optimizer."""
        update_successful, grad_norm, num_zeros_in_grad = super().step()

        # All-gather updated params. If overlap_param_gather is True, the all-gather
        # is deferred to the forward pre-hooks via DDP bucket infrastructure.
        if not self.overlap_param_gather:
            if self.use_buffer_param_sync:
                # Model params are views into the DDP param buffer
                # (ddp_config.use_distributed_optimizer=True).  The optimizer step
                # already copied updated fp32 main params → bf16 model params (=
                # buffer views), so the buffer is up-to-date.  Trigger the standard
                # buffer all-gather.
                for model_chunk in self.model_chunks:
                    model_chunk.start_param_sync(force_sync=True)
            else:
                self.allgather_params()

        return update_successful, grad_norm, num_zeros_in_grad

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
