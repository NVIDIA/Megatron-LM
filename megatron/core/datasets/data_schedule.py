# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

import enum
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import torch

from megatron.core.context_parallel import get_batches_on_this_cp_rank
from megatron.core.context_parallel.groups import get_valid_dynamic_context_parallel_group_sizes
from megatron.core.context_parallel_layout import ThdCpRoute, finalize_packed_seq_params
from megatron.core.datasets.data_schedule_utils import (
    align_sample_id_groups,
    broadcast_scalars,
    broadcast_tensor,
    build_packed_microbatches,
    create_data_iterator,
    get_batch_and_global_seqlens,
    get_packed_sequence_alignment,
    get_packed_sequence_length,
    next_hdp_group_packing_aware,
    pad_packed_batch_before_cp_slice,
    reroute_samples_to_dcp_ranks,
)
from megatron.core.full_cuda_graph import FULL_CUDA_GRAPH_STATIC_METADATA_KEY
from megatron.core.packed_seq_params import PackedSeqParams, resolve_thd_tail_padding_policy
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.multi_token_prediction import (
    mtp_on_this_rank as mtp_on_this_pipeline_rank,
)

try:
    # Register the TE CUDA kernels
    import transformer_engine  # pylint: disable=unused-import

    # Alias the PyTorch wrapper so we can call tex.* APIs
    import transformer_engine_torch as tex
except ImportError:
    # TE isn't installed or the torch wrapper is missing
    tex = None


# Sentinel key marking a batch that was already canonicalized to the THD
# full-iteration CUDA graph static-input contract outside the captured region.
THD_FULL_ITERATION_STATIC_BATCH_KEY = 'thd_full_iteration_static_batch'
_THD_FULL_ITERATION_CP_METADATA_KEY = 'thd_full_iteration_cp_metadata'
_THD_FULL_ITERATION_PACKED_SEQ_PARAMS_KEY = '_thd_full_iteration_packed_seq_params'


def _materialize_thd_cp_layout_signatures(
    packed_seq_params: PackedSeqParams,
    host_cu_q: Optional[List[int]],
    host_cu_kv: Optional[List[int]],
):
    """Return exact Q/KV CP boundaries with at most one normal host transfer."""
    if host_cu_q is not None and host_cu_kv is not None:
        return tuple(host_cu_q), tuple(host_cu_kv)

    cu_q = (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )
    cu_kv = (
        packed_seq_params.cu_seqlens_kv_padded
        if packed_seq_params.cu_seqlens_kv_padded is not None
        else packed_seq_params.cu_seqlens_kv
    )
    assert cu_q is not None and cu_kv is not None
    if cu_q is cu_kv:
        values = tuple(cu_q.detach().to(device='cpu', dtype=torch.int64).tolist())
        return values, values
    if cu_q.device != cu_kv.device:
        return (
            tuple(cu_q.detach().to(device='cpu', dtype=torch.int64).tolist()),
            tuple(cu_kv.detach().to(device='cpu', dtype=torch.int64).tolist()),
        )
    q_numel = cu_q.numel()
    values = torch.cat((cu_q.detach(), cu_kv.detach())).to(device='cpu', dtype=torch.int64).tolist()
    return tuple(values[:q_numel]), tuple(values[q_numel:])


def _build_thd_full_iteration_cp_metadata(packed_seq_params: PackedSeqParams) -> Dict[str, Any]:
    """Serialize finalized CP tensor inputs into containers understood by StaticBufferLoader."""
    route = packed_seq_params.cp_partition_route
    return {
        'cp_group': packed_seq_params.cp_group,
        'zigzag_index': None if route is None else route.zigzag_index,
        'contiguous_index': None if route is None else route.contiguous_index,
    }


def _build_thd_full_iteration_cp_layout_signature(
    packed_seq_params: PackedSeqParams,
) -> Optional[Dict[str, Any]]:
    """Return the exact packed geometry baked into CP>1 Python route metadata."""
    cp_group = packed_seq_params.cp_group
    if cp_group is None or cp_group.size() <= 1:
        return None
    route = packed_seq_params.cp_partition_route
    assert route is not None, "CP>1 THD static batches require a prebuilt partition route."

    # PRs layered on top of the generic THD graph support may attach compact
    # host boundaries for consumers such as DSA. Keep them in the immutable
    # metadata certificate without making this module depend on those consumers.
    host_cu_q = getattr(packed_seq_params, 'thd_cp_host_cu_seqlens_q', None)
    host_cu_kv = getattr(packed_seq_params, 'thd_cp_host_cu_seqlens_kv', None)
    cp_cu_q, cp_cu_kv = _materialize_thd_cp_layout_signatures(
        packed_seq_params, host_cu_q, host_cu_kv
    )
    return {
        'version': 1,
        'cp_size': cp_group.size(),
        'cp_rank': cp_group.rank(),
        'cp_partition_mode': packed_seq_params.cp_partition_mode,
        'cp_cu_seqlens_q': cp_cu_q,
        'cp_cu_seqlens_kv': cp_cu_kv,
        'zigzag_index_is_none': route.zigzag_index is None,
        'zigzag_split_sizes': tuple(route.zigzag_split_sizes),
        'contiguous_index_is_none': route.contiguous_index is None,
        'contiguous_split_sizes': tuple(route.contiguous_split_sizes),
        'thd_cp_host_cu_seqlens_q': None if host_cu_q is None else tuple(host_cu_q),
        'thd_cp_host_cu_seqlens_kv': None if host_cu_kv is None else tuple(host_cu_kv),
    }


def _restore_thd_full_iteration_cp_route(
    tensor_metadata: Dict[str, Any], layout_metadata: Optional[Dict[str, Any]]
):
    """Rebuild the lightweight route wrapper around static-buffer payloads."""
    if layout_metadata is None:
        return None
    for layout in ('zigzag', 'contiguous'):
        index = tensor_metadata[f'{layout}_index']
        if (index is None) != layout_metadata[f'{layout}_index_is_none']:
            raise RuntimeError(
                f"THD full-iteration {layout} route index presence does not match its "
                "graph-static metadata certificate."
            )
    return ThdCpRoute(
        zigzag_index=tensor_metadata['zigzag_index'],
        zigzag_split_sizes=list(layout_metadata['zigzag_split_sizes']),
        contiguous_index=tensor_metadata['contiguous_index'],
        contiguous_split_sizes=list(layout_metadata['contiguous_split_sizes']),
    )


def _unpack_thd_full_iteration_static_batch(batch: Dict[str, Any]):
    """Rebuild ``get_batch`` outputs from a pre-canonicalized static batch.

    Inside a captured full-iteration CUDA graph, batch tensors must be read
    from fixed addresses without shape discovery, host synchronization, or
    collective communication, so this only re-wraps the static tensors
    prepared by ``prepare_thd_static_batch_for_full_iteration_cuda_graph``.
    """
    layout_metadata = batch.get(FULL_CUDA_GRAPH_STATIC_METADATA_KEY, {}).get('thd_cp_layout')
    # Only CP>1 geometry is immutable across a static-buffer slot.  At CP1,
    # packed boundaries may change between replays, so do not retain consumer
    # caches attached to a PackedSeqParams instance from an earlier batch.
    cp_metadata = batch[_THD_FULL_ITERATION_CP_METADATA_KEY]
    packed_seq_params = (
        cp_metadata.get(_THD_FULL_ITERATION_PACKED_SEQ_PARAMS_KEY)
        if layout_metadata is not None
        else None
    )
    if packed_seq_params is None:
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=batch['cu_seqlens'],
            cu_seqlens_kv=batch['cu_seqlens'],
            cu_seqlens_q_padded=batch['cu_seqlens_padded'],
            cu_seqlens_kv_padded=batch['cu_seqlens_padded'],
            max_seqlen_q=batch['max_seqlen'],
            max_seqlen_kv=batch['max_seqlen'],
            local_cp_size=None,
            cp_group=cp_metadata['cp_group'],
            cp_partition_mode=batch['cp_partition_mode'],
            pad_between_seqs=batch['pad_between_seqs'],
            cp_partition_route=_restore_thd_full_iteration_cp_route(cp_metadata, layout_metadata),
        )
        for name in ('thd_cp_host_cu_seqlens_q', 'thd_cp_host_cu_seqlens_kv'):
            value = None if layout_metadata is None else layout_metadata.get(name)
            if value is not None:
                setattr(packed_seq_params, name, list(value))
        # ``pretrain_{gpt,hybrid}.get_batch`` finalizes every ordinary THD
        # batch. This marker makes that operation idempotent for metadata that
        # was already finalized outside capture.
        packed_seq_params._full_cuda_graph_cp_metadata_prebuilt = True
        if layout_metadata is not None:
            # The CP>1 static buffer owns this object for the lifetime of the
            # graph. In addition to preserving route tensor addresses, this
            # lets consumers populate per-layout caches during eager warmup and
            # reuse them during capture without rebuilding host-derived tensors.
            # StaticBufferLoader returns a shallow top-level copy, while this
            # nested metadata dict is shared with the captured slot; cache here
            # so the object survives the next replay without exposing a mutable
            # top-level entry to pipeline-stage batch tailoring.
            cp_metadata[_THD_FULL_ITERATION_PACKED_SEQ_PARAMS_KEY] = packed_seq_params
    return (
        batch['tokens'],
        batch['labels'],
        batch['loss_mask'],
        None,
        batch['position_ids'],
        packed_seq_params,
        batch['padding_mask'],
    )


def prepare_thd_static_batch_for_full_iteration_cuda_graph(
    data_iterator,
    vp_stage: Optional[int] = None,
    *,
    config,
    vpp_size: Optional[int] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> Dict[str, Any]:
    """Canonicalize one packed THD microbatch outside the full-iteration graph.

    Runs the eager THD batch path (padding-mask construction, CP slicing, TP
    broadcast and static-shape padding) so every tensor reaches the fixed
    shapes required for CUDA graph capture and replay: token-like tensors at
    the per-rank token capacity ``max_seqlen_per_dp_cp_rank`` and cu_seqlens
    tensors at ``thd_max_packed_sequences + 1`` entries, with
    ``max_seqlen_q/kv`` pinned to the static config upper bound. This function
    issues TP broadcasts, so all ranks must call it in the same order.

    Returns a dict suitable for ``StaticBufferLoader``. The captured
    ``get_batch_on_this_rank_for_sequence_packing`` path recognizes it via
    ``THD_FULL_ITERATION_STATIC_BATCH_KEY`` and reads the static tensors
    without further shape discovery.
    """
    # Static-shape requirements (max_seqlen_per_dp_cp_rank, thd_max_packed_sequences,
    # no dynamic CP) are validated once at TransformerConfig construction time.
    assert config is not None, "THD full-iteration batch preparation requires a config."
    assert getattr(config, 'cuda_graph_impl', 'none') == 'full_iteration', (
        "prepare_thd_static_batch_for_full_iteration_cuda_graph requires "
        f"cuda_graph_impl='full_iteration', got {getattr(config, 'cuda_graph_impl', 'none')}."
    )
    max_seqlen_per_dp_cp_rank = config.max_seqlen_per_dp_cp_rank
    thd_max_packed_sequences = config.thd_max_packed_sequences

    # Canonicalization mutates the batch dict in place (padding-mask insertion,
    # 2D views, sanitized values). Work on a shallow copy so callers that replay
    # the same raw batch — e.g. the PagedStashRunner overflow fallback — can
    # canonicalize it again.
    if data_iterator is not None:
        raw_batch = next(data_iterator)
        if isinstance(raw_batch, dict):
            raw_batch = dict(raw_batch)
        data_iterator = iter([raw_batch])

    tokens, labels, loss_mask, _, position_ids, packed_seq_params, padding_mask = (
        get_batch_on_this_rank_for_sequence_packing(
            data_iterator,
            vpp_size=vpp_size,
            mtp_on_this_rank=mtp_on_this_pipeline_rank(
                config, ignore_virtual=False, vp_stage=vp_stage
            ),
            vp_stage=vp_stage,
            dynamic_cp=False,
            pg_collection=pg_collection,
            config=config,
        )
    )

    # CP route construction reads packed boundaries on the host and creates
    # Python split metadata. Do it before StaticBufferLoader hands the batch to
    # the captured forward, then preserve the resulting tensors and lists in
    # the static slot. The exact CP>1 geometry is part of the graph contract;
    # StaticBufferLoader rejects a later batch that would invalidate it.
    finalize_packed_seq_params(packed_seq_params)
    cp_metadata = _build_thd_full_iteration_cp_metadata(packed_seq_params)
    cp_layout_signature = _build_thd_full_iteration_cp_layout_signature(packed_seq_params)

    # Enforce the static-input contract before the tensors reach graph buffers.
    expected_entries = thd_max_packed_sequences + 1
    for name, cu in (
        ('cu_seqlens', packed_seq_params.cu_seqlens_q),
        ('cu_seqlens_padded', packed_seq_params.cu_seqlens_q_padded),
    ):
        assert cu is not None and cu.numel() == expected_entries, (
            f"THD full-iteration static batch expects {name} with {expected_entries} entries "
            f"(thd_max_packed_sequences + 1), got "
            f"{None if cu is None else cu.numel()}."
        )
    for name, t in (
        ('tokens', tokens),
        ('labels', labels),
        ('loss_mask', loss_mask),
        ('position_ids', position_ids),
        ('padding_mask', padding_mask),
    ):
        assert t is None or t.shape[-1] == max_seqlen_per_dp_cp_rank, (
            f"THD full-iteration static batch expects {name} padded to the per-rank token "
            f"capacity ({max_seqlen_per_dp_cp_rank}), got {t.shape[-1]}."
        )
    assert isinstance(packed_seq_params.max_seqlen_q, int), (
        "THD full-iteration static batch expects a Python int max_seqlen (static upper bound), "
        f"got {type(packed_seq_params.max_seqlen_q)}."
    )
    assert packed_seq_params.pad_between_seqs is True, (
        "THD full-iteration static batch requires the batch-independent "
        "pad_between_seqs=True contract, "
        f"got {packed_seq_params.pad_between_seqs!r}."
    )

    static_batch = {
        THD_FULL_ITERATION_STATIC_BATCH_KEY: True,
        'tokens': tokens,
        'labels': labels,
        'loss_mask': loss_mask,
        'position_ids': position_ids,
        'padding_mask': padding_mask,
        'cu_seqlens': packed_seq_params.cu_seqlens_q,
        'cu_seqlens_padded': packed_seq_params.cu_seqlens_q_padded,
        'max_seqlen': packed_seq_params.max_seqlen_q,
        'cp_partition_mode': packed_seq_params.cp_partition_mode,
        'pad_between_seqs': packed_seq_params.pad_between_seqs,
        _THD_FULL_ITERATION_CP_METADATA_KEY: cp_metadata,
    }
    if cp_layout_signature is not None:
        static_batch[FULL_CUDA_GRAPH_STATIC_METADATA_KEY] = {'thd_cp_layout': cp_layout_signature}
    return static_batch


def _build_thd_padding_mask(
    cu_seqlens: torch.Tensor, cu_seqlens_padded: torch.Tensor
) -> torch.Tensor:
    """Build a 1D THD padding mask from scheduler sequence metadata."""
    assert cu_seqlens.dim() == 1
    assert cu_seqlens_padded.dim() == 1
    assert cu_seqlens.numel() == cu_seqlens_padded.numel()

    total_tokens = int(cu_seqlens_padded[-1].item())
    if total_tokens == 0:
        return torch.empty((0,), dtype=torch.bool, device=cu_seqlens.device)

    num_sequences = cu_seqlens.numel() - 1
    if num_sequences <= 0:
        return torch.ones((total_tokens,), dtype=torch.bool, device=cu_seqlens.device)

    positions = torch.arange(
        total_tokens, dtype=cu_seqlens_padded.dtype, device=cu_seqlens_padded.device
    )
    seq_indices = torch.searchsorted(cu_seqlens_padded[1:].contiguous(), positions, right=True)

    valid_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).clamp(min=0)
    valid_ends = cu_seqlens_padded[:-1] + valid_lengths
    return positions >= valid_ends[seq_indices]


def _sanitize_thd_padding_values(batch: Dict[str, Any], padding_mask: torch.Tensor) -> None:
    """Replace padded token-like slots with safe neutral values in-place."""
    assert padding_mask.dim() == 1
    pad_values = {'tokens': 0, 'labels': 0, 'loss_mask': 0.0, 'position_ids': 0}
    for key, pad_value in pad_values.items():
        tensor = batch.get(key)
        if tensor is None:
            continue
        assert tensor.dim() == 1, f"{key} must be 1D before CP slicing, got {tensor.dim()}D"
        assert tensor.numel() == padding_mask.numel(), (
            f"{key} length ({tensor.numel()}) must match padding_mask length "
            f"({padding_mask.numel()}) before CP slicing."
        )
        batch[key] = tensor.masked_fill(padding_mask, pad_value)


def _refresh_zigzag_cp_certificate(
    certificate: Any, cu_seqlens_padded: torch.Tensor, cp_size: int
) -> torch.Tensor:
    """Re-validate the scheduler's zigzag MTP certificate after pre-slice padding.

    Padding before the CP slice can align a previously non-divisible physical
    layout, extend the final sequence and append a dummy tail sequence. A zero
    certificate ("inspected, not certifiable") no longer describes the padded
    layout, so it becomes unknown (-1). A positive certificate is capped by the
    padded layout's minimum non-empty half-chunk, or becomes zero if a non-empty
    physical sequence is not divisible by twice the CP size. Negative (unknown)
    values are kept. Runs on device, without a host synchronization.
    """
    lengths = cu_seqlens_padded.diff()
    certificate = torch.as_tensor(certificate, dtype=torch.int32, device=lengths.device)
    divisor = 2 * cp_size
    nonempty = lengths > 0
    divisible = torch.all(~nonempty | (lengths.remainder(divisor) == 0))
    no_limit = torch.iinfo(torch.int32).max
    min_chunk = (
        torch.where(nonempty, torch.div(lengths, divisor, rounding_mode='floor'), no_limit)
        .min()
        .to(torch.int32)
        if lengths.numel()
        else torch.tensor(no_limit, dtype=torch.int32, device=lengths.device)
    )
    positive = torch.where(
        divisible, torch.minimum(certificate, min_chunk), torch.zeros_like(certificate)
    )
    return torch.where(
        certificate > 0, positive, torch.where(certificate == 0, -1, certificate)
    ).to(torch.int32)


class BasePackingScheduler:
    """Base class for sequence packing schedulers."""

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank: int,
        cp_size: int,
        dp_size: int,
        microbatch_group_size_per_vp_stage: Optional[int],
        pad_sequences: bool = False,
        max_num_seqs: Optional[int] = None,
    ):
        """
        Args:
            max_seqlen_per_dp_cp_rank: The maximum sequence length per DPxCP rank.
            cp_size: The context parallel size.
            dp_size: The data parallel size.
            microbatch_group_size_per_vp_stage: The microbatch group size per virtual
            pipeline stage, only used when enabling VPP, otherwise None.
            pad_sequences: Account for per-sequence zigzag padding when packing.
            max_num_seqs: Optional cap on the number of real packed sequences
            per microbatch. This excludes any dummy sequence later appended for
            THD padding.
        """
        self.max_seqlen_per_dp_cp_rank = max_seqlen_per_dp_cp_rank
        self.cp_size = cp_size
        self.dp_size = dp_size
        self.microbatch_group_size_per_vp_stage = microbatch_group_size_per_vp_stage
        self.pad_sequences = pad_sequences
        self.max_num_seqs = max_num_seqs

    def get_required_sample_keys(self):
        """Return the required key of each batch."""
        raise NotImplementedError

    def get_groups_and_subsamples(self, sample_id_seqlens):
        """schedule the samples into groups"""
        raise NotImplementedError

    def run(
        self,
        data_iterator,
        num_microbatches,
        dp_group,
        tp_group,
        pp_group,
        dp_cp_group,
        dev,
        config,
    ):
        """
        Run the scheduler and return the new data_iterator.

        Args:
            data_iterator: The data iterator.
            num_microbatches: The number of microbatches to fetch.
            dp_group: Data parallel process group.
            tp_group: Tensor parallel process group.
            pp_group: Pipeline parallel process group.
            dp_cp_group: Data parallel + context parallel process group.
            dev: CUDA device.
            config: Model parallel config.

        Returns:
            new_data_iterator: The new data iterator (or list for VPP).
            num_micro_batches: Number of micro batches after scheduling.
            seqlen_sum_this_global_batch: Total tokens for FLOPs calculation.
            seqlen_squared_sum_this_global_batch: Sum of squared seqlens for FLOPs.
        """
        raise NotImplementedError


class DpBalancedScheduler(BasePackingScheduler):
    """Packs sequences in their original order until reaching the max limit of sequence length."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_len_all_ranks = self.max_seqlen_per_dp_cp_rank * self.cp_size
        self.is_dynamic_cp = False

    def get_required_sample_keys(self):
        """Return the required key of each batch."""
        return [
            "tokens",
            "labels",
            "loss_mask",
            "position_ids",
            "original_seq_len",  # Length of the original sequence length, should be a gpu tensor.
            "padded_seq_len",  # Length of the padded sequence length, should be a gpu tensor.
        ]

    def get_groups_and_subsamples(self, sample_id_seqlens):
        """
        Packs sequences in their original order until reaching the max limit of sequence length.
        """
        sample_id_groups = []
        packed_id_groups = []
        sum_seqlen = 0
        single_microbatch = []

        for i in range(len(sample_id_seqlens)):
            seq_len = get_packed_sequence_length(
                sample_id_seqlens[i][1], self.cp_size, self.pad_sequences
            )
            if seq_len > self.max_seq_len_all_ranks:
                raise ValueError("A padded sequence exceeds the static CP packing capacity")
            if sum_seqlen + seq_len <= self.max_seq_len_all_ranks and (
                self.max_num_seqs is None or len(single_microbatch) < self.max_num_seqs
            ):
                single_microbatch.append(i)
                sum_seqlen += seq_len
            else:
                packed_id_groups.append(single_microbatch)
                single_microbatch = [i]
                sum_seqlen = seq_len
        if len(single_microbatch) > 0:
            packed_id_groups.append(single_microbatch)

        # we want the number of packed sequences to be multiple of dp_size
        # so we move few samples from previous microbatch
        # to the end of the microbatches if needed
        num_packed_sequence = len(packed_id_groups)

        # when enabling vpp, we want the number of packed sequences to be
        # multiple of dp_size * microbatch_group_size_per_vp_stage
        multiple = self.dp_size * (
            self.microbatch_group_size_per_vp_stage
            if self.microbatch_group_size_per_vp_stage is not None
            else 1
        )
        if num_packed_sequence % multiple != 0:
            remainder = num_packed_sequence % multiple
            num_to_move = multiple - remainder
            i = num_packed_sequence - 1
            while num_to_move > 0:
                assert i >= 0, "Not enough samples to move"
                if len(packed_id_groups[i]) > 1:
                    seq_id = packed_id_groups[i].pop()
                    packed_id_groups.append([seq_id])
                    num_to_move -= 1
                else:
                    i -= 1

        num_micro_batches = int(len(packed_id_groups) / self.dp_size)
        for i in range(num_micro_batches):
            sample_id_groups.append([])
            for j in range(self.cp_size * self.dp_size):
                seq_id = int(i * self.dp_size + j / self.cp_size)
                sample_id_groups[i].append(packed_id_groups[seq_id])
        return sample_id_groups

    def run(
        self,
        data_iterator,
        num_microbatches: int,
        dp_group,
        tp_group,
        pp_group,
        dp_cp_group,
        dev: torch.device,
        config,
    ):
        """
        Run the complete scheduling pipeline.

        Every PP stage owns the packed dataset on TP rank zero. Each stage runs
        the same schedule locally, then keeps only the data fields required by
        its pipeline/VPP position. This avoids a PP metadata broadcast and lets
        arbitrary PP layouts (including middle-stage MTP) consume the same
        per-microbatch runtime CP metadata.

        Args:
            data_iterator: The data iterator.
            num_microbatches: The number of microbatches to fetch.
            dp_group: Data parallel process group.
            tp_group: Tensor parallel process group.
            pp_group: Pipeline parallel process group.
            dp_cp_group: Data parallel + context parallel process group.
            dev: CUDA device.
            config: Model parallel config.

        Returns:
            new_data_iterator: The new data iterator (or list for VPP).
            num_micro_batches: Number of micro batches after scheduling.
            seqlen_sum_this_global_batch: Total tokens for FLOPs calculation.
            seqlen_squared_sum_this_global_batch: Sum of squared seqlens for FLOPs.
        """

        is_first_pp = pp_group.rank() == 0
        is_last_pp = pp_group.rank() == pp_group.size() - 1
        vpp_size = config.virtual_pipeline_model_parallel_size or 1
        mtp_on_this_pp = mtp_on_this_pipeline_rank(
            layout=getattr(config, 'pipeline_model_parallel_layout', None),
            mtp_num_layers=getattr(config, 'mtp_num_layers', None),
            ignore_virtual=True,
            pp_group=pp_group,
            vp_size=vpp_size,
        )

        vpp_needs_data = None
        if vpp_size > 1:
            assert data_iterator is None or len(data_iterator) == vpp_size
            if data_iterator is not None:
                data_iterator = next(
                    (iterator for iterator in data_iterator if iterator is not None), None
                )

            vpp_needs_data = [False] * vpp_size
            if is_first_pp:
                vpp_needs_data[0] = True
            if is_last_pp:
                vpp_needs_data[-1] = True
            if mtp_on_this_pp:
                for vp_stage in range(vpp_size):
                    if mtp_on_this_pipeline_rank(
                        layout=getattr(config, 'pipeline_model_parallel_layout', None),
                        mtp_num_layers=getattr(config, 'mtp_num_layers', None),
                        ignore_virtual=False,
                        vp_stage=vp_stage,
                        pp_group=pp_group,
                        vp_size=vpp_size,
                    ):
                        vpp_needs_data[vp_stage] = True

        # Only TP rank 0 reads data, so a rank that runs out must tell the others
        # before any collective; otherwise they block in the broadcasts and gathers below.
        # Reducing over TP, then DPxCP, then PP reaches every rank of the grid.
        batch_list = None
        exhausted = torch.zeros(1, dtype=torch.int32, device=dev)
        if data_iterator is not None:
            try:
                batch_list = [next(data_iterator) for _ in range(num_microbatches)]
            except StopIteration:
                exhausted.fill_(1)
        for group in (tp_group, dp_cp_group, pp_group):
            torch.distributed.all_reduce(exhausted, op=torch.distributed.ReduceOp.MAX, group=group)
        if exhausted.item():
            raise StopIteration

        if data_iterator is not None:
            assert tp_group.rank() == 0, "Only TP rank 0 should have a packed data iterator"

            # Step 1: Gather global sequence lengths for the fetched batches
            batch, global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered = (
                get_batch_and_global_seqlens(iter(batch_list), num_microbatches, dp_group)
            )

            # Step 2: Check required sample keys
            for sample in batch:
                for key in self.get_required_sample_keys():
                    assert key in sample, f"Batch missing required key {key}"

            # Retain only fields consumed on this PP rank. Metadata used by the
            # scheduler remains on every stage; MTP stages require both sides.
            keys_to_keep = {'original_seq_len', 'padded_seq_len'}
            if is_first_pp or mtp_on_this_pp:
                keys_to_keep.update(('tokens', 'position_ids'))
            if is_last_pp or mtp_on_this_pp:
                keys_to_keep.update(('labels', 'loss_mask'))
            for sample in batch:
                for key in list(sample):
                    if key not in keys_to_keep:
                        del sample[key]

            # Step 3: Schedule samples into groups
            sample_id_groups = self.get_groups_and_subsamples(global_id_seqlens)

            # Validate scheduling result
            set_gbs = set()
            for group in sample_id_groups:
                for sub in group:
                    set_gbs.update(sub)
            assert len(set_gbs) == len(global_id_seqlens), (
                f"set_gbs length: {len(set_gbs)} != "
                f"global_id_seqlens length: {len(global_id_seqlens)}"
            )

            # Step 4: Reroute samples to DCP ranks
            samples_this_rank_with_id = reroute_samples_to_dcp_ranks(
                batch,
                global_ids_this_rank,
                global_id_seqlens,
                sample_id_groups,
                offsets,
                dp_group,
                dp_cp_group,
                sample_keys=keys_to_keep,
            )

            dcp_rank = dp_cp_group.rank()
            num_micro_batches = len(sample_id_groups)

            # Step 5: Build packed microbatches
            new_samples = build_packed_microbatches(
                samples_this_rank_with_id,
                sample_id_groups,
                dcp_rank,
                dev,
                is_dynamic_cp=self.is_dynamic_cp,
                global_id_seqlens=global_id_seqlens,
            )

            # Step 6: Calculate FLOPs info
            seqlen_sum_this_global_batch = float(sum(seqlens_gathered))
            seqlen_squared_sum_this_global_batch = float(
                sum(seqlen**2 for seqlen in seqlens_gathered)
            )
        else:
            (
                new_samples,
                num_micro_batches,
                seqlen_sum_this_global_batch,
                seqlen_squared_sum_this_global_batch,
            ) = (None, None, None, None)

        # Broadcast the schedule shape and FLOPs metadata inside each TP group.
        num_micro_batches, seqlen_sum_this_global_batch, seqlen_squared_sum_this_global_batch = (
            broadcast_scalars(
                [
                    num_micro_batches,
                    seqlen_sum_this_global_batch,
                    seqlen_squared_sum_this_global_batch,
                ],
                tp_group,
                dev,
            )
        )
        num_micro_batches = int(num_micro_batches)

        # Build an independent iterator for each VPP stage so rerun/rewind state
        # and per-stage field stripping cannot leak between virtual stages.
        new_data_iterator = create_data_iterator(
            new_samples,
            tp_group,
            config,
            vpp_needs_data=vpp_needs_data,
            is_dynamic_cp=self.is_dynamic_cp,
        )

        return (
            new_data_iterator,
            num_micro_batches,
            seqlen_sum_this_global_batch,
            seqlen_squared_sum_this_global_batch,
        )


class PackingSchedulerEnum(enum.Enum):
    """Enum for supported sequence packing algorithms."""

    DP_BALANCED = "dp_balanced"
    DEFAULT_DYNAMIC_CP = "default_dynamic_cp"


class DefaultDynamicCPScheduler(DpBalancedScheduler):
    """Balance packed samples while selecting a runtime CP size per microbatch."""

    def __init__(
        self, *args, min_cp_size: int = 1, cp_group_sizes: Sequence[int] | None = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.is_dynamic_cp = True
        self.total_dpxcp_ranks = self.dp_size * self.cp_size
        valid_sizes = get_valid_dynamic_context_parallel_group_sizes(self.total_dpxcp_ranks)
        self.cp_group_sizes = tuple(
            sorted(set(valid_sizes if cp_group_sizes is None else cp_group_sizes))
        )
        if set(self.cp_group_sizes) - set(valid_sizes):
            raise ValueError(
                f"Unsupported Dynamic CP group sizes {list(self.cp_group_sizes)} for "
                f"DPxCP size {self.total_dpxcp_ranks}; expected a subset of {valid_sizes}"
            )
        if min_cp_size not in self.cp_group_sizes:
            raise ValueError(
                f"min_cp_size={min_cp_size} is not a valid group size for DPxCP size "
                f"{self.total_dpxcp_ranks}; expected one of {list(self.cp_group_sizes)}"
            )
        self.min_cp_size = min_cp_size

    def get_groups_and_subsamples(self, sample_id_seqlens):
        sample_id_seqlens = sorted(sample_id_seqlens, key=lambda item: item[1], reverse=True)
        lengths_by_id = dict(sample_id_seqlens)
        sample_id_groups = []

        while sample_id_seqlens:
            _, sample_id_seqlens, _, sample_ids = next_hdp_group_packing_aware(
                sample_id_seqlens,
                self.total_dpxcp_ranks,
                max_seq_len_per_rank=self.max_seqlen_per_dp_cp_rank,
                min_cp_size=self.min_cp_size,
                cp_group_sizes=self.cp_group_sizes,
                pad_sequences=self.pad_sequences,
            )
            sample_id_groups.append(sample_ids)

        if (
            self.microbatch_group_size_per_vp_stage is not None
            and self.microbatch_group_size_per_vp_stage > 1
        ):
            sample_id_groups = align_sample_id_groups(
                sample_id_groups,
                self.microbatch_group_size_per_vp_stage,
                cp_group_sizes=self.cp_group_sizes,
                sample_lengths=lengths_by_id,
                max_seq_len_per_rank=self.max_seqlen_per_dp_cp_rank,
                pad_sequences=self.pad_sequences,
            )

        return sample_id_groups


scheduler_map: Dict[PackingSchedulerEnum, Type[BasePackingScheduler]] = {
    PackingSchedulerEnum.DP_BALANCED: DpBalancedScheduler,
    PackingSchedulerEnum.DEFAULT_DYNAMIC_CP: DefaultDynamicCPScheduler,
}


def _get_scheduler_max_real_num_seqs(config) -> Optional[int]:
    """Return the scheduler cap for real THD sequences.

    ``thd_max_packed_sequences`` is the final static THD capacity, including the
    optional dummy sequence appended for a padding tail. The dp_balanced
    scheduler only packs real sequences, so reserve one slot when dummy-tail
    padding is enabled.
    """
    max_num_seqs = getattr(config, 'thd_max_packed_sequences', None)
    if max_num_seqs is None:
        return None

    max_num_seqs = int(max_num_seqs)
    if max_num_seqs < 1:
        raise ValueError(f"thd_max_packed_sequences must be >= 1, got {max_num_seqs}.")

    if (
        getattr(config, 'pad_packed_seq_alignment', None) is not None
        and resolve_thd_tail_padding_policy(config) == 'append_dummy_seq'
    ):
        if max_num_seqs < 2:
            raise ValueError(
                "thd_max_packed_sequences must be >= 2 when THD padding appends a dummy "
                "sequence, because thd_max_packed_sequences includes that dummy sequence."
            )
        return max_num_seqs - 1

    return max_num_seqs


def wrap_data_iterator(
    data_iterator, config, num_microbatches, pg_collection: ProcessGroupCollection
):
    """
    A wrapper function that wraps around an existing data_iterator
    and return the num_micro_batches for sequence packing.

    Args:
        data_iterator: The original data_iterator to wrap around
        config: The config object containing the max_seqlen_per_dp_cp_rank
        pg_collection: Caller-owned process groups; must define dp_cp, dp, tp, and pp.
    """

    # GTP_remat peers consume distinct samples, so packing must span them like the rest of DP.
    dp_cp_group = pg_collection.dp_cp_gtp_remat or pg_collection.dp_cp
    dp_group = pg_collection.dp_gtp_remat or pg_collection.dp
    tp_group = pg_collection.tp
    pp_group = pg_collection.pp
    assert (
        dp_cp_group is not None
        and dp_group is not None
        and tp_group is not None
        and pp_group is not None
    ), "dp_cp, dp, tp, and pp must be defined in pg_collection for sequence packing"

    dev = torch.cuda.current_device()
    dp_size = dp_group.size()
    cp_size = dp_cp_group.size() // dp_size

    # Convert string to enum
    scheduler_type = config.sequence_packing_scheduler
    scheduler_type = PackingSchedulerEnum[scheduler_type.upper()]

    scheduler_kwargs = {}
    if scheduler_type == PackingSchedulerEnum.DEFAULT_DYNAMIC_CP:
        scheduler_kwargs['min_cp_size'] = config.min_dynamic_context_parallel_size
        if pg_collection.dynamic_cp:
            scheduler_kwargs['cp_group_sizes'] = tuple(pg_collection.dynamic_cp)
        elif getattr(config, 'dynamic_context_parallel', True):
            raise ValueError("pg_collection.dynamic_cp is required for dynamic CP scheduling")
        # Otherwise this is the static-CP use of the packing-aware scheduler
        # (dynamic_context_parallel=False), which the config keeps legal. Initialization creates
        # no runtime CP groups in that mode, so the scheduler derives its candidate sizes from
        # the DPxCP grid and the consumer keeps the static CP group.

    capacity = config.max_seqlen_per_dp_cp_rank
    alignment, pad_enabled = get_packed_sequence_alignment(config, tp_group.size())
    scheduler_kwargs['pad_sequences'] = pad_enabled
    if pad_enabled:
        capacity -= capacity % alignment
        if capacity < alignment:
            raise ValueError(f"Packed sequence capacity must fit alignment {alignment}")

    # dp_balanced packs only real sequences, so the static THD capacity has to be
    # discounted by the dummy tail sequence; dynamic CP packs against the raw cap.
    scheduler_kwargs['max_num_seqs'] = (
        _get_scheduler_max_real_num_seqs(config)
        if scheduler_type == PackingSchedulerEnum.DP_BALANCED
        else getattr(config, 'thd_max_packed_sequences', None)
    )

    scheduler = scheduler_map[scheduler_type](
        capacity,
        cp_size,
        dp_size,
        # When VPP is enabled, align num_micro_batches to this multiple.
        (
            None
            if config.virtual_pipeline_model_parallel_size is None
            else config.microbatch_group_size_per_vp_stage
        ),
        **scheduler_kwargs,
    )

    (
        new_data_iterator,
        num_micro_batches,
        seqlen_sum_this_global_batch,
        seqlen_squared_sum_this_global_batch,
    ) = scheduler.run(
        data_iterator, num_microbatches, dp_group, tp_group, pp_group, dp_cp_group, dev, config
    )

    return (
        new_data_iterator,
        num_micro_batches,
        seqlen_sum_this_global_batch,
        seqlen_squared_sum_this_global_batch,
    )


def _get_dynamic_cp_group(
    group_size: int, groups: Dict[int, torch.distributed.ProcessGroup], name: str
) -> torch.distributed.ProcessGroup:
    """Select a pre-created runtime group from the caller's parallel grid."""
    group = groups.get(group_size)
    if group is None:
        raise ValueError(f"pg_collection.{name} has no group for runtime CP size {group_size}")
    return group


def _sequence_parallel_tp_cp_group(
    config,
    tp_group: torch.distributed.ProcessGroup,
    dynamic_cp: bool,
    local_cp_size: int,
    dynamic_tp_cp_group_func: Optional[Callable[..., torch.distributed.ProcessGroup]],
    tp_cp_group: Optional[torch.distributed.ProcessGroup],
) -> Optional[torch.distributed.ProcessGroup]:
    """Return the TP x CP group sequence parallelism needs, or None without it."""
    if not (config.sequence_parallel and tp_group.size() > 1):
        return None
    if dynamic_cp:
        if dynamic_tp_cp_group_func is None:
            raise ValueError(
                "dynamic_tp_cp_group_func is required for dynamic CP with sequence parallelism"
            )
        return dynamic_tp_cp_group_func(group_size=local_cp_size)
    if tp_cp_group is None:
        raise ValueError(
            "tp_cp_group (or pg_collection.tp_cp) is required for sequence parallelism"
        )
    return tp_cp_group


def get_batch_on_this_rank_for_sequence_packing(
    data_iterator,
    vpp_size: Optional[int] = None,
    mtp_on_this_rank: bool = False,
    vp_stage: Optional[int] = None,
    dynamic_cp: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
    config=None,
    return_context_parallel_batch: bool = False,
    dynamic_cp_group_func: Optional[Callable[..., torch.distributed.ProcessGroup]] = None,
    dynamic_tp_cp_group_func: Optional[Callable[..., torch.distributed.ProcessGroup]] = None,
    tp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    Get a batch of data for sequence packing.
    Args:
        data_iterator (Iterator): The data iterator to get the batch from.
        mtp_on_this_rank (bool): Whether to use multi-token prediction.
        vp_stage (Optional[int]): The stage of the pipeline.
        return_context_parallel_batch (bool): Return layout-keyed batch views
            instead of the legacy GPT tuple.
        dynamic_cp_group_func (Callable, optional): Overrides the collection's runtime CP
            group lookup for ``group_size`` ranks containing this rank.
        dynamic_tp_cp_group_func (Callable, optional): Returns the TP x runtime-DPxCP group
            of ``group_size`` ranks. Overrides ``pg_collection.dynamic_tp_cp``.
        tp_cp_group (torch.distributed.ProcessGroup, optional): Static TP x CP group for
            sequence parallelism. Defaults to ``pg_collection.tp_cp``.
        pg_collection (ProcessGroupCollection): Caller-owned parallel grid. Runtime groups
            default to its ``dynamic_cp`` and ``dynamic_tp_cp`` maps; callbacks may override them.
    Returns:
        tuple of (tokens, labels, loss_mask, attention_mask, position_ids,
        packed_seq_params, padding_mask)
    """

    # Full-iteration CUDA graph feeds every rank a pre-canonicalized static
    # batch (see prepare_thd_static_batch_for_full_iteration_cuda_graph). The
    # captured path must read those tensors from fixed addresses without shape
    # discovery, host synchronization, or dynamic allocation.
    prefetched_batch = None
    if data_iterator is not None:
        prefetched_batch = next(data_iterator)
        if isinstance(prefetched_batch, dict) and prefetched_batch.get(
            THD_FULL_ITERATION_STATIC_BATCH_KEY, False
        ):
            return _unpack_thd_full_iteration_static_batch(prefetched_batch)

    if pg_collection is None:
        raise ValueError("pg_collection is required for sequence packing")
    tp_group = pg_collection.tp
    pp_group = pg_collection.pp
    cp_group = pg_collection.cp
    if tp_group is None or pp_group is None or cp_group is None:
        raise ValueError("pg_collection must define tp, pp, and cp for sequence packing")
    if tp_cp_group is None:
        tp_cp_group = pg_collection.tp_cp
    if dynamic_cp and dynamic_cp_group_func is None:
        if not pg_collection.dynamic_cp:
            raise ValueError("pg_collection.dynamic_cp is required when dynamic_cp is True")
        dynamic_cp_group_func = partial(
            _get_dynamic_cp_group, groups=pg_collection.dynamic_cp, name="dynamic_cp"
        )
    if dynamic_cp and dynamic_tp_cp_group_func is None:
        dynamic_tp_cp_group_func = partial(
            _get_dynamic_cp_group, groups=pg_collection.dynamic_tp_cp or {}, name="dynamic_tp_cp"
        )

    tp_src_rank = torch.distributed.get_process_group_ranks(tp_group)[0]

    is_tp_rank_0 = tp_group.rank() == 0
    is_first_stage = pp_group.rank() == 0 and (vp_stage is None or vp_stage == 0)
    is_last_stage = pp_group.rank() == pp_group.size() - 1 and (
        vp_stage is None or vp_stage == vpp_size - 1
    )

    dev = torch.cuda.current_device()

    # data_iterator should return a batch including the following keys.
    batch_keys = ['cu_seqlens', 'cu_seqlens_padded', 'max_seqlen', 'zigzag_cp_min_chunk_size']
    if dynamic_cp:
        batch_keys.append('local_cp_size')
    if is_first_stage or mtp_on_this_rank:
        batch_keys.append('tokens')
        batch_keys.append('position_ids')
    if is_last_stage or mtp_on_this_rank:
        batch_keys.append('labels')
        batch_keys.append('loss_mask')

    # Get a batch from data_iterator or create an emtpy batch.
    if is_tp_rank_0:
        assert prefetched_batch is not None, "TP rank 0 requires a data_iterator"
        # Work on a shallow copy: the per-rank padding/slicing below rebinds the entries, and
        # replay paths (RerunDataIterator, the PagedStashRunner dropless retry) serve the same
        # dict objects again expecting the pre-slice tensors.
        batch = dict(prefetched_batch)
        # External iterators do not carry the optional scheduler certificate.
        # Unknown geometry keeps MTP on its established zigzag packed-CP roll path.
        if 'zigzag_cp_min_chunk_size' not in batch:
            batch['zigzag_cp_min_chunk_size'] = torch.tensor(-1, dtype=torch.int32, device=dev)
        for key in batch_keys:
            assert key in batch, f"{key} is missing in current batch."
    else:
        assert data_iterator is None, "Non TP 0 rank should not have data_iterator"
        batch = {}

    # The scheduler chooses one runtime CP group for this packed microbatch.
    # CP1 carries a real singleton group, but does not need THD partitioning.
    if dynamic_cp and is_tp_rank_0:
        local_cp_size = batch['local_cp_size']
        if isinstance(local_cp_size, torch.Tensor):
            local_cp_size = int(local_cp_size.item())
        cp_group = dynamic_cp_group_func(group_size=local_cp_size)

    # Build padding_mask before CP slicing while tensors still have the full
    # packed length represented by cu_seqlens_padded[-1].
    if is_tp_rank_0:
        batch['padding_mask'] = _build_thd_padding_mask(
            batch['cu_seqlens'], batch['cu_seqlens_padded']
        )
        _sanitize_thd_padding_values(batch, batch['padding_mask'])
        original_cu_seqlens_padded = batch['cu_seqlens_padded']
        pad_packed_batch_before_cp_slice(batch, config, cp_group.size(), tp_group.size())
        if batch['cu_seqlens_padded'] is not original_cu_seqlens_padded:
            batch['zigzag_cp_min_chunk_size'] = _refresh_zigzag_cp_certificate(
                batch['zigzag_cp_min_chunk_size'], batch['cu_seqlens_padded'], cp_group.size()
            )

    # dsv4_hybrid can give each CP rank one contiguous row range instead of TE's zigzag split.
    cp_partition_mode = getattr(config, 'cp_partition_mode', 'zigzag')

    # Partition padding_mask for context parallel on every PP stage. Partition
    # token-like tensors only on stages that own them.
    if is_tp_rank_0 and not return_context_parallel_batch:
        cp_size = local_cp_size if dynamic_cp else cp_group.size()
        cp_rank = cp_group.rank()
        # If cp_size == 1, no need to do further processing.
        if cp_size > 1:
            # Transformer Engine has a bug of cu_seqlens, we must treat cu_seqlens_padded as
            # cu_seqlens to get the correct result.
            # TODO: Revert this workaround once TE fixes the issue.
            cu_seqlens = batch["cu_seqlens_padded"]
            total_tokens = int(cu_seqlens[-1].item())
            if cp_partition_mode == "contiguous":
                # Each CP rank owns one equal row range of the padded pack.
                assert total_tokens % cp_size == 0, (
                    f"Contiguous CP needs the {total_tokens} packed tokens to divide by "
                    f"{cp_size}; set --pad-packed-seq-alignment."
                )
                local_tokens = total_tokens // cp_size
                index = torch.arange(
                    cp_rank * local_tokens, (cp_rank + 1) * local_tokens, device=cu_seqlens.device
                )
            else:
                assert (
                    tex is not None
                ), "Transformer Engine is required to use Context Parallel with THD format data."
                index = tex.thd_get_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)
            cp_slice_keys = ['padding_mask']
            if is_first_stage or mtp_on_this_rank:
                cp_slice_keys.extend(['tokens', 'position_ids'])
            if is_last_stage or mtp_on_this_rank:
                cp_slice_keys.extend(['labels', 'loss_mask'])
            for key in cp_slice_keys:
                batch[key] = batch[key].index_select(0, index)

    # Broadcast the receive-buffer shapes inside the TP group:
    # - cu_seqlen_size is needed to allocate cu_seqlens / cu_seqlens_padded on non TP 0 ranks.
    # - total_tokens is needed because padding_mask is prepared on every PP stage, and
    #   tokens/labels/loss_mask/position_ids use the same length on stages that own them.
    shapes = (
        [batch['cu_seqlens'].size(0), batch['padding_mask'].size(0)] if is_tp_rank_0 else [0, 0]
    )
    cu_seqlen_size, total_tokens = broadcast_scalars(shapes, tp_group, dev, dtype=torch.int32)

    # Step1: Prepare "tokens", "position_ids" on all ranks.
    if is_first_stage or mtp_on_this_rank:
        if is_tp_rank_0:
            assert batch['tokens'].dtype == torch.int64
            assert batch['position_ids'].dtype == torch.int64
            batch['tokens'] = batch['tokens'].view(1, total_tokens)
            batch['position_ids'] = batch['position_ids'].view(1, total_tokens)
        else:
            batch['tokens'] = torch.empty([1, total_tokens], dtype=torch.int64, device=dev)
            batch['position_ids'] = torch.empty([1, total_tokens], dtype=torch.int64, device=dev)
    else:
        # Non first stage rank doesn't need tokens and position_ids.
        batch['tokens'] = None
        batch['position_ids'] = None

    # Step2: Prepare "labels", "loss_mask" on all ranks.
    if is_last_stage or mtp_on_this_rank:
        if is_tp_rank_0:
            assert batch['labels'].dtype == torch.int64
            assert batch['loss_mask'].dtype == torch.float32
            batch['labels'] = batch['labels'].view(1, total_tokens)
            batch['loss_mask'] = batch['loss_mask'].view(1, total_tokens)
        else:
            batch['labels'] = torch.empty([1, total_tokens], dtype=torch.int64, device=dev)
            batch['loss_mask'] = torch.empty([1, total_tokens], dtype=torch.float32, device=dev)
    else:
        # Non last stage rank doesn't need labels and loss_mask.
        batch['labels'] = None
        batch['loss_mask'] = None

    # Step3: Prepare "padding_mask" on all TP ranks.
    if is_tp_rank_0:
        assert batch['padding_mask'].dtype == torch.bool
        batch['padding_mask'] = batch['padding_mask'].view(1, total_tokens)
    else:
        batch['padding_mask'] = torch.empty([1, total_tokens], dtype=torch.bool, device=dev)

    # Step4: Prepare "cu_seqlens", "cu_seqlens_padded", "max_seqlen" on all ranks.
    if is_tp_rank_0:
        assert batch['cu_seqlens'].dtype == torch.int32
        assert batch['cu_seqlens_padded'].dtype == torch.int32
        assert batch['cu_seqlens'].dim() == 1
        assert batch['cu_seqlens_padded'].dim() == 1
        if type(batch['max_seqlen']) == int:
            batch['max_seqlen'] = torch.tensor(batch['max_seqlen'], dtype=torch.int32, device=dev)
        else:
            assert batch['max_seqlen'].dtype == torch.int32
            assert batch['max_seqlen'].numel() == 1
    else:
        batch['cu_seqlens'] = torch.empty([cu_seqlen_size], dtype=torch.int32, device=dev)
        batch['cu_seqlens_padded'] = torch.empty([cu_seqlen_size], dtype=torch.int32, device=dev)
        batch['max_seqlen'] = torch.empty(1, dtype=torch.int32, device=dev)

    if is_tp_rank_0:
        if type(batch['zigzag_cp_min_chunk_size']) == int:
            batch['zigzag_cp_min_chunk_size'] = torch.tensor(
                batch['zigzag_cp_min_chunk_size'], dtype=torch.int32, device=dev
            )
        else:
            assert batch['zigzag_cp_min_chunk_size'].dtype == torch.int32
            assert batch['zigzag_cp_min_chunk_size'].numel() == 1
    else:
        batch['zigzag_cp_min_chunk_size'] = torch.empty(1, dtype=torch.int32, device=dev)

    if dynamic_cp:
        if is_tp_rank_0:
            if isinstance(batch['local_cp_size'], int):
                batch['local_cp_size'] = torch.tensor(
                    [batch['local_cp_size']], dtype=torch.int32, device=dev
                )
            else:
                batch['local_cp_size'] = batch['local_cp_size'].reshape(1).to(torch.int32)
        else:
            batch['local_cp_size'] = torch.empty(1, dtype=torch.int32, device=dev)
    else:
        batch['local_cp_size'] = None

    # Broadcast batch inside TP group.
    broadcast_tensor(batch['tokens'], tp_src_rank, tp_group)
    broadcast_tensor(batch['position_ids'], tp_src_rank, tp_group)
    broadcast_tensor(batch['labels'], tp_src_rank, tp_group)
    broadcast_tensor(batch['loss_mask'], tp_src_rank, tp_group)
    broadcast_tensor(batch['padding_mask'], tp_src_rank, tp_group)
    broadcast_tensor(batch['cu_seqlens'], tp_src_rank, tp_group)
    broadcast_tensor(batch['cu_seqlens_padded'], tp_src_rank, tp_group)
    broadcast_tensor(batch['max_seqlen'], tp_src_rank, tp_group)
    broadcast_tensor(batch['zigzag_cp_min_chunk_size'], tp_src_rank, tp_group)
    broadcast_tensor(batch['local_cp_size'], tp_src_rank, tp_group)

    if return_context_parallel_batch:
        if config is None:
            raise ValueError("config is required when returning ContextParallelBatch")
        runtime_cp_group = (
            dynamic_cp_group_func(group_size=int(batch['local_cp_size'].item()))
            if dynamic_cp
            else cp_group
        )
        batch['hybrid_cp_group'] = runtime_cp_group if dynamic_cp else None
        # The CP batch sharders expect (1, num_seqs + 1) cu_seqlens, as returned by
        # flatten_batch_for_packed_sequences; the per-document sharder reads row 0.
        batch['cu_seqlens'] = batch['cu_seqlens'].unsqueeze(0)
        batch['cu_seqlens_padded'] = batch['cu_seqlens_padded'].unsqueeze(0)
        additional_layouts = set()
        if config.linear_cp_layout != config.attention_cp_layout:
            additional_layouts.add(config.attention_cp_layout)
        return get_batches_on_this_cp_rank(
            batch,
            boundary_layout=config.linear_cp_layout,
            dynamic_cp=dynamic_cp,
            cp_group=runtime_cp_group,
            additional_layouts=additional_layouts,
            dynamic_cp_group_func=dynamic_cp_group_func,
            sequence_parallel=config.sequence_parallel,
            tp_group=tp_group,
            tp_cp_group=_sequence_parallel_tp_cp_group(
                config,
                tp_group,
                dynamic_cp,
                runtime_cp_group.size(),
                dynamic_tp_cp_group_func,
                tp_cp_group,
            ),
            tokens_per_sample=None,
        )

    # Extract the data from batch after broadcasting.
    tokens = batch['tokens']
    position_ids = batch['position_ids']
    labels = batch['labels']
    loss_mask = batch['loss_mask']
    padding_mask = batch['padding_mask']
    cu_seqlens = batch['cu_seqlens']
    cu_seqlens_padded = batch['cu_seqlens_padded']
    scalar_metadata = [batch['max_seqlen'].reshape(1), batch['zigzag_cp_min_chunk_size'].reshape(1)]
    if dynamic_cp:
        scalar_metadata.append(batch['local_cp_size'].reshape(1))
    scalar_metadata_host = torch.cat(scalar_metadata).cpu().tolist()
    max_seqlen = scalar_metadata_host[0]
    zigzag_cp_min_chunk_size = scalar_metadata_host[1]
    if zigzag_cp_min_chunk_size < 0 or (
        config is not None and getattr(config, 'cuda_graph_impl', 'none') == 'full_iteration'
    ):
        zigzag_cp_min_chunk_size = None
    local_cp_size = int(scalar_metadata_host[2]) if dynamic_cp else None
    runtime_cp_group = dynamic_cp_group_func(group_size=local_cp_size) if dynamic_cp else None

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        local_cp_size=local_cp_size,
        cp_group=runtime_cp_group,
        total_tokens=int(cu_seqlens_padded[-1].item()),
        # CUDA graphs capture this branch once. Static THD metadata therefore always takes
        # the padded path, which is correct for every gap layout.
        pad_between_seqs=(
            getattr(config, 'thd_max_packed_sequences', None) is not None
            or not torch.equal(cu_seqlens, cu_seqlens_padded)
        ),
        cp_partition_mode=cp_partition_mode,
        zigzag_cp_min_chunk_size=zigzag_cp_min_chunk_size,
    )

    # "attention_mask" is not valid for sequence packing, so set it to None.
    return tokens, labels, loss_mask, None, position_ids, packed_seq_params, padding_mask
