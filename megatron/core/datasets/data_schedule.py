# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

import enum
from typing import Any, Dict, Optional, Type

import torch

from megatron.core import parallel_state
from megatron.core.context_parallel import get_batches_on_this_cp_rank
from megatron.core.datasets.data_schedule_utils import (
    align_sample_id_groups,
    broadcast_scalars,
    broadcast_tensor,
    build_packed_microbatches,
    create_data_iterator,
    get_batch_and_global_seqlens,
    get_packed_sequence_alignment,
    next_hdp_group_packing_aware,
    pad_packed_batch_before_cp_slice,
    reroute_samples_to_dcp_ranks,
)
from megatron.core.packed_seq_params import PackedSeqParams
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


class BasePackingScheduler:
    """Base class for sequence packing schedulers."""

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank: int,
        cp_size: int,
        dp_size: int,
        microbatch_group_size_per_vp_stage: Optional[int],
    ):
        """
        Args:
            max_seqlen_per_dp_cp_rank: The maximum sequence length per DPxCP rank.
            cp_size: The context parallel size.
            dp_size: The data parallel size.
            microbatch_group_size_per_vp_stage: The microbatch group size per virtual
            pipeline stage, only used when enabling VPP, otherwise None.
        """
        self.max_seqlen_per_dp_cp_rank = max_seqlen_per_dp_cp_rank
        self.cp_size = cp_size
        self.dp_size = dp_size
        self.microbatch_group_size_per_vp_stage = microbatch_group_size_per_vp_stage

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
            if sum_seqlen + sample_id_seqlens[i][1] <= self.max_seq_len_all_ranks:
                single_microbatch.append(i)
                sum_seqlen += sample_id_seqlens[i][1]
            else:
                packed_id_groups.append(single_microbatch)
                single_microbatch = [i]
                sum_seqlen = sample_id_seqlens[i][1]
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
                assert i > 0, "Not enough samples to move"
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

        if data_iterator is not None:
            assert tp_group.rank() == 0, "Only TP rank 0 should have a packed data iterator"

            # Step 1: Fetch batches and gather global sequence lengths
            batch, global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered = (
                get_batch_and_global_seqlens(data_iterator, num_microbatches, dp_group)
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

    def __init__(self, *args, min_cp_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_dynamic_cp = True
        self.total_dpxcp_ranks = self.dp_size * self.cp_size
        self.cp_group_sizes = tuple(
            parallel_state.get_valid_dynamic_context_parallel_group_sizes(self.total_dpxcp_ranks)
        )
        if min_cp_size not in self.cp_group_sizes:
            raise ValueError(
                f"min_cp_size={min_cp_size} is not a valid group size for DPxCP size "
                f"{self.total_dpxcp_ranks}; expected one of {list(self.cp_group_sizes)}"
            )
        self.min_cp_size = min_cp_size

    def get_groups_and_subsamples(self, sample_id_seqlens):
        sample_id_seqlens = sorted(sample_id_seqlens, key=lambda item: item[1], reverse=True)
        sample_id_groups = []

        while sample_id_seqlens:
            _, sample_id_seqlens, _, sample_ids = next_hdp_group_packing_aware(
                sample_id_seqlens,
                self.total_dpxcp_ranks,
                max_seq_len_per_rank=self.max_seqlen_per_dp_cp_rank,
                min_cp_size=self.min_cp_size,
                cp_group_sizes=self.cp_group_sizes,
            )
            sample_id_groups.append(sample_ids)

        if (
            self.microbatch_group_size_per_vp_stage is not None
            and self.microbatch_group_size_per_vp_stage > 1
        ):
            sample_id_groups = align_sample_id_groups(
                sample_id_groups, self.microbatch_group_size_per_vp_stage
            )

        return sample_id_groups


scheduler_map: Dict[PackingSchedulerEnum, Type[BasePackingScheduler]] = {
    PackingSchedulerEnum.DP_BALANCED: DpBalancedScheduler,
    PackingSchedulerEnum.DEFAULT_DYNAMIC_CP: DefaultDynamicCPScheduler,
}


def wrap_data_iterator(
    data_iterator, config, num_microbatches, pg_collection: Optional[ProcessGroupCollection] = None
):
    """
    A wrapper function that wraps around an existing data_iterator
    and return the num_micro_batches for sequence packing.

    Args:
        data_iterator: The original data_iterator to wrap around
        config: The config object containing the max_seqlen_per_dp_cp_rank
        pg_collection: The process group collection.
    """

    if pg_collection is None:
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        dp_group = parallel_state.get_data_parallel_group()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
    else:
        dp_cp_group = pg_collection.dp_cp
        dp_group = pg_collection.dp
        tp_group = pg_collection.tp
        pp_group = pg_collection.pp
    assert (
        dp_cp_group is not None
        and dp_group is not None
        and tp_group is not None
        and pp_group is not None
    ), "dp_cp_group, dp_group, tp_group must not be None when using sequence packing"

    dev = torch.cuda.current_device()
    dp_size = dp_group.size()
    cp_size = dp_cp_group.size() // dp_size

    # Convert string to enum
    scheduler_type = config.sequence_packing_scheduler
    scheduler_type = PackingSchedulerEnum[scheduler_type.upper()]

    scheduler_kwargs = {}
    if scheduler_type == PackingSchedulerEnum.DEFAULT_DYNAMIC_CP:
        scheduler_kwargs['min_cp_size'] = config.min_dynamic_context_parallel_size

    capacity = config.max_seqlen_per_dp_cp_rank
    alignment, pad_enabled = get_packed_sequence_alignment(config, tp_group.size())
    if pad_enabled:
        capacity -= capacity % alignment
        if capacity < alignment:
            raise ValueError(f"Packed sequence capacity must fit alignment {alignment}")

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


def get_batch_on_this_rank_for_sequence_packing(
    data_iterator,
    vpp_size: Optional[int] = None,
    mtp_on_this_rank: bool = False,
    vp_stage: Optional[int] = None,
    dynamic_cp: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
    config=None,
    return_context_parallel_batch: bool = False,
):
    """
    Get a batch of data for sequence packing.
    Args:
        data_iterator (Iterator): The data iterator to get the batch from.
        mtp_on_this_rank (bool): Whether to use multi-token prediction.
        vp_stage (Optional[int]): The stage of the pipeline.
        return_context_parallel_batch (bool): Return layout-keyed batch views
            instead of the legacy GPT tuple.
    Returns:
        tuple of (tokens, labels, loss_mask, attention_mask, position_ids,
        packed_seq_params, padding_mask)
    """

    if pg_collection is None:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
    else:
        tp_group = pg_collection.tp
        pp_group = pg_collection.pp
        cp_group = pg_collection.cp

    tp_src_rank = torch.distributed.get_process_group_ranks(tp_group)[0]

    is_tp_rank_0 = tp_group.rank() == 0
    is_first_stage = pp_group.rank() == 0 and (vp_stage is None or vp_stage == 0)
    is_last_stage = pp_group.rank() == pp_group.size() - 1 and (
        vp_stage is None or vp_stage == vpp_size - 1
    )

    is_first_or_last_stage = is_first_stage or is_last_stage
    dev = torch.cuda.current_device()

    # data_iterator should return a batch including the following keys.
    batch_keys = ['cu_seqlens', 'cu_seqlens_padded', 'max_seqlen']
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
        assert data_iterator is not None
        batch = next(data_iterator)
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
        cp_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)

    # Build padding_mask before CP slicing while tensors still have the full
    # packed length represented by cu_seqlens_padded[-1].
    if is_tp_rank_0:
        batch['padding_mask'] = _build_thd_padding_mask(
            batch['cu_seqlens'], batch['cu_seqlens_padded']
        )
        _sanitize_thd_padding_values(batch, batch['padding_mask'])
        pad_packed_batch_before_cp_slice(batch, config, cp_group.size(), tp_group.size())

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
            assert (
                tex is not None
            ), "Transformer Engine is required to use Context Parallel with THD format data."
            index = tex.thd_get_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)
            cp_slice_keys = ['padding_mask']
            if is_first_or_last_stage or mtp_on_this_rank:
                cp_slice_keys.extend(['tokens', 'position_ids', 'labels', 'loss_mask'])
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
    broadcast_tensor(batch['local_cp_size'], tp_src_rank, tp_group)

    if return_context_parallel_batch:
        if config is None:
            raise ValueError("config is required when returning ContextParallelBatch")
        runtime_cp_group = (
            parallel_state.get_dynamic_data_context_parallel_groups(
                group_size=int(batch['local_cp_size'].item())
            )
            if dynamic_cp
            else cp_group
        )
        batch['hybrid_cp_group'] = runtime_cp_group if dynamic_cp else None
        additional_layouts = set()
        if config.linear_cp_layout != config.attention_cp_layout:
            additional_layouts.add(config.attention_cp_layout)
        return get_batches_on_this_cp_rank(
            batch,
            boundary_layout=config.linear_cp_layout,
            is_hybrid_cp=dynamic_cp,
            cp_group=runtime_cp_group,
            additional_layouts=additional_layouts,
            hybrid_cp_group_func=parallel_state.get_dynamic_data_context_parallel_groups,
            sequence_parallel=config.sequence_parallel,
            tp_group=tp_group,
            tp_cp_group=(
                parallel_state.get_dynamic_tensor_data_context_parallel_group(
                    group_size=int(batch['local_cp_size'].item())
                )
                if dynamic_cp and config.sequence_parallel and tp_group.size() > 1
                else (
                    parallel_state.get_tensor_and_context_parallel_group()
                    if config.sequence_parallel and tp_group.size() > 1
                    else None
                )
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
    max_seqlen = batch['max_seqlen'].item()
    local_cp_size = int(batch['local_cp_size'].item()) if dynamic_cp else None
    runtime_cp_group = (
        parallel_state.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)
        if dynamic_cp
        else None
    )

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
        pad_between_seqs=not torch.equal(cu_seqlens, cu_seqlens_padded),
    )

    # "attention_mask" is not valid for sequence packing, so set it to None.
    return tokens, labels, loss_mask, None, position_ids, packed_seq_params, padding_mask
