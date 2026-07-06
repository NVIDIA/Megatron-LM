# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, Optional, Type

import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule_utils import (
    _get_thd_partitioned_indices,
    align_sample_id_groups,
    broadcast_scalars,
    broadcast_tensor,
    build_packed_microbatches,
    create_data_iterator,
    get_batch_and_global_seqlens,
    next_hdp_group_packing_aware,
    reroute_samples_to_dcp_ranks,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank as mtp_is_on_rank


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
        pipeline_model_parallel_size: int = 1,
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
        self.pipeline_model_parallel_size = pipeline_model_parallel_size

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

        num_packed_sequence = len(packed_id_groups)

        # Every microbatch needs one packed group per DP rank. With VPP, a
        # partial final group is valid when it has at least PP microbatches;
        # otherwise split samples until the next full interleaving group.
        num_to_move = (-num_packed_sequence) % self.dp_size
        aligned_packed = num_packed_sequence + num_to_move
        if self.microbatch_group_size_per_vp_stage is not None:
            num_microbatches = aligned_packed // self.dp_size
            group_size = self.microbatch_group_size_per_vp_stage
            if num_microbatches < group_size:
                target_microbatches = group_size
            else:
                remainder = num_microbatches % group_size
                target_microbatches = (
                    num_microbatches
                    if remainder == 0 or remainder >= self.pipeline_model_parallel_size
                    else num_microbatches + group_size - remainder
                )
            num_to_move += self.dp_size * (target_microbatches - num_microbatches)

        if num_to_move > 0:
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

        Packed-sequence datasets are built on TP rank 0 of every PP stage. Each
        stage therefore runs the same schedule locally, retaining only the data
        fields required by that stage before the all-to-all transfer.

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

        total_dcp_gpus = dp_cp_group.size()
        is_first_pp = pp_group.rank() == 0
        is_last_pp = pp_group.rank() == pp_group.size() - 1
        mtp_on_this_pp = mtp_is_on_rank(
            layout=config.pipeline_model_parallel_layout,
            mtp_num_layers=config.mtp_num_layers,
            ignore_virtual=True,
        )

        vpp_size = config.virtual_pipeline_model_parallel_size or 1
        vpp_needs_data = None
        if vpp_size > 1:
            assert len(data_iterator) == vpp_size
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
                    if mtp_is_on_rank(
                        layout=config.pipeline_model_parallel_layout,
                        mtp_num_layers=config.mtp_num_layers,
                        ignore_virtual=False,
                        vp_stage=vp_stage,
                    ):
                        vpp_needs_data[vp_stage] = True

        if data_iterator is not None:
            assert tp_group.rank() == 0, "Only TP rank 0 should have data_iterator"

            # Step 1: Fetch batches and gather global sequence lengths
            (
                batch,
                global_id_seqlens,
                global_ids_this_rank,
                offsets,
                _padded_seqlens_gathered,
                original_seqlens_gathered,
            ) = get_batch_and_global_seqlens(data_iterator, num_microbatches, dp_group)

            # Step 2: Check required sample keys
            for key in self.get_required_sample_keys():
                assert (
                    key in batch[0]
                ), f"Batch missing required key {key}, provided keys: {batch[0].keys()}"

            # Avoid transferring fields that this pipeline stage never consumes.
            keys_to_keep = {'original_seq_len', 'padded_seq_len'}
            if is_first_pp or mtp_on_this_pp:
                keys_to_keep.update(['tokens', 'position_ids'])
            if is_last_pp or mtp_on_this_pp:
                keys_to_keep.update(['labels', 'loss_mask'])
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
                total_dcp_gpus,
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
            seqlen_sum_this_global_batch = float(sum(original_seqlens_gathered))
            seqlen_squared_sum_this_global_batch = float(
                sum(seqlen**2 for seqlen in original_seqlens_gathered)
            )
        else:
            (
                new_samples,
                num_micro_batches,
                seqlen_sum_this_global_batch,
                seqlen_squared_sum_this_global_batch,
            ) = (None, None, None, None)

        # Broadcast scalar schedule results to the remaining TP ranks.
        (num_micro_batches, seqlen_sum_this_global_batch, seqlen_squared_sum_this_global_batch) = (
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

        new_data_iterator = create_data_iterator(
            new_samples, tp_group, config, vpp_needs_data, is_dynamic_cp=self.is_dynamic_cp
        )

        return (
            new_data_iterator,
            num_micro_batches,
            seqlen_sum_this_global_batch,
            seqlen_squared_sum_this_global_batch,
        )


class DefaultDynamicCPScheduler(DpBalancedScheduler):
    """Balance packed sequences across variable-sized context-parallel groups."""

    def __init__(self, *args, min_cp_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_dynamic_cp = True
        self.total_dcp_gpus = self.dp_size * self.cp_size
        self.min_cp_size = min_cp_size

    def get_groups_and_subsamples(self, sample_id_seqlens):
        """Form packing-aware DCP microbatches until all samples are scheduled."""
        sample_id_groups = []
        remaining = sorted(sample_id_seqlens, key=lambda item: item[1], reverse=True)
        while remaining:
            _, remaining, _, sample_ids = next_hdp_group_packing_aware(
                remaining,
                self.total_dcp_gpus,
                max_seq_len_per_rank=self.max_seqlen_per_dp_cp_rank,
                min_cp_size=self.min_cp_size,
            )
            sample_id_groups.append(sample_ids)

        if self.microbatch_group_size_per_vp_stage is not None:
            sample_id_groups = align_sample_id_groups(
                sample_id_groups,
                self.microbatch_group_size_per_vp_stage,
                min_final_microbatches=self.pipeline_model_parallel_size,
            )
        return sample_id_groups


scheduler_map: Dict[str, Type[BasePackingScheduler]] = {
    'dp_balanced': DpBalancedScheduler,
    'default_dynamic_cp': DefaultDynamicCPScheduler,
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
        dp_cp_group: Data parallel context parallel group.
        pg_collection: The process group collection.
    """

    if pg_collection is None:
        # Compatibility fallback for callers that have not migrated process groups yet.
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

    scheduler_type = config.sequence_packing_scheduler
    scheduler_kwargs = {}
    if scheduler_type == 'default_dynamic_cp':
        scheduler_kwargs['min_cp_size'] = config.min_dynamic_context_parallel_size
    scheduler = scheduler_map[scheduler_type](
        config.max_seqlen_per_dp_cp_rank,
        cp_size,
        dp_size,
        # When VPP is enabled, align num_micro_batches to this multiple.
        (
            None
            if config.virtual_pipeline_model_parallel_size is None
            else config.microbatch_group_size_per_vp_stage
        ),
        pipeline_model_parallel_size=pp_group.size(),
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
    dynamic_cp_group_func=None,
):
    """
    Get a batch of data for sequence packing.
    Args:
        data_iterator (Iterator): The data iterator to get the batch from.
        mtp_on_this_rank (bool): Whether to use multi-token prediction.
        vp_stage (Optional[int]): The stage of the pipeline.
    Returns:
        tuple of (tokens, labels, loss_mask, attention_mask, position_ids,
        packed_seq_params, padding_mask)
    """

    if pg_collection is None:
        # Compatibility fallback for entrypoints that do not carry process groups yet.
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

    if dynamic_cp_group_func is None:
        # Compatibility fallback until ProcessGroupCollection carries the DCP group map.
        dynamic_cp_group_func = parallel_state.get_dynamic_data_context_parallel_groups

    # DCP selects a process group independently for every scheduled microbatch.
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

    # Partition padding_mask for context parallel on every PP stage. Partition
    # token-like tensors only on stages that own them.
    if is_tp_rank_0:
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        # If cp_size == 1, no need to do further processing.
        if cp_size > 1:
            # Transformer Engine has a bug of cu_seqlens, we must treat cu_seqlens_padded as
            # cu_seqlens to get the correct result.
            # TODO: Revert this workaround once TE fixes the issue.
            cu_seqlens = batch["cu_seqlens_padded"]
            total_tokens = int(cu_seqlens[-1].item())
            index = _get_thd_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)
            cp_slice_keys = ['padding_mask']
            if is_first_stage or mtp_on_this_rank:
                cp_slice_keys.extend(['tokens', 'position_ids'])
            if is_last_stage or mtp_on_this_rank:
                cp_slice_keys.extend(['labels', 'loss_mask'])
            for key in cp_slice_keys:
                if key in batch and batch[key] is not None:
                    batch[key] = batch[key].index_select(0, index)

    # Broadcast cu_seqlens_size because we need it to create placeholder for cu_seqlens and
    # cu_seqlens_padded for non TP 0 ranks.
    if is_tp_rank_0:
        cu_seqlen_size = torch.tensor(batch['cu_seqlens'].size(0), dtype=torch.int32, device=dev)
    else:
        cu_seqlen_size = torch.empty(1, dtype=torch.int32, device=dev)
    broadcast_tensor(cu_seqlen_size, tp_src_rank, tp_group)
    cu_seqlen_size = cu_seqlen_size.item()

    # Broadcast total_tokens because padding_mask is prepared on every PP stage.
    # Tokens/labels/loss_mask/position_ids use the same length on stages that own them.
    if is_tp_rank_0:
        total_tokens = (batch['cu_seqlens_padded'][-1].to(torch.int32) // cp_group.size()).reshape(
            1
        )
    else:
        total_tokens = torch.empty(1, dtype=torch.int32, device=dev)
    broadcast_tensor(total_tokens, tp_src_rank, tp_group)
    total_tokens = total_tokens.item()

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

    # Step5: Prepare the per-microbatch CP size on every TP rank.
    if dynamic_cp:
        if is_tp_rank_0:
            if type(batch['local_cp_size']) == int:
                batch['local_cp_size'] = torch.tensor(
                    batch['local_cp_size'], dtype=torch.int32, device=dev
                )
            else:
                assert batch['local_cp_size'].dtype == torch.int32
                assert batch['local_cp_size'].numel() == 1
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

    # Extract the data from batch after broadcasting.
    tokens = batch['tokens']
    position_ids = batch['position_ids']
    labels = batch['labels']
    loss_mask = batch['loss_mask']
    padding_mask = batch['padding_mask']
    cu_seqlens = batch['cu_seqlens']
    cu_seqlens_padded = batch['cu_seqlens_padded']
    max_seqlen = batch['max_seqlen'].item()
    local_cp_size = batch['local_cp_size'].item() if dynamic_cp else None
    cp_group = dynamic_cp_group_func(group_size=local_cp_size) if dynamic_cp else None

    # Preserve compact valid-token boundaries separately from physical padded
    # boundaries. Attention consumes both and uses pad_between_seqs when they differ.
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        local_cp_size=local_cp_size,
        cp_group=cp_group,
        pad_between_seqs=(True if not torch.equal(cu_seqlens, cu_seqlens_padded) else None),
    )

    # "attention_mask" is not valid for sequence packing, so set it to None.
    return tokens, labels, loss_mask, None, position_ids, packed_seq_params, padding_mask
