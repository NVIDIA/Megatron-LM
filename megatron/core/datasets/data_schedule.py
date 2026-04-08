# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, List, Optional, Type

import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule_utils import (
    align_sample_id_groups,
    broadcast_scalars,
    broadcast_tensor,
    broadcast_to_pp_group,
    build_packed_microbatches,
    create_data_iterator,
    dcp_get_total_workload,
    dcp_gpus_needed,
    dcp_make_buckets_equal,
    get_batch_and_global_seqlens,
    get_cp_slice_for_thd,
    next_hdp_group,
    reroute_samples_to_dcp_ranks,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.hybrid_cp_schedule import BalancedCPScheduler
from megatron.core.process_groups_config import ProcessGroupCollection


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

        Steps:
            1. Fetch batches and gather global sequence lengths
            2. Check required sample keys
            3. Schedule samples into groups
            4. Reroute samples to DCP ranks
            5. Build packed microbatches
            6. Calculate FLOPs info
            7. Broadcast to PP group (for middle PP stages)
            8. Broadcast to TP group (for non-TP-0 ranks)
            9. Handle VPP if enabled

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

        # Handle VPP: extract the correct data_iterator for this PP stage.
        # When VPP is enabled, data_iterator is a list with one entry per VPP stage.
        # We only need one data_iterator to run the schedule (all VPP stages on the
        # same PP rank share the same underlying dataset), so pick the first non-None.
        # Record which VPP stages had data so create_data_iterator knows which ones
        # need full samples vs metadata only.
        vpp_has_data = None
        if (
            config.virtual_pipeline_model_parallel_size is not None
            and config.virtual_pipeline_model_parallel_size > 1
        ):
            assert len(data_iterator) == config.virtual_pipeline_model_parallel_size
            vpp_has_data = [di is not None for di in data_iterator]
            extracted = None
            for di in data_iterator:
                if di is not None:
                    extracted = di
                    break
            data_iterator = extracted

        # data_iterator is not None on TP rank 0 for PP stages that need data
        # (first stage, last stage, or any stage with MTP).
        if data_iterator is not None:
            assert tp_group.rank() == 0, "Only TP rank 0 should have data_iterator"

            # Step 1: Fetch batches and gather global sequence lengths
            batch, global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered = (
                get_batch_and_global_seqlens(data_iterator, num_microbatches, dp_group)
            )

            # Step 2: Check required sample keys
            for key in self.get_required_sample_keys():
                assert (
                    key in batch[0]
                ), f"Batch missing required key {key}, provided keys: {batch[0].keys()}"

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
                tp_group,
                dp_cp_group,
                total_dcp_gpus,
            )

            dcp_rank = dp_cp_group.rank()
            num_micro_batches = len(sample_id_groups)

            # Step 5: Build packed microbatches
            new_samples = build_packed_microbatches(
                samples_this_rank_with_id, sample_id_groups, dcp_rank, dev, self.is_dynamic_cp
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

        # Step 7: Broadcast to PP group (for middle PP stages)
        if tp_group.rank() == 0:
            (
                new_samples,
                num_micro_batches,
                seqlen_sum_this_global_batch,
                seqlen_squared_sum_this_global_batch,
            ) = broadcast_to_pp_group(
                new_samples,
                num_micro_batches,
                seqlen_sum_this_global_batch,
                seqlen_squared_sum_this_global_batch,
                pp_group,
                dev,
                is_dynamic_cp=self.is_dynamic_cp,
            )

        # Step 8: Broadcast to TP group (for non-TP-0 ranks)
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

        # Step 9: create data_iterator and handle VPP if enabled
        new_data_iterator = create_data_iterator(
            new_samples, tp_group, config, vpp_has_data, self.is_dynamic_cp
        )

        return (
            new_data_iterator,
            num_micro_batches,
            seqlen_sum_this_global_batch,
            seqlen_squared_sum_this_global_batch,
        )


class DefaultDynamicCPScheduler(DpBalancedScheduler):
    """
    Dynamic CP scheduler that balances workload across variable CP sizes.
    """

    def __init__(self, *args, min_cp_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_dynamic_cp = True
        self.max_seq_len_per_rank = self.max_seqlen_per_dp_cp_rank
        self.total_hdp_gpus = self.dp_size * self.cp_size
        self.min_cp_size = min_cp_size

    def get_groups_and_subsamples(self, sample_id_seqlens):
        """
        This function recursively forms groups of sub-samples such that all DPxCP ranks
        have a roughly balanced workload in the group.
        """
        mslpr = self.max_seq_len_per_rank
        min_cp = self.min_cp_size
        workload_fn = lambda seq_len, cp_size=None: dcp_get_total_workload(
            seq_len, mslpr, cp_size, min_cp
        )
        gpus_fn = lambda seq_len: dcp_gpus_needed(seq_len, mslpr, min_cp)
        buckets_fn = lambda sample_seqlens, compute_est: dcp_make_buckets_equal(
            sample_seqlens, compute_est, mslpr, min_cp
        )

        groups = []
        sample_id_groups = []
        sample_id_seqlens = sorted(sample_id_seqlens, key=lambda x: x[1], reverse=True)
        while sample_id_seqlens:
            mb, sample_id_seqlens, exec_times, sample_ids = next_hdp_group(
                sample_id_seqlens,
                workload_fn,
                self.total_hdp_gpus,
                gpus_needed_fn=gpus_fn,
                make_buckets_equal_fn=buckets_fn,
                max_seq_len_per_rank=mslpr,
                get_total_workload_fn=workload_fn,
            )
            groups.append(mb)
            sample_id_groups.append(sample_ids)

        if (
            self.microbatch_group_size_per_vp_stage is not None
            and self.microbatch_group_size_per_vp_stage > 1
        ):
            sample_id_groups = align_sample_id_groups(
                sample_id_groups, self.microbatch_group_size_per_vp_stage
            )

        return sample_id_groups


scheduler_map: Dict[str, Type[BasePackingScheduler]] = {
    "dp_balanced": DpBalancedScheduler,
    "default_dynamic_cp": DefaultDynamicCPScheduler,
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

    # Look up the scheduler class by name
    scheduler_type = config.sequence_packing_scheduler

    scheduler_kwargs = {}
    if scheduler_type == 'default_dynamic_cp':
        scheduler_kwargs['min_cp_size'] = config.min_dynamic_context_parallel_size

    scheduler = scheduler_map[scheduler_type](
        config.max_seqlen_per_dp_cp_rank,
        cp_size,
        dp_size,
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
):
    """
    Get a batch of data for sequence packing.
    Args:
        data_iterator (Iterator): The data iterator to get the batch from.
        mtp_on_this_rank (bool): Whether to use multi-token prediction.
        vp_stage (Optional[int]): The stage of the pipeline.
    Returns:
        tuple of (tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params)
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

    # For dynamic CP, determine the correct cp_group from batch on TP rank 0.
    if dynamic_cp and is_tp_rank_0:
        local_cp_size_val = batch['local_cp_size']
        if isinstance(local_cp_size_val, torch.Tensor):
            local_cp_size_val = local_cp_size_val.item()
        cp_group = parallel_state.get_dynamic_data_context_parallel_groups(
            group_size=local_cp_size_val
        )

    # Partition tokens, position_ids, labels, loss_mask for context parallel.
    # Only TP rank 0 on stages that have data (first/last PP stage or MTP stage) needs this.
    if is_tp_rank_0 and (is_first_or_last_stage or mtp_on_this_rank):
        get_cp_slice_for_thd(batch, cp_group)

    # Broadcast cu_seqlens_size because we need it to create placeholder for cu_seqlens and
    # cu_seqlens_padded for non TP 0 ranks.
    if is_tp_rank_0:
        cu_seqlen_size = torch.tensor(batch['cu_seqlens'].size(0), dtype=torch.int32, device=dev)
    else:
        cu_seqlen_size = torch.empty(1, dtype=torch.int32, device=dev)
    broadcast_tensor(cu_seqlen_size, tp_src_rank, tp_group)
    cu_seqlen_size = cu_seqlen_size.item()

    # Broadcast total_tokens because we need it to create placeholder for tokens, position_ids,
    # labels, loss_mask for non TP 0 ranks. Only first stage, last stage,
    # and stage with mtp need this.

    if is_first_or_last_stage or mtp_on_this_rank:
        if is_tp_rank_0:
            total_tokens = torch.tensor(batch['tokens'].size(0), dtype=torch.int32, device=dev)
        else:
            total_tokens = torch.empty(1, dtype=torch.int32, device=dev)
        broadcast_tensor(total_tokens, tp_src_rank, tp_group)
        total_tokens = total_tokens.item()

    # Step1: Prepare "tokens", "position_ids" for first stage and stage with mtp on all TP ranks.
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

    # Step2: Prepare "labels", "loss_mask" for last stage and stage with mtp on all TP ranks.
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

    # Step3: Prepare "cu_seqlens", "cu_seqlens_padded", "max_seqlen" on all ranks.
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

    # Step4: Prepare "local_cp_size" if dynamic context parallel is enabled.
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
    broadcast_tensor(batch['cu_seqlens'], tp_src_rank, tp_group)
    broadcast_tensor(batch['cu_seqlens_padded'], tp_src_rank, tp_group)
    broadcast_tensor(batch['max_seqlen'], tp_src_rank, tp_group)
    broadcast_tensor(batch['local_cp_size'], tp_src_rank, tp_group)

    # Extract the data from batch after broadcasting.
    tokens = batch['tokens']
    position_ids = batch['position_ids']
    labels = batch['labels']
    loss_mask = batch['loss_mask']
    cu_seqlens = batch['cu_seqlens']
    cu_seqlens_padded = batch['cu_seqlens_padded']
    max_seqlen = batch['max_seqlen'].item()
    local_cp_size = batch['local_cp_size'].item() if dynamic_cp else None
    cp_group = (
        parallel_state.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)
        if dynamic_cp
        else None
    )

    # Transformer Engine has a bug of cu_seqlens, we must treat cu_seqlens_padded as cu_seqlens to
    # get the correct result.
    # TODO: Revert this workaround once TE fixes the issue.
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        local_cp_size=local_cp_size,
        cp_group=cp_group,
    )

    # "attention_mask" is not valid for sequence packing, so set it to None.
    return tokens, labels, loss_mask, None, position_ids, packed_seq_params


class HybridCPDataLoaderWrapper:
    """
    A wrapper class that wraps around an existing data_iterator.
    For every __next__ call,
    1. Each DP rank pulls a batch of packed samples.
    2. Extracts the sequence lengths of each sub-sample and all-gathers across the DP group.
    3. Schedules the sub-samples to the DPxCP ranks using the BalancedCPScheduler.
    4. Based on the schedule, reroutes the sub-samples to the correct rank using all-to-all.
    5. Returns the assigned sub-samples to this rank.

    Args:
        data_iterator: The original data_iterator to wrap around
        config: The config object containing the max_seqlen_per_dp_cp_rank
        dp_cp_group: Data parallel context parallel group.
    """

    def __init__(
        self, data_iterator, config, pg_collection: Optional[ProcessGroupCollection] = None
    ):
        self.data_iterator = data_iterator
        self.config = config
        if pg_collection is None:
            self.dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
            self.dp_group = parallel_state.get_data_parallel_group()
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            self.dp_cp_group = pg_collection.dp_cp
            self.dp_group = pg_collection.dp
            self.tp_group = pg_collection.tp
        assert (
            self.dp_cp_group is not None and self.dp_group is not None and self.tp_group is not None
        ), "dp_cp_group, dp_group, tp_group must not be None when using hybrid context parallel"

        self.cp_balancing_scheduler = BalancedCPScheduler(
            max_seq_len_per_rank=self.config.max_seqlen_per_dp_cp_rank, dp_cp_group=self.dp_cp_group
        )

        self.total_hdp_gpus = self.dp_cp_group.size()

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def get_global_seqlens(self, subsample_seqlens: torch.Tensor) -> List[int]:
        """Gather sequence lengths of all subsamples from all DP ranks."""
        local_len = torch.tensor([subsample_seqlens.shape[0]], dtype=torch.int32).cuda()
        dp_subsample_count = [torch.zeros_like(local_len) for _ in range(self.dp_group.size())]
        torch.distributed.all_gather(dp_subsample_count, local_len, group=self.dp_group)

        dp_subsample_counts = torch.stack(dp_subsample_count, dim=0).cpu().view(-1)
        max_sub_samples = int(dp_subsample_counts.max().item())

        if local_len.item() < max_sub_samples:
            subsample_seqlens_padded = torch.cat(
                [
                    subsample_seqlens,
                    torch.zeros(max_sub_samples - local_len.item(), dtype=torch.int32).cuda(),
                ],
                dim=0,
            )
        else:
            subsample_seqlens_padded = subsample_seqlens

        seqlens_gathered = [
            torch.empty_like(subsample_seqlens_padded) for _ in range(self.dp_group.size())
        ]
        torch.distributed.all_gather(
            seqlens_gathered, subsample_seqlens_padded, group=self.dp_group
        )

        for dp_rank, seqlen in enumerate(seqlens_gathered):
            seqlens_gathered[dp_rank] = seqlen[: dp_subsample_counts[dp_rank]]

        seqlens_gathered = torch.cat(seqlens_gathered, dim=0)
        seqlens_gathered = seqlens_gathered.cpu().tolist()

        csum = torch.cumsum(dp_subsample_counts, dim=0, dtype=torch.int32)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum[:-1]], dim=0)

        return seqlens_gathered, offsets

    def get_global_id_seqlens(self, num_local_subsamples, offsets, seqlens_gathered):
        """Calculate global IDs for each subsample."""
        dp_rank = self.dp_group.rank()
        global_ids = torch.arange(len(seqlens_gathered), dtype=torch.int32).cuda()
        global_id_seqlens = [(i, seqlens_gathered[i]) for i in range(len(global_ids))]
        global_ids_this_rank = global_ids[
            offsets[dp_rank] : offsets[dp_rank] + num_local_subsamples
        ]
        return global_id_seqlens, global_ids_this_rank

    def _gid_to_src_rank(self, gid: int, offsets: List[int]) -> int:
        """Map a global ID to its source HDP rank."""
        dp_src_rank = torch.bucketize(gid, offsets[1:] - 1)
        hdp_rank = (
            torch.distributed.get_process_group_ranks(self.dp_group)[dp_src_rank]
            // self.tp_group.size()
        )
        return hdp_rank

    def reroute_samples_to_hdp_ranks(
        self, batch, global_ids_this_rank, global_id_seqlens, sample_id_groups, offsets
    ):
        """Reroute sub-samples to correct ranks after scheduling."""
        gid2local_id = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
        hdp_rank = self.dp_cp_group.rank()
        dp_ranks = torch.distributed.get_process_group_ranks(self.dp_group)
        dp_ranks = [r // self.tp_group.size() for r in dp_ranks]

        data_keys = batch[0].keys()

        combined_sample_id_groups: List[List[int]] = [[] for _ in range(self.total_hdp_gpus)]

        for d in range(self.total_hdp_gpus):
            for sample_id_group in sample_id_groups:
                combined_sample_id_groups[d].extend(sample_id_group[d])

        for dest_rank in range(self.total_hdp_gpus):
            combined_sample_id_groups[dest_rank].sort()

        send_ids_sorted = [
            gid
            for d in dp_ranks
            for gid in combined_sample_id_groups[d]
            if gid in global_ids_this_rank
        ]

        send_lens_split = [0] * self.total_hdp_gpus
        for dest_rank in range(self.total_hdp_gpus):
            if dest_rank in dp_ranks:
                send_lens_split[dest_rank] = sum(
                    [
                        global_id_seqlens[gid][1]
                        for gid in combined_sample_id_groups[dest_rank]
                        if gid in global_ids_this_rank
                    ]
                )
            else:
                send_lens_split[dest_rank] = 0

        recv_sample_id_groups = [[] for _ in range(self.total_hdp_gpus)]
        for gid in combined_sample_id_groups[hdp_rank]:
            src_rank = self._gid_to_src_rank(gid, offsets)
            recv_sample_id_groups[src_rank].append(gid)

        recv_lens_split = [0] * self.total_hdp_gpus
        for src_rank in range(self.total_hdp_gpus):
            recv_lens_split[src_rank] = sum(
                [global_id_seqlens[gid][1] for gid in recv_sample_id_groups[src_rank]]
            )

        recv_ids_sorted = [
            gid for d in range(self.total_hdp_gpus) for gid in recv_sample_id_groups[d]
        ]
        recv_counts = [len(recv_sample_id_groups[d]) for d in range(self.total_hdp_gpus)]

        recv_samples = [{k: None for k in data_keys} for _ in range(sum(recv_counts))]

        def _pack_sample_by_key(key: str) -> torch.Tensor:
            flattened_tensors = []
            for gid in send_ids_sorted:
                t = batch[gid2local_id[gid]][key].to(torch.cuda.current_device(), non_blocking=True)
                flattened_tensors.append(t)
            return (
                torch.cat(flattened_tensors, dim=0)
                if flattened_tensors
                else torch.empty(0, device=torch.cuda.current_device(), dtype=batch[0][key].dtype)
            )

        def _unpack_sample_by_key(key: str, recv_tensor: torch.Tensor):
            cursor = 0
            for i, gid in enumerate(recv_ids_sorted):
                sample_len = global_id_seqlens[gid][1]
                recv_samples[i][key] = recv_tensor[cursor : cursor + sample_len]
                cursor += sample_len

        for key in data_keys:
            send_tensor = _pack_sample_by_key(key)
            recv_tensor = torch.empty(
                sum(recv_lens_split), device=torch.cuda.current_device(), dtype=send_tensor.dtype
            )
            torch.distributed.all_to_all_single(
                output=recv_tensor,
                input=send_tensor,
                output_split_sizes=recv_lens_split,
                input_split_sizes=send_lens_split,
                group=self.dp_cp_group,
            )
            _unpack_sample_by_key(key, recv_tensor)

        recv_sample_with_id = {
            recv_id: recv_samples[i] for i, recv_id in enumerate(recv_ids_sorted)
        }
        return recv_sample_with_id

    def unpack_batch(self, batch):
        """Unpack packed samples into a list of sub-samples."""
        batch_unpacked = []
        for sample in batch:
            for sub_sample in range(sample["cu_seqlens"].shape[0] - 1):
                sub_sample_dict = {}
                start_idx = sample["cu_seqlens"][sub_sample]
                end_idx = sample["cu_seqlens"][sub_sample + 1]
                if end_idx - start_idx == 0:
                    continue
                for key in sample.keys():
                    if key in ["cu_seqlens", "batch_idx", "max_seqlen"]:
                        continue
                    sub_sample_dict[key] = sample[key][start_idx:end_idx]
                batch_unpacked.append(sub_sample_dict)
        return batch_unpacked

    def __next__(self) -> Any:
        """Get the next batch, schedule, and reroute sub-samples."""
        if self.data_iterator is None:
            return None, None
        else:
            batch = next(self.data_iterator)
        subsample_seqlens = []
        for sample in batch:
            subsample_seqlens.extend(
                [
                    int(sample["cu_seqlens"][i + 1] - sample["cu_seqlens"][i])
                    for i in range(0, sample["cu_seqlens"].shape[0] - 1)
                ]
            )
        subsample_seqlens = torch.tensor(subsample_seqlens, dtype=torch.int32).cuda()
        subsample_seqlens = subsample_seqlens[subsample_seqlens != 0]

        seqlens_gathered, offsets = self.get_global_seqlens(subsample_seqlens)

        global_id_seqlens, global_ids_this_rank = self.get_global_id_seqlens(
            subsample_seqlens.shape[0], offsets, seqlens_gathered
        )

        groups, sample_id_groups = self.cp_balancing_scheduler.get_groups_and_subsamples(
            global_id_seqlens, self.config
        )

        batch = self.unpack_batch(batch)
        samples_this_rank_with_id = self.reroute_samples_to_hdp_ranks(
            batch, global_ids_this_rank, global_id_seqlens, sample_id_groups, offsets
        )
        return samples_this_rank_with_id, sample_id_groups
