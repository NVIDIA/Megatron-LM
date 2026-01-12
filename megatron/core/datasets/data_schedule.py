# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

import enum
from collections import deque
from functools import lru_cache
from math import ceil, log2
from typing import Callable, Deque, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.utils import is_te_min_version

try:
    # Register the TE CUDA kernels
    import transformer_engine  # pylint: disable=unused-import

    # Alias the PyTorch wrapper so we can call tex.* APIs
    import transformer_engine_torch as tex
except ImportError:
    # TE isn’t installed or the torch wrapper is missing
    tex = None


class PackingScheduler(enum.Enum):
    """Enum for supported sequence packing algorithms."""

    # custom hybrid-cp scheduler, schedule in samplers, only need to pack
    EMPTY_PACKING = "empty_scheduler_with_packing"
    # custom hybrid-cp scheduler, schedule in samplers and pack in collate_fn
    EMPTY_NO_PACKING = "empty_scheduler_no_packing"
    NAIVE_SEQUENCE_PACKING = "naive_sequence_packing"
    DEFAULT_HYBRID_CP = "default_hybrid_cp"


def wrap_dataloader(
    data_iterator,
    config,
    scheduler_type: Union[PackingScheduler, str],
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """
    A wrapper function that wraps around an existing data_iterator
    and return the num_micro_batches for sequence packing.

    Args:
        data_iterator: The original data_iterator to wrap around
        config: The config object containing the max_seqlen_per_dp_cp_rank
        dp_cp_group: Data parallel context parallel group.
    """

    scheduler_map: Dict[PackingScheduler, Type[BaseScheduler]] = {
        PackingScheduler.DEFAULT_HYBRID_CP: DefaultHybridCPscheduler,
        PackingScheduler.NAIVE_SEQUENCE_PACKING: NaiveSequencePackingScheduler,
        PackingScheduler.EMPTY_PACKING: EmptyPackingScheduler,
        PackingScheduler.EMPTY_NO_PACKING: EmptyNoPackingScheduler,
    }

    def _get_global_seqlens(subsample_seqlens: torch.Tensor, dp_group) -> List[int]:
        """
        Gathers the sequence lengths of all subsamples from all DP ranks.
        Each DP rank loads the same number of microbatches but each microbatch
        may have a different number of subsamples.

        We find the number of subsamples each rank holds and then gather the
        sequence lengths of all subsamples from all ranks.
        """
        # Collect the number of subsamples from all ranks
        local_len = torch.tensor([subsample_seqlens.shape[0]], dtype=torch.int32).cuda()
        dp_subsample_count = [torch.zeros_like(local_len) for _ in range(dp_group.size())]
        torch.distributed.all_gather(dp_subsample_count, local_len, group=dp_group)

        # Find the max number of subsamples across all ranks and pad subsample_seqlens to max length
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

        # Gather the subsample_seqlens from all ranks
        seqlens_gathered = [
            torch.empty_like(subsample_seqlens_padded) for _ in range(dp_group.size())
        ]
        torch.distributed.all_gather(seqlens_gathered, subsample_seqlens_padded, group=dp_group)

        # Trim each seqlens_gathered to the length of the correct sample
        for dp_rank, seqlen in enumerate(seqlens_gathered):
            seqlens_gathered[dp_rank] = seqlen[: dp_subsample_counts[dp_rank]]

        seqlens_gathered = torch.cat(seqlens_gathered, dim=0)
        seqlens_gathered = seqlens_gathered.cpu().tolist()

        # Calculate the offsets to assign unique global ID to each subsample.
        csum = torch.cumsum(dp_subsample_counts, dim=0, dtype=torch.int32)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum[:-1]], dim=0)

        return seqlens_gathered, offsets

    def _get_global_id_seqlens(num_local_subsamples, offsets, seqlens_gathered, dp_group):
        """
        Calculates the global ID for each subsample.

        We assign a unique global ID to each subsample.

        Returns:
        global_id_seqlens: list of (global_id, seqlen) tuples for scheduling.
        global_ids_this_rank: list of global IDs locally present on this rank.
        """
        dp_rank = dp_group.rank()
        global_ids = torch.arange(len(seqlens_gathered), dtype=torch.int32).cuda()
        # Create a list of (global_id, seqlen) tuples for scheduling
        global_id_seqlens = [(i, seqlens_gathered[i]) for i in range(len(global_ids))]
        # Get the global IDs locally present on this rank
        global_ids_this_rank = global_ids[
            offsets[dp_rank] : offsets[dp_rank] + num_local_subsamples
        ]

        return global_id_seqlens, global_ids_this_rank

    def _gid_to_src_rank(gid: int, offsets: List[int], dp_group, tp_group, dp_cp_group) -> int:
        dp_src_rank = torch.bucketize(gid, offsets[1:] - 1)
        # Since the torch.distributed.get_process_group_ranks
        # provides the global rank, we need to consider TP
        hdp_rank = (
            torch.distributed.get_process_group_ranks(dp_group)[dp_src_rank] // tp_group.size()
        ) % dp_cp_group.size()
        return hdp_rank

    def _reroute_samples_to_hdp_ranks(
        batch,
        global_ids_this_rank,
        global_id_seqlens,
        sample_id_groups,
        offsets,
        dp_group,
        tp_group,
        dp_cp_group,
        total_hdp_gpus,
    ):
        """
        Reroutes the sub-samples to the correct rank after scheduling.

        For each key in the batch dict, we perform an all-to-all communication
        to transfer the data to the correct ranks.
        Since all CP ranks within a DP group have the same data, we only need
        to transfer data between matching CP ranks.
        """
        gid2local_id = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
        hdp_rank = dp_cp_group.rank()
        dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
        # Here we actually want to get the DP group's rank within the HDP group,
        # we need to consider TP
        # tp-cp-ep-dp-pp
        dp_ranks = [(r // tp_group.size()) % dp_cp_group.size() for r in dp_ranks]

        data_keys = batch[0].keys()

        # Create the send plan
        combined_sample_id_groups: List[List[int]] = [[] for _ in range(total_hdp_gpus)]

        for d in range(total_hdp_gpus):
            for sample_id_group in sample_id_groups:
                combined_sample_id_groups[d].extend(sample_id_group[d])

        for dest_rank in range(total_hdp_gpus):
            combined_sample_id_groups[dest_rank].sort()

        # Filter out samples that are not present on this rank
        send_ids_sorted = [
            gid
            for d in dp_ranks
            for gid in combined_sample_id_groups[d]
            if gid in global_ids_this_rank
        ]
        # send_counts = [len(combined_sample_id_groups[d]) for d in range(total_hdp_gpus)]

        send_num_split = [0] * total_hdp_gpus
        send_lens_split = [0] * total_hdp_gpus
        for dest_rank in range(total_hdp_gpus):
            if dest_rank in dp_ranks:
                send_seq_lens = [
                    global_id_seqlens[gid][1]
                    for gid in combined_sample_id_groups[dest_rank]
                    if gid in global_ids_this_rank
                ]
                send_num_split[dest_rank] = len(send_seq_lens)
                send_lens_split[dest_rank] = sum(send_seq_lens)
            else:
                # We only need to share local data with DP ranks that have different data.
                send_lens_split[dest_rank] = 0

        # Create the recv plan
        recv_sample_id_groups = [[] for _ in range(total_hdp_gpus)]
        for gid in combined_sample_id_groups[hdp_rank]:
            src_rank = _gid_to_src_rank(gid, offsets, dp_group, tp_group, dp_cp_group)
            recv_sample_id_groups[src_rank].append(gid)

        recv_lens_split = [0] * total_hdp_gpus
        for src_rank in range(total_hdp_gpus):
            recv_lens_split[src_rank] = sum(
                [global_id_seqlens[gid][1] for gid in recv_sample_id_groups[src_rank]]
            )

        recv_ids_sorted = [gid for d in range(total_hdp_gpus) for gid in recv_sample_id_groups[d]]
        recv_counts = [len(recv_sample_id_groups[d]) for d in range(total_hdp_gpus)]

        recv_samples = [{k: None for k in data_keys} for _ in range(sum(recv_counts))]

        def _pack_sample_by_key(key: str) -> torch.Tensor:
            flattened_tensors = []
            for gid in send_ids_sorted:
                t = batch[gid2local_id[gid]][key].to(torch.cuda.current_device(), non_blocking=True)
                # flattened_tensors.append(t)
                flattened_tensors.append(t.reshape(-1))
            return (
                torch.cat(flattened_tensors, dim=0)
                if flattened_tensors
                else torch.empty(1, device=torch.cuda.current_device(), dtype=batch[0][key].dtype)
            )

        def _unpack_sample_by_key(key: str, recv_tensor: torch.Tensor):
            cursor = 0
            for i, gid in enumerate(recv_ids_sorted):
                sample_len = (
                    1
                    if key in ["original_seq_len", "padded_seq_len"]
                    else global_id_seqlens[gid][1]
                )
                recv_samples[i][key] = recv_tensor[cursor : cursor + sample_len]
                cursor += sample_len

        for key in data_keys:
            output_split_sizes, input_split_sizes = (
                (recv_counts, send_num_split)
                if key in ["original_seq_len", "padded_seq_len"]
                else (recv_lens_split, send_lens_split)
            )
            send_tensor = _pack_sample_by_key(key)
            recv_tensor_size = sum(output_split_sizes)
            recv_tensor = torch.empty(
                recv_tensor_size, device=torch.cuda.current_device(), dtype=send_tensor.dtype
            )
            torch.distributed.all_to_all_single(
                output=recv_tensor,
                input=send_tensor,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=dp_cp_group,
            )
            _unpack_sample_by_key(key, recv_tensor)

        recv_sample_with_id = {
            recv_id: recv_samples[i] for i, recv_id in enumerate(recv_ids_sorted)
        }
        return recv_sample_with_id

    def _unpack_batch(batch):
        """
        Unpacks the packed samples into a list of sub-samples.
        Since each sub-sample may be routed to different DPxCP ranks,
        we unpack the sample here to avoid unnecessarily transferring
        the entire packed sample.
        """
        batch_unpacked = []
        for sample in batch:
            sample_dict = {}
            for key in sample.keys():
                if key in ["cu_seqlens", "batch_idx", "max_seqlen"]:
                    continue
                sample_dict[key] = sample[key]
            batch_unpacked.append(sample_dict)
        return batch_unpacked

    def _broadcast_to_tp_group(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                parallel_state.get_tensor_model_parallel_src_rank(),
                group=parallel_state.get_tensor_model_parallel_group(),
            )

    def _broadcast_to_pp_group(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                parallel_state.get_pipeline_model_parallel_first_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
            )

    def _pack_sequences(
        samples: List,
        local_cp_size: torch.Tensor,
        padded_lengths: torch.Tensor,
        original_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        def _pack_tensors(tensors):
            return torch.cat([t.reshape(-1) for t in tensors], dim=0)

        tokens = _pack_tensors([sample["tokens"] for sample in samples])
        labels = _pack_tensors([sample["labels"] for sample in samples])
        loss_mask = _pack_tensors([sample["loss_mask"] for sample in samples])
        position_ids = _pack_tensors([sample["position_ids"] for sample in samples])

        new_sample = {}
        new_sample["tokens"] = tokens
        new_sample["labels"] = labels
        new_sample["loss_mask"] = loss_mask
        new_sample["position_ids"] = position_ids
        if local_cp_size is not None:
            # Accept either a Python int or a CUDA scalar tensor; keep as a scalar tensor on GPU.
            new_sample["local_cp_size"] = local_cp_size.to(device=dev, dtype=torch.int32)

        padded_lengths = padded_lengths.to(
            device=dev, dtype=torch.int32, non_blocking=True
        ).reshape(-1)
        cu_seqlens_padded = torch.empty(padded_lengths.numel() + 1, device=dev, dtype=torch.int32)
        cu_seqlens_padded[0] = 0
        cu_seqlens_padded[1:] = torch.cumsum(padded_lengths, dim=0)
        max_seqlen = torch.max(padded_lengths).to(dtype=torch.int32)

        new_sample["cu_seqlens_padded"] = cu_seqlens_padded
        new_sample["max_seqlen"] = max_seqlen

        # create cu_seqlens without padding

        original_lengths = original_lengths.to(
            device=dev, dtype=torch.int32, non_blocking=True
        ).reshape(-1)
        cu_seqlens = torch.empty(original_lengths.numel() + 1, device=dev, dtype=torch.int32)
        cu_seqlens[0] = 0
        cu_seqlens[1:] = torch.cumsum(original_lengths, dim=0).reshape(-1)
        new_sample["cu_seqlens"] = cu_seqlens

        return new_sample

    def _build_packed_microbatches(
        grouped_samples: List[List[Dict[str, torch.Tensor]]],
        scheduler_type: PackingScheduler,
        local_cp_sizes_gpu: Optional[torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Build packed samples for each microbatch given a pre-built list of `samples` per microbatch.

        Args:
            grouped_samples: List of length `num_microbatches`. Each element is the `samples` list
                (list[sample]) for that microbatch, where `sample` is the dict returned by
                `dataset.__getitem__`.
            scheduler_type: packing scheduler.
            local_cp_sizes_gpu: CUDA int32 tensor of shape [num_micro_batches]
            when DEFAULT_HYBRID_CP, otherwise None.

        Returns:
            new_samples: list of packed samples (dicts) length == num_micro_batches.
        """
        num_micro_batches = len(grouped_samples)
        seg_starts: List[int] = [0]
        original_lens_tensors = []
        padded_lens_tensors = []

        for i in range(num_micro_batches):
            samples = grouped_samples[i]
            seg_starts.append(seg_starts[-1] + len(samples))
            original_lens_tensors.extend([s["original_seq_len"].reshape(-1) for s in samples])
            padded_lens_tensors.extend([s["padded_seq_len"].reshape(-1) for s in samples])

        padded_lens_all_gpu = torch.cat(padded_lens_tensors, dim=0).to(dtype=torch.int32)
        original_lens_all_gpu = torch.cat(original_lens_tensors, dim=0).to(dtype=torch.int32)

        new_samples: List[Dict[str, torch.Tensor]] = []
        for i in range(num_micro_batches):
            samples = grouped_samples[i]

            lp = padded_lens_all_gpu[seg_starts[i] : seg_starts[i + 1]]
            lo = original_lens_all_gpu[seg_starts[i] : seg_starts[i + 1]]

            if scheduler_type is PackingScheduler.DEFAULT_HYBRID_CP:
                assert local_cp_sizes_gpu is not None
                partner_cp_arg = local_cp_sizes_gpu[i]
            else:
                partner_cp_arg = None

            new_sample = _pack_sequences(samples, partner_cp_arg, lp, lo)
            new_samples.append(new_sample)

        return new_samples

    # Convert string to enum if needed
    if isinstance(scheduler_type, str):
        try:
            scheduler_type = PackingScheduler[scheduler_type.upper()]
        except KeyError:
            available_scheduler = ", ".join([scheduler.name for scheduler in PackingScheduler])
            raise ValueError(
                f"Unknown packing scheduler: {scheduler_type}. "
                f"Available schedulers: {available_scheduler}"
            )

    if scheduler_type not in scheduler_map:
        available_scheduler = ", ".join([scheduler.name for scheduler in PackingScheduler])
        raise ValueError(
            f"Unknown scheduler: {scheduler}. " f"Available schedulers: {available_scheduler}"
        )

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
        dp_cp_group is not None and dp_group is not None and tp_group is not None
    ), "dp_cp_group, dp_group, tp_group must not be None when using hybrid context parallel"

    total_hdp_gpus = dp_cp_group.size()
    dev = torch.cuda.current_device()

    scheduler = scheduler_map[scheduler_type](
        config.max_seqlen_per_dp_cp_rank,
        parallel_state.get_context_parallel_world_size(),
        parallel_state.get_data_parallel_world_size(),
        # When VPP is enabled, align num_micro_batches to this multiple.
        (
            None
            if config.virtual_pipeline_model_parallel_size is None
            else config.microbatch_group_size_per_vp_stage
        ),
        config.hybrid_context_parallel,
    )
    if (
        config.virtual_pipeline_model_parallel_size is not None
        and config.virtual_pipeline_model_parallel_size > 1
    ):
        if pp_group.rank() == pp_group.size() - 1:
            assert len(data_iterator) == config.virtual_pipeline_model_parallel_size
            data_iterator = data_iterator[-1]
        else:
            data_iterator = data_iterator[0]

    if data_iterator is not None:
        batch = next(data_iterator)

        # check if the batch contains all the required keys
        assert scheduler.check_require_sample_keys(batch), "Batch missing required keys"
        # indicates TP rank 0, with PP stage 0 or -1.
        if scheduler_type is PackingScheduler.EMPTY_PACKING:
            # EMPTY scheduler does not schedule the data,
            # just packing sequences

            # Here, next(data_iterator) returns multiple samples, i.e., a list[sample].
            # Each sample is the value returned by dataset.__getitem__ (see
            # `megatron/training/datasets/sft_dataset.py` for the concrete sample fields).
            # This indicates that, according to the scheduler's result,
            #  these samples (sequences) should be packed together.
            num_micro_batches = batch[0]["num_micro_batches_left"] + 1

            batch_all = [batch] + [next(data_iterator) for _ in range(num_micro_batches - 1)]

            num_total_tokens = 0
            sequence_square_sum = 0

            # pack sequences in the same group and create a new data iterator
            new_samples = []
            for samples in batch_all:
                # In EMPTY scheduler, scheduler has already selected the grouping and
                # provides `local_cp_size` for each packed group.
                local_cp_size = samples[0]["local_cp_size"]
                # Convert local_cp_size to a python int for FLOPs accounting
                local_cp_size_int = (
                    int(local_cp_size.item())
                    if isinstance(local_cp_size, torch.Tensor)
                    else int(local_cp_size)
                )

                # Build per-sub-sample length tensors on GPU so `_pack_sequences` can compute
                # cu_seqlens and cu_seqlens_padded via GPU cumsum.
                padded_lengths = torch.cat(
                    [s["padded_seq_len"].reshape(-1) for s in samples], dim=0
                )
                original_lengths = torch.cat(
                    [s["original_seq_len"].reshape(-1) for s in samples], dim=0
                )

                new_sample = _pack_sequences(
                    samples, local_cp_size, padded_lengths, original_lengths
                )
                new_samples.append(new_sample)
                for sample in samples:
                    n = sample["tokens"].numel()
                    # tokens.numel() is a python int (no D2H). local_cp_size_int is python int.
                    num_total_tokens += n / local_cp_size_int
                    sequence_square_sum += n**2 / local_cp_size_int
            # allreduce to get the total number of microbatches
            flops_info_to_broadcast_this_hdp_group = torch.tensor(
                [num_total_tokens, sequence_square_sum], dtype=torch.float32, device=dev
            )
            torch.distributed.all_reduce(flops_info_to_broadcast_this_hdp_group, group=dp_cp_group)
            flops_info_cpu = flops_info_to_broadcast_this_hdp_group.cpu().numpy()
            num_total_tokens_this_global_batch = flops_info_cpu[0]
            sequence_square_sum_this_global_batch = flops_info_cpu[1]

        elif scheduler_type is PackingScheduler.EMPTY_NO_PACKING:
            # EMPTY_NO_PACKING scheduler does not schedule the data,
            # sequences are already packed, we just need to return the batch
            num_micro_batches = batch["num_micro_batches_left"] + 1
            num_total_tokens_this_global_batch = batch["num_total_tokens_this_global_batch"]
            sequence_square_sum_this_global_batch = batch["sequence_square_sum_this_global_batch"]
            new_samples = [batch] + [next(data_iterator) for _ in range(num_micro_batches - 1)]

        elif (
            scheduler_type is PackingScheduler.DEFAULT_HYBRID_CP
            or scheduler_type is PackingScheduler.NAIVE_SEQUENCE_PACKING
        ):

            subsample_seqlens = []
            for sample in batch:
                subsample_seqlens.extend([sample["tokens"].numel()])
            subsample_seqlens = torch.tensor(subsample_seqlens, dtype=torch.int32).cuda()
            subsample_seqlens = subsample_seqlens[subsample_seqlens != 0]

            seqlens_gathered, offsets = _get_global_seqlens(subsample_seqlens, dp_group)

            global_id_seqlens, global_ids_this_rank = _get_global_id_seqlens(
                subsample_seqlens.shape[0], offsets, seqlens_gathered, dp_group
            )

            sample_id_groups = scheduler.get_groups_and_subsamples(global_id_seqlens)

            set_gbs = set()
            for group in sample_id_groups:
                for sub in group:
                    set_gbs.update(sub)
            assert len(set_gbs) == len(
                global_id_seqlens
            ), f"set_gbs length: {len(set_gbs)} \
            != global_ids_this_rank length: {len(global_id_seqlens)}"

            batch = _unpack_batch(batch)
            samples_this_rank_with_id = _reroute_samples_to_hdp_ranks(
                batch,
                global_ids_this_rank,
                global_id_seqlens,
                sample_id_groups,
                offsets,
                dp_group,
                tp_group,
                dp_cp_group,
                total_hdp_gpus,
            )
            batch, sample_id_groups = samples_this_rank_with_id, sample_id_groups

            hdp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
            num_micro_batches = len(sample_id_groups)

            # local_cp_sizes_gpu is computed outside and passed in for DEFAULT_HYBRID_CP.
            if scheduler_type is PackingScheduler.DEFAULT_HYBRID_CP:
                # One H2D total
                local_cp_sizes_cpu: List[int] = []
                for i in range(num_micro_batches):
                    sample_ids_this_group = sample_id_groups[i][hdp_rank]
                    local_cp_sizes_cpu.append(
                        len(
                            [
                                1
                                for sample_ids in sample_id_groups[i]
                                if sample_ids_this_group[0] in sample_ids
                            ]
                        )
                    )
                local_cp_sizes_gpu = torch.tensor(local_cp_sizes_cpu, dtype=torch.int32, device=dev)
            else:
                local_cp_sizes_gpu = None

            grouped_samples = [
                [batch[sub_sample_id] for sub_sample_id in sample_id_groups[i][hdp_rank]]
                for i in range(num_micro_batches)
            ]

            new_samples = _build_packed_microbatches(
                grouped_samples=grouped_samples,
                scheduler_type=scheduler_type,
                local_cp_sizes_gpu=local_cp_sizes_gpu,
            )

            # calculate this two values for tflops calculation
            num_total_tokens_this_global_batch = float(sum(seqlens_gathered))
            sequence_square_sum_this_global_batch = float(
                sum(seqlen**2 for seqlen in seqlens_gathered)
            )

    # broadcast num_micro_batches, num_total_tokens_this_global_batch,
    # sequence_square_sum_this_global_batch, and packed_seq_params to PP group
    if pp_group.size() > 2 and tp_group.rank() == 0:
        if pp_group.rank() == 0:
            tensor_list = [
                torch.tensor(
                    [
                        num_micro_batches,
                        num_total_tokens_this_global_batch,
                        sequence_square_sum_this_global_batch,
                    ],
                    dtype=torch.float32,
                ).cuda()
            ]
            for sample in new_samples:
                tensor_list.append(sample["max_seqlen"].unsqueeze(0))
            for sample in new_samples:
                tensor_list.append(
                    sample["local_cp_size"].unsqueeze(0)
                    if scheduler_type is PackingScheduler.DEFAULT_HYBRID_CP
                    else torch.tensor([-1], dtype=torch.float32).cuda()
                )
            for sample in new_samples:
                tensor_list.append(sample["cu_seqlens"])
                tensor_list.append(sample["cu_seqlens_padded"])
            info_to_broadcast_this_pp_group = torch.cat(tensor_list, dim=0).to(
                device=dev, dtype=torch.float32
            )
            info_length_tensor = torch.tensor(
                info_to_broadcast_this_pp_group.shape[0], dtype=torch.int32
            ).cuda()
            _broadcast_to_pp_group(info_length_tensor)
            _broadcast_to_pp_group(info_to_broadcast_this_pp_group)
        else:
            info_length_tensor = torch.tensor(0, dtype=torch.int32).cuda()
            _broadcast_to_pp_group(info_length_tensor)
            info_to_broadcast_this_pp_group = torch.empty(
                info_length_tensor.item(), dtype=torch.float32
            ).cuda()
            _broadcast_to_pp_group(info_to_broadcast_this_pp_group)
            if pp_group.rank() != pp_group.size() - 1:
                info_numpy = info_to_broadcast_this_pp_group.cpu().numpy()
                num_micro_batches = int(info_numpy[0])
                num_total_tokens_this_global_batch = info_numpy[1]
                sequence_square_sum_this_global_batch = info_numpy[2]
                max_seqlens = info_to_broadcast_this_pp_group[3 : 3 + num_micro_batches]
                is_hybrid_cp = int(info_numpy[3 + num_micro_batches]) != -1
                local_cp_sizes = info_to_broadcast_this_pp_group[
                    3 + num_micro_batches : 3 + 2 * num_micro_batches
                ]
                cu_seqlens_list = []
                cu_seqlens_padded_list = []
                indices = np.where(info_numpy == 0)[0]
                for i in range(num_micro_batches):
                    cu_seqlens_list.append(
                        info_to_broadcast_this_pp_group[indices[i * 2] : indices[i * 2 + 1]]
                    )
                    if i == num_micro_batches - 1:
                        cu_seqlens_padded_list.append(
                            info_to_broadcast_this_pp_group[indices[i * 2 + 1] :]
                        )
                    else:
                        cu_seqlens_padded_list.append(
                            info_to_broadcast_this_pp_group[indices[i * 2 + 1] : indices[i * 2 + 2]]
                        )

                new_samples = []
                for i in range(num_micro_batches):
                    new_sample = {}
                    new_sample["max_seqlen"] = max_seqlens[i].to(torch.int32)
                    if is_hybrid_cp:
                        new_sample["local_cp_size"] = local_cp_sizes[i].to(torch.int32)
                    new_sample["cu_seqlens"] = cu_seqlens_list[i].to(torch.int32)
                    new_sample["cu_seqlens_padded"] = cu_seqlens_padded_list[i].to(torch.int32)
                    new_samples.append(new_sample)

    if tp_group.size() > 1:
        # TODO(tailaim): This triggers H2D/D2H transfers.
        # Ideally, we should perform the communication over a CPU process
        if tp_group.rank() == 0:
            info_to_broadcast_this_tpgroup = torch.tensor(
                [
                    num_micro_batches,
                    num_total_tokens_this_global_batch,
                    sequence_square_sum_this_global_batch,
                ],
                dtype=torch.float32,
                device=dev,
            )
            _broadcast_to_tp_group(info_to_broadcast_this_tpgroup)
        else:
            info_to_broadcast_this_tpgroup = torch.tensor(
                [0, 0, 0], dtype=torch.float32, device=dev
            )
            _broadcast_to_tp_group(info_to_broadcast_this_tpgroup)
            info_numpy = info_to_broadcast_this_tpgroup.cpu().numpy()
            num_micro_batches = int(info_numpy[0])
            num_total_tokens_this_global_batch = info_numpy[1]
            sequence_square_sum_this_global_batch = info_numpy[2]

    if (
        config.virtual_pipeline_model_parallel_size is not None
        and config.virtual_pipeline_model_parallel_size > 1
    ):
        vpp_size = config.virtual_pipeline_model_parallel_size
        if tp_group.rank() == 0:
            if pp_group.rank() == 0 or pp_group.rank() == pp_group.size() - 1:
                new_samples_for_other_ppstage = []
                for sample in new_samples:
                    new_sample_for_other_ppstage = {}
                    new_sample_for_other_ppstage["max_seqlen"] = sample["max_seqlen"]
                    new_sample_for_other_ppstage["cu_seqlens"] = sample["cu_seqlens"]
                    new_sample_for_other_ppstage["cu_seqlens_padded"] = sample["cu_seqlens_padded"]
                    if scheduler_type is PackingScheduler.DEFAULT_HYBRID_CP:
                        new_sample_for_other_ppstage["local_cp_size"] = sample["local_cp_size"]
                    new_samples_for_other_ppstage.append(new_sample_for_other_ppstage)
                if pp_group.rank() == 0:
                    new_data_iterator = [RerunDataIterator(iter(new_samples))] + [
                        RerunDataIterator(iter(new_samples_for_other_ppstage))
                        for _ in range(vpp_size - 1)
                    ]
                else:
                    new_data_iterator = [
                        RerunDataIterator(iter(new_samples_for_other_ppstage))
                        for _ in range(vpp_size - 1)
                    ] + [RerunDataIterator(iter(new_samples))]
            else:
                new_data_iterator = [RerunDataIterator(iter(new_samples)) for _ in range(vpp_size)]
        else:
            new_data_iterator = [None for _ in range(vpp_size)]
    else:
        new_data_iterator = RerunDataIterator(iter(new_samples)) if tp_group.rank() == 0 else None

    return (
        new_data_iterator,
        num_micro_batches,
        num_total_tokens_this_global_batch,
        sequence_square_sum_this_global_batch,
    )


class BaseScheduler:
    """
    Base class for sequence packing schedulers.
    """

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank: Optional[int],
        cp_size: int,
        dp_size: int,
        microbatch_group_size_per_vp_stage: Optional[int],
        hybrid_context_parallel: bool = False,
    ):
        self.max_seqlen_per_dp_cp_rank = max_seqlen_per_dp_cp_rank
        self.cp_size = cp_size
        self.dp_size = dp_size
        self.microbatch_group_size_per_vp_stage = microbatch_group_size_per_vp_stage
        self.hybrid_context_parallel = hybrid_context_parallel

    def check_require_sample_keys(self, batch: List[Dict]):
        """
        Required per-(sub)sample fields expected by the scheduler.
        """
        raise NotImplementedError

    def get_groups_and_subsamples(self, sample_id_seqlens):
        """
        This scheduler simply packs sequences in their original order
        until reaching the max sequence length.
        It does not reorder sequences nor perform any load balancing.
        """
        raise NotImplementedError


class EmptyPackingScheduler(BaseScheduler):
    """
    This scheduler only packs sequences in their original order
    and does not perform any load balancing.
    """

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank,
        cp_size,
        dp_size,
        microbatch_group_size_per_vp_stage,
        hybrid_context_parallel: bool = False,
    ):
        super().__init__(
            max_seqlen_per_dp_cp_rank,
            cp_size,
            dp_size,
            microbatch_group_size_per_vp_stage,
            hybrid_context_parallel=hybrid_context_parallel,
        )

    def check_require_sample_keys(self, batch: List[Dict]):
        """
        Required per-(sub)sample fields expected by the default hybrid CP
        - tokens[torch.Tensor]:1D tensor of input token ids for the (sub)sequence
        (typically shape [padded_seq_len], dtype int64).
        - labels[torch.Tensor]: 1D tensor of target token ids aligned with `tokens` for next-token
        prediction (typically shape [padded_seq_len], dtype int64).
        - loss_mask[torch.Tensor]: 1D tensor mask indicating which positions contribute to the
        loss (typically shape [padded_seq_len], dtype float); must already reflect
        any padding/EOD masking policy.
        - position_ids[torch.Tensor]: 1D tensor of positional indices used by the model
        (typically shape [padded_seq_len], dtype int64).
        - original_seq_len[torch.Tensor]: Scalar int32 tensor length of the unpadded (effective)
        sequence, used to build `cu_seqlens` for variable-length attention.
        - padded_seq_len[torch.Tensor]: Scalar int32 tensor length after padding/truncation,
        used to build `cu_seqlens_padded` and for max_seqlen computation.
        - local_cp_size[torch.Tensor]: Scalar int32 tensor of the partner CP size, used to build
        `local_cp_size` for Hybrid-CP.
        - num_micro_batches_left[int]: number of microbatches left to be fetched.
        """
        required_keys = [
            "tokens",
            "labels",
            "loss_mask",
            "position_ids",
            "original_seq_len",
            "padded_seq_len",
            "local_cp_size",
            "num_micro_batches_left",
        ]

        # - Each `next(data_iterator)` returns a microbatch worth of samples.
        # - The returned `batch` is a Python `list` of per-sample dicts
        # (i.e., `List[Dict[str, Tensor]]`).
        # - `batch[0]["num_micro_batches_left"]` indicates how many additional microbatches
        # should be fetched afterwards for the current step (so the caller will fetch
        # `num_micro_batches_left` more times, in addition to the first fetch).

        for key in required_keys:
            if key not in batch[0]:
                return False
        if "local_cp_size" in batch[0]:
            assert (
                self.hybrid_context_parallel
            ), "local_cp_size is only supported when using hybrid context parallel"
        return True

    def get_groups_and_subsamples(self, sample_id_seqlens):
        """
        This scheduler only packs sequences in their original order
        and does not perform any load balancing.
        """
        pass


class EmptyNoPackingScheduler(BaseScheduler):
    """
    It does not pack sequences, it only returns the original batch.
    """

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank,
        cp_size,
        dp_size,
        microbatch_group_size_per_vp_stage,
        hybrid_context_parallel: bool = False,
    ):
        super().__init__(
            max_seqlen_per_dp_cp_rank,
            cp_size,
            dp_size,
            microbatch_group_size_per_vp_stage,
            hybrid_context_parallel=hybrid_context_parallel,
        )

    def check_require_sample_keys(self, batch: List[Dict]):
        """
        Required per-(sub)sample fields expected by the default hybrid CP
        - tokens[torch.Tensor]:1D tensor of input token ids for the (sub)sequence
        (typically shape [padded_seq_len], dtype int64).
        - labels[torch.Tensor]: 1D tensor of target token ids aligned with `tokens` for next-token
        prediction (typically shape [padded_seq_len], dtype int64).
        - loss_mask[torch.Tensor]: 1D tensor mask indicating which positions contribute to the
        loss (typically shape [padded_seq_len], dtype float); must already reflect any
        padding/EOD masking policy.
        - position_ids[torch.Tensor]: 1D tensor of positional indices used by the model
        (typically shape [padded_seq_len], dtype int64).
        - cu_seqlens[torch.Tensor]: 1D int32 tensor of cumulative (prefix-sum)*original sequence
        lengths, shape [num_seqs + 1], with cu_seqlens[0] = 0. Used for variable-length
        attention over unpadded token streams.
        - cu_seqlens_padded[torch.Tensor]: 1D int32 tensor of cumulative (prefix-sum) padded
        sequence lengths, shape [num_seqs + 1], with cu_seqlens_padded[0] = 0.
        Used for packed layouts where each sequence occupies `padded_seq_len` tokens.
        - max_seqlen[torch.Tensor]: Scalar int32 tensor of the maximum sequence length in
        the microbatch (typically the max of padded lengths).
        - local_cp_size[torch.Tensor]: Scalar int32 tensor of the local_cp_size CP size, used to
        build `local_cp_size`.
        - num_micro_batches_left[int]: number of microbatches left to be fetched.
        - num_total_tokens_this_global_batch[float]: total number of tokens in the global batch.
        Used for tflops calculation.
        - sequence_square_sum_this_global_batch[float]: sum of the squares of the sequence lengths
        in the global batch. Used for tflops calculation.
        """
        required_keys = [
            "tokens",
            "labels",
            "loss_mask",
            "position_ids",
            "cu_seqlens",
            "cu_seqlens_padded",
            "max_seqlen",
            "local_cp_size",
            "num_micro_batches_left",
            "num_total_tokens_this_global_batch",
            "sequence_square_sum_this_global_batch",
        ]

        # Each call returns the packed sequences for one microbatch; the data_iterator will fetch
        # num_micro_batches_left more times.

        for key in required_keys:
            if key not in batch[0]:
                return False

        if "local_cp_size" in batch[0]:
            assert (
                self.hybrid_context_parallel
            ), "local_cp_size is only supported when using hybrid context parallel"

        return True

    def get_groups_and_subsamples(self, sample_id_seqlens):
        """
        This scheduler only packs sequences in their original order
        and does not perform any load balancing.
        """
        pass


class NaiveSequencePackingScheduler(BaseScheduler):
    """
    This scheduler simply packs sequences in their original order
    until reaching the max sequence length.
    It does not reorder sequences nor perform any load balancing.
    """

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank,
        cp_size,
        dp_size,
        microbatch_group_size_per_vp_stage,
        hybrid_context_parallel: bool = False,
    ):
        super().__init__(
            max_seqlen_per_dp_cp_rank,
            cp_size,
            dp_size,
            microbatch_group_size_per_vp_stage,
            hybrid_context_parallel=hybrid_context_parallel,
        )
        self.max_seq_len_all_ranks = self.max_seqlen_per_dp_cp_rank * self.cp_size

    def check_require_sample_keys(self, batch: List[Dict]):
        """
        Required per-(sub)sample fields expected by the default hybrid CP
        - tokens[torch.Tensor]:1D tensor of input token ids for the (sub)sequence
        (typically shape [padded_seq_len], dtype int64).
        - labels[torch.Tensor]: 1D tensor of target token ids aligned with `tokens` for next-token
        prediction (typically shape [padded_seq_len], dtype int64).
        - loss_mask[torch.Tensor]: 1D tensor mask indicating which positions contribute to the
        loss (typically shape [padded_seq_len], dtype float); must already reflect any
        padding/EOD masking policy.
        - position_ids[torch.Tensor]: 1D tensor of positional indices used by the model
        (typically shape [padded_seq_len], dtype int64).
        - original_seq_len[torch.Tensor]: Scalar int32 tensor length of the unpadded (effective)
        sequence, used to build `cu_seqlens` for variable-length attention.
        - padded_seq_len[torch.Tensor]: Scalar int32 tensor length after padding/truncation,
        used to build `cu_seqlens_padded` and for max_seqlen computation.
        """
        required_keys = [
            "tokens",
            "labels",
            "loss_mask",
            "position_ids",
            "original_seq_len",
            "padded_seq_len",
        ]
        # data_iterator returns all samples at once;
        # we only fetch it once, rather than iterating num_micro_batches times.
        for key in required_keys:
            if key not in batch[0]:
                return False
        return True

    def get_groups_and_subsamples(self, sample_id_seqlens):
        """
        This scheduler simply packs sequences in their original order
        until reaching the max sequence length.
        It does not reorder sequences nor perform any load balancing.
        """
        groups = []
        sample_id_groups = []
        packed_id_groups = []
        sum_seqlen = 0
        single_microbatch = []

        for i in range(len(sample_id_seqlens)):
            single_microbatch = [i]
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


class DefaultHybridCPscheduler(BaseScheduler):
    """
    This class provides the functionality to form groups of sub-samples
    such that all DPxCP ranks have a roughly balanced workload in the group.
    """

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank,
        cp_size,
        dp_size,
        microbatch_group_size_per_vp_stage,
        hybrid_context_parallel: bool = False,
    ):
        super().__init__(
            max_seqlen_per_dp_cp_rank,
            cp_size,
            dp_size,
            microbatch_group_size_per_vp_stage,
            hybrid_context_parallel=hybrid_context_parallel,
        )
        self.max_seq_len_per_rank = self.max_seqlen_per_dp_cp_rank
        self.total_hdp_gpus = self.dp_size * self.cp_size

    def check_require_sample_keys(self, batch: List[Dict]):
        """
        Required per-(sub)sample fields expected by the default hybrid CP
        - tokens[torch.Tensor]:1D tensor of input token ids for the (sub)sequence
        (typically shape [padded_seq_len], dtype int64).
        - labels[torch.Tensor]: 1D tensor of target token ids aligned with `tokens` for next-token
        prediction (typically shape [padded_seq_len], dtype int64).
        - loss_mask[torch.Tensor]: 1D tensor mask indicating which positions contribute to the
        loss (typically shape [padded_seq_len], dtype float); must already reflect any
        padding/EOD masking policy.
        - position_ids[torch.Tensor]: 1D tensor of positional indices used by the model
        (typically shape [padded_seq_len], dtype int64).
        - original_seq_len[torch.Tensor]: Scalar int32 tensor length of the unpadded (effective)
        sequence, used to build `cu_seqlens` for variable-length attention.
        - padded_seq_len[torch.Tensor]: Scalar int32 tensor length after padding/truncation,
        used to build `cu_seqlens_padded` and for max_seqlen computation.
        """
        required_keys = [
            "tokens",
            "labels",
            "loss_mask",
            "position_ids",
            "original_seq_len",
            "padded_seq_len",
        ]

        # data_iterator returns all samples at once;
        # we only fetch it once, rather than iterating num_micro_batches times.
        for key in required_keys:
            if key not in batch[0]:
                return False
        return True

    @lru_cache(maxsize=128)
    def get_total_workload(self, seq_length: int, cp_size: Optional[int] = None):
        """
        seq_length: sequence length of a sub-sample
        cp_size: total number of CP ranks working on this sub-sample

        Note:
        This function is used to estimate the relative workload intensity
        of a sub-sample. This is not meant to be an accurate flops calculator.

        Returns: workload of a sub-sample
        """
        if cp_size is None:
            cp_size = self.gpus_needed(seq_length)
        return (seq_length * seq_length) / cp_size

    @lru_cache(maxsize=128)
    def gpus_needed(self, seq_len: int) -> int:
        """
        Calculates the number of GPUs needed for a given sequence length
        and max sequence length per CP rank.
        This is used to determine the CP size of a sub-sample.

        The number is rounded up to the next power of 2 to match the available
        hybrid context parallel process group sizes.
        """
        return max(1, 2 ** ceil(log2((seq_len / self.max_seq_len_per_rank))))

    def make_buckets_equal(
        self,
        sample_seqlens: List[Tuple[int, int]],  # List of (sample_id, sequence_length) tuples
        compute_estimator: Callable[[int], float],
    ) -> List[Deque]:
        """
        Makes as many buckets as unique CP sizes needed.
        This keeps sample IDs tethered to their sequence lengths throughout the bucketing process.
        """
        # Extract just the sequence lengths for determining k
        seqlens = [seq_len for _, seq_len in sample_seqlens]

        # Determine k based on unique GPU categories needed
        k = len({self.gpus_needed(L) for L in seqlens})

        # Create a work target for each bucket
        # This is the total work divided by the number of buckets
        work = []
        for _, s in sample_seqlens:
            cp_size = self.gpus_needed(s)
            work.append(compute_estimator(s, cp_size))
        total_work = sum(work)
        target = total_work / k
        buckets, cur, cur_work = [], [], 0.0
        remaining_work = total_work
        remaining_k = k

        for i, (sample_id, seq_len) in enumerate(sample_seqlens):
            work = compute_estimator(seq_len)
            projected = cur_work + work

            # Check if we should close this bucket
            if cur and (
                projected > target * 1.1  # Too much work
                or len(sample_seqlens) - i <= remaining_k - len(buckets)
            ):  # Need to save sequences for remaining buckets
                buckets.append(deque(cur))
                cur, cur_work = [], 0.0
                remaining_work -= sum(compute_estimator(seq_len) for _, seq_len in cur)
                remaining_k -= 1

            cur.append((sample_id, seq_len))
            cur_work += work

        if cur:
            buckets.append(deque(cur))

        return buckets

    def next_hdp_group(
        self,
        sample_seqlens: List[Tuple[int, int]],  # List of (sample_id, sequence_length) tuples
        compute_estimator: Callable[[int], float],
        total_gpus: int,
        delta: float = 0.05,  # balance slack (e.g. 5 %)
        strategy: str = "dp",  # "dp" or "pp"
        eps_bucket: float = 0.10,  # ε target for bucket balance
    ) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:
        """
        Given a list of (sample_id, sequence_length) tuples, this function aims to assign
        sequences in a group such that all GPUs in the DPxCP group have a roughly balanced
        workload. Once each group is roughly balanced, we exit and return the
        group and the leftover sequences.

        The function performs the following passes in order to form a balanced microbatch:
        1. We create buckets of sequences that are roughly balanced.
        We try to create as many buckets as possible CP sizes.
        2. Given a bucket has sequences available, we assign the sample
            a. To a new set of GPUs if there are enough free GPUs.
            b. To an existing set of GPUs with the lowest load.
        3. We check if the group is balanced whenever we need to move onto a new CP size
        in the same set of GPUs.
        4. We trim the group if removing the last added sequence helps improve balance.
        5. If we run out of sequences to assign and there are empty GPUs,
        we redistribute work to empty GPUs by recursively increasing the CP size of a
        sample until no empty GPUs are left.

        #TODO: Add clarification on when we check for balance. What does prev_needed do?

        Returns (micro_batches, leftover_sample_seqlens, exec_times, sample_ids_per_gpu).
        """
        if not sample_seqlens:
            return (
                [[] for _ in range(total_gpus)],
                [],
                [0.0 for _ in range(total_gpus)],
                [[] for _ in range(total_gpus)],
            )

        # Get buckets of sequences with balanced work
        buckets = self.make_buckets_equal(sample_seqlens, compute_estimator)

        # Initialize tracking structures
        micro_batches = [[] for _ in range(total_gpus)]
        exec_times = [0.0 for _ in range(total_gpus)]
        sample_ids_per_gpu = [[] for _ in range(total_gpus)]
        # gid : seq_len
        packing_sequence_len = {}

        gpu_group_id = [None] * total_gpus
        group_members = {}
        group_size = {}
        next_gid = 0

        pp_cursor = 0
        prev_needed = None
        check_balance = False

        while buckets:
            # ---- Step 1 – pick the next sequence we COULD place ------------------
            sample_seq_tuple = bucket_idx = None
            needed = None

            scan_order = (
                range(len(buckets))
                if strategy == "dp"
                else [(pp_cursor + i) % len(buckets) for i in range(len(buckets))]
            )

            for idx in scan_order:
                if not buckets[idx]:
                    continue
                cand_tuple = buckets[idx][0]  # This is now (sample_id, seq_len)
                cand_seq_len = cand_tuple[1]
                needed = self.gpus_needed(cand_seq_len)

                # (a) Do we have an *existing* group of size `needed`?
                candidate_gids = [gid for gid, sz in group_size.items() if sz == needed]

                # (b) Or enough completely free GPUs to start a new group?
                free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
                if candidate_gids or len(free_ranks) >= needed:
                    sample_seq_tuple, bucket_idx = cand_tuple, idx
                    break

            # No place to put any remaining sequence – finish this micro‑batch
            if sample_seq_tuple is None:
                break

            # TODO[pmannan]: PP not yet supported. Add PP scheduling.
            if strategy == "pp":
                pp_cursor = (bucket_idx + 1) % len(buckets)

            sample_id, seq_len = sample_seq_tuple
            needed = self.gpus_needed(seq_len)
            if prev_needed is None:
                prev_needed = needed

            # (a)  Existing groups of exactly this size
            candidate_gids = [
                gid
                for gid, sz in group_size.items()
                if sz == needed
                and packing_sequence_len[gid] + seq_len / needed <= self.max_seq_len_per_rank
            ]
            if candidate_gids:
                best_gid, best_load = min(
                    (
                        (gid, max(exec_times[r] for r in group_members[gid]))
                        for gid in candidate_gids
                    ),
                    key=lambda t: t[1],
                )
            else:
                best_gid, best_load = None, float("inf")

            # (b)  Hypothetical **new** group from completely free GPUs
            free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
            if len(free_ranks) >= needed:
                free_sorted = sorted(free_ranks, key=lambda r: exec_times[r])
                new_members = free_sorted[:needed]
                new_load = exec_times[new_members[-1]]

                if new_load < best_load:
                    best_gid = None
                    chosen_members = new_members
                else:
                    chosen_members = group_members[best_gid]
            else:
                if best_gid is None:
                    break
                chosen_members = group_members[best_gid]

            # ---- Step 2 – if we decided to create a fresh group ----------------
            if best_gid is None:
                best_gid = next_gid
                next_gid += 1
                group_members[best_gid] = chosen_members
                group_size[best_gid] = needed
                for r in chosen_members:
                    gpu_group_id[r] = best_gid

            # ---- Step 3 – assign the sequence to every member of that group ------
            per_gpu_cost = compute_estimator(seq_len)

            packing_sequence_len[best_gid] = (
                packing_sequence_len.get(best_gid, 0) + seq_len / needed
            )
            for r in chosen_members:
                micro_batches[r].append(seq_len)
                exec_times[r] += per_gpu_cost
                sample_ids_per_gpu[r].append(sample_id)

            # Remove the sequence definitively from its bucket
            buckets[bucket_idx].popleft()

            # ---- Step 4 – tidy, balance‑check, maybe early‑exit ------------------
            while buckets and not buckets[0]:
                buckets.pop(0)
                pp_cursor %= max(1, len(buckets))

            # TODO: Removing this helps reduce the number of groups when we have
            # lots of samples with same CP size.
            # But because we don't exit as soon as we get balanced,
            # even if there is one group available that can take the next sample,
            # we will keep adding samples to the same group.
            # trim_overload() does not help because it only checks if removing the
            # last added sample helps.
            # We cannot check after adding every sample because there will always be imbalance
            # if we don't wait for future scheduling.

            # IMPORTANT: So we need a solution here
            if needed < prev_needed:
                # When we get into a lower CP size in the same group,
                # we can start checking for balance. There is still a gotcha here.
                # Let's say we have a group of 3 GPU 0-2, then we move onto group of 2.
                # We keep assigning group of 2 as we do in descending order but GPU 7/15
                # never sees a microbatch assigned to it
                # until we run out of samples with CP2.
                # This means we are never balanced as min(exec_times) will always be 0.
                # We need a smart way of identifying that we have run out of big samples
                # and if we are having to assign work to a GPU already working,
                # is it because there are empty GPUs?
                # Would assigning work to empty GPUs first by moving onto next CP bucket help?
                # But we need to remember to come back to this CP size bucket and then
                # check for balance. Maybe the scheduling algorithm should look at empty
                # GPUs and find work rather than going sequence by sequence.
                check_balance = True

            if (
                check_balance
                and buckets
                and max(exec_times) - min(exec_times) <= delta * max(exec_times)
            ):
                break

        # Gather leftovers (flatten remaining buckets, preserve order)
        leftovers = []
        for b in buckets:
            for sample_seq_tuple in b:
                leftovers.append(sample_seq_tuple)

        # ---------------------------------------------------------------------------
        def trim_overload():
            """
            Iteratively pop the most-recent sequence from the *most-loaded group*
            whenever doing so reduces the global slack.
            """
            while True:
                cur_max = max(exec_times)
                cur_min = min(exec_times)
                cur_slack = cur_max - cur_min
                if cur_slack <= delta * cur_max:
                    # Slack is already within limit.
                    break
                if cur_min == 0:
                    # There are empty GPUs that will be
                    # handled in the next step.
                    break

                max_r = exec_times.index(cur_max)
                gid = gpu_group_id[max_r]
                members = group_members[gid]

                if not micro_batches[max_r] or len(micro_batches[max_r]) <= 1:
                    break

                seq = micro_batches[max_r][-1]
                need = group_size[gid]
                per_gpu_cost = compute_estimator(seq)

                proj_times = exec_times[:]
                for r in members:
                    proj_times[r] -= per_gpu_cost

                proj_slack = max(proj_times) - min(proj_times)

                # Check if trimming the workload helps imbalance
                if proj_slack < cur_slack:
                    sample_id_to_remove = sample_ids_per_gpu[max_r][-1]
                    for r in members:
                        micro_batches[r].pop()
                        exec_times[r] -= per_gpu_cost
                        sample_ids_per_gpu[r].pop()
                    leftovers.append((sample_id_to_remove, seq))
                else:
                    break

        # TODO(tailaim): uncomment this to support different ranks have different num_microbatches
        # trim_overload()

        # Track samples in this group before redistribution to empty GPUs
        total_work_before = sum(len(mb) for mb in micro_batches)

        # Check for empty GPUs and redistribute work
        def fill_empty_gpus(
            micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size
        ):
            """
            Recursively check for empty GPUs and redistribute work by increasing
            the number of GPUs sharing samples. This ensures all GPUs have work.
            GPUs must be allocated consecutively so we may need to push existing
            work to other ranks in order to expand samples.
            """
            # Find empty GPUs
            empty_gpus = [i for i in range(total_gpus) if not micro_batches[i]]
            if not empty_gpus:
                return (
                    micro_batches,
                    exec_times,
                    sample_ids_per_gpu,
                    group_members,
                    group_size,
                )  # No empty GPUs, we're done

            # Find the smallest group size that exists
            existing_group_sizes = set(group_size.values())
            assert (
                existing_group_sizes
            ), "There should be at least one group existing, cannot reditribute, "
            "try to increase 'max-seqlen-per-dp-cp-rank'."

            min_group_size = min(existing_group_sizes)
            # We have Hybrid DPxCP groups for every power of 2 of GPUs or the entire DPxCP group.
            next_power = min(min_group_size * 2, total_gpus)

            # Find the first group of min_group_size that can be expanded
            expandable_gid = None
            expandable_members = None
            expandable_new_gpus = None

            for gid, size in group_size.items():
                if size == min_group_size:
                    members = group_members[gid]
                    needed_count = next_power - min_group_size
                    group_start_gpu = members[0]
                    group_end_gpu = members[-1]
                    empty_gpu = [idx for idx, work in enumerate(micro_batches) if not work][0]
                    assert not all(
                        work for work in micro_batches[empty_gpu : empty_gpu + needed_count]
                    ), f"Empty GPUs were detected but not enough to expand."
                    work_to_push = micro_batches[
                        group_end_gpu + 1 : empty_gpu
                    ]  # This is work of all other subsequent sub-samples
                    exec_times_to_push = exec_times[group_end_gpu + 1 : empty_gpu]
                    sample_ids_to_push = sample_ids_per_gpu[group_end_gpu + 1 : empty_gpu]

                    new_micro_batches = [[]] * len(micro_batches)
                    new_exec_times = [0.0] * len(exec_times)
                    new_sample_ids_per_gpu = [[]] * len(sample_ids_per_gpu)

                    # No change in work until the group selected for expansion
                    for i in range(group_start_gpu):
                        new_micro_batches[i] = micro_batches[i]
                        new_exec_times[i] = exec_times[i]
                        new_sample_ids_per_gpu[i] = sample_ids_per_gpu[i]

                    # The work is distributed across the expanded group
                    for i in range(group_start_gpu, group_end_gpu + needed_count + 1):
                        new_micro_batches[i] = micro_batches[group_end_gpu]
                        new_exec_times[i] = self.get_total_workload(
                            micro_batches[group_end_gpu][0], next_power
                        )
                        new_sample_ids_per_gpu[i] = sample_ids_per_gpu[group_end_gpu]

                    # Any assigned work on expanded GPUs is pushed
                    for i, work in enumerate(work_to_push):
                        new_micro_batches[group_end_gpu + needed_count + 1 + i] = work
                        new_exec_times[group_end_gpu + needed_count + 1 + i] = exec_times_to_push[i]
                        new_sample_ids_per_gpu[group_end_gpu + needed_count + 1 + i] = (
                            sample_ids_to_push[i]
                        )

                    group_size[gid] = next_power
                    group_members[gid] = list(range(members[0], members[-1] + needed_count + 1))
                    for pushed_gid in group_size.keys():
                        if pushed_gid > gid:
                            group_members[pushed_gid] = [
                                x + needed_count for x in group_members[pushed_gid]
                            ]

                    return (
                        new_micro_batches,
                        new_exec_times,
                        new_sample_ids_per_gpu,
                        group_members,
                        group_size,
                    )

        empty_gpus = any([not micro_batches[i] for i in range(total_gpus)])
        while empty_gpus:
            micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size = (
                fill_empty_gpus(
                    micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size
                )
            )
            empty_gpus = any([not micro_batches[i] for i in range(total_gpus)])

        # Assert that no sample has been completely removed
        total_work_after = sum(len(mb) for mb in micro_batches)
        assert (
            total_work_after >= total_work_before
        ), f"Samples were removed: {total_work_before} -> {total_work_after}"

        return micro_batches, leftovers, exec_times, sample_ids_per_gpu

    def align_sample_id_groups(self, sample_id_groups):
        """
        Align len(sample_id_groups) to microbatch_group_size_per_vp_stage (K) when VPP is enabled.
        i.e. if len(sample_id_groups) % K != 0, we need to add extra microbatches.
        """
        multiple = int(self.microbatch_group_size_per_vp_stage)
        remainder = (-len(sample_id_groups)) % multiple
        i = len(sample_id_groups) - 1

        def split_group(sample_id_group):
            total_hdp_ranks = len(sample_id_group)
            cu_ranks = [0]
            prev_cp_size = 0

            while cu_ranks[-1] != total_hdp_ranks:
                start_rank = cu_ranks[-1]
                sid0 = sample_id_group[start_rank][0]
                # Count contiguous ranks that share sid0 (CP group assumed contiguous).
                cp_size = 0
                for r in range(start_rank, total_hdp_ranks):
                    if sid0 in sample_id_group[r]:
                        cp_size += 1
                    else:
                        break
                assert (
                    prev_cp_size == 0 or cp_size <= prev_cp_size
                ), f"split_group: CP size is not decreasing: prev={prev_cp_size}, cur={cp_size}"
                cu_ranks.append(start_rank + cp_size)
                prev_cp_size = cp_size
            if len(cu_ranks) == 2:
                # can't split anymore
                return None, None

            k = 0
            while cu_ranks[k] < total_hdp_ranks // 2:
                k += 1

            # Keep original rank positions; zero out the other half, then expand CP to fill empties.
            old_mb = sample_id_group[: cu_ranks[k]] + [
                [] for _ in range(total_hdp_ranks - cu_ranks[k])
            ]
            new_mb = sample_id_group[cu_ranks[k] :] + [[] for _ in range(cu_ranks[k])]
            old_mb = fill_empty_by_expanding_cp(old_mb)
            new_mb = fill_empty_by_expanding_cp(new_mb)
            return new_mb, old_mb

        def fill_empty_by_expanding_cp(sample_id_group):
            def fill_empty(sample_id_group):
                empty_size = sum(1 for x in sample_id_group if len(x) == 0)
                i = len(sample_id_group) - 1 - empty_size
                prev_cp_size = 0
                while i >= 0:
                    sid0 = sample_id_group[i][0]
                    cp_size = 0
                    while sid0 in sample_id_group[i] and i >= 0:
                        cp_size += 1
                        i -= 1
                    if cp_size > prev_cp_size and prev_cp_size != 0:
                        # double cp_size of this group
                        start_idx = i + 1 + cp_size
                        end_idx = (
                            -empty_size + prev_cp_size if -empty_size + prev_cp_size < 0 else None
                        )
                        sample_id_group[start_idx + 2 * prev_cp_size : end_idx] = sample_id_group[
                            start_idx + prev_cp_size : -empty_size
                        ]
                        sample_id_group[start_idx + prev_cp_size : start_idx + 2 * prev_cp_size] = (
                            sample_id_group[start_idx : start_idx + prev_cp_size]
                        )
                        break
                    elif cp_size <= empty_size and i == -1:
                        end_idx = -empty_size + cp_size if -empty_size + cp_size < 0 else None
                        sample_id_group[2 * cp_size : end_idx] = sample_id_group[
                            cp_size:-empty_size
                        ]
                        sample_id_group[cp_size : 2 * cp_size] = sample_id_group[0:cp_size]
                        break
                    prev_cp_size = cp_size
                return sample_id_group

            while len(sample_id_group[-1]) == 0:
                sample_id_group = fill_empty(sample_id_group)
            return sample_id_group

        while remainder > 0:
            assert i >= 0, f'align_sample_id_groups: no tail microbatch has enough ids to split'
            group1, group2 = split_group(sample_id_groups[i])
            if group1 is not None and group2 is not None:
                sample_id_groups[i] = group1
                sample_id_groups.append(group2)
                remainder -= 1
            i -= 1

        return sample_id_groups

    def get_groups_and_subsamples(self, sample_id_seqlens):
        """
        This function recursively forms groups of sub-samples such that all DPxCP ranks
        have a roughly balanced workload in the group.
        """
        groups = []
        sample_id_groups = []
        # We assign a sample_id to each sub-sample in order to track assignment to each GPU.
        sample_id_seqlens = sorted(sample_id_seqlens, key=lambda x: x[1], reverse=True)
        while sample_id_seqlens:
            mb, sample_id_seqlens, exec_times, sample_ids = self.next_hdp_group(
                sample_id_seqlens, self.get_total_workload, self.total_hdp_gpus
            )
            groups.append(mb)
            sample_id_groups.append(sample_ids)

        if (
            self.microbatch_group_size_per_vp_stage is not None
            and self.microbatch_group_size_per_vp_stage > 1
        ):
            sample_id_groups = self.align_sample_id_groups(sample_id_groups)

        return sample_id_groups


def get_batch_on_this_rank_for_sequence_packing(
    data_iterator,
    mtp_on_this_rank: bool = False,
    vp_stage: Optional[int] = None,
    hybrid_context_parallel: bool = False,
):
    """
    Get a batch of data for sequence packing.
    Args:
        data_iterator (Iterator): The data iterator to get the batch from.
        mtp_on_this_rank (bool): Whether to use multi-token prediction.
        vp_stage (Optional[int]): The stage of the pipeline.
        hybrid_context_parallel (bool): Whether to use hybrid context parallel.
    Returns:
        tuple of (tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params)
    """

    def _broadcast_to_tp_group(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                parallel_state.get_tensor_model_parallel_src_rank(),
                group=parallel_state.get_tensor_model_parallel_group(),
            )

    is_tp_rank_0 = parallel_state.get_tensor_model_parallel_rank() == 0
    is_first_stage = parallel_state.is_pipeline_first_stage(
        ignore_virtual=vp_stage is None, vp_stage=vp_stage
    )
    is_last_stage = parallel_state.is_pipeline_last_stage(
        ignore_virtual=vp_stage is None, vp_stage=vp_stage
    )
    is_first_or_last_stage = is_first_stage or is_last_stage
    dev = torch.cuda.current_device()

    # data_iterator should return a batch including the following keys.
    batch_keys = [
        'tokens',
        'position_ids',
        'labels',
        'loss_mask',
        'cu_seqlens',
        'cu_seqlens_padded',
        'max_seqlen',
    ]
    if hybrid_context_parallel:
        batch_keys.append('local_cp_size')

    # Get a batch from data_iterator or create an emtpy batch.
    if is_tp_rank_0:
        assert data_iterator is not None
        batch = next(data_iterator)
        for key in batch_keys:
            assert key in batch, f"{key} is missing in current batch"
    else:
        assert data_iterator is None, "Non TP 0 rank should not have data_iterator"
        batch = {}

    # Partition tokens, position_ids, labels, loss_mask for context parallel, currently only
    # TP rank 0 and the first/last PP stage rank has these data.
    if is_tp_rank_0 and is_first_or_last_stage:
        # Get the proper cp_size and cp_rank based on hybrid context parallel is enabled or not.
        if hybrid_context_parallel:
            cp_size = batch['local_cp_size']
            if type(cp_size) == torch.Tensor:
                cp_size = cp_size.item()
            cp_rank = torch.distributed.get_rank(
                group=parallel_state.get_hybrid_data_context_parallel_groups(group_size=cp_size)
            )
        else:
            cp_size = parallel_state.get_context_parallel_world_size()
            cp_rank = parallel_state.get_context_parallel_rank()
        # If cp_size == 1, no need to do further processing.
        if cp_size > 1:
            assert tex is not None and is_te_min_version("1.10.0"), (
                "Please update Transformer Engine to >= 1.10 to use "
                "Context Parallel with THD format data"
            )
            total_tokens = batch['tokens'].size(0)
            # Transformer Engine has a bug of cu_seqlens, we must treat cu_seqlens_padded as
            # cu_seqlens to get the correct result.
            # TODO: Revert this workaround once TE fixes the issue.
            cu_seqlens = batch["cu_seqlens_padded"]
            index = tex.thd_get_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)
            for key in ['tokens', 'position_ids', 'labels', 'loss_mask']:
                batch[key] = batch[key].index_select(0, index)

    # Broadcast cu_seqlens_size because we need it to create placeholder for cu_seqlens and
    # cu_seqlens_padded for non TP 0 ranks.
    if is_tp_rank_0:
        cu_seqlen_size = torch.tensor(batch['cu_seqlens'].size(0), dtype=torch.int32, device=dev)
    else:
        cu_seqlen_size = torch.empty(1, dtype=torch.int32, device=dev)
    _broadcast_to_tp_group(cu_seqlen_size)
    cu_seqlen_size = cu_seqlen_size.item()

    # Broadcast total_tokens because we need it to create placeholder for tokens, position_ids,
    # labels, loss_mask for non TP 0 ranks. Only first or last stage need this.
    if is_first_or_last_stage:
        if is_tp_rank_0:
            total_tokens = torch.tensor(batch['tokens'].size(0), dtype=torch.int32, device=dev)
        else:
            total_tokens = torch.empty(1, dtype=torch.int32, device=dev)
        _broadcast_to_tp_group(total_tokens)
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
    if is_last_stage:
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

    # Step4(optional): Prepare "local_cp_size" if hybrid context parallel is enabled.
    if hybrid_context_parallel:
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

    # Broadcast batch inside TP group.
    _broadcast_to_tp_group(batch['tokens'])
    _broadcast_to_tp_group(batch['position_ids'])
    _broadcast_to_tp_group(batch['labels'])
    _broadcast_to_tp_group(batch['loss_mask'])
    _broadcast_to_tp_group(batch['cu_seqlens'])
    _broadcast_to_tp_group(batch['cu_seqlens_padded'])
    _broadcast_to_tp_group(batch['max_seqlen'])
    if hybrid_context_parallel:
        _broadcast_to_tp_group(batch['local_cp_size'])

    # Extract the data from batch after broadcasting.
    tokens = batch['tokens']
    position_ids = batch['position_ids']
    labels = batch['labels']
    loss_mask = batch['loss_mask']
    cu_seqlens = batch['cu_seqlens']
    cu_seqlens_padded = batch['cu_seqlens_padded']
    max_seqlen = batch['max_seqlen'].item()

    # Set the proper cp_group and local_cp_size when hybrid context parallel is enabled.
    if hybrid_context_parallel:
        local_cp_size = batch['local_cp_size'].item()
        cp_group = parallel_state.get_hybrid_data_context_parallel_groups(group_size=local_cp_size)
    else:
        local_cp_size = None
        cp_group = None

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
