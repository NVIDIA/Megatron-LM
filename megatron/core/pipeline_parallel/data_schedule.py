# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import enum
import sys
import copy
import nvtx
from collections import deque
from functools import lru_cache
import math
from math import ceil, log2
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.multiprocessing as mp

from megatron.core import parallel_state
from megatron.core.datasets.megatron_dataset import MegatronDataset

# from megatron.core.pipeline_parallel.utils import (
#     is_pp_first_stage,
#     is_pp_last_stage,
#     is_vp_first_stage,
#     is_vp_last_stage,
# )
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunDataIterator

# time simulator
from megatron.pipeline_simulator.simulator.schedules import SplitFuseSchedule, InterleavedSchedule
from megatron.pipeline_simulator.simulator.solver import test_with_schedule

class PackingScheduler(enum.Enum):
    """Enum for supported sequence packing algorithms."""

    HYBRID_CP = "hybrid_cp"
    HYBRID_CP_WITH_PP = "hybrid_cp_with_pp"
    NAIVE_SEQUENCE_PACKING = "naive_sequence_packing"
    # schedule in data_samplers, only need to pack, no need to schedule
    ONLY_PACKING_NO_SCHEDULING = "only_packing_no_scheduling"


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
    if torch.distributed.get_rank() == 0: print(f"{scheduler_type=}")

    scheduler_map = {
        "hybrid_cp": BalancedHybridCPscheduler,
        "hybrid_cp_with_pp": PipelineAwareBalancedHybridCPscheduler,
        "naive": NaiveSequencePackingScheduler,
        "only_packing_no_scheduling": OnlyPackingNoSchedulingScheduler,
    }

    scheduler_map: Dict[PackingScheduler, Type[BaseScheduler]] = {
        PackingScheduler.HYBRID_CP_WITH_PP: PipelineAwareBalancedHybridCPscheduler,
        PackingScheduler.HYBRID_CP: BalancedHybridCPscheduler,
        PackingScheduler.NAIVE_SEQUENCE_PACKING: NaiveSequencePackingScheduler,
        PackingScheduler.ONLY_PACKING_NO_SCHEDULING: OnlyPackingNoSchedulingScheduler,
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
        global_ids = torch.arange(len(seqlens_gathered), dtype=torch.int32)
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

    def cast_inputs_device(inputs, device, skip_device={}):
        if isinstance(inputs, (list, tuple)):
            return inputs.__class__(cast_inputs_device(v, device, skip_device) for v in inputs)
        elif isinstance(inputs, dict):
            return {k: v if k in skip_device else cast_inputs_device(v, device, skip_device=skip_device) for k, v in inputs.items()}
        elif isinstance(inputs, torch.Tensor):
            if not inputs.is_cuda:
                inputs = inputs.to(device=device, non_blocking=True) # here input is expected to be pinned

        return inputs

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
                else torch.empty(0, device=torch.cuda.current_device(), dtype=batch[0][key].dtype)
            )

        def _unpack_sample_by_key(key: str, recv_tensor: torch.Tensor):
            cursor = 0
            for i, gid in enumerate(recv_ids_sorted):
                sample_len = 1 if key in ["original_seq_len"] else global_id_seqlens[gid][1]
                recv_samples[i][key] = recv_tensor[cursor : cursor + sample_len]
                cursor += sample_len

        for key in data_keys:
            output_split_sizes, input_split_sizes = (
                (recv_counts, send_num_split)
                if key in ["original_seq_len"]
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
        samples: List[MegatronDataset], partner_cp_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        # TODO(tailaim): do we need attention_mask for sequence packing?

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
        if partner_cp_size is not None:
            new_sample["local_cp_size"] = torch.tensor(
                partner_cp_size, dtype=torch.int32, device=dev
            )

        # create cu_seqlens_padded
        lengths_padding = np.fromiter(
            (s["tokens"].numel() for s in samples), dtype=np.int32, count=len(samples)
        )
        cu_seqlens_padded = np.empty(len(samples) + 1, dtype=np.int32)
        cu_seqlens_padded[0] = 0
        cu_seqlens_padded[1:] = np.cumsum(lengths_padding, out=cu_seqlens_padded[1:])
        cu_seqlens_padded = (
            torch.from_numpy(cu_seqlens_padded)
            .to(device=dev, non_blocking=True, dtype=torch.int32)
            .reshape(-1)
        )
        new_sample["cu_seqlens_padded"] = cu_seqlens_padded

        # create max_seqlen
        max_seqlen = np.max(lengths_padding)
        max_seqlen = torch.tensor(max_seqlen, device=dev, dtype=torch.int32)
        new_sample["max_seqlen"] = max_seqlen

        # create cu_seqlens without padding
        lengths = torch.stack([s["original_seq_len"] for s in samples], dim=0).reshape(-1)
        cu_seqlens = torch.empty(lengths.numel() + 1, device=dev, dtype=torch.int32)
        cu_seqlens[0] = 0
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0).reshape(-1)
        new_sample["cu_seqlens"] = cu_seqlens

        return new_sample

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

    scheduler = scheduler_map[scheduler_type](config)
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
        # indicates TP rank 0, with PP stage 0 or -1.
        local_cp_size = None
        if scheduler_type is PackingScheduler.ONLY_PACKING_NO_SCHEDULING:
            # ONLY_PACKING_NO_SCHEDULING scheduler does not schedule the data,
            # just packing sequences

            # batch is a list of samples: List[MegatronDataset]
            batch = next(data_iterator)
            batch = cast_inputs_device(batch, dev)
            # print(f"{batch=}")
            num_micro_batches = batch[0]["num_micro_batches_left"] + 1

            batch_all = [batch] + [next(data_iterator) for _ in range(num_micro_batches - 1)]

            # calculate this two values for tflops calculation
            seqlens_gathered = [
                sample["tokens"].numel() for samples in batch_all for sample in samples
            ]
            num_total_tokens = 0
            sequence_square_sum = 0

            # pack sequences in the same group and create a new data iterator
            new_samples = []
            for samples in batch_all:
                partner_cp_size = samples[0]["local_cp_size"]
                new_sample = _pack_sequences(samples, partner_cp_size)
                new_samples.append(new_sample)
                for sample in samples:
                    num_total_tokens += sample["tokens"].numel() / partner_cp_size
                    sequence_square_sum += sample["tokens"].numel() ** 2 / partner_cp_size

        elif (
            scheduler_type is PackingScheduler.HYBRID_CP
            or scheduler_type is PackingScheduler.HYBRID_CP_WITH_PP
            or scheduler_type is PackingScheduler.NAIVE_SEQUENCE_PACKING
        ):
            batch = next(data_iterator)
            batch = cast_inputs_device(batch, dev)
            subsample_seqlens = []
            for sample in batch:
                subsample_seqlens.extend([sample["tokens"].numel()])
            subsample_seqlens = torch.tensor(subsample_seqlens, dtype=torch.int32).cuda()
            subsample_seqlens = subsample_seqlens[subsample_seqlens != 0]

            nvtx.push_range("_get_global_seqlens")
            seqlens_gathered, offsets = _get_global_seqlens(subsample_seqlens, dp_group)
            nvtx.pop_range()

            nvtx.push_range("_get_global_id_seqlens")
            global_id_seqlens, global_ids_this_rank = _get_global_id_seqlens(
                subsample_seqlens.shape[0], offsets, seqlens_gathered, dp_group
            )
            nvtx.pop_range()

            nvtx.push_range("scheduler.get_groups_and_subsamples")
            groups, sample_id_groups = scheduler.get_groups_and_subsamples(
                global_id_seqlens, config
            )
            nvtx.pop_range()

            set_gbs = set()
            for group in sample_id_groups:
                for sub in group:
                    set_gbs.update(sub)
            assert len(set_gbs) == len(
                global_id_seqlens
            ), f"set_gbs length: {len(set_gbs)} \
            != global_ids_this_rank length: {len(global_id_seqlens)}"

            nvtx.push_range("_unpack_batch")
            batch = _unpack_batch(batch)
            nvtx.pop_range()
            
            nvtx.push_range("_reroute_samples_to_hdp_ranks")
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
            nvtx.pop_range()
            batch, sample_id_groups = samples_this_rank_with_id, sample_id_groups

            hdp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
            num_micro_batches = len(sample_id_groups)
            # calculate this two values for tflops calculation
            num_total_tokens_this_GB = np.int64(sum(seqlens_gathered))
            sequence_square_sum_this_GB = np.int64(sum(seqlen**2 for seqlen in seqlens_gathered))

            new_samples = []
            cp_sizes = []
            for i in range(num_micro_batches):
                # pack sequences in the same group and create a new data iterator
                sample_ids_this_group = sample_id_groups[i][hdp_rank]
                samples = [batch[sub_sample_id] for sub_sample_id in sample_ids_this_group]
                partner_cp_size = (
                    len(
                        [
                            True
                            for sample_ids in sample_id_groups[i]
                            if sample_ids_this_group[0] in sample_ids
                        ]
                    )
                    if config.hybrid_context_parallel
                    else None
                )
                nvtx.push_range("_pack_sequences")
                cp_sizes.append(partner_cp_size)
                new_sample = _pack_sequences(samples, partner_cp_size)
                nvtx.pop_range()
                new_samples.append(new_sample)
            
            # if parallel_state.get_pipeline_model_parallel_rank() == 0: print(f"rank={torch.distributed.get_rank()}, {cp_sizes=}\n{sample_id_groups[0][hdp_rank]=}\n{sample_id_groups[1][hdp_rank]=}\n{sample_id_groups[2][hdp_rank]=}")

        if scheduler_type is PackingScheduler.ONLY_PACKING_NO_SCHEDULING:
            # allreduce to get the total number of microbatches
            mfu_info_to_broadcast_this_hdp_group = torch.tensor(
                [num_total_tokens, sequence_square_sum], dtype=torch.int64, device=dev
            )
            torch.distributed.all_reduce(mfu_info_to_broadcast_this_hdp_group, group=dp_cp_group)
            num_total_tokens_this_GB = mfu_info_to_broadcast_this_hdp_group[0].item()
            print(f"{num_total_tokens_this_GB=}")
            sequence_square_sum_this_GB = mfu_info_to_broadcast_this_hdp_group[1].item()

    # # broadcast num_micro_batches, num_total_tokens_this_GB, sequence_square_sum_this_GB,
    # #  and packed_seq_params to tp group
    # if pp_group.size() > 2 and tp_group.rank() == 0:
    #     if pp_group.rank() == 0:
    #         tensor_list = [
    #             torch.tensor(
    #                 [num_micro_batches, num_total_tokens_this_GB, sequence_square_sum_this_GB],
    #                 dtype=torch.int64,
    #             ).cuda()
    #         ]
    #         for sample in new_samples:
    #             tensor_list.append(sample["max_seqlen"].unsqueeze(0))
    #         for sample in new_samples:
    #             tensor_list.append(
    #                 sample["local_cp_size"].unsqueeze(0)
    #                 if scheduler_type is PackingScheduler.HYBRID_CP
    #                 else torch.tensor([-1], dtype=torch.int32).cuda()
    #             )
    #         for sample in new_samples:
    #             tensor_list.append(sample["cu_seqlens"])
    #             tensor_list.append(sample["cu_seqlens_padded"])
    #         info_to_broadcast_this_pp_group = torch.cat(tensor_list, dim=0).to(
    #             device=dev, dtype=torch.int64
    #         )
    #         info_length_tensor = torch.tensor(
    #             info_to_broadcast_this_pp_group.shape[0], dtype=torch.int32
    #         ).cuda()
    #         _broadcast_to_pp_group(info_length_tensor)
    #         _broadcast_to_pp_group(info_to_broadcast_this_pp_group)
    #     else:
    #         info_length_tensor = torch.tensor(0, dtype=torch.int32).cuda()
    #         _broadcast_to_pp_group(info_length_tensor)
    #         info_to_broadcast_this_pp_group = torch.empty(
    #             info_length_tensor.item(), dtype=torch.int64
    #         ).cuda()
    #         _broadcast_to_pp_group(info_to_broadcast_this_pp_group)
    #         if pp_group.rank() != pp_group.size() - 1:
    #             info_numpy = info_to_broadcast_this_pp_group.cpu().numpy()
    #             num_micro_batches = info_numpy[0]
    #             num_total_tokens_this_GB = info_numpy[1]
    #             sequence_square_sum_this_GB = info_numpy[2]
    #             max_seqlens = info_numpy[3 : 3 + num_micro_batches]
    #             local_cp_sizes = info_numpy[3 + num_micro_batches : 3 + 2 * num_micro_batches]
    #             cu_seqlens_list = []
    #             cu_seqlens_padded_list = []
    #             indices = np.where(info_numpy == 0)[0]
    #             for i in range(num_micro_batches):
    #                 cu_seqlens_list.append(info_numpy[indices[i * 2] : indices[i * 2 + 1]])
    #                 if i == num_micro_batches - 1:
    #                     cu_seqlens_padded_list.append(info_numpy[indices[i * 2 + 1] :])
    #                 else:
    #                     cu_seqlens_padded_list.append(
    #                         info_numpy[indices[i * 2 + 1] : indices[i * 2 + 2]]
    #                     )

    #             new_samples = []
    #             for i in range(num_micro_batches):
    #                 new_sample = {}
    #                 new_sample["max_seqlen"] = torch.tensor(
    #                     max_seqlens[i], dtype=torch.int32
    #                 ).cuda()
    #                 if local_cp_sizes[i] != -1:
    #                     new_sample["local_cp_size"] = torch.tensor(
    #                         local_cp_sizes[i], dtype=torch.int32
    #                     ).cuda()
    #                 new_sample["cu_seqlens"] = torch.tensor(
    #                     cu_seqlens_list[i], dtype=torch.int32
    #                 ).cuda()
    #                 new_sample["cu_seqlens_padded"] = torch.tensor(
    #                     cu_seqlens_padded_list[i], dtype=torch.int32
    #                 ).cuda()
    #                 new_samples.append(new_sample)

    if tp_group.size() > 1:
        if tp_group.rank() == 0:
            info_to_broadcast_this_tpgroup = torch.tensor(
                [num_micro_batches, num_total_tokens_this_GB, sequence_square_sum_this_GB],
                dtype=torch.int64,
                device=dev,
            )
            _broadcast_to_tp_group(info_to_broadcast_this_tpgroup)
        else:
            info_to_broadcast_this_tpgroup = torch.tensor([0, 0, 0], dtype=torch.int64, device=dev)
            _broadcast_to_tp_group(info_to_broadcast_this_tpgroup)
            info_numpy = info_to_broadcast_this_tpgroup.cpu().numpy()
            (num_micro_batches, num_total_tokens_this_GB, sequence_square_sum_this_GB) = info_numpy[
                :3
            ]

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
                    if config.hybrid_context_parallel:
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
        num_total_tokens_this_GB,
        sequence_square_sum_this_GB,
    )


class BaseScheduler:
    """
    Base class for sequence packing schedulers.
    """

    def __init__(self, config):
        pass


class NaiveSequencePackingScheduler(BaseScheduler):
    """
    This scheduler simply packs sequences in their original order
    until reaching the max sequence length.
    It does not reorder sequences nor perform any load balancing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.dp_size = int(parallel_state.get_data_parallel_world_size())
        self.cp_size = int(parallel_state.get_context_parallel_world_size())
        self.max_seq_len_all_ranks = config.max_seqlen_per_dp_cp_rank * self.cp_size

    def get_groups_and_subsamples(self, sample_id_seqlens, config):
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
            if sum_seqlen + sample_id_seqlens[i][1] <= self.max_seq_len_all_ranks:
            # if flag and sum_seqlen + sample_id_seqlens[i][1] <= self.max_seq_len_all_ranks:
                # flag = False
                single_microbatch.append(i)
                sum_seqlen += sample_id_seqlens[i][1]
            else:
                packed_id_groups.append(single_microbatch)
                single_microbatch = [i]
                sum_seqlen = sample_id_seqlens[i][1]
        if len(single_microbatch) > 0:
            packed_id_groups.append(single_microbatch)

        gbs_sum = 0
        for i in packed_id_groups:
            gbs_sum += len(i)
        assert gbs_sum == len(
            sample_id_seqlens
        ), f"gbs_sum: {gbs_sum} != sample_id_seqlens length: {len(sample_id_seqlens)}"

        groups.append(single_microbatch)
        packed_id_groups.append(single_microbatch)

        # we want the number of packed sequences to be multiple of dp_size
        # so we move few samples from previous microbatch
        # to the end of the microbatches if needed
        num_packed_sequence = len(packed_id_groups)
        if num_packed_sequence % self.dp_size != 0:
            # print(f"{num_packed_sequence=}, {self.dp_size=}, {len(sample_id_seqlens)=}")
            remainder = num_packed_sequence % self.dp_size
            num_to_move = self.dp_size - remainder
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
        return groups, sample_id_groups


class BalancedHybridCPscheduler(BaseScheduler):
    """
    This class provides the functionality to form groups of sub-samples
    such that all DPxCP ranks have a roughly balanced workload in the group.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_seq_len_per_rank = config.max_seqlen_per_dp_cp_rank
        self.num_subsamples = 0
        self.num_subsamples_processed = 0
        self.free_resources = []
        self.total_hdp_gpus = parallel_state.get_data_parallel_world_size(
            with_context_parallel=True
        )

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
    ) -> List[deque]:
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

    def get_groups_and_subsamples(self, sample_id_seqlens, config):
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
            if len(sample_ids) < self.total_hdp_gpus:
                sample_ids.extend([] * (self.total_hdp_gpus - len(sample_ids)))
            sample_id_groups.append(sample_ids)

        # if torch.distributed.get_rank() == 0:
        #     breakpoint()
        # torch.distributed.barrier()
        # if torch.distributed.get_rank() == 0: print(f"rank={torch.distributed.get_rank()}, {groups=}")
        # print(f"rank={torch.distributed.get_rank()}, {sample_id_groups=}")
        return groups, sample_id_groups


def compute_pp_bubble_ratio(PP, m, VPP=1):
    return (PP - 1) / (m * VPP + PP - 1)


def greedy_assign_bucket_to_dp(curr_m, indices_buckets, normal_indexes, except_buckets, except_bucket_num_per_sample, 
                             except_bucket_m_per_sample, except_bucket_dp_per_sample, buckets_for_current_m,
                             dp_size_for_current_m, used_flops, used_fwd_flops, used_bwd_flops, bucket_num_per_dp_curr_m,
                             all_flops, all_lengths, combination=None, config=None):
    """
    使用贪心算法将桶分配给数据并行(DP)组
    
    参数:
        curr_m: 当前处理的m值(微批次数量)
        indices_buckets: 所有桶的索引信息
        except_buckets: 特殊处理的序列桶
        except_bucket_num_per_sample: 每个特殊序列分配的桶数
        except_bucket_m_per_sample: 每个特殊序列分配的m值
        buckets_for_current_m: 当前m值对应的桶列表
        dp_size_for_current_m: 当前m值的DP组大小
        used_flops: 已使用的总FLOPs
        used_fwd_flops: 已使用的前向FLOPs
        used_bwd_flops: 已使用的后向FLOPs
        bucket_num_per_dp_curr_m: 每个DP rank的桶数量限制
        
    返回:
        包含以下内容的元组:
        - 每个DP rank的总FLOPs列表
        - 每个DP rank的前向FLOPs列表
        - 每个DP rank的后向FLOPs列表
        - 分配给每个DP rank的桶列表
        - 是否遇到空桶的标志
    """

    # args = get_args()
    
    # 初始化每个DP rank的统计列表
    fwd_flops_for_dp_per_m = [[] for _ in range(dp_size_for_current_m)]  # 前向FLOPs
    bwd_flops_for_dp_per_m = [[] for _ in range(dp_size_for_current_m)]  # 后向FLOPs
    seq_len_for_dp_per_m = [[] for _ in range(dp_size_for_current_m)]  # 每个 microbatch 的 seqlen
    buckets_for_dp = [[] for _ in range(dp_size_for_current_m)]  # 分配的桶
    sample_ids_for_dp = [[] for _ in range(dp_size_for_current_m)]  # 分配的 sample_id
    sample_lengths_for_dp = [[] for _ in range(dp_size_for_current_m)]  # 分配的 sample_length

    # 初始化每个DP rank的FLOPs总和和已用桶数
    fwd_flops_sum_per_dp_this_m = [0.0] * dp_size_for_current_m
    bucket_used_num_per_dp_this_m = [0] * dp_size_for_current_m
    prefix_sum_per_dp_this_m = 0  # 用于跟踪特殊序列桶的分配位置
    # 第一步：分配特殊序列(Seq1F1B)的桶
    # 遍历每个样本，判断其是否被分配到当前 m
    num_split_for_dp = [0] * dp_size_for_current_m
    # print_rank0(f"assign except bucket")
    for idx in range(len(except_bucket_m_per_sample)):
        # 只处理当前m值的特殊序列
        if (except_bucket_m_per_sample[idx] != curr_m):
            continue

        # 计算当前序列在except_buckets中的位置范围
        st = prefix_sum_per_dp_this_m
        ed = prefix_sum_per_dp_this_m + except_bucket_num_per_sample[idx]

        # 将序列片段分配到选定的DP rank
        for k in range(st, ed):
            # 记录FLOPs信息
            bucket_tmp = except_buckets[curr_m][k]

            bucket_tmp.fwd_flops = (bucket_tmp.fwd_flops, {}, bucket_tmp.cp_size, bucket_tmp.dp_index)
            bucket_tmp.bwd_flops = (bucket_tmp.bwd_flops, {}, bucket_tmp.cp_size, bucket_tmp.dp_index)

            fwd_flops_for_dp_per_m[bucket_tmp.dp_index].append(bucket_tmp.fwd_flops)
            bwd_flops_for_dp_per_m[bucket_tmp.dp_index].append(bucket_tmp.bwd_flops)
            
            # construct 2d array
            # correction for memory simulator
            seq_len_for_dp_per_m[bucket_tmp.dp_index].append([bucket_tmp.seq_len_sum // bucket_tmp.cp_size // config.min_hybrid_context_parallel_size * config.context_parallel_size])
            
            # 更新DP rank的负载统计
            fwd_flops_sum_per_dp_this_m[bucket_tmp.dp_index] += bucket_tmp.fwd_flops[0]

            buckets_for_dp[bucket_tmp.dp_index].append(bucket_tmp)
            sample_ids_for_dp[bucket_tmp.dp_index].append(bucket_tmp.samples)

            # 更新分配位置和桶使用计数
            num_split_for_dp[bucket_tmp.dp_index] += 1
            bucket_used_num_per_dp_this_m[bucket_tmp.dp_index] += 1
        prefix_sum_per_dp_this_m += except_bucket_num_per_sample[idx]
        # for ttt in bucket_used_num_per_dp_this_m:
        #     print_rank0(ttt, end='\t')
        # print_rank0("")

    # 第二步：分配普通桶
    # print_rank0(f"assign normal bucket")
    empty_bucket_flag = False
    for j in range(len(buckets_for_current_m)):
        # 寻找最适合的DP rank(负载最小且桶未满)
        min_flops = sys.float_info.max
        min_flops_dp_rank = -1
        for dp_rank in range(len(fwd_flops_sum_per_dp_this_m)):
            if (min_flops > fwd_flops_sum_per_dp_this_m[dp_rank]) and \
               (bucket_used_num_per_dp_this_m[dp_rank] < bucket_num_per_dp_curr_m):
                min_flops = fwd_flops_sum_per_dp_this_m[dp_rank]
                min_flops_dp_rank = dp_rank

        assert min_flops_dp_rank != -1  # 确保找到合适的DP rank

        # 获取当前桶ID并检查是否为空
        bucket_id = buckets_for_current_m[j][1]
        if not indices_buckets[bucket_id] or len(indices_buckets[bucket_id].samples) == 0:
            # for idx, except_b in enumerate(except_buckets[curr_m]):
            #     print_rank0(f"{idx=}, {except_b=}")
            #     print_rank0(except_b)
            # for idx, normal_b in enumerate(indices_buckets):
            #     print_rank0(f"{idx=}, {normal_b=}")
            #     print_rank0(normal_b)
            # import pdb; pdb.set_trace()
            empty_bucket_flag = True
        
        # for test only
        indices_buckets[bucket_id].samples_fwd_flops = [all_flops[1][indice] for indice in indices_buckets[bucket_id].samples]

        # tflops to time
        scale = 0.5
        length_sum = 0
        length_square_sum = 0
        # attn_fwd_tflops_sum = 0
        # gemm_fwd_tflops_sum = 0
        lengths = []
        # NOTE shenglong 
        hidden_size = config.hidden_size
        # hidden_size = 4096

        bucket_tmp = indices_buckets[bucket_id]
        for sample_id in bucket_tmp.samples:
            length = all_lengths[sample_id]
            # attn_fwd_tflops = attention_tflops(length, hidden_size, scale)
            # gemm_fwd_tflops = linear_tflops(length, config.hidden_size)

            length_sum += length
            lengths.append(length)
            length_square_sum += (length ** 2)
            # attn_fwd_tflops_sum += attn_fwd_tflops
            # gemm_fwd_tflops_sum += gemm_fwd_tflops

        # fwd_time, bwd_time, fwd_time_dict = flops_to_times(length_sum, length_square_sum, attn_fwd_tflops_sum)
        fwd_time, bwd_time = bucket_tmp.fwd_flops, bucket_tmp.bwd_flops     # TODO(wuguohao)
        fwd_time_dict = {}

        split_num = 1
        split_idx = 0
        bwd_time_dict = {} # TODO
        bucket_tmp.fwd_flops = (bucket_tmp.fwd_flops, fwd_time_dict, split_num, split_idx)
        bucket_tmp.bwd_flops = (bucket_tmp.bwd_flops, bwd_time_dict, split_num, split_idx)
        # bucket_tmp.fwd_flops = (bucket_tmp.fwd_flops, {"attn_fwd_time":attn_fwd_tflops_sum, "mlp_fc1_fwd_time":gemm_fwd_tflops_sum})

        # print_rank0(f"{lengths=}")
        assert length_sum == bucket_tmp.seq_len_sum, f"{length_sum=}, {bucket_tmp.seq_len_sum=}, {bucket_id=}"

        # !将带 offset 的 data index 替换为真实的 data index
        # indices_buckets[bucket_id].samples = [normal_indexes[indice] for indice in indices_buckets[bucket_id].samples]

        # 将桶分配给选定的DP rank
        # fwd_flops_for_dp_per_m[min_flops_dp_rank].append(used_fwd_flops[bucket_id])
        # bwd_flops_for_dp_per_m[min_flops_dp_rank].append(used_bwd_flops[bucket_id])
        fwd_flops_for_dp_per_m[min_flops_dp_rank].append(bucket_tmp.fwd_flops)
        bwd_flops_for_dp_per_m[min_flops_dp_rank].append(bucket_tmp.bwd_flops)
        # correction for memory simulator
        seq_len_for_dp_per_m[min_flops_dp_rank].append([bucket_tmp.seq_len_sum // config.min_hybrid_context_parallel_size * config.context_parallel_size])
        buckets_for_dp[min_flops_dp_rank].append(bucket_tmp)
        sample_ids_for_dp[min_flops_dp_rank].append(bucket_tmp.samples)

        # 更新DP rank的负载统计
        fwd_flops_sum_per_dp_this_m[min_flops_dp_rank] += (bucket_tmp.fwd_flops[0])
        bucket_used_num_per_dp_this_m[min_flops_dp_rank] += 1
        # for ttt in bucket_used_num_per_dp_this_m:
        #     print_rank0(ttt, end='\t')
        # print_rank0("")

    # print_rank0(f"aft asign normal bucket, {dp_size_for_current_m=}, {bucket_used_num_per_dp_this_m=}")
    
    for dp_rank in range(len(buckets_for_dp)):
        # print_rank0(f"rank {torch.distributed.get_rank()} bucket num for dp{len(buckets_for_dp[dp_rank])}")
        # num_fused = sum(1 for b in buckets_for_dp[dp_rank] if not isinstance(b, SplitBucket))
        for bucket_i, bucket in enumerate(buckets_for_dp[dp_rank]):
            bucket.num_split_bucket_this_dp = num_split_for_dp[dp_rank]
            # if isinstance(bucket, SplitBucket):
            #     # print_rank0(f"{dp_rank=}, {bucket_i=}, {bucket.fwd_flops=}")
            # else:
            #     # print_rank0(f"{dp_rank=}, {bucket_i=}, {bucket.samples_fwd_flops=} {len(bucket.samples)=}")
    assert len(buckets_for_dp) == len(sample_ids_for_dp), f"{len(sample_ids_for_dp)=}, {len(buckets_for_dp)=}"
    return fwd_flops_for_dp_per_m, bwd_flops_for_dp_per_m, buckets_for_dp, sample_ids_for_dp, seq_len_for_dp_per_m, empty_bucket_flag



def fwd_flops_update_rule(bucket, index, all_density, all_lengths, all_flops):
    if bucket.seq_len_sum + all_lengths[index] > bucket.target_length:    # add memory limit.
        return None

    new_fwd_flops = bucket.fwd_flops + all_flops[1][index]
    return new_fwd_flops * (new_fwd_flops / bucket.target_flops)


def length_update_rule(bucket, index, all_density, all_lengths, all_flops):
    return ((bucket.seq_len_sum + all_lengths[index]) - bucket.target_length)** 2 / bucket.target_length**2


class UpdateRule(enum.Enum):
    DENSITY = 1
    FW_FLOPS = 2
    LENGTH = 3


update_rule_mapping = {
    UpdateRule.FW_FLOPS: fwd_flops_update_rule,
    UpdateRule.LENGTH: length_update_rule
}


def fwd_flops_to_bwd_flops(pre_attn_fwd_time, attn_fwd_time, post_attn_fwd_time, mlp_fwd_time):
    attn_bwd_time = 2.77 * attn_fwd_time
    pre_attn_bwd_time = 2.7 * pre_attn_fwd_time
    post_attn_bwd_time = 2.7 * post_attn_fwd_time
    mlp_bwd_time = 2.7 * mlp_fwd_time

    return pre_attn_bwd_time, attn_bwd_time, post_attn_bwd_time, mlp_bwd_time


def attention_tflops(s, h, scale):
    # NOTE: only consider forward tflops
    s2 = s**2
    tflops = 2 * 2 * s2 * h / 1e12 * scale
    return tflops


def linear_tflops(s, h1, h2):
    # NOTE: only consider forward tflops
    tflops = 2 * s * h1 * h2 / 1e12
    return tflops


def TFLOPs(s1, config):
    """
        Only calculate one block TFLOPs here.
    """
    scale = 0.5

    ####### forward tflops ########
    gemm_fwd_tflops = linear_tflops(s1, config.hidden_size, config.hidden_size)
    attn_fwd_tflops = attention_tflops(s1, config.hidden_size, scale)

    pre_attn_fwd_tflops = 3 * gemm_fwd_tflops
    if config.num_query_groups is not None:
        pre_attn_fwd_tflops = gemm_fwd_tflops + 2 * gemm_fwd_tflops / config.num_query_groups
    post_attn_fwd_tflops = gemm_fwd_tflops
    mlp_fc1_h = config.ffn_hidden_size * 2 if config.gated_linear_unit else config.ffn_hidden_size
    mlp_fc2_h = config.ffn_hidden_size
    mlp_fc1_fwd_tflops = linear_tflops(s1, config.hidden_size, mlp_fc1_h)
    mlp_fc2_fwd_tflops = linear_tflops(s1, mlp_fc2_h, config.hidden_size)

    fwd_tflops = pre_attn_fwd_tflops + attn_fwd_tflops + post_attn_fwd_tflops + mlp_fc1_fwd_tflops + mlp_fc2_fwd_tflops
    
    ####### backward tflops ########
    pre_attn_bwd_tflops, attn_bwd_tflops, post_attn_bwd_tflops, mlp_bwd_tflops = \
        fwd_flops_to_bwd_flops(pre_attn_fwd_tflops, attn_fwd_tflops, post_attn_fwd_tflops, mlp_fc1_fwd_tflops + mlp_fc2_fwd_tflops)

    bwd_tflops = pre_attn_bwd_tflops + attn_bwd_tflops + post_attn_bwd_tflops + mlp_bwd_tflops

    ####### recompute tflops ########
    if config.recompute_granularity == "full":
        bwd_tflops += fwd_tflops
    else:
        # TODO: add other recompute method here.
        pass

    tot_tflops = fwd_tflops + bwd_tflops
    return tot_tflops, fwd_tflops, bwd_tflops


def compute_ratios(combination, PP):
    # PP = mpu.get_pipeline_model_parallel_world_size()
    VPP = 1
    ratios = []
    for num_m in range(1, len(combination)+1):
        ratio = num_m * PP * (PP * VPP + PP - 1) / (num_m * PP * VPP + PP - 1)
        ratios.append(ratio)
    return ratios


class Bucket:
    def __init__(self, target_flops, target_density, target_length, bucket_id, cp_size, samples, fwd_flops=0, bwd_flops=0, seq_len_sum=0, dp_index=-1):
        self.bucket_id = bucket_id
        self.samples = samples
        self.cp_size = cp_size
        self.target_flops = target_flops
        self.target_density = target_density
        self.target_length = target_length
        self.current_density = 0
        self.fwd_flops = fwd_flops
        self.bwd_flops = bwd_flops
        self.seq_len_sum = seq_len_sum
        self.dp_index = dp_index
        self.type = "Bucket"

    def __str__(self):
        return (
            f"Bucket {self.bucket_id}:\n"
            f"  Target Flop: {self.target_flops}\n"
            f"  Target Density: {self.target_density}\n"
            f"  Target Length: {self.target_length}\n"
            f"  Current Density: {self.current_density}\n"
            f"  Forward Flops: {self.fwd_flops}\n"
            f"  Backward Flops: {self.bwd_flops}\n"
            f"  Sequence Length Sum: {self.seq_len_sum}\n"
            f"  Samples: {self.samples}\n"
        )


def create_buckets(num_buckets, avg_fwd_flops_with_m, max_seq_len_for_fuse):
    total_bucket_num = 0
    all_buckets = []

    for i in range(len(num_buckets)):
        for j in range(num_buckets[i]):
            target_density =  avg_fwd_flops_with_m[i] / max_seq_len_for_fuse
            all_buckets.append(
                Bucket(
                    target_flops = avg_fwd_flops_with_m[i], 
                    target_density = target_density, 
                    target_length = max_seq_len_for_fuse, 
                    bucket_id = total_bucket_num,
                    cp_size = 1,
                    samples = [],
                )
            )
            total_bucket_num += 1
    
    return total_bucket_num, all_buckets


def assign_samples_to_buckets(
    sorted_indices,
    buckets,
    all_density,
    all_lengths,
    all_flops,
    update_rule=None,
    remaining_sample_indices=None,
    print_score=False,
):

    preassigned_samples = []
    if update_rule is UpdateRule.DENSITY:
        raise Exception()
        print_rank0("using density update rule")
        pre_assign_sample_to_empty_bucket(sorted_indices, buckets, all_density, all_lengths, all_flops, remaining_sample_indices, preassigned_samples)
    else:
        assert len(preassigned_samples) == 0
    update_rule = update_rule_mapping[update_rule]

    for index in sorted_indices:
        if index in preassigned_samples:
            print(f"{index=}, {preassigned_samples=}")
            continue

        min_score = float('inf')
        target_bucket = None

        score = None
        for bucket in buckets:
            score = update_rule(bucket, index, all_density, all_lengths, all_flops)
            if score is not None and score < min_score:
                min_score = score
                target_bucket = bucket

        if target_bucket is not None:
            target_bucket.fwd_flops += all_flops[1][index]
            target_bucket.bwd_flops += (2 * all_flops[1][index])    # TODO(wuguohao): use more precisely bwd_flops
            target_bucket.seq_len_sum += all_lengths[index]
            target_bucket.samples.append(index)
            remaining_sample_indices.remove(index)
            # if torch.distributed.get_rank() == 0: print(f"pop {index=}, {len(remaining_sample_indices)=}, length_rule={update_rule == length_update_rule}, flops_rule={update_rule == fwd_flops_update_rule} {remaining_sample_indices=}")
        else:
            if update_rule == length_update_rule:
                if torch.distributed.get_rank() == 0: print(f"skip {index=}, {score=}, {min_score=} {len(buckets)=}, {len(remaining_sample_indices)=}, length_rule={update_rule == length_update_rule}, flops_rule={update_rule == fwd_flops_update_rule} ")

    return remaining_sample_indices


def nearest_pow2(n: int) -> int:
    """
    将正整数 n 四舍五入到最接近的 2 的幂。
    n < 1 时返回 1。
    """
    if n < 1:
        return 1
    # lower = 2^(⌊log2 n⌋)
    lower = 1 << (n.bit_length() - 1)
    # upper = 2^(⌈log2 n⌉)
    upper = 1 << n.bit_length()
    # 距离较小者
    return lower if (n - lower) < (upper - n) else upper


def simulate_memory(chunks_list, config):
    from megatron.pipeline_simulator.hotsim.model import Model
    from megatron.pipeline_simulator.hotsim.memory_model import MemoryModel
    from megatron.pipeline_simulator.hotsim.training_config import TrainingConfig
    from megatron.pipeline_simulator.hotsim.schedule import build_splitfuse_schedule
    model = Model(
        name="Llama",
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.ffn_hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
    )
    ckpt_type = "no"
    if config.recompute_granularity == "full":
        ckpt_type = "full"
    # if config.kaimm_recompute_mlp_activation_func and config.kaimm_recompute_norm:
    #     if config.kaimm_recompute_mlp_fc1:
    #         ckpt_type = "partial+fc1"
    #     else:
    #         ckpt_type = "partial"

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        num_gpus = torch.distributed.get_world_size()
    else:
        num_gpus = parallel_state.get_tensor_model_parallel_world_size() \
             * parallel_state.get_pipeline_model_parallel_world_size() \
             * parallel_state.get_data_parallel_world_size()
    train_config = TrainingConfig(
        model=model,
        num_gpus=num_gpus,
        microbatch_size=1,
        tensor_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
        context_parallel_size=parallel_state.get_context_parallel_world_size(),
        data_parallel_size=parallel_state.get_data_parallel_world_size(),
        pipeline_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
        expert_parallel_size=parallel_state.get_expert_model_parallel_world_size(),
        num_model_chunks=1,
        ckpt=ckpt_type,
        offload_ratio=0,
        # offload_ratio=config.kaimm_offload_activation_ratio,
    )

    actions_by_rank = build_splitfuse_schedule(
        config.pipeline_model_parallel_size, chunks_list
    )

    memory_model = MemoryModel(train_config)
    memory_model.setup(chunks_list, actions_by_rank)
    memory_model.run()
    return max(memory_model.peak_memory_histogram)


def simulate_time(fwd_costs, bwd_costs, PP, VPP):
    # PP = mpu.get_pipeline_model_parallel_world_size()
    schedule = SplitFuseSchedule(PP, fwd_costs, bwd_costs)
    # num_VPP = 8
    # schedule = InterleavedSchedule(PP, num_VPP, fwd_costs, bwd_costs)
    return test_with_schedule(schedule)


def fill_bucket_with_samples(
    curr_except_index,
    sorted_indices,
    target_flops,
    all_flops,
    all_lengths,
    max_seq_len,
    remaining_sample_indices=None,
    total_num=0,
    consumed_num_buckets=0,
    assign_all_sample_to_except_bucket_flag=False,
):
    # assume sorted_indices is sorted by fwd flops in reversed order.
    selected_indices = []
    selected_fwd_flops = []
    selected_bwd_flops = []
    selected_lengths = []
    remained_flops = target_flops
    # print_rank0(f"###{max_num_samples_to_fill=}")
    length_sum = all_lengths[curr_except_index]
    for index in sorted_indices:
        # if max_num_samples_to_fill == 0: break
        sample_fwd_flops = all_flops[1][index]
        # sample_bwd_flops = all_flops[2][index]
        sample_bwd_flops = all_flops[2][index]  # TODO(wuguohao): more precisely bwd_flops
        extra_limit = total_num < len(remaining_sample_indices) and length_sum < (max_seq_len * consumed_num_buckets)
        extra_limit = (assign_all_sample_to_except_bucket_flag) or extra_limit  # skip extra_limit if `assign_all_sample_to_except_bucket_flag` is True

        exceed_ratio = 1.05
        # if assign_all_sample_to_except_bucket_flag:
        #     exceed_ratio = 1.5
        if sample_fwd_flops < remained_flops * exceed_ratio and extra_limit: # TODO: consume num buckets * max seq len
            # if torch.distributed.get_rank() == 0: print(f"{target_flops=}, {index=}, {sample_fwd_flops=}, {sample_bwd_flops=}")
            remained_flops -= sample_fwd_flops
            selected_indices.append(index)
            selected_fwd_flops.append(sample_fwd_flops)
            selected_bwd_flops.append(sample_bwd_flops)
            selected_lengths.append(all_lengths[index])
            remaining_sample_indices.remove(index)
            length_sum += all_lengths[index]
            # max_num_samples_to_fill -= 1

    selected_flops = [selected_fwd_flops, selected_bwd_flops]
    
    return selected_indices, selected_flops, selected_lengths, remained_flops, remaining_sample_indices


class PipelineAwareBalancedHybridCPscheduler(BaseScheduler):
    
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_len_per_rank = config.max_seqlen_per_dp_cp_rank
        self.num_subsamples = 0
        self.num_subsamples_processed = 0
        self.free_resources = []
        self.total_hdp_gpus = parallel_state.get_data_parallel_world_size(
            with_context_parallel=True
        )

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

    def get_groups_and_subsamples(self, sample_id_seqlens, config, return_cp_sizes=False):
        """
        This function recursively forms groups of sub-samples such that all DPxCP ranks
        have a roughly balanced workload in the group.
        """
        groups = []
        sample_id_groups = []
        cp_sizes = []
        # We assign a sample_id to each sub-sample in order to track assignment to each GPU.
        sample_id_seqlens = sorted(sample_id_seqlens, key=lambda x: x[1], reverse=True)
        # while sample_id_seqlens:
        #     mb, sample_id_seqlens, exec_times, sample_ids = self.next_hdp_group(
        #         sample_id_seqlens, self.get_total_workload, self.total_hdp_gpus, config=config
        #     )
        #     groups.append(mb)
        #     if len(sample_ids) < self.total_hdp_gpus:
        #         sample_ids.extend([] * (self.total_hdp_gpus - len(sample_ids)))
        #     sample_id_groups.append(sample_ids)

        _, _, best_indices_buckets, best_sample_ids, best_dp_combination, _ = self.next_hdp_group(
            sample_id_seqlens, self.get_total_workload, self.total_hdp_gpus, config=config
        )

        # print(best_indices_buckets[-1][0][0])
        # breakpoint()

        mi = -1
        for i in range(len(best_indices_buckets)):
            if len(best_indices_buckets[i]) > 0:
                mi = i
                break
        assert mi != -1
        best_sample_ids = best_sample_ids[mi]
        best_indices_buckets = best_indices_buckets[mi]

        # print(f"{len(best_indices_buckets)=}, {len(best_sample_ids)=}")
        assert len(best_indices_buckets) == len(best_sample_ids)
        # print(f"{best_sample_ids=}, {len(best_indices_buckets)=}, {len(best_indices_buckets[0])=}, {len(best_indices_buckets[1])=}")
        # breakpoint()

        def transpose_2d_list(matrix):
            return [list(row) for row in zip(*matrix)]

        local_sample_id_groups = transpose_2d_list(best_sample_ids)
        local_best_indices_buckets = transpose_2d_list(best_indices_buckets)
        # groups = 
        min_hybrid_context_parallel_size = config.min_hybrid_context_parallel_size
        for microbatch_idx in range(len(local_sample_id_groups)):
            sample_id_groups.append([])
            groups.append([])
            cp_sizes.append([])
            dpxcp = len(local_sample_id_groups[microbatch_idx]) * min_hybrid_context_parallel_size
            for dp_rank in range(dpxcp):
                # for min_hybrid_context_parallel_rank in range(min_hybrid_context_parallel_size):
                sample_id_groups[microbatch_idx].append([])
                groups[microbatch_idx].append([])
                cp_sizes[microbatch_idx].append([])
                origin_dp_rank = dp_rank // min_hybrid_context_parallel_size
                # if torch.distributed.get_rank() == 0: print(f"{microbatch_idx=}, {dp_rank=}, {origin_dp_rank=}, {local_sample_id_groups[microbatch_idx][origin_dp_rank]=}")
                for local_sample_idx in local_sample_id_groups[microbatch_idx][origin_dp_rank]:
                    sample_id_groups[microbatch_idx][dp_rank].append(sample_id_seqlens[local_sample_idx][0])
                    groups[microbatch_idx][dp_rank].append(sample_id_seqlens[local_sample_idx][1])
                    final_cp_size = local_best_indices_buckets[microbatch_idx][origin_dp_rank].cp_size * min_hybrid_context_parallel_size
                    cp_sizes[microbatch_idx][dp_rank].append(final_cp_size)

        # if torch.distributed.get_rank() == 0: print(f"{sample_id_groups=}")
        # if torch.distributed.get_rank() == 0: print(f"{cp_sizes=}")
        # if torch.distributed.get_rank() == 0: print(f"rank={torch.distributed.get_rank()}, {groups=}")
        # if torch.distributed.get_rank() == 0: print(f"rank={torch.distributed.get_rank()}, {cp_sizes=}")
        def flatten(lst):
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result

        # 示例
        # nested_list = [1, [2, 3], [4, [5, 6]], 7]
        # print(flatten(nested_list))  # [1, 2, 3, 4, 5, 6, 7]

        # print(f"rank={torch.distributed.get_rank()}, {flatten(sample_id_groups)=}")
        # breakpoint()

        if return_cp_sizes:
            return groups, sample_id_groups, cp_sizes

        return groups, sample_id_groups
    def split_sample(
        self,
        num_buckets: List[int],
        avg_fwd_flops_with_m: List[float],
        all_lengths,
        all_flops,
        except_indexes,
        normal_indexes,
        combination,
        DP, PP, UP, TP,
        max_split_size,
        max_seq_len,
        config,
    ):
        num_layers = config.num_layers               # 模型层数
        hidden_size = config.hidden_size             # 隐藏层大小
        num_heads = config.num_attention_heads       # 注意力头数
        assert hidden_size % num_heads == 0, "hidden_size should be divisible by num_heads"
        head_dim = hidden_size // num_heads        # 每个注意力头的维度
        ffn_size = config.ffn_hidden_size           # FFN层隐藏大小

        # 初始化特殊序列的桶分配结构
        except_buckets = [[] for _ in range(len(num_buckets))]  # 每个m值对应的特殊序列桶
        except_bucket_num = 0                      # 特殊序列桶计数器
        except_bucket_m_per_sample = []            # 记录每个样本分配到的m值
        except_bucket_dp_per_sample = []           # 记录每个样本分配到的dp值
        except_bucket_num_per_sample = []          # 记录每个样本分割的桶数量

        # 计算每个 m 下单个 dp 的桶数(相同 m 的不同 dp 的桶数相等)
        bucket_num_per_dp_per_m = []
        # import pdb;pdb.set_trace()
        for i in range(len(num_buckets)):
            if combination[i] > 0:
                assert num_buckets[i] % combination[i] == 0, f"{i=}, {num_buckets[i]=}, {combination[i]=}"
                bucket_num_per_dp_per_m.append(num_buckets[i] // combination[i])
            else:
                bucket_num_per_dp_per_m.append(0)
        # print_rank0(f"{bucket_num_per_dp_per_m=}")

        # 维护不同 dp 当前剩余桶数，使用该桶数去做大 UP
        remain_buckets_num_per_dp_per_m = []
        for i in range(len(num_buckets)):
            if combination[i] > 0:
                assert num_buckets[i] % combination[i] == 0, f"{i=}, {num_buckets[i]=}, {combination[i]=}"
                # import pdb;pdb.set_trace()
                remain_buckets_num_per_dp_per_m.append([num_buckets[i] // combination[i]] * combination[i])
            else:
                remain_buckets_num_per_dp_per_m.append([])

        # 遍历所有需要独占一路 DP 的序列
        single_sample_indexes = []   # 去掉需要独占一个 DP 的样本后的 except_indexes
        combination_used = [0] * len(combination)

        # 重新计算 桶的容积
        sum_fwd_flops = sum([all_flops[1][idx] for idx in except_indexes if idx not in single_sample_indexes]) + \
            sum([all_flops[1][idx] for idx in normal_indexes])
            
        ratios = compute_ratios(combination, PP=PP)
        avg_fwd_flops_with_m_new = []
        total_num = sum([(combination[j]-combination_used[j]) * ratios[j] for j in range(len(combination))])  # TODO: total num need to - exceed_buckets num
        mean_fwd_flops_with_m = sum_fwd_flops / total_num
        for i in range(1, len(combination)+1):
            avg_fwd_flops_with_m_new.append(mean_fwd_flops_with_m * ratios[i - 1] / i / PP)

        avg_fwd_flops_with_m = avg_fwd_flops_with_m_new

        non_zero_combination = [(combination[idx]-combination_used[idx]) != 0 for idx in range(len(combination))]
        first_non_zero_m = 1 + non_zero_combination.index(True)
        threshold = 2 * sum_fwd_flops / (first_non_zero_m * (DP-len(single_sample_indexes)) * PP)

        consumed_num_buckets_backup = {}
        consumed_num_buckets_raw_backup = {}
        for idx, index in enumerate(except_indexes):
            find_bucket = False
            for i in range(len(num_buckets)):
                # 只考虑有剩余桶的m值
                if combination[i] > 0:
                    # 计算当前序列需要的桶数量(向上取整)
                    consumed_num_buckets_raw = math.ceil(all_flops[1][index] / avg_fwd_flops_with_m[i])
                    remain_num_split_sample = len(except_indexes) - 1 - idx
                    consumed_num_buckets = min(nearest_pow2(consumed_num_buckets_raw), max_split_size, DP//config.min_hybrid_context_parallel_size, num_buckets[i]-remain_num_split_sample)
                    consumed_num_buckets_raw_backup[index] = consumed_num_buckets_raw
                    consumed_num_buckets_backup[index] = consumed_num_buckets
                    # 更新剩余桶数量
                    num_buckets[i] -= consumed_num_buckets
                    find_bucket = True
                    break


        assign_all_sample_to_except_bucket_flag = False
        assert sum(num_buckets) >= 0
        if sum(num_buckets) == 0:
            assign_all_sample_to_except_bucket_flag = True
        # if torch.distributed.get_rank() == 0: print(f"{num_buckets=}\n{consumed_num_buckets_raw_backup.keys()=}\n{consumed_num_buckets_raw_backup.values()=}\n{consumed_num_buckets_backup.keys()=}\n{consumed_num_buckets_backup.values()=}\n{except_indexes=}")
        
        for index in except_indexes:
            find_bucket = False
            for i in range(len(num_buckets)):
                # 只考虑有剩余桶的m值
                if combination[i] > 0:
                    # 计算当前序列需要的桶数量(向上取整)
                    # consumed_num_buckets_raw = math.ceil(all_flops[1][index] / avg_fwd_flops_with_m[i])
                    # consumed_num_buckets = min(min(nearest_pow2(consumed_num_buckets_raw), max_split_size), DP)
                    consumed_num_buckets = consumed_num_buckets_backup[index]
                    # print(f"{index=}, {i=}, {consumed_num_buckets_raw=}, {consumed_num_buckets=}")
                    remained_flops = consumed_num_buckets * avg_fwd_flops_with_m[i] - all_flops[1][index]

                    # choose the CP interval
                    max_value = -1
                    max_left = max_right = -1
                    max_indexes = [-1] * consumed_num_buckets
                    for j in range(combination[i]):
                        left = (j // consumed_num_buckets) * consumed_num_buckets
                        right = (j // consumed_num_buckets + 1) * consumed_num_buckets
                        min_value_this_interval = 10000000
                        # for dp size not divisible by consumed_num_buckets, continue to skip this search space
                        if right > len(remain_buckets_num_per_dp_per_m[i]):
                            continue

                        for k in range(left, right): #left close right close
                            min_value_this_interval = min(min_value_this_interval, remain_buckets_num_per_dp_per_m[i][k])
                        if max_value < min_value_this_interval:
                            max_value = min_value_this_interval
                            max_left = left
                            max_right = right
                            max_indexes = list(range(left, right))
                    
                    normal_indexes_copy = copy.deepcopy(normal_indexes)
                    selected_indices, selected_flops, selected_lengths, remained_flops, remaining_sample_indices = fill_bucket_with_samples(index, normal_indexes, remained_flops, all_flops, all_lengths, max_seq_len, normal_indexes_copy, total_num, consumed_num_buckets, assign_all_sample_to_except_bucket_flag)
                    # print(f"\n{len(selected_indices)=}, {selected_indices=}\n{len(remaining_sample_indices)=}, {remaining_sample_indices=}\n{sum(selected_lengths)=}, {sum(selected_flops[0])=}, {remained_flops=}, {all_lengths[index]=}, {max_seq_len*consumed_num_buckets=}")
                    normal_indexes = remaining_sample_indices
                    for j in range(consumed_num_buckets):
                        remain_buckets_num_per_dp_per_m[i][max_indexes[j]] -= 1

                    assert len(max_indexes) == consumed_num_buckets, f"{len(max_indexes)=}, {consumed_num_buckets=}"
                    # 将分割后的序列片段分配到各个桶中
                    for j in range(consumed_num_buckets):
                        bucket_fwd_flops = all_flops[1][index] + sum(selected_flops[0])
                        bucket_bwd_flops = (3 * all_flops[1][index]) + sum(selected_flops[1])   # TODO(wuguohao): more precisely bwd_flops
                        bucket_length = all_lengths[index] + sum(selected_lengths)
                        bucket_tmp = [index] + selected_indices
                        #shenglong target_flops=1 to handle except use all buckets
                        except_buckets[i].append(
                            Bucket(
                                bucket_id=except_bucket_num,
                                samples=bucket_tmp,
                                cp_size=consumed_num_buckets,
                                fwd_flops=bucket_fwd_flops/consumed_num_buckets,
                                bwd_flops=bucket_bwd_flops/consumed_num_buckets,
                                seq_len_sum=bucket_length,
                                target_flops=1, target_density=0, target_length=0,
                                dp_index=max_indexes[j],
                            )
                        )
                        except_bucket_num += 1  # 递增桶计数器

                    # 更新剩余桶数量
                    # num_buckets[i] -= consumed_num_buckets
                    # 记录分配信息
                    except_bucket_num_per_sample.append(consumed_num_buckets)
                    except_bucket_m_per_sample.append(i)
                    except_bucket_dp_per_sample.append(max_indexes)

                    find_bucket = True
                    break  # 成功分配到桶中，跳出循环

            if not find_bucket:
                raise NotImplementedError("not found a bucket for the sample")
        
        assert len(except_bucket_m_per_sample) == len(except_bucket_num_per_sample), f"{len(except_bucket_m_per_sample)=}, {len(except_bucket_num_per_sample)=}"
        return except_buckets, num_buckets, except_bucket_num_per_sample, except_bucket_m_per_sample, except_bucket_dp_per_sample, except_indexes, normal_indexes, avg_fwd_flops_with_m

    def next_hdp_group(
        self,
        sample_seqlens: List[Tuple[int, int]],  # List of (sample_id, sequence_length) tuples
        compute_estimator: Callable[[int], float],
        total_gpus: int,
        delta: float = 0.05,  # balance slack (e.g. 5 %)
        strategy: str = "dp",  # "dp" or "pp"
        eps_bucket: float = 0.10,  # ε target for bucket balance
        config = None,
    ) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:

        DP = parallel_state.get_data_parallel_world_size()
        PP = parallel_state.get_pipeline_model_parallel_world_size()
        UP = parallel_state.get_context_parallel_world_size()
        TP = parallel_state.get_tensor_model_parallel_world_size()
        VPP = 1
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            VPP = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        # if torch.distributed.get_rank() == 0:
        #     breakpoint()
        # torch.distributed.barrier()

        max_split_size = 8 # TODO: change to configurable args.
        max_seq_len = config.max_seqlen_per_dp_cp_rank

        all_lengths = [sample_seqlens[i][1] for i in range(len(sample_seqlens))]
        
        all_flops = []
        all_tot_flops = []
        all_fwd_flops = []
        all_bwd_flops = []
        for idx in range(len(all_lengths)):
            length = all_lengths[idx]
            flops = TFLOPs(length, config)
            all_tot_flops.append(flops[0])
            all_fwd_flops.append(flops[1])
            all_bwd_flops.append(flops[2])
        all_flops.append(all_tot_flops)
        all_flops.append(all_fwd_flops)
        all_flops.append(all_bwd_flops)

        all_density = [all_flops[1][i] / all_lengths[i] for i in range(len(all_lengths))]
        best_max_seq_per_m = 0
        sum_fwd_flops = sum(all_flops[1])

        assert len(all_lengths) == len(all_flops[1])

        def dynamic_loops_product(limits):
            min_max_flops_sum_per_iter = sys.float_info.max / 10.0
            best_indices_buckets = []
            best_sample_ids = []
            best_dp_combination = []

            assert DP % config.min_hybrid_context_parallel_size == 0
            limit_item = DP // config.min_hybrid_context_parallel_size

            for idx, limit in enumerate(limits):
                combination = [0] * len(limits)
                combination[idx] = limit
                if sum(combination) != limit_item:
                    continue

                num_buckets = [PP * i * combination[i - 1] for i in range(1, len(combination)+1)]
                num_buckets_sum = sum(num_buckets)

                if num_buckets_sum > len(all_lengths):
                    print(f"continue due to num_buckets_sum {num_buckets_sum=} > len(all_lengths) {len(all_lengths)=}")
                    continue

                ratios = compute_ratios(combination, PP=PP)
                avg_fwd_flops_with_m = []
                total_num = sum(combination[j] * ratios[j] for j in range(len(combination)))
                mean_fwd_flops_with_m = sum_fwd_flops / total_num

                for i in range(1, len(combination)+1):
                    avg_fwd_flops_with_m.append(mean_fwd_flops_with_m * ratios[i - 1] / i / PP)

                indices_buckets, sample_ids, max_flops_sum_per_iter, max_seq_per_m, used_flops = \
                    self.solver(all_lengths, all_flops, all_density, num_buckets, avg_fwd_flops_with_m, combination, DP, PP, UP, TP, VPP, max_seq_len, max_split_size, config)
                
                if max_flops_sum_per_iter < min_max_flops_sum_per_iter:
                    min_max_flops_sum_per_iter = max_flops_sum_per_iter
                    best_indices_buckets = indices_buckets
                    best_sample_ids = sample_ids
                    best_dp_combination = combination
                    # if torch.distributed.get_rank() == 0:
                    #     print(f"{best_dp_combination=}\n{best_indices_buckets=}\n{best_sample_ids=}")

            return min_max_flops_sum_per_iter, best_indices_buckets, best_sample_ids, best_dp_combination

        search_space = 6
        assert DP % config.min_hybrid_context_parallel_size == 0
        limit_item = DP // config.min_hybrid_context_parallel_size
        limits = [limit_item] * search_space

        min_max_flops_sum_per_iter, best_indices_buckets, best_sample_ids, best_dp_combination = dynamic_loops_product(limits)
        if torch.distributed.get_rank() == 0: print(f"{best_dp_combination=}")
        # assert all DP have the same num_microbatch
        sum_best_dp_combination = sum(best_dp_combination)
        best_m = -1
        for idx, num_dp in enumerate(best_dp_combination):
            if num_dp == sum_best_dp_combination:
                best_m = idx
                break
        assert best_m != -1, f"{best_dp_combination=}"

        if not best_dp_combination:
            raise Exception()

        best_var_m = 0
        return min_max_flops_sum_per_iter, best_max_seq_per_m, best_indices_buckets, best_sample_ids, best_dp_combination, best_var_m

    def solver(
        self,
        all_lengths: List[int],
        all_flops: List[List[float]],
        all_density: List[float],
        num_buckets: List[int],
        avg_fwd_flops_with_m: List[float],
        combination,
        DP, PP, UP, TP, VPP,
        max_seq_len,
        max_split_size,
        config,
    ):
        except_indexes = []
        normal_indexes = []

        non_zero_combination = [combination[idx] != 0 for idx in range(len(combination))]
        first_non_zero_m = 1 + non_zero_combination.index(True)

        sum_fwd_flops = sum(all_flops[1])
        threshold = 1.3 * sum_fwd_flops / (first_non_zero_m * DP * PP)
        for idx in range(len(all_flops[1])):
            if  all_flops[1][idx] > threshold:
                except_indexes.append(idx)
            else:
                normal_indexes.append(idx)
        # if torch.distributed.get_rank() == 0:
        #     print(f"\n{except_indexes=}")
        #     except_flops = []
        #     for idx in except_indexes:
        #         except_flops.append(all_flops[1][idx])
        #     print(f"{except_indexes=}\n{except_flops=}")
        except_indexes = sorted(except_indexes, key=lambda x: all_flops[1][x], reverse=True)
        normal_indexes = sorted(normal_indexes, key=lambda x: all_flops[1][x], reverse=True)

        except_buckets, num_buckets, except_bucket_num_per_sample, except_bucket_m_per_sample, except_bucket_dp_per_sample, except_indexes, normal_indexes, avg_fwd_flops_with_m = \
            self.split_sample(num_buckets, avg_fwd_flops_with_m, all_lengths, all_flops, except_indexes, normal_indexes, combination, DP, PP, UP, TP, max_split_size, max_seq_len, config)

        sum_remained_flops = sum([all_flops[1][index] for index in normal_indexes])

        # for the case that except indexes take all buckets
        if sum(num_buckets) != 0:
            max_seq_len_for_fuse = sum([all_lengths[idx] for idx in normal_indexes]) / sum(num_buckets)
        else:
            max_seq_len_for_fuse = 0
        

        if max_seq_len_for_fuse == 0:
            assert len(normal_indexes) == 0
        total_bucket_num, all_buckets = create_buckets(num_buckets, avg_fwd_flops_with_m, max_seq_len_for_fuse)
        sorted_indices_fwdflops = sorted(normal_indexes, key=lambda x: all_flops[1][x], reverse=True)
        sorted_all_buckets_fwd_flops = sorted(all_buckets, key=lambda bucket: bucket.fwd_flops)
        all_sample_index_copy = copy.deepcopy(sorted_indices_fwdflops)

        all_sample_index_copy_bef_flops = copy.deepcopy(all_sample_index_copy)
        all_sample_index_copy = assign_samples_to_buckets(sorted_indices_fwdflops,
                                    sorted_all_buckets_fwd_flops,
                                    all_density,
                                    all_lengths,
                                    all_flops,
                                    update_rule=UpdateRule.FW_FLOPS,
                                    remaining_sample_indices=all_sample_index_copy)
    
        # If there are some leftover of the samples 
        # (e.g. if put the sample in any of the bucket will cause the bucket exceed the memory limit),
        # we will use the length update rule to assign those samples to the bucket.
        # The all_sample_index_copy should contain only a few samples. Sorting might be unnecessary.
        sorted_indices_length = sorted(all_sample_index_copy, key=lambda x: all_lengths[x], reverse=True)
        sorted_all_buckets_length = sorted(all_buckets, key=lambda bucket: bucket.seq_len_sum)

        if len(all_sample_index_copy) > 0:
            all_sample_index_copy_bef_len = copy.deepcopy(all_sample_index_copy)
            all_sample_index_copy = assign_samples_to_buckets(sorted_indices_length, sorted_all_buckets_length, all_density, all_lengths, all_flops, update_rule=UpdateRule.LENGTH, remaining_sample_indices=all_sample_index_copy, print_score=True)

        assert len(all_sample_index_copy) == 0, f"sample {all_sample_index_copy} is not assigned to any bucket."

        indices_buckets = [[] for _ in range(total_bucket_num)]
        used_flops = [0.0] * total_bucket_num
        used_fwd_flops = [0.0] * total_bucket_num
        used_bwd_flops = [0.0] * total_bucket_num
        max_seq_per_m = 0
        seq_per_m = []
        
        for bucket in sorted_all_buckets_fwd_flops:
            bucket_id = bucket.bucket_id 
            indices_buckets[bucket_id] = bucket
            used_flops[bucket_id] = bucket.fwd_flops + bucket.bwd_flops
            used_fwd_flops[bucket_id] = bucket.fwd_flops
            used_bwd_flops[bucket_id] = bucket.bwd_flops
            max_seq_per_m = max(bucket.seq_len_sum, max_seq_per_m)
            seq_per_m.append(bucket.seq_len_sum)

        indices_buckets_2d = [[] for _ in range(len(num_buckets))]
        sample_ids_2d = [[] for _ in range(len(num_buckets))]
        new_cnt = 0
        max_sum_per_iter = 0.0
        rets = [0.0] * DP
        thread_cnt = 0

        max_iter_sum_among_dp_list = []
        for i in range(len(num_buckets)):
            if len(except_buckets[i]) + num_buckets[i] == 0:
                assert combination[i] == 0, f"{combination=}, {num_buckets=}, {len(except_buckets[i])=}"
                continue

            total_buckets_for_current_m = num_buckets[i] + len(except_buckets[i])
            num_m = i + 1
            bucket_num_per_dp_curr_m = num_m * PP
            assert total_buckets_for_current_m % bucket_num_per_dp_curr_m == 0, f"{total_buckets_for_current_m=}, {bucket_num_per_dp_curr_m=}"
            dp_size_for_current_m = total_buckets_for_current_m // bucket_num_per_dp_curr_m

            buckets_for_current_m = []
            for j in range(num_buckets[i]):
                buckets_for_current_m.append([used_flops[new_cnt], new_cnt, used_fwd_flops[new_cnt]])
                new_cnt += 1

            buckets_for_current_m.sort(key=lambda x: x[2])

            fwd_flops_for_dp_per_m, bwd_flops_for_dp_per_m, buckets_for_dp, sample_ids_for_dp, seq_len_for_dp_per_m, empty_bucket_flag = greedy_assign_bucket_to_dp(i, indices_buckets, normal_indexes, except_buckets, except_bucket_num_per_sample, except_bucket_m_per_sample, except_bucket_dp_per_sample, buckets_for_current_m, dp_size_for_current_m, used_flops, used_fwd_flops, used_bwd_flops, bucket_num_per_dp_curr_m, all_flops, all_lengths, combination, config)

            for j in range(len(buckets_for_dp)):
                indices_buckets_2d[i].append(buckets_for_dp[j])
                sample_ids_2d[i].append(sample_ids_for_dp[j])
            
            assert len(indices_buckets_2d) == len(sample_ids_2d), f"{len(indices_buckets_2d)=}, {len(sample_ids_2d)=}"

            bubble_time_list = []
            if empty_bucket_flag:
                print(f"error, found empty bucket, skip")
                max_sum_per_iter = sys.float_info.max / 10.0
            else:
                for m in range(len(fwd_flops_for_dp_per_m)):
                    total_bucket_num_for_current_dp = len(fwd_flops_for_dp_per_m[m])
                    forward_cost = [fwd_flops_for_dp_per_m[m][k][0] for k in range(len(fwd_flops_for_dp_per_m[m]))]
                    backward_cost = [bwd_flops_for_dp_per_m[m][k][0] for k in range(len(fwd_flops_for_dp_per_m[m]))]
                    seq_len_for_dp = seq_len_for_dp_per_m[m]
                    communication_cost = [0.0] * len(fwd_flops_for_dp_per_m[m])

                    forward_cost_cmp = []
                    backward_cost_cmp = []
                    assert len(fwd_flops_for_dp_per_m[m]) == len(bwd_flops_for_dp_per_m[m])
                    for k in range(len(fwd_flops_for_dp_per_m[m])):
                        split_num = fwd_flops_for_dp_per_m[m][k][2]
                        split_idx = fwd_flops_for_dp_per_m[m][k][3]
                        fwd_cost = fwd_flops_for_dp_per_m[m][k][0]
                        bwd_cost = bwd_flops_for_dp_per_m[m][k][0]

                        forward_cost_cmp.append([fwd_cost])
                        backward_cost_cmp.append([bwd_cost])

                    max_iter_sum_among_dp = simulate_time(forward_cost_cmp, backward_cost_cmp, PP, VPP)
                    
                    max_iter_sum_among_dp_list.append(max_iter_sum_among_dp)
                    max_sum_per_iter = max(max_sum_per_iter, max_iter_sum_among_dp)

                    peak_memory = simulate_memory(seq_len_for_dp, config)

                    forward_cost_cmp = torch.tensor(forward_cost_cmp).flatten().tolist()
                    backward_cost_cmp = torch.tensor(backward_cost_cmp).flatten().tolist()

                    fwd_cost_total = sum(forward_cost_cmp)
                    bwd_cost_total = sum(backward_cost_cmp)

                    fwd_bwd_cost_total = fwd_cost_total + bwd_cost_total
                    num_microbatch = (i+1) * PP
                    pp_bubble_ratio = compute_pp_bubble_ratio(PP, num_microbatch, VPP)

                    pp_bubble_time = fwd_bwd_cost_total / (1 - pp_bubble_ratio) - fwd_bwd_cost_total
                    bubble_idle_time = max_iter_sum_among_dp - fwd_bwd_cost_total
                    imbalanced_bubble_time = bubble_idle_time - pp_bubble_time

                    bubble_over_iter_time = bubble_idle_time / max_iter_sum_among_dp
                    bubble_over_compute_time = bubble_idle_time / fwd_bwd_cost_total

                    pp_bubble_over_iter_time = pp_bubble_time / max_iter_sum_among_dp
                    pp_bubble_over_compute_time = pp_bubble_time / fwd_bwd_cost_total

                    imbalanced_bubble_over_iter_time = imbalanced_bubble_time / max_iter_sum_among_dp
                    imbalanced_bubble_over_compute_time = imbalanced_bubble_time / fwd_bwd_cost_total

                    bubble_time_list.append({
                        "pp_bubble_ratio": pp_bubble_ratio,
                        "bubble_over_compute_time":bubble_over_compute_time,
                        "pp_bubble_over_compute_time":pp_bubble_over_compute_time,
                        "imbalanced_bubble_over_compute_time":imbalanced_bubble_over_compute_time,
                    })

                    # if peak_memory >= 70 * 1024**3:
                    #     max_sum_per_iter = sys.float_info.max / 10.0    # skip this m
                    #     print(f"rank={torch.distributed.get_rank()}, Peak memory usage: {peak_memory / 1024**3:.2f} GiB, {combination=}")

                # if torch.distributed.get_rank() == 0:
                #     for k in range(len(bubble_time_list)):
                #         for key in bubble_time_list[k].keys():
                #             bubble_time_list[k][key] = round(bubble_time_list[k][key], 3)
                #         print(f"{k=}, {bubble_time_list[k]}")

        max_max_iter_sum = max(max_iter_sum_among_dp_list)
        min_max_iter_sum = min(max_iter_sum_among_dp_list)
        sum_max_iter_sum = sum(max_iter_sum_among_dp_list)
        len_max_iter_sum = len(max_iter_sum_among_dp_list)
        mean_max_iter_sum = sum_max_iter_sum/len_max_iter_sum

        # print(f"{sample_ids_2d=}")

        return indices_buckets_2d, sample_ids_2d, max_sum_per_iter, max_seq_per_m, used_flops
            
class OnlyPackingNoSchedulingScheduler(BaseScheduler):
    """
    This scheduler only packs sequences in their original order
    and does not perform any load balancing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.dp_size = int(parallel_state.get_data_parallel_world_size())
        self.cp_size = int(parallel_state.get_context_parallel_world_size())
        self.max_seq_len_all_ranks = config.max_seqlen_per_dp_cp_rank * self.cp_size
