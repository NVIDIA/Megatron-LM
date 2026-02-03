# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

import enum
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.hybrid_cp_schedule import BalancedCPScheduler
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
        """
        Gathers the sequence lengths of all subsamples from all DP ranks.
        Each DP rank loads the same number of microbatches but each microbatch
        may have a different number of subsamples.

        We find the number of subsamples each rank holds and then gather the
        sequence lengths of all subsamples from all ranks.
        """
        # Collect the number of subsamples from all ranks
        local_len = torch.tensor([subsample_seqlens.shape[0]], dtype=torch.int32).cuda()
        dp_subsample_count = [torch.zeros_like(local_len) for _ in range(self.dp_group.size())]
        torch.distributed.all_gather(dp_subsample_count, local_len, group=self.dp_group)

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
            torch.empty_like(subsample_seqlens_padded) for _ in range(self.dp_group.size())
        ]
        torch.distributed.all_gather(
            seqlens_gathered, subsample_seqlens_padded, group=self.dp_group
        )

        # Trim each seqlens_gathered to the length of the correct sample
        for dp_rank, seqlen in enumerate(seqlens_gathered):
            seqlens_gathered[dp_rank] = seqlen[: dp_subsample_counts[dp_rank]]

        seqlens_gathered = torch.cat(seqlens_gathered, dim=0)
        seqlens_gathered = seqlens_gathered.cpu().tolist()

        # Calculate the offsets to assign unique global ID to each subsample.
        csum = torch.cumsum(dp_subsample_counts, dim=0, dtype=torch.int32)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum[:-1]], dim=0)

        return seqlens_gathered, offsets

    def get_global_id_seqlens(self, num_local_subsamples, offsets, seqlens_gathered):
        """
        Calculates the global ID for each subsample.

        We assign a unique global ID to each subsample.

        Returns:
        global_id_seqlens: list of (global_id, seqlen) tuples for scheduling.
        global_ids_this_rank: list of global IDs locally present on this rank.
        """
        dp_rank = self.dp_group.rank()
        global_ids = torch.arange(len(seqlens_gathered), dtype=torch.int32).cuda()
        # Create a list of (global_id, seqlen) tuples for scheduling
        global_id_seqlens = [(i, seqlens_gathered[i]) for i in range(len(global_ids))]
        # Get the global IDs locally present on this rank
        global_ids_this_rank = global_ids[
            offsets[dp_rank] : offsets[dp_rank] + num_local_subsamples
        ]

        return global_id_seqlens, global_ids_this_rank

    def _gid_to_src_rank(self, gid: int, offsets: List[int]) -> int:
        dp_src_rank = torch.bucketize(gid, offsets[1:] - 1)
        # Since the torch.distributed.get_process_group_ranks
        # provides the global rank, we need to consider TP
        hdp_rank = (
            torch.distributed.get_process_group_ranks(self.dp_group)[dp_src_rank]
            // self.tp_group.size()
        )
        return hdp_rank

    def reroute_samples_to_hdp_ranks(
        self, batch, global_ids_this_rank, global_id_seqlens, sample_id_groups, offsets
    ):
        """
        Reroutes the sub-samples to the correct rank after scheduling.

        For each key in the batch dict, we perform an all-to-all communication
        to transfer the data to the correct ranks.
        Since all CP ranks within a DP group have the same data, we only need
        to transfer data between matching CP ranks.
        """
        gid2local_id = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
        hdp_rank = self.dp_cp_group.rank()
        dp_ranks = torch.distributed.get_process_group_ranks(self.dp_group)
        # Here we actually want to get the DP group's rank within the HDP group,
        # we need to consider TP
        dp_ranks = [r // self.tp_group.size() for r in dp_ranks]

        data_keys = batch[0].keys()

        # Create the send plan
        combined_sample_id_groups: List[List[int]] = [[] for _ in range(self.total_hdp_gpus)]

        for d in range(self.total_hdp_gpus):
            for sample_id_group in sample_id_groups:
                combined_sample_id_groups[d].extend(sample_id_group[d])

        for dest_rank in range(self.total_hdp_gpus):
            combined_sample_id_groups[dest_rank].sort()

        # Filter out samples that are not present on this rank
        send_ids_sorted = [
            gid
            for d in dp_ranks
            for gid in combined_sample_id_groups[d]
            if gid in global_ids_this_rank
        ]
        # send_counts = [len(combined_sample_id_groups[d]) for d in range(self.total_hdp_gpus)]

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
                # We only need to share local data with DP ranks that have different data.
                send_lens_split[dest_rank] = 0

        # Create the recv plan
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
        """
        Unpacks the packed samples into a list of sub-samples.
        Since each sub-sample may be routed to different DPxCP ranks,
        we unpack the sample here to avoid unnecessarily transferring
        the entire packed sample.
        """
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
        """
        Get the next item from the dataset, pull scheduling metadata and return it.
        """
        if self.data_iterator is None:
            # TP0 reads from data_iterator, others receive via broadcast.
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


class PackingScheduler(enum.Enum):
    """Enum for supported sequence packing algorithms."""

    DEFAULT_SEQUENCE_PACKING = "default_sequence_packing"


def _broadcast_tensor(item, src_rank, group) -> None:
    """Broadcast a tensor from src_rank to all ranks in the group."""
    if item is not None:
        torch.distributed.broadcast(item, src_rank, group=group)


class BaseScheduler:
    """Base class for sequence packing schedulers."""

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank: int,
        cp_size: int,
        dp_size: int,
        microbatch_group_size_per_vp_stage: Optional[int],
    ):
        self.max_seqlen_per_dp_cp_rank = max_seqlen_per_dp_cp_rank
        self.cp_size = cp_size
        self.dp_size = dp_size
        self.microbatch_group_size_per_vp_stage = microbatch_group_size_per_vp_stage

    def get_require_sample_keys(self):
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
        """Run the scheduler and return the new data_iterator."""
        raise NotImplementedError

    @staticmethod
    def _get_global_seqlens(subsample_seqlens: torch.Tensor, dp_group) -> List[int]:
        """
        Gathers the sequence lengths of all subsamples from all DP ranks.

        Each DP rank has the same number of subsamples (num_microbatches),
        so we can directly all_gather without padding.
        """
        dp_size = dp_group.size()
        num_local_subsamples = subsample_seqlens.shape[0]

        # Gather the subsample_seqlens from all ranks
        seqlens_gathered = [torch.empty_like(subsample_seqlens) for _ in range(dp_size)]
        torch.distributed.all_gather(seqlens_gathered, subsample_seqlens, group=dp_group)

        seqlens_gathered = torch.cat(seqlens_gathered, dim=0)
        seqlens_gathered = seqlens_gathered.cpu().tolist()

        # Calculate the offsets to assign unique global ID to each subsample.
        # Since each rank has the same number of subsamples, offsets are evenly spaced.
        offsets = torch.arange(
            0, dp_size * num_local_subsamples, num_local_subsamples, dtype=torch.int32
        )

        return seqlens_gathered, offsets

    @staticmethod
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

    @staticmethod
    def _broadcast_to_pp_group(
        new_samples,
        num_micro_batches,
        seqlen_sum_this_global_batch,
        seqlen_squared_sum_this_global_batch,
        pp_group,
        dev,
    ):
        """
        Broadcast num_micro_batches, seqlen_sum_this_global_batch,
        seqlen_squared_sum_this_global_batch and metadata to middle PP stages.
        """

        pp_src_rank = torch.distributed.get_process_group_ranks(pp_group)[0]

        if pp_group.size() > 2:
            if pp_group.rank() == 0:
                tensor_list = [
                    torch.tensor(
                        [
                            num_micro_batches,
                            seqlen_sum_this_global_batch,
                            seqlen_squared_sum_this_global_batch,
                        ],
                        dtype=torch.float32,
                    ).cuda()
                ]
                for sample in new_samples:
                    tensor_list.append(sample["max_seqlen"].unsqueeze(0))
                for sample in new_samples:
                    tensor_list.append(sample["cu_seqlens"])
                    tensor_list.append(sample["cu_seqlens_padded"])
                info_to_broadcast = torch.cat(tensor_list, dim=0).to(
                    device=dev, dtype=torch.float32
                )
                info_length_tensor = torch.tensor(
                    info_to_broadcast.shape[0], dtype=torch.int32
                ).cuda()
                _broadcast_tensor(info_length_tensor, pp_src_rank, pp_group)
                _broadcast_tensor(info_to_broadcast, pp_src_rank, pp_group)
            else:
                info_length_tensor = torch.tensor(0, dtype=torch.int32).cuda()
                _broadcast_tensor(info_length_tensor, pp_src_rank, pp_group)
                info_to_broadcast = torch.empty(
                    info_length_tensor.item(), dtype=torch.float32
                ).cuda()
                _broadcast_tensor(info_to_broadcast, pp_src_rank, pp_group)
                if pp_group.rank() != pp_group.size() - 1:
                    # middle PP stages receive the broadcasted info and unpack it
                    info_numpy = info_to_broadcast.cpu().numpy()
                    num_micro_batches = int(info_numpy[0])
                    seqlen_sum_this_global_batch = info_numpy[1]
                    seqlen_squared_sum_this_global_batch = info_numpy[2]
                    max_seqlens = info_to_broadcast[3 : 3 + num_micro_batches]
                    cu_seqlens_list = []
                    cu_seqlens_padded_list = []
                    indices = np.where(info_numpy == 0)[0]
                    for i in range(num_micro_batches):
                        cu_seqlens_list.append(
                            info_to_broadcast[indices[i * 2] : indices[i * 2 + 1]]
                        )
                        if i == num_micro_batches - 1:
                            cu_seqlens_padded_list.append(info_to_broadcast[indices[i * 2 + 1] :])
                        else:
                            cu_seqlens_padded_list.append(
                                info_to_broadcast[indices[i * 2 + 1] : indices[i * 2 + 2]]
                            )

                    new_samples = []
                    for i in range(num_micro_batches):
                        new_sample = {}
                        new_sample["max_seqlen"] = max_seqlens[i].to(torch.int32)
                        new_sample["cu_seqlens"] = cu_seqlens_list[i].to(torch.int32)
                        new_sample["cu_seqlens_padded"] = cu_seqlens_padded_list[i].to(torch.int32)
                        new_samples.append(new_sample)

        return (
            new_samples,
            num_micro_batches,
            seqlen_sum_this_global_batch,
            seqlen_squared_sum_this_global_batch,
        )

    @staticmethod
    def _broadcast_scalars(values: List, group, dev, dtype=torch.float32) -> List:
        """
        Broadcast scalar values from rank 0 to all ranks in the group.

        Args:
            values: List of scalar values to broadcast (only used on rank 0).
            group: The process group to broadcast within.
            dev: The device to use for the tensor.
            dtype: The data type for the tensor.

        Returns:
            List of broadcasted values.
        """
        if group.size() <= 1:
            return values

        src_rank = torch.distributed.get_process_group_ranks(group)[0]
        num_values = len(values)

        if group.rank() == 0:
            info_to_broadcast = torch.tensor(values, dtype=dtype, device=dev)
        else:
            info_to_broadcast = torch.zeros(num_values, dtype=dtype, device=dev)

        _broadcast_tensor(info_to_broadcast, src_rank, group)

        if group.rank() != 0:
            values = info_to_broadcast.cpu().tolist()

        return values

    @staticmethod
    def _create_data_iterator(new_samples, pp_group, tp_group, config):
        """Handle virtual pipeline parallelism."""
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
                        new_sample_for_other_ppstage["cu_seqlens_padded"] = sample[
                            "cu_seqlens_padded"
                        ]
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
                    new_data_iterator = [
                        RerunDataIterator(iter(new_samples)) for _ in range(vpp_size)
                    ]
            else:
                new_data_iterator = [None for _ in range(vpp_size)]
        else:
            new_data_iterator = (
                RerunDataIterator(iter(new_samples)) if tp_group.rank() == 0 else None
            )

        return new_data_iterator

    @staticmethod
    def _reroute_samples_to_dcp_ranks(
        batch,
        global_ids_this_rank,
        global_id_seqlens,
        sample_id_groups,
        offsets,
        dp_group,
        tp_group,
        dp_cp_group,
        total_dcp_gpus,
    ):
        """
        Reroutes the sub-samples to the correct rank after scheduling.

        For each key in the batch dict, we perform an all-to-all communication
        to transfer the data to the correct ranks.
        """

        def _gid_to_src_rank(gid: int) -> int:
            dp_src_rank = torch.bucketize(gid, offsets[1:] - 1)
            dcp_rank = (
                torch.distributed.get_process_group_ranks(dp_group)[dp_src_rank] // tp_group.size()
            ) % dp_cp_group.size()
            return dcp_rank

        gid2local_id = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
        dcp_rank = dp_cp_group.rank()
        dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
        dp_ranks = [(r // tp_group.size()) % dp_cp_group.size() for r in dp_ranks]

        data_keys = batch[0].keys()

        # Create the send plan
        combined_sample_id_groups: List[List[int]] = [[] for _ in range(total_dcp_gpus)]
        for d in range(total_dcp_gpus):
            for sample_id_group in sample_id_groups:
                combined_sample_id_groups[d].extend(sample_id_group[d])
        for dest_rank in range(total_dcp_gpus):
            combined_sample_id_groups[dest_rank].sort()

        send_ids_sorted = [
            gid
            for d in dp_ranks
            for gid in combined_sample_id_groups[d]
            if gid in global_ids_this_rank
        ]

        send_num_split = [0] * total_dcp_gpus
        send_lens_split = [0] * total_dcp_gpus
        for dest_rank in range(total_dcp_gpus):
            if dest_rank in dp_ranks:
                send_seq_lens = [
                    global_id_seqlens[gid][1]
                    for gid in combined_sample_id_groups[dest_rank]
                    if gid in global_ids_this_rank
                ]
                send_num_split[dest_rank] = len(send_seq_lens)
                send_lens_split[dest_rank] = sum(send_seq_lens)
            else:
                send_lens_split[dest_rank] = 0

        # Create the recv plan
        recv_sample_id_groups = [[] for _ in range(total_dcp_gpus)]
        for gid in combined_sample_id_groups[dcp_rank]:
            src_rank = _gid_to_src_rank(gid)
            recv_sample_id_groups[src_rank].append(gid)

        recv_lens_split = [0] * total_dcp_gpus
        for src_rank in range(total_dcp_gpus):
            recv_lens_split[src_rank] = sum(
                [global_id_seqlens[gid][1] for gid in recv_sample_id_groups[src_rank]]
            )

        recv_ids_sorted = [gid for d in range(total_dcp_gpus) for gid in recv_sample_id_groups[d]]
        recv_counts = [len(recv_sample_id_groups[d]) for d in range(total_dcp_gpus)]

        recv_samples = [{k: None for k in data_keys} for _ in range(sum(recv_counts))]

        def _pack_sample_by_key(key: str) -> torch.Tensor:
            flattened_tensors = []
            for gid in send_ids_sorted:
                t = batch[gid2local_id[gid]][key].to(torch.cuda.current_device(), non_blocking=True)
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

    @staticmethod
    def _pack_sequences(
        samples: List,
        padded_lengths: torch.Tensor,
        original_lengths: torch.Tensor,
        dev: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Pack multiple samples into a single packed sample."""

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

        padded_lengths = padded_lengths.to(
            device=dev, dtype=torch.int32, non_blocking=True
        ).reshape(-1)
        cu_seqlens_padded = torch.empty(padded_lengths.numel() + 1, device=dev, dtype=torch.int32)
        cu_seqlens_padded[0] = 0
        cu_seqlens_padded[1:] = torch.cumsum(padded_lengths, dim=0)
        max_seqlen = torch.max(padded_lengths).to(dtype=torch.int32)

        new_sample["cu_seqlens_padded"] = cu_seqlens_padded
        new_sample["max_seqlen"] = max_seqlen

        original_lengths = original_lengths.to(
            device=dev, dtype=torch.int32, non_blocking=True
        ).reshape(-1)
        cu_seqlens = torch.empty(original_lengths.numel() + 1, device=dev, dtype=torch.int32)
        cu_seqlens[0] = 0
        cu_seqlens[1:] = torch.cumsum(original_lengths, dim=0).reshape(-1)
        new_sample["cu_seqlens"] = cu_seqlens

        return new_sample

    @staticmethod
    def _build_packed_microbatches(
        grouped_samples: List[List[Dict[str, torch.Tensor]]], dev: torch.device
    ) -> List[Dict[str, torch.Tensor]]:
        """Build packed samples for each microbatch."""
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
            new_sample = BaseScheduler._pack_sequences(samples, lp, lo, dev)
            new_samples.append(new_sample)

        return new_samples

    @staticmethod
    def _get_batch_and_global_seqlens(data_iterator, num_microbatches, dp_group):
        """
        Get the batch and global sequence lengths.
        Each DP rank loads the same number of sequences, so we need to gather the sequence
        lengths from all ranks then we can schedule the sequences into groups.
        Args:
            data_iterator: The data iterator.
            num_microbatches: The number of microbatches.
            dp_group: The data parallel group.

        Returns:
            batch: The batch.
            global_id_seqlens: The global sequence lengths.
            global_ids_this_rank: The global IDs locally present on this rank.
        """
        batch = [next(data_iterator) for _ in range(num_microbatches)]
        subsample_seqlens = []
        for sample in batch:
            subsample_seqlens.extend([sample["tokens"].numel()])
        subsample_seqlens = torch.tensor(subsample_seqlens, dtype=torch.int32).cuda()

        seqlens_gathered, offsets = BaseScheduler._get_global_seqlens(subsample_seqlens, dp_group)

        global_id_seqlens, global_ids_this_rank = BaseScheduler._get_global_id_seqlens(
            subsample_seqlens.shape[0], offsets, seqlens_gathered, dp_group
        )

        return batch, global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered


class DefaultSequencePackingScheduler(BaseScheduler):
    """Packs sequences in their original order until reaching the max limit of sequence length."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_len_all_ranks = self.max_seqlen_per_dp_cp_rank * self.cp_size

    def get_require_sample_keys(self):
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

        # Handle VPP: extract the correct data_iterator for this PP stage
        if (
            config.virtual_pipeline_model_parallel_size is not None
            and config.virtual_pipeline_model_parallel_size > 1
        ):
            # if enable VPP, data_iterator is a list of data_iterators for each VPP stage,
            # and only the first and last stage rank will have data_iterator,
            # other stages will have None.
            assert len(data_iterator) == config.virtual_pipeline_model_parallel_size
            if pp_group.rank() == 0:
                # the first stage
                data_iterator = data_iterator[0]
            elif pp_group.rank() == pp_group.size() - 1:
                # the last stage
                data_iterator = data_iterator[-1]
            else:
                data_iterator = None

        # data_iterator is not None when TP rank 0, with PP stage 0 or -1.
        if data_iterator is not None:
            assert tp_group.rank() == 0 and (
                pp_group.rank() == 0 or pp_group.rank() == pp_group.size() - 1
            ), f"Only TP rank 0 and PP stage 0 or -1 should have data_iterator"

            # Step 1: Fetch batches and gather global sequence lengths
            batch, global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered = (
                self._get_batch_and_global_seqlens(data_iterator, num_microbatches, dp_group)
            )

            # Step 2: Check required sample keys
            for key in self.get_require_sample_keys():
                assert key in batch[0], f"Batch missing required key {key}"

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
            samples_this_rank_with_id = self._reroute_samples_to_dcp_ranks(
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

            grouped_samples = [
                [
                    samples_this_rank_with_id[sub_sample_id]
                    for sub_sample_id in sample_id_groups[i][dcp_rank]
                ]
                for i in range(num_micro_batches)
            ]

            # Step 5: Build packed microbatches
            new_samples = self._build_packed_microbatches(grouped_samples, dev)

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
            ) = self._broadcast_to_pp_group(
                new_samples,
                num_micro_batches,
                seqlen_sum_this_global_batch,
                seqlen_squared_sum_this_global_batch,
                pp_group,
                dev,
            )

        # Step 8: Broadcast to TP group (for non-TP-0 ranks)
        (num_micro_batches, seqlen_sum_this_global_batch, seqlen_squared_sum_this_global_batch) = (
            self._broadcast_scalars(
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
        new_data_iterator = self._create_data_iterator(new_samples, pp_group, tp_group, config)

        return (
            new_data_iterator,
            num_micro_batches,
            seqlen_sum_this_global_batch,
            seqlen_squared_sum_this_global_batch,
        )


def wrap_dataloader(
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

    scheduler_map: Dict[PackingScheduler, Type[BaseScheduler]] = {
        PackingScheduler.DEFAULT_SEQUENCE_PACKING: DefaultSequencePackingScheduler
    }

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
    scheduler_type = PackingScheduler[scheduler_type.upper()]

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
    data_iterator, mtp_on_this_rank: bool = False, vp_stage: Optional[int] = None
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

    tp_src_rank = parallel_state.get_tensor_model_parallel_src_rank()
    tp_group = parallel_state.get_tensor_model_parallel_group()

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
    batch_keys = ['cu_seqlens', 'cu_seqlens_padded', 'max_seqlen']
    if is_first_stage:
        batch_keys.append('tokens')
        batch_keys.append('position_ids')
    if is_last_stage:
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

    # Partition tokens, position_ids, labels, loss_mask for context parallel, currently only
    # TP rank 0 and the first/last PP stage rank has these data.
    if is_tp_rank_0 and is_first_or_last_stage:
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
    _broadcast_tensor(cu_seqlen_size, tp_src_rank, tp_group)
    cu_seqlen_size = cu_seqlen_size.item()

    # Broadcast total_tokens because we need it to create placeholder for tokens, position_ids,
    # labels, loss_mask for non TP 0 ranks. Only first or last stage need this.
    if is_first_or_last_stage:
        if is_tp_rank_0:
            total_tokens = torch.tensor(batch['tokens'].size(0), dtype=torch.int32, device=dev)
        else:
            total_tokens = torch.empty(1, dtype=torch.int32, device=dev)
        _broadcast_tensor(total_tokens, tp_src_rank, tp_group)
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

    # Broadcast batch inside TP group.
    _broadcast_tensor(batch['tokens'], tp_src_rank, tp_group)
    _broadcast_tensor(batch['position_ids'], tp_src_rank, tp_group)
    _broadcast_tensor(batch['labels'], tp_src_rank, tp_group)
    _broadcast_tensor(batch['loss_mask'], tp_src_rank, tp_group)
    _broadcast_tensor(batch['cu_seqlens'], tp_src_rank, tp_group)
    _broadcast_tensor(batch['cu_seqlens_padded'], tp_src_rank, tp_group)
    _broadcast_tensor(batch['max_seqlen'], tp_src_rank, tp_group)

    # Extract the data from batch after broadcasting.
    tokens = batch['tokens']
    position_ids = batch['position_ids']
    labels = batch['labels']
    loss_mask = batch['loss_mask']
    cu_seqlens = batch['cu_seqlens']
    cu_seqlens_padded = batch['cu_seqlens_padded']
    max_seqlen = batch['max_seqlen'].item()

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
        local_cp_size=None,
        cp_group=None,
    )

    # "attention_mask" is not valid for sequence packing, so set it to None.
    return tokens, labels, loss_mask, None, position_ids, packed_seq_params
