# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

from typing import Any, List, Optional

import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule_utils import _get_global_seqlens_and_ids
from megatron.core.pipeline_parallel.hybrid_cp_schedule import BalancedCPScheduler
from megatron.core.process_groups_config import ProcessGroupCollection


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

        Delegates to ``data_schedule_utils._get_global_seqlens_and_ids``. The
        shared helper returns ``offsets`` with one extra trailing entry (the
        total subsample count); this method preserves the original contract of
        returning per-rank start offsets only.
        """
        _, _, offsets, seqlens_gathered = _get_global_seqlens_and_ids(
            subsample_seqlens, self.dp_group
        )
        return seqlens_gathered, offsets[:-1]

    def get_global_id_seqlens(self, num_local_subsamples, offsets, seqlens_gathered):
        """
        Calculates the global ID for each subsample.

        We assign a unique global ID to each subsample.

        Kept as a local implementation: the shared helper fuses this pure
        indexing step with the collective gather, so delegating here would
        re-run the all-gathers.

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

    # NOTE: data_schedule_utils.reroute_samples_to_dcp_ranks is the sequence-
    # packing generalization of this method. It is intentionally NOT used here
    # because it is not behavior-preserving for the hybrid-CP path:
    #   * its empty-send fallback allocates a 1-element (not 0-element) buffer,
    #     which fails all_to_all_single split validation when a rank has
    #     nothing to send (reachable here);
    #   * it transports the "original_seq_len"/"padded_seq_len" metadata keys
    #     with count-based splits, keys the hybrid-CP batch does not carry;
    #   * its rank mapping applies "% dp_cp_group.size()" for PP support,
    #     while hybrid CP asserts pipeline_model_parallel_size == 1.
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

    # NOTE: data_schedule_utils._unpack_batch is the sequence-packing variant
    # of this method. It is intentionally NOT used here because it changes the
    # sub-sample contract for the hybrid-CP path: it copies a fixed key list
    # (dropping any extra dataset keys), squeezes 2-D tensors in place, and
    # adds synthesized "original_seq_len"/"padded_seq_len" keys, which would
    # alter the all-to-all key set in reroute_samples_to_hdp_ranks and the
    # samples returned from __next__.
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

        global_id_seqlens, global_ids_this_rank, offsets, _ = _get_global_seqlens_and_ids(
            subsample_seqlens, self.dp_group
        )

        groups, sample_id_groups = self.cp_balancing_scheduler.get_groups_and_subsamples(
            global_id_seqlens, self.config
        )

        batch = self.unpack_batch(batch)
        samples_this_rank_with_id = self.reroute_samples_to_hdp_ranks(
            batch, global_ids_this_rank, global_id_seqlens, sample_id_groups, offsets
        )
        return samples_this_rank_with_id, sample_id_groups
