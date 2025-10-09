# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from collections import deque
from functools import lru_cache
from math import ceil, log2
from typing import Any, Callable, List, Optional, Tuple

import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunDataIterator


class HybridCPDatasetWrapper:
    """
    A wrapper class that wraps around any existing dataset.
    It adds batch_idx to the sample and returns the sample.

    This is required if certains DP ranks need the dataset idx
    to access the appropriate sub-sample assigned to them from
    the file system even if it was not originally assigned to them.

    Args:
        dataset: The original dataset to wrap around
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Any:
        """
        Get the next item from the dataset, add batch_idx and return it.
        """
        sample = self.dataset[idx]
        sample["batch_idx"] = idx
        assert (
            "cu_seqlens" in sample
        ), "cu_seqlens must be in the sample to use hybrid context parallel"
        return sample


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
        self.cp_balancing_scheduler = BalancedCPScheduler(
            max_seq_len_per_rank=self.config.max_seqlen_per_dp_cp_rank
        )
        if pg_collection is None:
            self.dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
            self.dp_group = parallel_state.get_data_parallel_group()
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            self.dp_cp_group = pg_collection.dp_cp
            self.dp_group = pg_collection.dp
            self.tp_group = pg_collection.tp
        assert (
            self.dp_cp_group is not None and self.dp_group is not None
        ), "dp_cp_group and dp_group not found"

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


class BalancedCPScheduler:
    """
    This class provides the functionality to form groups of sub-samples
    such that all DPxCP ranks have a roughly balanced workload in the group.
    """

    def __init__(self, max_seq_len_per_rank: int):
        self.max_seq_len_per_rank = max_seq_len_per_rank
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
            candidate_gids = [gid for gid, sz in group_size.items() if sz == needed]
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
            Iteratively pop the most‑recent sequence from the *most‑loaded group*
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

        trim_overload()

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
            "try to increase 'max-seqlen-per-cp-rank'."

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

        return groups, sample_id_groups


def hybrid_context_parallel_forward_backward(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    output_tensor_grad,
    forward_data_store,
    config,
    collect_non_loss_data,
    first_val_step,
    forward_only,
    no_sync_func,
    total_num_tokens,
    check_first_val_step,
    model_type,
):
    """
    Scheduler for Hybrid Context Parallel.

    This function performs the packed sample scheduling and determines
    1. The number of microbatches to schedule for each CP rank
    2. The number of groups each CP rank should execute
    3. The number of sub-samples per group each CP rank should execute

    A group is defined by a set of samples that can run across the CP domain without any barrier.
    There are many reasons why we may not be able to run endless samples within a single group.
    For example, if we have 8 GPUs,
    if GPU 0-5 are assigned a long sample that requires CP6,
    GPU 6-7 are assigned a short sample that requires CP2,
    The next sample which requires CP4 can be assigned GPU 4-7.
    But GPU 6-7 will finish first and get deadlocked if GPU 4-5 are not participating in the group.
    """
    from .schedules import backward_step, forward_step

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                parallel_state.get_tensor_model_parallel_src_rank(),
                group=parallel_state.get_tensor_model_parallel_group(),
            )

    def _broadcast_num_samples_this_group(num_samples_this_group):
        dev = torch.cuda.current_device()
        torch.distributed.barrier()

        n = 0 if num_samples_this_group is None else int(num_samples_this_group.numel())
        n = torch.tensor([n], dtype=torch.int64, device=dev)

        _broadcast(n)
        n = int(n.item())

        assert n > 0, "there should be at least 1 sub samples in the group"
        num_samples_this_group_broadcast = (
            torch.empty(n, dtype=torch.int32, device=dev)
            if num_samples_this_group is None
            else num_samples_this_group
        )
        _broadcast(num_samples_this_group_broadcast)
        return num_samples_this_group_broadcast

    def _get_new_data_iterator(sample_id_in_group, group_id):
        if is_first_tp_rank:
            sub_sample_id = sample_ids_this_group[sample_id_in_group]
            sample = batch[sub_sample_id]
            partner_cp_size = len(
                [True for sample_ids in sample_id_groups[group_id] if sub_sample_id in sample_ids]
            )
            sample["local_cp_size"] = torch.tensor(partner_cp_size, dtype=torch.int32)
            new_data_iterator = RerunDataIterator(iter([sample]))
            return new_data_iterator
        else:
            return None

    # We get data once per global batch and schedule the sub-samples.
    # TODO(pmannan): Should we wrap the data_iterator here instead of the training.py file?
    hdp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    is_first_tp_rank = parallel_state.get_tensor_model_parallel_rank() == 0

    if is_first_tp_rank:
        data = next(data_iterator)
        sample_id_groups = data[1]
        batch = data[0]
    else:
        data, sample_id_groups, batch = None, None, None

    num_samples_this_group = None
    if is_first_tp_rank:
        num_samples_this_group = torch.tensor(
            [len(group[hdp_rank]) for group in sample_id_groups], dtype=torch.int32, device='cuda'
        )

    num_samples_this_group = _broadcast_num_samples_this_group(num_samples_this_group)
    num_samples_this_group = num_samples_this_group.cpu().numpy()
    num_total_groups = num_samples_this_group.shape[0]

    # TODO: How does this variable affect downstream logic?
    num_microbatches = 1

    # Upto last group, we don't need any sync.
    with no_sync_func():
        for j in range(num_total_groups - 1):
            sample_ids_this_group = sample_id_groups[j][hdp_rank] if is_first_tp_rank else None
            for i in range(num_samples_this_group[j]):
                # Call forward step for each sub-sample
                new_data_iterator = _get_new_data_iterator(i, j)
                # TODO: Find the usage of current_microbatch and is_first_microbatch and
                # how that may affect my usage.
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    new_data_iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(
                        first_val_step, forward_only, num_microbatches == 1
                    ),
                    current_microbatch=num_microbatches - 1,
                )
                total_num_tokens += num_tokens.item()
                if not forward_only:
                    backward_step(
                        input_tensor, output_tensor, output_tensor_grad, model_type, config
                    )

            # Create a barrier at end of each group.
            # This barrier ensures that all ranks are prepared to change assigned CP group sizes and
            # no rank is starting a sub-sample ahead of it's partner ranks.
            torch.distributed.barrier(
                parallel_state.get_data_parallel_group(with_context_parallel=True)
            )

    # For the last group, we need to run the last sub-sample out of the context handler.
    with no_sync_func():
        sample_ids_this_group = sample_id_groups[-1][hdp_rank] if is_first_tp_rank else None
        for i in range(num_samples_this_group[-1] - 1):
            new_data_iterator = _get_new_data_iterator(i, -1)
            # Call forward step for each sub-sample
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                new_data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(
                    first_val_step, forward_only, num_microbatches == 1
                ),
                current_microbatch=num_microbatches - 1,
            )
            total_num_tokens += num_tokens.item()
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    # The last sub-sample of the last group of the last microbatch is
    # run out of the context handler.
    new_data_iterator = _get_new_data_iterator(-1, -1)
    # Call forward step for each sub-sample
    output_tensor, num_tokens = forward_step(
        forward_step_func,
        new_data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        is_first_microbatch=check_first_val_step(
            first_val_step, forward_only, num_microbatches == 1
        ),
        current_microbatch=num_microbatches - 1,
    )
    total_num_tokens += num_tokens.item()
    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    torch.distributed.barrier(parallel_state.get_data_parallel_group(with_context_parallel=True))

    return forward_data_store, total_num_tokens
