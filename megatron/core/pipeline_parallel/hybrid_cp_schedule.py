import torch
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.rerun_state_machine import RerunDataIterator
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from functools import lru_cache
from collections import deque
from math import ceil, log2
import heapq

class HybridCPWrapper():
    """
    A wrapper class that wraps around any existing dataset and prints samples
    as they are requested from the original dataset.
    
    This wrapper implements the standard PyTorch Dataset interface and can be
    used with any dataset that follows the same interface.
    
    Args:
        dataset: The original dataset to wrap around
        print_format: Format string for printing samples (default: "Sample {idx}: {sample}")
        print_func: Function to use for printing (default: print)
        max_print_length: Maximum length of sample to print (default: 200 chars)
        print_every: Print every Nth sample (default: 1, print all)
    """
    
    def __init__(
        self, 
        data_iterator,
        config,
    ):
        self.data_iterator = data_iterator
        self.sample_count = 0
        self.config = config
        self.cp_balancing_scheduler = BalancedCPScheduler(max_seq_len_per_rank=self.config.max_seqlen_per_cp_rank)

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def __next__(self) -> Any:
        """
        Get the next item from the dataset, pull scheduling metadata and return it.
        """
        sample = next(self.data_iterator)
        assert "cu_seqlens" in sample, "cu_seqlens must be in the sample"
        # TODO(milestone 2): Get cu_seqlens and all-gather the entire global batch worth cu_seqlens and then perform the scheduling.
        # But why should this scheduling information be integrated back into the data?
        groups, sample_id_groups = self.cp_balancing_scheduler.get_groups_and_subsamples(sample, self.config)
        sample["groups"] = groups
        sample["sample_id_groups"] = sample_id_groups
        return sample

class BalancedCPScheduler:
    def __init__(self, max_seq_len_per_rank: int):
        self.max_seq_len_per_rank = max_seq_len_per_rank
        self.num_subsamples = 0
        self.num_subsamples_processed = 0
        self.free_resources = []

    @lru_cache(maxsize=128)
    def get_total_workload(self, seq_length: int, cp_size: Optional[int] = None):
        """
        seq_length: sequence length of a sub-sample
        cp_size: total number of CP ranks working on this sub-sample

        Note:
        This function is used to estimate the relative workload intensity
        of a sub-sample. This is not meant to be an accurate flops calculator.

        Returns:
        workload: workload of a sub-sample
        """
        if cp_size is None:
            cp_size = self.gpus_needed(seq_length)
        return (seq_length * seq_length) / cp_size

    @lru_cache(maxsize=128)
    def gpus_needed(self, seq_len: int) -> int:
        return max(1, 2 ** ceil(log2((seq_len / self.max_seq_len_per_rank))))

    def make_buckets_equal(
        self,
        sample_seqlens: List[Tuple[int, int]],  # List of (sample_id, sequence_length) tuples
        compute_estimator: Callable[[int], float],
    ) -> List[deque]:
        """
        Modified version of make_buckets_equal_work that works with (sample_id, seq_len) tuples.
        This keeps sample IDs tethered to their sequence lengths throughout the bucketing process.
        """
        # Extract just the sequence lengths for determining k
        seqlens = [seq_len for _, seq_len in sample_seqlens]
        
        # Determine k based on unique GPU categories needed
        k = len({self.gpus_needed(L) for L in seqlens})
        
        # Use the existing contiguous_equal_buckets function but with sample_seqlens
        # We need to modify it to work with tuples
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
            if (cur and 
                (projected > target * 1.1 or  # Too much work
                len(sample_seqlens) - i <= remaining_k - len(buckets))):  # Need to save sequences for remaining buckets
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
        delta: float = 0.05,                # balance slack (e.g. 5 %)
        strategy: str = "dp",               # "dp" or "pp"
        eps_bucket: float = 0.10,           # ε target for bucket balance
    ) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:
        """
        Given a list of (sample_id, sequence_length) tuples, this function aims to assign
        sequences to a microbatch such that all GPUs in the CP domain have a roughly balanced workload.
        Once each microbatch is roughly balanced, we exit and return the microbatch and the leftover sequences.

        The function performs the following passes in order to form a balanced microbatch:
        1. We create buckets of sequences that are roughly balanced. 
        We try to create as many buckets as possible CP sizes.
        2. Given a bucket has sequences available, we assign the microbatch
            a. To a new set of GPUs if there are enough free GPUs.
            b. To an existing set of GPUs with the lowest load.
        3. We check if the microbatch is balanced whenever we need to move onto a new CP size in the same set of GPUs.
        4. We trim the microbatch if removing the last added sequence helps improve balance.
        5. If we run out of sequences to assign and there are empty GPUs, 
        we redistribute work to empty GPUs by recursively increasing the CP size of a sample until no empty GPUs are left..

        #TODO: Add clarification on when we check for balance. What does prev_needed do?

        Returns (*micro_batches*, *leftover_sample_seqlens*, *exec_times*, *sample_ids_per_gpu*).
        """
        if not sample_seqlens:
            return [[] for _ in range(total_gpus)], [], [0.0 for _ in range(total_gpus)], [[] for _ in range(total_gpus)]

        # Use the improved bucketing that works with (sample_id, seq_len) tuples
        buckets = self.make_buckets_equal(sample_seqlens, compute_estimator)

        # Initialize tracking structures
        micro_batches   = [[] for _ in range(total_gpus)]
        exec_times      = [0.0 for _ in range(total_gpus)]
        sample_ids_per_gpu = [[] for _ in range(total_gpus)]

        gpu_group_id    = [None] * total_gpus
        group_members   = {}
        group_size      = {}
        next_gid        = 0

        pp_cursor       = 0
        prev_needed     = None
        check_balance   = False

        while buckets:
            # ---- Step 1 – pick the next sequence we COULD place ------------------
            sample_seq_tuple = bucket_idx = None
            needed  = None

            scan_order = (
                range(len(buckets)) if strategy == "dp"
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

            if strategy == "pp":
                pp_cursor = (bucket_idx + 1) % len(buckets)

            sample_id, seq_len = sample_seq_tuple
            needed = self.gpus_needed(seq_len)
            if prev_needed is None:
                prev_needed = needed

            # (a)  Existing groups of exactly this size
            candidate_gids = [
                gid for gid, sz in group_size.items() if sz == needed
            ]
            if candidate_gids:
                best_gid, best_load = min(
                    ((gid, max(exec_times[r] for r in group_members[gid]))
                    for gid in candidate_gids),
                    key=lambda t: t[1]
                )
            else:
                best_gid, best_load = None, float("inf")

            # (b)  Hypothetical **new** group from completely free GPUs
            free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
            if len(free_ranks) >= needed:
                free_sorted = sorted(free_ranks, key=lambda r: exec_times[r])
                new_members = free_sorted[:needed]
                new_load    = exec_times[new_members[-1]]

                if new_load < best_load:
                    best_gid = None
                    chosen_members = new_members
                else:
                    chosen_members = group_members[best_gid]
            else:
                if best_gid is None:
                    print(f"No room to form a new group")
                chosen_members = group_members[best_gid]

            # ---- Step 2b – if we decided to create a fresh group ----------------
            if best_gid is None:
                best_gid = next_gid
                next_gid += 1
                group_members[best_gid] = chosen_members
                group_size[best_gid]    = needed
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

            # TODO: Should I pre-emptively break out if slack is already within delta?
            # Feels like if we have global batch level samples, we will have lots with same CP size.
            # So we can just keep adding samples.
            # We already have trim workload to handle imbalanced cases.
            # TODO: Removing this helps reduce the number of groups when we have lots of samples with same CP size.
            # But because we don't exit as soon as we get balanced, even if there is one group available that can take the next sample,
            # we will keep adding samples to the same group.
            # trim_overload() does not help because it only checks if removing the last added sample helps.
            # We cannot check after adding every sample because there will always be imbalance if we don't wait for future scheduling.

            # IMPORTANT: So we need a solution here
            if needed < prev_needed:
                # When we get into a lower CP size in the same group, we can start checking for balance.
                # There is still a gotcha here.
                # Let's say we have a group of 3 GPU 0-2, then we move onto group of 2.
                # We keep assigning group of 2 as we do in descending order but GPU 7/15 never sees a microbatch assigned to it
                # until we run out of samples with CP2.
                # This means we are never balanced as min(exec_times) will always be 0.
                # We need a smart way of identifying that we have run out of big samples and if we are having to 
                # assign work to a GPU already working, is it because there are empty GPUs?
                # Would assigning work to empty GPUs first by moving onto next CP bucket help?
                # But we need to remember to come back to this CP size bucket and then check for balance.
                # Maybe the scheduling algorithm should look at empty GPUs and find work rather than going 
                # sequence by sequence.
                check_balance = True

            if check_balance and buckets and max(exec_times) - min(exec_times) <= delta * max(exec_times):
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
                cur_max  = max(exec_times)
                cur_min  = min(exec_times)
                cur_slack = cur_max - cur_min
                if cur_slack <= delta * cur_max:
                    break

                max_r   = exec_times.index(cur_max)
                gid     = gpu_group_id[max_r]
                members = group_members[gid]

                if not micro_batches[max_r] or len(micro_batches[max_r]) <= 1:
                    break

                seq   = micro_batches[max_r][-1]
                need  = group_size[gid]
                per_gpu_cost = compute_estimator(seq)

                proj_times = exec_times[:]
                for r in members:
                    proj_times[r] -= per_gpu_cost

                proj_slack = max(proj_times) - min(proj_times)

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

        # Track work before redistribution
        total_work_before = sum(len(mb) for mb in micro_batches)

        # Check for empty GPUs and redistribute work
        def fill_empty_gpus(micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size):
            """
            Recursively check for empty GPUs and redistribute work by increasing
            the number of GPUs sharing samples. This ensures all GPUs have work.
            GPUs must be allocated consecutively.
            """
            # Find empty GPUs
            empty_gpus = [i for i in range(total_gpus) if not micro_batches[i]]
            if not empty_gpus:
                return micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size # No empty GPUs, we're done
            
            # Find the smallest group size that exists
            existing_group_sizes = set(group_size.values())
            if not existing_group_sizes:
                return  # No groups exist, cannot redistribute
            
            min_group_size = min(existing_group_sizes)
            next_power = min_group_size * 2
            
            # Find the first group of min_group_size that can be expanded
            expandable_gid = None
            expandable_members = None
            expandable_new_gpus = None
            
            for gid, size in group_size.items():
                if size == min_group_size:
                    members = group_members[gid]
                    # get_new_work_queue(members[-1], min_group_size)
                    needed_count = min_group_size
                    current_gpu = members[-1]
                    empty_gpu = [idx for idx, work in enumerate(micro_batches) if not work][0]
                    assert not all(work for work in micro_batches[empty_gpu:empty_gpu+needed_count]), f"Not enough empty GPUs to expand or there are empty GPUs between work scheduled which is not allowed."
                    work_to_push = micro_batches[current_gpu + 1 : empty_gpu] # This is work of all other subsequent sub-samples
                    exec_times_to_push = exec_times[current_gpu + 1 : empty_gpu]
                    sample_ids_to_push = sample_ids_per_gpu[current_gpu + 1 : empty_gpu]
                    

                    new_micro_batches = [[]] * len(micro_batches)
                    new_exec_times = [0.0] * len(exec_times)
                    new_sample_ids_per_gpu = [[]] * len(sample_ids_per_gpu)

                    for i in range(current_gpu+1):
                        new_micro_batches[i] = micro_batches[i]
                        new_exec_times[i] = exec_times[i]
                        new_sample_ids_per_gpu[i] = sample_ids_per_gpu[i]

                    for i in range(needed_count):
                        new_micro_batches[current_gpu + 1 +i] = micro_batches[current_gpu]
                        new_exec_times[current_gpu + 1 + i] = exec_times[current_gpu]
                        new_sample_ids_per_gpu[current_gpu + 1 + i] = sample_ids_per_gpu[current_gpu]

                    for i, work in enumerate(work_to_push):
                        new_micro_batches[current_gpu + needed_count + 1 + i] = work
                        new_exec_times[current_gpu + needed_count + 1 + i] = exec_times_to_push[i]
                        new_sample_ids_per_gpu[current_gpu + needed_count + 1 + i] = sample_ids_to_push[i]
                    
                    group_size[gid] = next_power
                    group_members[gid] = list(range(members[0], members[-1] + needed_count + 1))
                    for pushed_gid in group_size.keys():
                        if pushed_gid > gid:
                            group_members[pushed_gid] = [x + needed_count for x in group_members[pushed_gid]]                
                    
                    return new_micro_batches, new_exec_times, new_sample_ids_per_gpu, group_members, group_size
            

        empty_gpus = any([not micro_batches[i] for i in range(total_gpus)])
        while empty_gpus:
            micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size = fill_empty_gpus(micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size)
            empty_gpus = any([not micro_batches[i] for i in range(total_gpus)])

        # Assert that no work has been completely removed
        total_work_after = sum(len(mb) for mb in micro_batches)
        assert total_work_after >= total_work_before, f"Work was removed: {total_work_before} -> {total_work_after}"

        return micro_batches, leftovers, exec_times, sample_ids_per_gpu

    def get_groups_and_subsamples(
        self,
        data,
        config,
    ):
        # TODO: Protect for model parallelism
        # TODO: Reduce access to file system as much as possible.
        groups = []
        sample_id_groups = []
        assert "cu_seqlens" in data, (
            "data must have a cu_seqlens attribute to define the valid sequenece lengths "
            "of each sub-sample in a packed sample to use hybrid context parallel"
        )
        # We assign a sample_id to each sub-sample in order to track the right assignment to each GPU.
        # TODO (Milestone 2): Sample ID logic will have to change once we have global batch
        sample_id_seqlens = [(i, int(data["cu_seqlens"][0][i+1] - data["cu_seqlens"][0][i])) for i in range(0, data["cu_seqlens"][0].shape[0] - 1)]
        sample_id_seqlens = sorted(sample_id_seqlens, key=lambda x: x[1], reverse=True)
        while sample_id_seqlens:
            mb, sample_id_seqlens, exec_times, sample_ids = self.next_hdp_group(sample_id_seqlens, self.get_total_workload, config.context_parallel_size)
            groups.append(mb)
            if len(sample_ids) < config.context_parallel_size:
                sample_ids.extend([] * (config.context_parallel_size - len(sample_ids)))
            sample_id_groups.append(sample_ids)
        # print(f"groups: {groups}")
        # print(f"sample_id_groups: {sample_id_groups}")
        
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
    There are many reasons why we may not be able to run endless number of samples within a single group.
    For example, if we have 8 GPUs, 
    if GPU 0-5 are assigned a long sample that requires CP6, 
    GPU 6-7 are assigned a short sample that requires CP2,
    The next sample which requires CP4 can be assigned GPU 4-7.
    But GPU 6-7 will finish first and get deadlocked if GPU 4-5 are not participating in the group.

    As of now, the number of microbatches is pre-determined by GBS and DP size.
    We perform the scheduling for each microbatch.
    In the future, when we schedule over the entire global batch, we will remove the need for step #2 and
    number of microbatches will be determined by the number of groups.
    """
    from .schedules import forward_step, backward_step

    cp_balancing_scheduler = BalancedCPScheduler(max_seq_len_per_rank=config.max_seqlen_per_cp_rank)
    with no_sync_func():
        for i in range(num_microbatches - 1):
            data = next(data_iterator)
            # groups, sample_id_groups = cp_balancing_scheduler.get_groups_and_subsamples(data, model, config)
            groups = data["groups"]
            sample_id_groups = data["sample_id_groups"]
            for j in range(len(groups)):
                # Get sub-samples for the current CP rank
                # TODO: Update to DPxCP rank when milestone 2
                sample_ids_per_group = sample_id_groups[j][parallel_state.get_context_parallel_rank()]
                for k in range(len(sample_ids_per_group)):
                    # Call forward step for each sub-sample
                    sub_sample_id = sample_ids_per_group[k]
                    partner_cp_size = len([True for sample_ids in sample_id_groups[j] if sub_sample_id in sample_ids])
                    if partner_cp_size == 0:
                        assert False, f"rank: {torch.distributed.get_rank()}, sub_sample_id: {sub_sample_id} j: {j} k: {k} sample_ids_group: {sample_id_groups}"
                    data["local_cp_size"] = torch.tensor(partner_cp_size, dtype=torch.int32)
                    data["scheduled_id"] = torch.tensor(sub_sample_id, dtype=torch.int32)
                    new_data_iterator = RerunDataIterator(iter([data]))
                    # TODO: Change data iterator to the right sub-sample
                    # TODO: Find the usage of current_microbatch and is_first_microbatch and how that may affect my usage.
                    output_tensor, num_tokens = forward_step(
                        forward_step_func,
                        new_data_iterator,
                        model,
                        num_microbatches,
                        input_tensor,
                        forward_data_store,
                        config,
                        collect_non_loss_data,
                        is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                        current_microbatch=i,
                    )
                    total_num_tokens += num_tokens.item()
                    if not forward_only:
                        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

                # TODO: Move to DPxCP barrier
                torch.distributed.barrier(parallel_state.get_context_parallel_group())

    # Last microbatch
    # TODO: Call scheduler here.
    with no_sync_func():
        data = next(data_iterator)
        # groups, sample_id_groups = cp_balancing_scheduler.get_groups_and_subsamples(data, model, config)
        groups = data["groups"]
        sample_id_groups = data["sample_id_groups"]
        for j in range(len(groups) - 1):
            sample_ids_per_group = sample_id_groups[j][parallel_state.get_context_parallel_rank()]
            for k in range(len(sample_ids_per_group)):
                # Call forward step for each sub-sample
                sub_sample_id = sample_ids_per_group[k]
                
                partner_cp_size = len([True for sample_ids in sample_id_groups[j] if sub_sample_id in sample_ids])
                if partner_cp_size == 0:
                    assert False, f"rank: {torch.distributed.get_rank()}, sub_sample_id: {sub_sample_id} j: {j} k: {k} sample_ids_group: {sample_id_groups}"
                data["local_cp_size"] = torch.tensor(partner_cp_size, dtype=torch.int32)
                data["scheduled_id"] = torch.tensor(sub_sample_id, dtype=torch.int32)
                # TODO: What else should I update in data so that we can get the right sub-sample?
                new_data_iterator = RerunDataIterator(iter([data]))
                # TODO: Change data iterator to the right sub-sample
                # TODO: Find the usage of current_microbatch and is_first_microbatch and how that may affect my usage.
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    new_data_iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(first_val_step, forward_only, num_microbatches == 1),
                    current_microbatch=num_microbatches - 1,
                )
                total_num_tokens += num_tokens.item()
                if not forward_only:
                    backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
            
            # TODO: Move to DPxCP barrier
            torch.distributed.barrier(parallel_state.get_context_parallel_group())

    # For the last group, we need to run the last sub-sample out of the context handler.
    # TODO: Find num sub-samples per group in this group
    with no_sync_func():
        sample_ids_per_group = sample_id_groups[-1][parallel_state.get_context_parallel_rank()]
        for k in range(len(sample_ids_per_group) - 1):
            sub_sample_id = sample_ids_per_group[k]
            partner_cp_size = len([True for sample_ids in sample_id_groups[-1] if sub_sample_id in sample_ids])
            data["local_cp_size"] = torch.tensor(partner_cp_size, dtype=torch.int32)
            data["scheduled_id"] = torch.tensor(sub_sample_id, dtype=torch.int32)
            # TODO: What else should I update in data so that we can get the right sub-sample?
            new_data_iterator = RerunDataIterator(iter([data]))
            # TODO: Change data iterator to the right sub-sample
            # TODO: Find the usage of current_microbatch and is_first_microbatch and how that may affect my usage.
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
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, num_microbatches == 1),
                current_microbatch=num_microbatches - 1,
            )
            total_num_tokens += num_tokens.item()
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
        
    # The last sub-sample of the last group of the last microbatch is run out of the context handler.
    sub_sample_id = sample_ids_per_group[-1]
    partner_cp_size = len([True for sample_ids in sample_id_groups[-1] if sub_sample_id in sample_ids])
    if partner_cp_size == 0:
        assert False, f"rank: {torch.distributed.get_rank()}, sub_sample_id: {sub_sample_id} j: {j} k: {k} sample_ids_group: {sample_id_groups}"
    data["local_cp_size"] = torch.tensor(partner_cp_size, dtype=torch.int32)
    data["scheduled_id"] = torch.tensor(sub_sample_id, dtype=torch.int32)
    # TODO: What else should I update in data so that we can get the right sub-sample?
    new_data_iterator = RerunDataIterator(iter([data]))
    # TODO: Change data iterator to the right sub-sample
    # TODO: Find the usage of current_microbatch and is_first_microbatch and how that may affect my usage.
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
        is_first_microbatch=check_first_val_step(first_val_step, forward_only, num_microbatches == 1),
        current_microbatch=num_microbatches - 1,
    )
    total_num_tokens += num_tokens.item()
    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    # TODO: Move to DPxCP barrier
    torch.distributed.barrier(parallel_state.get_context_parallel_group())

    # TODO: Before returning forward_data_store, we need to change the loss.
    # Instead of letting reporting loss be calculated by train_step, can we just calculate it here?
    # Since we will need the global number of samples information to average the loss and local cp group size for each sample.
    return forward_data_store, total_num_tokens