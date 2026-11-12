# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import bisect
from typing import List, TypeVar

T = TypeVar("T")


def search_for_fit(numbers: List[int], capacity: int) -> int:
    """Finds the index of largest number that fits into the knapsack with the given capacity."""
    import bisect

    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(item_sizes: List[int], samples: List[T], max_capacity: int) -> List[List[T]]:
    """Greedy algorithm with binary search for the knapsack problem.

    Pack as many samples as possible given a maximum capacity and capacities of individual samples.
    Used if sequence packing is enabled.
    """
    assert len(item_sizes) == len(samples), "sample lengths and samples must have the same length."

    knapsacks = []

    if len(item_sizes) == 0:
        return knapsacks

    # Sort sample lengths and samples together.
    sorted_item_sizes, sorted_samples = zip(*sorted(zip(item_sizes, samples), key=lambda x: x[0]))
    sorted_item_sizes = list(sorted_item_sizes)
    sorted_samples = list(sorted_samples)

    # Check if all samples fit in the knapsack capacity.
    if sorted_item_sizes[-1] > max_capacity:
        raise ValueError(
            f"knapsack: A sample is larger {sorted_item_sizes[-1]} than the max_sequence_length {max_capacity}."
        )

    while sorted_item_sizes:
        current_knapsack = []
        remaining_capacity = max_capacity

        while True:
            idx = search_for_fit(sorted_item_sizes, remaining_capacity)
            if idx == -1:
                break  # Can't fit more samples.

            remaining_capacity -= sorted_item_sizes[idx]

            sorted_item_sizes.pop(idx)
            sample = sorted_samples.pop(idx)
            current_knapsack.append(sample)

        knapsacks.append(current_knapsack)

    return knapsacks


def balanced_greedy_knapsack(
    item_sizes: List[int], samples: List[T], max_capacity: int, delta: int = 20
) -> List[List[T]]:
    """Balanced greedy knapsack algorithm for distributing samples across knapsacks."""
    item_size_samples = list(zip(item_sizes, samples))
    item_size_samples.sort(key=lambda x: x[0], reverse=True)
    item_sizes = [item_size for item_size, _ in item_size_samples]
    samples = [sample for _, sample in item_size_samples]
    total_length = sum(item_sizes)
    min_knapsacks = int((total_length + max_capacity - 1) // max_capacity + delta)
    knapsacks = [[] for _ in range(min_knapsacks)]
    knapsack_lengths = [0] * min_knapsacks
    ks_index = 0
    sample_index = 0
    while sample_index < len(item_sizes):
        length = item_sizes[sample_index]
        if length > max_capacity:
            print(f"Warning: sample {sample_index} has length {length} > max_capacity. Skipping.")
            sample_index += 1
            continue
        if knapsack_lengths[ks_index] + length <= max_capacity:
            knapsacks[ks_index].append(samples[sample_index])
            knapsack_lengths[ks_index] += length
            sample_index += 1
        else:
            knapsacks.append([])
            knapsack_lengths.append(0)
        ks_index = knapsack_lengths.index(min(knapsack_lengths))
    return knapsacks


def bucketing_greedy_knapsack(
    item_sizes: list[int], samples: list, max_capacity: int
) -> list[list]:
    """
    Bucketing greedy knapsack algorithm for distributing samples across knapsacks.
    Each packed sample will be composed of samples with similar lengths.
    Bucketing is implicit - by sorting and packing greedily, similar-sized items
    naturally end up in the same batches.

    Note(pzelasko): This type of bucketing is inefficient hardware-wise because the 'max_capacity' heuristic
    results in batches that do not utilize the GPU memory equally across different sequence length buckets.
    This implementation can be improved through OOMptimizer, i.e. pre-computing a list of bucket bins (ideally with shapes aligned to hardware)
    and finding the maximum batch size that can still fit in the GPU memory during the full training step, for each bin.

    Args:
        item_sizes: List of sizes for each sample
        samples: List of samples to pack
        max_capacity: Maximum capacity (sum of sizes) for each batch

    Returns:
        List of batches, where each batch contains samples with similar sizes
    """
    assert len(item_sizes) == len(samples), "item_sizes and samples must have the same length."

    if len(item_sizes) == 0:
        return []

    # Check if any sample exceeds max_capacity
    if max(item_sizes) > max_capacity:
        raise ValueError(
            f"bucketing_greedy_knapsack: A sample has size {max(item_sizes)} "
            f"which exceeds max_capacity {max_capacity}."
        )

    # Sort by size (ascending)
    item_size_samples = list(zip(item_sizes, samples))
    item_size_samples.sort(key=lambda x: x[0])

    # Build batches by iterating through sorted items
    batches = []
    current_batch = []
    current_tot_size = 0

    for size, sample in item_size_samples:
        # If adding this item would exceed capacity, start a new batch
        if current_tot_size + size > max_capacity:
            if current_batch:
                batches.append(current_batch)
            current_batch = [sample]
            current_tot_size = size
        else:
            # Add item to current batch
            current_batch.append(sample)
            current_tot_size += size

    # Add the last batch if it has items
    if current_batch:
        batches.append(current_batch)

    return batches


def _sample_prompt_hash(sample: T, sample_idx: int) -> str:
    prompt_hash = getattr(sample, "prompt_hash", None)
    if prompt_hash is not None:
        return str(prompt_hash)

    sample_key = getattr(sample, "__key__", None)
    if sample_key is not None:
        return f"key:{sample_key}"

    return f"idx:{sample_idx}"


def streaming_prompt_dedup_first_fit_knapsack(
    item_sizes: List[int], samples: List[T], max_capacity: int, tolerance: int = 1000
) -> List[List[T]]:
    """Streaming first-fit packing modeled after Nano 3.5 text SFT offline packing.

    The algorithm preserves the incoming sample stream order more than the sorted
    knapsack variants. It keeps one current pack and a sorted list of unfinished
    packs, first trying to place a sample into the tightest unfinished pack that
    has enough remaining capacity. A sample is not placed into a pack that already
    contains the same prompt hash.
    """
    assert len(item_sizes) == len(samples), "sample lengths and samples must have the same length."

    full_packs: list[list[T]] = []
    unfinished_packs: list[dict] = []
    current_pack: list[T] = []
    current_prompt_hashes: set[str] = set()
    current_pack_tokens = 0

    def insert_unfinished_pack(pack: list[T], pack_tokens: int, prompt_hashes: set[str]) -> None:
        if not pack:
            return
        remaining = max_capacity - pack_tokens
        ranks = [unfinished_pack["num_tokens_to_full"] for unfinished_pack in unfinished_packs]
        insert_idx = bisect.bisect(ranks, remaining)
        unfinished_packs.insert(
            insert_idx,
            {
                "current_pack": pack,
                "current_pack_tokens": pack_tokens,
                "num_tokens_to_full": remaining,
                "current_prompt_hashes": prompt_hashes,
            },
        )

    for sample_idx, (current_sample_tokens, sample) in enumerate(zip(item_sizes, samples)):
        if current_sample_tokens > max_capacity:
            raise ValueError(
                "streaming_prompt_dedup_first_fit_knapsack: A sample has size "
                f"{current_sample_tokens} which exceeds max_capacity {max_capacity}."
            )

        prompt_hash = _sample_prompt_hash(sample, sample_idx)

        filled_unfinished_pack = None
        for unfinished_idx, unfinished_pack in enumerate(unfinished_packs):
            if (
                unfinished_pack["num_tokens_to_full"] >= current_sample_tokens
                and prompt_hash not in unfinished_pack["current_prompt_hashes"]
            ):
                unfinished_pack["current_pack"].append(sample)
                unfinished_pack["current_pack_tokens"] += current_sample_tokens
                unfinished_pack["num_tokens_to_full"] -= current_sample_tokens
                unfinished_pack["current_prompt_hashes"].add(prompt_hash)
                filled_unfinished_pack = unfinished_packs.pop(unfinished_idx)
                break

        if filled_unfinished_pack is not None:
            if filled_unfinished_pack["num_tokens_to_full"] < tolerance:
                full_packs.append(filled_unfinished_pack["current_pack"])
            else:
                insert_unfinished_pack(
                    filled_unfinished_pack["current_pack"],
                    filled_unfinished_pack["current_pack_tokens"],
                    filled_unfinished_pack["current_prompt_hashes"],
                )
            continue

        if current_pack and current_pack_tokens + current_sample_tokens > max_capacity:
            num_tokens_to_full = max_capacity - current_pack_tokens
            if num_tokens_to_full < tolerance:
                full_packs.append(current_pack)
            else:
                insert_unfinished_pack(current_pack, current_pack_tokens, current_prompt_hashes)

            current_pack = []
            current_prompt_hashes = set()
            current_pack_tokens = 0

        if prompt_hash not in current_prompt_hashes:
            current_prompt_hashes.add(prompt_hash)
            current_pack.append(sample)
            current_pack_tokens += current_sample_tokens
        else:
            insert_unfinished_pack(current_pack, current_pack_tokens, current_prompt_hashes)
            current_pack = [sample]
            current_prompt_hashes = {prompt_hash}
            current_pack_tokens = current_sample_tokens

    if current_pack:
        full_packs.append(current_pack)

    full_packs.extend(unfinished_pack["current_pack"] for unfinished_pack in unfinished_packs)

    return full_packs
