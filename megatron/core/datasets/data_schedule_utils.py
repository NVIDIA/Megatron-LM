# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

from typing import Dict, List

import torch

from megatron.core.rerun_state_machine import RerunDataIterator


def _unpack_batch(batch: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """Normalize batches into one dictionary per variable-length sequence."""

    if not batch:
        return []

    data_keys = ("tokens", "labels", "loss_mask", "position_ids")

    def _without_collate_dim(value):
        if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[0] == 1:
            return value.squeeze(0)
        return value

    # Variable-length datasets already emit one sequence per sample.
    if "padded_seq_len" in batch[0]:
        normalized_batch = []
        for sample in batch:
            normalized = {key: _without_collate_dim(value) for key, value in sample.items()}
            if "original_seq_len" not in normalized:
                normalized["original_seq_len"] = normalized["padded_seq_len"].clone()
            original_length = int(normalized["original_seq_len"].reshape(-1)[0].item())
            padded_length = int(normalized["padded_seq_len"].reshape(-1)[0].item())
            if original_length <= 0 or original_length > padded_length:
                raise ValueError(
                    "Each real sequence length must be positive and no larger than its "
                    "physical length"
                )
            for key in data_keys:
                if key in normalized and normalized[key].numel() != padded_length:
                    raise ValueError(f"{key} must contain padded_seq_len values")
            normalized_batch.append(normalized)
        return normalized_batch

    batch_unpacked = []
    original_sequence_lengths = []
    padded_sequence_lengths = []
    device = batch[0]["cu_seqlens"].device
    for sample in batch:
        cu_seqlens = _without_collate_dim(sample["cu_seqlens"])
        if "cu_seqlens_original" not in sample:
            raise ValueError(
                "Pre-packed samples require cu_seqlens_original so the packing scheduler "
                "can distinguish real tokens from physical padding."
            )
        cu_seqlens_original = _without_collate_dim(sample["cu_seqlens_original"])
        if cu_seqlens_original.numel() != cu_seqlens.numel():
            raise ValueError("cu_seqlens_original must have the same width as cu_seqlens")
        normalized_data = {
            key: _without_collate_dim(sample[key]) for key in data_keys if key in sample
        }
        for sequence_index in range(cu_seqlens.numel() - 1):
            start = int(cu_seqlens[sequence_index].item())
            end = int(cu_seqlens[sequence_index + 1].item())
            if end == start:
                continue
            original_length = int(
                (
                    cu_seqlens_original[sequence_index + 1] - cu_seqlens_original[sequence_index]
                ).item()
            )
            padded_length = end - start
            if original_length <= 0 or original_length > padded_length:
                raise ValueError(
                    "Each real sequence length must be positive and no larger than its "
                    "physical length"
                )
            batch_unpacked.append({key: value[start:end] for key, value in normalized_data.items()})
            original_sequence_lengths.append(original_length)
            padded_sequence_lengths.append(padded_length)

    original_lengths = torch.tensor(original_sequence_lengths, dtype=torch.int32, device=device)
    padded_lengths = torch.tensor(padded_sequence_lengths, dtype=torch.int32, device=device)
    for index, sample in enumerate(batch_unpacked):
        sample["original_seq_len"] = original_lengths[index : index + 1]
        sample["padded_seq_len"] = padded_lengths[index : index + 1]

    return batch_unpacked


def _get_global_seqlens_and_ids(subsample_seqlens: torch.Tensor, dp_group):
    """
    Gathers the sequence lengths of all subsamples from all DP ranks and calculates global IDs.
    """
    # Collect the number of subsamples from all ranks
    num_local_subsamples = subsample_seqlens.shape[0]
    device = subsample_seqlens.device
    local_len = torch.tensor([num_local_subsamples], dtype=torch.int32, device=device)
    dp_subsample_count = [torch.zeros_like(local_len) for _ in range(dp_group.size())]
    torch.distributed.all_gather(dp_subsample_count, local_len, group=dp_group)

    # Find the max number of subsamples across all ranks and pad subsample_seqlens to max length
    dp_subsample_counts = torch.stack(dp_subsample_count, dim=0).cpu().view(-1)
    max_sub_samples = int(dp_subsample_counts.max().item())

    if num_local_subsamples < max_sub_samples:
        subsample_seqlens_padded = torch.cat(
            [
                subsample_seqlens,
                torch.zeros(
                    max_sub_samples - num_local_subsamples, dtype=torch.int32, device=device
                ),
            ],
            dim=0,
        )
    else:
        subsample_seqlens_padded = subsample_seqlens

    # Gather the subsample_seqlens from all ranks
    seqlens_gathered = [torch.empty_like(subsample_seqlens_padded) for _ in range(dp_group.size())]
    torch.distributed.all_gather(seqlens_gathered, subsample_seqlens_padded, group=dp_group)

    # Trim each seqlens_gathered to the length of the correct sample
    for dp_rank, seqlen in enumerate(seqlens_gathered):
        seqlens_gathered[dp_rank] = seqlen[: dp_subsample_counts[dp_rank]]

    seqlens_gathered = torch.cat(seqlens_gathered, dim=0)
    seqlens_gathered = seqlens_gathered.cpu().tolist()

    # Calculate the offsets to assign unique global ID to each subsample.
    csum = torch.cumsum(dp_subsample_counts, dim=0, dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum], dim=0)

    # Calculate global ID for each subsample
    dp_rank = dp_group.rank()
    global_ids = torch.arange(len(seqlens_gathered), dtype=torch.int32, device=device)

    # Create a list of (global_id, seqlen) tuples for scheduling
    global_id_seqlens = [(i, seqlens_gathered[i]) for i in range(len(global_ids))]

    # Get the global IDs locally present on this rank
    start_idx = int(offsets[dp_rank].item())
    end_idx = int(offsets[dp_rank + 1].item())

    global_ids_this_rank = global_ids[start_idx:end_idx]

    return global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered


def _get_group_local_ranks(subgroup, parent_group) -> List[int]:
    """Map subgroup members to their local ranks in a containing process group."""

    parent_ranks = torch.distributed.get_process_group_ranks(parent_group)
    parent_rank_by_global_rank = {
        global_rank: rank for rank, global_rank in enumerate(parent_ranks)
    }
    subgroup_ranks = torch.distributed.get_process_group_ranks(subgroup)
    try:
        return [parent_rank_by_global_rank[global_rank] for global_rank in subgroup_ranks]
    except KeyError as exc:
        raise ValueError("subgroup must be contained in parent_group") from exc


def _pack_sequences(
    samples: List, padded_lengths: torch.Tensor, original_lengths: torch.Tensor, dev: torch.device
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

    padded_lengths = padded_lengths.to(device=dev, dtype=torch.int32, non_blocking=True).reshape(-1)
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


def broadcast_tensor(item, src_rank, group) -> None:
    """Broadcast a tensor from src_rank to all ranks in the group."""
    if item is not None:
        torch.distributed.broadcast(item, src_rank, group=group)


def _serialize_packed_metadata(
    samples: List[Dict[str, torch.Tensor]], device: torch.device
) -> torch.Tensor:
    """Serialize packed-sequence metadata without floating-point casts."""

    tensors = [torch.tensor([len(samples)], dtype=torch.int64, device=device)]
    for sample in samples:
        cu_seqlens = sample["cu_seqlens"].reshape(-1).to(dtype=torch.int64, device=device)
        cu_seqlens_padded = (
            sample["cu_seqlens_padded"].reshape(-1).to(dtype=torch.int64, device=device)
        )
        lengths = torch.tensor(
            [cu_seqlens.numel(), cu_seqlens_padded.numel()], dtype=torch.int64, device=device
        )
        tensors.append(
            torch.cat(
                (sample["max_seqlen"].reshape(1).to(dtype=torch.int64, device=device), lengths)
            )
        )
        tensors.extend((cu_seqlens, cu_seqlens_padded))
    return torch.cat(tensors)


def _deserialize_packed_metadata(payload: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
    """Deserialize length-prefixed packed-sequence metadata."""

    assert payload.dtype == torch.int64
    cursor = 0
    num_samples = int(payload[cursor].item())
    cursor += 1
    samples = []
    for _ in range(num_samples):
        max_seqlen = payload[cursor].to(torch.int32)
        cu_seqlens_len = int(payload[cursor + 1].item())
        cu_seqlens_padded_len = int(payload[cursor + 2].item())
        cursor += 3

        cu_seqlens = payload[cursor : cursor + cu_seqlens_len].to(torch.int32)
        cursor += cu_seqlens_len
        cu_seqlens_padded = payload[cursor : cursor + cu_seqlens_padded_len].to(torch.int32)
        cursor += cu_seqlens_padded_len
        samples.append(
            {
                "max_seqlen": max_seqlen,
                "cu_seqlens": cu_seqlens,
                "cu_seqlens_padded": cu_seqlens_padded,
            }
        )

    assert cursor == payload.numel(), (
        f"Packed metadata decoder consumed {cursor} values, "
        f"but payload contains {payload.numel()}."
    )
    return samples


def broadcast_to_pp_group(
    new_samples,
    num_micro_batches,
    seqlen_sum_this_global_batch,
    seqlen_squared_sum_this_global_batch,
    pp_group,
    dev,
    preserve_local_samples: bool = False,
):
    """Broadcast packed metadata and batch statistics to intermediate PP stages."""

    if pp_group.size() <= 2:
        return (
            new_samples,
            num_micro_batches,
            seqlen_sum_this_global_batch,
            seqlen_squared_sum_this_global_batch,
        )

    pp_src_rank = torch.distributed.get_process_group_ranks(pp_group)[0]
    if pp_group.rank() == 0:
        metadata = _serialize_packed_metadata(new_samples, dev)
        metadata_size = torch.tensor(metadata.numel(), dtype=torch.int64, device=dev)
        stats = torch.tensor(
            [seqlen_sum_this_global_batch, seqlen_squared_sum_this_global_batch],
            dtype=torch.float64,
            device=dev,
        )
    else:
        metadata_size = torch.zeros((), dtype=torch.int64, device=dev)
        stats = torch.zeros(2, dtype=torch.float64, device=dev)

    broadcast_tensor(metadata_size, pp_src_rank, pp_group)
    if pp_group.rank() != 0:
        metadata = torch.empty(int(metadata_size.item()), dtype=torch.int64, device=dev)
    broadcast_tensor(metadata, pp_src_rank, pp_group)
    broadcast_tensor(stats, pp_src_rank, pp_group)

    if pp_group.rank() not in (0, pp_group.size() - 1) and not preserve_local_samples:
        new_samples = _deserialize_packed_metadata(metadata)
        num_micro_batches = len(new_samples)
        seqlen_sum_this_global_batch = float(stats[0].item())
        seqlen_squared_sum_this_global_batch = float(stats[1].item())

    return (
        new_samples,
        num_micro_batches,
        seqlen_sum_this_global_batch,
        seqlen_squared_sum_this_global_batch,
    )


def broadcast_scalars(values: List, group, dev, dtype=torch.float32) -> List:
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

    broadcast_tensor(info_to_broadcast, src_rank, group)

    if group.rank() != 0:
        values = info_to_broadcast.cpu().tolist()

    return values


def create_data_iterator(
    new_samples, tp_group, config, vpp_needs_data=None, needs_full_data: bool = False
):
    """Create TP-rank-zero iterators with the standard dataloader batch dimension."""

    def _normalize_sample(sample):
        normalized = dict(sample)
        for key in (
            "tokens",
            "labels",
            "loss_mask",
            "position_ids",
            "cu_seqlens",
            "cu_seqlens_padded",
        ):
            value = normalized.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 1:
                normalized[key] = value.unsqueeze(0)
        if isinstance(normalized.get("max_seqlen"), torch.Tensor):
            normalized["max_seqlen"] = normalized["max_seqlen"].reshape(1)
        return normalized

    if tp_group.rank() != 0:
        if (
            config.virtual_pipeline_model_parallel_size is not None
            and config.virtual_pipeline_model_parallel_size > 1
        ):
            return [None] * config.virtual_pipeline_model_parallel_size
        return None

    new_samples = [_normalize_sample(sample) for sample in new_samples]
    metadata_keys = ("max_seqlen", "cu_seqlens", "cu_seqlens_padded")

    if (
        config.virtual_pipeline_model_parallel_size is not None
        and config.virtual_pipeline_model_parallel_size > 1
    ):
        vpp_size = config.virtual_pipeline_model_parallel_size
        assert vpp_needs_data is not None and len(vpp_needs_data) == vpp_size
        iterators = []
        for stage_needs_data in vpp_needs_data:
            if stage_needs_data:
                assert needs_full_data
                samples = [dict(sample) for sample in new_samples]
            else:
                samples = [{key: sample[key] for key in metadata_keys} for sample in new_samples]
            iterators.append(RerunDataIterator(iter(samples)))
        return iterators

    return RerunDataIterator(iter(new_samples))


def reroute_samples_to_dcp_ranks(
    batch,
    global_ids_this_rank,
    global_id_seqlens,
    sample_id_groups,
    offsets,
    dp_group,
    dp_cp_group,
    total_dcp_gpus,
):
    """
    Reroutes the sub-samples to the correct rank after scheduling.

    For each key in the batch dict, we perform an all-to-all communication
    to transfer the data to the correct ranks.
    """

    dp_ranks = _get_group_local_ranks(dp_group, dp_cp_group)

    def _gid_to_src_rank(gid: int) -> int:
        gid_tensor = torch.tensor(gid, dtype=offsets.dtype, device=offsets.device)
        dp_src_rank = int(torch.bucketize(gid_tensor, offsets[1:] - 1).item())
        return dp_ranks[dp_src_rank]

    gid2local_id = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
    global_ids_this_rank_set = set(gid2local_id)
    dcp_rank = dp_cp_group.rank()
    dp_rank_set = set(dp_ranks)

    data_keys = [
        key
        for key in (
            "tokens",
            "labels",
            "loss_mask",
            "position_ids",
            "original_seq_len",
            "padded_seq_len",
        )
        if key in batch[0]
    ]

    # Create the send plan
    combined_sample_id_groups: List[List[int]] = [[] for _ in range(total_dcp_gpus)]
    for d in range(total_dcp_gpus):
        for sample_id_group in sample_id_groups:
            combined_sample_id_groups[d].extend(sample_id_group[d])
    for dest_rank in range(total_dcp_gpus):
        combined_sample_id_groups[dest_rank].sort()

    send_ids_sorted = [
        gid
        for d in range(total_dcp_gpus)
        if d in dp_rank_set
        for gid in combined_sample_id_groups[d]
        if gid in global_ids_this_rank_set
    ]

    send_num_split = [0] * total_dcp_gpus
    send_lens_split = [0] * total_dcp_gpus
    for dest_rank in range(total_dcp_gpus):
        if dest_rank in dp_rank_set:
            send_seq_lens = [
                global_id_seqlens[gid][1]
                for gid in combined_sample_id_groups[dest_rank]
                if gid in global_ids_this_rank_set
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
            else torch.empty(0, device=torch.cuda.current_device(), dtype=batch[0][key].dtype)
        )

    def _unpack_sample_by_key(key: str, recv_tensor: torch.Tensor):
        cursor = 0
        for i, gid in enumerate(recv_ids_sorted):
            sample_len = (
                1 if key in ["original_seq_len", "padded_seq_len"] else global_id_seqlens[gid][1]
            )
            recv_samples[i][key] = recv_tensor[cursor : cursor + sample_len]
            cursor += sample_len
        assert cursor == recv_tensor.numel()

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

    recv_sample_with_id = {recv_id: recv_samples[i] for i, recv_id in enumerate(recv_ids_sorted)}
    return recv_sample_with_id


def build_packed_microbatches(
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
        lens_padded = padded_lens_all_gpu[seg_starts[i] : seg_starts[i + 1]]
        lens_original = original_lens_all_gpu[seg_starts[i] : seg_starts[i + 1]]
        new_sample = _pack_sequences(samples, lens_padded, lens_original, dev)
        new_samples.append(new_sample)

    return new_samples


def get_batch_and_global_seqlens(data_iterator, num_microbatches, dp_group):
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

    batch_list = [next(data_iterator) for _ in range(num_microbatches)]

    batch = []
    for item in batch_list:
        if isinstance(item, dict):
            batch.append(item)
        elif isinstance(item, list):
            batch.extend(item)
        else:
            raise ValueError(f"Invalid item type: {type(item)}")

    # in sft_dataset.py, sequences are already packed before rescheduling,
    # so we need to unpack them here and repack after rescheduling.
    # This is only to adapt to the current megatron-lm sft_dataset.
    # If you implement your own dataset, just have __getitem__ return List[Dict]
    # and this step can be skipped.
    batch = _unpack_batch(batch)
    if not batch:
        raise ValueError("Sequence packing requires at least one non-empty sample.")

    subsample_seqlens = torch.cat([sample["padded_seq_len"] for sample in batch]).to(
        dtype=torch.int32, device=torch.cuda.current_device()
    )

    global_id_seqlens, global_ids_this_rank, offsets, _ = _get_global_seqlens_and_ids(
        subsample_seqlens, dp_group
    )

    original_lengths = torch.cat([sample["original_seq_len"] for sample in batch]).to(
        dtype=torch.float64, device=torch.cuda.current_device()
    )
    global_stats = torch.stack((original_lengths.sum(), torch.square(original_lengths).sum()))
    torch.distributed.all_reduce(global_stats, group=dp_group)

    return (
        batch,
        global_id_seqlens,
        global_ids_this_rank,
        offsets,
        float(global_stats[0].item()),
        float(global_stats[1].item()),
    )
