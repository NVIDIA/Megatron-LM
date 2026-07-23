# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

from typing import Dict, List, Literal, Optional, Sequence

import torch

from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices
from megatron.core.rerun_state_machine import RerunDataIterator


def get_cp_slice_for_thd(
    batch,
    cp_group,
    keys: Optional[Sequence[str]] = None,
    cp_partition_mode: Literal["zigzag", "contiguous"] = "zigzag",
    partition_total_tokens: Optional[int] = None,
):
    """Partition sequence data for context parallelism in THD format.

    ``zigzag`` uses TE's THD partitioned indices. ``contiguous`` splits the
    flattened rows into equal rank-contiguous slices.

    Args:
        batch: Dict with packed sequence data.
        cp_group: Context parallel process group.
        keys: Sequence data keys to slice. Defaults to the original THD data tensors.
        cp_partition_mode: How to assign packed rows to CP ranks.
        partition_total_tokens: Optional total used to tail-pad tensors selected by
            ``keys`` before slicing. Existing cu_seqlens metadata is left unchanged.
    """
    cp_size = cp_group.size()
    if cp_size <= 1:
        return
    cp_rank = cp_group.rank()
    # Partition with padded cumulative lengths so CP slices match the THD
    # sequence boundaries consumed by attention kernels.
    cu_seqlens = batch["cu_seqlens_padded"]
    # Use cu_seqlens_padded[-1] for total_tokens instead of batch['tokens'].size(0):
    # under VPP, the last PP stage has labels/loss_mask but no tokens, so
    # batch['tokens'] is None on that stage. cu_seqlens_padded is always populated.
    total_tokens = (
        int(cu_seqlens[-1].item())
        if partition_total_tokens is None
        else int(partition_total_tokens)
    )
    if keys is None:
        keys = ('tokens', 'position_ids', 'labels', 'loss_mask')

    if partition_total_tokens is not None:
        for key in keys:
            if key not in batch or batch[key] is None:
                continue
            pad_len = total_tokens - batch[key].numel()
            if pad_len < 0:
                raise RuntimeError(
                    f"partition_total_tokens={total_tokens} is smaller than {key} length "
                    f"{batch[key].numel()}."
                )
            if pad_len > 0:
                pad_value = True if key == 'padding_mask' else 0
                batch[key] = torch.cat([batch[key], batch[key].new_full((pad_len,), pad_value)])

    if cp_partition_mode == "contiguous":
        if total_tokens % cp_size != 0:
            raise RuntimeError(
                f"Contiguous CP slicing requires total_tokens={total_tokens} to be divisible by "
                f"cp_size={cp_size}."
            )
        local_rows = total_tokens // cp_size
        row_slice = slice(cp_rank * local_rows, (cp_rank + 1) * local_rows)
        for key in keys:
            if key in batch and batch[key] is not None:
                batch[key] = batch[key][row_slice]
        return

    if cp_partition_mode != "zigzag":
        raise ValueError(f"Unsupported CP partition mode: {cp_partition_mode}")

    index = get_thd_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)
    for key in keys:
        if key in batch and batch[key] is not None:
            batch[key] = batch[key].index_select(0, index)


def _unpack_batch(batch: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """Normalize samples that are already unpacked by the varlen dataset."""
    for sample in batch:
        if "padded_seq_len" not in sample:
            raise KeyError("sequence packing samples must provide 'padded_seq_len'")
        for key, value in sample.items():
            if value.ndim == 2 and value.shape[0] == 1:
                sample[key] = value.squeeze(0)
        if "original_seq_len" not in sample:
            sample["original_seq_len"] = sample["padded_seq_len"].clone()
    return batch


def _get_global_seqlens_and_ids(
    padded_subsample_seqlens: torch.Tensor, original_subsample_seqlens: torch.Tensor, dp_group
):
    """
    Gathers the sequence lengths of all subsamples from all DP ranks and calculates global IDs.
    """
    # Collect the number of subsamples from all ranks
    assert padded_subsample_seqlens.shape == original_subsample_seqlens.shape
    num_local_subsamples = padded_subsample_seqlens.shape[0]
    local_len = torch.tensor(
        [num_local_subsamples], dtype=torch.int32, device=padded_subsample_seqlens.device
    )
    dp_subsample_count = [torch.zeros_like(local_len) for _ in range(dp_group.size())]
    torch.distributed.all_gather(dp_subsample_count, local_len, group=dp_group)

    # Gather padded and original lengths together so scheduling and FLOPs use
    # the same global sample ordering without an extra collective.
    dp_subsample_counts = torch.stack(dp_subsample_count, dim=0).cpu().view(-1)
    max_sub_samples = int(dp_subsample_counts.max().item())
    local_seqlens = torch.stack([padded_subsample_seqlens, original_subsample_seqlens], dim=1)

    if num_local_subsamples < max_sub_samples:
        local_seqlens = torch.cat(
            [
                local_seqlens,
                torch.zeros(
                    (max_sub_samples - num_local_subsamples, 2),
                    dtype=torch.int32,
                    device=local_seqlens.device,
                ),
            ],
            dim=0,
        )

    seqlens_gathered = [torch.empty_like(local_seqlens) for _ in range(dp_group.size())]
    torch.distributed.all_gather(seqlens_gathered, local_seqlens, group=dp_group)

    # Trim each seqlens_gathered to the length of the correct sample
    for dp_rank, seqlen in enumerate(seqlens_gathered):
        seqlens_gathered[dp_rank] = seqlen[: dp_subsample_counts[dp_rank]]

    seqlens_gathered = torch.cat(seqlens_gathered, dim=0).cpu()
    padded_seqlens_gathered = seqlens_gathered[:, 0].tolist()
    original_seqlens_gathered = seqlens_gathered[:, 1].tolist()

    # Calculate the offsets to assign unique global ID to each subsample.
    csum = torch.cumsum(dp_subsample_counts, dim=0, dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum], dim=0)

    # Calculate global ID for each subsample
    dp_rank = dp_group.rank()
    global_ids = torch.arange(
        len(padded_seqlens_gathered), dtype=torch.int32, device=padded_subsample_seqlens.device
    )

    # Create a list of (global_id, seqlen) tuples for scheduling
    global_id_seqlens = [(i, padded_seqlens_gathered[i]) for i in range(len(global_ids))]

    # Get the global IDs locally present on this rank
    start_idx = offsets[dp_rank]
    end_idx = offsets[dp_rank + 1]

    global_ids_this_rank = global_ids[start_idx:end_idx]

    return (
        global_id_seqlens,
        global_ids_this_rank,
        offsets,
        padded_seqlens_gathered,
        original_seqlens_gathered,
    )


def _pack_sequences(
    samples: List, padded_lengths: torch.Tensor, original_lengths: torch.Tensor, dev: torch.device
) -> Dict[str, torch.Tensor]:
    """Pack multiple samples into a single packed sample."""

    def _pack_tensors(tensors):
        return torch.cat([t.reshape(-1) for t in tensors], dim=0)

    new_sample = {}
    for key in ['tokens', 'labels', 'loss_mask', 'position_ids']:
        if key in samples[0]:
            new_sample[key] = _pack_tensors([sample[key] for sample in samples])

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


def create_data_iterator(new_samples, tp_group, config, vpp_needs_data=None):
    """Create independent iterators for the virtual pipeline stages."""
    if (
        config.virtual_pipeline_model_parallel_size is not None
        and config.virtual_pipeline_model_parallel_size > 1
    ):
        vpp_size = config.virtual_pipeline_model_parallel_size
        if tp_group.rank() == 0:
            new_data_iterator = []
            for vp_stage in range(vpp_size):
                if vpp_needs_data is not None and vpp_needs_data[vp_stage]:
                    samples = [dict(sample) for sample in new_samples]
                    new_data_iterator.append(RerunDataIterator(iter(samples)))
                else:
                    metadata_keys = ['max_seqlen', 'cu_seqlens', 'cu_seqlens_padded']
                    metadata = [
                        {key: sample[key] for key in metadata_keys if key in sample}
                        for sample in new_samples
                    ]
                    new_data_iterator.append(RerunDataIterator(iter(metadata)))
        else:
            new_data_iterator = [None for _ in range(vpp_size)]
    else:
        new_data_iterator = RerunDataIterator(iter(new_samples)) if tp_group.rank() == 0 else None

    return new_data_iterator


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

    dp_global_ranks = torch.distributed.get_process_group_ranks(dp_group)
    dp_cp_global_ranks = torch.distributed.get_process_group_ranks(dp_cp_group)
    global_to_dcp_rank = {global_rank: rank for rank, global_rank in enumerate(dp_cp_global_ranks)}

    def _gid_to_src_rank(gid: int) -> int:
        dp_src_rank = torch.bucketize(gid, offsets[1:] - 1)
        return global_to_dcp_rank[dp_global_ranks[int(dp_src_rank)]]

    gid2local_id = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
    dcp_rank = dp_cp_group.rank()
    dp_ranks = [global_to_dcp_rank[global_rank] for global_rank in dp_global_ranks]

    data_keys = batch[0].keys()

    # Create the send plan
    combined_sample_id_groups: List[List[int]] = [[] for _ in range(total_dcp_gpus)]
    for d in range(total_dcp_gpus):
        for sample_id_group in sample_id_groups:
            combined_sample_id_groups[d].extend(sample_id_group[d])
    for dest_rank in range(total_dcp_gpus):
        combined_sample_id_groups[dest_rank].sort()

    send_ids_sorted = [
        gid for d in dp_ranks for gid in combined_sample_id_groups[d] if gid in global_ids_this_rank
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

    # Normalize the optional leading batch dimension before scheduling.
    batch = _unpack_batch(batch)

    padded_subsample_seqlens = torch.cat([sample["padded_seq_len"] for sample in batch]).to(
        dtype=torch.int32, device=torch.cuda.current_device()
    )
    original_subsample_seqlens = torch.cat([sample["original_seq_len"] for sample in batch]).to(
        dtype=torch.int32, device=torch.cuda.current_device()
    )

    (
        global_id_seqlens,
        global_ids_this_rank,
        offsets,
        padded_seqlens_gathered,
        original_seqlens_gathered,
    ) = _get_global_seqlens_and_ids(padded_subsample_seqlens, original_subsample_seqlens, dp_group)

    return (
        batch,
        global_id_seqlens,
        global_ids_this_rank,
        offsets,
        padded_seqlens_gathered,
        original_seqlens_gathered,
    )
