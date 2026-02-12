# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

from typing import Dict, List

import numpy as np
import torch

from megatron.core.rerun_state_machine import RerunDataIterator


def _unpack_batch(batch: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """
    Unpacks the packed samples into a list of sub-samples.
    Since each sub-sample may be routed to different DPxCP ranks,
    we unpack the sample here to avoid unnecessarily transferring
    the entire packed sample.
    """
    batch_unpacked = []
    dev = batch[0]["tokens"].device
    original_seq_lens = []
    padded_seq_lens = []
    for sample in batch:
        for key in sample.keys():
            if len(sample[key].shape) == 2:
                # squeeze the redundant batch dimension added by
                # default collate_fn in pytorch dataloader
                # we need a custom collate_fn for THD to avoid this
                # current THD does not support micro_batch_size > 1 due to sft_dataset.py and
                # data_loader in data_samples.py
                sample[key] = sample[key].squeeze(0)
        for sub_sample in range(sample["cu_seqlens"].shape[0] - 1):
            sub_sample_dict = {}
            start_idx = sample["cu_seqlens"][sub_sample]
            end_idx = sample["cu_seqlens"][sub_sample + 1]
            if end_idx - start_idx == 0:
                continue
            for key in ["tokens", "labels", "loss_mask", "position_ids"]:
                sub_sample_dict[key] = sample[key][start_idx:end_idx]
            # Since sft_dataset.py does not provide cu_seqlens_original,
            # we assume original_seq_len equals padded_seq_len here.
            # Ideally the dataset should define the pre-padding seq_len.
            seq_len = (end_idx - start_idx).item()
            original_seq_lens.append(seq_len)
            padded_seq_lens.append(seq_len)
            batch_unpacked.append(sub_sample_dict)

    # Single H2D transfer for all seq lens
    original_seq_lens_cuda = torch.tensor(original_seq_lens, device=dev)
    padded_seq_lens_cuda = torch.tensor(padded_seq_lens, device=dev)
    for i, sub_sample_dict in enumerate(batch_unpacked):
        sub_sample_dict["original_seq_len"] = original_seq_lens_cuda[i : i + 1]
        sub_sample_dict["padded_seq_len"] = padded_seq_lens_cuda[i : i + 1]

    return batch_unpacked


def _get_global_seqlens_and_ids(subsample_seqlens: torch.Tensor, dp_group):
    """
    Gathers the sequence lengths of all subsamples from all DP ranks and calculates global IDs.
    """
    # Collect the number of subsamples from all ranks
    num_local_subsamples = subsample_seqlens.shape[0]
    local_len = torch.tensor([num_local_subsamples], dtype=torch.int32).cuda()
    dp_subsample_count = [torch.zeros_like(local_len) for _ in range(dp_group.size())]
    torch.distributed.all_gather(dp_subsample_count, local_len, group=dp_group)

    # Find the max number of subsamples across all ranks and pad subsample_seqlens to max length
    dp_subsample_counts = torch.stack(dp_subsample_count, dim=0).cpu().view(-1)
    max_sub_samples = int(dp_subsample_counts.max().item())

    if num_local_subsamples < max_sub_samples:
        subsample_seqlens_padded = torch.cat(
            [
                subsample_seqlens,
                torch.zeros(max_sub_samples - num_local_subsamples, dtype=torch.int32).cuda(),
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
    global_ids = torch.arange(len(seqlens_gathered), dtype=torch.int32).cuda()

    # Create a list of (global_id, seqlen) tuples for scheduling
    global_id_seqlens = [(i, seqlens_gathered[i]) for i in range(len(global_ids))]

    # Get the global IDs locally present on this rank
    start_idx = offsets[dp_rank]
    end_idx = offsets[dp_rank + 1]

    global_ids_this_rank = global_ids[start_idx:end_idx]

    return global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered


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


def broadcast_to_pp_group(
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
    Before this broadcast, the new_samples on middle PP stages are None,
    after this broadcast, the new_samples on middle PP stages contain the metadata but
    without tokens, labels, loss_mask, position_ids.
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
            info_to_broadcast = torch.cat(tensor_list, dim=0).to(device=dev, dtype=torch.float32)
            info_length_tensor = torch.tensor(info_to_broadcast.shape[0], dtype=torch.int32).cuda()
            broadcast_tensor(info_length_tensor, pp_src_rank, pp_group)
            broadcast_tensor(info_to_broadcast, pp_src_rank, pp_group)
        else:
            info_length_tensor = torch.tensor(0, dtype=torch.int32).cuda()
            broadcast_tensor(info_length_tensor, pp_src_rank, pp_group)
            info_to_broadcast = torch.empty(info_length_tensor.item(), dtype=torch.float32).cuda()
            broadcast_tensor(info_to_broadcast, pp_src_rank, pp_group)
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
                    cu_seqlens_list.append(info_to_broadcast[indices[i * 2] : indices[i * 2 + 1]])
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


def create_data_iterator(new_samples, pp_group, tp_group, config):
    """Handle virtual pipeline parallelism."""
    if (
        config.virtual_pipeline_model_parallel_size is not None
        and config.virtual_pipeline_model_parallel_size > 1
    ):
        vpp_size = config.virtual_pipeline_model_parallel_size
        if tp_group.rank() == 0:
            if pp_group.rank() == 0 or pp_group.rank() == pp_group.size() - 1:
                metadata = [
                    {k: sample[k] for k in ["max_seqlen", "cu_seqlens", "cu_seqlens_padded"]}
                    for sample in new_samples
                ]
                if pp_group.rank() == 0:
                    new_data_iterator = [RerunDataIterator(iter(new_samples))] + [
                        RerunDataIterator(iter(metadata)) for _ in range(vpp_size - 1)
                    ]
                else:
                    new_data_iterator = [
                        RerunDataIterator(iter(metadata)) for _ in range(vpp_size - 1)
                    ] + [RerunDataIterator(iter(new_samples))]
            else:
                # on middle PP stages, the new_samples are the metadata
                metadata = new_samples
                new_data_iterator = [RerunDataIterator(iter(metadata)) for _ in range(vpp_size)]
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
            else torch.empty(1, device=torch.cuda.current_device(), dtype=batch[0][key].dtype)
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

    # in sft_dataset.py, sequences are already packed before rescheduling,
    # so we need to unpack them here and repack after rescheduling.
    # This is only to adapt to the current megatron-lm sft_dataset.
    # If you implement your own dataset, just have __getitem__ return List[Dict]
    # and this step can be skipped.
    batch = _unpack_batch(batch)

    subsample_seqlens = torch.cat([sample["padded_seq_len"] for sample in batch]).to(
        dtype=torch.int32, device=torch.cuda.current_device()
    )

    global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered = (
        _get_global_seqlens_and_ids(subsample_seqlens, dp_group)
    )

    return batch, global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered
