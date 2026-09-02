# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from functools import lru_cache
from math import ceil, log2
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from megatron.core.parallel_state import get_valid_dynamic_context_parallel_group_sizes
from megatron.core.rerun_state_machine import RerunDataIterator

_DYNAMIC_CP_WORKLOAD_CAP_DELTA = 0.05

# Dynamic CP rerouting currently owns the text-only GPT/SFT sample schema.
# Multimodal metadata needs explicit element-layout and routing semantics
# before it can be added here.
_REROUTE_KEY_ORDER = (
    "tokens",
    "labels",
    "loss_mask",
    "position_ids",
    "original_seq_len",
    "padded_seq_len",
)
_REROUTE_KEY_SET = frozenset(_REROUTE_KEY_ORDER)
_REROUTE_SCALAR_KEYS = frozenset(("original_seq_len", "padded_seq_len"))


def _unpack_batch(batch: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """
    Unpacks the packed samples into a list of sub-samples.
    Since each sub-sample may be routed to different DPxCP ranks,
    we unpack the sample here to avoid unnecessarily transferring
    the entire packed sample.

    Two mutually exclusive input shapes are accepted, and every sample in
    ``batch`` must use the same one:

      * **Pre-packed** (e.g. :class:`SFTDataset`): each sample carries a
        ``cu_seqlens`` tensor and the tokens of multiple sub-samples
        concatenated together. We slice them apart and synthesize
        ``original_seq_len`` / ``padded_seq_len`` from the cu_seqlens deltas.

      * **Already unpacked** (e.g. :class:`VarlenDataset`): each sample is a
        single sub-sample that already carries ``padded_seq_len`` (and
        usually ``original_seq_len``). We just normalize the leading batch
        dimension introduced by the default collate_fn and return as-is.

    The shape is decided once for the whole batch and asserted per sample, so a
    dataset that emits both keys cannot silently bypass the ``cu_seqlens``
    slicing below.
    """
    if not batch:
        return batch

    # Pick the input shape from the first sample, then validate every sample
    # against it and normalize the collate dimension in the same pass.
    is_unpacked = "padded_seq_len" in batch[0]
    for i, sample in enumerate(batch):
        assert ("padded_seq_len" in sample) == is_unpacked, (
            f"_unpack_batch got a mixed batch: sample {i} and sample 0 disagree on "
            "whether they carry 'padded_seq_len' (already unpacked) or not (pre-packed)."
        )
        assert ("cu_seqlens" in sample) != is_unpacked, (
            f"_unpack_batch: sample {i} must carry exactly one of 'padded_seq_len' "
            "(already unpacked, e.g. VarlenDataset) or 'cu_seqlens' (pre-packed, "
            "e.g. SFTDataset)."
        )
        for key, value in sample.items():
            if value.ndim == 2:
                # Drop the redundant batch dimension added by the default
                # collate_fn in the pytorch dataloader. squeeze(0) is a silent
                # no-op when the leading dimension is not 1, so assert on it
                # instead of slicing along the batch dimension further down.
                # The packing path installs an identity collate_fn (see
                # build_pretraining_data_loader), which never adds this
                # dimension in the first place and therefore supports
                # micro_batch_size > 1; the default collate_fn only works here
                # with micro_batch_size == 1.
                assert value.shape[0] == 1, (
                    f"_unpack_batch got '{key}' with shape {tuple(value.shape)}; the "
                    "packed-sequence path needs one sub-sample per collated entry. Use "
                    "micro_batch_size 1 with the default collate_fn, or an identity "
                    "collate_fn."
                )
                sample[key] = value.squeeze(0)

    # Short-circuit for datasets that already emit one sub-sample per index.
    if is_unpacked:
        for sample in batch:
            if "original_seq_len" not in sample:
                sample["original_seq_len"] = sample["padded_seq_len"].clone()
        return batch

    batch_unpacked = []
    device = batch[0]["tokens"].device
    original_seq_lens = []
    padded_seq_lens = []
    for sample in batch:
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
    original_seq_lens_cuda = torch.tensor(original_seq_lens, device=device)
    padded_seq_lens_cuda = torch.tensor(padded_seq_lens, device=device)
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
    samples: List,
    padded_lengths: torch.Tensor,
    original_lengths: torch.Tensor,
    local_cp_size: Optional[torch.Tensor],
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
    if local_cp_size is not None:
        new_sample["local_cp_size"] = local_cp_size

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
    is_dynamic_cp: bool = False,
):
    """
    Broadcast num_micro_batches, seqlen_sum_this_global_batch,
    seqlen_squared_sum_this_global_batch and metadata to middle PP stages.
    Before this broadcast, the new_samples on middle PP stages are None,
    after this broadcast, the new_samples on middle PP stages contain the metadata but
    without tokens, labels, loss_mask, position_ids.

    Who needs what:

      * **PP rank 0 and the last PP rank** both own a data iterator (only TP rank 0
        on the first and last PP stage does), so both run the whole schedule ->
        reroute -> pack pipeline on the same input samples and independently end up
        with complete ``new_samples``: tokens, labels, loss_mask, position_ids *and*
        the packing metadata. Neither takes anything from this broadcast; the last
        stage in particular must keep its own labels / loss_mask.
      * **Middle PP stages** have no data iterator, so ``new_samples`` is None on
        entry. They only need the packing metadata (max_seqlen / cu_seqlens /
        cu_seqlens_padded) to rebuild the packed-sequence params, never the token
        tensors.

    The last PP rank still takes part in the transfer because
    ``torch.distributed.broadcast`` is a collective over ``pp_group``: every member
    has to call it or the group deadlocks. It therefore receives the payload and
    drops it, which is what the ``pp_group.rank() != pp_group.size() - 1`` guard
    below implements. Filtering it out of the transfer itself would require a
    separate "first + middle" process group, which is not worth an extra process
    group for a payload of a few hundred bytes per global batch.
    """

    pp_src_rank = torch.distributed.get_process_group_ranks(pp_group)[0]

    # size() > 2 asks "does a middle PP stage exist at all": with 1 or 2 PP ranks
    # every rank is a first and/or last stage and already owns its packed samples,
    # so there is nobody to broadcast to.
    if pp_group.size() > 2:
        if pp_group.rank() == 0:
            cu_seqlens_lengths = torch.tensor(
                [sample["cu_seqlens"].numel() for sample in new_samples],
                dtype=torch.float32,
                device=dev,
            )
            cu_seqlens_padded_lengths = torch.tensor(
                [sample["cu_seqlens_padded"].numel() for sample in new_samples],
                dtype=torch.float32,
                device=dev,
            )
            tensor_list = [
                torch.tensor(
                    [
                        num_micro_batches,
                        seqlen_sum_this_global_batch,
                        seqlen_squared_sum_this_global_batch,
                    ],
                    dtype=torch.float32,
                    device=dev,
                )
            ]
            for sample in new_samples:
                tensor_list.append(sample["max_seqlen"].reshape(1))
            if is_dynamic_cp:
                for sample in new_samples:
                    tensor_list.append(sample["local_cp_size"].reshape(1))
            tensor_list.append(cu_seqlens_lengths)
            tensor_list.append(cu_seqlens_padded_lengths)
            for sample in new_samples:
                tensor_list.append(sample["cu_seqlens"])
                tensor_list.append(sample["cu_seqlens_padded"])
            info_to_broadcast = torch.cat(tensor_list, dim=0).to(device=dev, dtype=torch.float32)
            info_length_tensor = torch.tensor(
                info_to_broadcast.shape[0], dtype=torch.int32, device=dev
            )
            broadcast_tensor(info_length_tensor, pp_src_rank, pp_group)
            broadcast_tensor(info_to_broadcast, pp_src_rank, pp_group)
        else:
            # Every non-source rank has to take part in the collective, including
            # the last PP stage.
            info_length_tensor = torch.tensor(0, dtype=torch.int32, device=dev)
            broadcast_tensor(info_length_tensor, pp_src_rank, pp_group)
            info_to_broadcast = torch.empty(
                info_length_tensor.item(), dtype=torch.float32, device=dev
            )
            broadcast_tensor(info_to_broadcast, pp_src_rank, pp_group)
            if pp_group.rank() != pp_group.size() - 1:
                # Middle PP stages receive the broadcasted info and unpack it.
                # Cu-seqlens lengths are encoded explicitly so zero values inside
                # the payload cannot be mistaken for tensor boundaries.
                # The last PP stage deliberately falls through: it built its own
                # new_samples from its own data iterator (with the labels and
                # loss_mask this payload does not carry), so it discards what it
                # just received rather than overwriting them.
                num_micro_batches = int(info_to_broadcast[0].item())
                seqlen_sum_this_global_batch = info_to_broadcast[1].item()
                seqlen_squared_sum_this_global_batch = info_to_broadcast[2].item()

                cursor = 3
                max_seqlens = info_to_broadcast[cursor : cursor + num_micro_batches]
                cursor += num_micro_batches
                if is_dynamic_cp:
                    local_cp_sizes = info_to_broadcast[cursor : cursor + num_micro_batches].to(
                        torch.int32
                    )
                    cursor += num_micro_batches
                else:
                    local_cp_sizes = None
                cu_seqlens_lengths = info_to_broadcast[cursor : cursor + num_micro_batches].to(
                    torch.int64
                )
                cursor += num_micro_batches
                cu_seqlens_padded_lengths = info_to_broadcast[
                    cursor : cursor + num_micro_batches
                ].to(torch.int64)
                cursor += num_micro_batches

                new_samples = []
                for i in range(num_micro_batches):
                    cu_seqlens_len = int(cu_seqlens_lengths[i].item())
                    cu_seqlens_padded_len = int(cu_seqlens_padded_lengths[i].item())
                    new_sample = {}
                    new_sample["max_seqlen"] = max_seqlens[i].to(torch.int32)
                    if is_dynamic_cp:
                        new_sample["local_cp_size"] = local_cp_sizes[i]
                    new_sample["cu_seqlens"] = info_to_broadcast[
                        cursor : cursor + cu_seqlens_len
                    ].to(torch.int32)
                    cursor += cu_seqlens_len
                    new_sample["cu_seqlens_padded"] = info_to_broadcast[
                        cursor : cursor + cu_seqlens_padded_len
                    ].to(torch.int32)
                    cursor += cu_seqlens_padded_len
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


def create_data_iterator(new_samples, pp_group, tp_group, config, is_dynamic_cp: bool = False):
    """Handle virtual pipeline parallelism."""
    if (
        config.virtual_pipeline_model_parallel_size is not None
        and config.virtual_pipeline_model_parallel_size > 1
    ):
        vpp_size = config.virtual_pipeline_model_parallel_size
        if tp_group.rank() == 0:
            if pp_group.rank() == 0 or pp_group.rank() == pp_group.size() - 1:
                metadata_keys = ["max_seqlen", "cu_seqlens", "cu_seqlens_padded"]
                if is_dynamic_cp:
                    metadata_keys.append("local_cp_size")
                metadata = [
                    {k: sample[k] for k in metadata_keys if k in sample} for sample in new_samples
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
    batch, global_ids_this_rank, global_id_seqlens, sample_id_groups, offsets, dp_group, dp_cp_group
):
    """
    Reroutes the sub-samples to the correct rank after scheduling.

    Each CP lane gathers the samples from its DP group, then keeps only the
    samples assigned to its DPxCP rank. Gathering within ``dp_group`` avoids
    collecting the identical input held by every CP sibling and avoids the
    fully connected P2P transport created by NCCL all-to-all.

    The gather is issued one data key at a time to bound temporary memory to
    one global field. Selected slices are cloned before the next key so the
    full gather buffer can be released.

    This is intentionally a text-only contract. Multimodal fields are rejected
    until their per-token, per-sample, or per-media routing layout is defined.
    """

    dcp_rank = dp_cp_group.rank()
    dp_rank = dp_group.rank()
    dp_size = dp_group.size()

    batch_keys = set(batch[0])
    unsupported_keys = batch_keys - _REROUTE_KEY_SET
    assert not unsupported_keys, (
        "Dynamic CP reroute currently supports only the text sample schema; "
        f"cannot reroute unsupported sample keys {sorted(unsupported_keys)}. "
        "extend _REROUTE_KEY_ORDER and classify their element layout."
    )
    for sample_idx, sample in enumerate(batch[1:], start=1):
        sample_keys = set(sample)
        assert sample_keys == batch_keys, (
            f"Sample {sample_idx} keys {sorted(sample_keys)} do not match sample 0 keys "
            f"{sorted(batch_keys)}."
        )
    data_keys = [key for key in _REROUTE_KEY_ORDER if key in batch_keys]

    offset_values = [int(value) for value in offsets.tolist()]
    assert (
        len(offset_values) == dp_size + 1
    ), f"Expected {dp_size + 1} DP offsets, got {len(offset_values)}."
    local_ids = [int(gid) for gid in global_ids_this_rank.tolist()]
    expected_local_ids = list(range(offset_values[dp_rank], offset_values[dp_rank + 1]))
    assert (
        local_ids == expected_local_ids
    ), f"Local sample IDs {local_ids} do not match DP-rank range {expected_local_ids}."
    assert len(batch) == len(
        local_ids
    ), f"Local batch size {len(batch)} does not match sample-ID count {len(local_ids)}."

    recv_ids = sorted(
        {gid for sample_id_group in sample_id_groups for gid in sample_id_group[dcp_rank]}
    )
    recv_samples = {gid: {key: None for key in data_keys} for gid in recv_ids}
    seq_len_by_gid = dict(global_id_seqlens)

    def _build_layout(is_scalar):
        sample_numels = {
            gid: 1 if is_scalar else int(seq_len_by_gid[gid]) for gid in range(offset_values[-1])
        }
        rank_numels = [
            sum(sample_numels[gid] for gid in range(offset_values[rank], offset_values[rank + 1]))
            for rank in range(dp_size)
        ]
        max_rank_numel = max(rank_numels)

        sample_slices = {}
        for source_rank in range(dp_size):
            cursor = source_rank * max_rank_numel
            for gid in range(offset_values[source_rank], offset_values[source_rank + 1]):
                sample_numel = sample_numels[gid]
                sample_slices[gid] = (cursor, sample_numel)
                cursor += sample_numel

        return rank_numels, max_rank_numel, sample_slices

    layouts = {False: _build_layout(is_scalar=False), True: _build_layout(is_scalar=True)}

    for key in data_keys:
        rank_numels, max_rank_numel, sample_slices = layouts[key in _REROUTE_SCALAR_KEYS]

        local_tensor = torch.cat(
            [
                sample[key].to(torch.cuda.current_device(), non_blocking=True).reshape(-1)
                for sample in batch
            ],
            dim=0,
        )
        assert (
            local_tensor.numel() == rank_numels[dp_rank]
        ), f"Packed {key} has {local_tensor.numel()} elements, expected {rank_numels[dp_rank]}."

        if local_tensor.numel() < max_rank_numel:
            gather_input = local_tensor.new_zeros(max_rank_numel)
            gather_input[: local_tensor.numel()].copy_(local_tensor)
        else:
            gather_input = local_tensor.contiguous()

        if dp_size == 1:
            gathered_tensor = gather_input
        else:
            gathered_tensor = local_tensor.new_empty(dp_size * max_rank_numel)
            torch.distributed.all_gather_into_tensor(gathered_tensor, gather_input, group=dp_group)

        for gid in recv_ids:
            start, sample_numel = sample_slices[gid]
            recv_samples[gid][key] = gathered_tensor[start : start + sample_numel].clone()

    return recv_samples


def build_packed_microbatches(
    samples_this_rank_with_id: Dict[int, Dict[str, torch.Tensor]],
    sample_id_groups: List[List[List[int]]],
    dcp_rank: int,
    dev: torch.device,
    is_dynamic_cp: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """Build this rank's packed samples and attach its runtime CP size."""
    num_micro_batches = len(sample_id_groups)
    seg_starts: List[int] = [0]
    original_lens_tensors = []
    padded_lens_tensors = []

    grouped_samples = [
        [
            samples_this_rank_with_id[sub_sample_id]
            for sub_sample_id in sample_id_groups[i][dcp_rank]
        ]
        for i in range(num_micro_batches)
    ]

    local_cp_sizes = None
    if is_dynamic_cp:
        local_cp_sizes = []
        for i in range(num_micro_batches):
            sample_ids_this_rank = sample_id_groups[i][dcp_rank]
            assert sample_ids_this_rank, "Dynamic CP must assign at least one sample to every rank"
            representative_id = sample_ids_this_rank[0]
            local_cp_sizes.append(
                sum(representative_id in rank_ids for rank_ids in sample_id_groups[i])
            )
        local_cp_sizes = torch.tensor(local_cp_sizes, dtype=torch.int32, device=dev)

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
        local_cp_size = local_cp_sizes[i] if is_dynamic_cp else None
        new_sample = _pack_sequences(samples, lens_padded, lens_original, local_cp_size, dev)
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
        batch (List[Dict[str, torch.Tensor]]): The sub-samples pulled from this rank's
            ``data_iterator`` over ``num_microbatches`` steps, flattened and unpacked
            (see :func:`_unpack_batch`). Every dict carries ``tokens`` / ``labels`` /
            ``loss_mask`` / ``position_ids`` plus the ``original_seq_len`` and
            ``padded_seq_len`` scalars used for scheduling.
        global_id_seqlens (List[Tuple[int, int]]): ``(global_id, padded_seq_len)`` for
            every sub-sample in the DP group, ordered by DP rank and then by local
            index. Identical on all ranks; this is the scheduler's input.
        global_ids_this_rank (torch.Tensor): int32 CUDA tensor holding the global IDs of
            the sub-samples loaded by this rank, i.e. ``batch[i]`` has global ID
            ``global_ids_this_rank[i]``.
        offsets (torch.Tensor): int32 CPU tensor of shape ``[dp_size + 1]`` with the
            exclusive prefix sum of the per-rank sub-sample counts, so DP rank ``r`` owns
            global IDs ``offsets[r]:offsets[r + 1]``. Used by
            :func:`reroute_samples_to_dcp_ranks` to map a global ID back to its source
            rank.
        seqlens_gathered (List[int]): Padded sequence length of every sub-sample in the
            DP group, indexed by global ID (``seqlens_gathered[gid]`` equals
            ``global_id_seqlens[gid][1]``). Handy for global-batch token counts such as
            the FLOPs accounting.
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


# =============================================================================
# Dynamic CP scheduling algorithms (used by DefaultDynamicCPScheduler)
# =============================================================================


def next_hdp_group_packing_aware(
    sample_seqlens: List[Tuple[int, int]],
    total_gpus: int,
    max_seq_len_per_rank: int,
    min_cp_size: int = 1,
    cp_group_sizes: Optional[Sequence[int]] = None,
) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:
    """Form one DCP microbatch with packing-aware CP group selection.

    This differs from the legacy DCP scheduler in two ways:
    1. Short sequences may use a larger CP group than their minimum required
       CP size when that lowers the critical-path rank workload.
    2. Candidate placements are bounded by ``tall * max_seq_len_per_rank``,
       the per-rank workload upper bound for packing sequences no longer than
       the local tallest sequence in the microbatch.

    The scheduler keeps the legacy invariant that each returned microbatch has
    no empty DPxCP rank after the fill step. For non-power-of-two DPxCP layouts,
    it falls back to the full DPxCP group if power-of-two expansion cannot fill
    every rank.
    """
    valid_group_sizes = get_valid_dynamic_context_parallel_group_sizes(total_gpus)
    if cp_group_sizes is None:
        cp_group_sizes = valid_group_sizes
    else:
        cp_group_sizes = sorted(set(cp_group_sizes))
        unsupported_group_sizes = set(cp_group_sizes) - set(valid_group_sizes)
        if unsupported_group_sizes:
            raise ValueError(
                f"Unsupported Dynamic CP group sizes {sorted(unsupported_group_sizes)} for "
                f"DPxCP size {total_gpus}; expected a subset of {valid_group_sizes}"
            )
    if min_cp_size not in cp_group_sizes:
        raise ValueError(
            f"min_cp_size={min_cp_size} is not available; configured Dynamic CP group "
            f"sizes are {list(cp_group_sizes)}"
        )
    enabled_cp_sizes = tuple(size for size in cp_group_sizes if size >= min_cp_size)

    if not sample_seqlens:
        return (
            [[] for _ in range(total_gpus)],
            [],
            [0.0 for _ in range(total_gpus)],
            [[] for _ in range(total_gpus)],
        )

    def cp_min_fn(seq_len: int) -> int:
        return dcp_gpus_needed(
            seq_len, max_seq_len_per_rank, min_cp_size, valid_group_sizes=enabled_cp_sizes
        )

    def workload(seq_len: int, cp_size: int) -> float:
        return (seq_len * seq_len) / cp_size

    sample_seqlens = sorted(sample_seqlens, key=lambda x: x[1], reverse=True)
    local_tall = sample_seqlens[0][1]
    cap = float(local_tall) * float(max_seq_len_per_rank) * (1.0 + _DYNAMIC_CP_WORKLOAD_CAP_DELTA)

    micro_batches: List[List[int]] = [[] for _ in range(total_gpus)]
    exec_times: List[float] = [0.0 for _ in range(total_gpus)]
    sample_ids_per_gpu: List[List[int]] = [[] for _ in range(total_gpus)]
    packing_sequence_len: Dict[int, float] = {}

    gpu_group_id: List[Optional[int]] = [None] * total_gpus
    group_members: Dict[int, List[int]] = {}
    group_size: Dict[int, int] = {}
    next_gid = 0

    sample_id, seq_len = sample_seqlens[0]
    cp_size = cp_min_fn(seq_len)
    assert cp_size <= total_gpus, (
        f"Sequence length {seq_len} requires CP size {cp_size}, "
        f"but only {total_gpus} DPxCP ranks are available."
    )
    group_id = next_gid
    next_gid += 1
    members = list(range(cp_size))
    group_members[group_id] = members
    group_size[group_id] = cp_size
    packing_sequence_len[group_id] = seq_len / cp_size
    per_gpu_cost = workload(seq_len, cp_size)
    for rank in members:
        gpu_group_id[rank] = group_id
        micro_batches[rank].append(seq_len)
        exec_times[rank] += per_gpu_cost
        sample_ids_per_gpu[rank].append(sample_id)

    leftovers: List[Tuple[int, int]] = []
    for sample_id, seq_len in sample_seqlens[1:]:
        min_needed = cp_min_fn(seq_len)
        best = None

        for cp_size in enabled_cp_sizes:
            if cp_size < min_needed:
                continue
            per_gpu_cost = workload(seq_len, cp_size)

            for group_id, size in list(group_size.items()):
                if size != cp_size:
                    continue
                if packing_sequence_len.get(group_id, 0) + seq_len / cp_size > max_seq_len_per_rank:
                    continue
                members = group_members[group_id]
                member_set = set(members)
                projected_max = max(
                    time + per_gpu_cost if rank in member_set else time
                    for rank, time in enumerate(exec_times)
                )
                if projected_max <= cap and (best is None or projected_max < best[0]):
                    best = (projected_max, cp_size, "add", group_id, None)

            free_ranks = [
                rank
                for rank, assigned_group_id in enumerate(gpu_group_id)
                if assigned_group_id is None
            ]
            if len(free_ranks) >= cp_size:
                chosen_members = sorted(free_ranks, key=lambda rank: exec_times[rank])[:cp_size]
                chosen_set = set(chosen_members)
                projected_max = max(
                    time + per_gpu_cost if rank in chosen_set else time
                    for rank, time in enumerate(exec_times)
                )
                if projected_max <= cap and (best is None or projected_max < best[0]):
                    best = (projected_max, cp_size, "new", None, chosen_members)

        if best is None:
            leftovers.append((sample_id, seq_len))
            continue

        _, selected_cp_size, action, group_id, chosen_members = best
        per_gpu_cost = workload(seq_len, selected_cp_size)
        if action == "add":
            members = group_members[group_id]
            packing_sequence_len[group_id] += seq_len / selected_cp_size
            for rank in members:
                micro_batches[rank].append(seq_len)
                exec_times[rank] += per_gpu_cost
                sample_ids_per_gpu[rank].append(sample_id)
        else:
            group_id = next_gid
            next_gid += 1
            group_members[group_id] = chosen_members
            group_size[group_id] = selected_cp_size
            packing_sequence_len[group_id] = seq_len / selected_cp_size
            for rank in chosen_members:
                gpu_group_id[rank] = group_id
                micro_batches[rank].append(seq_len)
                exec_times[rank] += per_gpu_cost
                sample_ids_per_gpu[rank].append(sample_id)

    def fill_empty_gpus_once() -> bool:
        nonlocal micro_batches, exec_times, sample_ids_per_gpu

        empty_ranks = [rank for rank, micro_batch in enumerate(micro_batches) if not micro_batch]
        if not empty_ranks:
            return False
        assert all(
            not micro_batches[rank] for rank in range(empty_ranks[0], total_gpus)
        ), "fill_empty_gpus_once assumes empty ranks are contiguous at the tail"

        existing_group_sizes = set(group_size.values())
        if not existing_group_sizes:
            return False
        min_group_size = min(existing_group_sizes)
        current_group_index = enabled_cp_sizes.index(min_group_size)
        if current_group_index + 1 >= len(enabled_cp_sizes):
            return False
        next_group_size = enabled_cp_sizes[current_group_index + 1]

        for group_id, size in list(group_size.items()):
            if size != min_group_size:
                continue

            members = group_members[group_id]
            needed_count = next_group_size - min_group_size
            group_start_rank = members[0]
            group_end_rank = members[-1]
            empty_rank = empty_ranks[0]
            if group_end_rank + 1 > empty_rank or group_end_rank + needed_count >= total_gpus:
                continue

            work_to_push = micro_batches[group_end_rank + 1 : empty_rank]
            exec_times_to_push = exec_times[group_end_rank + 1 : empty_rank]
            sample_ids_to_push = sample_ids_per_gpu[group_end_rank + 1 : empty_rank]

            new_micro_batches: List[List[int]] = [[] for _ in range(total_gpus)]
            new_exec_times: List[float] = [0.0 for _ in range(total_gpus)]
            new_sample_ids_per_gpu: List[List[int]] = [[] for _ in range(total_gpus)]

            for rank in range(group_start_rank):
                new_micro_batches[rank] = micro_batches[rank]
                new_exec_times[rank] = exec_times[rank]
                new_sample_ids_per_gpu[rank] = sample_ids_per_gpu[rank]

            for rank in range(group_start_rank, group_end_rank + needed_count + 1):
                new_micro_batches[rank] = list(micro_batches[group_end_rank])
                new_exec_times[rank] = sum(
                    workload(length, next_group_size) for length in micro_batches[group_end_rank]
                )
                new_sample_ids_per_gpu[rank] = list(sample_ids_per_gpu[group_end_rank])

            for idx, work in enumerate(work_to_push):
                target_rank = group_end_rank + needed_count + 1 + idx
                new_micro_batches[target_rank] = work
                new_exec_times[target_rank] = exec_times_to_push[idx]
                new_sample_ids_per_gpu[target_rank] = sample_ids_to_push[idx]

            group_size[group_id] = next_group_size
            group_members[group_id] = list(
                range(group_start_rank, group_end_rank + needed_count + 1)
            )
            for other_group_id in list(group_size.keys()):
                if other_group_id == group_id:
                    continue
                if min(group_members[other_group_id]) > group_end_rank:
                    group_members[other_group_id] = [
                        rank + needed_count for rank in group_members[other_group_id]
                    ]

            micro_batches = new_micro_batches
            exec_times = new_exec_times
            sample_ids_per_gpu = new_sample_ids_per_gpu
            return True

        return False

    def fill_with_full_dpxcp_group() -> None:
        nonlocal micro_batches, exec_times, sample_ids_per_gpu, leftovers

        selected: List[Tuple[int, int]] = []
        next_leftovers: List[Tuple[int, int]] = []
        packed_sequence_len = 0.0

        for sample_id, seq_len in sample_seqlens:
            per_rank_len = seq_len / total_gpus
            if packed_sequence_len + per_rank_len <= max_seq_len_per_rank:
                selected.append((sample_id, seq_len))
                packed_sequence_len += per_rank_len
            else:
                next_leftovers.append((sample_id, seq_len))

        assert selected, (
            "At least one sequence should fit in the full DPxCP group; "
            "try to increase 'max-seqlen-per-dp-cp-rank'."
        )

        selected_ids = [sample_id for sample_id, _ in selected]
        selected_lens = [seq_len for _, seq_len in selected]
        per_rank_work = sum(workload(seq_len, total_gpus) for _, seq_len in selected)

        micro_batches = [list(selected_lens) for _ in range(total_gpus)]
        exec_times = [per_rank_work for _ in range(total_gpus)]
        sample_ids_per_gpu = [list(selected_ids) for _ in range(total_gpus)]
        leftovers = next_leftovers

    while any(not micro_batch for micro_batch in micro_batches):
        if not fill_empty_gpus_once():
            fill_with_full_dpxcp_group()
            break

    return micro_batches, leftovers, exec_times, sample_ids_per_gpu


def align_sample_id_groups(sample_id_groups: List, microbatch_group_size_per_vp_stage: int) -> List:
    """Align len(sample_id_groups) to microbatch_group_size_per_vp_stage when VPP is enabled.

    Standalone version extracted from DefaultDynamicCPScheduler.
    """
    multiple = int(microbatch_group_size_per_vp_stage)
    remainder = (-len(sample_id_groups)) % multiple
    i = len(sample_id_groups) - 1

    def split_group(sample_id_group):
        total_hdp_ranks = len(sample_id_group)
        cu_ranks = [0]
        prev_cp_size = 0

        while cu_ranks[-1] != total_hdp_ranks:
            start_rank = cu_ranks[-1]
            sid0 = sample_id_group[start_rank][0]
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
            return None, None

        k = 0
        while cu_ranks[k] < total_hdp_ranks // 2:
            k += 1

        old_mb = sample_id_group[: cu_ranks[k]] + [[] for _ in range(total_hdp_ranks - cu_ranks[k])]
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
                    start_idx = i + 1 + cp_size
                    end_idx = -empty_size + prev_cp_size if -empty_size + prev_cp_size < 0 else None
                    sample_id_group[start_idx + 2 * prev_cp_size : end_idx] = sample_id_group[
                        start_idx + prev_cp_size : -empty_size
                    ]
                    sample_id_group[start_idx + prev_cp_size : start_idx + 2 * prev_cp_size] = (
                        sample_id_group[start_idx : start_idx + prev_cp_size]
                    )
                    break
                elif cp_size <= empty_size and i == -1:
                    end_idx = -empty_size + cp_size if -empty_size + cp_size < 0 else None
                    sample_id_group[2 * cp_size : end_idx] = sample_id_group[cp_size:-empty_size]
                    sample_id_group[cp_size : 2 * cp_size] = sample_id_group[0:cp_size]
                    break
                prev_cp_size = cp_size
            return sample_id_group

        while len(sample_id_group[-1]) == 0:
            sample_id_group = fill_empty(sample_id_group)
        return sample_id_group

    attempts_since_split = 0
    while remainder > 0:
        if i < 0:
            if attempts_since_split >= len(sample_id_groups):
                assert False, 'align_sample_id_groups: no tail microbatch has enough ids to split'
            i = len(sample_id_groups) - 1
        group1, group2 = split_group(sample_id_groups[i])
        if group1 is not None and group2 is not None:
            sample_id_groups[i] = group1
            sample_id_groups.append(group2)
            remainder -= 1
            attempts_since_split = 0
        else:
            attempts_since_split += 1
        i -= 1

    return sample_id_groups


# =============================================================================
# Workload estimation helpers for dynamic CP scheduling
# =============================================================================


@lru_cache(maxsize=128)
def dcp_gpus_needed(
    seq_len: int,
    max_seq_len_per_rank: int,
    min_cp_size: int = 1,
    valid_group_sizes: Optional[Tuple[int, ...]] = None,
) -> int:
    """Return the smallest configured runtime CP group that can hold a sequence."""
    if max_seq_len_per_rank < 1:
        raise ValueError("max_seq_len_per_rank must be positive")

    required_ranks = max(min_cp_size, ceil(seq_len / max_seq_len_per_rank), 1)
    if valid_group_sizes is not None:
        for group_size in valid_group_sizes:
            if group_size >= required_ranks:
                return group_size
        return required_ranks

    raw = max(1, 2 ** ceil(log2(max(required_ranks, 1))))
    return max(min_cp_size, raw)
