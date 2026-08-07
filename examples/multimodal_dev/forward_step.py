# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Forward step, TP broadcast, and loss for multimodal_dev training."""

import math
from functools import partial
from itertools import accumulate
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn.functional as F

from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)
from megatron.training import get_args

# -------------------------------------------------------------------
# dtype <-> int mapping for cross-rank broadcast
# -------------------------------------------------------------------

_DTYPE_MAP = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.bool: 5,
}
_ID_MAP = {v: k for k, v in _DTYPE_MAP.items()}


def per_segment_alignment(cp_size: int, tp_size: int, sequence_parallel: bool) -> int:
    """Padding multiple each individual THD segment must satisfy.

    The CP load-balanced split needs 2*CP chunks per segment and the
    sequence-parallel scatter needs a TP multiple. Data sources that
    predict this packer's physical cost import this rule rather than
    restating it, so the two can never drift apart.
    """
    if cp_size > 1:
        return 2 * cp_size * (tp_size if sequence_parallel else 1)
    return tp_size if sequence_parallel else 1


def _cp_local_target_multiple(cp_size: int, tp_size: int, sequence_parallel: bool) -> int:
    """Divisibility the CP-local packed target length must satisfy.

    Distinct from :func:`per_segment_alignment` (it drops the CP factor);
    the two must never be merged.
    """
    return (2 if cp_size > 1 else 1) * (tp_size if sequence_parallel else 1)


def _dtype_to_id(dtype):
    return _DTYPE_MAP.get(dtype, 0)


def _id_to_dtype(id_val):
    return _ID_MAP.get(id_val, torch.float32)


# -------------------------------------------------------------------
# Tensor broadcast helper
# -------------------------------------------------------------------


def _broadcast_tensor(tensor, src, group, device):
    """Broadcast a single tensor from *src* to all ranks in *group*."""
    ndim = torch.tensor(
        [len(tensor.shape) if tensor is not None else 0], dtype=torch.long, device=device
    )
    torch.distributed.broadcast(ndim, src, group=group)

    if ndim.item() == 0:
        return None

    if tensor is not None:
        shape_tensor = torch.tensor(list(tensor.shape), dtype=torch.long, device=device)
        dtype_id = torch.tensor([_dtype_to_id(tensor.dtype)], dtype=torch.long, device=device)
    else:
        shape_tensor = torch.zeros(ndim.item(), dtype=torch.long, device=device)
        dtype_id = torch.zeros(1, dtype=torch.long, device=device)

    torch.distributed.broadcast(shape_tensor, src, group=group)
    torch.distributed.broadcast(dtype_id, src, group=group)

    dtype = _id_to_dtype(dtype_id.item())
    shape = tuple(shape_tensor.tolist())

    if tensor is None:
        tensor = torch.empty(shape, dtype=dtype, device=device)
    # Zero-element tensors (e.g. pixel_values of a text-only microbatch) are
    # fully described by the shape broadcast above; skip the data collective.
    if tensor.numel() > 0:
        torch.distributed.broadcast(tensor, src, group=group)
    return tensor


# -------------------------------------------------------------------
# Batch broadcast across TP ranks
# -------------------------------------------------------------------


def _stage_batch_for_broadcast(data, device):
    """Move a source batch onto *device* and reject un-encodable dtypes.

    Belongs to the source's PROTECTED region, before the pack-status handshake:
    ``broadcast_data_batch`` otherwise stages and encodes dtypes inside its
    per-key loop, so a staging OOM or unmapped dtype raises between payload
    collectives with the peers already blocked — exactly the hang the handshake
    exists to prevent.
    """
    staged = {}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            staged[key] = value
            continue
        if value.dtype not in _DTYPE_MAP:
            raise TypeError(
                f"batch key '{key}' has dtype {value.dtype}, which the TP broadcast "
                f"cannot encode (known: {sorted(str(d) for d in _DTYPE_MAP)}); falling "
                "back would desynchronize the wire size. Add it to _DTYPE_MAP / _ID_MAP."
            )
        staged[key] = value.to(device)
    return staged


def broadcast_data_batch(data, device="cuda"):
    """Broadcast a data-batch dict from TP rank 0 to all TP ranks.

    Source tensors arrive staged (:func:`_stage_batch_for_broadcast`); the
    ``.to(device)`` below only covers direct callers such as tests.
    """
    src = get_tensor_model_parallel_src_rank()
    group = get_tensor_model_parallel_group()

    if data is None:
        data = {}

    if get_tensor_model_parallel_rank() == 0:
        keys = list(data.keys())
        key_str = ",".join(keys)
        key_bytes = key_str.encode("utf-8")
        key_len = torch.tensor([len(key_bytes)], dtype=torch.long, device=device)
    else:
        key_len = torch.zeros(1, dtype=torch.long, device=device)
        keys = []

    torch.distributed.broadcast(key_len, src, group=group)

    if get_tensor_model_parallel_rank() == 0:
        key_tensor = torch.tensor(list(key_bytes), dtype=torch.uint8, device=device)
    else:
        key_tensor = torch.zeros(key_len.item(), dtype=torch.uint8, device=device)

    torch.distributed.broadcast(key_tensor, src, group=group)

    if get_tensor_model_parallel_rank() != 0:
        key_str = bytes(key_tensor.cpu().tolist()).decode("utf-8")
        keys = key_str.split(",") if key_str else []

    result = {}
    for key in keys:
        tensor = data.get(key, None) if data else None
        if tensor is not None and isinstance(tensor, torch.Tensor):
            tensor = tensor.to(device)
        result[key] = _broadcast_tensor(
            tensor if isinstance(tensor, torch.Tensor) else None, src, group, device
        )

    return result


# -------------------------------------------------------------------
# Pack-status handshake across TP ranks
# -------------------------------------------------------------------

_PACK_ERROR_MSG_MAX_BYTES = 2048


def _propagate_pack_status(is_src, error, device="cuda"):
    """Synchronize the source rank's fetch/pack/validation outcome across TP ranks.

    This converts source-only failures — data-fetch errors in
    :func:`_fetch_batch_with_status` (a DataLoader worker crash, a dataset
    ``__getitem__`` error) as well as pack/validation failures (a ``seq_lens``
    sum mismatch, BSHD multi-segment or over-length rejects, any future pack
    error) — from a collective hang into a synchronized loud failure on every
    rank: without this handshake the TP source rank would raise before the
    next collective (``has_data`` broadcast or ``broadcast_data_batch``) while
    the peer ranks sit in that collective until the NCCL timeout.

    Every TP rank must call this right after the source-side fetch or packing
    block and before the following broadcast. The source rank broadcasts a
    ``[failed, msg_len]`` int64 header (plus the UTF-8 error message bytes,
    truncated to ``_PACK_ERROR_MSG_MAX_BYTES``, when failed); on failure all
    ranks raise together — the source re-raises the original exception, the
    peers raise a ``RuntimeError`` carrying the source's message.
    """
    if mpu.get_tensor_model_parallel_world_size() == 1:
        if error is not None:
            raise error
        return

    # Same source rank / group as broadcast_data_batch, so the handshake and
    # the payload broadcast are ordered on the same communicator.
    src = mpu.get_tensor_model_parallel_src_rank()
    group = mpu.get_tensor_model_parallel_group()

    if is_src:
        # The peers are already blocked in the header broadcast, so an
        # exception whose __str__ raises would strand them; degrade to a
        # placeholder message instead. (An allocation failure on the header
        # itself is NOT covered — recovering from that needs a preallocated
        # buffer, since the broadcast still has to happen.)
        try:
            msg_bytes = (
                str(error).encode("utf-8")[:_PACK_ERROR_MSG_MAX_BYTES] if error is not None else b""
            )
        except Exception:
            msg_bytes = b"<unprintable source-rank exception>"
        header = torch.tensor(
            [1 if error is not None else 0, len(msg_bytes)], dtype=torch.int64, device=device
        )
    else:
        header = torch.zeros(2, dtype=torch.int64, device=device)
    torch.distributed.broadcast(header, src, group=group)

    failed, msg_len = int(header[0].item()), int(header[1].item())
    if not failed:
        return

    if is_src:
        msg_tensor = torch.tensor(list(msg_bytes), dtype=torch.uint8, device=device)
    else:
        msg_tensor = torch.zeros(msg_len, dtype=torch.uint8, device=device)
    torch.distributed.broadcast(msg_tensor, src, group=group)

    if is_src:
        raise error
    msg = bytes(msg_tensor.cpu().tolist()).decode("utf-8", errors="replace")
    raise RuntimeError(
        "TP source rank failed while fetching/packing/validating the microbatch: " + msg
    )


# -------------------------------------------------------------------
# THD (packed sequence) helpers
# -------------------------------------------------------------------


def _build_packed_seq_params(seq_lengths: torch.Tensor, device: torch.device) -> PackedSeqParams:
    """Build ``PackedSeqParams`` from per-sample valid sequence lengths.

    Args:
        seq_lengths: ``[B]`` valid token counts per sample.
        device: Target device for cu_seqlens tensors.

    Returns:
        A ``PackedSeqParams`` instance with ``qkv_format='thd'``.
    """
    if not isinstance(seq_lengths, torch.Tensor):
        seq_lengths = torch.tensor(seq_lengths)
    lengths_t = seq_lengths.to(device=device, dtype=torch.int32)
    cu_seqlens = torch.zeros(lengths_t.numel() + 1, dtype=torch.int32, device=device)
    torch.cumsum(lengths_t, dim=0, out=cu_seqlens[1:])
    max_seqlen = int(lengths_t.max().item())
    return _build_packed_seq_params_from_cu_seqlens(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)


def _build_packed_seq_params_from_cu_seqlens(
    cu_seqlens: torch.Tensor, max_seqlen: int
) -> PackedSeqParams:
    """Build ``PackedSeqParams`` from packed cumulative sequence lengths.

    ``cu_seqlens`` must already be on the target compute device.
    """
    cs = cu_seqlens.to(dtype=torch.int32)
    total_tokens = int(cs[-1].item())
    return PackedSeqParams(
        cu_seqlens_q=cs,
        cu_seqlens_kv=cs,
        cu_seqlens_q_padded=cs,
        cu_seqlens_kv_padded=cs,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
        total_tokens=total_tokens,
    )


def _check_vision_patch_budget(
    pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, args
) -> None:
    """Fail fast when a microbatch's vision payload exceeds configured caps.

    The vision tower consumes the whole microbatch payload in one packed
    forward whose attention workspace scales stepwise with total raw patches;
    exceeding the memory envelope otherwise surfaces as an opaque CUDA OOM
    deep in backward. Checked identically on every TP rank on the broadcasted
    batch, so all ranks fail fast together with the same rich error message
    (source-side pack failures before the broadcast are covered separately by
    :func:`_propagate_pack_status`).

    Callers must invoke this AFTER the payload broadcast so every TP rank
    evaluates the identical batch and raises the same rich error natively,
    with no message re-encoding through the pack-status handshake.
    """
    max_total = getattr(args, "max_vision_patches_per_microbatch", None)
    max_per_image = getattr(args, "max_vision_patches_per_image", None)
    if max_total is None and max_per_image is None:
        return
    num_images = int(image_grid_thw.shape[0]) if image_grid_thw.dim() == 2 else 0
    total_patches = int(pixel_values.shape[0])
    if max_total is not None and total_patches > int(max_total):
        raise ValueError(
            f"Microbatch vision payload has {total_patches} raw patches across "
            f"{num_images} image(s), exceeding --max-vision-patches-per-microbatch="
            f"{int(max_total)}. Reduce the image count/size profile or the "
            "per-sample vision budget."
        )
    if max_per_image is not None and num_images:
        patches_per_image = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        worst = int(patches_per_image.max().item())
        if worst > int(max_per_image):
            worst_index = int(patches_per_image.argmax().item())
            raise ValueError(
                f"Image {worst_index} with grid {image_grid_thw[worst_index].tolist()} has "
                f"{worst} raw patches, exceeding --max-vision-patches-per-image="
                f"{int(max_per_image)}."
            )


def _segment_bounds(seq_lens: Optional[torch.Tensor], sample_len: int) -> list[tuple[int, int]]:
    """Per-segment (start, end) bounds of a sample's token axis.

    ``seq_lens`` is the optional per-sample segment-length vector emitted by
    the packed_document dataset; ``None`` means one segment spanning the
    whole sample.
    """
    if seq_lens is None:
        return [(0, sample_len)]
    lengths = [int(length) for length in seq_lens.tolist()]
    if any(length <= 0 for length in lengths):
        raise ValueError(f"seq_lens must be positive, got {lengths}.")
    if sum(lengths) != sample_len:
        raise ValueError(
            f"seq_lens sum {sum(lengths)} does not match the sample length {sample_len}."
        )
    ends = list(accumulate(lengths))
    return list(zip([0] + ends[:-1], ends))


def _append_cu_boundary(cu_seqlens: Optional[torch.Tensor], end: int) -> Optional[torch.Tensor]:
    """Append one cumulative sequence boundary on the existing device."""
    if cu_seqlens is None:
        return None
    boundary = torch.full((1,), int(end), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    return torch.cat((cu_seqlens, boundary), dim=0)


def _pad_multimodal_thd_batch(
    packed_batch: Dict[str, Any],
    packed_seq_params: PackedSeqParams,
    *,
    pad_alignment: Any,
    max_seqlen_per_dp_cp_rank: Optional[int],
    pad_by_appending_dummy_seq: bool,
    cp_size: int,
    tp_size: int,
    sequence_parallel: bool,
) -> tuple[Dict[str, Any], PackedSeqParams]:
    """Pad a pre-CP multimodal THD batch using CP-local alignment semantics.

    The core scheduler pads tensors after CP partitioning, but the multimodal
    path packs the complete global token buffer before model-side CP slicing.
    Resolve the requested target in CP-local coordinates, convert it back to a
    global target, and represent the physical tail as one ordinary dummy THD
    sequence.  The dummy logical and padded lengths are kept equal even when
    real samples already contain inter-sequence alignment padding.
    """
    if not pad_by_appending_dummy_seq:
        raise ValueError(
            "multimodal packed THD represents the physical tail as one dummy "
            "THD sequence whenever --pad-packed-seq-alignment is enabled; this "
            "is an implementation invariant of the packed path — disabling it "
            "(e.g. via the auto-generated "
            "--no-pad-packed-seq-by-appending-dummy-seq switch) is not "
            "supported"
        )

    global_actual = int(packed_batch["input_ids"].shape[-1])
    metadata_actual = int(packed_seq_params.cu_seqlens_q_padded[-1].item())
    if metadata_actual != global_actual:
        raise ValueError(
            "multimodal packed THD metadata does not cover the token buffer: "
            f"cu_seqlens_q_padded[-1]={metadata_actual}, tokens={global_actual}"
        )
    if global_actual % cp_size != 0:
        raise ValueError(f"global packed length {global_actual} must be divisible by CP={cp_size}")

    local_actual = global_actual // cp_size
    parallel_multiple = _cp_local_target_multiple(cp_size, tp_size, sequence_parallel)

    if pad_alignment == "max":
        if max_seqlen_per_dp_cp_rank is None:
            raise ValueError(
                "--max-seqlen-per-dp-cp-rank is required when " "--pad-packed-seq-alignment=max"
            )
        local_target = int(max_seqlen_per_dp_cp_rank)
        if local_target % parallel_multiple != 0:
            raise ValueError(
                f"CP-local packed target {local_target} must be divisible by "
                f"the CP/SP alignment {parallel_multiple}"
            )
    else:
        if isinstance(pad_alignment, bool):
            raise ValueError("--pad-packed-seq-alignment must be a positive integer or 'max'")
        user_alignment = int(pad_alignment)
        if user_alignment <= 0:
            raise ValueError("--pad-packed-seq-alignment must be a positive integer or 'max'")
        effective_alignment = math.lcm(user_alignment, parallel_multiple)
        local_target = math.ceil(local_actual / effective_alignment) * effective_alignment

    if local_target < local_actual:
        raise ValueError(
            f"CP-local packed length {local_actual} exceeds target {local_target}; "
            "increase --max-seqlen-per-dp-cp-rank"
        )

    global_target = local_target * cp_size
    physical_tail = global_target - global_actual
    if physical_tail == 0:
        packed_seq_params.pad_between_seqs = False
        return packed_batch, packed_seq_params

    if cp_size > 1 and physical_tail % (2 * cp_size) != 0:
        raise ValueError(f"dummy THD tail {physical_tail} must be divisible by 2*CP={2 * cp_size}")

    packed_batch["input_ids"] = F.pad(packed_batch["input_ids"], (0, physical_tail), value=0)
    packed_batch["labels"] = F.pad(packed_batch["labels"], (0, physical_tail), value=-100)
    packed_batch["loss_mask"] = F.pad(packed_batch["loss_mask"], (0, physical_tail), value=0)
    padding_tail = torch.ones(
        (*packed_batch["padding_mask"].shape[:-1], physical_tail),
        dtype=torch.bool,
        device=packed_batch["padding_mask"].device,
    )
    packed_batch["padding_mask"] = torch.cat((packed_batch["padding_mask"], padding_tail), dim=-1)

    logical_q_end = int(packed_seq_params.cu_seqlens_q[-1].item()) + physical_tail
    logical_kv_end = int(packed_seq_params.cu_seqlens_kv[-1].item()) + physical_tail
    packed_seq_params = PackedSeqParams(
        qkv_format=packed_seq_params.qkv_format,
        cu_seqlens_q=_append_cu_boundary(packed_seq_params.cu_seqlens_q, logical_q_end),
        cu_seqlens_kv=_append_cu_boundary(packed_seq_params.cu_seqlens_kv, logical_kv_end),
        cu_seqlens_q_padded=_append_cu_boundary(
            packed_seq_params.cu_seqlens_q_padded, global_target
        ),
        cu_seqlens_kv_padded=_append_cu_boundary(
            packed_seq_params.cu_seqlens_kv_padded, global_target
        ),
        max_seqlen_q=max(int(packed_seq_params.max_seqlen_q), physical_tail),
        max_seqlen_kv=max(int(packed_seq_params.max_seqlen_kv), physical_tail),
        local_cp_size=packed_seq_params.local_cp_size,
        cp_group=packed_seq_params.cp_group,
        total_tokens=global_target,
        pad_between_seqs=False,
        cp_partition_mode=packed_seq_params.cp_partition_mode,
    )
    return packed_batch, packed_seq_params


def pack_or_pad_batch(
    batch: Optional[list[Dict[str, Any]]],
    use_packed_sequence: bool = False,
    seq_length: Optional[int] = None,
    device="cuda",
) -> Dict[str, Any]:
    """Pack or pad a ``[B, S]`` batch into ``[1, T]`` THD or ``[B, S]`` BSHD.

    Must be invoked on every TP rank. On the TP source rank ``batch`` is
    the per-sample dict list from the dataset; on other TP ranks ``batch``
    may be ``None`` (the function relies on the trailing TP broadcast to
    distribute results). All metadata needed to reconstruct
    ``PackedSeqParams`` (``cu_seqlens``, ``cu_seqlens_padded``,
    ``max_seqlen``, ``total_tokens``) is broadcast alongside the data, so
    every rank can build an identical ``PackedSeqParams`` on its own.

    Source-side pack/validation errors (e.g. a ``seq_lens`` sum mismatch or a
    BSHD multi-segment/over-length reject) are propagated to every TP rank
    via a status handshake (:func:`_propagate_pack_status`) before the
    payload broadcast, so all ranks raise together instead of the peers
    hanging in the collective.
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    is_src = mpu.get_tensor_model_parallel_rank() == 0

    # SP is an explicit runtime option; TP>1 does not imply SP is enabled.
    # get_args() itself raises in test contexts where megatron globals are
    # not initialised.
    try:
        args = get_args()
    except AssertionError:
        args = None
    has_sp = bool(getattr(args, "sequence_parallel", False))

    divisible_by = per_segment_alignment(cp_size, tp_size, has_sp)

    if use_packed_sequence:
        packed_batch: Dict[str, Any] = {}

        pack_error: Optional[Exception] = None
        if is_src:
            try:
                assert batch is not None, "source TP rank must provide a batch"
                input_ids_list, labels_list, loss_mask_list = [], [], []
                pixel_values_list, image_grid_thw_list = [], []
                seqlens_list, seqlens_padded_list = [], []

                for sample in batch:
                    sample_len = sample["input_ids"].shape[0]
                    assert (
                        sample["labels"].shape
                        == sample["input_ids"].shape
                        == sample["loss_mask"].shape
                    ), "labels, input_ids, and loss_mask must have the same shape"
                    # A sample may carry multiple document segments (a packed
                    # window). Each segment becomes its own logical sequence: it is
                    # spliced into cu_seqlens and padded independently to the
                    # CP/SP alignment, so the physical layout always matches
                    # cu_seqlens_padded. Vision payloads stay sample-level:
                    # placeholder order inside the tokens already matches the
                    # pixel row order.
                    bounds = _segment_bounds(sample.get("seq_lens"), sample_len)
                    for start, end in bounds:
                        seqlen = end - start
                        target_len = math.ceil(seqlen / divisible_by) * divisible_by
                        input_ids_list.append(
                            F.pad(sample["input_ids"][start:end], (0, target_len - seqlen), value=0)
                        )
                        labels_list.append(
                            F.pad(sample["labels"][start:end], (0, target_len - seqlen), value=-100)
                        )
                        loss_mask_list.append(
                            F.pad(sample["loss_mask"][start:end], (0, target_len - seqlen), value=0)
                        )
                        seqlens_list.append(seqlen)
                        seqlens_padded_list.append(target_len)
                    pixel_values_list.append(sample["pixel_values"])
                    image_grid_thw_list.append(sample["image_grid_thw"])

                cu_seqlens = list(accumulate(seqlens_list, initial=0))
                cu_seqlens_padded = list(accumulate(seqlens_padded_list, initial=0))

                # padding_mask: True at collate-padded positions within each packed
                # sample. Real tokens occupy [cu_seqlens_padded[i], +seqlens_list[i]);
                # the tail up to cu_seqlens_padded[i+1] is padding. Consumed by MoE
                # routing in megatron.core to exclude padded tokens from aux loss,
                # z-loss, and expert-bias accumulation.
                total_tokens_padded = cu_seqlens_padded[-1]
                padding_mask_thd = torch.zeros(total_tokens_padded, dtype=torch.bool)
                for i, real_seqlen in enumerate(seqlens_list):
                    pad_start = cu_seqlens_padded[i] + real_seqlen
                    pad_end = cu_seqlens_padded[i + 1]
                    if pad_end > pad_start:
                        padding_mask_thd[pad_start:pad_end] = True

                packed_batch["input_ids"] = torch.concat(input_ids_list, dim=0).unsqueeze(0)
                packed_batch["labels"] = torch.concat(labels_list, dim=0).unsqueeze(0)
                packed_batch["loss_mask"] = torch.concat(loss_mask_list, dim=0).unsqueeze(0)
                packed_batch["padding_mask"] = padding_mask_thd.unsqueeze(0)
                packed_batch["pixel_values"] = torch.concat(pixel_values_list)
                packed_batch["image_grid_thw"] = torch.concat(image_grid_thw_list)
                # cu_seqlens / cu_seqlens_padded need to reach non-source TP ranks
                # so each rank can build an identical PackedSeqParams.
                packed_batch["cu_seqlens"] = torch.tensor(
                    cu_seqlens, dtype=torch.int32, device=device
                )
                packed_batch["cu_seqlens_padded"] = torch.tensor(
                    cu_seqlens_padded, dtype=torch.int32, device=device
                )
                # Verdict before staging: moving an over-budget payload to the
                # device is itself the OOM this guard exists to pre-empt.
                _check_vision_patch_budget(
                    packed_batch["pixel_values"], packed_batch["image_grid_thw"], args
                )
                packed_batch = _stage_batch_for_broadcast(packed_batch, device)
            except Exception as exc:
                pack_error = exc

        _propagate_pack_status(is_src, pack_error, device=device)
        packed_batch = broadcast_data_batch(packed_batch, device=device)

        cu_seqlens_t = packed_batch.pop("cu_seqlens")
        cu_seqlens_padded_t = packed_batch.pop("cu_seqlens_padded")
        # Derive max_seqlen / total_tokens from the (broadcast) cu_seqlens —
        # no extra collective needed.
        max_seqlen_q = int((cu_seqlens_padded_t[1:] - cu_seqlens_padded_t[:-1]).max().item())
        total_tokens = int(cu_seqlens_padded_t[-1].item())

        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_t,
            cu_seqlens_kv=cu_seqlens_t,
            cu_seqlens_q_padded=cu_seqlens_padded_t,
            cu_seqlens_kv_padded=cu_seqlens_padded_t,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_q,
            total_tokens=total_tokens,
            pad_between_seqs=False,
        )

        # Same predicate as pretrain_multimodal.validate_entry_args, which
        # rejects this at startup; None means unset on both sides.
        if getattr(args, "cuda_graph_impl", "none") not in (None, "none"):
            raise ValueError(
                "pretrain_multimodal packed THD does not yet support CUDA Graph; "
                "the graph path currently requires the core sequence-packing scheduler"
            )

        pad_alignment = getattr(args, "pad_packed_seq_alignment", None)
        if pad_alignment is not None:
            packed_batch, packed_seq_params = _pad_multimodal_thd_batch(
                packed_batch,
                packed_seq_params,
                pad_alignment=pad_alignment,
                max_seqlen_per_dp_cp_rank=getattr(args, "max_seqlen_per_dp_cp_rank", None),
                pad_by_appending_dummy_seq=getattr(
                    args, "pad_packed_seq_by_appending_dummy_seq", True
                ),
                cp_size=cp_size,
                tp_size=tp_size,
                sequence_parallel=has_sp,
            )

        packed_batch["packed_seq_params"] = packed_seq_params
        return packed_batch

    # ---------- padded (BSHD) branch ----------
    assert seq_length is not None, "seq_length must be provided when use_packed_sequence is False"
    padded_batch: Dict[str, Any] = {}

    pack_error: Optional[Exception] = None
    if is_src:
        try:
            assert batch is not None, "source TP rank must provide a batch"
            if any(
                sample.get("seq_lens") is not None and sample["seq_lens"].numel() > 1
                for sample in batch
            ):
                raise ValueError(
                    "Multi-segment packed samples require --use-packed-sequence "
                    "(THD); the padded BSHD layout has no segment representation."
                )
            max_seqlens = max(x["input_ids"].shape[0] for x in batch)
            if max_seqlens > seq_length:
                # F.pad with a negative pad would silently truncate the token
                # stream while pixel_values/image_grid_thw keep every image,
                # desynchronizing vision payloads from their placeholder tokens.
                raise ValueError(
                    f"A sample of length {max_seqlens} exceeds the --seq-length cap "
                    f"{seq_length}. The padded BSHD path never truncates. The "
                    "fixed-shape providers (mock, cord_v2) size their samples from "
                    "--total-seq-length while the packer caps at --seq-length, so "
                    "the usual cause is --total-seq-length > --seq-length; otherwise "
                    "the dataset provider is emitting over-length samples."
                )
            target_seqlens = min(max_seqlens, seq_length)
            # Round target seqlen up to the parallelism alignment factor so the
            # batched tensor is divisible for CP (+SP) splitting downstream.
            if divisible_by > 1:
                target_seqlens = math.ceil(target_seqlens / divisible_by) * divisible_by

            # Capture real lengths before in-place padding so we can build a
            # padding_mask for MoE routing (True at collate-padded positions).
            real_seqlens = [s["input_ids"].shape[0] for s in batch]

            for sample in batch:
                sample["input_ids"] = F.pad(
                    sample["input_ids"], (0, target_seqlens - sample["input_ids"].shape[0]), value=0
                )
                sample["labels"] = F.pad(
                    sample["labels"], (0, target_seqlens - sample["labels"].shape[0]), value=-100
                )
                sample["loss_mask"] = F.pad(
                    sample["loss_mask"], (0, target_seqlens - sample["loss_mask"].shape[0]), value=0
                )

            padded_batch["input_ids"] = torch.concat(
                [x["input_ids"].unsqueeze(0) for x in batch], dim=0
            )
            padded_batch["labels"] = torch.concat([x["labels"].unsqueeze(0) for x in batch], dim=0)
            padded_batch["loss_mask"] = torch.concat(
                [x["loss_mask"].unsqueeze(0) for x in batch], dim=0
            )
            # Keep None as the known-no-padding fast path for MoE routing.
            has_padding = any(real_seqlen < target_seqlens for real_seqlen in real_seqlens)
            if has_padding:
                positions = torch.arange(target_seqlens).unsqueeze(0)
                padded_batch["padding_mask"] = positions >= torch.tensor(real_seqlens).unsqueeze(1)
            padded_batch["pixel_values"] = torch.concat([x["pixel_values"] for x in batch])
            padded_batch["image_grid_thw"] = torch.concat([x["image_grid_thw"] for x in batch])
            # See the packed branch: verdict, then stage, both protected.
            _check_vision_patch_budget(
                padded_batch["pixel_values"], padded_batch["image_grid_thw"], args
            )
            padded_batch = _stage_batch_for_broadcast(padded_batch, device)
        except Exception as exc:
            pack_error = exc

    _propagate_pack_status(is_src, pack_error, device=device)
    padded_batch = broadcast_data_batch(padded_batch, device=device)
    return padded_batch


# -------------------------------------------------------------------
# get_batch
# -------------------------------------------------------------------


def _fetch_batch_with_status(
    data_iterator: Optional[Iterator[list[Dict[str, Any]]]], device="cuda"
):
    """Fetch the next batch on the TP source rank and synchronize the outcome.

    ``StopIteration`` is a normal end-of-data signal, not an error: it maps to
    ``has_data = 0`` with no failure handshake, and every rank returns
    ``(None, False)`` after the ``has_data`` broadcast. Any OTHER source-side
    fetch failure (a DataLoader worker crash, a dataset ``__getitem__`` error)
    is propagated to every TP rank via :func:`_propagate_pack_status` BEFORE
    the ``has_data`` broadcast, so all ranks raise together instead of the
    peers hanging in that collective until the NCCL timeout.

    Returns ``(data, has_data)``: the fetched sample list (``None`` on
    non-source ranks and at end of data) and whether this step carries data.
    """
    data = None
    fetch_error: Optional[Exception] = None
    is_src = get_tensor_model_parallel_rank() == 0
    if is_src:
        try:
            data = next(data_iterator)
            has_data = torch.tensor([1], dtype=torch.uint8, device=device)
        except StopIteration:
            has_data = torch.tensor([0], dtype=torch.uint8, device=device)
        except Exception as exc:
            fetch_error = exc
            has_data = None  # unreachable past the handshake below
    else:
        has_data = torch.empty(1, dtype=torch.uint8, device=device)

    # Synchronize the fetch outcome BEFORE the has_data broadcast; on failure
    # every rank raises here (source re-raises the original exception).
    _propagate_pack_status(is_src, fetch_error, device=device)

    src = get_tensor_model_parallel_src_rank()
    group = get_tensor_model_parallel_group()
    torch.distributed.broadcast(has_data, src, group=group)
    return data, bool(has_data.item())


def get_batch(data_iterator: Iterator[list[Dict[str, Any]]]):
    """Get a batch from *data_iterator* and broadcast across TP ranks."""
    device = "cuda"
    args = get_args()

    data, has_data = _fetch_batch_with_status(data_iterator, device=device)
    if not has_data:
        return None

    # Because broadcast will not broadcast packed_seq_params, we move it into pack_or_pad_batch
    batch = pack_or_pad_batch(data, args.use_packed_sequence, args.seq_length, device=device)

    # Fix shapes produced by default_collate.
    if "position_ids" in batch and batch["position_ids"] is not None:
        p = batch["position_ids"]
        if p.dim() == 3 and p.shape[1] == 3:
            batch["position_ids"] = p.permute(1, 0, 2).contiguous()

    if "pixel_values" in batch and batch["pixel_values"] is not None:
        pv = batch["pixel_values"]
        if pv.dim() == 3:
            B, P, D = pv.shape
            batch["pixel_values"] = pv.reshape(B * P, D)

    if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
        g = batch["image_grid_thw"]
        if g.dim() == 3:
            batch["image_grid_thw"] = g.squeeze(1)

    return batch


# -------------------------------------------------------------------
# Loss
# -------------------------------------------------------------------


def loss_func(loss_mask, output_tensor):
    """Compute masked language model loss."""
    losses = output_tensor.float()
    loss_mask = loss_mask.contiguous().view(-1).float()

    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])

    return (total_loss, total_tokens, {"lm loss": reporting_loss})


# -------------------------------------------------------------------
# Forward step
# -------------------------------------------------------------------


def forward_step(data_iterator, model):
    """Forward step for multimodal_dev training."""
    batch = get_batch(data_iterator)

    if batch is None:
        return None, None

    pixel_values = batch.get("pixel_values", None)
    if (
        pixel_values is not None
        and pixel_values.is_floating_point()
        and pixel_values.dtype == torch.float32
    ):
        pixel_values = pixel_values.bfloat16()

    # We don't provide position_ids, now. Let model handle it itself.
    output_tensor = model(
        input_ids=batch["input_ids"],
        position_ids=batch.get("position_ids"),
        attention_mask=batch.get("attention_mask", None),
        labels=batch.get("labels", None),
        loss_mask=batch.get("loss_mask", None),
        padding_mask=batch.get("padding_mask", None),
        pixel_values=pixel_values,
        image_grid_thw=batch.get("image_grid_thw", None),
        packed_seq_params=batch.get("packed_seq_params", None),
    )

    loss_mask = batch.get("loss_mask", None)
    if loss_mask is None:
        loss_mask = torch.ones_like(batch["input_ids"], dtype=torch.float)

    # Slice loss_mask the same way the model sliced its inputs, so the
    # mask aligns with the CP-shard output.  Delegated to MultimodalModel
    # so the slicing rule lives in one place.
    from examples.multimodal_dev.models.base import MultimodalModel

    loss_mask = MultimodalModel.cp_split_loss_mask(loss_mask, batch.get("packed_seq_params", None))

    return output_tensor, partial(loss_func, loss_mask)
