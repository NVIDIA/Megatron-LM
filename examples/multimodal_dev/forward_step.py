# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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
    torch.distributed.broadcast(tensor, src, group=group)
    return tensor


# -------------------------------------------------------------------
# Batch broadcast across TP ranks
# -------------------------------------------------------------------


def broadcast_data_batch(data, device="cuda"):
    """Broadcast a data-batch dict from TP rank 0 to all TP ranks."""
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
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    is_src = mpu.get_tensor_model_parallel_rank() == 0

    # SP is an explicit runtime option; TP>1 does not imply SP is enabled.
    # get_args() itself raises in test contexts where megatron globals are
    # not initialised.
    try:
        has_sp = bool(getattr(get_args(), "sequence_parallel", False))
    except AssertionError:
        has_sp = False

    if cp_size > 1:
        divisible_by = (tp_size * cp_size * 2) if has_sp else (cp_size * 2)
    else:
        divisible_by = tp_size if has_sp else 1

    if use_packed_sequence:
        packed_batch: Dict[str, Any] = {}

        if is_src:
            assert batch is not None, "source TP rank must provide a batch"
            input_ids_list, labels_list, loss_mask_list = [], [], []
            pixel_values_list, image_grid_thw_list = [], []
            seqlens_list, seqlens_padded_list = [], []

            for sample in batch:
                seqlen = sample["input_ids"].shape[0]
                assert (
                    sample["labels"].shape == sample["input_ids"].shape == sample["loss_mask"].shape
                ), "labels, input_ids, and loss_mask must have the same shape"
                target_len = math.ceil(seqlen / divisible_by) * divisible_by
                input_ids_list.append(F.pad(sample["input_ids"], (0, target_len - seqlen), value=0))
                labels_list.append(F.pad(sample["labels"], (0, target_len - seqlen), value=-100))
                loss_mask_list.append(F.pad(sample["loss_mask"], (0, target_len - seqlen), value=0))
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
            packed_batch["cu_seqlens"] = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            packed_batch["cu_seqlens_padded"] = torch.tensor(
                cu_seqlens_padded, dtype=torch.int32, device=device
            )

        packed_batch = broadcast_data_batch(packed_batch, device=device)

        cu_seqlens_t = packed_batch.pop("cu_seqlens")
        cu_seqlens_padded_t = packed_batch.pop("cu_seqlens_padded")
        # Derive max_seqlen / total_tokens from the (broadcast) cu_seqlens —
        # no extra collective needed.
        max_seqlen_q = int((cu_seqlens_padded_t[1:] - cu_seqlens_padded_t[:-1]).max().item())
        total_tokens = int(cu_seqlens_padded_t[-1].item())

        packed_batch["packed_seq_params"] = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_t,
            cu_seqlens_kv=cu_seqlens_t,
            cu_seqlens_q_padded=cu_seqlens_padded_t,
            cu_seqlens_kv_padded=cu_seqlens_padded_t,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_q,
            total_tokens=total_tokens,
        )
        return packed_batch

    # ---------- padded (BSHD) branch ----------
    assert seq_length is not None, "seq_length must be provided when use_packed_sequence is False"
    padded_batch: Dict[str, Any] = {}

    if is_src:
        assert batch is not None, "source TP rank must provide a batch"
        max_seqlens = max(x["input_ids"].shape[0] for x in batch)
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
        positions = torch.arange(target_seqlens).unsqueeze(0)
        padded_batch["padding_mask"] = positions >= torch.tensor(real_seqlens).unsqueeze(1)
        padded_batch["pixel_values"] = torch.concat([x["pixel_values"] for x in batch])
        padded_batch["image_grid_thw"] = torch.concat([x["image_grid_thw"] for x in batch])

    return broadcast_data_batch(padded_batch, device=device)


# -------------------------------------------------------------------
# get_batch
# -------------------------------------------------------------------


def get_batch(data_iterator: Iterator[list[Dict[str, Any]]]):
    """Get a batch from *data_iterator* and broadcast across TP ranks."""
    device = "cuda"
    args = get_args()

    if get_tensor_model_parallel_rank() == 0:
        try:
            data = next(data_iterator)
            has_data = torch.tensor([1], dtype=torch.uint8, device=device)
        except StopIteration:
            has_data = torch.tensor([0], dtype=torch.uint8, device=device)
            data = None
    else:
        has_data = torch.empty(1, dtype=torch.uint8, device=device)
        data = None

    src = get_tensor_model_parallel_src_rank()
    group = get_tensor_model_parallel_group()
    torch.distributed.broadcast(has_data, src, group=group)

    if has_data.item() == 0:
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
