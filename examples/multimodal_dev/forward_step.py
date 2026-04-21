# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Forward step, TP broadcast, and loss for multimodal_dev training."""

from functools import partial
from typing import Any, Dict, Iterator

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)
from megatron.core.utils import unwrap_model


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
        [len(tensor.shape) if tensor is not None else 0],
        dtype=torch.long,
        device=device,
    )
    torch.distributed.broadcast(ndim, src, group=group)

    if ndim.item() == 0:
        return None

    if tensor is not None:
        shape_tensor = torch.tensor(
            list(tensor.shape), dtype=torch.long, device=device,
        )
        dtype_id = torch.tensor(
            [_dtype_to_id(tensor.dtype)],
            dtype=torch.long,
            device=device,
        )
    else:
        shape_tensor = torch.zeros(
            ndim.item(), dtype=torch.long, device=device,
        )
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
        key_len = torch.tensor(
            [len(key_bytes)], dtype=torch.long, device=device,
        )
    else:
        key_len = torch.zeros(1, dtype=torch.long, device=device)
        keys = []

    torch.distributed.broadcast(key_len, src, group=group)

    if get_tensor_model_parallel_rank() == 0:
        key_tensor = torch.tensor(
            list(key_bytes), dtype=torch.uint8, device=device,
        )
    else:
        key_tensor = torch.zeros(
            key_len.item(), dtype=torch.uint8, device=device,
        )

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
            tensor if isinstance(tensor, torch.Tensor) else None,
            src, group, device,
        )

    return result


# -------------------------------------------------------------------
# THD (packed sequence) helpers
# -------------------------------------------------------------------

def _build_packed_seq_params(
    seq_lengths: torch.Tensor, device: torch.device,
) -> PackedSeqParams:
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
    cu_seqlens = torch.zeros(
        lengths_t.numel() + 1, dtype=torch.int32, device=device,
    )
    torch.cumsum(lengths_t, dim=0, out=cu_seqlens[1:])
    max_seqlen = int(lengths_t.max().item())
    return _build_packed_seq_params_from_cu_seqlens(
        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
    )


def _build_packed_seq_params_from_cu_seqlens(
    cu_seqlens: torch.Tensor, max_seqlen: int,
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


def _extract_valid_mask_and_lengths(
    batch: Dict[str, Any], seq_len: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve per-token validity mask and per-sample lengths.

    Priority:
    1) ``batch["cu_seqlens"]`` from data pipeline (preferred).
    2) ``batch["attention_mask"]``.
    3) Full-length fallback.
    """
    B = batch["input_ids"].shape[0]
    cu_seqlens = batch.get("cu_seqlens", None)
    if cu_seqlens is not None:
        cs = cu_seqlens.to(device=device, dtype=torch.int32)
        if cs.dim() == 2 and cs.shape[0] == B:
            if cs.shape[1] != 2:
                raise ValueError(
                    "Multi-segment cu_seqlens in [B, N] format is not supported yet; "
                    f"expected [B, 2], got [B, {cs.shape[1]}]"
                )
            # Per-sample cu_seqlens, usually [B, 2] => [[0, L_i], ...].
            seq_lengths = cs[:, 1] - cs[:, 0]
        elif cs.dim() == 1 and cs.numel() == B + 1:
            # Already a packed cumulative vector for this microbatch.
            seq_lengths = cs[1:] - cs[:-1]
        else:
            raise ValueError(
                f"Unsupported cu_seqlens shape {tuple(cs.shape)} for batch size {B}"
            )
        seq_lengths = seq_lengths.clamp(min=0, max=seq_len)
        arange_s = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_mask = arange_s < seq_lengths.unsqueeze(1)
        cu_for_pack = torch.zeros(
            seq_lengths.numel() + 1, dtype=torch.int32, device=device,
        )
        torch.cumsum(seq_lengths, dim=0, out=cu_for_pack[1:])
        return valid_mask, seq_lengths, cu_for_pack

    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        am = attention_mask.to(device=device)
        if am.dim() > 2:
            am = am.any(dim=-1)
            if am.dim() == 3:
                am = am.squeeze(1)
        valid_mask = am.to(dtype=torch.bool)
        seq_lengths = valid_mask.sum(dim=1, dtype=torch.int32)
        cu_for_pack = torch.zeros(
            seq_lengths.numel() + 1, dtype=torch.int32, device=device,
        )
        torch.cumsum(seq_lengths, dim=0, out=cu_for_pack[1:])
        return valid_mask, seq_lengths, cu_for_pack

    seq_lengths = torch.full(
        (B,), seq_len, dtype=torch.int32, device=device,
    )
    valid_mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)
    cu_for_pack = torch.arange(
        0, (B + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device,
    )
    return valid_mask, seq_lengths, cu_for_pack


def _pack_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Pack a ``[B, S]`` batch into ``[1, T]`` THD format.

    Concatenates valid tokens from each sample (stripping padding),
    builds ``PackedSeqParams``, and stores it in
    ``batch["packed_seq_params"]``.

    Args:
        batch: Dict with ``input_ids [B, S]``, ``position_ids [3, B, S]``,
               ``labels [B, S]``, ``loss_mask [B, S]``, and optionally
               ``attention_mask [B, S]``.

    Returns:
        Mutated *batch* dict with packed tensors.
    """
    input_ids = batch["input_ids"]  # [B, S]
    _, S = input_ids.shape
    device = input_ids.device

    valid_mask, seq_lengths, cu_for_pack = _extract_valid_mask_and_lengths(
        batch, seq_len=S, device=device,
    )
    is_full_length = bool(torch.all(seq_lengths == S).item())

    # Pack input_ids: [B, S] -> [1, T]
    if is_full_length:
        batch["input_ids"] = input_ids.reshape(1, -1)
    else:
        batch["input_ids"] = input_ids[valid_mask].unsqueeze(0)

    # Pack labels: [B, S] -> [1, T]
    if batch.get("labels") is not None:
        labels = batch["labels"]
        if is_full_length:
            batch["labels"] = labels.reshape(1, -1)
        else:
            batch["labels"] = labels[valid_mask].unsqueeze(0)

    # Pack loss_mask: [B, S] -> [1, T]
    if batch.get("loss_mask") is not None:
        loss_mask = batch["loss_mask"]
        if is_full_length:
            batch["loss_mask"] = loss_mask.reshape(1, -1)
        else:
            batch["loss_mask"] = loss_mask[valid_mask].unsqueeze(0)

    # Pack position_ids: [3, B, S] -> [3, 1, T] or [B, S] -> [1, T]
    if batch.get("position_ids") is not None:
        pos = batch["position_ids"]
        if pos.dim() == 3 and pos.shape[0] == 3:
            # MRoPE: [3, B, S] -> [3, 1, T]
            if is_full_length:
                batch["position_ids"] = pos.reshape(3, 1, -1)
            else:
                packed_pos = [pos[d][valid_mask] for d in range(3)]
                batch["position_ids"] = torch.stack(packed_pos, dim=0).unsqueeze(1)
        else:
            # Standard: [B, S] -> [1, T]
            if is_full_length:
                batch["position_ids"] = pos.reshape(1, -1)
            else:
                batch["position_ids"] = pos[valid_mask].unsqueeze(0)

    # THD uses cu_seqlens; attention_mask is no longer needed.
    batch["attention_mask"] = None

    max_seqlen = batch.get("max_seqlen", None)
    if isinstance(max_seqlen, torch.Tensor):
        max_seqlen = int(max_seqlen.reshape(-1).max().item())
    elif max_seqlen is not None:
        max_seqlen = int(max_seqlen)
    else:
        max_seqlen = int(seq_lengths.max().item())
    batch["packed_seq_params"] = _build_packed_seq_params_from_cu_seqlens(
        cu_for_pack, max_seqlen=max_seqlen,
    )
    # These are consumed at data side; keep only packed metadata.
    batch.pop("cu_seqlens", None)
    batch.pop("cu_seqlens_padded", None)
    batch.pop("max_seqlen", None)
    return batch


# -------------------------------------------------------------------
# get_batch
# -------------------------------------------------------------------

def get_batch(data_iterator: Iterator[Dict[str, Any]]):
    """Get a batch from *data_iterator* and broadcast across TP ranks."""
    device = "cuda"

    if get_tensor_model_parallel_rank() == 0:
        try:
            data = next(data_iterator)
            has_data = torch.tensor(
                [1], dtype=torch.uint8, device=device,
            )
        except StopIteration:
            has_data = torch.tensor(
                [0], dtype=torch.uint8, device=device,
            )
            data = None
    else:
        has_data = torch.empty(1, dtype=torch.uint8, device=device)
        data = None

    src = get_tensor_model_parallel_src_rank()
    group = get_tensor_model_parallel_group()
    torch.distributed.broadcast(has_data, src, group=group)

    if has_data.item() == 0:
        return None

    batch = broadcast_data_batch(data, device=device)

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

    if (
        "image_grid_thw" in batch
        and batch["image_grid_thw"] is not None
    ):
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
    reporting_loss = torch.cat(
        [total_loss.clone().detach().view(1), total_tokens.view(1)],
    )

    return (total_loss, total_tokens, {"lm loss": reporting_loss})


# -------------------------------------------------------------------
# Forward step
# -------------------------------------------------------------------

def forward_step(data_iterator, model):
    """Forward step for multimodal_dev training."""
    batch = get_batch(data_iterator)

    if batch is None:
        return None, None

    # Compute position_ids before packing (MRoPE needs [B, S] input_ids).
    position_ids = batch.get("position_ids", None)
    if position_ids is None:
        inner = unwrap_model(model)
        if hasattr(inner, "compute_position_ids"):
            position_ids = inner.compute_position_ids(
                input_ids=batch["input_ids"],
                image_grid_thw=batch.get("image_grid_thw", None),
            )
            batch["position_ids"] = position_ids

    # Pack sequences into THD format if enabled.
    # Lazy import avoids potential import cycles during Megatron init.
    from megatron.training import get_args
    args = get_args()
    packed_seq_params = None
    if getattr(args, "use_packed_sequence", False):
        batch = _pack_batch(batch)
        packed_seq_params = batch.pop("packed_seq_params")
        position_ids = batch["position_ids"]
        if getattr(args, "sequence_parallel", False):
            raise NotImplementedError(
                "multimodal_dev THD packed sequences currently do not support "
                "sequence_parallel safely; disable --sequence-parallel or "
                "--use-packed-sequence."
            )

    pixel_values = batch.get("pixel_values", None)
    if (
        pixel_values is not None
        and pixel_values.is_floating_point()
        and pixel_values.dtype == torch.float32
    ):
        pixel_values = pixel_values.bfloat16()

    output_tensor = model(
        input_ids=batch["input_ids"],
        position_ids=position_ids,
        attention_mask=batch.get("attention_mask", None),
        labels=batch.get("labels", None),
        loss_mask=batch.get("loss_mask", None),
        pixel_values=pixel_values,
        image_grid_thw=batch.get("image_grid_thw", None),
        packed_seq_params=packed_seq_params,
    )

    loss_mask = batch.get("loss_mask", None)
    if loss_mask is None:
        loss_mask = torch.ones_like(
            batch["input_ids"], dtype=torch.float,
        )

    return output_tensor, partial(loss_func, loss_mask)
