# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Forward step, TP broadcast, and loss for multimodal_dev training."""

from functools import partial
from typing import Any, Dict, Iterator

import torch

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

    position_ids = batch.get("position_ids", None)
    if position_ids is None:
        inner = unwrap_model(model)
        if hasattr(inner, "compute_position_ids"):
            position_ids = inner.compute_position_ids(
                input_ids=batch["input_ids"],
                image_grid_thw=batch.get("image_grid_thw", None),
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
    )

    loss_mask = batch.get("loss_mask", None)
    if loss_mask is None:
        loss_mask = torch.ones_like(
            batch["input_ids"], dtype=torch.float,
        )

    return output_tensor, partial(loss_func, loss_mask)
