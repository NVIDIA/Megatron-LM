# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Forward step, TP broadcast, and loss for multimodal_dev training."""

from functools import partial
from typing import Any, Dict, Iterator
import math

import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)
from megatron.core import mpu
from itertools import accumulate
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




def pack_or_pad_batch(batch: list[Dict[str, Any]], use_packed_sequence: bool=False, seq_length: int=None) -> list[Dict[str, Any]]:
    """Pack or pad a ``[B, S]`` batch into ``[1, T]`` THD format."""
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    
    divisible_by = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    # NOTE: don't consider fp8 padding now
    
    if use_packed_sequence:
        input_ids_list, labels_list, loss_mask_list, pixel_values_list, image_grid_thw_list = [], [], [], [], []
        seqlens_list, seqlens_padded_list = [], []

        # NOTE: for attention_mask, we don't use attention mask
        #       for position_ids, let model handle it itself
        #       we don't cut input id, althrough it exceeds seq_length

        packed_batch = dict()

        for sample in batch:
            seqlen = sample["input_ids"].shape[0]
            assert sample["labels"].shape == sample["input_ids"].shape == sample["loss_mask"].shape, "labels, input_ids, and loss_mask must have the same shape"
            target_len = math.ceil(seqlen / divisible_by) * divisible_by
            input_ids = F.pad(sample["input_ids"], (0, target_len - seqlen), value=0)
            labels = F.pad(sample["labels"], (0, target_len - seqlen), value=-100)
            loss_mask = F.pad(sample["loss_mask"], (0, target_len - seqlen), value=0)

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            loss_mask_list.append(loss_mask)
            seqlens_list.append(seqlen)
            seqlens_padded_list.append(target_len)
            pixel_values_list.append(sample["pixel_values"])
            image_grid_thw_list.append(sample["image_grid_thw"])

        cu_seqlens = list(accumulate(seqlens_list, initial=0))
        cu_seqlens_padded = list(accumulate(seqlens_padded_list, initial=0))

        packed_batch["input_ids"] = torch.concat(input_ids_list, dim=0).unsqueeze(0)
        packed_batch["labels"] = torch.concat(labels_list, dim=0).unsqueeze(0)
        packed_batch["loss_mask"] = torch.concat(loss_mask_list, dim=0).unsqueeze(0)

        # TODO, maybe pixel_values's seqlens needs to be recorded. 
        packed_batch["pixel_values"] = torch.concat(pixel_values_list)
        packed_batch["image_grid_thw"] = torch.concat(image_grid_thw_list)

        packed_batch["packed_seq_params"] = PackedSeqParams(
            cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32),
            cu_seqlens_kv=torch.tensor(cu_seqlens, dtype=torch.int32),
            cu_seqlens_q_padded=torch.tensor(cu_seqlens_padded, dtype=torch.int32),
            cu_seqlens_kv_padded=torch.tensor(cu_seqlens_padded, dtype=torch.int32),
            max_seqlen_q=max(seqlens_padded_list),
            max_seqlen_kv=max(seqlens_padded_list),
        )
        return packed_batch
    else:
        assert seq_length is not None, "seq_length must be provided when use_packed_sequence is False"
        max_seqlens = max([x["input_ids"].shape[0] for x in batch])
        target_seqlens = min(max_seqlens, seq_length)
        padded_batch = dict()
        
        for sample in batch:
            sample["input_ids"] = F.pad(sample["input_ids"], (0, target_seqlens - sample["input_ids"].shape[0]), value=0)
            sample["labels"] = F.pad(sample["labels"], (0, target_seqlens - sample["labels"].shape[0]), value=-100)
            sample["loss_mask"] = F.pad(sample["loss_mask"], (0, target_seqlens - sample["loss_mask"].shape[0]), value=0)

        padded_batch["input_ids"] = torch.concat([x["input_ids"].unsqueeze(0) for x in batch], dim=0)
        padded_batch["labels"] = torch.concat([x["labels"].unsqueeze(0) for x in batch], dim=0)
        padded_batch["loss_mask"] = torch.concat([x["loss_mask"].unsqueeze(0) for x in batch], dim=0)
        padded_batch["pixel_values"] = torch.concat([x["pixel_values"] for x in batch])
        padded_batch["image_grid_thw"] = torch.concat([x["image_grid_thw"] for x in batch])
        return padded_batch


# -------------------------------------------------------------------
# get_batch
# -------------------------------------------------------------------

def get_batch(data_iterator: Iterator[Dict[str, Any]]):
    """Get a batch from *data_iterator* and broadcast across TP ranks."""
    device = "cuda"
    args = get_args()

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

    data = pack_or_pad_batch(data, args.use_packed_sequence, args.seq_length)
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
        pixel_values=pixel_values,
        image_grid_thw=batch.get("image_grid_thw", None),
        packed_seq_params=batch.get("packed_seq_params", None),
    )

    loss_mask = batch.get("loss_mask", None)
    if loss_mask is None:
        loss_mask = torch.ones_like(
            batch["input_ids"], dtype=torch.float,
        )

    return output_tensor, partial(loss_func, loss_mask)
