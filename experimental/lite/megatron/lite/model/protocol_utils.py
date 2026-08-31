# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Helpers shared by model protocol forward steps.

The verl/runtime layers hand each protocol a raw, model-agnostic ``PackedBatch``
(true per-sequence lengths, no padding, no ``PackedSeqParams``). Each model owns
its pack/unpack pair: ``pack_thd_forward_kwargs`` pads + CP-splits the batch into
model forward kwargs, and ``unpack_thd_forward_output`` reverses a model output
back to jagged true-length form. THD models share the zigzag-CP pair below;
models with a different CP layout (e.g. DeepSeek-V4 contiguous DSA) provide their
own pair.
"""

from __future__ import annotations

from typing import Any

import torch
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.parallel.thd import (
    pack_nested_thd,
    parallel_state_from_model,
    prepare_packed_thd_kwargs_for_context_parallel,
    split_packed_to_cp_local,
    thd_pack_meta,
    unpack_thd_to_nested,
)
from megatron.lite.primitive.utils.packed_seq import PackedSeqParams
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.loss import get_loss_context


def _parallel_state(model) -> ParallelState:
    return parallel_state_from_model(model) or ParallelState()


def router_replay_roots(chunk) -> list:
    """Select decoder layers for R3, excluding MTP-only routers.

    The current rollout configuration does not run MTP, so route tensors have
    one layer-axis entry per main decoder router.  MTP layers therefore keep
    their native routing instead of consuming that rollout-owned axis.  If a
    future rollout enables MTP speculative decoding, this assumption must be
    reevaluated.
    """
    model = getattr(chunk, "model", chunk)
    layers = getattr(model, "layers", None)
    if layers is None:
        return [chunk]
    values = getattr(layers, "values", None)
    return list(values()) if callable(values) else list(layers)


def nested_from_packed(tensor: torch.Tensor | None, seq_lens: torch.Tensor):
    """Split a 1-D packed (true, unpadded) tensor back into a jagged nested tensor."""
    if tensor is None:
        return None
    if tensor.dim() == 2 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 1:
        raise ValueError(f"PackedBatch tensor must be 1-D, got {tuple(tensor.shape)}.")
    pieces = []
    offset = 0
    for length_t in seq_lens:
        length = int(length_t.item())
        pieces.append(tensor.narrow(0, offset, length))
        offset += length
    if offset != tensor.numel():
        raise ValueError(
            f"PackedBatch sizes sum to {offset}, tensor has {tensor.numel()} tokens."
        )
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def pack_thd_forward_kwargs(model, batch: PackedBatch) -> dict[str, Any]:
    """Pad + zigzag-CP-split a raw THD batch into model forward kwargs.

    Pads each sequence to the TE/zigzag alignment, then CP-splits tokens,
    labels, masks and position ids through the shared THD primitive — the same
    layout the model was validated against, now produced inside the protocol
    rather than the connector.
    """
    ps = _parallel_state(model)
    seq_lens = batch.seq_lens
    packed = pack_nested_thd(
        nested_from_packed(batch.input_ids, seq_lens),
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_rank=ps.cp_rank,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
        split_cp=False,
        labels=nested_from_packed(batch.labels, seq_lens),
        roll_labels=batch.labels is not None,
        loss_mask=nested_from_packed(batch.loss_mask, seq_lens),
        roll_loss_mask=batch.loss_mask is not None,
    )
    max_seqlen = (
        int(packed.padded_lengths.max().item()) if packed.padded_lengths.numel() else 0
    )
    # pack_nested_thd already returns [1, T] token rows; do not unsqueeze again.
    kwargs: dict[str, Any] = {
        "input_ids": packed.input_ids,
        "labels": packed.labels,
        "loss_mask": packed.loss_mask,
        "position_ids": packed.position_ids,
        "packed_seq_params": PackedSeqParams.from_cu_seqlens(
            packed.cu_seqlens_padded, max_seqlen=max_seqlen
        ),
    }
    prepare_packed_thd_kwargs_for_context_parallel(model, kwargs)
    return kwargs


def unpack_thd_forward_output(
    model, batch: PackedBatch, output: torch.Tensor
) -> torch.Tensor:
    """Reverse a zigzag-CP THD model output back to jagged true-length form."""
    ps = _parallel_state(model)
    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
    )
    return unpack_thd_to_nested(output, meta, contiguous=False)


def pack_routed_experts(
    model, batch: PackedBatch, routed_experts, *, contiguous: bool = False
) -> list[torch.Tensor]:
    """Pack jagged ``[batch, seq, layers, topk]`` R3 routes for local routers."""

    ps = _parallel_state(model)
    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
    )
    rows = (
        list(routed_experts.unbind(0))
        if getattr(routed_experts, "is_nested", False)
        else [routed_experts[i] for i in range(routed_experts.size(0))]
    )
    if len(rows) != int(meta.lengths.numel()):
        raise ValueError(
            f"routed_experts has {len(rows)} sequences, batch has {int(meta.lengths.numel())}."
        )
    if not rows or rows[0].dim() != 3:
        shape = None if not rows else tuple(rows[0].shape)
        raise ValueError(
            f"routed_experts rows must be [seq, layers, topk], got {shape}."
        )
    num_layers, topk = int(rows[0].size(1)), int(rows[0].size(2))
    total_padded = int(meta.cu_seqlens_padded[-1].item())
    device = batch.input_ids.device
    full = torch.zeros(total_padded, num_layers, topk, dtype=torch.long, device=device)
    for idx, row in enumerate(rows):
        length = int(meta.lengths[idx].item())
        route_length = int(row.size(0))
        # Rollout routes are attached to consumed next-token logits, so vLLM
        # normally returns one fewer route row than the actor input has tokens.
        # The missing final row predicts no token and is excluded by the R3
        # replay mask below; leave its placeholder zeroed so native routing is
        # used there.  Also accept full-length rows for offline/test producers.
        if route_length not in (length - 1, length):
            raise ValueError(
                f"routed_experts seq {idx} has {route_length} tokens, "
                f"expected {length - 1} rollout rows or {length} full rows."
            )
        start = int(meta.cu_seqlens_padded[idx].item())
        full[start : start + route_length] = row.to(device=device, dtype=torch.long)

    if contiguous:
        if ps.cp_size > 1:
            local_len = total_padded // ps.cp_size
            full = full[ps.cp_rank * local_len : (ps.cp_rank + 1) * local_len]
        local = full
    else:
        local = split_packed_to_cp_local(
            full,
            cu_seqlens_padded=meta.cu_seqlens_padded,
            cp_size=ps.cp_size,
            cp_rank=ps.cp_rank,
            dim=0,
        )
    if ps.tp_size > 1:
        tp_local = local.size(0) // ps.tp_size
        local = local[ps.tp_rank * tp_local : (ps.tp_rank + 1) * tp_local]
    return [local[:, layer, :].contiguous() for layer in range(num_layers)]


def pack_r3_replay_mask(
    model, batch: PackedBatch, *, contiguous: bool = False
) -> torch.Tensor:
    """Pack a caller-provided causal R3 replay mask for the local layout.

    This helper performs only model-local THD/CP/TP layout conversion. The mask
    semantics are defined by the caller and carried with the packed batch; a
    missing mask fails loudly.
    """

    if batch.r3_replay_mask is None:
        raise ValueError("R3 router replay requires batch.r3_replay_mask.")
    if batch.r3_replay_mask.numel() != int(batch.seq_lens.sum().item()):
        raise ValueError(
            "batch.r3_replay_mask must contain one entry per packed input token."
        )
    rows = []
    offset = 0
    for length_tensor in batch.seq_lens:
        length = int(length_tensor.item())
        row = batch.r3_replay_mask[offset : offset + length]
        rows.append(row.to(device=batch.input_ids.device)[:, None, None])
        offset += length
    nested = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    return pack_routed_experts(model, batch, nested, contiguous=contiguous)[0][
        :, 0
    ].bool()


def add_loss_context_kwargs(
    kwargs: dict[str, Any], *, include_return_log_probs: bool = False
) -> None:
    loss_context = get_loss_context()
    if loss_context is None:
        return
    kwargs["temperature"] = loss_context.temperature
    kwargs["calculate_entropy"] = loss_context.calculate_entropy
    if include_return_log_probs:
        kwargs["return_log_probs"] = loss_context.return_log_probs


def _resolve_cross_entropy_fusion(model) -> bool:
    """Read the ``cross_entropy_fusion`` flag through any module wrappers.

    ``set_cross_entropy_fusion`` runs at build time on the bare model chunks;
    by the time ``_forward_step`` runs, the runtime has wrapped each chunk in a
    data-parallel container. Megatron's ``_BaseDataParallel`` keeps the wrapped
    module in ``self.module`` and does not proxy unknown attributes to it, so a
    plain ``getattr(wrapper, "cross_entropy_fusion")`` misses the attribute
    entirely and silently falls back to the default.

    The consequence was invisible: the flag was set correctly, appeared as
    ``True`` in every config dump, and the model still took the unfused branch —
    materialising the full ``[tokens, vocab/tp]`` logits and upcasting them to
    fp32. Walk the ``.module`` chain instead, checking each level's own
    ``__dict__`` so an attribute on an inner chunk is found no matter how many
    wrappers were added.
    """

    seen: set[int] = set()
    current = model
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        own = getattr(current, "__dict__", None)
        if isinstance(own, dict) and "cross_entropy_fusion" in own:
            return bool(own["cross_entropy_fusion"])
        current = getattr(current, "module", None)
    return False


def add_cross_entropy_fusion(kwargs: dict[str, Any], model) -> None:
    kwargs["use_fused_kernels"] = _resolve_cross_entropy_fusion(model)


def set_cross_entropy_fusion(chunks: list, enabled: bool) -> None:
    for chunk in chunks:
        chunk.cross_entropy_fusion = bool(enabled)


__all__ = [
    "_resolve_cross_entropy_fusion",
    "add_cross_entropy_fusion",
    "add_loss_context_kwargs",
    "nested_from_packed",
    "pack_r3_replay_mask",
    "pack_routed_experts",
    "pack_thd_forward_kwargs",
    "set_cross_entropy_fusion",
    "unpack_thd_forward_output",
]
