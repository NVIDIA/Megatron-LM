# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Three-phase schedule for colocated MIMO training with LLM PP>1.

Phase 1: Encoder forward + communicate for the full batch (all ranks synchronized).
Phase 2: LLM 1F1B pipeline with detached encoder embeddings sliced per microbatch.
Phase 3: Encoder backward for the full batch (all ranks synchronized).

Encoder runs on all ranks (PP=1) and its TP/DP collectives require all ranks
to participate simultaneously. The 1F1B pipeline staggers ranks across PP stages,
so encoder collectives cannot run inside the pipeline. The three-phase design
separates encoder (synchronized) from LLM (pipelined) by detaching the autograd
graph at the encoder-LLM boundary.

Shape contract: encoder input tensors are 3D ``[seq, batch, hidden]`` with
the batch dim at ``dim=1``. Encoder output embeddings are either 3D
``[seq, batch, hidden]`` (batch dim = 1) or 2D ``[seq*batch, hidden]``
(batch dim = 0); the bridge may collapse the leading two dims. Other
layouts (e.g. ``[B, C, H, W]`` images) are not supported.

DP-direction contract: fan-in (enc_dp > llm_dp), fan-out (enc_dp < llm_dp),
and equal-DP are all supported. The ColocatedBridgeCommunicator handles
the encoder-side reshape on both forward (fan-in: all-gather, fan-out:
narrow) and backward (fan-in: scatter, fan-out: all-gather). The schedule's
job is to hand each side its correctly-sized slice of the global batch:

  * Fan-in: data iterator yields LLM-DP-sized per-rank batches; the
    schedule narrows encoder inputs to the encoder rank's smaller slot
    in ``_slice_for_encoder_dp`` before encode_and_communicate.
  * Fan-out: data iterator yields encoder-DP-sized per-rank batches; the
    bridge narrows encoder embeddings to the LLM-DP rank's slot inside
    encode_and_communicate, and ``_build_lm_microbatches`` narrows the
    LLM-side passthrough fields (input_ids, labels, loss_mask,
    position_ids) to the same slot so they line up with the bridge
    output for the LLM forward.
"""

from contextlib import contextmanager
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel import schedules


def colocated_forward_backward_with_pp(
    mimo_model,
    data_iterator,
    num_microbatches: int,
    encoder_grid: Optional[HyperCommGrid] = None,
    llm_grid: Optional[HyperCommGrid] = None,
    encoder_name: str = "images",
    forward_only: bool = False,
    **schedule_kwargs,
):
    """Three-phase colocated training: encoder batch -> LLM pipeline -> encoder backward.

    Args:
        mimo_model: MimoModel with colocated communicators and lm_has_pp=True.
        data_iterator: Yields dicts with input_ids, labels, etc.
        num_microbatches: Number of microbatches for the LLM pipeline.
        encoder_grid: Encoder HyperCommGrid (for DP fan-in slicing).
        llm_grid: LLM HyperCommGrid (for PP group).
        encoder_name: Modality name for the encoder (e.g., "images").
        forward_only: Skip backward passes if True.
        **schedule_kwargs: Passed to forward_backward_pipelining_without_interleaving.
            Must include p2p_communicator, pg_collection, seq_length, micro_batch_size.
    """
    pp_group = llm_grid.get_pg("pp") if llm_grid and 'pp' in llm_grid.dim_names else None
    is_pp_first = pp_group is None or pp_group.rank() == 0

    # ── Phase 1: Encoder forward on full batch (one pass) ────────────────
    # All ranks participate (encoder is PP=1, communicate is collective).
    all_batches = [next(data_iterator) for _ in range(num_microbatches)]
    full_encoder_input = _concat_encoder_inputs(all_batches, encoder_name)
    _slice_for_encoder_dp(full_encoder_input, encoder_grid, llm_grid)

    enc_out = mimo_model.encode_and_communicate({encoder_name: full_encoder_input})

    # Detach so Phase 2 runs no encoder collectives; microbatch views accumulate
    # .grad into detached_full.grad automatically.
    detached_full = {k: v.detach().requires_grad_(True) for k, v in enc_out.items()}
    lm_data = _build_lm_microbatches(
        detached_full, all_batches, num_microbatches, encoder_grid, llm_grid
    )

    # ── Phase 2: LLM 1F1B pipeline ──────────────────────────────────────
    # Only LLM P2P communication (within PP group). No encoder collectives.
    cache_iter = iter(lm_data)

    def _lm_forward_step(data_iterator_unused, model, *args):
        cached = next(cache_iter)
        forward_kwargs = dict(
            input_ids=cached['input_ids'],
            labels=cached['labels'],
            loss_mask=cached['loss_mask'],
            position_ids=cached['position_ids'],
            encoder_embeddings=cached['encoder_embeddings'],
        )
        if cached.get('attention_mask') is not None:
            forward_kwargs['attention_mask'] = cached['attention_mask']
        if cached.get('packing_kwargs') is not None:
            forward_kwargs['packing_kwargs'] = cached['packing_kwargs']
        output_tensor, loss_mask = model(**forward_kwargs)
        return output_tensor, partial(_loss_func, cached['loss_mask'])

    # Swap in a capturing finalize so the inner PP schedule does not run DDP
    # grad sync before Phase 3 has produced encoder grads. The capture also
    # records ``num_tokens`` and ``force_all_reduce`` that the inner schedule
    # would have passed — we forward them to the original finalize after
    # Phase 3 so per-token-loss configs see the correct global divisor and
    # any caller-requested all-reduce semantics are preserved.
    with _deferred_finalize(mimo_model.config) as (original_finalize, capture):
        losses = schedules.forward_backward_pipelining_without_interleaving(
            forward_step_func=_lm_forward_step,
            data_iterator=cache_iter,
            model=[mimo_model],
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            **schedule_kwargs,
        )

    # ── Phase 3: Encoder backward (one pass, all ranks sync) ────────────
    # detached_full.grad was populated by Phase 2's per-microbatch LLM backward
    # (accumulated across microbatch view slices on PP stage 0).
    # Broadcast to PP stage 1+ then run one encoder backward for the full batch.
    if not forward_only and enc_out:
        _broadcast_encoder_grad(detached_full, enc_out, pp_group, is_pp_first)
        for key in enc_out:
            grad = detached_full[key].grad
            if grad is not None:
                torch.autograd.backward(enc_out[key], grad_tensors=grad)

    # Single post-Phase-3 finalize: reduces LLM grads (from Phase 2) and
    # encoder grads (from Phase 3) together. Without this call, encoder
    # grads remain local to each rank and Adam steps on un-reduced grads,
    # causing silent divergence from the equal-DP reference. Forward the
    # captured force_all_reduce so callers requesting that semantics
    # (e.g. final-microbatch sync with overlap_grad_reduce) get it.
    if not forward_only and original_finalize is not None:
        original_finalize(
            [mimo_model],
            capture.num_tokens,
            pg_collection=schedule_kwargs.get('pg_collection'),
            force_all_reduce=capture.force_all_reduce,
        )

    return losses


# ── Helpers ──────────────────────────────────────────────────────────────


def _fan_out_slot(encoder_grid, llm_grid):
    """Return ``(scale, slot)`` for fan-out LLM-side narrowing.

    For fan-out (``llm_dp > enc_dp``) the data iterator yields encoder-DP-
    sized per-rank batches. The bridge narrows encoder embeddings to this
    LLM-DP rank's slot inside ``encode_and_communicate``; LLM-side fields
    (input_ids, labels, ...) must be narrowed to the SAME slot so they
    line up with the bridge output for the LLM forward. Returns
    ``(scale, slot)`` where ``slot`` is this rank's index inside the
    fan-out sibling group; ``(1, 0)`` for equal-DP and fan-in (where the
    LLM-side fields are already correctly sized for the LLM-DP rank).
    """
    if encoder_grid is None or llm_grid is None:
        return 1, 0
    enc_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()
    if llm_dp <= enc_dp:
        return 1, 0
    scale = llm_dp // enc_dp
    slot = llm_grid.get_pg("dp").rank() % scale
    return scale, slot


def _modality_present(batch, encoder_name):
    """Return True iff this batch carries inputs for ``encoder_name``."""
    mod_in = batch.get('modality_inputs')
    return bool(mod_in) and encoder_name in mod_in and mod_in[encoder_name] is not None


def _concat_encoder_inputs(all_batches, encoder_name):
    """Concatenate encoder inputs from all microbatches along batch dim (dim=1).

    All encoder input tensors must be 3D ``[seq, batch, hidden]``. All
    microbatches must uniformly have or lack ``modality_inputs[encoder_name]``;
    mixed batches are rejected because Phase 2 reuses one detached encoder
    output across every LLM microbatch.
    """
    first = all_batches[0]
    has_first = _modality_present(first, encoder_name)
    for idx, b in enumerate(all_batches):
        if _modality_present(b, encoder_name) != has_first:
            raise ValueError(
                f"colocated_forward_backward_with_pp requires uniform "
                f"modality_inputs across microbatches for '{encoder_name}'; "
                f"microbatch 0 has it = {has_first} but microbatch {idx} differs."
            )
    if not has_first:
        return {}
    result = {}
    for enc_name in first['modality_inputs'][encoder_name]:
        result[enc_name] = {}
        for key in first['modality_inputs'][encoder_name][enc_name]:
            vals = [b['modality_inputs'][encoder_name][enc_name][key] for b in all_batches]
            tensors = [v for v in vals if isinstance(v, torch.Tensor)]
            if tensors:
                for v in tensors:
                    if v.ndim != 3:
                        raise ValueError(
                            f"encoder input '{enc_name}.{key}' must be 3D "
                            f"[seq, batch, hidden], got shape={tuple(v.shape)}"
                        )
                result[enc_name][key] = torch.cat(tensors, dim=1)
            else:
                result[enc_name][key] = vals[0]
    return result


def _slice_for_encoder_dp(full_encoder_input, encoder_grid, llm_grid):
    """Slice concatenated encoder input for fan-in (enc_dp > llm_dp).

    Encoder input tensors must be 3D ``[seq, batch, hidden]``. For fan-in
    the data iterator yields LLM-DP-sized per-rank batches; this helper
    narrows them to the encoder rank's smaller slot before forward.
    Equal-DP and fan-out (where the per-rank batch is already encoder-DP-
    sized — the bridge narrows on the LLM side) are no-ops.
    """
    if encoder_grid is None or llm_grid is None:
        return
    enc_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()
    if enc_dp <= llm_dp:
        return
    scale = enc_dp // llm_dp
    slot = encoder_grid.get_pg("dp").rank() % scale
    for enc_name in full_encoder_input:
        for key, tensor in full_encoder_input[enc_name].items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim != 3:
                raise ValueError(
                    f"encoder input '{enc_name}.{key}' must be 3D "
                    f"[seq, batch, hidden], got shape={tuple(tensor.shape)}"
                )
            bs = tensor.shape[1]
            ss = bs // scale
            if ss == 0:
                raise ValueError(
                    f"Encoder fan-in produces zero-sized batch: "
                    f"total_batch={bs}, scale={scale}. Increase micro_batch_size."
                )
            full_encoder_input[enc_name][key] = tensor[
                :, slot * ss : (slot + 1) * ss, :
            ].contiguous()


def _build_lm_microbatches(
    detached_full, all_batches, num_microbatches, encoder_grid=None, llm_grid=None
):
    """Slice detached encoder output into per-microbatch views for the LLM pipeline.

    Encoder embeddings are either 3D ``[seq, batch, hidden]`` (batch dim = 1)
    or 2D ``[seq*batch, hidden]`` (batch dim = 0); the bridge may collapse
    the leading two dims. Other layouts are rejected. Pass-through fields
    (input_ids, labels, loss_mask, position_ids, attention_mask, packing_kwargs)
    are copied per microbatch from the corresponding ``all_batches`` entry.

    For fan-out (``llm_dp > enc_dp``) the per-microbatch passthrough fields
    arrive at the encoder-DP-sized batch; this helper narrows them to the
    LLM-DP rank's slot via :func:`_fan_out_slot` so they line up with the
    bridge-narrowed encoder embeddings. Fan-in and equal-DP leave the
    fields unchanged (``scale=1, slot=0``).
    """
    fan_out_scale, fan_out_slot = _fan_out_slot(encoder_grid, llm_grid)

    def _maybe_narrow(tensor):
        """Narrow a batch-dim-0 tensor to this LLM-DP rank's fan-out slot."""
        if fan_out_scale == 1 or tensor is None or not isinstance(tensor, torch.Tensor):
            return tensor
        bs = tensor.shape[0]
        if bs % fan_out_scale != 0:
            raise ValueError(
                f"Fan-out narrowing: tensor batch={bs} not divisible by " f"scale={fan_out_scale}."
            )
        ss = bs // fan_out_scale
        return tensor[fan_out_slot * ss : (fan_out_slot + 1) * ss].contiguous()

    def _maybe_narrow_attn(tensor, ref_batch):
        """Narrow ``attention_mask`` only when its dim-0 matches the input batch.

        Some callers pass attention_mask as ``[b, 1, s, s]`` (batch-first,
        narrow the way ``input_ids`` is narrowed); others pass shapes that
        broadcast across batch (e.g. ``[1, 1, s, s]`` causal mask). We only
        narrow when dim-0 equals the pre-narrowing batch size, leaving
        broadcastable masks alone.
        """
        if (
            fan_out_scale == 1
            or tensor is None
            or not isinstance(tensor, torch.Tensor)
            or ref_batch is None
            or not isinstance(ref_batch, torch.Tensor)
            or tensor.ndim < 1
            or tensor.shape[0] != ref_batch.shape[0]
        ):
            return tensor
        return _maybe_narrow(tensor)

    def _passthrough(batch_idx):
        b = all_batches[batch_idx]
        input_ids = b.get('input_ids')
        return {
            'input_ids': _maybe_narrow(input_ids),
            'labels': _maybe_narrow(b.get('labels')),
            'loss_mask': _maybe_narrow(b.get('loss_mask')),
            'position_ids': _maybe_narrow(b.get('position_ids')),
            'attention_mask': _maybe_narrow_attn(b.get('attention_mask'), input_ids),
            'packing_kwargs': b.get('packing_kwargs'),
        }

    if not detached_full:
        # Text-only batch: no encoder embeddings to slice
        return [{'encoder_embeddings': {}, **_passthrough(mb)} for mb in range(num_microbatches)]

    sample = next(iter(detached_full.values()))
    if sample.ndim not in (2, 3):
        raise ValueError(
            f"encoder output must be 2D [seq*batch, hidden] or 3D "
            f"[seq, batch, hidden], got shape={tuple(sample.shape)}"
        )
    batch_dim = 1 if sample.ndim == 3 else 0
    total_batch = sample.shape[batch_dim]
    if total_batch % num_microbatches != 0:
        raise ValueError(
            f"Encoder output batch dim ({total_batch}) must be divisible "
            f"by num_microbatches ({num_microbatches})"
        )
    mb_size = total_batch // num_microbatches

    lm_data = []
    for mb in range(num_microbatches):
        s, e = mb * mb_size, (mb + 1) * mb_size
        mb_enc = {k: (v[:, s:e, :] if v.ndim == 3 else v[s:e, :]) for k, v in detached_full.items()}
        lm_data.append({'encoder_embeddings': mb_enc, **_passthrough(mb)})
    return lm_data


def _broadcast_encoder_grad(detached_full, enc_out, pp_group, is_pp_first):
    """Broadcast encoder gradient from PP stage 0 to stage 1+ ranks."""
    if pp_group is None or pp_group.size() <= 1:
        return
    src = dist.get_global_rank(pp_group, 0)
    for key in enc_out:
        if is_pp_first:
            if detached_full[key].grad is None:
                raise RuntimeError(
                    f"No encoder gradient on PP stage 0 for '{key}'; "
                    f"Phase 2 LLM backward did not populate detached_full.grad."
                )
            dist.broadcast(detached_full[key].grad, src=src, group=pp_group)
        else:
            grad = torch.empty_like(detached_full[key])
            dist.broadcast(grad, src=src, group=pp_group)
            detached_full[key].grad = grad


def _loss_func(loss_mask, output_tensor):
    """Default loss function for the LLM pipeline.

    Returns the 3-tuple ``(local_sum, local_num_tokens, log_dict)`` contract
    expected when ``calculate_per_token_loss=True`` is set on the
    TransformerConfig. When it is not set, the schedule divides
    ``local_sum`` by ``local_num_tokens`` (clamped to 1), so the 3-tuple
    form is also safe for standard per-microbatch-mean configs.
    """
    if output_tensor is None:
        zero_loss = torch.tensor(0.0, device='cuda', requires_grad=True)
        zero_count = torch.tensor(0, device='cuda', dtype=torch.int)
        return zero_loss, zero_count, {'loss_reduced': 0.0}
    masked = output_tensor.float() * loss_mask.float()
    local_sum = masked.sum()
    local_num_tokens = loss_mask.float().sum().to(torch.int)
    return local_sum, local_num_tokens, {'loss_reduced': local_sum.detach().item()}


class _CapturingFinalize:
    """Capture finalize args the inner PP schedule would have passed.

    The three-phase schedule defers grad finalization until after Phase 3
    runs encoder backward. Replacing the config's ``finalize_model_grads_func``
    with this object absorbs the inner schedule's invocation and stores
    ``num_tokens`` (required for ``calculate_per_token_loss=True`` configs
    whose finalize hook divides by the global valid-token count) and
    ``force_all_reduce`` (preserves any caller-requested all-reduce
    semantics on the final microbatch) so the post-Phase-3 call to the
    original finalize can forward both.
    """

    def __init__(self):
        self.num_tokens = None
        self.force_all_reduce = False

    def __call__(self, model_list, num_tokens, *args, **kwargs):
        self.num_tokens = num_tokens
        self.force_all_reduce = kwargs.get('force_all_reduce', False)
        return None


@contextmanager
def _deferred_finalize(config):
    """Suppress the PP schedule's end-of-run DDP grad sync; yield the
    original finalize and a capture object so callers can invoke the
    original (with the captured ``num_tokens``) once after Phase 3.
    """
    original = config.finalize_model_grads_func
    capture = _CapturingFinalize()
    config.finalize_model_grads_func = capture
    try:
        yield original, capture
    finally:
        config.finalize_model_grads_func = original
