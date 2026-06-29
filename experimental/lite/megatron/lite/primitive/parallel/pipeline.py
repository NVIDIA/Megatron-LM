# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pipeline parallel: 1F1B schedule with correct backward handling."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.utils import ensure_divisible
from megatron.lite.runtime.contracts.loss import split_loss_context, use_loss_context

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel.state import ParallelState


def forward_backward_pipelining(
    forward_step_fn: Callable,
    model_chunks: list,
    data_iter,
    config,
    ps: ParallelState,
    tensor_shape: tuple[int, ...] | None = None,
    grad_sync_fn: Callable[[], None] | None = None,
    pre_forward_hook: Callable[[torch.Tensor], None] | None = None,
    loss_fn: Callable | None = None,
    forward_only: bool = False,
) -> list[dict]:
    """
    Run forward and backward passes with pipeline parallelism.

    Args:
        forward_step_fn: Callable(model, batch) -> output_dict with "loss" and "hidden_states"
        model_chunks: local model chunks. ``len>1`` enables interleaved VPP.
        data_iter: iterator yielding micro-batches
        config: training config
        ps: parallel state
        tensor_shape: shape of hidden states passed between stages [B, S, H]
        grad_sync_fn: called before the last micro-batch's backward to enable
                       overlapped gradient ReduceScatter in DistributedOptimizer.

    Returns:
        List of output dicts from forward passes.
    """
    if ps.pp_size <= 1:
        if forward_only:
            return _forward_only_no_pipeline(
                forward_step_fn,
                model_chunks[0],
                data_iter,
                config,
                ps,
                pre_forward_hook=pre_forward_hook,
                loss_fn=loss_fn,
            )
        return _no_pipeline(
            forward_step_fn,
            model_chunks[0],
            data_iter,
            config,
            ps,
            grad_sync_fn=grad_sync_fn,
            pre_forward_hook=pre_forward_hook,
            loss_fn=loss_fn,
        )

    if tensor_shape is None:
        raise ValueError("tensor_shape is required when PP > 1")

    num_microbatches = _num_microbatches_from_config(config, ps)

    if forward_only:
        return _forward_only_pipeline_schedule(
            forward_step_fn,
            model_chunks,
            data_iter,
            num_microbatches,
            ps,
            tensor_shape,
            pre_forward_hook=pre_forward_hook,
            loss_fn=loss_fn,
        )

    if len(model_chunks) > 1:
        return _interleaved_1f1b_schedule(
            forward_step_fn,
            model_chunks,
            data_iter,
            num_microbatches,
            config,
            ps,
            tensor_shape,
            grad_sync_fn=grad_sync_fn,
            pre_forward_hook=pre_forward_hook,
            loss_fn=loss_fn,
        )

    return _1f1b_schedule(
        forward_step_fn,
        model_chunks[0],
        data_iter,
        num_microbatches,
        config,
        ps,
        tensor_shape,
        grad_sync_fn=grad_sync_fn,
        pre_forward_hook=pre_forward_hook,
        loss_fn=loss_fn,
    )


# ══════════════════════════════════════════════════════════════════════
# No pipeline (PP=1)
# ══════════════════════════════════════════════════════════════════════
def _num_microbatches_from_config(config, ps: ParallelState) -> int:
    explicit = getattr(config, "num_microbatches", None)
    if explicit is not None:
        return int(explicit)
    return ensure_divisible(config.gbs, config.mbs * ps.dp_size)


def _set_aux_loss_scale(pre_forward_hook, num_microbatches: int) -> None:
    if pre_forward_hook is not None:
        scale = torch.tensor(1.0 / num_microbatches, device="cuda")
        pre_forward_hook(scale)


def _batch_get(batch, key: str):
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _apply_external_loss(
    out: dict, batch, loss_fn, loss_context=None
) -> tuple[torch.Tensor, dict] | tuple[None, None]:
    if loss_fn is None:
        return None, None
    # Mirror run_microbatch_loop: pass loss_context as 3rd arg when present.
    if loss_context is None:
        loss, metrics = loss_fn(out, batch)
    else:
        loss, metrics = loss_fn(out, batch, loss_context)
    out["loss"] = loss
    out["_loss_fn_metrics"] = metrics
    return loss, metrics


def _compact_pipeline_output(out: dict | None) -> dict:
    if not out:
        return {}
    compact: dict = {}
    if "model_output" in out:
        compact["model_output"] = out["model_output"]
    if "loss" in out and out["loss"] is not None:
        loss = out["loss"]
        compact["loss"] = loss.detach().item() if isinstance(loss, torch.Tensor) else float(loss)
    if "_loss_fn_metrics" in out:
        compact["metrics"] = out["_loss_fn_metrics"]
    elif "metrics" in out:
        compact["metrics"] = out["metrics"]
    return compact


def _no_pipeline(
    forward_step_fn,
    model,
    data_iter,
    config,
    ps,
    *,
    grad_sync_fn=None,
    pre_forward_hook=None,
    loss_fn=None,
):
    num_microbatches = _num_microbatches_from_config(config, ps)
    outputs = []
    for i in range(num_microbatches):
        if grad_sync_fn and i == num_microbatches - 1:
            grad_sync_fn()
        batch = next(data_iter)
        _set_aux_loss_scale(pre_forward_hook, num_microbatches)
        output = forward_step_fn(model, batch)
        if loss_fn is not None:
            loss, _metrics = _apply_external_loss(output, batch, loss_fn)
            assert loss is not None
        else:
            loss = output["loss"]
        loss = loss / num_microbatches
        loss.backward()
        outputs.append(_compact_pipeline_output(output))
    return outputs


def _forward_only_no_pipeline(
    forward_step_fn, model, data_iter, config, ps, *, pre_forward_hook=None, loss_fn=None
):
    num_microbatches = _num_microbatches_from_config(config, ps)
    del ps
    outputs = []
    for _ in range(num_microbatches):
        batch = next(data_iter)
        _set_aux_loss_scale(pre_forward_hook, num_microbatches)
        output = forward_step_fn(model, batch)
        _apply_external_loss(output, batch, loss_fn)
        outputs.append(_compact_pipeline_output(output))
    return outputs


# ══════════════════════════════════════════════════════════════════════
# 1F1B Schedule
# ══════════════════════════════════════════════════════════════════════
def _1f1b_schedule(
    forward_step_fn,
    model,
    data_iter,
    num_microbatches: int,
    config,
    ps: ParallelState,
    tensor_shape: tuple[int, ...],
    *,
    grad_sync_fn=None,
    pre_forward_hook=None,
    loss_fn=None,
):
    """
    1-Forward-1-Backward pipeline schedule using batch_isend_irecv.

    Communication is always done via combined send+recv to avoid deadlocks.
    """
    num_warmup = min(ps.pp_size - ps.pp_rank - 1, num_microbatches)
    num_steady = num_microbatches - num_warmup

    # Split each microbatch into (PackedBatch, LossContext) like run_microbatch_loop; the connector
    # yields (batch, loss_context) tuples, so forward_step must receive the unwrapped batch.
    batches = [split_loss_context(next(data_iter)) for _ in range(num_microbatches)]
    mb_idx = 0

    input_tensors: list[torch.Tensor | None] = []
    output_hiddens: list[torch.Tensor | None] = []
    losses: list[torch.Tensor | None] = []
    outputs: list[dict] = []

    # Fix 3: Pre-allocate recv buffers to avoid torch.empty per P2P call.
    _fwd_recv_buf = (
        torch.empty(tensor_shape, dtype=_PIPELINE_TENSOR_DTYPE, device="cuda")
        if not ps.pp_is_first
        else None
    )
    _bwd_recv_buf = (
        torch.empty(tensor_shape, dtype=_PIPELINE_TENSOR_DTYPE, device="cuda")
        if not ps.pp_is_last
        else None
    )

    def _run_forward(input_tensor, batch, loss_context=None):
        _set_aux_loss_scale(pre_forward_hook, num_microbatches)
        if not ps.pp_is_first:
            # `model` is the dist_opt DDP-wrapped chunk; set_input_tensor lives on the base lite model.
            from megatron.lite.primitive.ckpt.hf_weights import unwrap_model

            unwrap_model(model).set_input_tensor(input_tensor)
        with use_loss_context(loss_context):
            out = forward_step_fn(model, batch)
            if ps.pp_is_last:
                _apply_external_loss(out, batch, loss_fn, loss_context)
        return out

    def _run_backward(inp_t, hid_t, loss_t, grad_t):
        if ps.pp_is_last:
            if loss_t is not None:
                loss_t.backward()
        else:
            if hid_t is not None and hid_t.requires_grad:
                torch.autograd.backward(hid_t, grad_t)
        return inp_t.grad if inp_t is not None else None

    def _p2p(send_fwd=None, send_bwd=None, recv_fwd=False, recv_bwd=False):
        return _send_recv_pipeline(
            send_fwd,
            send_bwd,
            recv_fwd,
            recv_bwd,
            ps,
            tensor_shape,
            fwd_recv_buf=_fwd_recv_buf,
            bwd_recv_buf=_bwd_recv_buf,
        )

    # ── Warmup: pure forward passes ──
    fwd_input: torch.Tensor | None = None
    for k in range(num_warmup):
        if not ps.pp_is_first and k == 0:
            fwd_input, _ = _p2p(recv_fwd=True)

        batch, loss_ctx = batches[mb_idx]
        mb_idx += 1
        current_input = fwd_input
        out = _run_forward(fwd_input, batch, loss_ctx)
        hidden = out.get("hidden_states")
        loss_s = out["loss"] / num_microbatches if "loss" in out and ps.pp_is_last else None

        need_recv_next = not ps.pp_is_first and k < num_warmup - 1
        if not ps.pp_is_last:
            fwd_input, _ = _p2p(send_fwd=hidden, recv_fwd=need_recv_next)
        elif need_recv_next:
            fwd_input, _ = _p2p(recv_fwd=True)

        input_tensors.append(current_input if not ps.pp_is_first else None)
        output_hiddens.append(hidden)
        losses.append(loss_s)
        # Fix 4: only keep loss from output, drop logits/hidden references
        outputs.append(_compact_pipeline_output(out))

    # ── Steady: interleaved forward + backward ──
    for k in range(num_steady):
        if grad_sync_fn and num_warmup == 0 and k == num_steady - 1:
            grad_sync_fn()

        if not ps.pp_is_first and k == 0 and num_warmup == 0:
            fwd_input, _ = _p2p(recv_fwd=True)

        batch, loss_ctx = batches[mb_idx]
        mb_idx += 1
        out = _run_forward(fwd_input, batch, loss_ctx)
        hidden = out.get("hidden_states")
        loss_s = out["loss"] / num_microbatches if "loss" in out and ps.pp_is_last else None

        input_tensors.append(fwd_input if not ps.pp_is_first else None)
        output_hiddens.append(hidden)
        losses.append(loss_s)
        outputs.append(_compact_pipeline_output(out))

        send_fwd = hidden if not ps.pp_is_last else None
        need_bwd = not ps.pp_is_last
        _, bwd_grad = _p2p(send_fwd=send_fwd, recv_bwd=need_bwd)

        old_inp = input_tensors.pop(0)
        old_hid = output_hiddens.pop(0)
        old_loss = losses.pop(0)
        in_grad = _run_backward(old_inp, old_hid, old_loss, bwd_grad)

        send_bwd = in_grad if not ps.pp_is_first else None
        need_fwd = not ps.pp_is_first and k < num_steady - 1
        fwd_input, _ = _p2p(send_bwd=send_bwd, recv_fwd=need_fwd)

    # ── Cooldown: drain remaining backwards ──
    for k in range(num_warmup):
        if grad_sync_fn and k == num_warmup - 1:
            grad_sync_fn()

        need_bwd = not ps.pp_is_last
        _, bwd_grad = _p2p(recv_bwd=need_bwd)

        old_inp = input_tensors.pop(0)
        old_hid = output_hiddens.pop(0)
        old_loss = losses.pop(0)
        in_grad = _run_backward(old_inp, old_hid, old_loss, bwd_grad)

        send_bwd = in_grad if not ps.pp_is_first else None
        if send_bwd is not None:
            _p2p(send_bwd=send_bwd)

    return outputs


# ══════════════════════════════════════════════════════════════════════
# Pipeline communication helpers
# ══════════════════════════════════════════════════════════════════════
_PIPELINE_TENSOR_DTYPE = torch.bfloat16


def _deallocate_output_tensor(tensor: torch.Tensor | None) -> None:
    """Free a large output tensor after it has been sent to the next stage."""
    if tensor is not None:
        tensor.data = torch.empty(1, device=tensor.device, dtype=tensor.dtype)


# ══════════════════════════════════════════════════════════════════════
# Interleaved 1F1B (VPP) Schedule
# ══════════════════════════════════════════════════════════════════════
def _build_schedule_table(
    num_microbatches: int, num_chunks: int, group_size: int
) -> list[tuple[int, int]]:
    """Build (microbatch_id, model_chunk_id) table for VPP scheduling."""
    table: list[tuple[int, int]] = []
    for start in range(0, num_microbatches, group_size):
        end = min(start + group_size, num_microbatches)
        for chunk in range(num_chunks):
            for mb in range(start, end):
                table.append((mb, chunk))
    return table


def _send_recv_pipeline(
    send_fwd: torch.Tensor | None,
    send_bwd: torch.Tensor | None,
    recv_fwd: bool,
    recv_bwd: bool,
    ps: ParallelState,
    tensor_shape: tuple[int, ...],
    *,
    fwd_recv_buf: torch.Tensor | None = None,
    bwd_recv_buf: torch.Tensor | None = None,
    batch_p2p: bool = True,
    clone_recv: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """P2P communication between pipeline stages."""
    _dbg = int(os.environ.get("MEGATRON_LITE_PP_DEBUG", "0"))
    rank = dist.get_rank()

    ops: list[dist.P2POp] = []
    fwd_buf: torch.Tensor | None = None
    bwd_buf: torch.Tensor | None = None

    p2p_group = ps.pp_group

    if send_fwd is not None:
        t = send_fwd.to(_PIPELINE_TENSOR_DTYPE)
        ops.append(dist.P2POp(dist.isend, t, ps.pp_next_rank, p2p_group))
    if recv_fwd:
        fwd_buf = (
            fwd_recv_buf
            if fwd_recv_buf is not None
            else torch.empty(tensor_shape, dtype=_PIPELINE_TENSOR_DTYPE, device="cuda")
        )
        ops.append(dist.P2POp(dist.irecv, fwd_buf, ps.pp_prev_rank, p2p_group))
    if send_bwd is not None:
        t = send_bwd.to(_PIPELINE_TENSOR_DTYPE)
        ops.append(dist.P2POp(dist.isend, t, ps.pp_prev_rank, p2p_group))
    if recv_bwd:
        bwd_buf = (
            bwd_recv_buf
            if bwd_recv_buf is not None
            else torch.empty(tensor_shape, dtype=_PIPELINE_TENSOR_DTYPE, device="cuda")
        )
        ops.append(dist.P2POp(dist.irecv, bwd_buf, ps.pp_next_rank, p2p_group))

    if ops:
        if _dbg:
            desc = []
            if send_fwd is not None:
                desc.append(f"send_fwd→{ps.pp_next_rank}({list(send_fwd.shape)})")
            if recv_fwd:
                desc.append(f"recv_fwd←{ps.pp_prev_rank}")
            if send_bwd is not None:
                desc.append(f"send_bwd→{ps.pp_prev_rank}")
            if recv_bwd:
                desc.append(f"recv_bwd←{ps.pp_next_rank}")
            op_name = "batch_isend_irecv" if batch_p2p else "isend_irecv"
            print(f"[P2P r{rank}] {op_name}: {' '.join(desc)}", flush=True)
        if batch_p2p:
            reqs = dist.batch_isend_irecv(ops)
        else:
            direct_tensors = []
            reqs = []
            if send_fwd is not None:
                t = send_fwd.to(_PIPELINE_TENSOR_DTYPE)
                direct_tensors.append(t)
                reqs.append(dist.isend(t, ps.pp_next_rank, group=p2p_group))
            if recv_fwd:
                reqs.append(dist.irecv(fwd_buf, ps.pp_prev_rank, group=p2p_group))
            if send_bwd is not None:
                t = send_bwd.to(_PIPELINE_TENSOR_DTYPE)
                direct_tensors.append(t)
                reqs.append(dist.isend(t, ps.pp_prev_rank, group=p2p_group))
            if recv_bwd:
                reqs.append(dist.irecv(bwd_buf, ps.pp_next_rank, group=p2p_group))
        for req in reqs:
            req.wait()
        if _dbg:
            print(f"[P2P r{rank}] batch done", flush=True)

    if fwd_buf is not None:
        if clone_recv:
            fwd_buf = fwd_buf.clone()
        fwd_buf.grad = None
        fwd_buf.requires_grad_()
    if bwd_buf is not None and clone_recv:
        bwd_buf = bwd_buf.clone()
    return fwd_buf, bwd_buf


def _pipeline_stage_barrier(ps: ParallelState) -> None:
    if ps.pp_cpu_group is not None and ps.pp_size > 1:
        dist.barrier(group=ps.pp_cpu_group)


def _set_virtual_pipeline_rank(ps: ParallelState, chunk_id: int | None, num_chunks: int) -> None:
    if chunk_id is None or num_chunks <= 1:
        ps.virtual_pipeline_size = None
        ps.virtual_pipeline_rank = None
        return
    ps.virtual_pipeline_size = num_chunks
    ps.virtual_pipeline_rank = chunk_id


def _run_pipeline_chunk_forward(
    forward_step_fn,
    model,
    batch,
    input_tensor: torch.Tensor | None,
    *,
    is_first_stage: bool,
    is_last_stage: bool,
    num_microbatches: int,
    pre_forward_hook=None,
    loss_fn=None,
) -> dict:
    _set_aux_loss_scale(pre_forward_hook, num_microbatches)
    if not is_first_stage:
        model.set_input_tensor(input_tensor)
    out = forward_step_fn(model, batch)
    if is_last_stage:
        _apply_external_loss(out, batch, loss_fn)
    return out


def _forward_only_pipeline_schedule(
    forward_step_fn,
    model_chunks: list,
    data_iter,
    num_microbatches: int,
    ps: ParallelState,
    tensor_shape: tuple[int, ...],
    *,
    pre_forward_hook=None,
    loss_fn=None,
):
    """Simple PP/VPP forward-only schedule used for log-prob inference."""
    num_chunks = len(model_chunks)
    total_stages = ps.pp_size * num_chunks
    outputs: list[dict] = []

    for _mb in range(num_microbatches):
        batch = next(data_iter)
        _set_aux_loss_scale(pre_forward_hook, num_microbatches)
        pending_activation: torch.Tensor | None = None
        last_output: dict | None = None
        for stage_id in range(total_stages):
            stage_pp_rank = stage_id % ps.pp_size
            is_local_stage = stage_pp_rank == ps.pp_rank
            hidden: torch.Tensor | None = None
            if is_local_stage:
                chunk_id = stage_id // ps.pp_size
                _set_virtual_pipeline_rank(ps, chunk_id, num_chunks)
                model = model_chunks[chunk_id]
                is_first_stage = stage_id == 0
                is_last_stage = stage_id == total_stages - 1
                activation = None if is_first_stage else pending_activation
                pending_activation = None
                out = _run_pipeline_chunk_forward(
                    forward_step_fn,
                    model,
                    batch,
                    activation,
                    is_first_stage=is_first_stage,
                    is_last_stage=is_last_stage,
                    num_microbatches=num_microbatches,
                    pre_forward_hook=None,
                    loss_fn=loss_fn,
                )
                if is_last_stage:
                    last_output = out
                else:
                    hidden = out.get("hidden_states")

            if stage_id < total_stages - 1:
                recv_next = (stage_id + 1) % ps.pp_size == ps.pp_rank
                _pipeline_stage_barrier(ps)
                fwd_buf, _ = _send_recv_pipeline(
                    hidden if is_local_stage else None,
                    None,
                    recv_next,
                    False,
                    ps,
                    tensor_shape,
                    batch_p2p=False,
                    clone_recv=True,
                )
                if recv_next:
                    pending_activation = fwd_buf

        outputs.append(_compact_pipeline_output(last_output) if last_output is not None else {})

    _set_virtual_pipeline_rank(ps, None, num_chunks)
    return outputs


def _interleaved_1f1b_schedule(
    forward_step_fn,
    model_chunks: list,
    data_iter,
    num_microbatches: int,
    config,
    ps: ParallelState,
    tensor_shape: tuple[int, ...],
    *,
    grad_sync_fn=None,
    pre_forward_hook=None,
    loss_fn=None,
):
    """
    Correct non-overlapped schedule for Virtual Pipeline Parallelism (VPP).

    Local chunks are laid out in global layer order as
    ``chunk_id * pp_size + pp_rank``.  The previous interleaved 1F1B bringup
    schedule could ask the first physical PP stage to receive the next virtual
    chunk before the last physical stage had produced it.  This schedule keeps
    the same VPP semantics but runs one micro-batch at a time in global stage
    order, which is slower but deterministic and easy to validate against
    Megatron.
    """
    num_chunks = len(model_chunks)
    total_stages = ps.pp_size * num_chunks
    rank = dist.get_rank()
    outputs: list[dict] = []

    _dbg = int(os.environ.get("MEGATRON_LITE_PP_DEBUG", "0"))
    if _dbg:
        print(
            f"[VPP r{rank}] entered simple schedule pp_rank={ps.pp_rank} "
            f"microbatches={num_microbatches} chunks={num_chunks} total_stages={total_stages}",
            flush=True,
        )

    for mb_id in range(num_microbatches):
        batch = next(data_iter)
        _set_aux_loss_scale(pre_forward_hook, num_microbatches)
        saved: dict[
            int, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict]
        ] = {}
        pending_activation: torch.Tensor | None = None

        # Forward in true virtual-stage order:
        # chunk0/rank0 -> chunk0/rank1 -> ... -> chunkN/rankP.
        for stage_id in range(total_stages):
            stage_pp_rank = stage_id % ps.pp_size
            is_local_stage = stage_pp_rank == ps.pp_rank
            hidden: torch.Tensor | None = None
            if is_local_stage:
                chunk_id = stage_id // ps.pp_size
                _set_virtual_pipeline_rank(ps, chunk_id, num_chunks)
                model = model_chunks[chunk_id]
                is_first_stage = stage_id == 0
                is_last_stage = stage_id == total_stages - 1
                activation = None if is_first_stage else pending_activation
                pending_activation = None
                if _dbg:
                    activation_shape = None if activation is None else tuple(activation.shape)
                    print(
                        f"[VPP r{rank}] mb={mb_id} fwd stage={stage_id} "
                        f"chunk={chunk_id} activation_shape={activation_shape}",
                        flush=True,
                    )
                out = _run_pipeline_chunk_forward(
                    forward_step_fn,
                    model,
                    batch,
                    activation,
                    is_first_stage=is_first_stage,
                    is_last_stage=is_last_stage,
                    num_microbatches=num_microbatches,
                    pre_forward_hook=None,
                    loss_fn=loss_fn,
                )

                hidden = out.get("hidden_states")
                if _dbg:
                    hidden_shape = None if hidden is None else tuple(hidden.shape)
                    print(
                        f"[VPP r{rank}] mb={mb_id} fwd stage={stage_id} "
                        f"chunk={chunk_id} hidden_shape={hidden_shape}",
                        flush=True,
                    )
                loss = out["loss"] / num_microbatches if is_last_stage and "loss" in out else None
                saved[stage_id] = (activation, hidden, loss, out)

                if is_last_stage:
                    outputs.append(_compact_pipeline_output(out))

            if stage_id < total_stages - 1:
                recv_next = (stage_id + 1) % ps.pp_size == ps.pp_rank
                if _dbg and (is_local_stage or recv_next):
                    print(
                        f"[VPP r{rank}] mb={mb_id} fwd boundary={stage_id} "
                        f"send={is_local_stage and hidden is not None} recv_next={recv_next}",
                        flush=True,
                    )
                _pipeline_stage_barrier(ps)
                fwd_buf, _ = _send_recv_pipeline(
                    hidden if is_local_stage else None,
                    None,
                    recv_next,
                    False,
                    ps,
                    tensor_shape,
                    batch_p2p=False,
                    clone_recv=True,
                )
                if recv_next:
                    pending_activation = fwd_buf

        if grad_sync_fn and mb_id == num_microbatches - 1:
            grad_sync_fn()

        # Backward in the reverse virtual-stage order.
        pending_grad: torch.Tensor | None = None
        for stage_id in range(total_stages - 1, -1, -1):
            stage_pp_rank = stage_id % ps.pp_size
            is_local_stage = stage_pp_rank == ps.pp_rank
            inp_grad: torch.Tensor | None = None
            if is_local_stage:
                chunk_id = stage_id // ps.pp_size
                _set_virtual_pipeline_rank(ps, chunk_id, num_chunks)
                is_first_stage = stage_id == 0
                is_last_stage = stage_id == total_stages - 1
                inp, out_t, loss, _out = saved[stage_id]
                grad = None if is_last_stage else pending_grad
                pending_grad = None
                if _dbg:
                    print(f"[VPP r{rank}] mb={mb_id} bwd stage={stage_id}", flush=True)
                if is_last_stage:
                    if loss is not None:
                        loss.backward()
                elif out_t is not None and out_t.requires_grad:
                    torch.autograd.backward(out_t, grad)
                if not is_first_stage:
                    inp_grad = inp.grad if inp is not None else None

            if stage_id > 0:
                recv_prev = (stage_id - 1) % ps.pp_size == ps.pp_rank
                if _dbg and (is_local_stage or recv_prev):
                    print(
                        f"[VPP r{rank}] mb={mb_id} bwd boundary={stage_id} "
                        f"send={is_local_stage and inp_grad is not None} recv_prev={recv_prev}",
                        flush=True,
                    )
                _pipeline_stage_barrier(ps)
                _, bwd_buf = _send_recv_pipeline(
                    None,
                    inp_grad if is_local_stage else None,
                    False,
                    recv_prev,
                    ps,
                    tensor_shape,
                    batch_p2p=False,
                    clone_recv=True,
                )
                if recv_prev:
                    pending_grad = bwd_buf

        if _dbg:
            print(f"[VPP r{rank}] mb={mb_id} complete", flush=True)

    _set_virtual_pipeline_rank(ps, None, num_chunks)
    return outputs


__all__ = ["forward_backward_pipelining"]
