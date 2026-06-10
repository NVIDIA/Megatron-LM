"""Reusable train-step primitives owned by Megatron Lite."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.protocols import ExpertClassifierFn, default_expert_classifier


def run_microbatch_loop(
    model,
    data_iter,
    num_microbatches: int,
    forward_fn,
    optimizer=None,
    dist_opt: bool = False,
    pre_forward_hook: Callable[[torch.Tensor], None] | None = None,
    loss_fn: Callable | None = None,
):
    """Run forward-backward over microbatches with loss accumulation.

    Args:
        forward_fn: ``forward_fn(model, batch) -> dict`` with at least ``"loss"`` key
            (when ``loss_fn`` is None) or model outputs (when ``loss_fn`` is provided).
        pre_forward_hook: Optional callable ``hook(scale: torch.Tensor) -> None``
            invoked once per microbatch, right before ``forward_fn``. ``scale`` is
            ``1.0 / num_microbatches`` (matches MC's ``schedules.forward_step``,
            see `pipeline_parallel/schedules.py`). Used e.g. by MoE aux-loss
            scale-setting; runtime stays model-agnostic by passing the hook
            through from the model bundle's extras.
            # TODO: once CP is supported, align with MC's
            # `schedules:297`-style scale of `cp_group_size / num_microbatches`
            # (currently assumes ``cp_group_size == 1``).
        loss_fn: Optional external loss function.
            ``loss_fn(model_output: dict, batch) -> (loss: Tensor, metrics: dict)``.
            When provided, ``forward_fn`` output is passed to ``loss_fn`` instead of
            reading ``out["loss"]`` directly. This enables RLHF policy/value losses.
    """
    last_out = None
    all_metrics: list[dict] = []
    for mb in range(num_microbatches):
        batch = next(data_iter)
        if pre_forward_hook is not None:
            scale = torch.tensor(1.0 / num_microbatches, device="cuda")
            pre_forward_hook(scale)
        out = forward_fn(model, batch)
        if dist_opt and optimizer is not None and mb == num_microbatches - 1:
            optimizer.grad_sync_enabled = True
        if loss_fn is not None:
            loss, metrics = loss_fn(out, batch)
            (loss / num_microbatches).backward()
            out["loss"] = loss.detach()
            all_metrics.append(metrics)
        else:
            (out["loss"] / num_microbatches).backward()
        last_out = out
    if last_out is not None and all_metrics:
        last_out["_loss_fn_metrics"] = all_metrics
    return last_out


def compute_and_clip_grad_norm(
    model,
    optimizer,
    max_norm: float,
    use_dist_opt: bool,
    sp_params=None,
    sp_group=None,
    *,
    report_global_norm: bool = False,
    ps: ParallelState | None = None,
    is_expert_param: ExpertClassifierFn = default_expert_classifier,
):
    """SP AllReduce + finish grad sync + clip grad norm. Returns grad_norm."""
    if sp_params:
        sp_grads = [p.grad for p in sp_params if p.grad is not None]
        if sp_grads:
            flat = torch.cat([g.view(-1) for g in sp_grads])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=sp_group)
            offset = 0
            for g in sp_grads:
                n = g.numel()
                g.copy_(flat[offset : offset + n].view_as(g))
                offset += n
    report_norm = None
    if report_global_norm:
        if ps is None:
            raise ValueError("`ps` is required when `report_global_norm=True`.")
        report_norm = compute_global_grad_norm(model, ps, is_expert_param=is_expert_param)
    if use_dist_opt:
        optimizer.finish_grad_sync()
        return optimizer.clip_grad_norm()
    local_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return report_norm if report_norm is not None else local_norm


def compute_global_grad_norm(
    model, ps: ParallelState, *, is_expert_param: ExpertClassifierFn = default_expert_classifier
) -> torch.Tensor:
    """Compute benchmark global grad norm with dist-opt-aligned reduction order."""
    dense_sq = _bucketed_grad_sq_sum(
        model,
        include_param=lambda name: not is_expert_param(name),
        replica_group=ps.dp_cp_group,
        replica_size=ps.dp_cp_size,
    )
    expert_sq = _bucketed_grad_sq_sum(
        model,
        include_param=is_expert_param,
        replica_group=ps.ep_dp_group,
        replica_size=ps.expert_dp_size,
    )

    if ps.tp_size > 1 and ps.tp_group is not None:
        dist.all_reduce(dense_sq, group=ps.tp_group)

    if ps.ep_size > 1 and ps.ep_group is not None:
        dist.all_reduce(expert_sq, group=ps.ep_group)
    if ps.etp_size > 1 and ps.etp_group is not None:
        dist.all_reduce(expert_sq, group=ps.etp_group)

    total_sq = dense_sq + expert_sq
    if ps.pp_size > 1 and ps.pp_group is not None:
        dist.all_reduce(total_sq, group=ps.pp_group)
    return total_sq.sqrt()


def _bucketed_grad_sq_sum(
    model,
    *,
    include_param: Callable[[str], bool],
    replica_group,
    replica_size: int,
    max_bucket_bytes: int = 80 * 1024 * 1024,
) -> torch.Tensor:
    """Accumulate squared norm after averaging replica grads within each bucket."""
    total_sq = torch.zeros(1, device="cuda")
    bucket: list[torch.Tensor] = []
    bucket_bytes = 0

    def flush_bucket() -> None:
        nonlocal bucket_bytes, bucket, total_sq
        if not bucket:
            return
        flat = torch.cat(bucket)
        if replica_size > 1 and replica_group is not None:
            dist.all_reduce(flat, group=replica_group)
            flat.div_(replica_size)
        total_sq += flat.float().norm().pow(2)
        bucket = []
        bucket_bytes = 0

    for name, param in model.named_parameters():
        if param.grad is None or not include_param(name):
            continue
        grad = param.grad.view(-1)
        grad_bytes = grad.numel() * grad.element_size()
        if bucket and bucket_bytes + grad_bytes > max_bucket_bytes:
            flush_bucket()
        bucket.append(grad)
        bucket_bytes += grad_bytes

    flush_bucket()
    return total_sq


def optimizer_step(optimizer) -> None:
    """Execute optimizer step."""
    optimizer.step()


__all__ = [
    "compute_and_clip_grad_norm",
    "compute_global_grad_norm",
    "optimizer_step",
    "run_microbatch_loop",
]
