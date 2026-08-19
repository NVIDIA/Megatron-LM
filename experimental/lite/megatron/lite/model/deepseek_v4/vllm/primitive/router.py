from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ._contract import own_visible_tensor


class _VLLMFixedRouteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, logits, visible_op, renormalize, route_scale):
        weights, ids = visible_op(logits)
        # The returned route IDs are subsequently consumed by the dispatcher,
        # whose permutation/remap path may reuse or mutate their storage.  The
        # fixed-active-set VJP must retain the original global expert IDs.
        ctx.save_for_backward(logits, ids.clone())
        ctx.renormalize = renormalize
        ctx.route_scale = route_scale
        ctx.mark_non_differentiable(ids)
        return own_visible_tensor(weights), ids

    @staticmethod
    def backward(ctx: Any, grad_weights, _grad_ids):
        logits, ids = ctx.saved_tensors
        with torch.enable_grad():
            replay = logits.detach().float().requires_grad_(True)
            scores = torch.sqrt(F.softplus(replay))
            selected = scores.gather(-1, ids.long())
            if ctx.renormalize:
                selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            selected = selected * ctx.route_scale
            (dlogits,) = torch.autograd.grad(selected, replay, grad_weights.float())
        return dlogits.to(logits.dtype), None, None, None


def fixed_route_vjp(visible_op, logits, *, renormalize: bool, route_scale: float):
    return _VLLMFixedRouteFunction.apply(
        logits, visible_op, renormalize, route_scale
    )
