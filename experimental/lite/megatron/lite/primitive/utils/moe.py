# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE routing, permutation, and router GEMM helpers for MLite primitives."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.permutation import moe_permute as fused_permute
from transformer_engine.pytorch.permutation import (
    moe_permute_and_pad_with_probs as fused_permute_and_pad_with_probs,
)
from transformer_engine.pytorch.permutation import (
    moe_permute_with_probs as fused_permute_with_probs,
)
from transformer_engine.pytorch.permutation import moe_unpermute as fused_unpermute
from transformer_engine.pytorch.router import (
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
    fused_topk_with_score_function,
)


def _te_general_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    *,
    layout: str = "TN",
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    grad: bool = False,
):
    kwargs = dict(
        out_dtype=out_dtype,
        quantization_params=None,
        gelu=None,
        gelu_in=None,
        accumulate=False,
        layout=layout,
        out=out,
        bias=bias,
        use_split_accumulator=False,
        grad=grad,
        ub=None,
        ub_type=None,
        extra_output=None,
        bulk_overlap=False,
    )
    return general_gemm(a, b, **kwargs)


def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
    *,
    fused: bool = False,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if padding_mask is not None:
        probs = probs * padding_mask.unsqueeze(-1)

    if fused:
        return fused_moe_aux_loss(
            probs=probs,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=topk,
            num_experts=num_experts,
            coeff=moe_aux_loss_coeff,
        )

    aggregated_probs_per_expert = probs.sum(dim=0)
    return torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )


def permute(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
    tokens_per_expert: Optional[torch.Tensor] = None,
    align_size: int = 0,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if fused and probs is None:
        permuted_input, sorted_indices = fused_permute(
            tokens, routing_map, num_out_tokens=num_out_tokens
        )
        return permuted_input, None, sorted_indices, None, tokens_per_expert

    if fused and probs is not None:
        if tokens_per_expert is not None and align_size > 0:
            return fused_permute_and_pad_with_probs(
                tokens, probs, routing_map, tokens_per_expert, align_size
            )
        output, permuted_probs, row_id_map = fused_permute_with_probs(
            tokens, probs, routing_map, num_out_tokens=num_out_tokens
        )
        return output, permuted_probs, row_id_map, None, tokens_per_expert

    num_tokens, _hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None
    if drop_and_pad and num_out_tokens is not None:
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
            :, :capacity
        ].contiguous()
        sorted_indices = sorted_indices.view(-1)

        if probs is not None:
            probs_t_1d = probs.T.contiguous().view(-1)
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1d = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            permuted_probs = probs_t_1d.index_select(0, indices_1d)
    else:
        if num_out_tokens is None:
            raise AssertionError("num_out_tokens is required for argsort-based permute")

        routing_map = routing_map.bool().T.contiguous()
        flat_sorted = routing_map.reshape(-1).argsort(descending=True, stable=True)
        flat_sorted = flat_sorted[:num_out_tokens]
        sorted_indices = flat_sorted % num_tokens

        if probs is not None:
            permuted_probs = probs.T.contiguous().reshape(-1)[flat_sorted]

    return (
        tokens.index_select(0, sorted_indices),
        permuted_probs,
        sorted_indices,
        None,
        tokens_per_expert,
    )


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: Optional[torch.Tensor] = None,
    routing_map: Optional[torch.Tensor] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
    pad_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if fused:
        kwargs = {}
        if pad_offsets is not None:
            kwargs["pad_offsets"] = pad_offsets
        return fused_unpermute(
            permuted_tokens,
            sorted_indices,
            merging_probs=probs,
            restore_shape=restore_shape,
            **kwargs,
        )

    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)
            probs_t_1d = probs.T.contiguous().view(-1)
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1d = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)
            permuted_probs = probs_t_1d.index_select(0, indices_1d)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    if torch.are_deterministic_algorithms_enabled():
        output_tokens.index_add_(0, sorted_indices, permuted_tokens)
    else:
        output_tokens.scatter_add_(
            0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens
        )
    return output_tokens.to(dtype=input_dtype)


def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    group_scores = (
        scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )
    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    return torch.topk(masked_scores, k=topk, dim=-1)


def topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    fused: bool = False,
    dense_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    num_tokens, num_experts = logits.shape
    if fused:
        return fused_topk_with_score_function(
            logits=logits,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
        )

    def compute_topk(
        scores: torch.Tensor,
        k: int,
        groups: Optional[int] = None,
        groups_topk: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if groups_topk:
            assert groups is not None
            return group_limited_topk(
                scores=scores,
                topk=k,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=groups,
                group_topk=groups_topk,
            )
        return torch.topk(scores, k=k, dim=1, sorted=torch.is_grad_enabled())

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    elif score_function in ("sigmoid", "sqrtsoftplus"):
        if score_function == "sigmoid":
            scores = torch.sigmoid(logits.float())
        else:
            scores = torch.nn.functional.softplus(logits.float()).sqrt()
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias.float()
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor
    probs = probs.type_as(logits)

    if dense_output:
        return probs, top_indices

    if torch.are_deterministic_algorithms_enabled():
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)
        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_(
            (rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False
        )
        routing_map = routing_map.bool()
    else:
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_probs, routing_map


def compute_routing_scores_for_aux_loss(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
    *,
    fused: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if fused:
        routing_map, scores = fused_compute_score_for_moe_aux_loss(
            logits=logits, topk=topk, score_function=score_function
        )
    else:
        if score_function == "softmax":
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        elif score_function == "sigmoid":
            scores = torch.sigmoid(logits.float())
            scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        elif score_function == "sqrtsoftplus":
            scores = torch.nn.functional.softplus(logits.float()).sqrt()
            scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            raise ValueError(f"Invalid score_function: {score_function}")
        _, top_indices = torch.topk(scores, k=topk, dim=1)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    if padding_mask is not None:
        valid_mask = (~padding_mask).unsqueeze(-1)
        routing_map = routing_map * valid_mask
        scores = scores * valid_mask
    return routing_map, scores


class RouterGatingLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        router_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.save_for_backward(inp, weight, bias)
        ctx.router_dtype = router_dtype
        ctx.input_dtype = inp.dtype
        ctx.weight_dtype = weight.dtype
        inp_shape = inp.shape
        inp = inp.view(-1, inp_shape[-1])

        gemm_out = None
        if router_dtype != torch.float64:
            gemm_out = _te_general_gemm(weight, inp, router_dtype, layout="TN", bias=bias)
        if gemm_out is not None:
            output = gemm_out[0]
        elif bias is None:
            output = torch.mm(inp.to(router_dtype), weight.to(router_dtype).t())
        else:
            output = torch.addmm(
                bias.to(router_dtype), inp.to(router_dtype), weight.to(router_dtype).t()
            )
        return output.view(*inp_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, weight, bias = ctx.saved_tensors
        inp_shape = inp.shape
        grad_shape = grad_output.shape
        inp = inp.view(-1, inp_shape[-1])
        grad_output = grad_output.view(-1, grad_shape[-1])

        grad_input_out = grad_weight_out = None
        if ctx.router_dtype != torch.float64:
            grad_input_out = _te_general_gemm(
                weight.to(ctx.router_dtype), grad_output, ctx.router_dtype, layout="NN", grad=True
            )
            grad_weight_out = _te_general_gemm(
                inp.to(ctx.router_dtype), grad_output, ctx.router_dtype, layout="NT", grad=True
            )
        if grad_input_out is not None and grad_weight_out is not None:
            grad_input = grad_input_out[0].to(ctx.input_dtype)
            grad_weight = grad_weight_out[0].to(ctx.weight_dtype)
        else:
            grad_input = torch.mm(grad_output, weight.to(ctx.router_dtype)).to(ctx.input_dtype)
            grad_weight = torch.mm(grad_output.t(), inp.to(ctx.router_dtype)).to(ctx.weight_dtype)
        grad_bias = grad_output.sum(dim=0).to(ctx.weight_dtype) if bias is not None else None
        return grad_input.view(*inp_shape), grad_weight, grad_bias, None


def router_gating_linear(
    inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, router_dtype: torch.dtype
) -> torch.Tensor:
    return RouterGatingLinearFunction.apply(inp, weight, bias, router_dtype)


__all__ = [
    "compute_routing_scores_for_aux_loss",
    "group_limited_topk",
    "permute",
    "router_gating_linear",
    "switch_load_balancing_loss_func",
    "topk_routing_with_score_function",
    "unpermute",
]
