# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The THD CSA output projection, without the round trip through BSHD.

Reaching the shared BSHD projection from the THD attention output cost three
full materialisations of the largest tensor in the module: a
``permute(...).contiguous()`` into ``[1, np, total, hn]``, an unfused inverse
RoPE that concatenates the untouched nope channels back onto the rotated ones,
and a ``transpose(1, 2).reshape(...)`` that copies again on the way out. A
per-module NVTX profile put ``direct_copy`` and ``CatArrayBatchedCopy`` at 64%
of ``F::CompressedSparseAttention``, and mcore's DSv4 attention does none of it
-- it leaves the output in THD and calls
``fused_mla_rope_out_of_place(..., inverse=True)``.

Two things have to hold for that substitution, and neither is visible in a
shape: the grouping must be reachable by a plain ``view``, and Core's fused
inverse must agree with the ``apply_partial_rope(cos, -sin)`` it replaces.
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.modules.attention.csa import apply_partial_rope

pytestmark = [pytest.mark.mlite]

TOTAL, HEADS, HEAD_DIM, GROUPS = 17, 8, 6, 2
ROPE_DIM = 4


def test_thd_grouping_is_a_free_view_of_the_bshd_reshape() -> None:
    """``(total, np, hn) -> (1, total, g, per*hn)`` must be the removed reshape.

    The dropped route went out through ``[1, np, total, hn]`` and came back with
    ``transpose(1, 2).reshape(...)``. If the two disagree on element order the
    output projection silently mixes heads across groups, which is finite,
    correctly shaped and wrong -- so this is asserted bitwise.
    """
    per = HEADS // GROUPS
    t = torch.randn(TOTAL, HEADS, HEAD_DIM)
    view = t.view(1, TOTAL, GROUPS, per * HEAD_DIM)
    reshape = (
        t.permute(1, 0, 2)
        .unsqueeze(0)
        .contiguous()
        .transpose(1, 2)
        .reshape(1, TOTAL, GROUPS, per * HEAD_DIM)
    )
    assert torch.equal(view, reshape)


def test_grouping_is_head_major_not_head_dim_major() -> None:
    """Guard the guard: a group axis taken over the wrong stride must differ.

    Without this, a regression that grouped the head-dim axis instead of the
    head axis would still produce the right shape and pass the test above on any
    input where the two happen to coincide.
    """
    per = HEADS // GROUPS
    t = torch.randn(TOTAL, HEADS, HEAD_DIM)
    correct = t.view(1, TOTAL, GROUPS, per * HEAD_DIM)
    transposed = t.transpose(1, 2).reshape(1, TOTAL, GROUPS, per * HEAD_DIM)
    assert not torch.equal(correct, transposed)


def _cos_sin(dtype: torch.dtype, device: str):
    """Full-length ``cat(freqs, freqs)`` tables, as the CSA builders return."""
    pos = torch.arange(TOTAL, device=device, dtype=torch.float32)
    inv = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float32) / ROPE_DIM)
    )
    freqs = pos.unsqueeze(-1) * inv.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


@pytest.mark.gpus(1)
def test_fused_inverse_rope_matches_apply_partial_rope() -> None:
    """Core's fused inverse must agree with ``apply_partial_rope(cos, -sin)``.

    bf16 through a rotate-and-concatenate is worth about a percent of the
    signal, so the bound is set at that scale rather than at an ulp; the point
    of the test is that the two undo the *same* rotation, which a wrong
    interleaving convention or a wrong sign would break by order unity.
    """
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_out_of_place

    torch.manual_seed(0)
    dtype = torch.bfloat16
    cu = torch.tensor([0, TOTAL], device="cuda", dtype=torch.int32)
    cos, sin = _cos_sin(dtype, "cuda")
    nope = HEAD_DIM - ROPE_DIM
    context = torch.randn(TOTAL, HEADS, HEAD_DIM, device="cuda", dtype=dtype)

    fused = fused_mla_rope_out_of_place(
        context, cos, sin, nope, ROPE_DIM, cu, 0, 1, inverse=True, remove_interleaving=True
    )
    eager = apply_partial_rope(
        context.permute(1, 0, 2).unsqueeze(0), cos.unsqueeze(0), -sin.unsqueeze(0), ROPE_DIM
    )
    eager = eager.squeeze(0).permute(1, 0, 2)
    scale = eager.float().abs().max()
    assert (fused.float() - eager.float()).abs().max() < 2e-2 * scale


@pytest.mark.gpus(1)
def test_forward_rotation_is_not_mistaken_for_the_inverse() -> None:
    """Negative control: ``inverse=False`` must not satisfy the bound above.

    The rotation is norm-preserving, so a dropped ``inverse`` flag changes no
    shape, no dtype and no magnitude -- only the values. Without this control
    the equivalence test could pass on a fixture with near-zero positions.
    """
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_out_of_place

    torch.manual_seed(0)
    dtype = torch.bfloat16
    cu = torch.tensor([0, TOTAL], device="cuda", dtype=torch.int32)
    cos, sin = _cos_sin(dtype, "cuda")
    nope = HEAD_DIM - ROPE_DIM
    context = torch.randn(TOTAL, HEADS, HEAD_DIM, device="cuda", dtype=dtype)

    wrong = fused_mla_rope_out_of_place(
        context, cos, sin, nope, ROPE_DIM, cu, 0, 1, inverse=False, remove_interleaving=True
    )
    eager = apply_partial_rope(
        context.permute(1, 0, 2).unsqueeze(0), cos.unsqueeze(0), -sin.unsqueeze(0), ROPE_DIM
    )
    eager = eager.squeeze(0).permute(1, 0, 2)
    scale = eager.float().abs().max()
    assert (wrong.float() - eager.float()).abs().max() > 2e-2 * scale
