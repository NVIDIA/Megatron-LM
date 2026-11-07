# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``apply_rotary_pos_emb_with_cos_sin`` (flash-attn rotary kernel and torch fallback).

Call site: ``megatron/core/models/common/embeddings/rope_utils.py``; used by the static
flash-decode prefill (``Attention._adjust_key_value_for_inference``) to RoPE queries and keys
before they are cached, so that the decode kernel finds RoPE'd keys.

Contract (from the RoPE definition and the call site's precomputed tables):

* Inputs: ``t [S, B, H, D]`` (any strides; production passes views), ``cos``/``sin [S, d_rot/2]``
  holding ``cos(theta[s, i])``/``sin(theta[s, i])`` for the token at row ``s``
  (``theta[s, i] = pos_s * base ** (-2 i / d_rot)``, from ``RotaryEmbedding.get_cos_sin``),
  ``rotary_interleaved``.
* Output ``[S, B, H, D]`` in t's dtype: each pair of channels ``(a, b)`` becomes
  ``(a cos - b sin, a sin + b cos)``; the pairs are ``(i, i + d_rot/2)`` (NeoX) or
  ``(2i, 2i + 1)`` (interleaved). Channels ``>= d_rot`` pass through unchanged. ``t`` is not
  modified.
* The function first rounds cos/sin to t's dtype; the tests pass tables already in t's dtype, so
  the reference sees the kernel's exact inputs.

Paths: the flash-attn Triton kernel ``flash_attn.layers.rotary.apply_rotary_emb`` when it imports,
else a torch fallback (``_apply_rotary_pos_emb_bshd`` on a table built from cos and sin), forced
here by setting ``rope_utils.apply_rotary_emb_flash = None``.

Reference: fp64 pairwise rotation from the definition (``attention_reference.rope_reference``).
Tolerance class: floating point, bounded by ``rope_lowp`` (the rotation with torch's elementwise
ops in t's dtype); pass-through channels exact.

Variants covered: both paths; ``d_rot`` = D, D/2, 16; both pairings; S = 1, 2, 17, 8193 (large
positions); bf16, fp16, fp32; contiguous and strided ``t``; D = 64, 128; GQA-sized head counts.

Main: the fallback rotates by ``cos(cos theta)`` and ``sin(sin theta)`` with a misplaced
interleave (d7, the audit's R11 record; fixed by upstream PR #7465, not one of ours), so every
``test_fallback_*`` case fails on main and passes once #7465 lands. The flash path is the control.
The fallback runs on CPU, so these cases need no GPU.
"""

import pytest
import torch

from megatron.core.models.common.embeddings import rope_utils
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.attention import attention_reference as ar

BASE = 10000.0


def _inputs(seq, batch, heads, dim, rot_dim, dtype, strided, device):
    g = ou.make_generator(device=device)
    if strided:
        # A [B, S, H, D] buffer seen as [S, B, H, D], as the flash-decode prefill passes views.
        t = torch.randn(batch, seq, heads, dim, device=device, generator=g).to(dtype)
        t = t.permute(1, 0, 2, 3)
    else:
        t = torch.randn(seq, batch, heads, dim, device=device, generator=g).to(dtype)
    # Positions run from a nonzero offset too, so large angles are exercised for short S.
    positions = torch.arange(seq, device=device) + (0 if seq > 2 else 8190)
    theta = ar.rope_angles(positions, rot_dim, BASE)
    return t, torch.cos(theta).to(dtype), torch.sin(theta).to(dtype)


def _check(t, cos, sin, interleaved, what):
    before = t.clone()
    out = rope_utils.apply_rotary_pos_emb_with_cos_sin(t, cos, sin, rotary_interleaved=interleaved)
    ou.assert_exact(t, before, what=f"{what}: input (read-only)")
    assert out.shape == t.shape and out.dtype == t.dtype, f"{what}: {out.shape} {out.dtype}"
    c, s = cos[:, None, None, :], sin[:, None, None, :]
    ref = ou.fp64_reference(ar.rope_reference, t, c, s, interleaved=interleaved)
    trusted = ar.rope_lowp(t, c, s, interleaved=interleaved)
    rot_dim = 2 * cos.shape[-1]
    ou.assert_exact(out[..., rot_dim:], t[..., rot_dim:], what=f"{what}: pass-through channels")
    flat = lambda x: x.reshape(-1, x.shape[-1])  # noqa: E731  (per (token, batch, head) row)
    ou.assert_close_to_reference(flat(out), flat(ref), flat(trusted), dim=-1, what=what)


def _cases():
    factors = {
        "seq": [1, 2, 17, 8193],
        "rot": ["D", "D/2", "16"],
        "interleaved": [False, True],
        "dtype": [torch.bfloat16, torch.float16, torch.float32],
        "strided": [False, True],
        "dim": [64, 128],
    }
    cases = ou.pairwise_cases(factors)
    return [
        pytest.param(c, id="-".join(f"{k}={str(v).replace('torch.', '')}" for k, v in c.items()))
        for c in cases
    ]


def _rot_dim(spec: str, dim: int) -> int:
    return {"D": dim, "D/2": dim // 2, "16": 16}[spec]


@pytest.mark.parametrize("case", _cases())
def test_fallback_matches_reference(case, monkeypatch):
    """The torch fallback (flash-attn's rotary module missing) rotates pairs by theta."""
    monkeypatch.setattr(rope_utils, "apply_rotary_emb_flash", None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rot_dim = _rot_dim(case["rot"], case["dim"])
    t, cos, sin = _inputs(
        case["seq"], 2, 4, case["dim"], rot_dim, case["dtype"], case["strided"], device
    )
    _check(t, cos, sin, case["interleaved"], f"fallback {case}")


def test_fallback_is_identity_at_position_zero(monkeypatch):
    """theta = 0 (cos 1, sin 0) leaves every channel unchanged, bit for bit."""
    monkeypatch.setattr(rope_utils, "apply_rotary_emb_flash", None)
    t = torch.randn(1, 2, 4, 64, generator=ou.make_generator()).to(torch.bfloat16)
    cos = torch.ones(1, 32, dtype=torch.bfloat16)
    sin = torch.zeros(1, 32, dtype=torch.bfloat16)
    for interleaved in (False, True):
        out = rope_utils.apply_rotary_pos_emb_with_cos_sin(t, cos, sin, interleaved)
        ou.assert_exact(out, t, what=f"fallback at theta = 0 (interleaved={interleaved})")


@pytest.mark.parametrize(
    "case", [pytest.param(c.values[0], id=c.id, marks=ou.requires_sm(ou.HOPPER)) for c in _cases()]
)
def test_flash_rotary_matches_reference(case):
    """flash-attn's rotary kernel (the production path) rotates pairs by theta."""
    if rope_utils.apply_rotary_emb_flash is None:
        if ou.session_target_sm() is not None:
            pytest.fail("flash_attn.layers.rotary is not importable", pytrace=False)
        pytest.skip("flash_attn.layers.rotary is not importable")
    rot_dim = _rot_dim(case["rot"], case["dim"])
    t, cos, sin = _inputs(
        case["seq"], 2, 4, case["dim"], rot_dim, case["dtype"], case["strided"], "cuda"
    )
    _check(t, cos, sin, case["interleaved"], f"flash rotary {case}")


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_flash_rotary_small_case_for_sanitizer():
    """A small strided case of the flash rotary kernel (compute-sanitizer memcheck case)."""
    if rope_utils.apply_rotary_emb_flash is None:
        pytest.fail("flash_attn.layers.rotary is not importable", pytrace=False)
    t, cos, sin = _inputs(3, 2, 2, 64, 32, torch.bfloat16, True, "cuda")
    _check(t, cos, sin, False, "flash rotary sanitizer case")
