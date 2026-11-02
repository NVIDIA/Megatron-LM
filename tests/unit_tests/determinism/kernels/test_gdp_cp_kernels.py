# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay the FLA context-parallel kernels the GDP chunkwise CP backend launches directly.

``megatron/core/ssm/context_parallel/gdp.py`` launches FLA's CP summary kernels (forward
``[delta_S | M]``, backward ``[gamma | M^T]``) and its CP merge kernel itself, outside FLA's
wrappers, for both BLH (several sequences per launch) and varlen boundary fragments.
"""

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

try:
    from megatron.core.ssm.context_parallel import gdp

    HAVE_FLA_GDP_CP = True
except ImportError:
    HAVE_FLA_GDP_CP = False

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    pytest.mark.skipif(not HAVE_FLA_GDP_CP, reason="FLA GDP CP kernels are not installed"),
]

BATCH, LENGTH, KEY_HEADS, VALUE_HEADS, VALUE_DIM = 2, 200, 2, 4, 64


def _fragments(key_dim, varlen):
    seeded()
    shape = (BATCH, LENGTH)
    k = torch.nn.functional.normalize(
        torch.randn(*shape, KEY_HEADS, key_dim, device="cuda"), dim=-1
    )
    beta = torch.rand(*shape, VALUE_HEADS, 1, device="cuda")
    tensors = dict(
        q=torch.randn(*shape, KEY_HEADS, key_dim, device="cuda"),
        k=k,
        # beta * k keeps every per-token transition a contraction, as the WY form does.
        w=beta * k.repeat_interleave(VALUE_HEADS // KEY_HEADS, dim=2),
        u=torch.randn(*shape, VALUE_HEADS, VALUE_DIM, device="cuda"),
        do=torch.randn(*shape, VALUE_HEADS, VALUE_DIM, device="cuda"),
        dv=torch.randn(*shape, VALUE_HEADS, VALUE_DIM, device="cuda"),
    )
    # Activations in BF16 as the backend receives them; the log-space gate cumsum stays FP32.
    tensors = {name: t.bfloat16() for name, t in tensors.items()}
    tensors["g"] = -0.05 * torch.rand(*shape, VALUE_HEADS, device="cuda").cumsum(1)
    if not varlen:
        return tensors, None
    tensors = {name: t.flatten(0, 1).unsqueeze(0) for name, t in tensors.items()}
    return tensors, torch.tensor([0, 77, BATCH * LENGTH], dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("varlen", [False, True], ids=["blh", "varlen"])
@pytest.mark.parametrize("key_dim", [64, 96, 128])
def test_fla_gdp_cp_summaries_replay(key_dim, varlen):
    t, cu_seqlens = _fragments(key_dim, varlen)
    forward = assert_replays_bit_exact(
        lambda **x: gdp._compute_fragment_summary(**x, cu_seqlens=cu_seqlens),
        dict(k=t["k"], w=t["w"], u=t["u"], g=t["g"]),
        backward=False,
        contention=True,
        what=f"FLA GDP CP forward summary [K={key_dim}, {'varlen' if varlen else 'blh'}]",
    )
    backward = assert_replays_bit_exact(
        lambda **x: gdp._compute_backward_fragment_summary(
            **x, scale=key_dim**-0.5, cu_seqlens=cu_seqlens
        ),
        dict(q=t["q"], k=t["k"], w=t["w"], g=t["g"], do=t["do"], dv=t["dv"]),
        backward=False,
        contention=True,
        what=f"FLA GDP CP backward summary [K={key_dim}, {'varlen' if varlen else 'blh'}]",
    )
    for summary in (forward[0]["out"], backward[0]["out"]):
        assert torch.isfinite(summary).all() and summary.abs().sum() > 0


@pytest.mark.parametrize("batched", [False, True], ids=["varlen", "blh"])
@pytest.mark.parametrize("forward", [True, False], ids=["fwd", "bwd"])
@pytest.mark.parametrize("key_dim", [64, 96, 128])
def test_fla_gdp_cp_merge_replays(key_dim, forward, batched):
    seeded()
    state = (BATCH,) if batched else ()
    summaries = 0.2 * torch.randn(
        3, *state, VALUE_HEADS, key_dim, VALUE_DIM + key_dim, device="cuda"
    )
    output = torch.empty(*state, VALUE_HEADS, key_dim, VALUE_DIM, device="cuda")
    merged = assert_replays_bit_exact(
        lambda packed_summaries: gdp._merge_affine_summaries(
            packed_summaries, output, forward
        ).clone(),
        dict(packed_summaries=summaries),
        backward=False,
        contention=True,
        what=f"FLA GDP CP merge [K={key_dim}, {'fwd' if forward else 'bwd'}]",
    )
    assert torch.isfinite(merged[0]["out"]).all()
