# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for FlashInfer's ``CuteDslMoEWrapper(...).run`` (per-token NVFP4 MoE) through Megatron's
``prepare_routed_nvfp4_weights`` and ``flashinfer_routed_nvfp4_moe``
(``megatron/core/inference/moe/flashinfer_nvfp4.py``).

Contract (from the call site, MCore's MoE math and FlashInfer's documented API):

* Per token ``t``: ``out[t] = sum_k p[t, k] * W2[e] @ act(W1[e] @ x[t])`` over routes to local
  experts ``offset <= e < offset + E_local`` (-1 and non-local routes add nothing), with FP32
  routing weights. ``act`` is squared ReLU (non-gated ``W1 [E, I, H]``) or MCore's SwiGLU
  (gated ``W1 [E, 2I, H]`` stored ``[gate; up]``: ``silu(gate) * up``).
* Megatron quantizes the BF16 expert weights to NVFP4 (per-expert FP32 scale ``448 * 6 / amax``,
  E4M3 block scales per 16, H and I padded to 256, gated FC1 reordered to FlashInfer's 64-row
  ``[up, gate]`` interleave) and every token to per-token NVFP4 (per-token scale
  ``amax / (448 * 6)``); the kernel requantizes the FC2 input per token (``fc2_input_scale =
  1 / 2688``). Quantization error is the only approximation the contract allows.
* Output: BF16 ``[rows, H]``. ``out``, ``token_capacity`` and ``use_bounded_rows`` behave as for
  routed MXFP8 (only ``out[:rows]`` is written; the bounded prefix is ``min(capacity, T)``).
  The activation enum must match the weights' gating, else ValueError; non-FP32 probabilities
  raise TypeError; a wrong hidden size raises ValueError.
* The fused finalize (default) sums a token's routes with BF16 atomics; under deterministic
  algorithms or batch-invariant mode Megatron selects the two-stage finalize, which must then be
  bitwise reproducible and independent of the other tokens in the batch.
* Graph capture replays the eager result; refreshing the weights in place
  (``prepare_routed_nvfp4_weights(..., out=weights)``) reaches a captured graph. Inputs are read
  only.

Oracle: ``spec.moe_reference`` in fp64 on the unquantized BF16 inputs. The inputs make every
quantization lossless: NVFP4-exact ``W2`` and tokens (E2M1 values, a +-6 in every 16-block, block
exponents within 3 of the row's), and a rank-1 ``W1 = u v^T`` with ``u = +-2**-n`` in
constant-level blocks of 16 along I. Then each route's FC2 input is ``c**2`` (or ``c`` times
``|z_gate|`` for SwiGLU, whose gates are kept beyond |64| so ``silu(g) = g`` or 0 in FP32 and
FP64) times a pattern that per-token NVFP4 represents exactly for any ``c``. Per-token
magnitudes span 2**-13..2**13 (2**4..2**12 gated), so a per-token scale error shows (a cosine
check would not see it for Relu2: ``relu(s z)**2 = s**2 relu(z)**2``). Tolerance class: floating
point, per row, bounded by a trusted fp32 torch implementation whose weighted route outputs are
summed in BF16, as the fused finalize does. Two-stage finalize: bitwise replay and batch
invariance (metamorphic).

Variants: {Relu2, SwiGLU} x {fused, two-stage finalize}; tactics are chosen per token count, so
tokens cover 1, 127/128/129 (the 128-row ``moe_sort`` tile), 256, 512; top-k 1, 2, 8; H and I
256-aligned, padded (320, 384 -> 512) and production (2688, 1856); EP offsets 0, interior and
last. ``FLASHINFER_NVFP4_4OVER6*`` are unset. Hardware: Blackwell (SM100/103).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec

RELU2, SWIGLU = 6, 3  # FlashInfer ActivationType values Megatron passes
GATE_MIN = 64.0  # |gate| beyond this: sigmoid is exactly 1 (or 0) in FP32 and its silu exact


def _two_stage(monkeypatch):
    """Make Megatron select the two-stage finalize, as batch-invariant mode does."""
    from megatron.core.inference.moe import flashinfer_nvfp4

    monkeypatch.setattr(flashinfer_nvfp4, "is_batch_invariant_mode_enabled", lambda: True)


class Problem:
    """Structured per-token NVFP4 MoE inputs (see the module docstring), on CUDA."""

    def __init__(
        self,
        *,
        tokens,
        hidden,
        inter,
        experts,
        local,
        offset,
        top_k,
        gated,
        pattern="random",
        seed=0,
        v0=None,
    ):
        g = ou.make_generator(ou.DEFAULT_SEED + seed)
        self.experts, self.local, self.offset, self.gated = experts, local, offset, gated
        kw = dict(row_block=16, generator=g, levels=2, v_scale_exponents=(-1, 1), v0=v0)
        u, v = spec.rank1_factors(local, inter, hidden, **kw)
        self.v0 = v[0] / v[0, 0]
        w1 = u[:, :, None] * v[:, None, :]
        if gated:  # MCore order [gate; up]; both halves rank 1 with the same v
            up, _ = spec.rank1_factors(local, inter, hidden, **dict(kw, v0=self.v0))
            w1 = torch.cat([w1, up[:, :, None] * v[:, None, :]], dim=1)
        self.w1 = w1
        self.w2 = spec.nvfp4_exact_values(local * hidden, inter, generator=g).reshape(
            local, hidden, inter
        )
        rows = (4, 12) if gated else (-13, 13)  # gated: large tokens keep every gate large
        x = spec.nvfp4_exact_values(tokens, hidden, generator=g, row_exponents=rows)
        if gated:  # v0 . x > 0 and large: every gate is beyond GATE_MIN
            x = x.abs() * torch.sign(self.v0)
        empty_rows, empty_expert = (), None
        if pattern == "empty-rows":
            empty_rows = list(range(0, tokens, 3))
        elif pattern == "empty-expert":
            empty_expert = offset + local // 2
        self.ids, self.probs = spec.make_routing(
            tokens,
            top_k,
            experts,
            generator=g,
            device_="cuda",
            empty_expert=empty_expert,
            empty_rows=empty_rows,
            dropped_routes=(tokens * top_k) // 4 if pattern == "random" else 0,
        )
        self.x = x.to(torch.bfloat16).cuda()
        for name, t in (("tokens", x), ("W1", w1), ("W2", self.w2)):
            assert torch.equal(t.to(torch.bfloat16).double(), t), f"{name} not exact in BF16"
        if gated:
            gates = torch.einsum("eih,th->eti", w1[:, :inter], x)
            live = x.abs().amax(-1) > 0
            assert bool((gates[:, live].abs() >= GATE_MIN).all()), "a gate is below GATE_MIN"

    def weights(self, out=None):
        from megatron.core.inference.moe.flashinfer_nvfp4 import prepare_routed_nvfp4_weights

        return prepare_routed_nvfp4_weights(
            self.w1.to(torch.bfloat16).cuda(), self.w2.to(torch.bfloat16).cuda(), out=out
        )

    @property
    def activation(self):
        return spec.swiglu if self.gated else spec.relu2

    def reference(self, rows=None, inputs=None):
        inputs = self if inputs is None else inputs
        rows = inputs.x.shape[0] if rows is None else rows
        x, ids, probs = inputs.x[:rows], inputs.ids[:rows], inputs.probs[:rows]
        w1, w2 = self.w1.cuda(), self.w2.cuda()
        ref = ou.fp64_reference(
            spec.moe_reference,
            x,
            w1,
            w2,
            ids,
            probs,
            activation=self.activation,
            local_offset=self.offset,
        )
        with spec.exact_fp32_matmul():
            trusted = spec.moe_reference(
                x.float(),
                w1.float(),
                w2.float(),
                ids,
                probs,
                activation=self.activation,
                local_offset=self.offset,
                out_dtype=torch.bfloat16,
            )
        return ref, trusted

    def run(self, weights, inputs=None, **kw):
        from megatron.core.inference.moe.flashinfer_nvfp4 import flashinfer_routed_nvfp4_moe

        inputs = self if inputs is None else inputs
        tensors = (inputs.x, inputs.ids, inputs.probs)
        before = [t.clone() for t in tensors]
        out = flashinfer_routed_nvfp4_moe(
            *tensors,
            weights,
            num_experts=self.experts,
            local_expert_offset=self.offset,
            activation_type=SWIGLU if self.gated else RELU2,
            **kw,
        )
        for name, t, b in zip(("hidden", "routing", "probs"), tensors, before):
            ou.assert_exact(t, b, what=f"{name} is read-only")
        return out


def _check(problem, out, what, rows=None, inputs=None):
    ref, trusted = problem.reference(rows, inputs)
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape, (out.dtype, out.shape)
    return ou.assert_close_to_reference(out, ref, trusted, dim=-1, what=what)


def test_structured_inputs_are_lossless():
    """CPU check of the oracle's inputs: tokens and weights are NVFP4-exact (per token and per
    tensor), every gate is beyond GATE_MIN, and every route's FC2 input is per-token NVFP4-exact,
    for both activations."""
    g = ou.make_generator()
    inter, hidden = 96, 160
    for gated in (False, True):
        kw = dict(row_block=16, generator=g, levels=2, v_scale_exponents=(-1, 1))
        u, v = spec.rank1_factors(3, inter, hidden, **kw)
        v0 = v[0] / v[0, 0]
        x = spec.nvfp4_exact_values(
            32, hidden, generator=g, row_exponents=(4, 12) if gated else (-13, 13)
        )
        w1 = u[:, :, None] * v[:, None, :]
        z = torch.einsum("eih,th->eti", w1, x)
        if gated:
            x = x.abs() * torch.sign(v0)
            up, _ = spec.rank1_factors(3, inter, hidden, **dict(kw, v0=v0))
            w_up = up[:, :, None] * v[:, None, :]
            gate = torch.einsum("eih,th->eti", w1, x)
            assert bool((gate.abs() >= GATE_MIN).all()), "a gate is below GATE_MIN"
            a = torch.where(gate > 0, gate, torch.zeros_like(gate)) * torch.einsum(
                "eih,th->eti", w_up, x
            )
            w1 = torch.cat([w1, w_up], dim=1)
        else:
            a = spec.relu2(z)
        ou.assert_exact(spec.nvfp4_per_token_roundtrip(x), x, what="tokens per token")
        for e in range(3):
            ou.assert_exact(spec.nvfp4_weight_roundtrip(w1[e]), w1[e], what="W1 per tensor")
        a = a.reshape(-1, inter)
        ou.assert_exact(spec.nvfp4_per_token_roundtrip(a), a, what=f"FC2 input (gated={gated})")
    w2 = spec.nvfp4_exact_values(256, 96, generator=g)
    ou.assert_exact(spec.nvfp4_weight_roundtrip(w2), w2, what="W2 per tensor")


@pytest.mark.parametrize("bad", ["activation", "probs", "hidden"])
def test_call_site_rejects_bad_arguments(bad, monkeypatch):
    """Argument checks run before any FlashInfer kernel (CPU; FlashInfer's import is faked)."""
    from megatron.core.inference.moe import flashinfer_nvfp4 as m

    monkeypatch.setattr(m, "require_flashinfer_routed_nvfp4", lambda: None)
    monkeypatch.setattr(
        m,
        "ActivationType",
        type(
            "A",
            (),
            {
                "Swiglu": type("S", (), {"value": 3, "name": "Swiglu"}),
                "Relu2": type("R", (), {"value": 6, "name": "Relu2"}),
            },
        ),
        raising=False,
    )
    weights = type("W", (), {"gated": False, "hidden_size": 64})()
    x = torch.zeros(4, 32 if bad == "hidden" else 64, dtype=torch.bfloat16)
    ids = torch.zeros(4, 2, dtype=torch.int64)
    probs = torch.zeros(4, 2, dtype=torch.bfloat16 if bad == "probs" else torch.float32)
    activation = SWIGLU if bad == "activation" else RELU2
    with pytest.raises(TypeError if bad == "probs" else ValueError):
        m.flashinfer_routed_nvfp4_moe(
            x, ids, probs, weights, num_experts=8, local_expert_offset=0, activation_type=activation
        )


LAYOUTS = [(8, 8, 0), (8, 2, 2), (8, 2, 6), (64, 8, 56), (64, 16, 24)]
SHAPES = [(256, 256), (320, 384), (512, 256)]
CASES = ou.pairwise_cases(
    {
        "tokens": [1, 2, 127, 128, 129, 256, 512],
        "gated": [False, True],
        "finalize": ["fused", "two-stage"],
        "layout": LAYOUTS,
        "top_k": [1, 2, 8],
        "shape": SHAPES,
        "pattern": ["random", "empty-expert", "empty-rows"],
    }
)


def _case_id(c):
    e, l, o = c["layout"]
    act = "swiglu" if c["gated"] else "relu2"
    return (
        f"T{c['tokens']}-{act}-{c['finalize']}-E{e}l{l}o{o}-k{c['top_k']}"
        f"-H{c['shape'][0]}I{c['shape'][1]}-{c['pattern']}"
    )


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("case", [pytest.param(c, id=_case_id(c)) for c in CASES])
def test_matches_fp64_moe(case, monkeypatch):
    spec.require_flashinfer()
    spec.clear_nvfp4_recipe_env(monkeypatch)
    if case["finalize"] == "two-stage":
        _two_stage(monkeypatch)
    e, local, offset = case["layout"]
    top_k = min(case["top_k"], e - 1 - (case["pattern"] == "empty-expert"))
    problem = Problem(
        tokens=case["tokens"],
        hidden=case["shape"][0],
        inter=case["shape"][1],
        experts=e,
        local=local,
        offset=offset,
        top_k=top_k,
        gated=case["gated"],
        pattern=case["pattern"],
    )
    weights = problem.weights()
    out = problem.run(weights)
    _check(problem, out, what=_case_id(case))
    if case["finalize"] == "two-stage":
        ou.assert_exact(problem.run(weights), out, what="two-stage finalize replays bitwise")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("gated", [False, True], ids=["relu2", "swiglu"])
def test_production_shape(gated, monkeypatch):
    """H = 2688 (padded to 2816), I = 1856 (padded to 2048), 128 experts, 8 local at 64, top-6."""
    spec.require_flashinfer()
    spec.clear_nvfp4_recipe_env(monkeypatch)
    problem = Problem(
        tokens=256, hidden=2688, inter=1856, experts=128, local=8, offset=64, top_k=6, gated=gated
    )
    _check(problem, problem.run(problem.weights()), what="production")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.sanitizer
@pytest.mark.parametrize("bounded", [True, False], ids=["bounded", "unbounded"])
def test_bounded_rows_write_only_the_prefix(bounded, monkeypatch):
    spec.require_flashinfer()
    spec.clear_nvfp4_recipe_env(monkeypatch)
    problem = Problem(
        tokens=512, hidden=256, inter=256, experts=8, local=2, offset=2, top_k=2, gated=False
    )
    problem.ids[40:64] = -1
    weights = problem.weights()
    out = ou.guarded_empty((512, 256), torch.float32)
    result = problem.run(weights, out=out.tensor, token_capacity=64, use_bounded_rows=bounded)
    assert result is out.tensor
    ou.check_guards(out=out)
    rows = 64 if bounded else 512
    head = out.tensor[:rows]
    ou.assert_exact(head, head.bfloat16().float(), what="out holds the BF16 kernel output")
    _check(problem, head.bfloat16(), what="bounded prefix" if bounded else "all rows", rows=rows)
    if bounded:
        ou.assert_poisoned(out.tensor[64:], what="rows past the capacity")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("gated", [False, True], ids=["relu2", "swiglu"])
def test_cuda_graph_replay_and_weight_refresh(gated, monkeypatch):
    spec.require_flashinfer()
    spec.clear_nvfp4_recipe_env(monkeypatch)
    from megatron.core.inference.moe.flashinfer_nvfp4 import flashinfer_routed_nvfp4_moe

    shape = dict(
        tokens=128, hidden=256, inter=256, experts=8, local=4, offset=4, top_k=2, gated=gated
    )
    first = Problem(**shape, seed=0)
    second = Problem(**shape, pattern="empty-expert", seed=1, v0=first.v0)
    third = Problem(**shape, pattern="empty-rows", seed=2, v0=first.v0)
    weights = first.weights()
    x, ids, probs = first.x.clone(), first.ids.clone(), first.probs.clone()
    act = SWIGLU if gated else RELU2

    def call():
        return flashinfer_routed_nvfp4_moe(
            x, ids, probs, weights, num_experts=8, local_expert_offset=4, activation_type=act
        )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = call()

    def replay(inputs):
        x.copy_(inputs.x)
        ids.copy_(inputs.ids)
        probs.copy_(inputs.probs)
        graph.replay()
        return static_out.clone()

    for inputs in (second, third):
        _check(first, replay(inputs), what="graph replay", inputs=inputs)
    second.weights(out=weights)  # refresh in place
    _check(second, replay(third), what="graph replay after refresh", inputs=third)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("gated", [False, True], ids=["relu2", "swiglu"])
def test_two_stage_finalize_is_batch_invariant(gated, monkeypatch):
    """Metamorphic: with the two-stage finalize (batch-invariant mode), a token's output is
    bitwise the same alone and at the last row of batches crossing the 128-row tile; random
    BF16 inputs, so quantization and reduction order would show."""
    spec.require_flashinfer()
    spec.clear_nvfp4_recipe_env(monkeypatch)
    _two_stage(monkeypatch)
    from megatron.core.inference.moe.flashinfer_nvfp4 import (
        flashinfer_routed_nvfp4_moe,
        prepare_routed_nvfp4_weights,
    )

    g = ou.make_generator()
    local, hidden, inter = 4, 512, 256
    rows1 = 2 * inter if gated else inter
    w1 = (torch.randn(local, rows1, hidden, generator=g) / hidden**0.5).bfloat16().cuda()
    w2 = (torch.randn(local, hidden, inter, generator=g) / inter**0.5).bfloat16().cuda()
    weights = prepare_routed_nvfp4_weights(w1, w2)
    batch = 512
    x = torch.randn(batch, hidden, generator=g).bfloat16().cuda()
    ids, probs = spec.make_routing(batch, 2, 8, generator=g, device_="cuda")

    def run(rows):
        order = torch.arange(rows, device="cuda")
        order[[0, rows - 1]] = order[[rows - 1, 0]]
        out = flashinfer_routed_nvfp4_moe(
            x[order],
            ids[order],
            probs[order],
            weights,
            num_experts=8,
            local_expert_offset=4,
            activation_type=SWIGLU if gated else RELU2,
        )
        return out[rows - 1]

    alone = run(1)
    for rows in (2, 127, 128, 129, 512):
        ou.assert_exact(run(rows), alone, what=f"token at the last of {rows} rows")
