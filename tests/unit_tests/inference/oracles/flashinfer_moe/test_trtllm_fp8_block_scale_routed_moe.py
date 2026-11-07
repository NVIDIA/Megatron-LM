# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for FlashInfer's ``trtllm_fp8_block_scale_routed_moe`` (routed MXFP8 MoE) through
Megatron's ``flashinfer_routed_mxfp8_moe`` (``megatron/core/inference/moe/flashinfer_mxfp8.py``)
and the dispatch in ``InferenceGroupedMLP._flashinfer_forward``
(``megatron/core/transformer/moe/experts.py``).

Contract (from the call site, MCore's MoE math and FlashInfer's documented API):

* Per token ``t``: ``out[t] = sum_k bf16(p[t, k]) * W2[e] @ relu(W1[e] @ x[t])**2`` over the
  routes ``e = ids[t, k]`` with ``offset <= e < offset + E_local``; -1 and non-local routes add
  nothing. Routing weights are rounded to BF16 by the packed ``(expert << 16) | bf16(p)``
  format (documented), and consumed as given (no softmax or renormalization).
* Weights are the canonical MXFP8 stacks (``W1 [E_local, I, H]``, ``W2 [E_local, H, I]``) after
  ``prepare_routed_mxfp8_weights``; activations are quantized to MXFP8 by
  ``quantize_routed_mxfp8_input``; the FC2 input is requantized to MXFP8 inside the kernel.
  ``H`` and ``I`` are zero-padded to multiples of 128.
* Output: BF16 ``[rows, H]``. With ``out``, the processed rows are copied into ``out[:rows]``
  and nothing else in ``out`` is written. With ``token_capacity`` and ``use_bounded_rows`` only
  ``rows = min(capacity, T)`` leading rows are processed. ``T <= 0`` or ``capacity <= 0`` raise
  ValueError; non-FP32 probabilities raise TypeError; mismatched sizes raise ValueError.
* The output of a token does not depend on the other tokens or on the row count (the controller
  alternates capacity and full row counts, and the config accepts batch-invariant mode).
* Graph capture and replay give the eager result; refitting the weights in place
  (``prepare_routed_mxfp8_weights(..., out=routed)``) reaches a captured graph.
* Inputs are read only.

Oracle: ``spec.moe_reference`` in fp64 on the unquantized inputs. The inputs are built so that
every quantization is lossless (``spec``): MXFP8-exact ``W2``; a rank-1 ``W1 = u v^T`` with
``u = +-2**-n`` constant-level blocks of 32 along I; tokens with ``2**p`` power-of-two nonzeros
in distinct 32-blocks such that ``v . x`` is a power of two. Then ``relu(W1 x)**2`` is a power of
two times ``u**2``, which MXFP8 represents exactly whatever its scale rule (RCEIL or OCP floor),
and the fp64 MoE of the unquantized values is the exact contract. Tolerance class: floating point,
per row, bounded by a trusted implementation on the same device: torch in fp32 (TF32 off) whose
weighted route outputs are rounded to BF16 and summed in BF16, the least precise finalize the
contract allows (FlashInfer documents its finalize only as producing the BF16 output); the
kernel may be 2x less accurate than that. Batch invariance is a metamorphic property (bitwise),
on random MXFP8 inputs.

Variants and structure: one activation (Relu2, the only one config allows); the kernel's tile
along tokens is chosen per token count (``tune_max_num_tokens = max(rows, 128)``) and per
expert load, so tokens cover 1, 7/8/9, 63/64/65, 127/128/129 and 512 with E_local 1..16 and
top-k 1, 2, 6, 8; H and I cover 128 (I only), 256, padding (160, 96) and production (2688,
1856); EP offsets 0, interior and last; routing with an empty expert, all -1 rows and dropped
routes. FlashInfer 0.7.2.post1 has no TRT-LLM kernel config for a (padded) hidden size of 128
("No valid config found", GB200), which Megatron's config accepts: finding O4b-N3, tested on its
own (``test_hidden_128_is_served_or_rejected_by_megatron``) so the numeric cases use H >= 160.
Hardware: Blackwell (FlashInfer's TRT-LLM routed MoE is SM100-only).
"""

from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec

RELU2 = 6  # FlashInfer ActivationType.Relu2 (tllm_enums), what Megatron passes


class Problem:
    """Structured routed-MoE inputs (see the module docstring), on CUDA."""

    def __init__(
        self, *, tokens, hidden, inter, experts, local, offset, top_k, pattern, seed=0, v0=None
    ):
        g = ou.make_generator(ou.DEFAULT_SEED + seed)
        self.experts, self.local, self.offset, self.top_k = experts, local, offset, top_k
        u, v = spec.rank1_factors(
            local, inter, hidden, row_block=32, generator=g, v_scale_exponents=(-1, 1), v0=v0
        )
        self.w1 = u[:, :, None] * v[:, None, :]  # fp64 [E_local, I, H]
        self.w2 = spec.mx_exact_values(local * hidden, inter, generator=g).reshape(
            local, hidden, inter
        )
        self.v0 = v[0] / v[0, 0]  # the shared +-1, +-1/2 pattern (v[e] = +-2**j_e * v0)
        x = spec.few_hot_tokens(tokens, self.v0, generator=g)
        empty_rows, empty_expert, dropped = (), None, 0
        if pattern == "empty-rows":
            empty_rows = list(range(0, tokens, 3))
            x[1::4] = 0.0  # all-zero tokens too
        elif pattern == "empty-expert":
            empty_expert = offset + local // 2
        elif pattern == "dropped-routes":
            dropped = (tokens * top_k) // 3
        self.ids, self.probs = spec.make_routing(
            tokens,
            top_k,
            experts,
            generator=g,
            device_="cuda",
            empty_expert=empty_expert,
            empty_rows=empty_rows,
            dropped_routes=dropped,
        )
        self.x = x.to(torch.bfloat16).cuda()
        assert torch.equal(self.x.double().cpu(), x), "tokens are not exact in BF16"
        self.fc1 = spec.canonical_mxfp8(self.w1)
        self.fc2 = spec.canonical_mxfp8(self.w2)

    def weights(self):
        from megatron.core.inference.moe.flashinfer_mxfp8 import prepare_routed_mxfp8_weights

        return (
            prepare_routed_mxfp8_weights(spec.to_cuda(self.fc1)),
            prepare_routed_mxfp8_weights(spec.to_cuda(self.fc2)),
        )

    def reference(self, rows=None, inputs=None):
        """fp64 reference and trusted fp32 result for this problem's weights and the first
        ``rows`` tokens of ``inputs`` (another problem sharing ``v0``; default: this one)."""
        inputs = self if inputs is None else inputs
        rows = inputs.x.shape[0] if rows is None else rows
        x, ids = inputs.x[:rows], inputs.ids[:rows]
        probs = inputs.probs[:rows].to(torch.bfloat16)  # the packed routing format's precision
        w1, w2 = self.w1.cuda(), self.w2.cuda()
        ref = ou.fp64_reference(
            spec.moe_reference,
            x,
            w1,
            w2,
            ids,
            probs,
            activation=spec.relu2,
            local_offset=self.offset,
        )
        with spec.exact_fp32_matmul():
            trusted = spec.moe_reference(
                x.float(),
                w1.float(),
                w2.float(),
                ids,
                probs.float(),
                activation=spec.relu2,
                local_offset=self.offset,
                out_dtype=torch.bfloat16,
            )
        return ref, trusted

    def run(self, fc1, fc2, **kw):
        from megatron.core.inference.moe.flashinfer_mxfp8 import flashinfer_routed_mxfp8_moe

        inputs = (self.x, self.ids, self.probs)
        before = [t.clone() for t in inputs]
        out = flashinfer_routed_mxfp8_moe(
            self.x,
            self.ids,
            self.probs,
            fc1,
            fc2,
            num_experts=self.experts,
            local_expert_offset=self.offset,
            activation_type=RELU2,
            **kw,
        )
        for name, t, b in zip(("hidden", "routing", "probs"), inputs, before):
            ou.assert_exact(t, b, what=f"{name} is read-only")
        return out


EXPERT_LAYOUTS = [  # (num_experts, local experts, local offset)
    (8, 8, 0),
    (8, 2, 2),
    (8, 2, 6),
    (8, 1, 3),
    (64, 8, 0),
    (64, 8, 56),
    (64, 16, 24),
]
SHAPES = [(256, 128), (256, 256), (160, 96), (512, 384)]
TOKENS = ou.boundary_sizes(8, 64, 128, multiples=(), degenerate=True, production=(512,), minimum=1)
CASES = ou.pairwise_cases(
    {
        "tokens": TOKENS,
        "layout": EXPERT_LAYOUTS,
        "top_k": [1, 2, 8],
        "shape": SHAPES,
        "pattern": ["random", "empty-expert", "empty-rows", "dropped-routes"],
    }
)


def _case_id(c):
    e, l, o = c["layout"]
    return (
        f"T{c['tokens']}-E{e}l{l}o{o}-k{c['top_k']}-H{c['shape'][0]}I{c['shape'][1]}-{c['pattern']}"
    )


def _problem(c, seed=0):
    e, local, offset = c["layout"]
    top_k = min(c["top_k"], e - 1 - (c["pattern"] == "empty-expert"))  # FlashInfer: top_k < E
    return Problem(
        tokens=c["tokens"],
        hidden=c["shape"][0],
        inter=c["shape"][1],
        experts=e,
        local=local,
        offset=offset,
        top_k=top_k,
        pattern=c["pattern"],
        seed=seed,
    )


def _check(problem, out, what, rows=None, inputs=None):
    ref, trusted = problem.reference(rows, inputs)
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape, (out.dtype, out.shape)
    return ou.assert_close_to_reference(out, ref, trusted, dim=-1, what=what)


def test_structured_inputs_are_lossless():
    """CPU check of the oracle's inputs: weights and tokens are MXFP8-exact, and every route's
    activation (the FC2 input, padded with zeros to 128) is MXFP8-exact under both the RCEIL and
    the OCP floor scale rule."""
    g = ou.make_generator()
    u, v = spec.rank1_factors(4, 96, 160, row_block=32, generator=g, v_scale_exponents=(-1, 1))
    w1 = u[:, :, None] * v[:, None, :]
    for e in range(4):
        ou.assert_exact(spec.mxfp8_roundtrip(w1[e]), w1[e], what="W1")
    x = spec.few_hot_tokens(64, v[0] / v[0, 0], generator=g)
    ou.assert_exact(spec.mxfp8_roundtrip(x), x, what="tokens")
    a = spec.relu2(torch.einsum("eih,th->eti", w1, x)).reshape(-1, 96)
    a = torch.nn.functional.pad(a, (0, 32))
    ou.assert_exact(spec.mxfp8_roundtrip(a), a, what="FC2 input")
    amax = a.reshape(a.shape[0], -1, 32).abs().amax(-1)
    exponent = torch.floor(torch.log2(torch.where(amax > 0, amax, torch.ones_like(amax))))
    mantissa = amax / spec.pow2(exponent)
    assert bool(((amax == 0) | (mantissa <= 1.75)).all()), "OCP floor and RCEIL would differ"


@pytest.mark.parametrize("bad", ["rows", "capacity", "probs", "hidden"])
def test_call_site_rejects_bad_arguments(bad):
    """Argument checks run before any FlashInfer call (CPU)."""
    from megatron.core.inference.moe.flashinfer_mxfp8 import (
        FlashInferRoutedMXFP8Weight,
        flashinfer_routed_mxfp8_moe,
    )

    def weight(rows, cols):
        return FlashInferRoutedMXFP8Weight(
            data=torch.empty(1, 128, 128),
            scale=torch.empty(1, 512),
            logical_rows=rows,
            logical_cols=cols,
            padded_rows=128,
            padded_cols=128,
        )

    fc1, fc2 = weight(64, 96), weight(96, 64)
    tokens = 0 if bad == "rows" else 4
    x = torch.zeros(tokens, 96 if bad != "hidden" else 64, dtype=torch.bfloat16)
    ids = torch.zeros(tokens, 2, dtype=torch.int64)
    probs = torch.zeros(tokens, 2, dtype=torch.bfloat16 if bad == "probs" else torch.float32)
    kw = dict(token_capacity=0, use_bounded_rows=True) if bad == "capacity" else {}
    error = TypeError if bad == "probs" else ValueError
    with pytest.raises(error):
        flashinfer_routed_mxfp8_moe(
            x,
            ids,
            probs,
            fc1,
            fc2,
            num_experts=8,
            local_expert_offset=0,
            activation_type=RELU2,
            **kw,
        )


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("case", [pytest.param(c, id=_case_id(c)) for c in CASES])
def test_matches_fp64_moe(case):
    spec.require_flashinfer()
    problem = _problem(case)
    fc1, fc2 = problem.weights()
    _check(problem, problem.run(fc1, fc2), what=_case_id(case))


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("tokens", [256, 512])
def test_production_shape(tokens):
    """H = 2688, I = 1856 (128-padding on both), 128 experts, 8 local at offset 64, top-6."""
    spec.require_flashinfer()
    problem = Problem(
        tokens=tokens,
        hidden=2688,
        inter=1856,
        experts=128,
        local=8,
        offset=64,
        top_k=6,
        pattern="dropped-routes",
    )
    fc1, fc2 = problem.weights()
    _check(problem, problem.run(fc1, fc2), what=f"production T={tokens}")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("inter", [128, 256])
def test_hidden_128_is_served_or_rejected_by_megatron(inter):
    """Finding O4b-N3 (GB200): an expert hidden size of 128 (Megatron pads H to 128) passes
    Megatron's config validation for routed MXFP8, but FlashInfer 0.7.2.post1's TRT-LLM batched
    GEMM has no config for it and the first forward dies with a tvm InternalError ("No valid
    config found for the given problem shape"). The contract: the MoE is computed correctly, or
    Megatron rejects the size with a ValueError that names it - never an opaque kernel error."""
    spec.require_flashinfer()
    problem = Problem(
        tokens=64, hidden=128, inter=inter, experts=8, local=8, offset=0, top_k=1, pattern="random"
    )
    try:
        fc1, fc2 = problem.weights()
        out = problem.run(fc1, fc2)
    except ValueError as error:
        assert "hidden" in str(error).lower(), f"rejected without naming the hidden size: {error}"
        return
    except Exception as error:  # noqa: BLE001 - the finding is which error escapes
        pytest.fail(
            f"hidden size 128 reached FlashInfer and failed there ({type(error).__name__}: "
            f"{str(error).strip().splitlines()[0][:200]}); Megatron should reject it when the "
            "model is configured",
            pytrace=False,
        )
    _check(problem, out, what="hidden 128")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.sanitizer
@pytest.mark.parametrize("bounded", [True, False], ids=["bounded", "unbounded"])
def test_bounded_rows_write_only_the_prefix(bounded):
    """``token_capacity=64`` with 512 rows: bounded processes rows [:64] (rows 40..63 route
    nowhere) and leaves the rest of ``out`` (the FP32 RSV buffer in production) untouched;
    unbounded processes all 512."""
    spec.require_flashinfer()
    problem = Problem(
        tokens=512, hidden=256, inter=128, experts=8, local=2, offset=2, top_k=2, pattern="random"
    )
    problem.ids[40:64] = -1
    fc1, fc2 = problem.weights()
    out = ou.guarded_empty((512, 256), torch.float32)
    result = problem.run(fc1, fc2, out=out.tensor, token_capacity=64, use_bounded_rows=bounded)
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
def test_dispatch_through_inference_grouped_mlp():
    """``_flashinfer_forward`` derives ``num_experts = E_local * EP`` and ``offset = rank *
    E_local`` from the EP group and routes MXFP8 weights here (EP 4, rank 2, 2 local experts)."""
    spec.require_flashinfer()
    from flashinfer.fused_moe import ActivationType

    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    problem = Problem(
        tokens=65, hidden=256, inter=256, experts=8, local=2, offset=4, top_k=2, pattern="random"
    )
    fc1, fc2 = problem.weights()
    experts = SimpleNamespace(
        _uses_mxfp8_weights=True,
        _fc1_weight=fc1,
        _fc2_weight=fc2,
        _nvfp4_weights=None,
        _flashinfer_activation_type=ActivationType.Relu2,
        _activation_clamp_scale=None,
        _flashinfer_clamp_limit=None,
        _nvls_dispatcher=False,
        _flashinfer_mxfp8_token_capacity=None,
        num_local_experts=2,
        ep_group=SimpleNamespace(size=lambda: 4, rank=lambda: 2),
        config=SimpleNamespace(deterministic_mode=False, batch_invariant_mode=False),
    )
    out, bias = InferenceGroupedMLP._flashinfer_forward(
        experts, problem.x, problem.ids, problem.probs
    )
    assert bias is None
    _check(problem, out, what="InferenceGroupedMLP dispatch")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
def test_cuda_graph_replay_and_refit():
    """A captured call replays the eager result for new inputs, and an in-place refit of the
    weights reaches the graph."""
    spec.require_flashinfer()
    from megatron.core.inference.moe.flashinfer_mxfp8 import (
        flashinfer_routed_mxfp8_moe,
        prepare_routed_mxfp8_weights,
    )

    shape = dict(tokens=128, hidden=256, inter=256, experts=8, local=4, offset=4, top_k=2)
    first = Problem(**shape, pattern="random", seed=0)
    # Same v0, so every problem's tokens give lossless activations with every problem's weights.
    second = Problem(**shape, pattern="dropped-routes", seed=1, v0=first.v0)
    third = Problem(**shape, pattern="empty-expert", seed=2, v0=first.v0)
    fc1, fc2 = first.weights()
    x, ids, probs = first.x.clone(), first.ids.clone(), first.probs.clone()

    def call():
        return flashinfer_routed_mxfp8_moe(
            x, ids, probs, fc1, fc2, num_experts=8, local_expert_offset=4, activation_type=RELU2
        )

    call()  # warm up (JIT, tactic choice) outside capture
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
        replayed = replay(inputs)
        ou.assert_exact(replayed, call(), what="graph replay vs eager")
        _check(first, replayed, what="graph replay", inputs=inputs)
    # Refit: the second problem's weights, written into the captured buffers in place.
    for routed, canonical in ((fc1, second.fc1), (fc2, second.fc2)):
        prepare_routed_mxfp8_weights(spec.to_cuda(canonical), out=routed)
    _check(second, replay(third), what="graph replay after refit", inputs=third)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
def test_token_output_is_independent_of_the_batch():
    """Metamorphic: a token's output is bitwise the same alone and at row ``r`` of batches whose
    row counts cross the kernel's token tiles (random MXFP8 inputs, not the structured ones, so
    the reduction order would show)."""
    spec.require_flashinfer()
    from megatron.core.inference.moe.flashinfer_mxfp8 import (
        flashinfer_routed_mxfp8_moe,
        prepare_routed_mxfp8_weights,
    )

    g = ou.make_generator()
    local, hidden, inter = 4, 512, 384
    w1 = spec.mx_exact_values(local * inter, hidden, generator=g).reshape(local, inter, hidden)
    w2 = spec.mx_exact_values(local * hidden, inter, generator=g).reshape(local, hidden, inter)
    fc1 = prepare_routed_mxfp8_weights(spec.to_cuda(spec.canonical_mxfp8(w1)))
    fc2 = prepare_routed_mxfp8_weights(spec.to_cuda(spec.canonical_mxfp8(w2)))
    batch = 512
    x = (torch.randn(batch, hidden, generator=g) * 2.0**-4).to(torch.bfloat16).cuda()
    ids, probs = spec.make_routing(batch, 2, 8, generator=g, device_="cuda")

    def run(rows, at):
        order = torch.arange(rows, device="cuda")
        order[[0, at]] = order[[at, 0]]  # the probe token (row 0) moves to row `at`
        out = flashinfer_routed_mxfp8_moe(
            x[order],
            ids[order],
            probs[order],
            fc1,
            fc2,
            num_experts=8,
            local_expert_offset=4,
            activation_type=RELU2,
        )
        return out[at]

    alone = run(1, 0)
    for rows in (7, 8, 9, 64, 65, 128, 129, 512):
        ou.assert_exact(run(rows, rows - 1), alone, what=f"token at row {rows - 1} of {rows}")
