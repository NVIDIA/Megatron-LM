# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `InferenceTopKRouter`, the inference MoE router.

Kernel: `megatron/core/transformer/moe/router.py` (`InferenceTopKRouter._forward` under
`InferenceMode`), which dispatches to `torch.compile`'d `topk_routing_with_score_function`
(`moe_utils.py`), the Triton sigmoid top-k kernel (`inference/moe/sigmoid_topk.py`, backend
'triton', the default), eager routing in batch-invariant mode, or TE's fused router
(`moe_router_fusion`). Every MoE layer of an inference-optimized model routes through it.

Contract (written from the routing semantics, `TransformerConfig`'s documented router options and
the eval-mode `TopKRouter` the model was trained with):
- Input [S, B, H]; tokens in `input.view(-1, H)` order. Router logits z = input @ W^T computed
  in fp32 (`moe_router_dtype='fp32'`, required for inference-optimized models).
- Output `(probs, experts)`, both [T, topk] (T = S * B), probs fp32 and experts int64; the order
  of a token's k entries is unspecified, so they are compared as sets keyed by expert id.
- Score functions: softmax selects the k largest logits, probs = softmax over those k logits;
  softmax with pre-softmax selects by softmax(z) over all experts and keeps those values;
  sigmoid (and sqrtsoftplus) selects by s(z) + expert_bias (the bias steers selection only),
  probs = s(z) of the selected, divided by their sum when k > 1. A scaling factor multiplies
  the probs. Sinkhorn routing in eval mode selects and weights by sigmoid(z) (k = 1) or
  softmax(z) over all experts (k > 1), not renormalized, unscaled. Hash layers take the experts
  from the token-id table and weight them by the score function as above.
- Group-limited routing is not supported and must be rejected at construction.
- Quantile balancing (`moe_router_load_balancing_type='quantile_balancing'`, documented in
  `TransformerConfig` and `qb_dual_update`): micro-batch QB selects the k largest z - beta,
  with the per-expert bias beta (`qb_beta`), and weights those experts with the score function
  as above; global-batch QB (sigmoid only) selects the k largest sigmoid(z) + b with its derived
  bias b (`expert_bias`) and weights them with sigmoid(z) normalized. Inference reads both biases
  and never updates them or the QB accumulators; micro-batch QB with TE's fused router must be
  refused, not routed without beta.
- Router replay (`moe_enable_routing_replay`, `docs/api-guide/router_replay.md`): RECORD routes
  as without replay and records exactly the routed experts (as a tensor, into the first T rows of
  a static [max_tokens, k] buffer, or into one bank and layer of the double-buffered
  [2, max_tokens, layers, k] buffer), writing nothing else; REPLAY_FORWARD routes each token to
  its target experts, weighted by the score function as above; with no action, routing is
  unchanged. Under CUDA-graph replay the recording lands in the bank selected at replay time.
- Dispatch (exact, from `inference_moe_router_backend`'s documentation): backend 'triton' with
  sigmoid scores, an expert bias, fp32 logits, 1 < k <= min(32, E), E <= 512 and no
  batch-invariant mode, group-limited routing, router replay, micro-batch QB or fusion runs the
  Triton kernel (global-batch QB's bias is an expert bias); batch-invariant mode runs the routing
  eagerly; everything else runs the compiled routing.
- CUDA-graph replay with new inputs gives the eager bits.

Reference: fp64 routing written from the semantics above (`routing_reference`), on logits built
to be exact in bf16 (the router weight is the identity) with a selection margin, so that every
correct implementation selects the same experts. Trusted implementation: the same reference
evaluated in fp32 on the device. QB and replay have their own fp64 references
(`qb_reference`, `replay_reference`); QB logits keep the margin on z - beta, and z - beta is
exact in fp32 (`make_qb_beta`). The eval-mode TopKRouter outside InferenceMode is checked against
the same QB references (R06's cross-check). Tolerance class: exact for expert sets, shapes,
dtypes, dispatch, QB state and recorded indices (and every untouched buffer element); floating
point for probs (bounded over the tensor, factor 4, see PROB_FACTOR); bit-exact for CUDA-graph
replay. `test_gating_logits_are_fp32` checks the router GEMM itself against fp64.

Structure (read after the contract): `_forward` squeezes dim 1 of the logits (main), takes QB
precomputed indices, then chooses the Triton kernel (`_uses_sigmoid_topk_kernel`), eager
routing (batch-invariant) or the compiled routing, which may call TE's fused router. The
sigmoid kernel has variants by k (< 8 left-fold sum, >= 8 tree sum, `tl.topk` for k >= 8), E
(power-of-two padding up to 512) and rows per program, and it orders routing scores by integer
keys built from their bits, with a separate rule for negative scores. Covered: E in {8, 64, 256}
(and 1024 for the dispatch boundary), k in {1, 2, 6, 8, 32, E}, T in {0, 1, 7, 128, 129, 4096},
B in {1, 2, 3}, every score function, expert bias, scaling, sinkhorn, hash layers, fusion, all
three paths, CUDA-graph replay, and expert biases that make the k-th selected score (and the
order among the selected ones) negative (`test_negative_routing_scores_select_like_fp64`).
QB (`_forward`: micro-batch QB precomputes the indices, so it takes the compiled routing; global
QB's bias goes through the expert-bias route, so the Triton kernel where the shape allows):
micro-batch pairwise over 3 score functions x 3 paths x E {8, 64, 256} x k {2, 6, 8} (k < E) x
T {0, 1, 7, 128, 129} x scaling; global-batch pairwise over 3 paths x E x k {2, 6, 8, 32} x T x
scaling; CUDA-graph replay with an in-place bias update for both scopes on the compiled and
Triton backends; fusion refused. Router replay (`RouterReplay.get_replay_topk` inside the
compiled routing; `record_indices` has a tensor, a static-buffer and a banked index_put branch):
RECORD pairwise over 4 score kinds x 3 paths x {no action, tensor, static, banked} x E x
T {0, 1, 7, 129}; REPLAY_FORWARD pairwise over 4 kinds x 3 paths x E x T x scaling, with targets
that differ from the natural top-k; RECORD captured in a CUDA graph into the static and the
banked buffer, the bank switched between replays.

Main vs open PRs (#8562, d4-d9 "Return dense [num_tokens, topk] routing from InferenceTopKRouter
with router fusion and top-1", which also routes sinkhorn like the eval router):
- `test_top1_returns_dense_topk` and `test_batched_input_layout` fail on main: the compiled and
  BI paths return `probs.squeeze(1)` ([T] for k = 1) and `logits.squeeze(1)` leaves [S, B, E]
  for B > 1 (d9).
- `test_fused_router_returns_dense_topk` fails on main: the fused branch returns [T, E] (d4).
- `test_sinkhorn_matches_eval_router` fails on main: sinkhorn is ignored (step 3 R06-F1).
- `test_gating_logits_are_fp32` fails on main for batch-invariant backends 'triton' and
  'deepgemm': their TE GEMM replacement rounds the fp32-output router GEMM to bf16 (finding
  O8a-N3); no open PR changes it.
All other tests pass on main and on #8562.

Hardware: H100 (1 GPU), with Transformer Engine.
"""

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.moe.sigmoid_topk import is_sigmoid_topk_supported
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.moe import router as router_module
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.router import InferenceTopKRouter
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.test_utilities import Utils

pytestmark = [ou.requires_sm(ou.HOPPER)]

_DEVICE = "cuda"

KINDS = {
    "softmax": dict(score="softmax", pre_softmax=False, bias=False),
    "softmax_pre": dict(score="softmax", pre_softmax=True, bias=False),
    "sigmoid": dict(score="sigmoid", pre_softmax=False, bias=False),
    "sigmoid_bias": dict(score="sigmoid", pre_softmax=False, bias=True),
    "sqrtsoftplus": dict(score="sqrtsoftplus", pre_softmax=False, bias=False),
}
PATHS = ("compiled", "triton", "batch_invariant")
# fp32 probs: the Triton kernel's tl.sigmoid (and Inductor's generated exp) lower fp32 exp to the
# approximate ex2.approx after a log2(e) prescale (about 2**-22 relative), while the trusted
# torch ops use CUDA's expf (1-2 ulp), so a correct path may measure a few times torch's error.
# A wrong score function, normalization or scaling errs by 1e-3 or more. Probs are bounded over
# the whole tensor: they share one scale, and a token holds only k of them.
PROB_FACTOR = 4.0
SIGMOID_KERNEL_MAX_TOPK = 32  # inference/moe/sigmoid_topk.py: is_sigmoid_topk_supported
SIGMOID_KERNEL_MAX_EXPERTS = 512


@pytest.fixture(scope="module")
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def _fresh_state(model_parallel):
    """Every test compiles afresh (no recompile limit falls back to eager) with BI mode off."""
    torch._dynamo.reset()
    assert not bik.is_batch_invariant_mode_enabled(), "batch-invariant mode leaked into the test"
    yield
    assert not bik.is_batch_invariant_mode_enabled(), "the test left batch-invariant mode on"


def build_router(num_experts, topk, kind, *, backend="torch", scaling=None, hidden=None, **extra):
    """An InferenceTopKRouter whose gating weight is the identity unless ``hidden`` is given."""
    k = KINDS[kind]
    config = TransformerConfig(
        num_layers=2,
        hidden_size=hidden or num_experts,
        ffn_hidden_size=64,
        num_attention_heads=1,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=64,
        moe_router_topk=topk,
        moe_router_score_function=k["score"],
        moe_router_pre_softmax=k["pre_softmax"],
        moe_router_enable_expert_bias=k["bias"],
        moe_router_topk_scaling_factor=scaling,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type=extra.pop("load_balancing", "none"),
        moe_aux_loss_coeff=0.0,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        activation_func=squared_relu,
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="inference_optimized",
        use_cpu_initialization=True,
        inference_moe_router_backend=backend,
        **extra,
    )
    router = InferenceTopKRouter(config=config, pg_collection=get_default_pg_collection())
    router = router.cuda().eval()
    if hidden is None:
        with torch.no_grad():
            router.weight.data = torch.eye(num_experts, device=_DEVICE, dtype=torch.bfloat16)
    return router


def selection_keys(logits, kind, bias):
    """What each score function selects on (fp64 or fp32, from the contract)."""
    k = KINDS[kind]
    if k["score"] == "softmax":
        return logits  # softmax is monotone: the same experts as the logits
    scores = torch.sigmoid(logits) if k["score"] == "sigmoid" else _sqrtsoftplus(logits)
    return scores + bias if k["bias"] else scores


def _sqrtsoftplus(z):
    return torch.nn.functional.softplus(z).sqrt()


def weights_of(logits, experts, kind, scaling):
    """Probabilities of the given experts (from the contract)."""
    k = KINDS[kind]
    if k["score"] == "softmax" and k["pre_softmax"]:
        probs = torch.softmax(logits, dim=-1).gather(-1, experts)
    elif k["score"] == "softmax":
        probs = torch.softmax(logits.gather(-1, experts), dim=-1)
    else:
        scores = torch.sigmoid(logits) if k["score"] == "sigmoid" else _sqrtsoftplus(logits)
        probs = scores.gather(-1, experts)
        if experts.shape[-1] > 1:
            probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs * scaling if scaling else probs


def routing_reference(logits, topk, kind, bias, scaling):
    """(probs, experts), each [T, topk], experts sorted ascending per token."""
    keys = selection_keys(logits, kind, bias)
    experts = keys.topk(topk, dim=-1).indices.sort(dim=-1).values
    return weights_of(logits, experts, kind, scaling), experts


def sinkhorn_reference(logits, topk):
    """Eval-mode sinkhorn: sigmoid (k = 1) or softmax over all experts, unnormalized."""
    act = torch.sigmoid(logits) if topk == 1 else torch.softmax(logits, dim=-1)
    experts = act.topk(topk, dim=-1).indices.sort(dim=-1).values
    return act.gather(-1, experts), experts


def bf16_pool(span):
    """Every bf16 value v with 2**-4 <= |v| <= span / 2, ascending (1792 values for span 16)."""
    bits = torch.arange(0, 0x7F80, dtype=torch.int32).to(torch.int16)  # positive finite bf16
    values = bits.view(torch.bfloat16).double()
    values = values[(values >= 2.0**-4) & (values <= span / 2)]
    return torch.cat([-values.flip(0), values])


def make_logits(tokens, num_experts, topk, kind, bias, seed, keys=None):
    """bf16-exact logits with a selection margin, a different set of values for every token.

    Each token draws ``num_experts`` distinct bf16 values (in random order) from a pool, so the
    logits (and the identity gating GEMM's fp32 output) are exact and the probs differ from token
    to token: with one shared set, every token's probs would repeat the same few values and an
    rms bound would rest on those few. A token is redrawn until its k-th and (k+1)-th selection
    keys differ by more than 1e-4 of the k-th, far above fp32 rounding, so no correct
    implementation can select differently. ``keys`` (fp64 row -> selection keys) replaces the
    score function's keys, e.g. for quantile balancing's z - beta.
    """
    span = 16.0 if KINDS[kind]["score"] == "softmax" else 8.0
    pool = bf16_pool(span)
    assert pool.numel() >= num_experts, "not enough distinct bf16 values in the span"
    bias = None if bias is None else bias.detach().double().cpu()
    g = ou.make_generator(seed)
    rows = []
    for _ in range(tokens):
        for _attempt in range(1000):
            row = pool[torch.randperm(pool.numel(), generator=g)[:num_experts]]
            if topk >= num_experts:
                break
            row_keys = selection_keys(row[None], kind, bias) if keys is None else keys(row[None])
            top = row_keys.topk(topk + 1, dim=-1).values[0]
            if top[topk - 1] - top[topk] > 1e-4 * max(1.0, abs(float(top[topk - 1]))):
                break
        else:
            raise AssertionError("could not build a token with a selection margin")
        rows.append(row)
    logits = torch.stack(rows) if rows else torch.empty(0, num_experts, dtype=torch.float64)
    out = logits.to(torch.bfloat16)
    assert torch.equal(out.double(), logits), "grid values must be exact in bf16"
    return out.to(_DEVICE)


def make_bias(num_experts, seed=5):
    """Expert biases in [-0.25, 0.25], multiples of 1/64 (exact in bf16 and fp32)."""
    g = ou.make_generator(seed)
    return (torch.randint(-16, 17, (num_experts,), generator=g).double() / 64).float().to(_DEVICE)


def make_negative_bias(num_experts, topk, seed=9):
    """Expert biases that make the k-th largest routing score negative.

    ``topk // 2`` experts keep biases in [-1/4, 1/4], so their scores sigmoid(z) + b have either
    sign; every other expert's bias is in [-2, -1], so its score is negative (sigmoid < 1). At
    most ``topk // 2`` scores are positive: the k-th selected score, the margin below it and the
    order among the remaining selected experts are all decided among negative scores. Multiples
    of 1/64 (exact in bf16 and fp32).
    """
    g = ou.make_generator(seed)
    bias = torch.randint(-128, -63, (num_experts,), generator=g).double() / 64
    keep = torch.randperm(num_experts, generator=g)[: topk // 2]
    bias[keep] = torch.randint(-16, 17, (keep.numel(),), generator=g).double() / 64
    return bias.float().to(_DEVICE)


def run_router(router, x, path):
    """Route x [S, B, H] under InferenceMode on the given path."""
    with torch.no_grad(), InferenceMode.active():
        if path == "batch_invariant":
            with bik.set_batch_invariant_mode(True, backend="te_native"):
                return router(x)
        return router(x)


def check_routing(probs, experts, logits, topk, kind, bias, scaling, what):
    """Exact shapes, dtypes and expert sets; probs bounded against fp64 per token."""
    check_against(
        probs, experts, logits, topk, routing_reference, (topk, kind, bias, scaling), what
    )


def check_against(probs, experts, logits, topk, reference, args, what):
    """``check_routing`` against ``reference(logits, *args)``, a (probs, sorted experts) pair."""
    tokens = logits.shape[0]
    assert tuple(probs.shape) == (tokens, topk), f"{what}: probs {tuple(probs.shape)}"
    assert tuple(experts.shape) == (tokens, topk), f"{what}: experts {tuple(experts.shape)}"
    assert probs.dtype == torch.float32, f"{what}: probs dtype {probs.dtype}"
    assert experts.dtype == torch.int64, f"{what}: experts dtype {experts.dtype}"
    order = experts.argsort(dim=-1)
    got_experts, got_probs = experts.gather(-1, order), probs.gather(-1, order)
    ref_probs, ref_experts = ou.fp64_reference(reference, logits, *args)
    ou.assert_exact(got_experts, ref_experts, what=f"{what}: expert sets")
    trusted, trusted_experts = reference(logits.float(), *args)
    assert torch.equal(trusted_experts, ref_experts), "the fp32 trusted routing selects differently"
    ou.assert_close_to_reference(
        got_probs, ref_probs, trusted, factor=PROB_FACTOR, what=f"{what}: probs"
    )


def _setup(num_experts, topk, kind, tokens, seed, batch=1, **router_kwargs):
    router = build_router(num_experts, topk, kind, **router_kwargs)
    bias = None
    if KINDS[kind]["bias"]:
        bias = make_bias(num_experts)
        router.expert_bias.copy_(bias)
    logits = make_logits(tokens * batch, num_experts, topk, kind, bias, seed)
    x = logits.view(tokens, batch, num_experts)
    return router, bias, logits, x


def _id(case):
    return "-".join(f"{k}{v}" for k, v in case.items())


_CASES = ou.pairwise_cases(
    {
        "kind": ("softmax", "softmax_pre", "sigmoid", "sigmoid_bias"),
        "path": PATHS,
        "e": (8, 64, 256),
        "topk": (2, 6, 8, 32),
        "t": (0, 1, 7, 128, 129, 4096),
        "scaling": (None, 2.5),
    }
)


@pytest.mark.parametrize("case", _CASES, ids=[_id(c) for c in _CASES])
def test_routing_matches_fp64(case):
    """Every score function on every path: exact expert sets, probs against fp64."""
    num_experts, topk = case["e"], min(case["topk"], case["e"])
    backend = "torch" if case["path"] == "compiled" else "triton"
    router, bias, logits, x = _setup(
        num_experts, topk, case["kind"], case["t"], seed=1, backend=backend, scaling=case["scaling"]
    )
    probs, experts = run_router(router, x, case["path"])
    check_routing(probs, experts, logits, topk, case["kind"], bias, case["scaling"], _id(case))


_NEGATIVE_CASES = [(64, 2), (64, 6), (64, 8), (256, 32)]  # k < 8 and k >= 8 kernel variants


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "num_experts,topk", _NEGATIVE_CASES, ids=[f"e{e}-k{k}" for e, k in _NEGATIVE_CASES]
)
def test_negative_routing_scores_select_like_fp64(num_experts, topk, path):
    """sigmoid(z) + bias selects by value when the scores are negative: an expert bias (e.g. the
    aux-loss-free load-balancing bias) can push most scores below zero, and the k largest must
    still be selected, with probs from sigmoid(z) of those experts."""
    backend = "torch" if path == "compiled" else "triton"
    router = build_router(num_experts, topk, "sigmoid_bias", backend=backend)
    bias = make_negative_bias(num_experts, topk)
    router.expert_bias.copy_(bias)
    tokens = 129
    logits = make_logits(tokens, num_experts, topk, "sigmoid_bias", bias, seed=10)
    keys = selection_keys(logits.double(), "sigmoid_bias", bias.double())
    assert bool((keys.topk(topk, dim=-1).values[:, -1] < 0).all()), "a k-th score is not negative"
    probs, experts = run_router(router, logits.view(tokens, 1, num_experts), path)
    check_routing(
        probs,
        experts,
        logits,
        topk,
        "sigmoid_bias",
        bias,
        None,
        f"negative scores E={num_experts} k={topk} {path}",
    )


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kind", ["softmax_pre", "sigmoid", "sigmoid_bias"])
def test_top1_returns_dense_topk(kind, path):
    """k = 1 gives [T, 1] outputs. Fails on main (probs.squeeze(1), d9); passes with #8562."""
    backend = "torch" if path == "compiled" else "triton"
    for tokens in (1, 7, 128):
        router, bias, logits, x = _setup(64, 1, kind, tokens, seed=tokens, backend=backend)
        probs, experts = run_router(router, x, path)
        check_routing(
            probs, experts, logits, 1, kind, bias, None, f"top-1 {kind} {path} T={tokens}"
        )


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kind", ["softmax", "sigmoid_bias"])
@pytest.mark.parametrize("batch", [2, 3])
def test_batched_input_layout(batch, kind, path):
    """[S, B, H] inputs with B > 1 route every token (sequence-major order).

    Fails on main (logits.squeeze(1) keeps [S, B, E], d9); passes with #8562.
    """
    backend = "torch" if path == "compiled" else "triton"
    router, bias, logits, x = _setup(64, 6, kind, 5, seed=batch, batch=batch, backend=backend)
    probs, experts = run_router(router, x, path)
    check_routing(probs, experts, logits, 6, kind, bias, None, f"B={batch} {kind} {path}")


@pytest.mark.parametrize("path", ["compiled", "batch_invariant"])
@pytest.mark.parametrize(
    "kind,topk", [("softmax", 2), ("softmax_pre", 1), ("softmax_pre", 6), ("sigmoid_bias", 6)]
)
def test_fused_router_returns_dense_topk(kind, topk, path):
    """moe_router_fusion=True (TE's fused router) keeps the dense contract.

    Fails on main (the fused branch returns [T, E], d4); passes with #8562, which fuses when TE
    writes dense top-k indices and otherwise routes unfused.
    """
    router, bias, logits, x = _setup(
        64, topk, kind, 129, seed=3, backend="torch", moe_router_fusion=True
    )
    probs, experts = run_router(router, x, path)
    check_routing(probs, experts, logits, topk, kind, bias, None, f"fused {kind} k={topk} {path}")


@pytest.mark.parametrize("path", ["compiled", "batch_invariant"])
@pytest.mark.parametrize("topk", [1, 2])
def test_sinkhorn_matches_eval_router(topk, path):
    """Sinkhorn routing weights like the eval-mode TopKRouter: sigmoid / full softmax, unscaled.

    Fails on main (InferenceTopKRouter ignores the routing type; step 3 R06-F1); passes with
    #8562.
    """
    kind = "softmax_pre" if topk == 1 else "softmax"
    router = build_router(64, topk, kind, backend="torch", load_balancing="sinkhorn", scaling=None)
    logits = make_logits(129, 64, topk, kind, None, seed=4)
    probs, experts = run_router(router, logits.view(129, 1, 64), path)
    assert tuple(probs.shape) == (129, topk) and tuple(experts.shape) == (129, topk)
    order = experts.argsort(dim=-1)
    ref_probs, ref_experts = ou.fp64_reference(sinkhorn_reference, logits, topk)
    ou.assert_exact(experts.gather(-1, order), ref_experts, what="sinkhorn expert sets")
    trusted, _ = sinkhorn_reference(logits.float(), topk)
    ou.assert_close_to_reference(
        probs.gather(-1, order), ref_probs, trusted, factor=PROB_FACTOR, what="sinkhorn probs"
    )


def test_group_limited_routing_is_rejected():
    """Group-limited routing cannot be served by the inference router: construction raises."""
    with pytest.raises((AssertionError, ValueError)):
        build_router(64, 4, "sigmoid_bias", moe_router_num_groups=8, moe_router_group_topk=2)


@pytest.mark.parametrize("kind", ["softmax", "softmax_pre", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("topk", [1, 4])
def test_hash_layer_routing(kind, topk):
    """Hash layers: experts from the token-id table, weights from the score function."""
    if kind == "softmax" and topk == 1:
        kind = "softmax_pre"  # the config requires pre-softmax for top-1 softmax
    num_experts, vocab, seq, batch = 64, 97, 9, 2
    router = build_router(
        num_experts, topk, kind, backend="triton", moe_num_hash_layers=1, hash_moe_vocab_size=vocab
    )
    router.set_layer_number(1)
    assert router.is_hash_layer
    g = ou.make_generator(6)
    table = torch.stack([torch.randperm(num_experts, generator=g)[:topk] for _ in range(vocab)])
    router.tid2eid.copy_(table.to(router.tid2eid.dtype))
    ids = torch.randint(0, vocab, (batch, seq), generator=g).to(_DEVICE)
    logits = make_logits(seq * batch, num_experts, topk, kind, None, seed=8)
    x = logits.view(seq, batch, num_experts)
    with torch.no_grad(), InferenceMode.active():
        probs, experts = router(x, input_ids=ids)
    want = table.to(_DEVICE)[ids.t().reshape(-1)]
    assert tuple(probs.shape) == (seq * batch, topk) and tuple(experts.shape) == want.shape
    want_sorted = want.long().sort(dim=-1).values
    order = experts.long().argsort(dim=-1)
    ou.assert_exact(experts.long().gather(-1, order), want_sorted, what="hash expert sets")
    ref = ou.fp64_reference(weights_of, logits, want_sorted, kind, None)
    trusted = weights_of(logits.float(), want_sorted, kind, None)
    ou.assert_close_to_reference(
        probs.gather(-1, order), ref, trusted, factor=PROB_FACTOR, what=f"hash {kind} k={topk}"
    )


class _PathSpy:
    """Records which routing implementation ran: the sigmoid kernel or the compiled routing.

    Eager routing (batch-invariant mode) is the case where neither ran; its result is checked
    like every other. The eager function itself is not wrapped, because the compiled routing
    traces through it.
    """

    def __init__(self, monkeypatch):
        self.calls = []
        kernel = router_module.sigmoid_topk_routing
        compiled = InferenceTopKRouter.__dict__["_compiled_topk_routing"].__func__

        def kernel_spy(*args, **kwargs):
            self.calls.append("triton_kernel")
            return kernel(*args, **kwargs)

        def compiled_spy(*args, **kwargs):
            self.calls.append("compiled")
            return compiled(*args, **kwargs)

        monkeypatch.setattr(router_module, "sigmoid_topk_routing", kernel_spy)
        monkeypatch.setattr(
            InferenceTopKRouter, "_compiled_topk_routing", staticmethod(compiled_spy)
        )


_DISPATCH = [
    # (backend, kind, E, topk, batch-invariant, expected implementation)
    ("triton", "sigmoid_bias", 64, 6, False, "triton_kernel"),
    ("triton", "sigmoid_bias", 64, SIGMOID_KERNEL_MAX_TOPK, False, "triton_kernel"),
    ("triton", "sigmoid_bias", 64, SIGMOID_KERNEL_MAX_TOPK + 1, False, "compiled"),
    ("triton", "sigmoid_bias", SIGMOID_KERNEL_MAX_EXPERTS, 8, False, "triton_kernel"),
    ("triton", "sigmoid_bias", 2 * SIGMOID_KERNEL_MAX_EXPERTS, 8, False, "compiled"),
    ("triton", "sigmoid", 64, 6, False, "compiled"),
    ("triton", "softmax", 64, 6, False, "compiled"),
    ("torch", "sigmoid_bias", 64, 6, False, "compiled"),
    ("triton", "sigmoid_bias", 64, 6, True, "eager"),
    ("torch", "softmax", 64, 6, True, "eager"),
]


@pytest.mark.parametrize(
    "backend,kind,num_experts,topk,batch_invariant,expected",
    _DISPATCH,
    ids=[f"{b}-{k}-e{e}-k{t}-bi{int(bi)}" for b, k, e, t, bi, _ in _DISPATCH],
)
def test_dispatch_decisions(
    backend, kind, num_experts, topk, batch_invariant, expected, monkeypatch
):
    """Exactly one routing implementation runs, the documented one, and its result is right."""
    router, bias, logits, x = _setup(num_experts, topk, kind, 33, seed=2, backend=backend)
    spy = _PathSpy(monkeypatch)
    probs, experts = run_router(router, x, "batch_invariant" if batch_invariant else "compiled")
    want = [] if expected == "eager" else [expected]
    assert spy.calls == want, f"ran {spy.calls}, expected {want} ({expected})"
    check_routing(probs, experts, logits, topk, kind, bias, None, f"dispatch {expected}")


@pytest.mark.parametrize("path", ["compiled", "triton"])
def test_cuda_graph_replay(path):
    """Captured once, replayed with new tokens: the eager bits, and right against fp64."""
    backend = "torch" if path == "compiled" else "triton"
    tokens, num_experts, topk, kind = 128, 64, 6, "sigmoid_bias"
    router, bias, logits, x = _setup(num_experts, topk, kind, tokens, seed=1, backend=backend)
    x_buf = x.clone()
    with torch.no_grad(), InferenceMode.active():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):  # compile and warm up outside the capture
                router(x_buf)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            probs_buf, experts_buf = router(x_buf)
        for seed in (11, 12, 13):
            new = make_logits(tokens, num_experts, topk, kind, bias, seed)
            x_buf.copy_(new.view(tokens, 1, num_experts))
            graph.replay()
            eager_probs, eager_experts = router(new.view(tokens, 1, num_experts))
            ou.assert_exact(experts_buf, eager_experts, what=f"replay experts seed {seed}")
            ou.assert_exact(probs_buf, eager_probs, what=f"replay probs seed {seed}")
            check_routing(probs_buf, experts_buf, new, topk, kind, bias, None, f"replay {seed}")


@pytest.mark.sanitizer
@pytest.mark.parametrize("path", ["triton", "compiled"])
def test_sanitizer_case(path, monkeypatch):
    """Small case for compute-sanitizer: 160 experts (not a power of two, so the sigmoid kernel
    masks 96 of its 256 columns), top-6 with scaling and an expert bias, and 7 tokens, so the
    kernel's 4-row blocks end in a partial block; the compiled routing on the same shape. The
    spy checks which implementation ran."""
    num_experts, topk, tokens, kind, scaling = 160, 6, 7, "sigmoid_bias", 2.5
    backend = "torch" if path == "compiled" else "triton"
    router, bias, logits, x = _setup(
        num_experts, topk, kind, tokens, seed=3, backend=backend, scaling=scaling
    )
    spy = _PathSpy(monkeypatch)
    probs, experts = run_router(router, x, path)
    want = "triton_kernel" if path == "triton" else "compiled"
    assert spy.calls == [want], f"ran {spy.calls}, expected {want}"
    check_routing(probs, experts, logits, topk, kind, bias, scaling, f"sanitizer {path}")


def _gemm_reference(x, w):
    return x @ w.t()


@pytest.mark.parametrize("bi_backend", [None, "te_native", "triton", "deepgemm"])
def test_gating_logits_are_fp32(bi_backend):
    """Router logits are the fp32 product x @ W^T of the bf16 operands, in and out of BI mode.

    Fails on main for 'triton' and 'deepgemm' (finding O8a-N3): the batch-invariant TE GEMM
    computes the bf16 product in bf16 and casts it to fp32 afterwards, so logits carry bf16
    rounding (relative 2**-9) and expert selection can flip near ties. No open PR changes it.
    """
    if bi_backend == "deepgemm":
        assert bik.HAVE_DEEPGEMM_BF16, "the deepgemm backend needs DeepGEMM's bf16 bindings"
    # The trusted fp32 GEMM multiplies bf16 values, which TF32 represents exactly, and TF32
    # tensor cores accumulate in fp32: its result does not depend on TF32 being allowed (the
    # H100 CI image reports torch.backends.cuda.matmul.allow_tf32 = True).
    hidden, num_experts, tokens = 1024, 64, 129
    router = build_router(num_experts, 6, "sigmoid_bias", hidden=hidden)
    g = ou.make_generator(7, device=_DEVICE)
    w = (torch.randn(num_experts, hidden, device=_DEVICE, generator=g) / 32).to(torch.bfloat16)
    with torch.no_grad():
        router.weight.data = w
    x = torch.randn(tokens, 1, hidden, device=_DEVICE, generator=g).to(torch.bfloat16)
    with torch.no_grad(), InferenceMode.active():
        if bi_backend is None:
            logits = router.gating(x)
        else:
            with bik.set_batch_invariant_mode(True, backend=bi_backend):
                logits = router.gating(x)
    logits = logits.reshape(tokens, num_experts)
    assert logits.dtype == torch.float32, f"router logits dtype {logits.dtype}"
    x2 = x.view(tokens, hidden)
    ou.assert_close_to_reference(
        logits,
        ou.fp64_reference(_gemm_reference, x2, w),
        torch.mm(x2.float(), w.float().t()),
        dim=-1,
        what=f"router logits (batch-invariant backend {bi_backend})",
    )


# ---- Quantile balancing ------------------------------------------------------------------------
#
# Micro-batch QB selects the k largest z - beta (per-expert `qb_beta`) and weights them with the
# score function, exactly as without QB. Global-batch QB derives a bias for sigmoid scores
# (`expert_bias`) and selects the k largest sigmoid(z) + bias, weighted by sigmoid(z) normalized.
# In inference both biases are read, never updated (the QB accumulators stay as they were).

QB_SCOPES = ("micro_batch", "global_batch")


def build_qb_router(num_experts, topk, kind, scope, **kwargs):
    """An InferenceTopKRouter with quantile balancing at the given estimation scope."""
    return build_router(
        num_experts,
        topk,
        kind,
        load_balancing="quantile_balancing",
        moe_router_quantile_balancing_estimation_scope=scope,
        **kwargs,
    )


def make_qb_beta(num_experts, seed=11):
    """Micro-batch QB biases in [-2, 2], multiples of 1/64.

    The bf16 logits of `make_logits` are multiples of 2**-11 below 8 in magnitude, so z - beta
    is a multiple of 2**-11 below 2**4: exact in fp32, and the margin holds on every path.
    """
    g = ou.make_generator(seed)
    beta = torch.randint(-128, 129, (num_experts,), generator=g).double() / 64
    return beta.float().to(_DEVICE)


def make_qb_bias(num_experts, seed=12):
    """Global-batch QB biases in [-1, 1] (`qb_bin_bounds`' default range), multiples of 1/64."""
    g = ou.make_generator(seed)
    bias = torch.randint(-64, 65, (num_experts,), generator=g).double() / 64
    return bias.float().to(_DEVICE)


def qb_keys(beta):
    """Micro-batch QB selection keys z - beta, for `make_logits`' margin (fp64 rows)."""
    beta = beta.detach().double().cpu()
    return lambda rows: rows - beta


def qb_reference(logits, topk, kind, beta, scaling):
    """Micro-batch QB: the k largest z - beta, weighted by the score function."""
    experts = (logits - beta).topk(topk, dim=-1).indices.sort(dim=-1).values
    return weights_of(logits, experts, kind, scaling), experts


def eval_router_routing(router, x):
    """Route outside InferenceMode: the eval-mode TopKRouter the model was trained with, its
    [T, E] probs and map converted to dense [T, k] (experts ascending)."""
    topk = router.topk
    with torch.no_grad():
        probs, routing_map = router(x)
    experts = routing_map.nonzero()[:, 1].view(-1, topk)
    return probs.gather(-1, experts), experts


def _qb_id(case):
    return "-".join(f"{k}{v}" for k, v in case.items())


_QB_MICRO_CASES = ou.pairwise_cases(
    {
        "kind": ("softmax", "softmax_pre", "sigmoid"),
        "path": PATHS,
        "e": (8, 64, 256),
        "topk": (2, 6, 8),
        "t": (0, 1, 7, 128, 129),
        "scaling": (None, 2.5),
    }
)


@pytest.mark.parametrize("case", _QB_MICRO_CASES, ids=[_qb_id(c) for c in _QB_MICRO_CASES])
def test_quantile_balancing_micro_batch_selects_on_logits_minus_beta(case, monkeypatch):
    """Micro-batch QB on every path: the experts with the k largest z - beta, probs from the
    score function, and the QB state read only. The compiled routing (or eager routing in
    batch-invariant mode) runs, never the sigmoid kernel, which knows no QB bias. The eval-mode
    router outside InferenceMode selects the same experts."""
    num_experts, kind, tokens = case["e"], case["kind"], case["t"]
    topk = min(case["topk"], num_experts - 1)  # QB requires k < E
    backend = "torch" if case["path"] == "compiled" else "triton"
    router = build_qb_router(
        num_experts, topk, kind, "micro_batch", backend=backend, scaling=case["scaling"]
    )
    beta = make_qb_beta(num_experts)
    router.qb_beta.copy_(beta)
    state = {n: getattr(router, n).clone() for n in ("qb_beta", "qb_beta_accum", "qb_beta_count")}
    logits = make_logits(tokens, num_experts, topk, kind, None, seed=21, keys=qb_keys(beta))
    x = logits.view(tokens, 1, num_experts)
    spy = _PathSpy(monkeypatch)
    probs, experts = run_router(router, x, case["path"])
    want = [] if case["path"] == "batch_invariant" else ["compiled"]
    assert spy.calls == want, f"ran {spy.calls}, expected {want}"
    args = (topk, kind, beta, case["scaling"])
    check_against(probs, experts, logits, topk, qb_reference, args, f"QB micro {_qb_id(case)}")
    for name, before in state.items():
        ou.assert_exact(getattr(router, name), before, what=f"{name} after inference routing")
    if tokens:
        eval_probs, eval_experts = eval_router_routing(router, x)
        check_against(
            eval_probs, eval_experts, logits, topk, qb_reference, args, "eval-mode QB router"
        )


_QB_GLOBAL_CASES = ou.pairwise_cases(
    {
        "path": PATHS,
        "e": (8, 64, 256),
        "topk": (2, 6, 8, 32),
        "t": (0, 1, 7, 128, 129),
        "scaling": (None, 2.5),
    }
)


@pytest.mark.parametrize("case", _QB_GLOBAL_CASES, ids=[_qb_id(c) for c in _QB_GLOBAL_CASES])
def test_quantile_balancing_global_batch_routes_with_its_bias(case, monkeypatch):
    """Global-batch QB on every path: the k largest sigmoid(z) + bias, probs sigmoid(z)
    normalized, histogram and bias read only. The QB bias is an expert bias, so the Triton
    backend runs the sigmoid kernel wherever it supports the shape. The eval-mode router
    selects the same experts."""
    num_experts, tokens, scaling = case["e"], case["t"], case["scaling"]
    topk = min(case["topk"], num_experts - 1)
    backend = "torch" if case["path"] == "compiled" else "triton"
    router = build_qb_router(
        num_experts, topk, "sigmoid", "global_batch", backend=backend, scaling=scaling
    )
    assert router.qb_beta is None and router.expert_bias is not None
    bias = make_qb_bias(num_experts)
    router.expert_bias.copy_(bias)
    state = {n: getattr(router, n).clone() for n in ("expert_bias", "qb_histogram")}
    logits = make_logits(tokens, num_experts, topk, "sigmoid_bias", bias, seed=22)
    x = logits.view(tokens, 1, num_experts)
    spy = _PathSpy(monkeypatch)
    probs, experts = run_router(router, x, case["path"])
    if case["path"] == "batch_invariant":
        want = []
    elif case["path"] == "triton" and is_sigmoid_topk_supported(num_experts, topk):
        want = ["triton_kernel"]
    else:
        want = ["compiled"]
    assert spy.calls == want, f"ran {spy.calls}, expected {want}"
    what = f"QB global {_qb_id(case)}"
    check_routing(probs, experts, logits, topk, "sigmoid_bias", bias, scaling, what)
    for name, before in state.items():
        ou.assert_exact(getattr(router, name), before, what=f"{name} after inference routing")
    if tokens:
        eval_probs, eval_experts = eval_router_routing(router, x)
        check_routing(
            eval_probs, eval_experts, logits, topk, "sigmoid_bias", bias, scaling, "eval QB"
        )


@pytest.mark.parametrize("path", ["compiled", "triton"])
@pytest.mark.parametrize("scope", QB_SCOPES)
def test_quantile_balancing_cuda_graph_replay(scope, path):
    """Captured once, replayed with new tokens: the eager bits, right against fp64. A QB bias
    updated in place (as a weight refit does) is the one the next replay routes with."""
    tokens, num_experts, topk = 128, 64, 6
    backend = "torch" if path == "compiled" else "triton"
    if scope == "micro_batch":
        kind = "softmax"
        router = build_qb_router(num_experts, topk, kind, scope, backend=backend)
        bias_buffer, biases = router.qb_beta, [make_qb_beta(num_experts, s) for s in (11, 13)]
    else:
        kind = "sigmoid_bias"
        router = build_qb_router(num_experts, topk, "sigmoid", scope, backend=backend)
        bias_buffer, biases = router.expert_bias, [make_qb_bias(num_experts, s) for s in (12, 14)]

    def logits_for(bias, seed):
        if scope == "micro_batch":
            return make_logits(tokens, num_experts, topk, kind, None, seed, keys=qb_keys(bias))
        return make_logits(tokens, num_experts, topk, kind, bias, seed)

    def check(probs, experts, logits, bias, what):
        if scope == "micro_batch":
            args = (topk, kind, bias, None)
            check_against(probs, experts, logits, topk, qb_reference, args, what)
        else:
            check_routing(probs, experts, logits, topk, kind, bias, None, what)

    bias_buffer.copy_(biases[0])
    x_buf = logits_for(biases[0], 1).view(tokens, 1, num_experts).clone()
    with torch.no_grad(), InferenceMode.active():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):  # compile and warm up outside the capture
                router(x_buf)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            probs_buf, experts_buf = router(x_buf)
        for seed, bias in ((31, biases[0]), (32, biases[1]), (33, biases[0])):
            bias_buffer.copy_(bias)
            new = logits_for(bias, seed)
            x_buf.copy_(new.view(tokens, 1, num_experts))
            graph.replay()
            eager_probs, eager_experts = router(new.view(tokens, 1, num_experts))
            ou.assert_exact(experts_buf, eager_experts, what=f"replay experts seed {seed}")
            ou.assert_exact(probs_buf, eager_probs, what=f"replay probs seed {seed}")
            check(probs_buf, experts_buf, new, bias, f"QB {scope} replay {seed}")


def test_quantile_balancing_with_router_fusion_is_rejected():
    """Micro-batch QB selects outside the score function, which TE's fused router cannot take:
    the router must refuse, not route without the QB bias. (Under torch.compile the refusal can
    surface as a RuntimeError wrapping the assertion.)"""
    router = build_qb_router(64, 6, "sigmoid", "micro_batch", moe_router_fusion=True)
    router.qb_beta.copy_(make_qb_beta(64))
    logits = make_logits(7, 64, 6, "sigmoid", None, seed=23, keys=qb_keys(router.qb_beta))
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        run_router(router, logits.view(7, 1, 64), "compiled")


# ---- Router replay ----------------------------------------------------------------------------
#
# With `moe_enable_routing_replay`, each router owns a RouterReplay. RECORD routes as usual and
# records the routed experts: as a tensor, into a static [max_tokens, k] buffer (rows past the
# step's tokens untouched), or into bank `buffer_index` and layer `layer_index` of a
# double-buffered [2, max_tokens, layers, k] buffer (the async engine's), every other row,
# layer and bank untouched. REPLAY_FORWARD routes each token to the target experts, weighted as
# the score function weights those experts (an expert bias steers selection only). Without an
# action, routing is unchanged. The Triton sigmoid kernel does not record or replay, so the
# compiled routing runs.

SENTINEL = -7  # never an expert id
REPLAY_MAX_TOKENS = 160


@pytest.fixture
def replay_registry():
    """RouterReplay keeps every instance in a class-level list; start and end empty."""
    RouterReplay.clear_global_router_replay_instances()
    yield
    RouterReplay.clear_global_router_replay_action()
    RouterReplay.clear_global_indices()
    RouterReplay.clear_global_static_buffers()
    RouterReplay.clear_global_router_replay_instances()


def replay_reference(logits, target, kind, scaling):
    """REPLAY_FORWARD: the target experts, weighted by the score function."""
    experts = target.sort(dim=-1).values
    return weights_of(logits, experts, kind, scaling), experts


def make_record_buffer(record, topk):
    """(buffer, set_static_buffer kwargs, the view of the buffer this router records into)."""
    if record == "static":
        buffer = torch.full((REPLAY_MAX_TOKENS, 2, topk), SENTINEL, dtype=torch.int32)
        buffer = buffer.to(_DEVICE)

        def select(b):
            return b[:, 1, :]  # layer 1 of 2, as RouterReplay.set_global_static_buffers gives

        return buffer, dict(buffer=select(buffer)), select
    buffer = torch.full((2, REPLAY_MAX_TOKENS, 2, topk), SENTINEL, dtype=torch.int32)
    buffer = buffer.to(_DEVICE)
    bank = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
    layer = torch.tensor([1], dtype=torch.int64, device=_DEVICE)

    def select(b):
        return b[int(bank), :, int(layer), :]

    return buffer, dict(buffer=buffer, buffer_index=bank, layer_index=layer), select


def check_recorded_rows(buffer, select, experts, what):
    """The first T rows of the router's view hold the routed experts (as sets); every other
    element of the buffer keeps its sentinel."""
    tokens = experts.shape[0]
    rows = select(buffer)[:tokens].long().sort(dim=-1).values
    ou.assert_exact(rows, experts.sort(dim=-1).values, what=f"{what}: recorded rows")
    expected = torch.full_like(buffer, SENTINEL)
    select(expected)[:tokens] = select(buffer)[:tokens]
    ou.assert_exact(buffer, expected, what=f"{what}: elements outside the recorded rows")


_RECORD_CASES = ou.pairwise_cases(
    {
        "kind": ("softmax", "softmax_pre", "sigmoid", "sigmoid_bias"),
        "path": PATHS,
        "record": ("off", "tensor", "static", "banked"),
        "e": (8, 64, 256),
        "t": (0, 1, 7, 129),
    }
)


@pytest.mark.parametrize("case", _RECORD_CASES, ids=[_qb_id(c) for c in _RECORD_CASES])
def test_router_replay_record_routes_and_records(case, monkeypatch, replay_registry):
    """RECORD (and replay enabled without an action) routes as the router without replay does;
    RECORD records exactly the routed experts and writes nothing else. The compiled routing
    runs (eager routing in batch-invariant mode)."""
    num_experts, kind, tokens, topk = case["e"], case["kind"], case["t"], 6
    backend = "torch" if case["path"] == "compiled" else "triton"
    router, bias, logits, x = _setup(
        num_experts, topk, kind, tokens, seed=41, backend=backend, moe_enable_routing_replay=True
    )
    replay = router.router_replay
    assert replay is not None, "moe_enable_routing_replay builds no RouterReplay"
    buffer = select = None
    if case["record"] != "off":
        replay.set_router_replay_action(RouterReplayAction.RECORD)
    if case["record"] in ("static", "banked"):
        buffer, kwargs, select = make_record_buffer(case["record"], topk)
        replay.set_static_buffer(**kwargs)
    spy = _PathSpy(monkeypatch)
    probs, experts = run_router(router, x, case["path"])
    want = [] if case["path"] == "batch_invariant" else ["compiled"]
    assert spy.calls == want, f"ran {spy.calls}, expected {want}"
    what = f"record {_qb_id(case)}"
    check_routing(probs, experts, logits, topk, kind, bias, None, what)
    recorded = replay.get_recorded_indices()
    if case["record"] == "off":
        assert recorded is None, "recorded without the RECORD action"
        return
    ou.assert_exact(
        recorded.long().sort(dim=-1).values,
        experts.sort(dim=-1).values,
        what=f"{what}: recorded experts",
    )
    if buffer is not None:
        check_recorded_rows(buffer, select, experts, what)


def make_replay_targets(logits, topk, kind, bias, seed):
    """Target experts per token, never the token's own top-k for even tokens: its weakest
    selected expert is swapped for its lowest-scoring one. Odd tokens get k random experts.
    Each row is in random order."""
    keys = selection_keys(
        logits.double().cpu(), kind, None if bias is None else bias.double().cpu()
    )
    tokens, num_experts = keys.shape
    g = ou.make_generator(seed)
    rows = []
    for t in range(tokens):
        if t % 2 == 0:
            row = keys[t].topk(topk).indices.clone()
            row[-1] = keys[t].argmin()
        else:
            row = torch.randperm(num_experts, generator=g)[:topk]
        rows.append(row[torch.randperm(topk, generator=g)])
    targets = torch.stack(rows) if rows else torch.empty(0, topk, dtype=torch.int64)
    return targets.long().to(_DEVICE)


_REPLAY_CASES = ou.pairwise_cases(
    {
        "kind": ("softmax", "softmax_pre", "sigmoid", "sigmoid_bias"),
        "path": PATHS,
        "e": (8, 64, 256),
        "t": (0, 1, 7, 129),
        "scaling": (None, 2.5),
    }
)


@pytest.mark.parametrize("case", _REPLAY_CASES, ids=[_qb_id(c) for c in _REPLAY_CASES])
def test_router_replay_forward_routes_to_the_target_experts(case, monkeypatch, replay_registry):
    """REPLAY_FORWARD: every token goes to its target experts (whatever its scores), weighted
    as the score function weights those experts and scaled; the compiled routing runs."""
    num_experts, kind, tokens, topk = case["e"], case["kind"], case["t"], 6
    backend = "torch" if case["path"] == "compiled" else "triton"
    router, bias, logits, x = _setup(
        num_experts,
        topk,
        kind,
        tokens,
        seed=42,
        backend=backend,
        scaling=case["scaling"],
        moe_enable_routing_replay=True,
    )
    targets = make_replay_targets(logits, topk, kind, bias, seed=43)
    replay = router.router_replay
    replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    replay.set_target_indices(targets)
    spy = _PathSpy(monkeypatch)
    probs, experts = run_router(router, x, case["path"])
    want = [] if case["path"] == "batch_invariant" else ["compiled"]
    assert spy.calls == want, f"ran {spy.calls}, expected {want}"
    args = (targets, kind, case["scaling"])
    what = f"replay {_qb_id(case)}"
    check_against(probs, experts, logits, topk, replay_reference, args, what)


@pytest.mark.parametrize("record", ["static", "banked"])
@pytest.mark.parametrize("path", ["compiled", "triton"])
def test_router_replay_records_under_cuda_graph_replay(path, record, replay_registry):
    """RECORD captured into a CUDA graph, as the engine's warmup captures it: every replay
    writes the replayed experts into the static buffer and nothing else. With the
    double-buffered layout, into the bank the GPU selector names at replay time."""
    tokens, num_experts, topk, kind = 128, 64, 6, "sigmoid_bias"
    backend = "torch" if path == "compiled" else "triton"
    router, bias, logits, x = _setup(
        num_experts, topk, kind, tokens, seed=44, backend=backend, moe_enable_routing_replay=True
    )
    replay = router.router_replay
    replay.set_router_replay_action(RouterReplayAction.RECORD)
    buffer, kwargs, select = make_record_buffer(record, topk)
    replay.set_static_buffer(**kwargs)
    x_buf = x.clone()
    with torch.no_grad(), InferenceMode.active():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):  # compile and warm up outside the capture
                router(x_buf)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            probs_buf, experts_buf = router(x_buf)
        for step, seed in enumerate((45, 46, 47)):
            if record == "banked":
                kwargs["buffer_index"].fill_(step % 2)
            buffer.fill_(SENTINEL)
            new = make_logits(tokens, num_experts, topk, kind, bias, seed)
            x_buf.copy_(new.view(tokens, 1, num_experts))
            graph.replay()
            torch.cuda.synchronize()
            what = f"graph record {record} step {step}"
            check_routing(probs_buf, experts_buf, new, topk, kind, bias, None, what)
            check_recorded_rows(buffer, select, experts_buf, what)
