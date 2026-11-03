# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the KDA core: ``KimiDeltaAttention`` forward at TP=1 around FLA's ``chunk_kda``.

Covers the ``chunk_kda`` call site (megatron/core/ssm/gated_delta_net/kda.py, ``_forward_compute``)
together with what feeds and follows it: projections, causal conv, q/k L2 norm, decay gate, beta,
gated output norm and ``out_proj`` (step 3 review R01, delta record D1 and its oracle plan).

Contract (written from the Kimi Delta Attention semantics, the call site and FLA 0.5.1's documented
``chunk_kda`` API, before reading the kernel):

* Input ``x`` [T, B, D] sequence-first, or a THD pack [T, 1, D] with ``cu_seqlens`` in which each
  sequence is computed on its own, from zero history. TP=1, no CP, no inference context. Output
  [T, B, D] in the model dtype; the returned bias is None.
* Projections, no bias. Direct gates: ``in_proj(x) = [q | k | v | f | gate]`` (q, k, f: H*K
  channels; v, gate: H*V). Two-stage gates (``kda_two_stage_gates``): ``in_proj(x) = [q | k | v]``,
  ``f = f_b(f_a(x))``, ``gate = g_b(g_a(x))``. ``b = beta_proj(x)`` [H].
* Depthwise causal conv of width W over the q|k|v channels with zero history at every sequence
  start, then SiLU: ``c[t] = silu(sum_j w[:, j] * x[t - W + 1 + j])``.
* q and k L2-normalised per head over K, once: ``x / sqrt(|x|^2 + 1e-6)`` (outside the kernel with
  direct gates, inside it with two-stage gates).
* Decay in log space, per channel: ``g = -exp(A_log[h]) * softplus(f + dt_bias[h])``; when
  ``kda_lower_bound`` is set, ``g = lower_bound * sigmoid(exp(A_log[h]) * (f + dt_bias[h]))``
  (FLA keys the form on ``lower_bound``; ``kda_safe_gate`` only selects a kernel path and requires
  ``lower_bound`` in [-5, 0)). ``beta = sigmoid(b)`` in fp32. A_log [H], dt_bias [H*K] are fp32.
* Recurrence per head from ``S = 0`` (K x V): ``S <- diag(exp(g_t)) S``,
  ``u_t = beta_t (v_t - S^T k_t)``, ``S <- S + k_t u_t^T``, ``o_t = S^T q_t / sqrt(K)``.
* Output norm per head over V. Direct gates: ``out_norm(o) * sigmoid(gate)``, ``out_norm`` as
  configured (RMSNorm, or LayerNorm with its bias; zero-centered gamma uses ``w + 1``). Two-stage
  gates: Kimi Linear's gated RMSNorm ``o / rms(o) * (w + [1 if zero-centered]) * sigmoid(gate)``,
  no bias, whatever ``normalization`` says (#8882's contract; main's forward already ignores the
  LayerNorm bias). The gain ``w + 1`` is exact (formed in fp32).
* ``out = out_proj(norm_out)``.

Tolerance class: floating point. Three outputs per case, each with
``ou.assert_close_to_reference(actual, fp64_reference, trusted)``:
1. the kernel's output ``o``, captured at the call site, against the recurrence above in fp64 on the
   exact tensors the module handed to ``chunk_kda`` (``CHUNKED_FACTOR``, see there);
2. the gated output norm on the exact tensors the module handed to it (default factor 2);
3. the module's output against the whole contract in fp64, fed the module's own weights
   (``CHUNKED_FACTOR``: it inherits the kernel's error).
The trusted implementation is the same plain-PyTorch reference run in fp32 on the same device
(TF32 off), rounding to bf16 exactly where the module hands bf16 tensors on (projection outputs,
conv output, normalised q/k, the kernel output, the norm output, ``out_proj``). The reference is a
token-by-token loop: it shares neither FLA's chunked WY algorithm nor its blocking (and is not FLA's
``naive.py``). Exact checks: the kernel's inputs are read-only, the norm receives the kernel's
output unchanged, and the call site passes ``use_gate_in_kernel=True``, the L2-norm flag of its gate
mode, ``lower_bound``, the default scale, no initial or final state and the pack's ``cu_seqlens``.

Variants and branches covered: direct and two-stage gates (``use_qk_l2norm_in_kernel`` False/True);
gate forms softplus (the config defaults), ``lower_bound=-5`` with ``safe_gate`` and ``lower_bound=-1``
without; ``deterministic_mode`` off (FLA causal conv, FLA l2norm) and on (``F.conv1d`` fallback,
``torch_l2norm``); fixed-length SBHD and THD packs (``cu_seqlens``); RMSNorm, LayerNorm and
zero-centered gamma output norms. Sizes from the kernel's structure: FLA's KDA chunk is 64 tokens
(``chunk_kda_fwd(chunk_size=64)``) and the conv width W=4: T in {1, 2, W-1, W, 63, 64, 65, 127,
128, 129}; packs with sequence boundaries at chunk offsets 0, 1 and 63 and single-token sequences;
head dim K in {32, 64, 128, 96 (not a power of two: FLA pads to 128 and masks)}; heads per rank
H in {1, 4}; factors combined pairwise. Production (``test_kda_production_shapes``): a Kimi Linear
layer at TP 1 (hidden 2304, H = 32, K = V = 128) with its two-stage gates on a 4096-token pack of
six sequences (2048, 1024, 769, 191, 63, 1: boundaries at chunk offsets 0, 1 and 63), and with
direct gates and the safe gate on 4 x 1024 tokens; same three checks and factors.

Memory: the call site's kernel inputs (q, k, v, g, beta, A_log, dt_bias) are handed to the kernel
in NaN guard buffers, so a read past any of them turns outputs NaN and a write fails the guard
check; the kernel must leave them unchanged. Three small K=96 cases (T=1 two-stage, T=65 direct,
a pack with single-token sequences) are marked ``sanitizer``.

Main vs open PRs:
* ``test_kda_matches_fp64_reference[two_stage-zero_centered]`` fails on main, at the gated norm:
  two-stage KDA forms the zero-centered gain ``w + 1`` in bf16, so a learned ``w = 0.0038`` becomes
  a gain of exactly 1.0 (R01-F1, a systematic 0.38% error on every output, about 2.5x the trusted
  rms error). #8882 (fix/kda-gated-norm-contract) forms the gain in fp32 and makes it pass.
  ``[direct-zero_centered]`` expects Transformer Engine's RMSNorm to form ``w + 1`` in fp32 as
  well (not verified by reading TE). Every other case is expected to pass on main.
* #8882 also builds the two-stage ``out_norm`` as a weight-only RMSNorm whatever ``normalization``
  says; ``[two_stage-layernorm_config]`` checks that contract and passes on both (main builds a
  LayerNorm but its forward never reads the bias).
* #8883 (fix/kda-fla-deterministic) pins FLA's autotuned configs in deterministic mode; it changes
  which tiles run, not the math, so the deterministic cases pass with and without it.
* #8861 (fix/fla-core-g-overrun) moves FLA to 0.6.0; the contract is the documented API, which
  0.6.0 keeps, so the test runs unchanged on it.
* #8710 (n13) adds a single-token ``fused_recurrent_kda`` path for context-free MTP drafts; a plain
  forward never takes it, so it is not covered here.

Hardware: one H100 (FLA, Transformer Engine); the production cases also on one GB200, where FLA
autotunes over a different tile list. Not covered: TP>1 / SP / CP (collectives, owned by
other tests), the inference-optimized spec's projections (``InferenceColumnParallelLinear`` needs
the symmetric-memory setup of the inference engine), FP8/MXFP8 projections, the backward pass.
"""

import contextlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pytest
import torch
import torch.nn.functional as F

from tests.unit_tests.inference.oracles import oracle_utils as ou

CHUNK = 64
"""FLA's KDA chunk length (fla/ops/kda/chunk_fwd.py: ``chunk_kda_fwd(chunk_size=64)``)."""

CONV_WIDTH = 4
"""Conv width W (``linear_conv_kernel_dim``) of the KDA configs in tree."""

L2NORM_EPS = 1e-6
"""Epsilon of KDA's q/k L2 norm (``torch_l2norm`` / FLA ``l2norm_fwd`` default)."""

CHUNKED_FACTOR = 4.0
"""Bound factor for the kernel output and the module output, which inherits it.

FLA's chunked algorithm keeps more of its intermediates in the input dtype (bf16) than the
contract does: the intra-chunk matrices ``Aqk``/``Akk`` and the WY factors ``w``, ``u``, ``kg``
(fla/ops/kda/chunk_intra.py, wy_fast.py: ``dtype=k.dtype``), the chunk-entry states ``h`` and
``v_new`` (fla/ops/common/chunk_delta_h.py: ``h = k.new_empty(...)``, ``v_new = empty_like(u)``).
Each of these roundings perturbs ``o`` about as much as its final bf16 rounding, which is the only
rounding the trusted fp32 recurrence makes there, so a correct kernel measures 2-3x the trusted
error. Wiring errors (gate form, scale, beta, L2-norm placement, conv history, sequence
boundaries) are O(1) and the gate or state kept in bf16 instead of fp32 is ~10x; the gated norm,
where R01-F1's gain error lives, keeps the default factor 2 on its own inputs."""

HIDDEN = 128
BATCH = 2

GATE_FORMS = {
    # name: (kda_safe_gate, kda_lower_bound)
    "softplus": (False, None),
    "safe_lb-5": (True, -5.0),
    "lb-1": (False, -1.0),
}

SEQ_LENS = sorted(
    {1, 2}
    | set(ou.boundary_sizes(CONV_WIDTH, multiples=(), degenerate=False))
    | set(ou.boundary_sizes(CHUNK, multiples=(2,), degenerate=False))
)
"""1, 2, W-1, W, W+1, 63, 64, 65, 127, 128, 129."""

PACKS = {
    # Sequence boundaries at chunk offsets 0, 1, 63 and single-token sequences.
    "pack-0-1-63": (64, 1, 62, 1, 65),
    "pack-1-2-63-0": (1, 1, 61, 129),
    # Sequences shorter than and equal to the conv width inside a pack.
    "pack-short": (3, 4, 127),
}

KIMI_LINEAR = dict(hidden=2304, heads=32, head_dim=128)
"""A Kimi Linear KDA layer at TP 1: Kimi-Linear-48B-A3B's published config (hidden 2304, 32 KDA heads
with K = V = 128, conv width 4, low-rank gates; no such config is in this tree, so taken from the
model card as recalled, offline)."""

PRODUCTION_PACKS = {
    # A 4096-token prefill pack: sequence boundaries at chunk offsets 0, 0, 1, 0, 63, 0.
    "pack-prod-4096": (2048, 1024, 769, 191, 63, 1)
}


# ---------------------------------------------------------------------------------------------
# Reference (plain PyTorch; fp64 via ou.fp64_reference, fp32 + interface rounding as trusted)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KDASpec:
    """The contract's parameters for one case."""

    heads: int
    head_dim: int
    two_stage: bool
    lower_bound: Optional[float]
    normalization: str = "RMSNorm"
    zero_centered: bool = False
    eps: float = 1e-5
    conv_rounds_preact: bool = False
    """The ``F.conv1d`` fallback (deterministic mode) rounds the conv output before SiLU."""


def _rounder(interface: Optional[torch.dtype]):
    if interface is None:
        return lambda t: t
    return lambda t: t.to(interface).to(t.dtype)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """``x / sqrt(|x|^2 + eps)`` over the last dim."""
    return x / torch.sqrt(x.square().sum(-1, keepdim=True) + L2NORM_EPS)


def kda_decay(
    f: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: Optional[float]
) -> torch.Tensor:
    """Log-space decay [..., H, K] from the raw gate ``f`` [..., H, K]."""
    heads, head_dim = f.shape[-2:]
    a = A_log.reshape(heads, 1).exp()
    z = f + dt_bias.reshape(heads, head_dim)
    if lower_bound is None:
        return -a * F.softplus(z)
    return lower_bound * torch.sigmoid(a * z)


def delta_rule(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Token-by-token gated delta rule from a zero state.

    q, k, g: [N, T, H, K]; v: [N, T, H, V]; beta: [N, T, H]. Returns o [N, T, H, V].
    """
    n, steps, heads, head_dim = q.shape
    state = q.new_zeros(n, heads, head_dim, v.shape[-1])
    scale = head_dim**-0.5
    out = []
    for t in range(steps):
        state = state * g[:, t].exp().unsqueeze(-1)
        predicted = torch.einsum("nhk,nhkv->nhv", k[:, t], state)
        update = beta[:, t].unsqueeze(-1) * (v[:, t] - predicted)
        state = state + k[:, t].unsqueeze(-1) * update.unsqueeze(-2)
        out.append(torch.einsum("nhk,nhkv->nhv", q[:, t] * scale, state))
    return torch.stack(out, dim=1)


def causal_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv with zero history: x [N, T, C], weight [C, W] -> [N, T, C]."""
    width = weight.shape[-1]
    steps = x.shape[1]
    padded = F.pad(x, (0, 0, width - 1, 0))
    return sum(weight[:, j] * padded[:, j : j + steps] for j in range(width))


def _sequences(total: int, cu_seqlens: Optional[Sequence[int]]) -> List[slice]:
    if cu_seqlens is None:
        return [slice(0, total)]
    return [slice(a, b) for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:])]


def kda_kernel_reference(
    q, k, v, g, beta, A_log, dt_bias, *, l2norm_in_kernel, lower_bound, cu_seqlens, interface=None
):
    """``chunk_kda(..., use_gate_in_kernel=True)`` per the contract: [B, T, H, V]."""
    r = _rounder(interface)
    if l2norm_in_kernel:
        q, k = r(l2_normalize(q)), r(l2_normalize(k))
    decay = kda_decay(g, A_log, dt_bias, lower_bound)
    pieces = [
        delta_rule(q[:, s], k[:, s], v[:, s], decay[:, s], beta[:, s])
        for s in _sequences(q.shape[1], cu_seqlens)
    ]
    return r(torch.cat(pieces, dim=1))


def _out_norm(o, gate, w, bias, spec: KDASpec, r):
    gain = w + 1 if spec.zero_centered else w
    if spec.two_stage:
        # Kimi Linear's gated RMSNorm: one rounding, after norm, gain and gate.
        rms = torch.sqrt(o.square().mean(-1, keepdim=True) + spec.eps)
        return r(o / rms * gain * torch.sigmoid(gate))
    if spec.normalization == "LayerNorm":
        mean = o.mean(-1, keepdim=True)
        var = (o - mean).square().mean(-1, keepdim=True)
        normed = (o - mean) / torch.sqrt(var + spec.eps) * gain + bias
    else:
        normed = o / torch.sqrt(o.square().mean(-1, keepdim=True) + spec.eps) * gain
    return r(r(normed) * torch.sigmoid(gate))


def gated_norm_reference(o, gate, weight, bias, spec: KDASpec, interface=None):
    """``_apply_gated_norm`` per the contract: o, gate [..., V] -> [..., V]."""
    return _out_norm(o, gate, weight, bias, spec, _rounder(interface))


def kda_module_reference(
    hidden: torch.Tensor,
    params: Dict[str, torch.Tensor],
    spec: KDASpec,
    cu_seqlens: Optional[Sequence[int]] = None,
    interface: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """The whole contract: hidden [T, B, D] -> out [T, B, D].

    ``interface=None`` computes exactly (in the inputs' precision); a dtype rounds every tensor the
    module hands on in that dtype (the trusted implementation).
    """
    r = _rounder(interface)
    heads, head_dim = spec.heads, spec.head_dim
    qk_dim = v_dim = heads * head_dim
    x = hidden.transpose(0, 1)  # [B, T, D]
    batch, steps, _ = x.shape

    projected = r(x @ params["in_proj"].T)
    beta = torch.sigmoid(r(x @ params["beta_proj"].T))
    if spec.two_stage:
        qkv = projected
        f = r(r(x @ params["f_a_proj"].T) @ params["f_b_proj"].T)
        gate = r(r(x @ params["g_a_proj"].T) @ params["g_b_proj"].T)
    else:
        qkv, f, gate = torch.split(projected, [2 * qk_dim + v_dim, qk_dim, v_dim], dim=-1)

    conv_weight = params["conv1d"].reshape(qkv.shape[-1], -1)
    pieces = []
    for s in _sequences(steps, cu_seqlens):
        pre = causal_conv(qkv[:, s], conv_weight)
        if spec.conv_rounds_preact:
            pre = r(pre)
        pieces.append(r(F.silu(pre)))
    conv = torch.cat(pieces, dim=1)

    q, k, v = torch.split(conv, [qk_dim, qk_dim, v_dim], dim=-1)
    q = r(l2_normalize(q.reshape(batch, steps, heads, head_dim)))
    k = r(l2_normalize(k.reshape(batch, steps, heads, head_dim)))
    v = v.reshape(batch, steps, heads, head_dim)
    decay = kda_decay(
        f.reshape(batch, steps, heads, head_dim),
        params["A_log"],
        params["dt_bias"],
        spec.lower_bound,
    )
    o = torch.cat(
        [
            delta_rule(q[:, s], k[:, s], v[:, s], decay[:, s], beta[:, s])
            for s in _sequences(steps, cu_seqlens)
        ],
        dim=1,
    )
    o = r(o)
    normed = _out_norm(
        o,
        gate.reshape(batch, steps, heads, head_dim),
        params["out_norm"],
        params.get("out_norm_bias"),
        spec,
        r,
    )
    out = r(normed.reshape(batch, steps, v_dim) @ params["out_proj"].T)
    return out.transpose(0, 1)


@contextlib.contextmanager
def _ieee_fp32_matmul():
    """The trusted fp32 reference must not run its GEMMs in TF32."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def trusted(fn, *args, interface=torch.bfloat16, **kwargs):
    """Run a reference in fp32 (TF32 off), rounding to ``interface`` at the module's interfaces."""

    def fp32(obj):
        if isinstance(obj, torch.Tensor) and obj.is_floating_point():
            return obj.float()
        if isinstance(obj, dict):
            return {key: fp32(value) for key, value in obj.items()}
        return obj

    with _ieee_fp32_matmul():
        out = fn(
            *[fp32(a) for a in args], interface=interface, **{k: fp32(v) for k, v in kwargs.items()}
        )
    return out.to(interface)


# ---------------------------------------------------------------------------------------------
# The module under test
# ---------------------------------------------------------------------------------------------


class _GuardedKernelCall:
    """Wraps the bound ``gated_delta_rule``: guarded inputs, read-only check, captured call."""

    INPUTS = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")

    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = []

    def __call__(self, **kwargs):
        guarded = {name: ou.guarded_like(kwargs[name].contiguous()) for name in self.INPUTS}
        before = {name: buf.tensor.clone() for name, buf in guarded.items()}
        for name, buf in guarded.items():
            kwargs[name] = buf.tensor
        result = self.kernel(**kwargs)
        ou.check_guards(**guarded)
        for name, buf in guarded.items():
            ou.assert_exact(buf.tensor, before[name], what=f"chunk_kda input {name} (read-only)")
        self.calls.append(({**kwargs, **before}, result[0].detach().clone()))
        return result


class _CapturedCall:
    """Wraps a bound method: records its tensor arguments and result."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, *args):
        result = self.fn(*args)
        self.calls.append(([a.detach().clone() for a in args], result.detach().clone()))
        return result


@dataclass(frozen=True)
class Case:
    """One forward: layout (``steps`` for SBHD or ``pack`` for THD) and module configuration."""

    head_dim: int
    heads: int
    gates: str
    gate_form: str
    steps: Optional[int] = None
    pack: Optional[str] = None
    deterministic: bool = False
    normalization: str = "RMSNorm"
    zero_centered: bool = False
    gain: Optional[float] = None
    """Fill ``out_norm.weight`` with this value (else random around 1, or 0 when zero-centered)."""
    hidden: int = HIDDEN
    batch: int = BATCH
    """Sequences of an SBHD input (``steps`` each)."""


@pytest.fixture(scope="module")
def model_parallel():
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


def _build_module(case: Case):
    from megatron.core import parallel_state
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net.kda import HAVE_FLA_KDA, KimiDeltaAttention
    from megatron.core.ssm.kda_layer_config import KDALayerConfig
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not HAVE_FLA_KDA:
        pytest.fail("FLA with KDA support is required for the KDA oracle", pytrace=False)
    model_parallel_cuda_manual_seed(123)
    safe_gate, lower_bound = GATE_FORMS[case.gate_form]
    config = KDALayerConfig(
        hidden_size=case.hidden,
        num_layers=1,
        num_attention_heads=case.heads,
        num_query_groups=case.heads,
        kv_channels=case.head_dim,
        linear_conv_kernel_dim=CONV_WIDTH,
        linear_key_head_dim=case.head_dim,
        linear_value_head_dim=case.head_dim,
        linear_num_key_heads=case.heads,
        linear_num_value_heads=case.heads,
        kda_two_stage_gates=case.gates == "two_stage",
        kda_safe_gate=safe_gate,
        kda_lower_bound=lower_bound,
        normalization=case.normalization,
        layernorm_zero_centered_gamma=case.zero_centered,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        deterministic_mode=case.deterministic,
        experimental_attention_variant="kda",
        is_hybrid_model=True,
        transformer_impl="transformer_engine",
    )
    spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
    module = KimiDeltaAttention(
        config,
        submodules=spec.submodules,
        layer_number=1,
        pg_collection=ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        ),
    ).cuda()
    return module.eval(), config


def _randomize(module, case: Case, generator: torch.Generator) -> Dict[str, torch.Tensor]:
    """Give every parameter seeded, well-conditioned values; return them by contract name."""

    def normal(param, std):
        return torch.randn(param.shape, generator=generator, device=param.device) * std

    def uniform(param, lo, hi):
        return torch.rand(param.shape, generator=generator, device=param.device) * (hi - lo) + lo

    head_dim = case.head_dim
    values = {}
    with torch.no_grad():
        hidden = case.hidden
        linears = {"in_proj": hidden, "beta_proj": hidden, "out_proj": case.heads * head_dim}
        if case.gates == "two_stage":
            linears.update(f_a_proj=hidden, f_b_proj=head_dim, g_a_proj=hidden, g_b_proj=head_dim)
        for name, fan_in in linears.items():
            weight = getattr(module, name).weight
            weight.copy_(normal(weight, fan_in**-0.5))
            values[name] = weight.detach()
        conv = module.conv1d.weight
        conv.copy_(normal(conv, 0.5))
        values["conv1d"] = conv.detach()
        module.A_log.copy_(uniform(module.A_log, 1.0, 16.0).log())
        dt = uniform(module.dt_bias, math.log(1e-3), math.log(1e-1)).exp()
        module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        values["A_log"], values["dt_bias"] = module.A_log.detach(), module.dt_bias.detach()
        norm_weight = module.out_norm.weight
        if case.gain is not None:
            norm_weight.fill_(case.gain)
        else:
            lo, hi = (-0.5, 0.5) if case.zero_centered else (0.5, 1.5)
            norm_weight.copy_(uniform(norm_weight, lo, hi))
        values["out_norm"] = norm_weight.detach()
        bias = getattr(module.out_norm, "bias", None)
        if bias is not None:
            bias.copy_(normal(bias, 0.5))
            if case.normalization == "LayerNorm" and case.gates != "two_stage":
                values["out_norm_bias"] = bias.detach()
    return values


def _run(case: Case):
    from megatron.core.packed_seq_params import PackedSeqParams

    module, config = _build_module(case)
    g = ou.make_generator(device="cuda")
    params = _randomize(module, case, g)
    if case.pack is not None:
        lengths = {**PACKS, **PRODUCTION_PACKS}[case.pack]
        cu = [0]
        for n in lengths:
            cu.append(cu[-1] + n)
        cu_tensor = torch.tensor(cu, dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_tensor,
            cu_seqlens_kv=cu_tensor,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        )
        shape = (cu[-1], 1, case.hidden)
        cu_seqlens = cu
    else:
        packed, cu_seqlens = None, None
        shape = (case.steps, case.batch, case.hidden)
    hidden = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
    hidden_before = hidden.clone()

    call = _GuardedKernelCall(module.gated_delta_rule)
    module.gated_delta_rule = call
    norm_call = _CapturedCall(module._apply_gated_norm)
    module._apply_gated_norm = norm_call
    with torch.no_grad():
        out, bias = module(hidden, None, packed_seq_params=packed)
    ou.assert_exact(hidden, hidden_before, what="hidden_states (read-only)")
    assert bias is None
    assert out.shape == shape and out.dtype == torch.bfloat16
    assert len(call.calls) == 1, f"expected one chunk_kda call, got {len(call.calls)}"
    assert len(norm_call.calls) == 1, f"expected one gated norm, got {len(norm_call.calls)}"
    kwargs, kernel_out = call.calls[0]
    (norm_in, norm_gate), norm_out = norm_call.calls[0]

    spec = KDASpec(
        heads=case.heads,
        head_dim=case.head_dim,
        two_stage=case.gates == "two_stage",
        lower_bound=GATE_FORMS[case.gate_form][1],
        normalization=case.normalization,
        zero_centered=case.zero_centered,
        eps=config.layernorm_epsilon,
        conv_rounds_preact=case.deterministic,
    )
    # The call site's contract with the kernel.
    assert kwargs["use_gate_in_kernel"] is True
    assert kwargs["use_qk_l2norm_in_kernel"] is spec.two_stage
    assert kwargs["lower_bound"] == spec.lower_bound
    assert kwargs.get("scale") is None, "KDA scales q by K**-0.5 (the kernel default)"
    assert kwargs["initial_state"] is None and kwargs["output_final_state"] is False
    if cu_seqlens is None:
        assert kwargs["cu_seqlens"] is None
    else:
        ou.assert_exact(kwargs["cu_seqlens"].cpu(), torch.tensor(cu_seqlens), what="cu_seqlens")

    kernel_args = [kwargs[name] for name in _GuardedKernelCall.INPUTS]
    kernel_kw = dict(
        l2norm_in_kernel=spec.two_stage, lower_bound=spec.lower_bound, cu_seqlens=cu_seqlens
    )
    ou.assert_close_to_reference(
        kernel_out,
        ou.fp64_reference(kda_kernel_reference, *kernel_args, **kernel_kw),
        trusted(kda_kernel_reference, *kernel_args, **kernel_kw),
        factor=CHUNKED_FACTOR,
        what=f"chunk_kda output o ({case})",
    )
    # The norm sees the kernel's output: the call site must hand it on unchanged.
    ou.assert_exact(norm_in, kernel_out, what="gated norm input (the kernel's output)")
    norm_args = (
        norm_in.reshape(-1, case.head_dim),
        norm_gate.reshape(-1, case.head_dim),
        params["out_norm"],
        params.get("out_norm_bias"),
        spec,
    )
    ou.assert_close_to_reference(
        norm_out,
        ou.fp64_reference(gated_norm_reference, *norm_args),
        trusted(gated_norm_reference, *norm_args),
        what=f"gated output norm ({case})",
    )
    ou.assert_close_to_reference(
        out,
        ou.fp64_reference(kda_module_reference, hidden, params, spec, cu_seqlens),
        trusted(kda_module_reference, hidden, params, spec, cu_seqlens),
        factor=CHUNKED_FACTOR,
        what=f"KimiDeltaAttention output ({case})",
    )


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------

HEAD_DIMS = (32, 64, 128, 96)
HEADS = (1, 4)


def _sbhd_cases() -> List[Case]:
    return [
        Case(**c)
        for c in ou.pairwise_cases(
            {
                "steps": SEQ_LENS,
                "head_dim": HEAD_DIMS,
                "heads": HEADS,
                "gates": ("direct", "two_stage"),
                "gate_form": tuple(GATE_FORMS),
                "deterministic": (False, True),
            }
        )
    ]


def _thd_cases() -> List[Case]:
    # Packed KDA requires FLA's causal conv: deterministic mode rejects THD input.
    return [
        Case(**c)
        for c in ou.pairwise_cases(
            {
                "pack": tuple(PACKS),
                "head_dim": HEAD_DIMS,
                "heads": HEADS,
                "gates": ("direct", "two_stage"),
                "gate_form": tuple(GATE_FORMS),
            }
        )
    ]


def _case_id(case: Case) -> str:
    layout = f"T{case.steps}" if case.pack is None else case.pack
    det = "-det" if case.deterministic else ""
    return f"{layout}-K{case.head_dim}-H{case.heads}-{case.gates}-{case.gate_form}{det}"


SPECIAL_CASES = [
    # Zero-centered gain w = 0.0038 lies in (-2**-9, 2**-8): bf16(w + 1) == 1.0, a 0.38% error.
    # Two-stage fails on main (R01-F1); #8882 forms the gain in fp32.
    pytest.param(
        Case(64, 4, "two_stage", "safe_lb-5", steps=65, zero_centered=True, gain=0.0038),
        id="two_stage-zero_centered",
    ),
    pytest.param(
        Case(64, 4, "direct", "safe_lb-5", steps=65, zero_centered=True, gain=0.0038),
        id="direct-zero_centered",
    ),
    pytest.param(
        Case(64, 4, "direct", "softplus", steps=65, normalization="LayerNorm"),
        id="direct-layernorm",
    ),
    # Two-stage gates run the gated RMSNorm whatever `normalization` says; any LayerNorm bias is
    # not part of the math (#8882 builds out_norm without one; main never reads it).
    pytest.param(
        Case(64, 4, "two_stage", "softplus", steps=65, normalization="LayerNorm"),
        id="two_stage-layernorm_config",
    ),
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "case", [pytest.param(c, id=_case_id(c)) for c in _sbhd_cases() + _thd_cases()] + SPECIAL_CASES
)
def test_kda_matches_fp64_reference(model_parallel, case):
    """KDA forward at TP=1 against the fp64 contract: kernel output, gated norm, module output.

    ``[two_stage-zero_centered]`` fails on main (R01-F1) and passes with #8882; see the module
    docstring.
    """
    _run(case)


PRODUCTION_CASES = [
    # Kimi Linear's own layer: low-rank (two-stage) gates, softplus decay, one 4096-token pack.
    pytest.param(
        Case(gates="two_stage", gate_form="softplus", pack="pack-prod-4096", **KIMI_LINEAR),
        id="kimi_linear-pack4096-two_stage-softplus",
    ),
    # The same head layout with direct gates and the safe-gate kernel path, 4 x 1024 tokens (SBHD).
    pytest.param(
        Case(gates="direct", gate_form="safe_lb-5", steps=1024, batch=4, **KIMI_LINEAR),
        id="kimi_linear-T1024x4-direct-safe_lb-5",
    ),
]


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("case", PRODUCTION_CASES)
def test_kda_production_shapes(model_parallel, case):
    """A Kimi Linear KDA layer (32 heads, K = V = 128, hidden 2304) on a 4096-token prefill pack and
    an SBHD batch, with the checks of ``test_kda_matches_fp64_reference``. Also on GB200: FLA picks
    its forward tiles from per-architecture autotune lists (``IS_NVIDIA_HOPPER``: num_warps
    [2, 4] on Hopper, up to 8 or 16 elsewhere, fla/ops/common/chunk_delta_h.py, chunk_o.py)."""
    _run(case)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(Case(96, 4, "two_stage", "softplus", steps=1), id="K96-T1-two_stage"),
        pytest.param(Case(96, 4, "direct", "lb-1", pack="pack-0-1-63"), id="K96-pack-direct"),
        pytest.param(Case(96, 1, "direct", "safe_lb-5", steps=65), id="K96-T65-direct"),
    ],
)
def test_kda_memory_safety(model_parallel, case):
    """Small K=96 (non-power-of-two head dim) cases for the compute-sanitizer pass.

    The kernel inputs sit in NaN guard buffers (``_GuardedKernelCall``); the same numeric
    assertions as ``test_kda_matches_fp64_reference`` apply.
    """
    _run(case)
