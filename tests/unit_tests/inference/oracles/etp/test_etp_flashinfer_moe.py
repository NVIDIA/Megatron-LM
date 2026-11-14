# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the expert-tensor-parallel inference MoE layer with the FlashInfer backends (GB200).

Contract: ``etp_moe.py`` -- every real token's output is ``sum_k p_k FC2_e(act(FC1_e x))`` with
the *unsharded* expert weights, whatever the attention TP, expert TP (each rank holds FFN columns
``[r F/ETP, (r+1) F/ETP)``), EP and token placement; padding rows route to no expert (output 0).
For the FlashInfer backends the documented differences are: the dispatcher sets every gathered
routing row to -1 before the gather, the kernels write the whole reduce-scatter buffer, routed
MXFP8 packs probabilities as bf16, and quantization (MXFP8, NVFP4) is the only approximation
besides the bf16 outputs. A layout the backend cannot run must be rejected when the config or
the layer is built, not at the first forward. With bounded rows
(``inference_flashinfer_mxfp8_token_capacity``) every gathered token must still get its output.

Under test: ``MoELayer`` (``get_inference_optimized_moe_spec``, NVLS dispatcher over ``tp_ep``)
with ``inference_grouped_gemm_backend='flashinfer'``:
- ``nvfp4-relu2`` / ``nvfp4-swiglu``: per-token NVFP4 (``inference_moe_nvfp4_recipe``, CuTe DSL);
- ``mxfp8-routed``: MXFP8 weights, squared ReLU, TRT-LLM routed MoE;
- ``mxfp8-cutlass``: MXFP8 weights, tanh-clamped squared ReLU, CUTLASS ``ClampedRelu2``;
- ``bf16-cutlass`` / ``bf16-cutlass-clamp``: BF16 CUTLASS fused MoE (``Relu2``, ``ClampedRelu2``).
The CI image installs FlashInfer without ``flashinfer-jit-cache`` (#8873 adds it), so
``MoELayer``'s guard against a slow first-forward JIT build is bypassed for the CUTLASS and
routed MXFP8 kinds; the kernels are JIT-built on first use (a persistent cache keeps them warm).

Oracle: inputs on which every quantization is lossless (``lossless_problem``: rank-one FC1 with
block-constant levels, grid tokens and FC2 weights, after O4b), so the fp64 MoE of the values is
the exact contract; the trusted implementation is torch with the FlashInfer design's least precise
rounding points (bf16 route outputs accumulated in bf16 per rank, fp32 reduce-scatter, bf16
output); per row, factor 2. Memory: NaN-payload poison in the NVLS buffers, 0xA5 bytes past them.

Layouts that fail on ``main`` are left to the fix PRs, which test them:
- R03-F2: with TP 1, ETP 2, EP 2, 64 requests and capacity 128, ``main`` bounds the rows by
  ``64 * EP = 128`` while the gather holds four replicas' 256 decode rows; tp_ep ranks 2 and 3
  get stale rows. #8876 (``fix/etp-ep-wide-decisions``) tests it in
  ``test_flashinfer_moe_bounded_rows.py``
  (``test_decode_steps_bound_rows_only_when_every_replica_fits``).
- R03-F4: FFN 384 at ETP 2 passes ``main``'s global 128-alignment check, and the 192-wide CUTLASS
  MXFP8 shards raise at the first forward. #8877 (``fix/cutlass-mxfp8-etp-alignment``) rejects the
  config and tests it in ``test_flashinfer_mxfp8.py``
  (``test_flashinfer_cutlass_mxfp8_rejects_unaligned_expert_tp_shards``).
- ETP-N1: routed MXFP8 with top-k = number of experts is accepted, then FlashInfer rejects it at
  the first forward. #9317 (``fix/flashinfer-routed-topk``) rejects the config and tests it in
  ``test_flashinfer_mxfp8.py``
  (``test_routed_mxfp8_config_rejects_routing_flashinfer_cannot_launch``).

Hardware: GB200, 2 and 4 GPUs in one NVLink domain (NVLS multicast), FlashInfer 0.7.2.post1.
"""

import contextlib

import pytest

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.etp import etp_moe as etp

pytestmark = [pytest.mark.launch_on_gb200, ou.requires_sm(ou.BLACKWELL)]

E, H, FFN = 8, 256, 512  # FFN shards of 512 (ETP 1) and 256 (ETP 2): 128-aligned for CUTLASS
M = 64
LAYOUTS = {
    layout.name: layout
    for layout in (
        etp.Layout(1, 1, 2),
        etp.Layout(1, 2, 1),
        etp.Layout(2, 2, 1),
        etp.Layout(1, 2, 2),
    )
}
KINDS = {
    "nvfp4-relu2": dict(activation="squared_relu", recipe="nvfp4", backend="flashinfer"),
    "nvfp4-swiglu": dict(activation="swiglu", recipe="nvfp4", backend="flashinfer"),
    "mxfp8-routed": dict(activation="squared_relu", recipe="mxfp8", backend="flashinfer"),
    "mxfp8-cutlass": dict(activation="clamp", recipe="mxfp8", backend="flashinfer"),
    "bf16-cutlass": dict(activation="squared_relu", recipe="bf16", backend="flashinfer"),
    "bf16-cutlass-clamp": dict(activation="clamp", recipe="bf16", backend="flashinfer"),
    # MCore's MXFP8 grouped GEMM (scaled_grouped_mm, Blackwell only) under the torch and vLLM
    # backends.
    "torch-mxfp8-relu2": dict(activation="squared_relu", recipe="mxfp8", backend="torch"),
    "torch-mxfp8-swiglu": dict(activation="swiglu", recipe="mxfp8", backend="torch"),
    "vllm-mxfp8-clamp": dict(activation="clamp", recipe="mxfp8", backend="vllm"),
}
PADDING = 64.0  # finite padding-row hidden states: FlashInfer quantizes every row it is given


@pytest.fixture(scope="module")
def env(request):
    with etp.parallel_layout(LAYOUTS[request.param]) as env:
        yield env


def _require(kind: str, clamp: bool) -> None:
    """The kernels this hardware targets must be present (a skip is not evidence here)."""
    from megatron.core.inference.moe import flashinfer_mxfp8, flashinfer_nvfp4

    recipe = KINDS[kind]["recipe"]
    if KINDS[kind]["backend"] != "flashinfer":
        from megatron.core.inference.moe.fused_moe import HAVE_SCALED_GMM

        assert HAVE_SCALED_GMM, "MXFP8 grouped GEMMs need torch scaled_grouped_mm"
    elif recipe == "nvfp4":
        assert flashinfer_nvfp4.HAVE_FLASHINFER_ROUTED_NVFP4, "per-token NVFP4 MoE unavailable"
    elif recipe == "mxfp8" and clamp:
        flashinfer_mxfp8.require_flashinfer_cutlass_mxfp8()
    elif recipe == "mxfp8":
        flashinfer_mxfp8.require_flashinfer_routed_mxfp8()


def _config(problem, layout, kind, **overrides):
    recipe = KINDS[kind]["recipe"]
    extra = {}
    if recipe == "mxfp8":
        extra = dict(fp8="hybrid", fp8_recipe="mxfp8", fp8_param=True, use_cpu_initialization=False)
    elif recipe == "nvfp4":
        extra = dict(inference_moe_nvfp4_recipe="per_token")
    extra.update(overrides)
    return etp.make_config(problem, layout, KINDS[kind]["backend"], **extra)


@contextlib.contextmanager
def _without_jit_cache_guard():
    """MoELayer requires flashinfer-jit-cache for the CUTLASS/routed kinds; the image lacks it."""
    from megatron.core.inference import utils

    check = utils.check_flashinfer_jit_cache_installed
    utils.check_flashinfer_jit_cache_installed = lambda *args, **kwargs: None
    try:
        yield
    finally:
        utils.check_flashinfer_jit_cache_installed = check


def _fp8_context(config):
    if not config.fp8:
        return contextlib.nullcontext()
    from megatron.core.fp8_utils import get_fp8_context

    return get_fp8_context(config, 0)


def _build(config, problem, env, kind):
    """The production layer with this rank's shards: MXFP8 weights through the checkpoint path
    (TE MXFP8 parameters converted by ``quantize_model_to_mxfp8``, then refit in place)."""
    from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec

    recipe = KINDS[kind]["recipe"]
    _require(kind, problem.clamp is not None)
    with _without_jit_cache_guard():
        if recipe == "mxfp8":
            from megatron.core.fp8_utils import get_fp8_context
            from megatron.core.inference.quantization.utils import quantize_model_to_mxfp8

            with get_fp8_context(config, 0, is_init=True):
                layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
            quantize_model_to_mxfp8(layer, backend="triton")
        else:
            layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
    etp.load_shards(layer, problem, env)
    return layer


def _trusted(layout, kind="mxfp8-routed"):
    """FlashInfer kinds: the least precise FlashInfer finalize; MCore MXFP8: the torch/vLLM
    design's rounding points (bf16 route outputs, fp32 unpermute and reduce-scatter)."""

    def fn(problem, x, routing, probs):
        if KINDS[kind]["backend"] == "flashinfer":
            return etp.trusted_flashinfer(problem, x, routing, probs, layout)
        return etp.trusted(problem, x, routing, probs, layout.etp)

    return fn


def _cases():
    cases = ou.pairwise_cases(
        {"kind": list(KINDS), "counts": ["uneven", "padded", "full"], "topk": ["2", "local"]}
    )
    params = [
        pytest.param(
            dict(case, seed=7000 + 10 * i), id=f"{case['kind']}-{case['counts']}-k{case['topk']}"
        )
        for i, case in enumerate(cases)
    ]
    return sorted(params, key=lambda p: p.values[0]["topk"])


@pytest.mark.parametrize("case", _cases())
@pytest.mark.parametrize("env", list(LAYOUTS), indirect=True, scope="module")
def test_matches_fp64_reference(env, case):
    """Every real token against the fp64 MoE of the unsharded (lossless) weights."""
    layout, kind = env.layout, case["kind"]
    topk = E // layout.ep if case["topk"] == "local" else int(case["topk"])
    if kind == "mxfp8-routed" and topk >= E:
        # TRT-LLM's routed MoE needs num_experts > top_k (ETP-N1, #9317); E - 2 keeps the
        # probability rows 8-byte aligned for main's NVLS gather.
        topk = E - 2
    problem = etp.lossless_problem(E, H, FFN, topk, KINDS[kind]["activation"], case["seed"])
    counts = etp.replica_counts(case["counts"], layout, M)
    tokens = etp.lossless_tokens(problem, sum(n for n, _ in counts), case["seed"] + 1, ep=layout.ep)
    batch = etp.rank_batch(tokens, counts, env, padding_hidden=PADDING)
    etp.allocate_nvls(env, M, topk, H)
    config = _config(problem, layout, kind)
    layer = _build(config, problem, env, kind)
    with _fp8_context(config):
        out = etp.run_step(
            env, layer, problem, batch, flashinfer=KINDS[kind]["backend"] == "flashinfer"
        )
    etp.check_output(
        out,
        batch,
        tokens,
        problem,
        layout.etp,
        trusted_fn=_trusted(layout, kind),
        what=f"{layout.name} rank {env.rank} {kind}",
    )


@pytest.mark.sanitizer
@pytest.mark.parametrize("kind", ["nvfp4-relu2", "mxfp8-routed"])
@pytest.mark.parametrize("env", ["w2-tp1-etp2-ep1"], indirect=True, scope="module")
def test_small_case_for_the_sanitizer(env, kind):
    """A small ETP 2 step with uneven counts and padding, for compute-sanitizer memcheck."""
    layout = env.layout
    problem = etp.lossless_problem(E, H, FFN, 2, KINDS[kind]["activation"], seed=501)
    counts = [(5, 8), (1, 1)]
    tokens = etp.lossless_tokens(problem, 6, seed=502, ep=layout.ep)
    batch = etp.rank_batch(tokens, counts, env, padding_hidden=PADDING)
    etp.allocate_nvls(env, 16, 2, H)
    config = _config(problem, layout, kind)
    layer = _build(config, problem, env, kind)
    with _fp8_context(config):
        out = etp.run_step(env, layer, problem, batch, flashinfer=True)
    etp.check_output(
        out,
        batch,
        tokens,
        problem,
        layout.etp,
        trusted_fn=_trusted(layout, kind),
        what=f"{layout.name} rank {env.rank} {kind} small",
    )
