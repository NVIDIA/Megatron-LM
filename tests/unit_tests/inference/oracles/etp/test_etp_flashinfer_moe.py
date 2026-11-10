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

Cases that fail on ``main`` because of a bug an open fix PR fixes (expect-fail tasks there; they
pass on the probe tree that merges the fix):
- ``test_bounded_rows_hold_every_replica``: R03-F2, fixed by #8876 (t3-etp). With TP 1, ETP 2,
  EP 2, 64 requests and capacity 128, ``main`` bounds the rows by ``64 * EP = 128`` while the
  gather holds four replicas' 256 decode rows; tp_ep ranks 2 and 3 get stale rows.
- ``test_cutlass_mxfp8_unaligned_expert_shard``: R03-F4, fixed by #8877 (t3-r03f4). FFN 384 at
  ETP 2 passes ``main``'s global 128-alignment check; the 192-wide shards raise at the first
  forward. With the fix the config is rejected when it is built.

Hardware: GB200, 2 and 4 GPUs in one NVLink domain (NVLS multicast), FlashInfer 0.7.2.post1.
"""

import contextlib
import gc
from types import SimpleNamespace

import pytest
import torch

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
    "nvfp4-relu2": dict(activation="squared_relu", recipe="nvfp4"),
    "nvfp4-swiglu": dict(activation="swiglu", recipe="nvfp4"),
    "mxfp8-routed": dict(activation="squared_relu", recipe="mxfp8"),
    "mxfp8-cutlass": dict(activation="clamp", recipe="mxfp8"),
    "bf16-cutlass": dict(activation="squared_relu", recipe="bf16"),
    "bf16-cutlass-clamp": dict(activation="clamp", recipe="bf16"),
}
PADDING = 64.0  # finite padding-row hidden states: FlashInfer quantizes every row it is given


@pytest.fixture(scope="module")
def env(request):
    with etp.parallel_layout(LAYOUTS[request.param]) as env:
        yield env


def _require(recipe: str, clamp: bool) -> None:
    """The kernels this hardware targets must be present (a skip is not evidence here)."""
    from megatron.core.inference.moe import flashinfer_mxfp8, flashinfer_nvfp4

    if recipe == "nvfp4":
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
    return etp.make_config(problem, layout, "flashinfer", **extra)


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
    _require(recipe, problem.clamp is not None)
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


def _trusted(layout):
    def fn(problem, x, routing, probs):
        return etp.trusted_flashinfer(problem, x, routing, probs, layout)

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
    problem = etp.lossless_problem(E, H, FFN, topk, KINDS[kind]["activation"], case["seed"])
    counts = etp.replica_counts(case["counts"], layout, M)
    tokens = etp.lossless_tokens(problem, sum(n for n, _ in counts), case["seed"] + 1, ep=layout.ep)
    batch = etp.rank_batch(tokens, counts, env, padding_hidden=PADDING)
    etp.allocate_nvls(env, M, topk, H)
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
        trusted_fn=_trusted(layout),
        what=f"{layout.name} rank {env.rank} {kind}",
    )


@pytest.mark.parametrize("kind", ["mxfp8-routed", "nvfp4-relu2"])
@pytest.mark.parametrize("env", ["w4-tp1-etp2-ep2"], indirect=True, scope="module")
def test_bounded_rows_hold_every_replica(env, kind):
    """A decode-only step with bounded FlashInfer rows must still compute every gathered token.

    TP 1, ETP 2, EP 2: the tp_ep group holds four replicas. Each serves 64 decode requests
    (``max_requests``), so the gather holds 256 rows; the token capacity is 128. Each rank's
    ``DynamicInferenceContext`` sets up the step and the controller's
    ``_dynamic_step_forward_logits`` decides the bounded rows, then runs the real MoE layer as
    the model. Fails on ``main`` (R03-F2: the bound counts EP replicas only, 64 * 2 = 128, so
    rows [128, 256) -- tp_ep ranks 2 and 3 -- keep stale reduce-scatter rows); passes with #8876.
    """
    from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
    from megatron.core.inference.config import InferenceConfig
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )
    from megatron.core.inference.utils import InferenceMode

    layout, requests, capacity = env.layout, 64, 128
    problem = etp.lossless_problem(E, H, FFN, 2, "squared_relu", seed=301)
    config = _config(problem, layout, kind, inference_flashinfer_mxfp8_token_capacity=capacity)
    etp.release_buffers()
    env.nvls_dims = None
    context = DynamicInferenceContext(
        model_config=config,
        inference_config=InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.05,
            block_size_tokens=256,
            max_tokens=256,
            max_requests=requests,
            unified_memory_level=0,
        ),
    )
    try:
        layer = _build(config, problem, env, kind)
        counts = [(requests, requests)] * layout.replicas
        tokens = etp.lossless_tokens(problem, requests * layout.replicas, seed=302, ep=layout.ep)
        batch = etp.rank_batch(tokens, counts, env)
        context.add_dummy_requests_for_cudagraph_capture(
            InferenceBatchDimensions(
                token_count=requests, prefill_req_count=0, decode_req_count=requests
            )
        )
        context.initialize_attention_state()
        assert int(context.gpu_view.real_token_count.item()) == requests

        step = {}
        x, probs, routing = batch.x.clone(), batch.probs.clone(), batch.routing.clone()
        etp.bypass_router(layer, probs, routing)

        def model_forward(inference_input):
            step["bounded"] = InferenceMode.use_bounded_mxfp8_rows()
            out, _ = layer(x.view(requests, 1, H))
            step["out"] = out.view(requests, H)
            return torch.zeros(1, 1, 1, device="cuda")

        controller = SimpleNamespace(  # the attributes the controller's forward reads
            inference_wrapped_model=SimpleNamespace(
                inference_context=context, run_one_forward_step=model_forward
            ),
            model_config=config,
            num_speculative_tokens=0,
            model_is_pipeline_parallel=False,
            _enable_cuda_graph=False,
            _overlap_gc=False,
            _select_async_sched_routing_buffer=lambda: None,
        )
        stale = etp.local_expert_range(problem, env)[0]
        etp.poison_nvls(stale)
        with torch.no_grad(), InferenceMode.active(), _fp8_context(config):
            TextGenerationController._dynamic_step_forward_logits(controller, None, None)
        etp.check_nvls(
            etp.gathered_rows(env, requests), stale, flashinfer=True, what=f"{layout.name}"
        )
        etp.check_output(
            step["out"],
            batch,
            tokens,
            problem,
            layout.etp,
            trusted_fn=_trusted(layout),
            what=(
                f"{layout.name} tp_ep rank {env.tp_ep_rank} {kind} (bounded rows"
                f" {step['bounded']}, capacity {capacity}, gathered rows"
                f" {requests * layout.tp_ep})"
            ),
        )
    finally:
        del context
        gc.collect()
        etp.release_buffers()
        env.nvls_dims = None


@pytest.mark.parametrize("env", ["w2-tp1-etp2-ep1"], indirect=True, scope="module")
def test_cutlass_mxfp8_unaligned_expert_shard(env):
    """FFN 384 at ETP 2 on the CUTLASS MXFP8 path: reject at construction, or serve correctly.

    Each rank's FFN shard is 192 wide; the CUTLASS MXFP8 packing needs 128-aligned rows and
    columns. Either ``TransformerConfig`` or the experts must reject the layout when built, or
    the layer must compute the reference. Fails on ``main`` (R03-F4: accepted, then the first
    forward raises in ``prepare_cutlass_mxfp8_weights``, on every rank alike, after the gather);
    passes with #8877, which rejects the config.
    """
    layout = env.layout
    problem = etp.lossless_problem(E, H, 384, 2, "clamp", seed=401)
    try:
        config = _config(problem, layout, "mxfp8-cutlass")
        layer = _build(config, problem, env, "mxfp8-cutlass")
    except ValueError as exc:
        assert "128" in str(exc) and "expert" in str(exc).lower(), str(exc)
        return
    counts = etp.replica_counts("uneven", layout, M)
    tokens = etp.lossless_tokens(problem, sum(n for n, _ in counts), seed=402, ep=layout.ep)
    batch = etp.rank_batch(tokens, counts, env, padding_hidden=PADDING)
    etp.allocate_nvls(env, M, 2, H)
    with _fp8_context(config):
        out = etp.run_step(env, layer, problem, batch, flashinfer=True)
    etp.check_output(
        out,
        batch,
        tokens,
        problem,
        layout.etp,
        trusted_fn=_trusted(layout),
        what=f"{layout.name} rank {env.rank} unaligned CUTLASS MXFP8 shard",
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
        trusted_fn=_trusted(layout),
        what=f"{layout.name} rank {env.rank} {kind} small",
    )
