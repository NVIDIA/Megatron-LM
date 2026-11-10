# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the Shortcut MoE inference overlap (#8257):
`ShortcutMoEBlock._forward_inference_overlap`
(megatron/core/models/hybrid/shortcut_block.py:459, taken at :592), which runs the routed half of a
shortcut MoE layer (router -> NVLS all-gather-v dispatch -> routed experts -> NVLS reduce-scatter-v
combine) on the process-wide MoE side stream while the paired Mamba or attention layer runs on the
main stream, inside the engine's CUDA graphs.

Contract (from the docstrings of `_forward_inference_overlap` and `forward`, the config option
`InferenceConfig.inference_shortcut_moe_overlap` and its validation,
`InferenceConfig.validate_shortcut_moe_model`,
config.py:853-888, written before reading the method body): the option changes only *where* the
routed work runs, never *what* it computes. The pair's output is defined by the eager, single-stream
schedule (`forward` with the option off): the router reads the pair input (the shortcut), the
predecessor runs its ordinary cache-aware forward, the shared expert reads the predecessor's
output, and the routed and shared outputs join before the MoE residual. With the option on, the
same kernels run on the same data, the main stream's kernels in the same order and the side
stream's kernels in the same order, so every output is the same, bit for bit:

* the logits of every step (decode-only, prefill-only, mixed and chunked-prefill steps), and the
  generated tokens; through them every KV-cache and Mamba-state update of every earlier step;
* eagerly and under CUDA-graph capture and replay (the fork and the join are captured);
* with uneven token counts per expert-parallel rank, ranks with no requests that run the engine's
  dummy forward, and requests admitted while others decode;
* at any relative timing of the two streams. A missing or misplaced fork lets the side stream read
  the pair input before the main stream has written it; a missing join lets the main stream read
  the combined routed output before the side stream has written it. Delays injected into each
  stream (`torch.cuda._sleep`) make either hazard produce wrong values instead of a lucky pass.

The reference is the option-off run of the same model on the same requests. A metamorphic test is
used because no independent numeric reference can see a stream-ordering bug that a correct kernel
schedule would hide: a race shows as different bits, never as a larger error. Tolerance class:
exact (bitwise), justified by the contract above (identical kernels, operands and per-stream order).
A third run, option off again, is the control: it shows the option-off result is reproducible on
this system (the NVLS in-switch reduction and every kernel are run-to-run deterministic for a fixed
batch), so a difference between the option-on and option-off runs is the option's.

Non-vacuity, asserted in every case: the overlap path ran (`_forward_inference_overlap` was entered
with the option on and never with it off); every NVLS dispatch and combine of the option-on run was
issued on the shortcut side stream, and none of the option-off run; the dispatcher is the NVLS one
with symmetric-memory handles (real multimem collectives, not the NCCL fallback); the runs took
decode-only, prefill and mixed steps, chunked prefills, dummy-forward steps on idle ranks and steps
whose token counts differ across ranks; in the graph cases, steps were replayed from CUDA graphs.

Model: a 4-layer hybrid `ME*E` (Mamba -> shortcut MoE, attention -> shortcut MoE), so both
predecessor types run and two pairs share the dispatcher's symmetric buffers, the vLLM fused-MoE
buffers and the side stream within a step and across steps (the buffer-reuse hazard).
Nemotron-3-Nano-like routing (sigmoid, expert bias, top-6 of 16, shared expert, post norm), BF16,
TP 1, EP = the whole node (8 on H100, 4 on GB200), the default async scheduler. The `wide` cases use
three wide residual streams, which add the shortcut residual read on the side stream. The
`graphs-nano3` case has Nemotron 3 Nano's production widths (hidden 2688, 128 experts of 1856,
a 3712-wide shared expert).

Not covered here (see packet o5): an fp64 per-token reference of the MoE half of the pair (the
kernels it runs have their own oracles: the NVLS collectives in this package, the fused MoE in
packet O3); EP sizes other than the whole node; FP8/FP4 and batch-invariant mode, which the option
rejects at configuration.
"""

import gc
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.activations import squared_relu
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.engine_factory import build_dynamic_inference_engine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_inference_stack_spec,
    wide_residual_hybrid_inference_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
    delete_cuda_graphs,
)
from megatron.core.transformer.enums import AttnBackend, InferenceCudaGraphScope
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.nvls import nvls_oracle_utils as nu
from tests.unit_tests.test_utilities import Utils

VOCAB_SIZE = 256
MAX_SEQ_LEN = 512
MAX_TOKENS = 128
"""Per-rank token budget of a step: prompts longer than this take chunked prefill."""
MAX_REQUESTS = 16
NUM_CUDA_GRAPHS = 4
NUM_EXPERTS = 16
"""Two local experts per rank at EP 8 (H100), four at EP 4 (GB200)."""
PATTERN = "ME*E"
PRODUCER_DELAY_CYCLES = 100_000_000
"""~50 ms (H100 and GB200 run near 2 GHz) before the main stream writes the pair input."""
ROUTER_DELAY_CYCLES = 20_000_000
"""~10 ms before the side stream's router reads the pair input."""
COMBINE_DELAY_CYCLES = 100_000_000
"""~50 ms before the side stream's combine writes the routed output.

Sizing. An eager step is host-bound: the host launches the routed half's kernels, then the
predecessor's, then the join, so either stream's kernels can finish before the other stream's
dependent kernel is even launched, and a 1 ms delay hides a missing fork or join (mutants
o5p-sc-001..003 survived the eager cases at 1 ms; only graph capture failed). Each delay must exceed
the host time it competes with (a few ms of launches), and the router delay must stay shorter than
the producer delay, or it would itself wait out the missing fork."""
MAX_STEPS = 200

# (step at which the request arrives, prompt length, tokens to generate), per EP rank modulo 4.
# Rank 0: a chunked prefill (200 > MAX_TOKENS), a short prompt, and a request admitted while the
# others decode (mixed steps). Rank 1: idle for three steps (dummy forwards), then a burst whose
# second prompt is chunked. Rank 2: one short request that finishes early (dummy forwards after).
# Rank 3: a long prompt and a request admitted one step later. Ranks 4-7 shift the lengths, so all
# eight ranks hold different token counts in most steps.
_PLANS = {
    0: [(0, 200, 6), (0, 9, 3), (5, 70, 4)],
    1: [(3, 33, 5), (3, 150, 3)],
    2: [(0, 2, 2)],
    3: [(0, 131, 3), (1, 64, 7)],
}


def gpu_test(fn):
    """NVLS needs Hopper or Blackwell; GB200 CI launches the file through launch_on_gb200."""
    return pytest.mark.launch_on_gb200(ou.requires_sm(ou.HOPPER, ou.BLACKWELL)(fn))


def _request_plan(rank):
    shift = 7 * (rank // 4)
    return [(step, length + shift, tokens) for step, length, tokens in _PLANS[rank % 4]]


def _requirements():
    """The reason this model cannot run here, or None (identical on every rank)."""
    from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
    from megatron.core.utils import is_fa_min_version

    if nu.WORLD_SIZE < 2:
        return "the overlap needs expert parallelism: run with at least 2 ranks"
    available, reason = _check_mamba_sequence_packing_support(for_inference_not_training=True)
    if not available:
        return reason
    if not is_fa_min_version("2.7.3"):
        return "dynamic batching needs flash-attn >= 2.7.3"
    return None


@pytest.fixture(scope="module")
def expert_parallel():
    """EP over the whole job (TP 1), initialized once: the dispatcher's symmetric buffers must
    never outlive their process group."""
    reason = _requirements()
    if reason is not None:
        nu.unavailable(reason)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=nu.WORLD_SIZE,
    )
    yield parallel_state.get_expert_model_parallel_rank()
    try:
        delete_cuda_graphs()
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
    finally:
        Utils.destroy_model_parallel()


TEST_DIMS = dict(
    hidden_size=128,
    ffn_hidden_size=128,
    num_attention_heads=4,
    num_query_groups=2,
    num_moe_experts=NUM_EXPERTS,
    moe_ffn_hidden_size=128,
    moe_shared_expert_intermediate_size=256,
)
NANO3_DIMS = dict(
    hidden_size=2688,
    ffn_hidden_size=1856,
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    num_moe_experts=128,
    moe_ffn_hidden_size=1856,
    moe_shared_expert_intermediate_size=3712,
)
"""Nemotron 3 Nano 30B-A3B's layer widths and experts
(examples/post_training/modelopt/conf/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.sh); its Mamba
dimensions and routing are the test model's."""


def _build_model(wide, dims=None):
    model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
    torch.manual_seed(123)
    config = TransformerConfig(
        num_layers=len(PATTERN),
        **(TEST_DIMS if dims is None else dims),
        moe_router_topk=6,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_topk_scaling_factor=2.5,
        moe_router_dtype="fp32",
        moe_shared_expert_overlap=False,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_shortcut_connection=True,
        moe_shortcut_post_norm=True,
        inference_moe_token_dispatcher_type="nvls",
        activation_func=squared_relu,
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="inference_optimized",
        tensor_model_parallel_size=1,
        expert_model_parallel_size=nu.WORLD_SIZE,
        expert_tensor_parallel_size=1,
        use_cpu_initialization=True,
        attention_backend=AttnBackend.local,
        cuda_graph_impl="local",
        inference_cuda_graph_scope=InferenceCudaGraphScope.block,
        moe_pad_experts_for_cuda_graph_inference=False,
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_num_heads=64,
        is_hybrid_model=True,
        wide_residual=WideResidualConfig(num_streams=3) if wide else None,
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=(
            wide_residual_hybrid_inference_stack_spec if wide else hybrid_inference_stack_spec
        ),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQ_LEN,
        hybrid_layer_pattern=PATTERN,
    ).cuda()
    model.eval()
    blocks = [m for m in model.modules() if isinstance(m, ShortcutMoEBlock)]
    assert len(blocks) == 2, f"expected two shortcut pairs in {PATTERN}, found {len(blocks)}"
    return model


def _reset_cuda_graph_state(model):
    """Drop every captured graph, so each run captures its own against its own context."""
    delete_cuda_graphs()
    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []
    _CudagraphGlobalRecord.cudagraph_inference_record = []
    CudaGraphManager.global_mempool = None
    for module in model.modules():
        if isinstance(module, CudaGraphManager):
            module.cudagraph_runners.clear()
            module.custom_cudagraphs_lookup_table.clear()


class _Spy:
    """Counts entries into the overlap path and records the stream of every NVLS dispatch and
    combine (eager calls and calls made while capturing a graph)."""

    def __init__(self, monkeypatch):
        self.overlap_calls = 0
        self.collective_streams = []
        spy = self
        overlap = ShortcutMoEBlock._forward_inference_overlap
        dispatch = NVLSAllGatherVDispatcher.token_dispatch
        combine = NVLSAllGatherVDispatcher.token_combine

        def counted_overlap(block, *args, **kwargs):
            spy.overlap_calls += 1
            return overlap(block, *args, **kwargs)

        def recorded_dispatch(dispatcher, *args, **kwargs):
            spy.collective_streams.append(torch.cuda.current_stream().cuda_stream)
            return dispatch(dispatcher, *args, **kwargs)

        def recorded_combine(dispatcher, *args, **kwargs):
            spy.collective_streams.append(torch.cuda.current_stream().cuda_stream)
            return combine(dispatcher, *args, **kwargs)

        monkeypatch.setattr(ShortcutMoEBlock, "_forward_inference_overlap", counted_overlap)
        monkeypatch.setattr(NVLSAllGatherVDispatcher, "token_dispatch", recorded_dispatch)
        monkeypatch.setattr(NVLSAllGatherVDispatcher, "token_combine", recorded_combine)

    def reset(self):
        self.overlap_calls = 0
        self.collective_streams = []


def _inject_delays(monkeypatch):
    """Delay each stream in turn (values unchanged).

    Main stream: the pair input is re-materialized after a ~50 ms sleep, so a side stream that does
    not wait for the fork reads it before it is written. Side stream: a ~10 ms sleep before the
    router (a main-stream write to the pair input after the fork would be seen) and a ~50 ms sleep
    before the combine (a main stream that does not wait for the join reads the combined output
    early).
    """
    forward = ShortcutMoEBlock.forward
    preprocess = ShortcutMoEBlock._moe_router_preprocess
    launch_combine = ShortcutMoEBlock._launch_combine

    def late_input(hidden_states):
        torch.cuda._sleep(PRODUCER_DELAY_CYCLES)
        return hidden_states.clone()

    def delayed_forward(block, *args, **kwargs):
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = late_input(kwargs["hidden_states"])
        else:
            args = (late_input(args[0]),) + tuple(args[1:])
        return forward(block, *args, **kwargs)

    def delayed_preprocess(block, *args, **kwargs):
        torch.cuda._sleep(ROUTER_DELAY_CYCLES)
        return preprocess(block, *args, **kwargs)

    def delayed_combine(block, *args, **kwargs):
        torch.cuda._sleep(COMBINE_DELAY_CYCLES)
        return launch_combine(block, *args, **kwargs)

    monkeypatch.setattr(ShortcutMoEBlock, "forward", delayed_forward)
    monkeypatch.setattr(ShortcutMoEBlock, "_moe_router_preprocess", delayed_preprocess)
    monkeypatch.setattr(ShortcutMoEBlock, "_launch_combine", delayed_combine)


def _build_engine(model, *, overlap, graphs):
    _reset_cuda_graph_state(model)
    config = InferenceConfig(
        max_sequence_length=MAX_SEQ_LEN,
        buffer_size_gb=0.25,
        mamba_inference_state_config=MambaInferenceStateConfig.from_model(model),
        max_tokens=MAX_TOKENS,
        max_requests=MAX_REQUESTS,
        enable_chunked_prefill=True,
        num_cuda_graphs=NUM_CUDA_GRAPHS if graphs else None,
        use_cuda_graphs_for_non_decode_steps=True,
        unified_memory_level=0,
        sampling_backend="torch",
        inference_shortcut_moe_overlap=overlap,
    )
    tokenizer = SimpleNamespace(vocab_size=VOCAB_SIZE, detokenize=lambda tokens: "")
    return build_dynamic_inference_engine(model=model, tokenizer=tokenizer, inference_config=config)


class _Recorder:
    """Records every forward of an engine: step kind, batch shape, graph use, last-token logits."""

    def __init__(self, engine):
        self.context = engine.context
        self.steps = []
        self.dummy = False
        wrapper = engine.controller.inference_wrapped_model
        forward = wrapper.run_one_forward_step

        def recording(inference_input, *args, **kwargs):
            logits = forward(inference_input, *args, **kwargs)
            ctx = self.context
            dims = ctx.batch_dimensions
            rows = ctx.get_active_request_count()
            self.steps.append(
                dict(
                    dummy=self.dummy,
                    graph=bool(ctx.using_cuda_graph_this_step()),
                    prefill=int(dims.prefill_req_count),
                    decode=int(dims.decode_req_count),
                    tokens=int(dims.token_count),
                    logits=None if self.dummy else logits[0, :rows].detach().clone(),
                )
            )
            return logits

        wrapper.run_one_forward_step = recording


def _make_request(rank, index, length, tokens):
    generator = torch.Generator().manual_seed(nu.case_seed("shortcut", rank, index))
    prompt = torch.randint(0, VOCAB_SIZE, (length,), generator=generator, dtype=torch.int64)
    return DynamicInferenceRequest(
        request_id=index,
        prompt_tokens=prompt.cuda(),
        sampling_params=SamplingParams(num_tokens_to_generate=tokens, termination_id=-1, top_k=1),
    )


def _drive(engine, rank, recorder):
    """Run this rank's requests to completion in lockstep with every EP rank.

    Mirrors the coordinator loop (`run_engine_with_coordinator`, engines/dynamic/loop.py): each
    iteration the ranks agree whether any of them has work; a rank with none runs the engine's
    dummy forward so its peers' expert collectives complete.
    """
    plan = _request_plan(rank)
    finished, step = {}, 0
    while True:
        for index, (arrival, length, tokens) in enumerate(plan):
            if arrival == step:
                engine._add_request(_make_request(rank, index, length, tokens))
        later = sum(1 for arrival, _, _ in plan if arrival > step)
        local = int(engine.has_unfinished_requests())
        flags = torch.tensor([local, later], dtype=torch.int32, device="cuda")
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        if int(flags.sum()) == 0:
            break
        if int(flags[0]) > 0:
            if local:
                result = engine.step_modern()
                for request in result["finished_requests"]:
                    finished[request.request_id] = list(request.generated_tokens)
            else:
                recorder.dummy = True
                engine.controller.dummy_forward()
                recorder.dummy = False
                engine.context.step_count += 1
                engine.context.prefix_cache_lru_clock += 1
        step += 1
        assert step < MAX_STEPS, "the requests did not finish"
    torch.cuda.synchronize()
    return finished


def _run(model, rank, spy, *, overlap, graphs):
    """One engine run of this rank's request plan; returns its record."""
    spy.reset()
    engine = _build_engine(model, overlap=overlap, graphs=graphs)
    recorder = _Recorder(engine)
    finished = _drive(engine, rank, recorder)
    dispatcher = None
    for module in model.modules():
        if isinstance(module, ShortcutMoEBlock):
            dispatcher = module.moe_layer.mlp.token_dispatcher
    record = dict(
        finished=finished,
        steps=[
            dict(s, logits=None if s["logits"] is None else s["logits"].cpu())
            for s in recorder.steps
        ],
        overlap_calls=spy.overlap_calls,
        collective_streams=list(spy.collective_streams),
        dispatcher=type(dispatcher).__name__,
        symmetric=NVLSAllGatherVDispatcher._symm_agv_hidden is not None
        and NVLSAllGatherVDispatcher._symm_agv_hidden["handle"] is not None
        and NVLSAllGatherVDispatcher._symm_rsv["handle"] is not None,
    )
    del engine, recorder
    _reset_cuda_graph_state(model)
    gc.collect()
    torch.cuda.empty_cache()
    nu.sync_world()
    return record


def _assert_same_run(actual, expected, *, what):
    """Bitwise equality of two runs' step records and generated tokens."""
    assert sorted(actual["finished"]) == sorted(
        expected["finished"]
    ), f"{what}: finished requests {sorted(actual['finished'])} != {sorted(expected['finished'])}"
    for request_id, tokens in expected["finished"].items():
        assert actual["finished"][request_id] == tokens, (
            f"{what}: request {request_id} generated {actual['finished'][request_id]}, expected "
            f"{tokens}"
        )
    shape = lambda s: (s["dummy"], s["graph"], s["prefill"], s["decode"], s["tokens"])
    assert [shape(s) for s in actual["steps"]] == [shape(s) for s in expected["steps"]], (
        f"{what}: the step sequences differ (dummy, graph, prefill, decode, tokens):\n"
        f"  got      {[shape(s) for s in actual['steps']]}\n"
        f"  expected {[shape(s) for s in expected['steps']]}"
    )
    for index, (got, want) in enumerate(zip(actual["steps"], expected["steps"])):
        if want["logits"] is not None:
            ou.assert_exact(got["logits"], want["logits"], what=f"{what}: step {index} logits")


def _assert_not_vacuous(on, off, *, graphs, side_stream):
    """Local half: the overlap ran on the side stream, and the option-off run never touched it."""
    assert (
        on["dispatcher"] == "NVLSAllGatherVDispatcher" and on["symmetric"]
    ), f"the MoE layers did not use the NVLS dispatcher with symmetric memory: {on['dispatcher']}"
    assert on["overlap_calls"] > 0, "the option-on run never entered _forward_inference_overlap"
    assert off["overlap_calls"] == 0, "the option-off run entered _forward_inference_overlap"
    assert on["collective_streams"], "the option-on run issued no NVLS dispatch or combine"
    stray = [s for s in on["collective_streams"] if s != side_stream]
    assert not stray, (
        f"{len(stray)} of {len(on['collective_streams'])} NVLS dispatches/combines of the option-on"
        " run were not on the shortcut side stream"
    )
    assert (
        side_stream not in off["collective_streams"]
    ), "the option-off run issued NVLS collectives on the shortcut side stream"
    if graphs:
        assert any(s["graph"] for s in on["steps"]), "no step of the option-on run used a graph"


def _assert_coverage(records, *, graphs):
    """Global half (every rank's step records): the runs took every kind of step."""
    steps = [r["steps"] for r in records]
    real = [s for rank_steps in steps for s in rank_steps if not s["dummy"]]
    assert any(s["prefill"] == 0 and s["decode"] > 0 for s in real), "no decode-only step"
    assert any(s["prefill"] > 0 and s["decode"] == 0 for s in real), "no prefill-only step"
    assert any(s["prefill"] > 0 and s["decode"] > 0 for s in real), "no mixed step"
    assert any(s["dummy"] for rank_steps in steps for s in rank_steps), "no dummy-forward step"
    counts = list(zip(*[[None if s["dummy"] else s["tokens"] for s in r] for r in steps]))
    assert any(len(set(c)) > 1 for c in counts), "every step had equal token counts on all ranks"
    if graphs:
        assert any(s["graph"] and s["prefill"] > 0 for s in real), "no prefill step from a graph"
        assert any(s["graph"] and s["prefill"] == 0 for s in real), "no decode step from a graph"


def _overlap_case(monkeypatch, rank, *, wide, graphs, delayed, dims=None):
    spy = _Spy(monkeypatch)
    if delayed:
        _inject_delays(monkeypatch)
    model = _build_model(wide, dims)
    runs = {}
    # The first run also absorbs one-time effects (Triton autotuning, JIT, allocator warmup), so
    # the two runs compared are both on warm caches.
    for name in ("off-first", "on", "off"):
        runs[name] = _run(model, rank, spy, overlap=name == "on", graphs=graphs)
    side_stream = ShortcutMoEBlock._get_moe_overlap_stream().cuda_stream

    nu.agree(
        lambda: _assert_not_vacuous(
            runs["on"], runs["off"], graphs=graphs, side_stream=side_stream
        ),
        "non-vacuity",
    )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(
        gathered, dict(steps=[dict(s, logits=None) for s in runs["on"]["steps"]])
    )
    nu.agree(lambda: _assert_coverage(gathered, graphs=graphs), "step coverage")
    nu.agree(
        lambda: _assert_same_run(runs["off"], runs["off-first"], what="control (off vs off)"),
        "the option-off run is not reproducible on this system",
    )
    nu.agree(
        lambda: _assert_same_run(runs["on"], runs["off"], what="overlap on vs off"),
        "Shortcut MoE overlap changed the output",
    )


@gpu_test
@pytest.mark.parametrize(
    "graphs,wide,dims",
    [
        pytest.param(False, False, None, id="eager"),
        pytest.param(True, False, None, id="graphs"),
        pytest.param(True, True, None, id="graphs-wide"),
        pytest.param(True, False, NANO3_DIMS, id="graphs-nano3"),
    ],
)
def test_overlap_is_bitwise_invariant(expert_parallel, monkeypatch, graphs, wide, dims):
    """Option on vs off over decode, prefill, mixed and chunked-prefill steps with uneven per-rank
    loads and dummy ranks, eager and from CUDA graphs: bitwise equal logits and tokens, with an
    option-off control run. `graphs-nano3` has Nemotron 3 Nano's widths and 128 experts."""
    _overlap_case(monkeypatch, expert_parallel, wide=wide, graphs=graphs, delayed=False, dims=dims)


@gpu_test
@pytest.mark.parametrize(
    "graphs", [pytest.param(False, id="eager"), pytest.param(True, id="graphs")]
)
def test_overlap_with_delayed_streams(expert_parallel, monkeypatch, graphs):
    """As above, with each stream delayed in turn: a missing fork or join changes the output."""
    _overlap_case(monkeypatch, expert_parallel, wide=False, graphs=graphs, delayed=True)


# ---------------------------------------------------------------------------------------------
# fp64 per-token reference of a whole pair, with the overlap on
# ---------------------------------------------------------------------------------------------

TIE_MARGIN = 1e-3
"""Tokens whose fp64 top-k selection is decided by less than this score margin are left out: the
kernel's router reads a BF16-rounded norm output, so it may pick the other expert of a near tie.
The routing scores are sigmoid outputs of logits of size ~0.2, so the BF16 perturbation of a score
is ~1e-4."""


def _rms_norm(x, weight, eps, exact):
    """RMSNorm as the model's norm computes it: fp32 math rounded to BF16 (or fp64 throughout)."""
    if exact:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
    y = x.float()
    return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).bfloat16()


def _mlp(x, fc1, fc2, exact):
    """Squared-ReLU MLP: fp64, or BF16 GEMMs (fp32 accumulation) with the activation in fp32."""
    if exact:
        return torch.relu(x @ fc1.T).pow(2) @ fc2.T
    hidden = torch.relu((x @ fc1.T).float()).pow(2).bfloat16()
    return hidden @ fc2.T


def _pair_reference(h, c, params, selection, exact):
    """The pair's output for pair inputs ``h`` and predecessor outputs ``c`` ([T, H]).

    The contract of a shortcut pair in inference: the router and the routed experts read the
    shortcut, RMSNorm_s(h); the shared expert reads RMSNorm_pre(c); their sum passes the post-norm
    and is added to the residual c. Sigmoid routing: scores sigmoid(W_r x); the top-k of scores +
    expert bias are selected (``selection``, from the fp64 scores); their scores, normalized to
    sum 1 and scaled by the top-k scaling factor, weight the experts. ``exact``: fp64 (the
    reference); otherwise BF16 storage with fp32 router math and fp32 expert accumulation, the
    precision the model runs at (the trusted implementation).
    """
    p = params
    dtype = torch.float64 if exact else torch.bfloat16
    acc = torch.float64 if exact else torch.float32
    h, c = h.to(dtype), c.to(dtype)
    w = {name: t.to(dtype) for name, t in p["weights"].items()}
    shortcut = _rms_norm(h, w["shortcut_norm"], p["eps"], exact)
    scores = torch.sigmoid(shortcut.to(acc) @ w["router"].to(acc).T)
    chosen = scores.gather(1, selection)
    probs = chosen / chosen.sum(-1, keepdim=True) * p["scaling"]
    routed = torch.zeros(h.shape, dtype=acc, device=h.device)
    for expert in range(w["fc1"].shape[0]):
        rows, slots = (selection == expert).nonzero(as_tuple=True)
        if rows.numel():
            out = _mlp(shortcut[rows], w["fc1"][expert], w["fc2"][expert], exact)
            routed.index_add_(0, rows, out.to(acc) * probs[rows, slots].unsqueeze(-1))
    routed = routed.to(dtype)
    shared = _mlp(
        _rms_norm(c, w["pre_norm"], p["eps"], exact), w["shared_fc1"], w["shared_fc2"], exact
    )
    return c + _rms_norm(routed + shared, w["post_norm"], p["eps"], exact)


def _pair_params(block):
    """The pair's weights, with every rank's local experts gathered in global expert order."""
    moe = block.moe_layer.mlp
    experts = moe.experts
    local = len(moe.local_expert_indices)

    def gathered(linear):
        stacked = torch.stack([getattr(linear, f"weight{i}").detach() for i in range(local)])
        parts = [torch.empty_like(stacked) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, stacked.contiguous())
        indices = [None] * dist.get_world_size()
        dist.all_gather_object(indices, list(moe.local_expert_indices))
        order = [e for rank_indices in indices for e in rank_indices]
        assert sorted(order) == list(range(NUM_EXPERTS)), f"expert placement {order}"
        full = torch.cat(parts)
        return full[torch.tensor(order).argsort().to(full.device)]

    router = moe.router
    bias = getattr(router, "expert_bias", None)
    return dict(
        eps=block.config.layernorm_epsilon,
        scaling=block.config.moe_router_topk_scaling_factor,
        topk=block.config.moe_router_topk,
        bias=None if bias is None else bias.detach().double(),
        weights=dict(
            shortcut_norm=block.shortcut_pre_mlp_layernorm.weight.detach(),
            router=router.weight.detach(),
            fc1=gathered(experts.linear_fc1),
            fc2=gathered(experts.linear_fc2),
            pre_norm=block.moe_layer.pre_mlp_layernorm.weight.detach(),
            shared_fc1=moe.shared_experts.linear_fc1.weight.detach(),
            shared_fc2=moe.shared_experts.linear_fc2.weight.detach(),
            post_norm=block.shortcut_post_norm.weight.detach(),
        ),
    )


def _capture_pairs(model, engine, recorder):
    """Record each pair's real-token input, predecessor output and output on every real step."""
    captured = {id(b): [] for b in model.modules() if isinstance(b, ShortcutMoEBlock)}
    pending = {}
    handles = []
    for block in model.modules():
        if not isinstance(block, ShortcutMoEBlock):
            continue

        def compute_hook(module, args, output, key=id(block)):
            pending[key] = output[0] if isinstance(output, tuple) else output

        def block_hook(module, args, kwargs, output, key=id(block)):
            if recorder.dummy:
                return
            n = engine.context.active_token_count
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            captured[key].append(
                tuple(t[:n].reshape(n, -1).detach().clone() for t in (hidden, pending[key], output))
            )

        handles.append(block.compute_layer.register_forward_hook(compute_hook))
        handles.append(block.register_forward_hook(block_hook, with_kwargs=True))
    return captured, handles


def _check_pair(block, tokens, params):
    h = torch.cat([t[0] for t in tokens])
    c = torch.cat([t[1] for t in tokens])
    out = torch.cat([t[2] for t in tokens])
    w = params["weights"]
    shortcut64 = _rms_norm(h.double(), w["shortcut_norm"].double(), params["eps"], True)
    scores64 = torch.sigmoid(shortcut64 @ w["router"].double().T)
    ranked = scores64 + (0 if params["bias"] is None else params["bias"])
    top = ranked.topk(params["topk"] + 1, dim=-1)
    selection = top.indices[:, : params["topk"]]
    shortcut16 = _rms_norm(h, w["shortcut_norm"], params["eps"], False)
    ranked16 = torch.sigmoid(shortcut16.float() @ w["router"].float().T).double()
    ranked16 = ranked16 + (0 if params["bias"] is None else params["bias"])
    same_set = (
        ranked16.topk(params["topk"], dim=-1).indices.sort(-1).values == selection.sort(-1).values
    ).all(-1)
    margin = top.values[:, params["topk"] - 1] - top.values[:, params["topk"]]
    keep = same_set & (margin > TIE_MARGIN)
    assert (
        int(keep.sum()) * 2 >= keep.numel()
    ), f"only {int(keep.sum())} of {keep.numel()} tokens have a clear top-{params['topk']}"
    ref = ou.fp64_reference(
        lambda hh, cc: _pair_reference(hh, cc, params, selection[keep], True), h[keep], c[keep]
    )
    trusted = _pair_reference(h[keep], c[keep], params, selection[keep], False)
    ou.assert_close_to_reference(
        out[keep], ref, trusted, dim=-1, what=f"pair (layer {block.layer_number}) output"
    )
    return int(keep.sum()), keep.numel()


@gpu_test
def test_pair_output_matches_fp64_reference(expert_parallel, monkeypatch):
    """With the option on (eager), every real token of both pairs matches an fp64 per-token
    reference of the pair (shortcut norm, sigmoid top-k routing over all ranks' experts, the
    shared expert on the predecessor's output, post-norm and residual), bounded by a BF16 trusted
    implementation with the model's rounding points. Tokens at a near tie in the routing are left
    out (``TIE_MARGIN``); at least half must remain."""
    model = _build_model(wide=False)
    engine = _build_engine(model, overlap=True, graphs=False)
    recorder = _Recorder(engine)
    captured, handles = _capture_pairs(model, engine, recorder)
    try:
        _drive(engine, expert_parallel, recorder)
    finally:
        for handle in handles:
            handle.remove()
    blocks = [b for b in model.modules() if isinstance(b, ShortcutMoEBlock)]
    params = [_pair_params(block) for block in blocks]
    counts = nu.agree(
        lambda: [
            _check_pair(block, captured[id(block)], p)
            for block, p in zip(blocks, params)
            if captured[id(block)]
        ],
        "Shortcut MoE pair output vs fp64",
    )
    checked = sum(k for k, _ in counts)
    assert checked > 0 or not any(captured.values()), "no token was checked"
    del engine, recorder
    _reset_cuda_graph_state(model)
    nu.sync_world()
