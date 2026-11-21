# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the inference MoE layer under expert tensor parallelism, torch and vLLM backends.

Contract: ``etp_moe.py`` (written before reading the dispatcher or the experts). In short, every
real token's output is ``sum_k p_k FC2_e(act(FC1_e x))`` with the *unsharded* expert weights,
whatever the attention TP, expert TP (FFN shards: ``[gate_r; up_r]`` for SwiGLU), EP and token
placement; padding rows route to no expert (output 0); the NVLS buffers are not written past the
step's gathered rows or past their end.

Under test: ``MoELayer`` from ``get_inference_optimized_moe_spec`` with
``inference_moe_token_dispatcher_type='nvls'``, i.e. ``NVLSAllGatherVDispatcher`` over the
expert tensor-and-model-parallel group (``tp_ep``: all-gather-v of hidden states, routing and
probabilities, reduce-scatter-v of fp32 partial outputs) and ``InferenceGroupedMLP`` with the
``torch`` (``mcore_fused_moe``, grouped_mm or DeepGEMM in batch-invariant mode) and ``vllm``
(``vllm_fused_moe``) backends, each rank holding FFN columns ``[r F/ETP, (r+1) F/ETP)``. The
router is bypassed (it has its own oracle in O8a) except in ``test_router_through_the_layer``;
``test_context_sizes_buffers_for_the_layout`` lets ``DynamicInferenceContext`` size and allocate
the buffers and publish the real token count.

Oracles and tolerance classes:
- fp64 per-token MoE from the unsharded weights; the trusted implementation is torch at the
  design's rounding points on the same ETP shards (bf16 GEMM outputs, fp32 combine; the
  probability at the activation in batch-invariant mode); per row, factor 2 (harness default);
- exact: integer experts and tokens on which every rounding point is exact (``integer_problem``),
  so the output is the fp64 reference rounded once, bit for bit;
- metamorphic: in batch-invariant mode a token's output does not change, bit for bit, when the
  tokens move between ranks or the step repeats;
- memory: NaN-payload poison in every NVLS buffer and 0xA5 bytes past each one; padding rows
  poisoned with NaN hidden states, NaN probabilities and valid expert ids; read-only inputs.

Inputs derived from the structure: layouts TP 1/2 x ETP 1/2/4 x EP 1/2 (ETP < TP, = TP, > TP;
tp_ep spanning 1, 2 or 4 replicas); per-replica token counts 1, uneven odd, capacity - 1 beside
capacity, every rank at capacity (the gather buffer full), graph-bucket padding with 0, 1 and odd
real tokens; FFN shard widths 64, 96 (a partial 64-column vLLM tile) and 128 (vLLM BLOCK_SIZE_N 64
and 128); top-k 2 and E/EP (odd top-k fails on main: the NVLS all-gather-v refuses 4- and
12-byte fp32 probability rows; #8562 (``fix/d4-d9-inference-router-shape``) adds a 32-bit path
and tests top-1 through the layer in ``test_moe_dispatching_and_routing.py``); routing to one EP
rank only, with an empty expert, with a hot expert; SwiGLU, squared ReLU and the tanh clamp;
batch-invariant mode on and off; CUDA-graph
replay with the real count moving; production shapes (Nemotron 3 Nano and Qwen3-30B-A3B experts).

Hardware: H100, 2 and 4 GPUs with NVLink and NVLS multicast. Layout ``w<N>-...`` needs a
multiple of N ranks; CI runs the bucket on 8.
"""

import gc

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.etp import etp_moe as etp

pytestmark = ou.requires_sm(ou.HOPPER)

E, H = 8, 128
M = 64  # NVLS per-rank capacity (per_rank_worst_case_token_count) for these cases
ALL = list(etp.LAYOUTS)


@pytest.fixture(scope="module")
def env(request):
    with etp.parallel_layout(etp.LAYOUTS[request.param]) as env:
        yield env


def _topk(choice: str, layout: etp.Layout) -> int:
    return E // layout.ep if choice == "local" else int(choice)


def _sweep():
    cases = ou.pairwise_cases(
        {
            "backend": ["vllm", "torch"],
            "activation": ["squared_relu", "swiglu", "clamp"],
            "ffn_per_rank": [64, 96, 128],
            "topk": ["2", "local"],
            "routing": list(etp.ROUTINGS),
            "counts": list(etp.COUNTS),
            "bi": [False, True],
        }
    )
    params = []
    for i, case in enumerate(cases):
        if case["bi"] and case["backend"] == "torch":
            # DeepGEMM's grouped GEMM (the batch-invariant torch path) at the N and K O8a covers.
            case["ffn_per_rank"] = 128
        case["seed"] = 1000 + 10 * i
        name = "-".join(
            [
                case["backend"],
                case["activation"],
                f"f{case['ffn_per_rank']}",
                f"k{case['topk']}",
                case["routing"],
                case["counts"],
                "bi" if case["bi"] else "nobi",
            ]
        )
        params.append(pytest.param(case, id=name))
    # Group by top-k: the NVLS buffers are sized for it.
    return sorted(params, key=lambda p: p.values[0]["topk"])


@pytest.mark.parametrize("case", _sweep())
@pytest.mark.parametrize("env", ALL, indirect=True, scope="module")
def test_matches_fp64_reference(env, case):
    """Every real token against the fp64 MoE of the unsharded weights; padding rows are 0."""
    layout, backend, bi = env.layout, case["backend"], case["bi"]
    if bi:
        etp.require_batch_invariant_backend(backend)
    topk = _topk(case["topk"], layout)
    routing = case["routing"]
    if routing == "empty-expert" and topk >= E:
        routing = "random"  # every expert is chosen when top-k = E
    problem = etp.random_problem(
        E, H, case["ffn_per_rank"] * layout.etp, topk, case["activation"], case["seed"]
    )
    counts = etp.replica_counts(case["counts"], layout, M)
    tokens = etp.make_tokens(
        problem, sum(n for n, _ in counts), case["seed"] + 1, routing, layout.ep
    )
    batch = etp.rank_batch(tokens, counts, env)
    etp.allocate_nvls(env, M, topk, H)
    config = etp.make_config(problem, layout, backend, batch_invariant=bi)
    layer = etp.build_layer(config, problem, env)
    with etp.batch_invariant_mode(bi):
        out = etp.run_step(env, layer, problem, batch)
    etp.check_output(
        out,
        batch,
        tokens,
        problem,
        layout.etp,
        probs_at_activation=bi,
        what=f"{layout.name} rank {env.rank} {backend}",
    )


def _exact_cases():
    cases = ou.pairwise_cases(
        {
            "backend": ["vllm", "torch"],
            "bi": [False, True],
            "topk": ["2", "local"],
            "routing": list(etp.ROUTINGS),
            "counts": ["uneven", "padded", "full"],
        }
    )
    return sorted(
        (
            pytest.param(
                dict(case, seed=5000 + 10 * i),
                id="-".join(
                    [
                        case["backend"],
                        "bi" if case["bi"] else "nobi",
                        f"k{case['topk']}",
                        case["routing"],
                        case["counts"],
                    ]
                ),
            )
            for i, case in enumerate(cases)
        ),
        key=lambda p: p.values[0]["topk"],
    )


@pytest.mark.parametrize("case", _exact_cases())
@pytest.mark.parametrize("env", ALL, indirect=True, scope="module")
def test_exact_on_integer_inputs(env, case):
    """Integer experts and tokens: the layer output is the fp64 MoE rounded once, bit for bit.

    Exactness (``integer_problem``): FC1 outputs are integers <= 4 in magnitude, squared ReLU
    <= 16, every expert-TP shard's FC2 output an integer <= 128, probabilities powers of two, so
    every bf16 rounding point and every fp32 sum in any order is exact. A dropped or doubled
    shard, a wrong expert, a misplaced token or a lost probability changes the bits.
    """
    layout, backend, bi = env.layout, case["backend"], case["bi"]
    if bi:
        etp.require_batch_invariant_backend(backend)
    topk = _topk(case["topk"], layout)
    routing = case["routing"]
    if routing == "empty-expert" and topk >= E:
        routing = "random"
    problem = etp.integer_problem(E, H, 128 * layout.etp, topk, case["seed"])
    counts = etp.replica_counts(case["counts"], layout, M)
    tokens = etp.make_tokens(
        problem, sum(n for n, _ in counts), case["seed"] + 1, routing, layout.ep
    )
    batch = etp.rank_batch(tokens, counts, env)
    etp.allocate_nvls(env, M, topk, H)
    layer = etp.build_layer(
        etp.make_config(problem, layout, backend, batch_invariant=bi), problem, env
    )
    with etp.batch_invariant_mode(bi):
        out = etp.run_step(env, layer, problem, batch)
    etp.check_output(
        out,
        batch,
        tokens,
        problem,
        layout.etp,
        exact=True,
        what=f"{layout.name} rank {env.rank} {backend} exact",
    )


@pytest.mark.parametrize("bi", [False, True], ids=["nobi", "bi"])
@pytest.mark.parametrize("backend", ["vllm", "torch"])
@pytest.mark.parametrize("env", ALL, indirect=True, scope="module")
def test_rank_token_permutation(env, backend, bi):
    """The same tokens placed differently across the tp_ep group's ranks.

    Placement B reverses the replicas' token counts and shuffles which token goes where (for a
    single sequence-parallel replica it moves tokens between the TP shards). Each placement must
    match the fp64 reference; in batch-invariant mode every token's output must also be bitwise
    identical in both placements and when placement A runs again.
    """
    layout = env.layout
    if bi:
        etp.require_batch_invariant_backend(backend)
    activation = "swiglu" if backend == "vllm" else "squared_relu"
    problem = etp.random_problem(E, H, 128 * layout.etp, 2, activation, seed=31)
    counts_a = etp.replica_counts("uneven", layout, M)
    tp = layout.tp
    counts_b = [(n, max(tp, -(-n // tp) * tp)) for n, _ in reversed(counts_a)]
    total = sum(n for n, _ in counts_a)
    tokens = etp.make_tokens(problem, total, seed=32, routing="random", ep=layout.ep)
    order_b = torch.randperm(total, device="cuda", generator=ou.make_generator(33, "cuda"))
    etp.allocate_nvls(env, M, 2, H)
    layer = etp.build_layer(
        etp.make_config(problem, layout, backend, batch_invariant=bi), problem, env
    )
    by_token = {}
    for label, counts, order in (
        ("A", counts_a, None),
        ("B", counts_b, order_b),
        ("A2", counts_a, None),
    ):
        batch = etp.rank_batch(tokens, counts, env, order)
        with etp.batch_invariant_mode(bi):
            out = etp.run_step(env, layer, problem, batch)
        what = f"{layout.name} rank {env.rank} {backend} placement {label}"
        etp.check_output(out, batch, tokens, problem, layout.etp, probs_at_activation=bi, what=what)
        by_token[label] = etp.gather_by_token(env, out, batch, total, M)
    if bi:
        ou.assert_exact(by_token["B"], by_token["A"], what=f"{layout.name}: placement B vs A")
        ou.assert_exact(by_token["A2"], by_token["A"], what=f"{layout.name}: repeated step")


def _graph_steps(layout: etp.Layout, padded: int):
    """Real tokens per replica in each replay: all real, uneven, an idle replica (0 real)."""
    reps = range(layout.replicas)
    return [
        [padded for _ in reps],
        [padded - 5 if j % 2 == 0 else 1 for j in reps],
        [0 if j == 0 else padded - 1 for j in reps] if layout.replicas > 1 else [3],
    ]


@pytest.mark.parametrize("bi", [False, True], ids=["nobi", "bi"])
@pytest.mark.parametrize("backend", ["vllm", "torch"])
@pytest.mark.parametrize("env", ALL, indirect=True, scope="module")
def test_cuda_graph_replay(env, backend, bi):
    """One captured step replayed with new tokens and a moving real-token count.

    Each rank captures a 32-row bucket; replays change the tokens and the count the context
    publishes (all real, uneven, an idle replica whose rows are all padding). Every replay must
    match the fp64 reference, with padding rows 0.
    """
    layout = env.layout
    if bi:
        etp.require_batch_invariant_backend(backend)
    rows = 32
    padded = rows * layout.tp
    problem = etp.random_problem(E, H, 128 * layout.etp, 2, "squared_relu", seed=41)
    etp.allocate_nvls(env, M, 2, H)
    layer = etp.build_layer(
        etp.make_config(problem, layout, backend, batch_invariant=bi), problem, env
    )
    static_x = torch.zeros(rows, H, dtype=torch.bfloat16, device="cuda")
    static_p = torch.full((rows, 2), 0.5, dtype=torch.float32, device="cuda")
    static_r = torch.zeros(rows, 2, dtype=torch.int64, device="cuda")
    static_r[:, 1] = 1
    count = etp.bind_real_count(padded)
    etp.bypass_router(layer, static_p, static_r)

    with etp.batch_invariant_mode(bi):
        etp.sync_all()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                etp.forward(layer, static_x)
        torch.cuda.current_stream().wait_stream(stream)
        etp.sync_all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = etp.forward(layer, static_x)
        etp.sync_all()

        stale_expert = etp.local_expert_range(problem, env)[0]
        for step, reals in enumerate(_graph_steps(layout, padded)):
            counts = [(n, padded) for n in reals]
            tokens = etp.make_tokens(problem, sum(reals), seed=50 + step, ep=layout.ep)
            batch = etp.rank_batch(tokens, counts, env)
            etp.poison_nvls(stale_expert)
            static_x.copy_(batch.x)
            static_p.copy_(batch.probs)
            static_r.copy_(batch.routing)
            count.fill_(batch.real)
            etp.sync_all()
            graph.replay()
            etp.check_nvls(
                rows * layout.tp_ep,
                stale_expert,
                flashinfer=False,
                what=f"{layout.name} replay {step}",
            )
            etp.check_output(
                static_out.clone(),
                batch,
                tokens,
                problem,
                layout.etp,
                probs_at_activation=bi,
                what=f"{layout.name} rank {env.rank} {backend} replay {step}",
            )
    del graph
    etp.sync_all()


@pytest.mark.parametrize("backend", ["vllm", "torch"])
@pytest.mark.parametrize("env", ALL, indirect=True, scope="module")
def test_router_through_the_layer(env, backend):
    """The whole layer forward, router included: the reference takes the router's own choice.

    The router's output is recorded on its way to the dispatcher, before the padding mask edits
    the routing in place.
    """
    layout = env.layout
    problem = etp.random_problem(E, H, 128 * layout.etp, 2, "squared_relu", seed=61)
    counts = etp.replica_counts("uneven", layout, M)
    tokens = etp.make_tokens(problem, sum(n for n, _ in counts), seed=62, ep=layout.ep)
    batch = etp.rank_batch(tokens, counts, env, padding_hidden=0.0)
    etp.allocate_nvls(env, M, 2, H)
    layer = etp.build_layer(etp.make_config(problem, layout, backend), problem, env)
    route, seen = layer.route, {}

    def spy(*args, **kwargs):
        probs, routing = route(*args, **kwargs)
        seen["probs"], seen["routing"] = probs.clone(), routing.clone()
        return probs, routing

    layer.route = spy
    out = etp.run_step(env, layer, problem, batch, route=True)
    assert seen["routing"].shape == (batch.x.shape[0], 2), seen["routing"].shape
    assert seen["probs"].dtype == torch.float32
    routed = etp.Tokens(
        batch.x[batch.is_real], seen["routing"][batch.is_real], seen["probs"][batch.is_real], E
    )
    real_rows = batch.is_real.nonzero().view(-1)
    check = etp.Batch(batch.x, batch.routing, batch.probs, torch.full_like(batch.ids, -1), 0)
    check.ids[real_rows] = torch.arange(real_rows.numel(), device="cuda")
    etp.check_output(
        out, check, routed, problem, layout.etp, what=f"{layout.name} rank {env.rank} router"
    )


@pytest.mark.parametrize("activation", ["swiglu", "squared_relu"])
@pytest.mark.parametrize("backend", ["vllm", "torch"])
@pytest.mark.parametrize("env", ALL, indirect=True, scope="module")
def test_context_sizes_buffers_for_the_layout(env, backend, activation):
    """``DynamicInferenceContext`` sizes the MoE buffers for the layout and masks its padding.

    - The NVLS buffers span tp_ep with ``round_up_tokens(max_tokens) / TP`` rows per rank.
    - The vLLM intermediates are sized for that gather and the FC1 width of one expert-TP shard:
      a step at the gather's capacity is served entirely from the pool (no allocation, so CUDA
      graphs keep stable addresses), and its largest request is the pool.
    - A prefill step through the context: its padded token count and published real count drive
      the layer, whose output must match the fp64 reference with padding rows 0.
    """
    from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
    from megatron.core.inference.config import InferenceConfig
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.inference.moe.vllm_fused_moe import VllmFusedMoeBuffers
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    layout = env.layout
    problem = etp.random_problem(E, H, 128 * layout.etp, 2, activation, seed=71)
    config = etp.make_config(problem, layout, backend)
    etp.release_buffers()
    env.nvls_dims = None
    max_tokens = 128
    context = DynamicInferenceContext(
        model_config=config,
        inference_config=InferenceConfig(
            max_sequence_length=256,
            buffer_size_gb=0.05,
            block_size_tokens=256,
            max_tokens=max_tokens,
            max_requests=16,
            unified_memory_level=0,
        ),
    )
    try:
        d = NVLSAllGatherVDispatcher
        per_rank = context.round_up_tokens(max_tokens) // layout.tp
        rows = per_rank * layout.tp_ep
        assert d._per_rank_worst_case_token_count == per_rank
        for key, tensor in etp._nvls_tensors().items():
            assert tensor.shape[0] == rows, f"{key}: {tuple(tensor.shape)}, expected {rows} rows"
            handle = SymmetricMemoryManager.get_buffer(key).symm_mem_hdl
            assert handle.world_size == layout.tp_ep, f"{key} spans {handle.world_size} ranks"
        assert d._real_token_count_tensor.data_ptr() == context.gpu_view.real_token_count.data_ptr()

        # Every replica prefills a sequence that pads to the per-rank capacity.
        full = per_rank * layout.tp
        reals = [full - 3 if j % 2 == 0 else full for j in range(layout.replicas)]
        real = reals[env.replica]
        context.add_dummy_requests_for_cudagraph_capture(
            InferenceBatchDimensions(token_count=real, prefill_req_count=1, decode_req_count=0)
        )
        context.initialize_attention_state()
        padded = context.padded_active_token_count
        assert padded == full, f"padded {padded}, expected {full}"
        assert int(context.gpu_view.real_token_count.item()) == real
        counts = [(n, full) for n in reals]
        tokens = etp.make_tokens(problem, sum(reals), seed=72, ep=layout.ep)
        batch = etp.rank_batch(tokens, counts, env)
        layer = etp.build_layer(config, problem, env)

        requests = []
        if backend == "vllm":
            pool = VllmFusedMoeBuffers._buffers
            assert pool is not None, "the context allocates the vLLM buffers"
            get = VllmFusedMoeBuffers.get.__func__

            def spy(cls, name, shape, dtype, device):
                out = get(cls, name, shape, dtype, device)
                served = out.untyped_storage().data_ptr() == pool[name].untyped_storage().data_ptr()
                requests.append((name, tuple(shape), served))
                return out

            VllmFusedMoeBuffers.get = classmethod(spy)
        try:
            out = etp.run_step(env, layer, problem, batch, bind=False)
        finally:
            if backend == "vllm":
                VllmFusedMoeBuffers.get = classmethod(get)
        etp.check_output(
            out, batch, tokens, problem, layout.etp, what=f"{layout.name} rank {env.rank} context"
        )
        if backend == "vllm":
            missed = [r for r in requests if not r[2]]
            assert requests and not missed, f"vLLM requests not served from the pool: {missed}"
            largest = max(
                (r for r in requests if r[0] == "intermediate1"), key=lambda r: r[1][0] * r[1][1]
            )
            width = problem.ffn // layout.etp * (2 if problem.glu else 1)
            assert largest[1] == (rows * problem.topk, width), largest
            assert pool["intermediate1"].numel() == rows * problem.topk * width
    finally:
        del context
        gc.collect()
        etp.release_buffers()
        env.nvls_dims = None


PRODUCTION = {
    # Nemotron 3 Nano experts: squared ReLU, 128 experts, top-6.
    "nemotron3-nano": dict(experts=128, hidden=2688, ffn=1856, topk=6, activation="squared_relu"),
    # Qwen3-30B-A3B experts: SwiGLU, 128 experts, top-8.
    "qwen3-30b-a3b": dict(experts=128, hidden=2048, ffn=768, topk=8, activation="swiglu"),
}


@pytest.mark.parametrize("backend", ["vllm", "torch"])
@pytest.mark.parametrize("model", list(PRODUCTION))
@pytest.mark.parametrize(
    "env", ["w2-tp1-etp2-ep1", "w4-tp1-etp2-ep2"], indirect=True, scope="module"
)
def test_production_shapes(env, model, backend):
    """Production expert shapes at ETP 2 (FFN shards of 928 and 384), uneven capacity counts."""
    layout = env.layout
    spec = PRODUCTION[model]
    problem = etp.random_problem(
        spec["experts"], spec["hidden"], spec["ffn"], spec["topk"], spec["activation"], seed=81
    )
    per_rank_max = 128
    reals = [[128, 37, 128, 1][j % 4] for j in range(layout.replicas)]
    counts = [(n, n) for n in reals]
    tokens = etp.make_tokens(problem, sum(reals), seed=82, ep=layout.ep)
    batch = etp.rank_batch(tokens, counts, env)
    etp.allocate_nvls(env, per_rank_max, problem.topk, problem.hidden)
    config = etp.make_config(problem, layout, backend, use_cpu_initialization=False)
    layer = etp.build_layer(config, problem, env)
    out = etp.run_step(env, layer, problem, batch)
    etp.check_output(
        out, batch, tokens, problem, layout.etp, what=f"{layout.name} rank {env.rank} {model}"
    )
    del layer
    etp.release_buffers()
    env.nvls_dims = None


@pytest.mark.sanitizer
@pytest.mark.parametrize("backend", ["vllm", "torch"])
@pytest.mark.parametrize("env", ["w2-tp1-etp2-ep1"], indirect=True, scope="module")
def test_small_case_for_the_sanitizer(env, backend):
    """A small ETP 2 step with uneven counts and padding, for compute-sanitizer memcheck."""
    layout = env.layout
    problem = etp.random_problem(E, H, 64 * layout.etp, 2, "swiglu", seed=91)
    counts = [(5, 8), (1, 1)]
    tokens = etp.make_tokens(problem, 6, seed=92, ep=layout.ep)
    batch = etp.rank_batch(tokens, counts, env)
    etp.allocate_nvls(env, 16, 2, H)
    layer = etp.build_layer(etp.make_config(problem, layout, backend), problem, env)
    out = etp.run_step(env, layer, problem, batch)
    etp.check_output(
        out, batch, tokens, problem, layout.etp, what=f"{layout.name} rank {env.rank} small"
    )
