# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded FlashInfer MoE rows across the replicas that share the NVLS all-gather.

With ``inference_flashinfer_mxfp8_token_capacity`` set, the routed MXFP8 and per-token NVFP4
experts of a decode-only step process only the first ``capacity`` rows of the tokens gathered
over the expert tensor-and-model-parallel (tp_ep) group, and write only those rows of the shared
reduce-scatter buffer. That is correct only when every gathered token sits in those rows, i.e.
when every data-parallel replica in tp_ep runs a decode-only step and all of their decode tokens
fit. Here four ranks are four replicas (TP = 1) in two layouts, tp_ep = EP = 4 and tp_ep =
ETP 2 x EP 2. The bounded-row choice comes from the production path: each rank's
``DynamicInferenceContext`` agrees on the step's phase over a ZMQ communicator on tp_ep, and the
controller's forward applies it. The real NVLS dispatcher and FlashInfer kernels then run one MoE
step, and every token is checked against an fp64 per-token reference.

Inputs and weights are E2M1-grid values times a power of two with one +-6 * 2**-k per 16-block, so
MXFP8 and NVFP4 quantization of inputs and weights is exact and only the kernels' intermediate
requantization remains (relative L2 error <= 0.041 MXFP8, <= 0.105 NVFP4). A missing expert
contribution or a stale reduce-scatter row is far above the tolerances.

The GB200 unit-test job runs this file on four SM100 ranks. There, a missing FlashInfer kernel,
pyzmq or NVLS multicast is a failure, not a skip; other platforms skip it.
"""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import zmq

from megatron.core.activations import squared_relu
from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

WORLD_SIZE = 4
CAPACITY = 128
HIDDEN = 256
FFN = 256  # each expert-TP rank holds FFN // ETP rows of FC1 and columns of FC2
TOPK = 2
NUM_EXPERTS = 8
MAX_TOKENS = 256  # NVLS per-rank worst case
DECODE_TOKENS = 4
PREFILL_TOKENS = 160
SENTINEL = 1000.0  # what a previous step left in the reduce-scatter buffer
TOLERANCE = {"mxfp8": 0.15, "nvfp4": 0.35}
RELU2 = 6  # flashinfer ActivationType.Relu2
EXP_X, EXP_FC1, EXP_FC2 = 2, 5, 6
PG_TIMEOUT = timedelta(seconds=120)
WARMUP_TIMEOUT = timedelta(minutes=30)  # a cold FlashInfer JIT build takes minutes

# (ETP, EP, max_requests). Both layouts are four replicas; max_requests * EP = 128 in both, but
# only 4 x 32 decode rows fit the capacity, so the ETP 2 x EP 2 layout never bounds its rows.
LAYOUTS = {"ep4": (1, 4, 32), "etp2-ep2": (2, 2, 64)}

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

pytestmark = pytest.mark.launch_on_gb200


def _skip_reason():
    if not torch.cuda.is_available():
        return "needs CUDA GPUs"
    if Utils.world_size != WORLD_SIZE:
        return f"needs exactly {WORLD_SIZE} ranks; WORLD_SIZE={Utils.world_size}"
    if torch.cuda.get_device_capability()[0] != 10:
        return "routed MXFP8 and per-token NVFP4 MoE kernels need an SM100-family GPU"
    return None


def _require_kernels():
    """The hardware these kernels target must have them; the CI image pins FlashInfer 0.7."""
    from megatron.core.inference.moe.flashinfer_mxfp8 import (
        _FLASHINFER_ROUTED_MXFP8_IMPORT_ERROR,
        HAVE_FLASHINFER_ROUTED_MXFP8,
    )
    from megatron.core.inference.moe.flashinfer_nvfp4 import (
        _FLASHINFER_NVFP4_IMPORT_ERROR,
        HAVE_FLASHINFER_ROUTED_NVFP4,
    )

    assert (
        HAVE_FLASHINFER_ROUTED_MXFP8
    ), f"FlashInfer routed MXFP8 unavailable: {_FLASHINFER_ROUTED_MXFP8_IMPORT_ERROR!r}"
    assert (
        HAVE_FLASHINFER_ROUTED_NVFP4
    ), f"FlashInfer per-token NVFP4 MoE unavailable: {_FLASHINFER_NVFP4_IMPORT_ERROR!r}"


def _on_all_ranks(ok):
    """True only if `ok` holds on every rank, so all ranks pass or fail together."""
    flag = torch.tensor([int(bool(ok))], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _set_timeout(group, timeout):
    setter = getattr(dist.distributed_c10d, "set_timeout", None) or getattr(
        dist.distributed_c10d, "_set_pg_timeout"
    )
    setter(timeout, group)


def _grid(shape, exponent, gen):
    """E2M1-grid values times 2**-exponent with one +-6 * 2**-exponent in every 16-block."""
    numel = 1
    for size in shape:
        numel *= size
    vals = _E2M1[torch.randint(0, 8, (numel,), generator=gen)]
    sign = torch.randint(0, 2, (numel,), generator=gen).float() * 2 - 1
    v = (vals * sign).reshape(-1, 16)
    pos = torch.randint(0, 16, (v.shape[0],), generator=gen)
    pin = 6.0 * (torch.randint(0, 2, (v.shape[0],), generator=gen).float() * 2 - 1)
    v[torch.arange(v.shape[0]), pos] = pin
    return (v.reshape(shape) * 2.0**-exponent).to(torch.bfloat16)


def _config(backend, etp, ep):
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=FFN,
        moe_router_topk=TOPK,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=etp,
        expert_model_parallel_size=ep,
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_token_dispatcher_type="nvls",
        inference_flashinfer_mxfp8_token_capacity=CAPACITY,
        activation_func=squared_relu,
    )
    if backend == "mxfp8":
        kwargs.update(fp8="hybrid", fp8_recipe="mxfp8", fp8_param=True)
    else:
        kwargs.update(inference_moe_nvfp4_recipe="per_token")
    return TransformerConfig(**kwargs)


def _prepare_mxfp8(weight_stack, exponent):
    from megatron.core.inference.moe.flashinfer_mxfp8 import prepare_routed_mxfp8_weights
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    quantized = [MXFP8Tensor.from_bf16(w.contiguous(), backend="triton") for w in weight_stack]
    for q, w in zip(quantized, weight_stack):
        assert torch.equal(q.data.float(), w.float() * 2.0 ** (6 + exponent)), "MXFP8 inexact"
        assert torch.all(q.scale.view(torch.uint8) == 127 - 6 - exponent), "MXFP8 scale inexact"
    stacked = MXFP8Tensor(
        data=torch.stack([q.data for q in quantized]).contiguous(),
        scale=torch.stack([q.scale for q in quantized]).contiguous(),
        backend="triton",
        dtype=torch.bfloat16,
    )
    return prepare_routed_mxfp8_weights(stacked)


def _warm_up_kernels(env):
    """Build or load the FlashInfer JIT modules, with and without bounded rows."""
    from megatron.core.inference.moe.flashinfer_mxfp8 import flashinfer_routed_mxfp8_moe
    from megatron.core.inference.moe.flashinfer_nvfp4 import flashinfer_routed_nvfp4_moe

    gen = torch.Generator().manual_seed(7)
    tokens = 16
    x = _grid((tokens, HIDDEN), EXP_X, gen).cuda()
    first = env.ep_rank * env.local + torch.arange(tokens) % env.local
    second = env.ep_rank * env.local + (torch.arange(tokens) + 1) % env.local
    routing = torch.stack([first, second], dim=1).cuda()
    probs = torch.full((tokens, TOPK), 0.5, device="cuda")
    offset = env.ep_rank * env.local
    for bounds in ({}, dict(token_capacity=8, use_bounded_rows=True)):
        flashinfer_routed_mxfp8_moe(
            x,
            routing,
            probs,
            env.mx_fc1,
            env.mx_fc2,
            num_experts=NUM_EXPERTS,
            local_expert_offset=offset,
            activation_type=RELU2,
            **bounds,
        )
        flashinfer_routed_nvfp4_moe(
            x,
            routing,
            probs,
            env.nvfp4,
            num_experts=NUM_EXPERTS,
            local_expert_offset=offset,
            activation_type=RELU2,
            **bounds,
        )
    torch.cuda.synchronize()


@pytest.fixture(scope="module", params=list(LAYOUTS))
def env(request):
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)
    _require_kernels()

    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.inference.engines.async_zmq_communicator import AsyncZMQCommunicator
    from megatron.core.inference.moe.flashinfer_nvfp4 import prepare_routed_nvfp4_weights
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    from megatron.core.parallel_state import (
        get_expert_model_parallel_group,
        get_expert_tensor_and_model_parallel_group,
        get_expert_tensor_parallel_group,
    )
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    etp, ep, max_requests = LAYOUTS[request.param]
    Utils.initialize_model_parallel(
        1, 1, expert_model_parallel_size=ep, expert_tensor_parallel_size=etp
    )
    ep_group = get_expert_model_parallel_group()
    etp_group = get_expert_tensor_parallel_group()
    tp_ep_group = get_expert_tensor_and_model_parallel_group()
    warmup_group = dist.new_group(backend="gloo", timeout=WARMUP_TIMEOUT)
    zmq_context = zmq.Context()
    communicator = None
    try:
        local = NUM_EXPERTS // ep
        ep_rank, etp_rank = dist.get_rank(ep_group), dist.get_rank(etp_group)
        gen = torch.Generator().manual_seed(1234)  # the same global weights on every rank
        fc1 = _grid((NUM_EXPERTS, FFN, HIDDEN), EXP_FC1, gen).cuda()
        fc2 = _grid((NUM_EXPERTS, HIDDEN, FFN), EXP_FC2, gen).cuda()
        experts = slice(ep_rank * local, (ep_rank + 1) * local)
        shard = slice(etp_rank * (FFN // etp), (etp_rank + 1) * (FFN // etp))
        fc1_shard = fc1[experts, shard, :].contiguous()
        fc2_shard = fc2[experts, :, shard].contiguous()
        env = SimpleNamespace(
            layout=request.param,
            etp=etp,
            ep=ep,
            max_requests=max_requests,
            ep_rank=ep_rank,
            tp_ep_rank=dist.get_rank(tp_ep_group),
            ep_group=ep_group,
            tp_ep_group=tp_ep_group,
            local=local,
            fc1=fc1,
            fc2=fc2,
            nvfp4=None,
        )

        # Every rank prepares weights and builds the FlashInfer modules at once (FlashInfer
        # serializes builds with file locks); the ranks then meet on gloo with a long timeout.
        error = None
        try:
            env.mx_fc1 = _prepare_mxfp8(fc1_shard, EXP_FC1)
            env.mx_fc2 = _prepare_mxfp8(fc2_shard, EXP_FC2)
            env.nvfp4 = prepare_routed_nvfp4_weights(fc1_shard, fc2_shard)
            _warm_up_kernels(env)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [None] * Utils.world_size
        dist.all_gather_object(errors, (dist.get_rank(), error), group=warmup_group)
        failed = [(rank, e) for rank, e in errors if e is not None]
        assert not failed, f"weight preparation or kernel warm-up failed: {failed}"
        for group in (None, ep_group, etp_group, tp_ep_group):
            _set_timeout(group, PG_TIMEOUT)

        # The serving context sizes and allocates the NVLS buffers and decides the bounded rows.
        NVLSAllGatherVDispatcher._delete_buffers()
        communicator = AsyncZMQCommunicator(zmq_context, process_group=tp_ep_group)
        try:
            env.context = DynamicInferenceContext(
                model_config=_config("mxfp8", etp, ep),
                inference_config=InferenceConfig(
                    max_sequence_length=512,
                    buffer_size_gb=0.05,
                    block_size_tokens=256,
                    max_tokens=MAX_TOKENS,
                    max_requests=max_requests,
                    unified_memory_level=0,
                ),
            )
            handle = NVLSAllGatherVDispatcher._symm_rsv["handle"]
            error = None if bool(getattr(handle, "multicast_ptr", 0)) else "no multicast pointer"
        except RuntimeError as exc:
            error = str(exc)
        assert _on_all_ranks(error is None), f"NVLS symmetric memory unavailable: {error}"
        env.context.set_ep_zmq_communicator(communicator)
        NVLSAllGatherVDispatcher._symm_agv_hidden["tensor"].zero_()
        NVLSAllGatherVDispatcher._symm_agv_probs["tensor"].zero_()
        torch.cuda.synchronize()
        dist.barrier()
        yield env
    finally:
        torch.cuda.synchronize()
        if communicator is not None:
            communicator.close()
        zmq_context.term()
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        dist.destroy_process_group(warmup_group)
        Utils.destroy_model_parallel()


def _global_tokens(env, total, seed):
    """The same global batch on every rank: slot 0 routes to an expert of EP rank 0 and slot 1
    to another EP rank's, so every token needs the outputs of several tp_ep ranks."""
    gen = torch.Generator().manual_seed(seed)
    hidden = _grid((total, HIDDEN), EXP_X, gen)
    slot0 = torch.randint(0, env.local, (total,), generator=gen)
    slot1 = torch.randint(env.local, NUM_EXPERTS, (total,), generator=gen)
    routing = torch.stack([slot0, slot1], dim=1).to(torch.int64)
    p0 = torch.randint(10, 15, (total,), generator=gen).float() / 16
    probs = torch.stack([p0, 1 - p0], dim=1).to(torch.float32)
    return hidden, routing, probs


def _reference_moe(x, routing, probs, fc1, fc2):
    out = torch.zeros(x.shape[0], fc2.shape[1], dtype=torch.float64, device=x.device)
    xd = x.double()
    for slot in range(routing.shape[1]):
        for expert in routing[:, slot].unique().tolist():
            rows = routing[:, slot] == expert
            h = torch.relu(xd[rows] @ fc1[expert].double().t()).square()
            out[rows] += probs[rows, slot : slot + 1].double() * (h @ fc2[expert].double().t())
    return out


def _sync():
    torch.cuda.synchronize()
    dist.barrier()


def _run_step(env, backend, tokens_per_rank, decode_only, seed):
    """One serving step on every tp_ep rank; returns every rank's bounded flag and token errors.

    Each rank sets up a batch of its phase in the context, agrees on the phase with its peers and
    initializes the attention state as the controller does, then runs the controller's forward
    with one MoE layer (real NVLS dispatcher, real FlashInfer kernels) in place of the model.
    """
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    rank = env.tp_ep_rank
    n = tokens_per_rank[rank]
    hidden, routing, probs = _global_tokens(env, sum(tokens_per_rank), seed)
    start = sum(tokens_per_rank[:rank])  # gathered rows follow the tp_ep rank order
    x = hidden[start : start + n].cuda()
    r = routing[start : start + n].cuda()
    p = probs[start : start + n].cuda()

    context = env.context
    context.reset()
    context.add_dummy_requests_for_cudagraph_capture(
        InferenceBatchDimensions(
            token_count=n,
            prefill_req_count=0 if decode_only[rank] else 1,
            decode_req_count=n if decode_only[rank] else 0,
        )
    )
    context.agree_on_moe_decode_only(decode_only[rank])
    context.initialize_attention_state()

    config = _config(backend, env.etp, env.ep)
    dispatcher = NVLSAllGatherVDispatcher(
        num_local_experts=env.local,
        local_expert_indices=[env.ep_rank * env.local + i for i in range(env.local)],
        config=config,
        pg_collection=get_default_pg_collection(),
        runs_metadata_sync=True,
    )
    from flashinfer.fused_moe.core import ActivationType

    experts = SimpleNamespace(  # the production attributes InferenceGroupedMLP reads
        _activation_clamp_scale=None,
        _uses_mxfp8_weights=backend == "mxfp8",
        _nvfp4_weights=env.nvfp4 if backend == "nvfp4" else None,
        _fc1_weight=env.mx_fc1 if backend == "mxfp8" else None,
        _fc2_weight=env.mx_fc2 if backend == "mxfp8" else None,
        num_local_experts=env.local,
        ep_group=env.ep_group,
        _flashinfer_activation_type=ActivationType.Relu2,
        _nvls_dispatcher=True,
        _flashinfer_mxfp8_token_capacity=CAPACITY,
    )
    step = {}

    def moe_layer(inference_input):
        step["bounded"] = InferenceMode.use_bounded_mxfp8_rows()
        h, pr = dispatcher.dispatch_preprocess(x, r, p)
        h, pr = dispatcher.token_dispatch(h, pr)
        expert_out, _ = InferenceGroupedMLP._flashinfer_forward(
            experts, h, dispatcher.routing_map, pr
        )
        step["out"] = dispatcher.combine_postprocess(dispatcher.token_combine(expert_out))
        return torch.zeros(1, 1, 1, device="cuda")

    controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(
            inference_context=context, run_one_forward_step=moe_layer
        ),
        model_config=config,
        num_speculative_tokens=0,
        model_is_pipeline_parallel=False,
        _enable_cuda_graph=False,
        _select_async_sched_routing_buffer=lambda: None,
    )
    rsv = NVLSAllGatherVDispatcher._get_rsv_tensor()
    _sync()
    rsv.fill_(SENTINEL)
    _sync()
    with torch.no_grad(), InferenceMode.active():
        TextGenerationController._dynamic_step_forward_logits(controller, None, None)
    _sync()

    ref = _reference_moe(x, r, p, env.fc1, env.fc2)
    rel = (step["out"].double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)
    stats = dict(rank=rank, start=start, bounded=step["bounded"], rel=rel.cpu().tolist())
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, stats, group=env.tp_ep_group)
    return sorted(gathered, key=lambda s: s["rank"])


def _check(gathered, backend, expected_bounded, label):
    tokens = [(g["start"] + i, g["rank"], rel) for g in gathered for i, rel in enumerate(g["rel"])]
    bad = [t for t in tokens if not t[2] <= TOLERANCE[backend]]
    bounded = [g["bounded"] for g in gathered]
    assert bounded == [expected_bounded] * WORLD_SIZE, f"{label}: bounded rows {bounded}"
    assert not bad, (
        f"{label}: {len(bad)} of {len(tokens)} tokens differ from the reference (relative L2 up "
        f"to {max(t[2] for t in bad):.4g}, tolerance {TOLERANCE[backend]}) at gathered rows "
        f"{sorted(t[0] for t in bad)[0]}..{sorted(t[0] for t in bad)[-1]} on tp_ep ranks "
        f"{sorted({t[1] for t in bad})}"
    )


@pytest.mark.parametrize("backend", ["mxfp8", "nvfp4"])
def test_prefill_steps_match_reference(env, backend):
    """No replica is decode-only, so no rank bounds its rows; peers' rows run past the capacity."""
    tokens = [DECODE_TOKENS] + [PREFILL_TOKENS] * (WORLD_SIZE - 1)
    gathered = _run_step(env, backend, tokens, [False] * WORLD_SIZE, seed=11)
    _check(gathered, backend, False, f"[{env.layout}, {backend}] all prefill")


@pytest.mark.parametrize("backend", ["mxfp8", "nvfp4"])
def test_decode_steps_bound_rows_only_when_every_replica_fits(env, backend):
    """Every replica decodes max_requests tokens. The bounded rows are used when the four
    replicas' decode rows fit the capacity (EP 4: 4 x 32 = 128) and not otherwise (ETP 2 x
    EP 2: 4 x 64 = 256), and every gathered row gets every rank's expert outputs."""
    tokens = [env.max_requests] * WORLD_SIZE
    gathered = _run_step(env, backend, tokens, [True] * WORLD_SIZE, seed=12)
    fits = WORLD_SIZE * env.max_requests <= CAPACITY
    _check(gathered, backend, fits, f"[{env.layout}, {backend}] all decode")


@pytest.mark.parametrize("backend", ["mxfp8", "nvfp4"])
def test_decode_replica_does_not_bound_rows_while_a_peer_prefills(env, backend):
    """tp_ep rank 0 runs a decode-only step whose tokens fit while its peers prefill; their
    tokens sit past the capacity in the gathered rows, so rank 0 must not bound its rows."""
    tokens = [DECODE_TOKENS] + [PREFILL_TOKENS] * (WORLD_SIZE - 1)
    decode_only = [True] + [False] * (WORLD_SIZE - 1)
    gathered = _run_step(env, backend, tokens, decode_only, seed=13)
    _check(gathered, backend, False, f"[{env.layout}, {backend}] rank 0 decodes, peers prefill")
