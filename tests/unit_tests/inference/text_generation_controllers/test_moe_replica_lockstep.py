# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lockstep between the data-parallel replicas that share the inference MoE collectives.

The inference MoE dispatchers gather tokens over the expert tensor-and-model-parallel (tp_ep)
group. When it spans more than one data-parallel replica (ETP * EP > TP, e.g. ETP = 2 > TP = 1
with EP = 1), the replicas must issue the same MoE collectives every step, so an idle replica's
dummy step must mirror its peers' MTP forwards, and they must agree on per-step choices that
change those collectives, such as the bounded FlashInfer MoE rows.

The tests run on every pair of ranks (tp_ep groups of two) and agree on each verdict across all
ranks before asserting it, so a mismatch fails every rank instead of hanging a collective.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import zmq

from megatron.core.activations import squared_relu
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.async_zmq_communicator import AsyncZMQCommunicator
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.utils import get_moe_dp_replica_count
from megatron.core.parallel_state import get_expert_tensor_and_model_parallel_group
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.text_generation_controllers.test_mtp_kv_cache_controller import (
    VOCAB_SIZE,
    _build_step,
    _make_context,
    _make_controller,
    _make_model,
)
from tests.unit_tests.test_utilities import Utils

HIDDEN = 128
TOPK = 2
NUM_EXPERTS = 4


@pytest.mark.parametrize(
    "tp, etp, ep, replicas",
    [
        (1, 1, 1, 1),
        (1, 1, 2, 2),
        # ETP > TP: the expert-TP peers are other data-parallel replicas.
        (1, 2, 1, 2),
        (1, 2, 2, 4),
        (2, 2, 2, 2),
        # tp_ep inside one attention TP group: a single replica.
        (2, 2, 1, 1),
        (2, 1, 2, 1),
        (4, 1, 2, 1),
    ],
)
def test_moe_dp_replica_count(tp, etp, ep, replicas):
    config = SimpleNamespace(
        tensor_model_parallel_size=tp,
        expert_tensor_parallel_size=etp,
        expert_model_parallel_size=ep,
    )
    assert get_moe_dp_replica_count(config) == replicas


@pytest.mark.parametrize("tp, etp, ep", [(4, 1, 3), (2, 1, 3), (4, 3, 2), (4, 2, 3)])
def test_moe_dp_replica_count_rejects_tp_ep_groups_straddling_tp_groups(tp, etp, ep):
    """With TP 4 and EP 3 the tp_ep group [3, 4, 5] holds parts of the TP groups {0..3} and
    {4..7}, whose ranks would join different engine consensus groups."""
    config = SimpleNamespace(
        tensor_model_parallel_size=tp,
        expert_tensor_parallel_size=etp,
        expert_model_parallel_size=ep,
    )
    with pytest.raises(ValueError, match="must divide one another"):
        get_moe_dp_replica_count(config)


def _require_rank_pairs():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA GPUs")
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip(f"needs an even number of ranks; WORLD_SIZE={Utils.world_size}")


def _on_all_ranks(ok):
    """True only if `ok` holds on every rank, so all ranks pass or fail together."""
    flag = torch.tensor([int(bool(ok))], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


@contextlib.contextmanager
def _tp_ep_pairs(etp, ep):
    """Groups with TP = 1 and tp_ep groups of two ranks (two data-parallel replicas each)."""
    Utils.initialize_model_parallel(
        1, 1, expert_model_parallel_size=ep, expert_tensor_parallel_size=etp
    )
    try:
        tp_ep = get_expert_tensor_and_model_parallel_group()
        assert dist.get_world_size(tp_ep) == 2
        yield tp_ep
    finally:
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


# --------------------------------------------------------------------------------------------
# An idle replica's dummy step issues the MTP MoE collectives of its active peer.
# --------------------------------------------------------------------------------------------


def _mtp_config(etp, ep):
    return TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=128,
        moe_router_topk=TOPK,
        moe_router_dtype="fp32",
        mtp_num_layers=1,
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="inference_optimized",
        inference_grouped_gemm_backend="vllm",
        inference_moe_token_dispatcher_type="nvls",
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=etp,
        expert_model_parallel_size=ep,
    )


def _route_mtp_forwards(model, moe_step):
    """Run `moe_step(token_count)` for every MTP layer forward of the stub model, as the MTP
    layer's MoE sublayer would dispatch and combine its tokens over tp_ep."""
    draft_step, layer = model.compute_mtp_single_step, model.mtp.layers[0]
    commit_step = layer.forward_single_position

    def compute_mtp_single_step(**kwargs):
        moe_step(kwargs["hidden_states"].shape[0])
        return draft_step(**kwargs)

    def forward_single_position(**kwargs):
        moe_step(kwargs["hidden_states"].shape[0])
        return commit_step(**kwargs)

    model.compute_mtp_single_step = compute_mtp_single_step
    layer.forward_single_position = forward_single_position


def _active_replica_mtp_forwards(etp, ep, moe_step):
    """The active replica: the real `_compute_serial_mtp_and_sample` (commit pass and draft
    loop) on a stub model. Returns the MTP layer forwards it issued."""
    controller, model, _, _ = _build_step(num_decode_requests=2, accepted=(2, 1))
    _route_mtp_forwards(model, moe_step)
    controller._compute_serial_mtp_and_sample(base_position=torch.tensor([10, 12], device="cuda"))
    return model.all_forwards


def _idle_replica_mtp_forwards(etp, ep, moe_step):
    """The idle replica: the dummy MTP half of `controller.dummy_forward()`, with this layout.
    Returns the MTP layer forwards it issued."""
    context = _make_context(num_decode_requests=0)
    model = _make_model()

    def compute_mtp_single_step(**kwargs):
        model.all_forwards.append(("mtp_step", kwargs))
        hidden = kwargs["hidden_states"]
        return hidden, torch.zeros((hidden.shape[0], 1, VOCAB_SIZE), device="cuda")

    model.compute_mtp_single_step = compute_mtp_single_step
    controller = _make_controller(context, model)
    controller.model_config.expert_tensor_parallel_size = etp
    controller.model_config.expert_model_parallel_size = ep
    _route_mtp_forwards(model, moe_step)
    controller._run_dummy_serial_mtp_forward()
    return model.all_forwards


def _real_moe_step(config):
    """One real MoE dispatch and combine through the NVLS dispatcher over tp_ep."""
    pg_collection = get_default_pg_collection()
    ep_rank = dist.get_rank(pg_collection.ep)
    local = NUM_EXPERTS // dist.get_world_size(pg_collection.ep)
    dispatcher = NVLSAllGatherVDispatcher(
        num_local_experts=local,
        local_expert_indices=[ep_rank * local + i for i in range(local)],
        config=config,
        pg_collection=pg_collection,
        runs_metadata_sync=True,
    )

    def step(n):
        x = torch.ones(n, 1, HIDDEN, device="cuda", dtype=torch.bfloat16)
        routing = torch.arange(TOPK, device="cuda").repeat(n, 1)
        probs = torch.full((n, TOPK), 1.0 / TOPK, device="cuda")
        h, p = dispatcher.dispatch_preprocess(x, routing, probs)
        dispatcher.token_dispatch(h, p)
        out = dispatcher.token_combine(NVLSAllGatherVDispatcher._get_rsv_tensor())
        dispatcher.combine_postprocess(out)

    return step


@pytest.mark.parametrize("etp, ep", [(1, 2), (2, 1)], ids=["ep", "etp-gt-tp"])
def test_idle_replica_dummy_step_issues_its_peers_mtp_moe_collectives(etp, ep):
    """With ETP > TP and EP = 1 the expert-TP peers are separate replicas, so the idle one's
    dummy step must issue the MTP MoE collectives that its active peer's real MTP phase issues."""
    _require_rank_pairs()
    config = _mtp_config(etp, ep)
    with _tp_ep_pairs(etp, ep) as tp_ep:
        is_active = dist.get_rank(tp_ep) == 0
        mtp_forwards = _active_replica_mtp_forwards if is_active else _idle_replica_mtp_forwards

        # Count first: a mismatched NVLS collective would never complete.
        calls = len(mtp_forwards(etp, ep, moe_step=lambda n: None))
        counts = [None, None]
        dist.all_gather_object(counts, (is_active, calls), group=tp_ep)
        (active,) = [calls for peer_active, calls in counts if peer_active]
        (idle,) = [calls for peer_active, calls in counts if not peer_active]
        assert _on_all_ranks(idle == active), (
            f"TP=1, ETP={etp}, EP={ep}: the idle replica's dummy step issued {idle} MTP MoE "
            f"dispatches, its active tp_ep peer {active}"
        )

        # The same paths with real NVLS collectives over tp_ep then complete.
        NVLSAllGatherVDispatcher.allocate_buffers(
            per_rank_worst_case_token_count=16, topk=TOPK, hidden_size=HIDDEN, tp_ep_group=tp_ep
        )
        mtp_forwards(etp, ep, moe_step=_real_moe_step(config))
        torch.cuda.synchronize()
        dist.barrier()


# --------------------------------------------------------------------------------------------
# The replicas agree on the bounded FlashInfer MoE rows.
# --------------------------------------------------------------------------------------------


def _bounded_rows_context(communicator):
    """The context of an NVLS FlashInfer MoE model whose 128-row token capacity holds both
    replicas' 64 decode rows (TP = 1, ETP = 1, EP = 2)."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=128,
        moe_router_topk=TOPK,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        activation_func=squared_relu,
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_nvfp4_recipe="per_token",
        inference_moe_token_dispatcher_type="nvls",
        inference_flashinfer_mxfp8_token_capacity=128,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=2,
    )
    context = DynamicInferenceContext(
        model_config=config,
        inference_config=InferenceConfig(
            max_sequence_length=256,
            buffer_size_gb=0.05,
            block_size_tokens=128,
            max_tokens=256,
            max_requests=64,
            num_cuda_graphs=8,
            unified_memory_level=0,
        ),
    )
    context.set_ep_zmq_communicator(communicator)
    return context


@pytest.mark.parametrize(
    "phases, bounded",
    [(("decode", "decode"), True), (("decode", "prefill"), False), (("prefill", "prefill"), False)],
)
def test_replicas_agree_on_bounded_moe_rows(phases, bounded):
    """A decode-only replica bounds its MoE rows only when its tp_ep peer is decode-only too;
    otherwise the peer's prefill tokens sit past the capacity in the gathered rows. Either way
    it replays a decode-only CUDA graph, captured with or without the bound."""
    _require_rank_pairs()
    with _tp_ep_pairs(etp=1, ep=2) as tp_ep:
        zmq_context = zmq.Context()
        communicator = AsyncZMQCommunicator(zmq_context, process_group=tp_ep)
        try:
            context = _bounded_rows_context(communicator)
            phase = phases[dist.get_rank(tp_ep)]
            if phase == "decode":
                # An idle replica's dummy step is decode-only.
                context.agree_on_moe_decode_only(True)
                context.initialize_attention_state(is_expert_parallel_dummy_cuda_graph_step=True)
            else:
                context.add_request(
                    DynamicInferenceRequest(
                        request_id=1,
                        prompt_tokens=torch.arange(0, 96, device="cpu"),
                        sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=-1),
                    )
                )
                context.agree_on_moe_decode_only(context.num_prefill_requests == 0)
                context.initialize_attention_state()
            # A decode rank replays a graph whose key says whether it bounds the rows.
            graph = None
            if phase == "decode":
                graph = (context.using_cuda_graph_this_step(), context.cuda_graph_key[1])
            outcome = (phase, context.use_bounded_mxfp8_rows, graph)
            outcomes = [None, None]
            dist.all_gather_object(outcomes, outcome, group=tp_ep)
            expected = [
                (phase, bounded, (True, bounded)) if phase == "decode" else (phase, False, None)
                for phase in phases
            ]
            assert _on_all_ranks(outcomes == expected), f"{outcomes} != {expected}"
        finally:
            communicator.close()
            zmq_context.term()
