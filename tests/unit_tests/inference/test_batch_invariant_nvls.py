# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real EP4 NVLS invariance; each collective/backend needs a fresh interpreter."""

import os

import pytest
import torch

from megatron.core.inference.moe import batch_invariant
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
from megatron.core.parallel_state import get_expert_model_parallel_group
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe import experts, moe_utils, token_dispatcher_inference
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from tests.unit_tests.inference.test_moe_dispatching_and_routing import _make_base_config
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def nvls_runtime():
    assert Utils.world_size == 4, "this manifest requires four real NVLink EP ranks"
    backend = os.environ.get("MCORE_BI_TEST_BACKEND", "te_native")
    collective = os.environ.get("MCORE_BI_TEST_COLLECTIVE", "ordered")
    if backend == "te_native":
        assert os.environ.get("CUBLASLT_WORKSPACE_SIZE") == "0"
    assert not bik.is_batch_invariant_mode_enabled(), "inherited a live BI backend"
    try:
        # Explicit global selection precedes model construction and the first GEMM.
        bik.enable_batch_invariant_mode(backend=backend, collective=collective)
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=4)
        assert bik.get_batch_invariant_backend() == backend
        assert bik.get_batch_invariant_collective() == collective
        yield backend, collective
    finally:
        try:
            if Utils.inited:
                torch.cuda.synchronize()
                torch.distributed.barrier()
        finally:
            NVLSAllGatherVDispatcher._delete_buffers()
            SymmetricMemoryManager.destroy()
            InferenceMode.unset_active()
            bik.disable_batch_invariant_mode()
            Utils.destroy_model_parallel()


def _build_nvls_layer(backend, collective, grouped_backend):
    """Build the shared native NVLS layer and target."""
    torch.manual_seed(321)
    model_parallel_cuda_manual_seed(321, inference_rng_tracker=True, force_reset_rng=True)
    config = _make_base_config(
        expert_model_parallel_size=4,
        inference_grouped_gemm_backend=grouped_backend,
        inference_moe_token_dispatcher_type="nvls",
        batch_invariant_mode=True,
        batch_invariant_backend=backend,
        batch_invariant_collective=collective,
        attention_backend=AttnBackend.flash,
        flash_attention_version=3,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        moe_shared_expert_intermediate_size=None,
    )
    NVLSAllGatherVDispatcher.allocate_buffers(
        per_rank_worst_case_token_count=128,
        topk=config.moe_router_topk,
        hidden_size=config.hidden_size,
        ep_group=get_expert_model_parallel_group(),
    )
    layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
    dispatcher = layer._inference_token_dispatcher
    assert isinstance(dispatcher, NVLSAllGatherVDispatcher)
    target = (
        torch.sin(torch.arange(8 * config.hidden_size, device="cuda") / 29)
        .reshape(8, 1, config.hidden_size)
        .bfloat16()
    )
    return config, layer, target


@torch.inference_mode()
def test_nvls_empty_router_width(nvls_runtime, monkeypatch):
    backend, collective = nvls_runtime
    config, layer, target = _build_nvls_layer(backend, collective, "vllm")
    original = moe_utils.te_general_gemm
    assert original is not None
    calls = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(tuple(args[1].shape))
        return result

    monkeypatch.setattr(moe_utils, "te_general_gemm", observe)
    rank = torch.distributed.get_rank()
    counts = (64, 128, 64) if rank == 0 else (0, 32 * rank, 0)
    reference = None
    for count in counts:
        inputs = target[:1].repeat(count, 1, 1)
        if rank == 0:
            inputs[:8] = target
        with InferenceMode.active():
            logits = layer.router.gating(inputs)
        assert logits.shape == (count, 1, config.num_moe_experts)
        assert logits.dtype == torch.float32 and logits.is_cuda
        if rank == 0:
            selected = logits[:8].clone()
            assert selected.abs().max() > 0
            if reference is None:
                reference = selected
            assert torch.equal(selected, reference)
    assert calls == [(count, config.hidden_size) for count in counts]


@pytest.mark.parametrize("grouped_backend", ["vllm", "torch"])
@pytest.mark.parametrize("peers", ["idle", "uneven"])
@torch.inference_mode()
def test_nvls_target_batch_invariance(nvls_runtime, monkeypatch, grouped_backend, peers):
    backend, collective = nvls_runtime
    rank = torch.distributed.get_rank()
    config, layer, target = _build_nvls_layer(backend, collective, grouped_backend)
    phase, evidence = {}, []
    original_gate = layer.router.gating

    def gate(inputs):
        result = original_gate(inputs)
        phase["physical"] = tuple(inputs.shape)
        if rank == 0:
            phase["gate"] = result[phase["target_row"] : phase["target_row"] + 8].clone()
        return result

    monkeypatch.setattr(layer.router, "gating", gate)
    original_router_gemm = moe_utils.te_general_gemm
    assert original_router_gemm is not None

    def router_gemm(*args, **kwargs):
        phase["router_gemms"] = phase.get("router_gemms", 0) + 1
        return original_router_gemm(*args, **kwargs)

    monkeypatch.setattr(moe_utils, "te_general_gemm", router_gemm)
    expert_name = "vllm_fused_moe" if grouped_backend == "vllm" else "mcore_fused_moe"
    original_experts = getattr(experts, expert_name)

    def expert_call(*args, **kwargs):
        valid = int(kwargs["valid_tokens"].item())
        assert valid == sum(phase["counts"])
        routing = kwargs["routing_map"][:valid]
        start = phase["target_row"]  # target is on EP rank zero, the first AGV segment
        phase["target_probs"] = args[1][start : start + 8].clone()
        phase["target_experts"] = routing[start : start + 8].clone()
        phase["loads"] = torch.bincount(routing.flatten(), minlength=config.num_moe_experts)
        assert kwargs["out"] is NVLSAllGatherVDispatcher._get_rsv_tensor()
        phase["expert_kernel"] = expert_name
        phase["experts"] = phase.get("experts", 0) + 1
        return original_experts(*args, **kwargs)

    monkeypatch.setattr(experts, expert_name, expert_call)
    original_agv = token_dispatcher_inference.multimem_all_gatherv_3tensor

    def all_gatherv(*args, **kwargs):
        phase["all_gathervs"] = phase.get("all_gathervs", 0) + 1
        return original_agv(*args, **kwargs)

    monkeypatch.setattr(token_dispatcher_inference, "multimem_all_gatherv_3tensor", all_gatherv)
    for owner, name in (
        (batch_invariant, "ordered_reduce_scatter_v"),
        (token_dispatcher_inference, "multimem_reduce_scatter_v"),
    ):
        original = getattr(owner, name)

        def reduce(*args, original=original, name=name, **kwargs):
            assert args[0].shape[0] == phase["counts"][rank]
            phase.setdefault("collectives", []).append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, reduce)

    def run(counts, target_row):
        phase.clear()
        phase.update(counts=counts, target_row=target_row)
        # Repeating a real target input deliberately changes the load on its
        # actual routed experts; no synthetic routing map replaces the router.
        inputs = target[:1].repeat(counts[rank], 1, 1)
        if rank == 0:
            inputs[target_row : target_row + 8] = target
        with InferenceMode.active():
            output, bias = layer(inputs)
        assert bias is None and output.shape == inputs.shape
        assert phase["physical"] == (counts[rank], 1, config.hidden_size)
        assert phase["router_gemms"] == phase["all_gathervs"] == 1
        assert phase["experts"] == 1
        assert phase["expert_kernel"] == expert_name
        metadata = NVLSAllGatherVDispatcher._step_metadata.tolist()
        assert metadata == [sum(counts), sum(counts[:rank]), max(counts)]
        phase["metadata"] = metadata
        expected_collective = (
            "ordered_reduce_scatter_v" if collective == "ordered" else "multimem_reduce_scatter_v"
        )
        assert phase["collectives"] == [expected_collective]
        # Independent oracle for the real target partials written by each rank's
        # experts into symmetric memory; NCCL here only gathers oracle inputs.
        partial = NVLSAllGatherVDispatcher._get_rsv_tensor()[target_row : target_row + 8].clone()
        partials = [torch.empty_like(partial) for _ in range(4)]
        torch.distributed.all_gather(partials, partial)
        phase["partial_sums"] = [contribution.double().sum().item() for contribution in partials]
        if collective == "multimem":
            expected = torch.stack(partials).double().sum(dim=0).float()
        else:
            expected = torch.zeros_like(partial)
            for contribution in partials:
                expected = expected + contribution
        if rank == 0:
            selected = output[target_row : target_row + 8, 0].clone()
            assert selected.abs().max() > 0
            assert torch.equal(selected, expected.bfloat16()), "actual NVLS combine changed target"
            phase["output"] = selected
            phase["output_bits"] = selected.view(torch.int16).long().sum().item()
        evidence.append(dict(phase))

    run([64, 0, 0, 0], 0)
    wide = [128, 0, 0, 0] if peers == "idle" else [128, 32, 64, 96]
    run(wide, 120)
    run(wide, 0)
    # Return to the smaller shape with idle peers: stale expert/symmetric-buffer
    # data from the preceding wide execution must not affect the target.
    run([64, 0, 0, 0], 0)
    for actual in evidence[1:]:
        assert torch.equal(actual["target_probs"], evidence[0]["target_probs"])
        assert torch.equal(actual["target_experts"], evidence[0]["target_experts"])
        if rank == 0:
            assert torch.equal(actual["gate"], evidence[0]["gate"])
            assert torch.equal(actual["output"], evidence[0]["output"])
    selected_experts = evidence[0]["target_experts"][0]
    assert torch.all(
        evidence[1]["loads"][selected_experts] > evidence[0]["loads"][selected_experts]
    )
    assert torch.equal(evidence[1]["loads"], evidence[2]["loads"])
    assert torch.equal(evidence[0]["loads"], evidence[3]["loads"])
    assert not torch.equal(evidence[1]["loads"].min(), evidence[1]["loads"].max())
    if rank == 0:
        assert {row["physical"][0] for row in evidence} == {64, 128}
    else:
        assert evidence[0]["physical"][0] == evidence[-1]["physical"][0] == 0
    print(
        "BI_NVLS_WITNESS",
        rank,
        backend,
        collective,
        grouped_backend,
        peers,
        "router_kernel",
        "te_general_gemm",
        "expert_kernel",
        expert_name,
        "dispatch_kernel",
        "multimem_all_gatherv_3tensor",
        "combine_kernels",
        [row["collectives"][0] for row in evidence],
        "physical",
        [row["physical"] for row in evidence],
        "metadata",
        [row["metadata"] for row in evidence],
        "target_rows",
        [row["target_row"] for row in evidence],
        "target_experts",
        [row["target_experts"].tolist() for row in evidence],
        "expert_loads",
        [row["loads"].tolist() for row in evidence],
        "partial_sums",
        [row["partial_sums"] for row in evidence],
        "output_bits",
        [row.get("output_bits") for row in evidence],
    )
