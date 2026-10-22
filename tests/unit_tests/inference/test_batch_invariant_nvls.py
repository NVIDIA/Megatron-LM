# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real EP4 NVLS invariance; each collective/backend needs a fresh interpreter."""

import os

import pytest
import torch

from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
from megatron.core.parallel_state import get_expert_model_parallel_group
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe import moe_utils
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
