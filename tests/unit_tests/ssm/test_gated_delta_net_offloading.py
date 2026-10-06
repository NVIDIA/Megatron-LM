# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Correctness and lifecycle coverage for BF16/FLA recurrence offloading."""

import copy
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_gated_delta_net_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import ChunkOffloadHandler
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    OffloadTensorGroup,
    PipelineOffloadManager,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _config(**kwargs: Any) -> TransformerConfig:
    defaults: dict[str, Any] = dict(
        num_layers=3,
        hidden_size=256,
        num_attention_heads=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        normalization="RMSNorm",
        activation_func=torch.nn.functional.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        experimental_attention_variant="gdn",
        linear_attention_freq=[1, 1, 1],
        fine_grained_activation_offloading=True,
        offload_modules=["gdn_core_attn"],
        min_offloaded_tensor_size=1024,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"experimental_attention_variant": None}, "only supports GDN"),
        ({"experimental_attention_variant": "gdn2"}, "only supports GDN"),
        ({"bf16": False, "params_dtype": torch.float32}, "BF16"),
        ({"fp8": "e4m3"}, "BF16"),
        ({"fp4": "e2m1"}, "BF16"),
        ({"deterministic_mode": True}, "FLA"),
        (
            {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            },
            "full recomputation",
        ),
        ({"cuda_graph_impl": "transformer_engine"}, "CUDA graphs"),
    ],
)
def test_reject_unsupported_gdn_offload(kwargs: dict[str, Any], match: str) -> None:
    """Unsupported paths fail at configuration time instead of silently doing nothing."""
    with pytest.raises(ValueError, match=match):
        _config(**kwargs)


def test_gdn_offload_config() -> None:
    """Canonical and legacy GDN names accept the new opt-in module."""
    for variant in ("gdn", "gated_delta_net"):
        config = _config(experimental_attention_variant=variant)
        assert config.experimental_attention_variant == "gdn"
    config = _config(recompute_granularity="selective", recompute_modules=["gdn_norm_out"])
    assert config.offload_modules == ["gdn_core_attn"]
    assert not TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=2
    ).offload_modules


@pytest.fixture
def gdn_offload_groups() -> Iterator[ProcessGroupCollection]:
    """Isolate the manager and initialize the real TP/CP groups."""
    Utils.initialize_model_parallel()
    off_interface.reset_instance()
    model_parallel_cuda_manual_seed(31)
    yield ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    torch.cuda.synchronize()
    off_interface.reset_instance()
    Utils.destroy_model_parallel()


def _build(config: TransformerConfig, groups: ProcessGroupCollection) -> torch.nn.ModuleList:
    spec = get_gated_delta_net_module_spec(config)
    return (
        torch.nn.ModuleList(
            [
                spec.module(config, spec.submodules, i + 1, pg_collection=groups)
                for i in range(config.num_layers)
            ]
        )
        .cuda()
        .bfloat16()
    )


def _run(
    model: torch.nn.ModuleList,
    source: torch.Tensor,
    packed: PackedSeqParams | None = None,
    offload: bool = False,
    fraction: float = 1.0,
    threshold: int = 1024,
    detached_inputs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    x = source.detach().clone().requires_grad_()
    if offload:
        off_interface.init_chunk_handler(0, None, None, threshold, 0, fraction)
    output = x
    for layer in model:
        layer_input = output.detach() if detached_inputs else output
        y, _ = layer(layer_input, None, packed_seq_params=packed)
        output = output + y
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    assert x.grad is not None
    return output.detach(), x.grad, grads


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.parametrize(
    "fraction,threshold", [(0.0, 1024), (0.5, 1024), (1.0, 1024), (1.0, 10**9)]
)
@pytest.mark.parametrize("recompute_norm", [False, True])
@pytest.mark.parametrize("value_heads", [4, 8], ids=["shared_qk", "expanded_qk"])
def test_gdn_offload_replay(
    gdn_offload_groups: ProcessGroupCollection,
    fraction: float,
    threshold: int,
    recompute_norm: bool,
    value_heads: int,
) -> None:
    """Outputs and every gradient agree through warmup, policy selection, and pool reuse."""
    config = _config(
        activation_offload_fraction=fraction,
        min_offloaded_tensor_size=threshold,
        linear_num_value_heads=value_heads,
    )
    if recompute_norm:
        config.recompute_granularity = "selective"
        config.recompute_modules = ["gdn_norm_out"]
    baseline_config = copy.deepcopy(config)
    baseline_config.fine_grained_activation_offloading = False
    baseline_config.offload_modules = []
    baseline = _build(baseline_config, gdn_offload_groups)
    offloaded = _build(config, gdn_offload_groups)
    offloaded.load_state_dict(baseline.state_dict())
    torch.manual_seed(19)
    manager = PipelineOffloadManager.get_instance()
    for iteration in range(3):
        # Different activations expose stale copies when cached groups/pools are reused.
        source = torch.randn(128, 2, 256, device="cuda", dtype=torch.bfloat16)
        reference = _run(baseline, source)
        result = _run(offloaded, source, offload=True, fraction=fraction, threshold=threshold)
        assert torch.equal(reference[0], result[0])
        assert torch.equal(reference[1], result[1])
        assert result[2].keys() == reference[2].keys()
        for name in reference[2]:
            assert torch.equal(reference[2][name], result[2][name]), name
        assert manager.cpu_tensor_pool.get_pool_status()["global_stats"]["current_in_use"] == 0
        off_interface.reset(process_group=torch.distributed.group.WORLD)
        if iteration == 0:
            groups = manager._cached_chunks_forward[0].offload_groups
            assert len(groups) == 3
            assert groups[-1].offload is False
            expected = (
                0 if threshold == 10**9 else (0 if fraction == 0 else 1 if fraction == 0.5 else 2)
            )
            assert sum(g.offload and g.total_offload_bytes > 0 for g in groups) == expected
            if threshold != 10**9 and fraction > 0:
                assert manager.offload_summary_total_bytes > 0
            else:
                assert manager.offload_summary_total_bytes == 0


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
def test_gdn_offload_packed_sequence(gdn_offload_groups: ProcessGroupCollection) -> None:
    """Packed-sequence metadata survives the saved-tensor hooks."""
    config = _config()
    baseline_config = copy.deepcopy(config)
    baseline_config.fine_grained_activation_offloading = False
    baseline_config.offload_modules = []
    baseline = _build(baseline_config, gdn_offload_groups)
    offloaded = _build(config, gdn_offload_groups)
    offloaded.load_state_dict(baseline.state_dict())
    cu = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    for _ in range(3):
        source = torch.randn(128, 1, 256, device="cuda", dtype=torch.bfloat16)
        reference = _run(baseline, source, packed)
        result = _run(offloaded, source, packed, offload=True)
        assert torch.equal(reference[0], result[0])
        assert torch.equal(reference[1], result[1])
        assert reference[2].keys() == result[2].keys()
        for name in reference[2]:
            assert torch.equal(reference[2][name], result[2][name]), name
        off_interface.reset(process_group=torch.distributed.group.WORLD)


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.parametrize("frozen_query", [False, True])
def test_gdn_offload_delayed_d2h(
    gdn_offload_groups: ProcessGroupCollection, monkeypatch: pytest.MonkeyPatch, frozen_query: bool
) -> None:
    """Warmup and missing query gradients still order fallback H2D after delayed D2H."""
    config = _config()
    baseline_config = copy.deepcopy(config)
    baseline_config.fine_grained_activation_offloading = False
    baseline_config.offload_modules = []
    baseline = _build(baseline_config, gdn_offload_groups)
    offloaded = _build(config, gdn_offload_groups)
    offloaded.load_state_dict(baseline.state_dict())
    if frozen_query:
        for model in (baseline, offloaded):
            for layer in model:
                layer.in_proj.requires_grad_(False)
                layer.conv1d.requires_grad_(False)

    bulk_offload = ChunkOffloadHandler.bulk_offload_group

    def delayed_offload(handler: ChunkOffloadHandler, group: OffloadTensorGroup) -> None:
        with torch.cuda.stream(handler.d2h_stream):
            torch.cuda._sleep(150_000_000)
        bulk_offload(handler, group)

    monkeypatch.setattr(ChunkOffloadHandler, "bulk_offload_group", delayed_offload)
    for _ in range(3):
        source = torch.randn(128, 2, 256, device="cuda", dtype=torch.bfloat16)
        # Detach recurrence inputs to exercise trainable gates without query gradients.
        reference = _run(baseline, source, detached_inputs=frozen_query)
        result = _run(offloaded, source, offload=True, detached_inputs=frozen_query)
        assert torch.equal(reference[0], result[0])
        assert torch.equal(reference[1], result[1])
        assert reference[2].keys() == result[2].keys()
        for name in reference[2]:
            assert torch.equal(reference[2][name], result[2][name]), name
        manager = PipelineOffloadManager.get_instance()
        assert manager.cpu_tensor_pool.get_pool_status()["global_stats"]["current_in_use"] == 0
        off_interface.reset(process_group=torch.distributed.group.WORLD)


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.parametrize("training,grad_enabled", [(False, True), (True, False), (False, False)])
def test_gdn_offload_no_grad_bypass(
    gdn_offload_groups: ProcessGroupCollection,
    monkeypatch: pytest.MonkeyPatch,
    training: bool,
    grad_enabled: bool,
) -> None:
    """Inference-style forwards do not initialize an offload manager or record groups."""
    model = _build(_config(), gdn_offload_groups)
    model.train(training)
    monkeypatch.setattr(
        PipelineOffloadManager,
        "get_instance",
        lambda: pytest.fail("Eval/no-grad forward should not access the offload manager."),
    )
    source = torch.randn(64, 1, 256, device="cuda", dtype=torch.bfloat16)
    with torch.set_grad_enabled(grad_enabled):
        for layer in model:
            source, _ = layer(source, None)


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
def test_gdn_offload_frozen_recurrence_bypass(
    gdn_offload_groups: ProcessGroupCollection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen recurrence bypasses offload even with global grad mode enabled."""
    model = _build(_config(), gdn_offload_groups).requires_grad_(False)
    monkeypatch.setattr(
        PipelineOffloadManager,
        "get_instance",
        lambda: pytest.fail("A non-differentiable recurrence should not access the manager."),
    )
    source = torch.randn(64, 1, 256, device="cuda", dtype=torch.bfloat16)
    with torch.enable_grad():
        for layer in model:
            source, _ = layer(source, None)
    assert not source.requires_grad
