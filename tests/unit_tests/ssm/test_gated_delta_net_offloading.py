# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import replace
from typing import Any

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_gated_delta_net_module_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.layers.utils import create_layer_config
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import ChunkOffloadHandler
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA, GatedDeltaNet2
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.ssm.test_gdn_offload_training import _training_config
from tests.unit_tests.test_utilities import Utils


def _config(**overrides):
    defaults = dict(num_layers=3, linear_attention_freq=[1, 1, 1], linear_num_value_heads=8)
    return replace(_training_config(1, False, 1, 1, None, 1.0, False), **(defaults | overrides))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"experimental_attention_variant": None}, "only supports GDN"),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["gdn"]},
            "whole-GDN recomputation",
        ),
        (
            {
                "experimental_attention_variant": "gdn2",
                "recompute_granularity": "selective",
                "recompute_modules": ["gdn"],
            },
            "whole-GDN recomputation",
        ),
        ({"bf16": False}, "BF16"),
        ({"gdn_kernel_backend": "torch"}, "FLA"),
        ({"cuda_graph_impl": "local"}, "Transformer Engine CUDA graphs"),
        ({"fp8": "e4m3"}, "BF16"),
        ({"fp4": "e2m1"}, "BF16"),
    ],
)
def test_gdn_offload_config_rejection(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _config(**kwargs)


@pytest.fixture
def gdn_groups():
    Utils.initialize_model_parallel()
    off_interface.reset_instance()
    model_parallel_cuda_manual_seed(31)
    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        torch.cuda.synchronize()
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "symbol,model_enabled,layer_enabled",
    [
        pytest.param(None, True, True, id="pattern"),
        pytest.param("*", True, True, id="config-list"),
        pytest.param("G", True, False, id="layer-off"),
        pytest.param("G", False, True, id="model-off"),
    ],
)
def test_hybrid_offload_requires_gdn(gdn_groups, symbol, model_enabled, layer_enabled):
    enabled = _training_config(1, False, 1, 1, None, 1.0, False, native_hybrid=True)
    disabled = replace(enabled, fine_grained_activation_offloading=False, offload_modules=[])
    architecture = (
        {"hybrid_layer_pattern": "*" * enabled.num_layers}
        if symbol is None
        else {
            "hybrid_layer_config_list": [
                create_layer_config(enabled if layer_enabled else disabled, symbol)
            ]
            * enabled.num_layers
        }
    )
    with pytest.raises(ValueError, match="gdn_core_attn offloading requires"):
        HybridModel(
            config=enabled if model_enabled else disabled,
            vocab_size=512,
            max_sequence_length=128,
            pg_collection=gdn_groups,
            **architecture,
        )


def _model(config, groups):
    spec = get_gated_delta_net_module_spec(config)
    return (
        torch.nn.ModuleList(
            spec.module(config, spec.submodules, i + 1, pg_collection=groups) for i in range(3)
        )
        .cuda()
        .bfloat16()
    )


def _pair(config, groups):
    baseline = _model(
        replace(config, fine_grained_activation_offloading=False, offload_modules=[]), groups
    )
    offloaded = _model(config, groups)
    offloaded.load_state_dict(baseline.state_dict())
    return baseline, offloaded


def _run(model, source, packed=None, *, detached=False):
    model.zero_grad(set_to_none=True)
    output = x = source.detach().requires_grad_()
    for layer in model:
        y, _ = layer(output.detach() if detached else output, None, packed_seq_params=packed)
        output = output + y
    output.float().square().mean().backward()
    return output.detach(), x.grad, {name: p.grad for name, p in model.named_parameters()}


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.parametrize(
    "variant,fused,fraction,threshold",
    [
        pytest.param("gdn", False, 1.0, 1024, id="gdn-packed"),
        pytest.param("gdn2", False, 0.5, 1024, id="gdn2-packed-norm"),
        pytest.param("gdn", True, 0.5, 1024, id="gdn-packed-fused-norm"),
        pytest.param("gdn", False, 1.0, 10**9, id="gdn-below-threshold"),
    ],
)
def test_gdn_packed_offload(gdn_groups, variant, fused, fraction, threshold, monkeypatch):
    if variant == "gdn2":
        pytest.importorskip("fla.ops.gdn2")
    if fused:
        pytest.importorskip("causal_conv1d", minversion="1.6.1")
        monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    recompute_norm = fused or variant == "gdn2"
    config = _config(
        experimental_attention_variant=variant,
        gdn_pre_gated_delta_rule_fusion=fused,
        recompute_granularity="selective" if recompute_norm else None,
        recompute_modules=["gdn_norm_out"] if recompute_norm else [],
        activation_offload_fraction=fraction,
        min_offloaded_tensor_size=threshold,
    )
    baseline, offloaded = _pair(config, gdn_groups)
    cu = torch.tensor([0, 32, 128], device="cuda", dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    manager = PipelineOffloadManager.get_instance()
    for _ in range(3):
        source = torch.randn(128, 1, 256, device="cuda", dtype=torch.bfloat16)
        expected = _run(baseline, source, packed)
        off_interface.init_chunk_handler(0, None, None, threshold, 0, fraction)
        torch.testing.assert_close(_run(offloaded, source, packed), expected, rtol=0, atol=0)
        assert manager.cpu_tensor_pool.get_pool_status()["global_stats"]["current_in_use"] == 0
        off_interface.reset(process_group=torch.distributed.group.WORLD)
        assert (manager.offload_summary_total_bytes > 0) == (threshold == 1024)


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.parametrize(
    "variant,mode", [("gdn", "eval"), ("gdn", "no-grad"), ("gdn", "frozen"), ("gdn2", "eval")]
)
def test_gdn_resident_forward(gdn_groups, variant, mode):
    if variant == "gdn2":
        pytest.importorskip("fla.ops.gdn2")
    model = _model(_config(experimental_attention_variant=variant), gdn_groups)
    model.train(mode != "eval").requires_grad_(mode != "frozen")
    off_interface.init_chunk_handler(0, None, None, 1024, 0, 1.0)
    chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    output = torch.randn(64, 1, 256, device="cuda", dtype=torch.bfloat16)
    with torch.set_grad_enabled(mode != "no-grad"):
        for layer in model:
            output, _ = layer(output, None)
        if mode == "eval":
            output.float().square().mean().backward()
        else:
            assert not output.requires_grad
    assert chunk._offloaded_group_index == 0


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.parametrize(
    "value_heads,frozen_gates",
    [
        pytest.param(4, False, id="trainable-unexpanded"),
        pytest.param(8, False, id="trainable-expanded"),
        pytest.param(4, True, id="frozen"),
    ],
)
def test_gdn2_offload_retains_live_sigmoid_gates(
    gdn_groups: ProcessGroupCollection,
    value_heads: int,
    frozen_gates: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip copies of live sigmoid outputs while allowing expanded or frozen gates."""
    pytest.importorskip("fla.ops.gdn2")
    config = _config(experimental_attention_variant="gdn2", linear_num_value_heads=value_heads)
    baseline, offloaded = _pair(config, gdn_groups)
    if frozen_gates:
        for model in (baseline, offloaded):
            for layer in model:
                layer.in_proj.requires_grad_(False)
                layer.conv1d.requires_grad_(False)

    source = torch.randn(128, 1, 256, device="cuda", dtype=torch.bfloat16)
    reference = _run(baseline, source, detached=frozen_gates)
    gate_copies: list[tuple[dict[str, int], set[int]]] = []
    run_core = GatedDeltaNet2._run_gated_delta_rule
    offload_tensor = ChunkOffloadHandler.offload

    def record_gates(
        layer: GatedDeltaNet2,
        kernel_inputs: dict[str, torch.Tensor],
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        gate_copies.append(
            ({name: kernel_inputs[name].untyped_storage().data_ptr() for name in ("b", "w")}, set())
        )
        return run_core(layer, kernel_inputs, cu_seqlens)

    def record_copy(handler: ChunkOffloadHandler, tensor: torch.Tensor, **kwargs: Any) -> Any:
        _, copied = gate_copies[-1]
        copied.add(tensor.untyped_storage().data_ptr())
        return offload_tensor(handler, tensor, **kwargs)

    monkeypatch.setattr(GatedDeltaNet2, "_run_gated_delta_rule", record_gates)
    monkeypatch.setattr(ChunkOffloadHandler, "offload", record_copy)
    off_interface.init_chunk_handler(0, None, None, 1024, 0, 1.0)
    torch.testing.assert_close(
        _run(offloaded, source, detached=frozen_gates), reference, rtol=0, atol=0
    )
    assert len(gate_copies) == config.num_layers
    for gates, copied in gate_copies:
        assert (gates["w"] in copied) == frozen_gates
        assert (gates["b"] in copied) == (frozen_gates or value_heads != 4)
