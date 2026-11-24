# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import replace

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_gated_delta_net_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA
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
        ({"bf16": False}, "BF16"),
        ({"gdn_kernel_backend": "torch"}, "FLA"),
        ({"cuda_graph_impl": "local"}, "CUDA graphs"),
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
@pytest.mark.parametrize("variant,mode", [("gdn", "eval"), ("gdn", "no-grad"), ("gdn", "frozen")])
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
