# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Non-GTP training regression for PR #7450 planner/replica alignment."""

import faulthandler
import os
import traceback
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.replica_expert_runtime import _ReplicaProjection
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe import test_scheduler_bf16_layer_parity as parity

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(scope="module")
def distributed_world():
    if not torch.cuda.is_available():
        pytest.skip("Replica training tests require CUDA.")
    Utils.set_world_size(world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"]))
    Utils.initialize_distributed()
    return dist.group.WORLD


@pytest.fixture(autouse=True)
def report_stalled_test():
    faulthandler.dump_traceback_later(90, repeat=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("backend", ["replica_peer_tma", "replica_nccl"])
def test_bf16_microbatch_accumulation_and_adam(distributed_world, monkeypatch, backend):
    assert dist.get_world_size() == 4
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    original = parity._config
    monkeypatch.setattr(
        parity,
        "_config",
        lambda backend, capacity=None: replace(
            original(backend, capacity),
            moe_scheduler_planner_type="moon_ep",
            moe_router_mock_maxvio=4,
        ),
    )
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    g = torch.Generator(device="cuda").manual_seed(413 + dist.get_rank())
    inputs = [
        torch.randn(256, 1, 256, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(4)
    ]
    upstream = [
        torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=g) * 0.01 for x in inputs
    ]
    bind = _ReplicaProjection.prepare_runtime_parameters
    seen = []

    def inspect(projection, dtype):
        bind(projection, dtype)
        seen.append(projection.accumulates_main_grad)
        assert projection.accumulates_main_grad
        assert all(
            a.data_ptr() == p.main_grad.data_ptr()
            for a, p in zip(projection.native_grad, projection.parameters)
        )
        for runtime in projection.runtime_parameters:
            assert not hasattr(runtime, "grad_added_to_main_grad")

    try:
        initial, reference, _ = parity._run(None, pg, None, inputs, upstream)
        monkeypatch.setattr(_ReplicaProjection, "prepare_runtime_parameters", inspect)
        _, actual, dispatch = parity._run(backend, pg, initial, inputs, upstream)
        assert seen and not any(d["overflow"] for d in dispatch)
        failures = []
        for name in reference:
            m = parity._metrics(reference[name], actual[name])
            limit = 2.5e-2 if name.startswith("master_delta/") else 1e-2
            if not m["finite"] or m["relative_l2"] > limit:
                failures.append((name, m))
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, failures)
        assert not any(reports), reports
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("routing", ["mock", "learned", "squared_relu"])
def test_precision_training_three_updates(distributed_world, monkeypatch, precision, routing):
    from megatron.core.activations import squared_relu
    from tests.unit_tests.transformer.moe import scheduler_precision_parity_utils as precision_utils
    from tests.unit_tests.transformer.moe.scheduler_precision_parity_utils import run_leg

    if routing == "squared_relu":
        original_config = precision_utils.config_for
        monkeypatch.setattr(
            precision_utils,
            "config_for",
            lambda *args, **kwargs: replace(
                original_config(*args, **kwargs),
                activation_func=squared_relu,
                gated_linear_unit=False,
                use_fused_weighted_squared_relu=True,
            ),
        )
        routing = "learned"
    assert dist.get_world_size() == 4
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    g = torch.Generator(device="cuda").manual_seed(837 + dist.get_rank())
    inputs = [
        [
            torch.randn(256, 1, 256, device="cuda", dtype=torch.bfloat16, generator=g)
            for _ in range(4)
        ]
        for _ in range(3)
    ]
    upstream = [
        [torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=g) * 0.01 for x in xs]
        for xs in inputs
    ]
    try:
        reference, _ = run_leg(precision, pg, inputs, upstream, routing=routing)
        actual, diagnostics = run_leg(
            precision,
            pg,
            inputs,
            upstream,
            ("moon_ep", "peer_tma", "shared", "reuse"),
            routing=routing,
        )
        failures = []
        assert reference.keys() == actual.keys()
        for name in reference:
            if name.startswith("initial"):
                if not torch.equal(reference[name], actual[name]):
                    failures.append((name, "initialization differs"))
                continue
            m = parity._metrics(reference[name], actual[name])
            limit = 0.025 if "/master_delta/" in name else 0.01
            if not m["finite"] or m["relative_l2"] > limit:
                failures.append((name, m))
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, failures)
        assert not any(reports), reports
        assert len(diagnostics) == 3
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_fsdp_direct_main_grad_two_updates(distributed_world, monkeypatch, precision):
    """Real sharded owner buckets, two microbatches/update and optimizer continuation."""
    from tests.unit_tests.transformer.moe import scheduler_precision_parity_utils as utility

    assert dist.get_world_size() == 4
    original = utility.config_for
    monkeypatch.setattr(
        utility,
        "config_for",
        lambda *args, **kwargs: replace(original(*args, **kwargs), expert_model_parallel_size=2),
    )
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    generator = torch.Generator(device="cuda").manual_seed(91 + dist.get_rank())
    inputs = [
        [
            torch.randn(128, 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
            for _ in range(2)
        ]
        for _ in range(2)
    ]
    upstream = [
        [torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=generator) * 0.01 for x in xs]
        for xs in inputs
    ]
    bind = _ReplicaProjection.prepare_runtime_parameters
    seen = []

    def inspect(projection, dtype):
        bind(projection, dtype)
        for grad, owner in zip(projection.native_grad, projection.parameters):
            assert grad.data_ptr() == owner.main_grad.data_ptr()
        seen.append(True)

    try:
        reference, _ = utility.run_leg(precision, pg, inputs, upstream, fsdp=True)
        monkeypatch.setattr(_ReplicaProjection, "prepare_runtime_parameters", inspect)
        actual, _ = utility.run_leg(
            precision, pg, inputs, upstream, ("echo", "nccl", "shared", "reuse"), fsdp=True
        )
        assert seen and reference.keys() == actual.keys()
        failures = []
        for name in reference:
            if name.startswith("initial"):
                assert torch.equal(reference[name], actual[name]), name
                continue
            metrics = parity._metrics(reference[name], actual[name])
            if not metrics["finite"] or metrics["relative_l2"] > (
                0.025 if "/master_delta/" in name else 0.01
            ):
                failures.append((name, metrics))
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, failures)
        assert not any(reports), reports
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("layout", [None, "contiguous"])
@pytest.mark.parametrize("mode", [None, "headwise"])
def test_dispatcher_config_preserves_defaulted_linear_cp_inputs(layout, mode):
    from types import SimpleNamespace

    from megatron.core.transformer.moe.moe_layer import MoELayer
    from tests.unit_tests.transformer.moe.test_echo_moe_scheduler import _scheduler_config

    config = _scheduler_config(
        experimental_attention_variant="gdn",
        linear_attention_freq=1,
        linear_cp_layout=layout,
        linear_cp_mode=mode,
    )
    before = dict(vars(config))
    layer = SimpleNamespace(config=config, num_physical_experts=8)
    physical = MoELayer._get_token_dispatcher_config(layer)
    assert physical.num_moe_experts == 8
    assert not physical.moe_enable_scheduler
    assert physical.linear_cp_mode == config.linear_cp_mode
    assert physical.linear_cp_layout == config.linear_cp_layout
    assert physical._linear_cp_mode_was_default == config._linear_cp_mode_was_default
    assert physical._linear_cp_layout_explicit == config._linear_cp_layout_explicit
    assert vars(config) == before
    # An explicitly incompatible user setting remains an error.
    with pytest.raises(ValueError, match="requires linear_cp_layout='contiguous'"):
        _scheduler_config(
            experimental_attention_variant="gdn",
            linear_cp_mode="chunkwise",
            linear_cp_layout="zigzag",
        )
