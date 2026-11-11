# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimizer mappings must retain model identity across checkpoint dequantization."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import (
    ShardedTensor,
    ShardedTensorFactory,
    get_checkpoint_parameter_id,
    register_checkpoint_parameter,
)
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

pytestmark = pytest.mark.launch_on_gb200


def _model_entry(factory):
    param = torch.nn.Parameter(torch.arange(8, dtype=torch.float16).reshape(2, 4))
    payload = param.detach().to(torch.bfloat16)
    register_checkpoint_parameter(payload, param)
    shard = ShardedTensor.from_rank_offsets("weight", payload, replica_id=(0, 0, 0))
    if not factory:
        return param, shard

    def build(key, data, replica_id, flattened_range):
        assert flattened_range is None
        return {"rows": ShardedTensor.from_rank_offsets(key, data, replica_id=replica_id)}

    entry = ShardedTensorFactory("weight", payload, build, lambda parts: parts["rows"], (0, 0, 0))
    return param, entry


@pytest.mark.parametrize("factory", [False, True])
def test_optimizer_maps_original_parameter_and_keeps_moment_dtype(factory):
    param, entry = _model_entry(factory)
    other = torch.nn.Parameter(torch.ones_like(param))
    other_entry = ShardedTensor.from_rank_offsets("other", other)
    mapping = get_param_id_to_sharded_param_map({"w": entry, "other": other_entry}, [other, param])
    assert mapping == {0: other_entry, 1: entry}
    moment = torch.full_like(param, 123, dtype=torch.float32)
    optimizer_entry = make_sharded_optimizer_tensor(mapping[1], moment, "optimizer.exp_avg")
    if factory:
        optimizer_entry = optimizer_entry.build()["rows"]
    assert optimizer_entry.key == "optimizer.exp_avg.weight"
    assert optimizer_entry.data is moment
    assert get_checkpoint_parameter_id(optimizer_entry.data) == id(moment)
    assert optimizer_entry.dtype == torch.float32
    assert optimizer_entry.global_shape == param.shape


def _check_distributed_optimizer_mapping(param, entry, layout):
    master = torch.zeros(param.numel(), device=entry.data.device, dtype=torch.float32)
    moment = torch.full_like(master, 123)
    param_range = SimpleNamespace(start=0, end=param.numel())
    ranges = [{torch.float32: [{"param_map": {param: {"param": param_range}}}]}]
    buffer = SimpleNamespace(buckets=[None], param_index_map={param: (0, param.numel(), 0)})
    optimizer = SimpleNamespace(
        gbuf_ranges=ranges,
        buffers=[buffer],
        distributed_optimizer_instance_id=0,
        get_parameter_state_dp_zero=lambda **kwargs: {0: {torch.float32: {"exp_avg": moment}}},
        _get_main_param_and_optimizer_states=lambda _: {"param": master, "exp_avg": moment},
    )
    method = getattr(DistributedOptimizer, f"sharded_param_state_{layout}")
    state = method(optimizer, {"w": entry}, metadata={})
    saved_moment = state[0]["exp_avg"]
    assert saved_moment.key == "optimizer.state.exp_avg.weight"
    assert saved_moment.data.dtype == torch.float32
    assert torch.equal(saved_moment.data.flatten(), moment)
    assert saved_moment.replica_id == entry.replica_id


@pytest.fixture
def native_mxfp8_entry():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch import fp8_autocast, fp8_model_init, is_mxfp8_available

    from megatron.core import parallel_state
    from megatron.core.fp8_utils import is_float8tensor
    from megatron.core.tensor_parallel.gtp_api import HAVE_GTP, attach_gtp_to_presharded_module
    from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
    from tests.unit_tests.test_utilities import Utils

    available, reason = is_mxfp8_available(return_reason=True)
    if not HAVE_GTP or not available:
        pytest.skip(f"Native MXFP8 GTP unavailable: {reason}")
    Utils.initialize_model_parallel()
    try:
        gtp_group = torch.distributed.group.WORLD
        if gtp_group.size() < 2:
            pytest.skip("Requires at least two GTP ranks")
        tp_group = parallel_state.get_tensor_model_parallel_group()
        recipe = MXFP8BlockScaling()
        with fp8_model_init(enabled=True, recipe=recipe, preserve_high_precision_init_val=True):
            layer = te.Linear(128, 32, bias=False, params_dtype=torch.bfloat16, device="cuda")
        layer.gtp_remat_size = gtp_group.size()
        attach_gtp_to_presharded_module(layer, gtp_group, 0, replica_group=tp_group)
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            layer(torch.randn(32, 128, dtype=torch.bfloat16, device="cuda"))
        param = layer.weight
        entry = make_tp_sharded_tensor_for_checkpoint(
            param, "weight", tp_group=tp_group, dp_cp_group=tp_group
        )
        assert is_float8tensor(param)
        assert entry.data is not param and not is_float8tensor(entry.data)
        assert entry.data._gtp_dequant_src is param
        yield param, entry, tp_group, gtp_group
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("layout", ["fully_reshardable", "fs_model_space"])
def test_native_mxfp8_distributed_optimizer_mapping(native_mxfp8_entry, layout):
    from megatron.core.transformer.mlp import apply_swiglu_sharded_factory

    param, entry, _, _ = native_mxfp8_entry
    if layout == "fs_model_space":
        # Flattened optimizer ranges require factory-backed model metadata.
        entry = apply_swiglu_sharded_factory(entry, ())
    del entry.data._gtp_dequant_src
    _check_distributed_optimizer_mapping(param, entry, layout)


def test_native_mxfp8_checkpoint_roundtrip(native_mxfp8_entry, tmp_path_dist_ckpt):
    from megatron.core import dist_checkpointing
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    param, entry, _, _ = native_mxfp8_entry
    del entry.data._gtp_dequant_src
    assert get_param_id_to_sharded_param_map({"w": entry}, [param]) == {0: entry}
    moment = torch.full_like(entry.data, 123, dtype=torch.float32)
    optimizer_entry = make_sharded_optimizer_tensor(entry, moment, "optimizer.exp_avg")
    with TempNamedDir(tmp_path_dist_ckpt / "gtp_parameter_identity", sync=True) as path:
        dist_checkpointing.save({"weight": entry, "moment": optimizer_entry}, path)
        loaded = dist_checkpointing.load_plain_tensors(path)
    start = entry.global_offset[0]
    torch.testing.assert_close(
        loaded["weight"][start : start + param.shape[0]].cpu(), entry.data.cpu(), rtol=0, atol=0
    )
    assert loaded["optimizer.exp_avg.weight"].dtype == torch.float32
    assert torch.all(loaded["optimizer.exp_avg.weight"] == 123)
    assert not hasattr(loaded["weight"], "_checkpoint_parameter_id")


def test_native_mxfp8_muon_checkpoint_mapping(native_mxfp8_entry):
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.emerging_optimizers import (
        HAVE_EMERGING_OPTIMIZERS,
        TensorParallelMuon,
    )
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
    from megatron.core.process_groups_config import ProcessGroupCollection

    if not HAVE_EMERGING_OPTIMIZERS:
        pytest.skip("Muon requires emerging_optimizers")
    param, entry, tp_group, gtp_group = native_mxfp8_entry
    del entry.data._gtp_dequant_src
    groups = ProcessGroupCollection(tp=tp_group, expt_tp=tp_group, gtp_remat=gtp_group)
    raw_optimizer = TensorParallelMuon([param], pg_collection=groups)
    optimizer = Float16OptimizerWithFloat16Params(
        raw_optimizer, OptimizerConfig(bf16=True), None, None
    )
    master = raw_optimizer.param_groups[0]["params"][0]
    momentum = torch.full_like(master, 123)
    raw_optimizer.state[master]["momentum_buffer"] = momentum
    # Generic matching must succeed before Muon's GTP-specific fallback.
    with patch(
        "megatron.core.tensor_parallel.gtp_api.is_gtp_param",
        side_effect=AssertionError("Unexpected GTP backfill"),
    ):
        state = optimizer.sharded_state_dict({"w": entry})
    saved_momentum = state["optimizer"]["state"][0]["momentum_buffer"]
    assert saved_momentum.data is momentum
    assert saved_momentum.dtype == torch.float32
    assert saved_momentum.key == "optimizer.state.momentum_buffer.weight"


def test_checkpoint_parameter_registration_is_explicit_and_composable():
    param = torch.nn.Parameter(torch.ones(2, 4))
    payload = param.detach().to(torch.bfloat16)
    register_checkpoint_parameter(payload, param)
    converted = payload.float()
    register_checkpoint_parameter(converted, payload)
    assert get_checkpoint_parameter_id(converted) == id(param)
    # GDP uses a fresh view to keep its semantic factory out of optimizer matching.
    view = payload.view_as(payload)
    assert get_checkpoint_parameter_id(view) == id(view)
    assert get_checkpoint_parameter_id(view) != id(param)
    assert get_checkpoint_parameter_id(param) == id(param)


def test_checkpoint_parameter_registration_rejects_changed_shape():
    param = torch.nn.Parameter(torch.ones(2, 4))
    with pytest.raises(ValueError, match="same shape"):
        register_checkpoint_parameter(param.flatten(), param)
