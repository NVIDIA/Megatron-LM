# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing import load, save
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class _MixedPrecisionModel(MegatronModule):
    """Both parameter dtypes and weight-decay groups must survive state loading."""

    def __init__(self):
        super().__init__(
            TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1, bf16=True)
        )
        self.weight = torch.nn.Parameter(
            torch.full((128, 128), 0.25, device="cuda", dtype=torch.bfloat16)
        )
        # Cover each DP shard, including the standard eight-rank bucket padding.
        self.gate = torch.nn.Parameter(torch.full((1024,), 0.125, device="cuda"))
        for param in self.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    def forward(self):
        return sum(param.float().square().mean() for param in self.parameters())


@pytest.fixture(autouse=True)
def _model_parallel():
    pytest.importorskip("transformer_engine.pytorch.optimizers")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _make_model_and_optimizer(
    store_param_remainders=True, moment_dtype=torch.float32, master_dtype=torch.float32
):
    module = _MixedPrecisionModel()
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    model = DistributedDataParallel(
        module.config,
        DistributedDataParallelConfig(use_distributed_optimizer=True, grad_reduce_in_fp32=False),
        module,
        pg_collection=pg_collection,
    )
    config = OptimizerConfig(
        optimizer="adam",
        lr=0.001,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        store_param_remainders=store_param_remainders,
        main_params_dtype=master_dtype,
        exp_avg_dtype=moment_dtype,
        exp_avg_sq_dtype=moment_dtype,
        clip_grad=0.0,
    )
    optimizer = get_megatron_optimizer(
        config, [model], pg_collection=pg_collection, use_gloo_process_groups=False
    )
    if isinstance(optimizer, ChainedOptimizer):
        assert len(optimizer.chained_optimizers) == 1
        optimizer = optimizer.chained_optimizers[0]
    assert isinstance(optimizer, DistributedOptimizer)
    assert not optimizer.optimizer.state
    return model, optimizer


def _step(model, optimizer):
    optimizer.zero_grad()
    model.zero_grad_buffer()
    model().backward()
    model.finish_grad_sync()
    success, _, _ = optimizer.step()
    assert success


def _assert_same_state(first, second):
    assert len(first.param_groups) == len(second.param_groups)
    for first_group, second_group in zip(first.param_groups, second.param_groups):
        assert first_group["step"] == second_group["step"]
        assert len(first_group["params"]) == len(second_group["params"])
        for first_param, second_param in zip(first_group["params"], second_group["params"]):
            torch.testing.assert_close(first_param, second_param, rtol=0, atol=0)
            assert first.state[first_param].keys() == second.state[second_param].keys()
            for key, value in first.state[first_param].items():
                torch.testing.assert_close(value, second.state[second_param][key], rtol=0, atol=0)


@pytest.mark.parametrize("store_param_remainders", [False, True])
def test_preallocate_and_reload_reuse_native_state_buffers(store_param_remainders):
    _, optimizer = _make_model_and_optimizer(store_param_remainders)
    inner = optimizer.optimizer
    metadata = optimizer.state_dict()
    expected_groups = []
    for index, group in enumerate(metadata["optimizer"]["param_groups"]):
        group.update(step=7, lr=0.002 + index * 0.001)
        expected_groups.append((group["wd_mult"], group["lr"]))
    metadata["optimizer"]["param_groups"].reverse()

    with patch.object(inner, "load_state_dict", wraps=inner.load_state_dict) as inner_load:
        optimizer.load_state_dict(metadata)
    assert inner_load.call_count == 1
    assert inner_load.call_args.args[0]["state"] == {}
    assert [(group["wd_mult"], group["lr"]) for group in inner.param_groups] == expected_groups

    params = optimizer.get_parameters()
    assert {param.dtype for param in params} == {torch.float32, torch.bfloat16}
    for index, param in enumerate(params):
        state = inner.state[param]
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
        expected_master_dtype = (
            torch.int16
            if store_param_remainders and param.dtype == torch.bfloat16
            else torch.float32
        )
        assert state["master_param"].dtype == expected_master_dtype
        for value in state.values():
            value.fill_(index + 1)

    buffers = {param: dict(inner.state[param]) for param in params}
    expected_state = {
        param: {key: value.clone() for key, value in state.items()}
        for param, state in inner.state.items()
    }
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with patch.object(inner, "load_state_dict", wraps=inner.load_state_dict) as inner_load:
        optimizer.load_state_dict(optimizer.state_dict())
    torch.cuda.synchronize()
    assert inner_load.call_args.args[0]["state"] == {}
    assert torch.cuda.max_memory_allocated() == allocated_before
    for param, state in buffers.items():
        assert inner.param_groups[0]["step"] == 7
        for key, value in state.items():
            assert inner.state[param][key] is value
            torch.testing.assert_close(inner.state[param][key], expected_state[param][key])


@pytest.mark.parametrize(
    "master_dtype,moment_dtype", [(torch.float16, torch.float32), (torch.float32, torch.float16)]
)
def test_lower_precision_states_keep_te_loading_path(master_dtype, moment_dtype):
    model, optimizer = _make_model_and_optimizer(
        store_param_remainders=False, master_dtype=master_dtype, moment_dtype=moment_dtype
    )
    _step(model, optimizer)
    inner = optimizer.optimizer
    with patch.object(inner, "load_state_dict", wraps=inner.load_state_dict) as inner_load:
        optimizer.load_state_dict(optimizer.state_dict())
    assert inner_load.call_args.args[0]["state"]
    for state in inner.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == moment_dtype
        assert state["master_param"].dtype == master_dtype


@pytest.mark.parametrize("store_param_remainders", [False, True])
def test_precision_aware_dp_reshardable_resume_matches_uninterrupted(
    tmp_path_dist_ckpt, store_param_remainders
):
    model, optimizer = _make_model_and_optimizer(store_param_remainders)
    for _ in range(2):
        _step(model, optimizer)
    metadata = {"distrib_optim_sharding_type": "dp_reshardable", "dp_cp_group": model.dp_cp_group}

    with TempNamedDir(tmp_path_dist_ckpt / "precision_aware_resume", sync=True) as checkpoint_dir:
        model_state = model.module.sharded_state_dict(metadata=metadata)
        save(
            {
                "model": model_state,
                "optimizer": optimizer.sharded_state_dict(model_state, metadata=metadata),
            },
            checkpoint_dir,
        )
        resumed_model, resumed_optimizer = _make_model_and_optimizer(store_param_remainders)
        inner = resumed_optimizer.optimizer
        with patch.object(inner, "load_state_dict", wraps=inner.load_state_dict) as inner_load:
            resumed_model_state = resumed_model.module.sharded_state_dict(metadata=metadata)
            state_dict = load(
                {
                    "model": resumed_model_state,
                    "optimizer": resumed_optimizer.sharded_state_dict(
                        resumed_model_state, metadata=metadata, is_loading=True
                    ),
                },
                checkpoint_dir,
            )
            resumed_model.module.load_state_dict(state_dict["model"])
            resumed_optimizer.load_state_dict(state_dict["optimizer"])
        assert inner_load.call_count == 2
        assert all(call.args[0]["state"] == {} for call in inner_load.call_args_list)
        _assert_same_state(optimizer.optimizer, resumed_optimizer.optimizer)

        for _ in range(2):
            _step(model, optimizer)
            _step(resumed_model, resumed_optimizer)
            _assert_same_state(optimizer.optimizer, resumed_optimizer.optimizer)
            for param, resumed_param in zip(model.parameters(), resumed_model.parameters()):
                torch.testing.assert_close(param, resumed_param, rtol=0, atol=0)
