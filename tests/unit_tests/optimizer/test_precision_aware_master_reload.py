# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reload precision-aware master weights without restoring optimizer history."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _reload_optimizer(initialized=False, has_reload_api=True):
    """A shard-routing spy; TE's representation is tested in TE, not reproduced here."""
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    shard = param.detach()[1:3]
    inner = SimpleNamespace(param_groups=[{"params": [shard]}], state={})
    if initialized:
        inner.state[shard] = {"master_param": torch.zeros(2)}
    if has_reload_api:
        inner.update_param = Mock()
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.optimizer = inner
    optimizer.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=True)
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.model_param_group_index_map = {param: (0, 0)}
    optimizer.model_chunks = [SimpleNamespace(start_param_sync=Mock())]
    optimizer._get_model_param_range_map = lambda _: {"param": Range(1, 3)}
    optimizer._build_model_param_to_state_dict_param_map = lambda state: state
    return optimizer, param, shard


def test_reload_te_masters_routes_exact_checkpoint_shard():
    optimizer, param, shard = _reload_optimizer()
    source = torch.tensor([1.001, 1.00390625, -1.00390625, 2.0])
    saved = source.clone()
    optimizer._reload_te_main_params({param: source})
    optimizer.optimizer.update_param.assert_called_once()
    actual_param, actual_source = optimizer.optimizer.update_param.call_args.args
    assert actual_param is shard
    assert actual_source.dtype == torch.float32
    assert torch.equal(actual_source, source[1:3])
    assert torch.equal(source, saved)
    optimizer.model_chunks[0].start_param_sync.assert_called_once_with(force_sync=True)


def test_model_to_master_copy_delegates_to_te_reload():
    optimizer, param, _ = _reload_optimizer()
    optimizer._copy_model_params_to_main_params({param: torch.full((4,), 1.001)})
    optimizer.optimizer.update_param.assert_called_once()


def test_master_to_model_copy_does_not_reload_te_state():
    optimizer, _, _ = _reload_optimizer()
    optimizer.is_stub_optimizer = False
    optimizer._copy_main_params_to_model_params()
    optimizer.optimizer.update_param.assert_not_called()


def test_reload_te_masters_refreshes_initialized_state_from_model():
    optimizer, param, shard = _reload_optimizer(initialized=True)
    with torch.no_grad():
        param.fill_(2.0)
    optimizer._reload_te_main_params()
    actual_param, actual_source = optimizer.optimizer.update_param.call_args.args
    assert actual_param is shard
    assert torch.equal(actual_source, torch.full((2,), 2.0))


def test_reload_te_masters_keeps_fresh_state_lazy_with_older_te():
    optimizer, _, _ = _reload_optimizer(has_reload_api=False)
    optimizer._reload_te_main_params()
    assert not optimizer.optimizer.state
    # Fresh and initialized ranks must both participate in shard synchronization.
    optimizer.model_chunks[0].start_param_sync.assert_called_once_with(force_sync=True)


@pytest.mark.parametrize("initialized, checkpoint", [(False, True), (True, False)])
def test_reload_te_masters_requires_te_api_for_explicit_reload(initialized, checkpoint):
    optimizer, param, _ = _reload_optimizer(initialized, has_reload_api=False)
    source = {param: torch.full((4,), 1.001)} if checkpoint else None
    with pytest.raises(RuntimeError, match="FusedAdam.update_param"):
        optimizer._reload_te_main_params(source)
    optimizer.model_chunks[0].start_param_sync.assert_not_called()


def test_reload_te_masters_syncs_ranks_without_local_shards():
    optimizer, _, _ = _reload_optimizer()
    optimizer.model_param_group_index_map = {}
    optimizer._reload_te_main_params({})
    optimizer.optimizer.update_param.assert_not_called()
    optimizer.model_chunks[0].start_param_sync.assert_called_once_with(force_sync=True)


class _MixedPrecisionLinear(torch.nn.Linear):
    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight).float() + self.bias


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _model_and_optimizer(fp8_recipe, store_remainders):
    te = pytest.importorskip("transformer_engine.pytorch")
    if not hasattr(te.optimizers.FusedAdam, "update_param"):
        pytest.skip("Requires the companion TE FusedAdam.update_param change")
    module = _MixedPrecisionLinear(128, 128, dtype=torch.bfloat16, device="cuda")
    module.bias.data = module.bias.float()
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        module,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=True,
            fp8_recipe=fp8_recipe,
            store_param_remainders=store_remainders,
        ),
        [model],
    )
    return model, optimizer


def _step(model, optimizer):
    optimizer.zero_grad()
    model.zero_grad_buffer()
    model(torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")).square().mean().backward()
    model.finish_grad_sync()
    assert optimizer.step()[0]


@pytest.mark.parametrize("fp8_recipe", [None, "delayed"])
@pytest.mark.parametrize("store_remainders", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
def test_reload_fp32_checkpoint_preserves_precision_and_adam_state(
    model_parallel, fp8_recipe, store_remainders, initialized
):
    """Exercise existing PAO configurations independently of BF16/non-delayed FP8 support."""
    model, optimizer = _model_and_optimizer(fp8_recipe, store_remainders)
    if initialized:
        _step(model, optimizer)
    (distributed,) = optimizer.chained_optimizers
    adam = distributed.optimizer
    moments = {
        param: {name: state[name].clone() for name in ("exp_avg", "exp_avg_sq")}
        for param, state in adam.state.items()
        if state
    }
    steps = [group.get("step") for group in adam.param_groups]
    checkpoint = {
        name: torch.full(param.shape, 1.001, dtype=torch.float32, device="cuda")
        for name, param in model.module.named_parameters()
    }
    checkpoint["weight"].view(-1)[:4] = torch.tensor(
        [1.00390625, 1.01171875, -1.00390625, -1.01171875], device="cuda"
    )
    # Model loading followed by reload(state_dict) is the --no-load-optim
    # --load-main-params-from-ckpt sequence. No optimizer checkpoint is restored.
    model.module.load_state_dict(checkpoint)
    optimizer.reload_model_params({"model": checkpoint})
    assert [group.get("step") for group in adam.param_groups] == steps
    for param, state in adam.state.items():
        for name in ("exp_avg", "exp_avg_sq"):
            if initialized:
                assert torch.equal(state[name], moments[param][name])
            else:
                assert torch.count_nonzero(state[name]) == 0

    # With zero gradients and lr=0 the TE kernel reconstructs and re-encodes the master.
    # It must keep the same remainder/high-bit representation across this actual step.
    before = {param: state["master_param"].clone() for param, state in adam.state.items() if state}
    model_before = {name: param.detach().clone() for name, param in model.module.named_parameters()}
    for group in adam.param_groups:
        group["lr"] = 0.0
    optimizer.zero_grad()
    model.zero_grad_buffer()
    for param in model.module.parameters():
        param.main_grad.zero_()
    assert optimizer.step()[0]
    for param, value in before.items():
        assert torch.equal(adam.state[param]["master_param"], value)
    for name, param in model.module.named_parameters():
        assert torch.equal(param, model_before[name])

    # For full masters, check the supplied FP32 values directly. TE's remainder
    # encoding is covered independently by its API tests, including rounding ties.
    for name, param in model.module.named_parameters():
        if param not in distributed.model_param_group_index_map:
            continue
        group_index, group_order = distributed.model_param_group_index_map[param]
        inner = adam.param_groups[group_index]["params"][group_order]
        if adam.state[inner]["master_param"].dtype == torch.float32:
            span = distributed._get_model_param_range_map(param)["param"]
            assert torch.equal(
                adam.state[inner]["master_param"], checkpoint[name].view(-1)[span.start : span.end]
            )
    # A real nonzero update must also work when reload created the optimizer state.
    for group in adam.param_groups:
        group["lr"] = 0.01
    _step(model, optimizer)


@pytest.mark.parametrize("store_remainders", [False, True])
def test_reload_existing_masters_after_model_replacement(model_parallel, store_remainders):
    model, optimizer = _model_and_optimizer(None, store_remainders)
    _step(model, optimizer)
    with torch.no_grad():
        for param in model.module.parameters():
            param.fill_(2.0)
    optimizer.reload_model_params()
    (distributed,) = optimizer.chained_optimizers
    for param, state in distributed.optimizer.state.items():
        master = state["master_param"]
        expected = (
            torch.zeros_like(master)
            if master.dtype == torch.int16
            else torch.full_like(master, 2.0)
        )
        assert torch.equal(master, expected)
    _step(model, optimizer)
