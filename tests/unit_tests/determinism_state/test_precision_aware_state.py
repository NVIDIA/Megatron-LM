# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contracts for raw optimizer storage; these fixtures are not TE GPU evidence."""

from types import SimpleNamespace

import pytest
import torch

from tools.determinism import training_state as states
from tools.determinism.megatron_state import capture_precision_aware_adam_state


class RawAdamFixture(torch.optim.Optimizer):
    """Model the pinned storage schema while forbidding checkpoint conversion."""

    def __init__(self, parameters):
        super().__init__(parameters, {"lr": 0.01, "betas": (0.9, 0.999), "eps": 1e-8})
        self.capturable = self.fuse_unscale = False
        self.master_weights = self.use_decoupled_grad = self.store_param_remainders = True
        self.master_weight_dtype = torch.float32
        self.exp_avg_dtype = self.exp_avg_sq_dtype = torch.float16
        self.name_to_dtype_map = {
            "exp_avg": torch.float16,
            "exp_avg_sq": torch.float16,
            "master_param": torch.float32,
        }
        self.dtype_to_range_map = {
            torch.float16: torch.tensor([32752.0]),
            torch.uint8: torch.tensor([448.0]),
        }
        self.adam_w_mode, self.set_grad_none = 1, None
        self._dummy_overflow_buf = torch.zeros(1, dtype=torch.int32)
        self._scales = {}
        for group in self.param_groups:
            group["step"] = 3
            for parameter in group["params"]:
                parameter.decoupled_grad = torch.ones_like(parameter, dtype=torch.float32)
                self.state[parameter] = {
                    "exp_avg": torch.full_like(parameter, 0.5, dtype=torch.float16),
                    "exp_avg_sq": torch.full_like(parameter, 0.25, dtype=torch.float16),
                    "master_param": torch.full_like(parameter, -1, dtype=torch.int16),
                }
                self._scales[parameter] = {
                    "exp_avg": torch.tensor([2.0]),
                    "exp_avg_sq": torch.tensor([4.0]),
                }

    def state_dict(self):
        pytest.fail("Checkpoint conversion cannot capture raw precision-aware state")


def fixture():
    model = torch.nn.Linear(4, 2, dtype=torch.bfloat16)
    model_parameters = list(model.parameters())
    shards = [p.detach().view(-1) for p in model_parameters]
    inner = RawAdamFixture(shards)
    config = SimpleNamespace(
        use_precision_aware_optimizer=True,
        use_distributed_optimizer=True,
        bf16=True,
        store_param_remainders=True,
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        exp_avg_dtype=torch.float16,
        exp_avg_sq_dtype=torch.float16,
        optimizer_cpu_offload=False,
    )
    wrapper = SimpleNamespace(
        optimizer=inner,
        config=config,
        model_param_group_index_map={p: (0, i) for i, p in enumerate(model_parameters)},
        _get_model_param_range_map=lambda p: {"param": SimpleNamespace(start=0, end=p.numel())},
        grad_scaler=None,
        get_loss_scale=lambda: torch.tensor([1.0]),
    )
    return model, wrapper, shards


def test_raw_moments_scales_remainders_and_shard_mapping_are_preserved():
    model, wrapper, shards = fixture()
    state, precision = capture_precision_aware_adam_state(wrapper, [model])
    assert (
        set(state["state"]["state"]) == set(state["scales"]) == set(state["model_shards"]) == {0, 1}
    )
    for identifier, parameter in enumerate(shards):
        for name in ("exp_avg", "exp_avg_sq", "master_param"):
            assert (
                state["state"]["state"][identifier][name]
                is wrapper.optimizer.state[parameter][name]
            )
        assert state["scales"][identifier] is wrapper.optimizer._scales[parameter]
        assert precision["parameter_shards"][identifier]["value"] is parameter
    assert state["model_shards"][0]["parameter"] == "weight"
    assert state["model_shards"][1]["parameter"] == "bias"
    assert state["model_shards"][0]["end"] == model.weight.numel()


@pytest.mark.parametrize(
    "change",
    [
        "missing_moment",
        "missing_remainder",
        "converted_remainder",
        "extra_state",
        "missing_scales",
        "missing_scale",
        "extra_scale",
        "wrong_scale_dtype",
        "wrong_moment_dtype",
        "missing_gradient",
        "missing_parameter",
        "extra_parameter",
        "missing_mapping",
        "wrong_mapping",
        "wrong_range",
        "copied_shard",
        "capturable",
        "offload",
        "wrong_dtype_policy",
        "wrong_config",
        "state_hook",
    ],
)
def test_unknown_or_incomplete_precision_state_is_unverified(change):
    model, wrapper, shards = fixture()
    inner, parameter = wrapper.optimizer, shards[0]
    if change == "missing_moment":
        del inner.state[parameter]["exp_avg_sq"]
    elif change == "missing_remainder":
        del inner.state[parameter]["master_param"]
    elif change == "converted_remainder":
        inner.state[parameter]["master_param"] = inner.state[parameter]["master_param"].float()
    elif change == "extra_state":
        inner.state[parameter]["unknown_buffer"] = torch.ones(1)
    elif change == "missing_scales":
        del inner._scales[parameter]
    elif change == "missing_scale":
        del inner._scales[parameter]["exp_avg_sq"]
    elif change == "extra_scale":
        inner._scales[parameter]["master_param"] = torch.ones(1)
    elif change == "wrong_scale_dtype":
        inner._scales[parameter]["exp_avg"] = torch.ones(1, dtype=torch.float16)
    elif change == "wrong_moment_dtype":
        inner.state[parameter]["exp_avg"] = inner.state[parameter]["exp_avg"].float()
    elif change == "missing_gradient":
        parameter.decoupled_grad = None
    elif change == "missing_parameter":
        del inner.state[parameter]
    elif change == "extra_parameter":
        inner.state[torch.ones(2)] = {}
    elif change == "missing_mapping":
        del wrapper.model_param_group_index_map[model.weight]
    elif change == "wrong_mapping":
        wrapper.model_param_group_index_map[model.weight] = (0, 1)
    elif change == "wrong_range":
        wrapper._get_model_param_range_map = lambda p: {
            "param": SimpleNamespace(start=1, end=p.numel())
        }
    elif change == "copied_shard":
        parameter.data = parameter.clone()
    elif change == "capturable":
        inner.capturable = True
    elif change == "offload":
        wrapper.config.optimizer_cpu_offload = True
    elif change == "wrong_dtype_policy":
        inner.name_to_dtype_map["exp_avg"] = torch.bfloat16
    elif change == "wrong_config":
        wrapper.config.store_param_remainders = False
    elif change == "state_hook":
        inner.register_state_dict_post_hook(lambda *_: None)
    with pytest.raises(states.UnverifiedState):
        capture_precision_aware_adam_state(wrapper, [model])


@pytest.mark.parametrize("component", ["exp_avg_sq", "master_param", "scale"])
def test_independent_raw_storage_perturbations_break_byte_comparison(
    tmp_path, monkeypatch, component
):
    model, wrapper, shards = fixture()
    provenance = {
        "source_revision": "cpu-fixture",
        "source_dirty": False,
        "recipe": {"name": "raw-storage-contract"},
        "hardware": {"device": "cpu"},
        "software": {"torch": str(torch.__version__)},
    }
    for index, name in enumerate(("reference", "changed")):
        if index:
            if component == "scale":
                wrapper.optimizer._scales[shards[0]]["exp_avg_sq"].view(torch.uint8)[0] ^= 1
            else:
                wrapper.optimizer.state[shards[0]][component].view(torch.uint8)[0] ^= 1
        optimizer, precision = capture_precision_aware_adam_state(wrapper, [model])
        snapshot = {
            "model": model.state_dict(),
            "gradients": {"weight": torch.ones_like(model.weight)},
            "optimizer": optimizer,
            "precision": precision,
            "rng": {"state": [1]},
            "scheduler": {"step": 3},
            "dataloader": {"cursor": 12},
        }
        monkeypatch.setattr(states, "PROCESS_ID", f"fixture-{index}")
        states.write_snapshot(
            tmp_path / name,
            snapshot,
            step=3,
            rank=0,
            world_size=1,
            run_id=name,
            provenance=provenance,
        )
        states.complete_rank(
            tmp_path / name, rank=0, world_size=1, run_id=name, steps=[3], provenance=provenance
        )
    result = states.compare_runs(
        tmp_path / "reference", tmp_path / "changed", steps=[3], world_size=1, comparison="fresh"
    )
    assert result["status"] == "different"
    path = result["first_difference"]["path"]
    assert path[:2] == ["optimizer", "scales" if component == "scale" else "state"]
