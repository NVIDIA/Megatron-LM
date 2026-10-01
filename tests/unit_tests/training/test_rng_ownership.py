# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Behavioral coverage for RNG policy after legacy argument construction."""

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.training import arguments, checkpointing, global_vars, initialize
from megatron.training.config.common_config import RNGConfig


def _args_without_rng():
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.world_size = 1
    for item in fields(RNGConfig):
        delattr(args, item.name)
    return args


@pytest.mark.parametrize("enabled", [False, True])
def test_tracker_cli_flags_remain_outside_rng_config(enabled):
    from megatron.training.argument_utils import _default_config_from_args

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(["--te-rng-tracker", "--inference-rng-tracker"] if enabled else [])
    assert args.te_rng_tracker is enabled
    assert args.inference_rng_tracker is enabled
    config = _default_config_from_args(RNGConfig, args)
    assert asdict(config) == {"seed": 1234, "data_parallel_random_init": False}


@pytest.mark.parametrize("deleted", [False, True])
def test_wandb_metadata_uses_owner_without_modifying_live_args(
    monkeypatch, tmp_path, deleted, run_config
):
    args = Namespace(
        seed=1,
        te_rng_tracker=False,
        inference_rng_tracker=False,
        data_parallel_random_init=False,
        iteration=17,
        wandb_project="project",
        wandb_exp_name="run",
        wandb_save_dir=str(tmp_path),
        wandb_entity=None,
        rank=0,
        world_size=1,
    )
    rng = RNGConfig(seed=987, data_parallel_random_init=True)
    run_config.rng = rng
    if deleted:
        for item in fields(RNGConfig):
            delattr(args, item.name)
    original = vars(args).copy()
    import sys

    wandb = SimpleNamespace(init=Mock())
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setattr(global_vars, "_GLOBAL_WANDB_WRITER", None)
    global_vars._set_wandb_writer(args)
    metadata = wandb.init.call_args.kwargs["config"]
    assert metadata["iteration"] == 17
    assert metadata["te_rng_tracker"] is False
    assert metadata["inference_rng_tracker"] is False
    for name, value in asdict(rng).items():
        assert metadata[name] == value
    assert vars(args) == original


@pytest.mark.parametrize("te_tracker", [False, True])
@pytest.mark.parametrize("inference_tracker", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("skip", [False, True])
def test_initialization_uses_owner_and_preserves_deferred_seeding(
    monkeypatch, lazy, skip, te_tracker, inference_tracker, run_config
):
    args = _args_without_rng()
    args.lazy_mpu_init = lazy
    args.te_rng_tracker = te_tracker
    args.inference_rng_tracker = inference_tracker
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    for name in (
        "setup_logging",
        "initialize_rerun_state_machine",
        "_initialize_distributed",
        "_init_autoresume",
        "set_default_log_ranks",
        "print_rank_0",
    ):
        monkeypatch.setattr(initialize, name, Mock())
    monkeypatch.setattr(initialize.mpu, "set_tensor_model_parallel_world_size", Mock())
    monkeypatch.setattr(initialize.mpu, "set_tensor_model_parallel_rank", Mock())
    seed = Mock()
    monkeypatch.setattr(initialize, "_set_random_seed", seed)
    rng = RNGConfig(seed=919, data_parallel_random_init=True)
    run_config.rng = rng
    finish = initialize.initialize_megatron(
        allow_no_cuda=True, skip_random_seed=skip, skip_dependency_compilation=True
    )
    if lazy:
        seed.assert_not_called()
        finish()
    if skip:
        seed.assert_not_called()
    else:
        assert seed.call_args.args == (919, True, te_tracker, inference_tracker)
    assert not hasattr(args, "seed")


@pytest.mark.parametrize("dp_random", [False, True])
def test_parallel_seed_offsets_are_unchanged(monkeypatch, dp_random):
    pp, dp = object(), object()
    monkeypatch.setattr(initialize, "get_pg_rank", lambda group: 2 if group is pp else 3)
    python_seed, numpy_seed, torch_seed = Mock(), Mock(), Mock()
    monkeypatch.setattr(initialize.random, "seed", python_seed)
    monkeypatch.setattr(initialize.np.random, "seed", numpy_seed)
    monkeypatch.setattr(initialize.torch, "manual_seed", torch_seed)
    monkeypatch.setattr(initialize.torch.cuda, "device_count", lambda: 0)
    initialize._set_random_seed(101, dp_random, pp_group=pp, dp_group=dp)
    for seeded in (python_seed, numpy_seed, torch_seed):
        seeded.assert_called_once_with(301 + (30 if dp_random else 0))


@pytest.mark.parametrize("seed", [None, 0, -1])
def test_invalid_seed_rejected_by_container_validation(seed, run_config):
    rng = RNGConfig(seed=seed)
    run_config.rng = rng
    with pytest.raises(ValueError, match="positive integer"):
        run_config.validate()
    with pytest.raises(ValueError, match="positive integer"):
        initialize._set_random_seed(rng.seed)


@pytest.mark.parametrize("dp_random", [False, True])
def test_checkpoint_gather_uses_owner_without_global_args(monkeypatch, dp_random, run_config):
    run_config.rng.data_parallel_random_init = dp_random
    monkeypatch.setattr(checkpointing, "get_args", Mock(side_effect=AssertionError("args read")))
    monkeypatch.setattr(checkpointing.torch.cuda, "get_rng_state", lambda: "cuda")
    monkeypatch.setattr(
        checkpointing.tensor_parallel,
        "get_cuda_rng_tracker",
        lambda: SimpleNamespace(get_states=lambda: {"model": "state"}),
    )
    monkeypatch.setattr(checkpointing, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(checkpointing, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(checkpointing.torch.distributed, "is_initialized", lambda: True)
    group = object()

    def gather(output, state, *, group):
        output[:] = [state, {"peer": "state"}]

    collect = Mock(side_effect=gather)
    monkeypatch.setattr(checkpointing.torch.distributed, "all_gather_object", collect)
    state = checkpointing.get_rng_state("torch", group, group, dp_group=group, dp_cp_group=group)
    assert len(state) == (2 if dp_random else 1)
    assert collect.call_count == int(dp_random)
    assert state[0]["cuda_rng_state"] == "cuda"


@pytest.mark.parametrize("current_dp", [False, True])
@pytest.mark.parametrize("saved_dp", [False, True])
@pytest.mark.parametrize("saved_config_dp", [None, False, True])
def test_checkpoint_policy_retains_current_run_precedence(
    monkeypatch, current_dp, saved_dp, saved_config_dp, run_config
):
    args = _args_without_rng()
    args.use_dist_ckpt = False
    checkpoint_args = Namespace(**vars(args), seed=123, data_parallel_random_init=saved_dp)
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)
    rng = RNGConfig(seed=987, data_parallel_random_init=current_dp)
    run_config.rng = rng
    checkpoint_config = (
        None if saved_config_dp is None else {"rng": {"data_parallel_random_init": saved_config_dp}}
    )
    effective_saved_dp = saved_dp if saved_config_dp is None else saved_config_dp
    if current_dp and not effective_saved_dp:
        with pytest.raises(AssertionError, match="data_parallel_random_init"):
            checkpointing.check_checkpoint_args(
                checkpoint_args, checkpoint_config=checkpoint_config
            )
    else:
        checkpointing.check_checkpoint_args(checkpoint_args, checkpoint_config=checkpoint_config)
    assert rng.seed == 987
    assert rng.data_parallel_random_init == current_dp
    assert not hasattr(args, "seed")


def test_tensorboard_records_owned_rng_values(monkeypatch, run_config):
    args = Namespace(iteration=19, seed=1, te_rng_tracker=True)
    writer = Mock()
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "get_tensorboard_writer", lambda: writer)
    run_config.rng = RNGConfig(seed=321)
    initialize.write_args_to_tensorboard()
    writer.add_text.assert_any_call("seed", "321", global_step=19)
    writer.add_text.assert_any_call("te_rng_tracker", "True", global_step=19)
    assert vars(args) == {"iteration": 19, "seed": 1, "te_rng_tracker": True}


def test_cached_logits_identity_uses_dataset_seed_not_legacy_args(monkeypatch, run_config):
    from megatron.training.distillation import utils_logits

    args = Namespace(seed=999, seq_length=32, train_samples=16)
    monkeypatch.setattr(utils_logits, "get_args", lambda: args)
    monkeypatch.setattr(utils_logits, "_blend_identifiers", lambda args: {"mock": True})
    run_config.rng.seed = 123
    first_hash, first_fields = utils_logits.compute_dataset_hash()
    del args.seed
    assert utils_logits.compute_dataset_hash() == (first_hash, first_fields)
    assert first_fields["seed"] == 123
    run_config.rng.seed = 124
    second_hash, _ = utils_logits.compute_dataset_hash()
    assert first_hash != second_hash
