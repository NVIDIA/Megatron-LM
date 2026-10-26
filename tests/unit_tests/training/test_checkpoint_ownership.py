# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint settings are owned by CheckpointConfig; the CLI and the containers agree on them."""

from argparse import Namespace
from dataclasses import asdict, replace
from unittest.mock import Mock, patch

import pytest

from megatron.training import global_vars
from megatron.training.argument_utils import (
    checkpoint_config_from_args,
    inference_cfg_container_from_args,
    project_checkpoint_policy,
)
from megatron.training.arguments import parse_args, validate_args
from megatron.training.config import (
    CheckpointConfig,
    InferenceConfigContainer,
    InferenceSetupConfig,
)

MODEL_ARGV = (
    "--num-layers 2 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--global-batch-size 8 --seq-length 32 --max-position-embeddings 32"
).split()

CLI_CASE = (
    "--save /ckpt/out --save-interval 10 --save-retain-interval 20 --load /ckpt/in "
    "--no-save-optim --no-load-rng --no-ckpt-fully-parallel-save --ckpt-fully-parallel-load "
    "--dist-ckpt-workers 4 --override-ckpt-iteration 7 --ckpt-drop-redundant-extra-state "
    "--async-save --use-persistent-ckpt-worker --dist-ckpt-strictness log_all"
).split()


def _parsed(*extra):
    with patch("sys.argv", ["test_checkpoint_ownership.py", *MODEL_ARGV, *extra]):
        return parse_args()


def _validated(*extra, defaults=None):
    return validate_args(_parsed(*extra), defaults or {})


def _inference_container(**checkpoint_kwargs):
    return InferenceConfigContainer(
        model=None,
        checkpoint=CheckpointConfig(**checkpoint_kwargs),
        inference=InferenceSetupConfig(),
    )


def test_cli_and_native_settings_match_without_aliasing():
    args = _validated(*CLI_CASE)
    config = checkpoint_config_from_args(args)
    native = CheckpointConfig(
        save="/ckpt/out",
        save_interval=10,
        save_retain_interval=20,
        load="/ckpt/in",
        save_optim=False,
        load_rng=False,
        fully_parallel_save=False,
        fully_parallel_load=True,
        dist_ckpt_workers=4,
        override_ckpt_iteration=7,
        ckpt_drop_redundant_extra_state=True,
        async_save=True,
        use_persistent_ckpt_worker=True,
        dist_ckpt_strictness="log_all",
    )
    assert asdict(config) == asdict(native)
    assert config.use_dist_ckpt is True
    # The section is a snapshot: later edits to args must not leak into it.
    args.save = "/elsewhere"
    args.no_save_optim = False
    assert config.save == "/ckpt/out" and config.save_optim is False


def test_the_deprecated_flags_are_inert_and_not_fields():
    args = _validated("--use-dist-ckpt", "--dist-ckpt-format", "x", "--ckpt-fully-parallel-save")
    config = checkpoint_config_from_args(args)
    assert config == CheckpointConfig()
    for name in ("use_dist_ckpt_deprecated", "dist_ckpt_format_deprecated"):
        assert name not in asdict(config)


@pytest.mark.parametrize("argv", [(), tuple(CLI_CASE)])
def test_projection_is_the_inverse_of_the_factory(argv):
    args = _validated(*argv)
    before = dict(vars(args))
    config = checkpoint_config_from_args(args)
    # A validated namespace projects its own section onto itself.
    assert vars(project_checkpoint_policy(args, config)) == before
    other = replace(
        config,
        save="/other",
        save_optim=False,
        load_rng=False,
        fully_parallel_save=False,
        fully_parallel_load=True,
        ckpt_format="torch",
        dist_ckpt_workers=9,
    )
    projected = project_checkpoint_policy(args, other)
    assert checkpoint_config_from_args(projected) == other
    assert projected.no_save_optim is True and projected.no_load_rng is True
    assert projected.ckpt_fully_parallel_save is False and projected.use_dist_ckpt is False
    # The live namespace is never written.
    assert dict(vars(args)) == before


def test_a_sparse_namespace_keeps_the_field_defaults():
    config = checkpoint_config_from_args(Namespace(save="/s", save_interval=3))
    assert config == CheckpointConfig(save="/s", save_interval=3)
    aliased = checkpoint_config_from_args(
        Namespace(save_optim=True, no_save_optim=True, ckpt_fully_parallel_save=False)
    )
    assert aliased.save_optim is False and aliased.fully_parallel_save is False


@pytest.mark.parametrize(
    "argv, match",
    [
        (["--ckpt-convert-format", "torch"], None),
        (["--ckpt-convert-format", "torch", "--ckpt-convert-save", "/c"], None),
        (["--save", "/s"], None),
        (["--save", "/s", "--save-interval", "0"], None),
        (["--save", "/s", "--save-interval", "10", "--save-retain-interval", "15"], None),
        (
            ["--non-persistent-ckpt-type", "local", "--non-persistent-save-interval", "5"],
            "without specifying --local-ckpt-dir",
        ),
        (["--load-main-params-from-ckpt"], "must be used with --no-load-optim"),
        (["--stream-ckpt-dequant", "--no-load-optim", "--finetune"], "inference-only load path"),
        (
            ["--stream-ckpt-dequant", "--no-load-optim", "--load-main-params-from-ckpt"],
            "cannot be used with --load-main-params-from-ckpt",
        ),
        (
            [
                "--stream-ckpt-dequant",
                "--no-load-optim",
                "--ckpt-fully-parallel-load",
                "--ckpt-fully-parallel-load-exchange-algo",
                "gather_rounds",
            ],
            "only with",
        ),
        (["--override-ckpt-iteration", "3", "--finetune"], "Cannot override checkpoint iteration"),
        (["--replication"], "requires the value of --replication-jump"),
        (["--replication", "--replication-jump", "2"], "non_persistent_ckpt_type == 'local'"),
        (
            [
                "--ckpt-pg-tensors-cache-path",
                "/cache",
                "--ckpt-fully-parallel-load-process-group",
                "ep_dp",
            ],
            "requires --ckpt-fully-parallel-save-process-group",
        ),
        (["--verify-integrity", "--ckpt-format", "torch"], "only supported with torch_dist"),
    ],
)
def test_validate_args_and_the_section_reject_the_same_inputs(argv, match):
    with pytest.raises(AssertionError, match=match):
        _validated(*argv)
    with pytest.raises(AssertionError, match=match):
        checkpoint_config_from_args(_parsed(*argv)).validate()


def test_validation_accepts_what_validate_args_accepted():
    for argv in ([], CLI_CASE, ["--replication-jump", "2"], ["--async-save"]):
        args = _validated(*argv)
        checkpoint_config_from_args(args).validate()
        inference_cfg_container_from_args(args, build_model_config=False).validate()


def test_inference_defaults_apply_before_the_checkpoint_checks():
    defaults = {"no_load_optim": True, "no_load_rng": True}
    args = _validated("--stream-ckpt-dequant", "--load", "/ckpt", defaults=defaults)
    assert args.no_load_optim is True
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    cfg.validate()
    assert cfg.checkpoint.load_optim is False and cfg.checkpoint.stream_ckpt_dequant is True
    with pytest.raises(AssertionError, match="requires --no-load-optim"):
        _validated("--stream-ckpt-dequant", "--load", "/ckpt")


@pytest.mark.parametrize("supported", [False, True])
def test_format_prerequisites_follow_the_effective_async_policy(monkeypatch, supported):
    from megatron.training import utils

    monkeypatch.setattr(utils, "has_nvrx_checkpointing_async_support", lambda: supported)
    # Without a persistent worker a distributed async save is disabled at finalize, so the
    # capability is not required; with one it is.
    CheckpointConfig(async_save=True, ckpt_format="torch_dcp").validate()
    config = CheckpointConfig(
        async_save=True, ckpt_format="torch_dcp", use_persistent_ckpt_worker=True
    )
    if supported:
        config.validate()
    else:
        with pytest.raises(AssertionError, match="nvidia-resiliency-ext"):
            config.validate()


def test_construction_runs_no_checks():
    assert not hasattr(CheckpointConfig, "__post_init__")
    config = CheckpointConfig(verify_integrity=True, ckpt_format="torch", async_strategy="other")
    with pytest.raises(AssertionError, match="async_strategy"):
        config.validate()


@pytest.mark.parametrize(
    "setup, match",
    [
        (
            dict(dist=dict(use_torch_fsdp2=True), checkpoint=dict(ckpt_format="torch")),
            "--use-torch-fsdp2 requires",
        ),
        (dict(checkpoint=dict(ckpt_format="torch_dcp")), "only tested with FSDP"),
        (dict(ddp=dict(use_megatron_fsdp=True)), "Megatron-FSDP requires"),
        (dict(checkpoint=dict(ckpt_format="fsdp_dtensor")), "only tested with Megatron FSDP"),
        (
            dict(
                dist=dict(use_gloo_process_groups=False),
                optimizer=dict(use_distributed_optimizer=True),
                checkpoint=dict(ckpt_format="torch"),
            ),
            None,
        ),
        (
            dict(
                ddp=dict(overlap_param_gather=True),
                checkpoint=dict(save="/s", save_interval=1, save_params_interval=1),
            ),
            None,
        ),
    ],
)
def test_training_container_checks_the_format_against_its_other_sections(run_config, setup, match):
    for section, values in setup.items():
        for name, value in values.items():
            setattr(getattr(run_config, section), name, value)
    with pytest.raises(AssertionError, match=match):
        run_config.validate()


def test_inference_container_checks_the_format_against_dist_only():
    cfg = _inference_container(ckpt_format="torch")
    cfg.dist.use_torch_fsdp2 = True
    with pytest.raises(AssertionError, match="--use-torch-fsdp2 requires"):
        cfg.validate()
    with pytest.raises(AssertionError, match="only tested with FSDP"):
        _inference_container(ckpt_format="torch_dcp").validate()
    # No ddp or optimizer section: those rules do not apply to inference.
    _inference_container(ckpt_format="fsdp_dtensor").validate()


@pytest.fixture
def warnings_log(monkeypatch):
    from megatron.training import utils

    log = []
    monkeypatch.setattr(utils, "warn_rank_0", lambda message, rank=None: log.append(message))
    return log


def _flippable():
    return dict(
        async_save=True,
        use_persistent_ckpt_worker=False,
        replication_jump=3,
        ckpt_convert_format="torch",
        ckpt_convert_save="/converted",
        load="/ckpt",
    )


@pytest.mark.parametrize("inference", [False, True])
def test_finalize_applies_the_legacy_flips_once(monkeypatch, run_config, warnings_log, inference):
    if inference:
        cfg = _inference_container(**_flippable())
    else:
        cfg = run_config
        cfg.checkpoint = CheckpointConfig(**_flippable())
        cfg.validation.skip_train = True
        cfg.train.micro_batch_size = 1
        monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", Namespace(batch_parallel_size=1))
    cfg.validate()
    assert warnings_log == []
    cfg.finalize()
    checkpoint = cfg.checkpoint
    assert checkpoint.exit_on_missing_checkpoint is True
    assert checkpoint.async_save is False
    assert checkpoint.replication_jump is None
    assert checkpoint.load_optim is inference
    assert len(warnings_log) == (2 if inference else 3)
    # Already derived: nothing changes and nothing is repeated.
    checkpoint.finalize_checkpoint_config(skip_train=not inference)
    assert len(warnings_log) == (2 if inference else 3)


def test_the_flips_keep_their_legacy_conditions(warnings_log):
    legacy_torch = CheckpointConfig(async_save=True, ckpt_format="torch")
    legacy_torch.finalize_checkpoint_config()
    assert legacy_torch.async_save is True
    zero_jump = CheckpointConfig(replication_jump=0)
    zero_jump.finalize_checkpoint_config()
    assert zero_jump.replication_jump == 0
    rl = CheckpointConfig()
    rl.finalize_checkpoint_config(skip_train=True, perform_rl_step=True)
    assert rl.load_optim is True
    assert warnings_log == []


def test_set_global_variables_finalizes_an_inference_container(monkeypatch):
    args = _validated()
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    cfg.checkpoint.replication_jump = 5
    for name in ("_GLOBAL_ARGS", "_GLOBAL_RUN_CONFIG"):
        monkeypatch.setattr(global_vars, name, None)
    services = Mock()
    monkeypatch.setattr(global_vars, "initialize_runtime_services", services)
    monkeypatch.setattr(global_vars, "_set_train_state", Mock())
    with patch("megatron.training.utils.warn_rank_0"):
        global_vars.set_global_variables(args, cfg)
    assert cfg._finalized
    assert cfg.checkpoint.replication_jump is None
    services.assert_called_once_with(args, build_tokenizer=True, training=False)
