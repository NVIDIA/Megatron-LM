# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint settings are owned by CheckpointConfig; the CLI and the containers agree on them."""

import ast
import inspect
from argparse import Namespace
from dataclasses import asdict, fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import megatron.training
from megatron.training import async_utils, global_vars, training
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
from megatron.training.utils import common_utils

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
    "name, alias",
    [
        ("save_optim", "no_save_optim"),
        ("save_rng", "no_save_rng"),
        ("load_optim", "no_load_optim"),
        ("load_rng", "no_load_rng"),
        ("fully_parallel_save", "ckpt_fully_parallel_save"),
        ("fully_parallel_load", "ckpt_fully_parallel_load"),
    ],
)
@pytest.mark.parametrize("spelling", ["canonical", "alias", "both", "none"])
@pytest.mark.parametrize("value", [True, False])
def test_projection_round_trips_every_spelling_of_the_aliased_fields(name, alias, spelling, value):
    """The persisted arguments reconstruct the effective policy whatever spelling the namespace
    uses for an aliased field (a namespace built in Python carries the canonical name, a parsed
    one the CLI alias, a sparse one neither), and the live namespace is never written."""
    negative = alias.startswith("no_")
    attrs = {}
    if spelling in ("canonical", "both"):
        attrs[name] = not value  # The namespace starts against the policy.
    if spelling in ("alias", "both"):
        attrs[alias] = value if negative else not value
    args = Namespace(**attrs)
    before = dict(vars(args))
    effective = replace(checkpoint_config_from_args(args), **{name: value})
    projected = project_checkpoint_policy(args, effective)
    assert getattr(checkpoint_config_from_args(projected), name) is value
    assert checkpoint_config_from_args(projected) == effective
    assert dict(vars(args)) == before
    if spelling == "canonical":
        assert not hasattr(projected, alias)  # No second spelling is introduced.


@pytest.mark.parametrize(
    "argv, match",
    [
        (["--ckpt-convert-format", "torch"], None),
        (["--ckpt-convert-format", "torch", "--ckpt-convert-save", "/c"], None),
        (["--save-interval", "-1"], "save_interval must be non-negative"),
        (["--save-retain-interval", "20"], "requires a positive save_interval"),
        (
            ["--save", "/s", "--save-interval", "10", "--save-retain-interval", "15"],
            "divisible by save_interval",
        ),
        (
            ["--non-persistent-ckpt-type", "local", "--non-persistent-save-interval", "5"],
            "without specifying --local-ckpt-dir",
        ),
        (["--load-main-params-from-ckpt"], "must be used with load_optim=False"),
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
    # A save directory without a periodic interval (unset or 0) is accepted: it saves at the end
    # of training and on the exit paths only, as Megatron-Bridge reads it.
    for argv in (
        [],
        CLI_CASE,
        ["--replication-jump", "2"],
        ["--async-save"],
        ["--save", "/s"],
        ["--save", "/s", "--save-interval", "0"],
    ):
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


# ---- consumers read the owner, with the namespace stripped and with the namespace disagreeing ----

CHECKPOINT_NAMES = tuple(f.name for f in fields(CheckpointConfig)) + (
    "no_save_optim",
    "no_save_rng",
    "no_load_optim",
    "no_load_rng",
    "ckpt_fully_parallel_save",
    "ckpt_fully_parallel_load",
    "use_dist_ckpt",
)


def _args_without_checkpoint(args):
    """The parsed namespace with every checkpoint-owned name removed."""
    for name in CHECKPOINT_NAMES:
        if hasattr(args, name):
            delattr(args, name)
    return args


def _args_contradicting(args, checkpoint):
    """The parsed namespace with every checkpoint-owned name set against the section."""
    for item in fields(CheckpointConfig):
        value = getattr(checkpoint, item.name)
        if isinstance(value, bool):
            setattr(args, item.name, not value)
        elif isinstance(value, str) or value is None:
            setattr(args, item.name, "/contradiction")
        elif isinstance(value, int):
            setattr(args, item.name, value + 7)
    for name in ("save_optim", "save_rng", "load_optim", "load_rng"):
        setattr(args, f"no_{name}", getattr(checkpoint, name))
    args.ckpt_fully_parallel_save = not checkpoint.fully_parallel_save
    args.ckpt_fully_parallel_load = not checkpoint.fully_parallel_load
    args.use_dist_ckpt = not checkpoint.use_dist_ckpt
    return args


@pytest.mark.parametrize("namespace", ["stripped", "contradicting"])
@pytest.mark.parametrize("async_save", [False, True])
def test_async_finalization_follows_the_section(monkeypatch, run_config, namespace, async_save):
    run_config.checkpoint.async_save = async_save
    args = _validated()
    args = (
        _args_without_checkpoint(args)
        if namespace == "stripped"
        else _args_contradicting(args, run_config.checkpoint)
    )
    monkeypatch.setattr(async_utils, "get_args", lambda: args, raising=False)
    queue = Mock()
    monkeypatch.setattr(async_utils, "_async_calls_queue", queue)
    monkeypatch.setattr(async_utils, "_tag_current_span_call_idx", Mock())
    monkeypatch.setattr(async_utils, "is_empty_async_queue", lambda: True)
    async_utils.maybe_finalize_async_save(blocking=False)
    assert queue.maybe_finalize_async_calls.called is async_save


@pytest.mark.parametrize("persistent", [False, True])
def test_lazy_async_queue_follows_the_section(monkeypatch, run_config, persistent):
    """The lazy queue used to read use_persistent_ckpt_worker through a getattr with a False
    default, which a stripped namespace alone would not catch; the namespace disagrees instead."""
    core = pytest.importorskip("nvidia_resiliency_ext.checkpointing.async_ckpt.core")
    run_config.checkpoint.use_persistent_ckpt_worker = persistent
    args = _validated()
    args.use_persistent_ckpt_worker = not persistent
    monkeypatch.setattr(async_utils, "get_args", lambda: args, raising=False)
    monkeypatch.setattr(async_utils, "_async_calls_queue", None)
    queue_cls = Mock()
    monkeypatch.setattr(core, "AsyncCallsQueue", queue_cls)
    async_utils._get_async_calls_queue()
    queue_cls.assert_called_once_with(persistent=persistent)


@pytest.mark.parametrize("section_saves", [False, True])
def test_autoresume_saves_only_when_the_section_has_a_save_directory(
    monkeypatch, run_config, section_saves
):
    run_config.checkpoint.save = "/section/save" if section_saves else None
    args = _validated()
    args.save = None if section_saves else "/namespace/save"
    monkeypatch.setattr(common_utils, "get_args", lambda: args)
    autoresume = SimpleNamespace(termination_requested=lambda: True, request_resume=Mock())
    monkeypatch.setattr(common_utils, "get_adlr_autoresume", lambda: autoresume)
    monkeypatch.setattr(common_utils.torch.distributed, "barrier", Mock())
    monkeypatch.setattr(common_utils.torch.distributed, "get_rank", lambda: 0)
    save = Mock()
    monkeypatch.setattr("megatron.training.checkpointing.save_checkpoint", save)
    with pytest.raises(SystemExit):
        common_utils.check_adlr_autoresume_termination(3, [], None, None)
    assert save.called is section_saves


def test_pretrain_finalizes_the_container_before_fault_tolerance_reads_it():
    """Fault tolerance captures the save directory and the async mode at setup; it must see the
    derived values (the async flip, the conversion exit flag)."""
    source = inspect.getsource(training.pretrain)
    assert source.index("cfg_container.finalize()") < source.index("ft_integration.setup()")


# ---- no consumer inside the migrated boundary reads a checkpoint name from args ----

CONSUMER_MODULES = (
    "megatron/training/checkpointing.py",
    "megatron/training/training.py",
    "megatron/training/async_utils.py",
    "megatron/training/initialize.py",
    "megatron/training/ft_integration.py",
    "megatron/training/global_vars.py",
    "megatron/training/utils/common_utils.py",
    "megatron/rl/rl_utils.py",
    "megatron/post_training/checkpointing.py",
    "megatron/post_training/model_builder.py",
    "megatron/inference/utils.py",
)
# Legacy helpers that maintain the argument namespace itself rather than consume it.
ARGS_MAINTAINERS = {("megatron/training/utils/common_utils.py", "update_use_dist_ckpt")}


def _args_reads(path, names, skip_functions):
    """`args.<name>` and `getattr(args, "<name>")` uses outside the skipped functions."""
    module = ast.parse(path.read_text())
    skipped = [
        range(node.lineno, node.end_lineno + 1)
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name in skip_functions
    ]
    reads = []
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and node.attr in names
        ):
            name = node.attr
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("getattr", "hasattr")
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "args"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in names
        ):
            name = node.args[1].value
        else:
            continue
        if not any(node.lineno in span for span in skipped):
            reads.append(f"{path.name}:{node.lineno} args.{name}")
    return reads


def test_consumers_read_no_checkpoint_name_from_args():
    root = Path(megatron.training.__file__).resolve().parents[2]
    reads = []
    for module in CONSUMER_MODULES:
        skip = {function for owner, function in ARGS_MAINTAINERS if owner == module}
        reads += _args_reads(root / module, set(CHECKPOINT_NAMES), skip)
    assert reads == []
