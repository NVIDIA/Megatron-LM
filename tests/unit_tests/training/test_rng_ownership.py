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


@pytest.mark.parametrize("graph", ["none", "local", "transformer_engine"])
@pytest.mark.parametrize("transformer", ["local", "transformer_engine"])
@pytest.mark.parametrize("enabled", [False, True])
def test_tracker_resolution_matches_cli_policy(monkeypatch, graph, transformer, enabled):
    warning = Mock()
    monkeypatch.setattr("megatron.training.utils.warn_rank_0", warning)
    rng = RNGConfig(te_rng_tracker=enabled)
    rng.resolve_cuda_graphs(transformer_impl=transformer, cuda_graph_impl=graph, rank=0)
    required = graph != "none" and "transformer_engine" in (graph, transformer)
    assert rng.te_rng_tracker == (enabled or required)
    rng.resolve_cuda_graphs(transformer_impl=transformer, cuda_graph_impl=graph, rank=0)
    assert warning.call_count == int(required and not enabled)


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
    rng = RNGConfig(
        seed=987, te_rng_tracker=True, inference_rng_tracker=True, data_parallel_random_init=True
    )
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
    for name, value in asdict(rng).items():
        assert metadata[name] == value
    assert vars(args) == original


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("skip", [False, True])
def test_initialization_uses_owner_and_preserves_deferred_seeding(
    monkeypatch, lazy, skip, run_config
):
    args = _args_without_rng()
    args.lazy_mpu_init = lazy
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
    rng = RNGConfig(
        seed=919, data_parallel_random_init=True, te_rng_tracker=True, inference_rng_tracker=True
    )
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
        assert seed.call_args.args == (919, True, True, True)
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


@pytest.mark.parametrize("model_kind", ["gpt", "hybrid"])
def test_native_builder_derives_tracker_and_preserves_sampling_seed(
    monkeypatch, model_kind, run_config
):
    from megatron.core.transformer import TransformerConfig
    from megatron.training import training
    from megatron.training.models import gpt, hybrid

    config_cls = gpt.GPTModelConfig if model_kind == "gpt" else hybrid.HybridModelConfig
    transformer = TransformerConfig(
        num_layers=2, hidden_size=32, num_attention_heads=4, inference_sampling_seed=777
    )
    run_config.model = config_cls(transformer=transformer, vocab_size=128, seq_length=16)
    run_config.rng = RNGConfig(seed=987, inference_rng_tracker=True)
    args = _args_without_rng()
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", Mock())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "is_gtp_remat_active", lambda args: False)
    monkeypatch.setattr(training, "_add_model_freeze_pre_wrap_hook", Mock())
    monkeypatch.setattr("megatron.training.utils.start_memory_history_recording", Mock())

    class ObservedModel(Exception):
        pass

    def construct(model_config):
        assert model_config.transformer.inference_sampling_seed == 777
        assert model_config.transformer.inference_rng_tracker is True
        raise ObservedModel

    monkeypatch.setattr(config_cls, "get_builder_cls", lambda self: construct)
    with pytest.raises(ObservedModel):
        training.setup_model_and_optimizer(Mock(), cfg_container=run_config)


@pytest.mark.parametrize("source", ["args", "yaml", "native", "provided", "separate"])
@pytest.mark.parametrize("enabled", [False, True])
def test_shared_transformer_resolution(monkeypatch, source, enabled, run_config):
    from megatron.core.transformer import TransformerConfig
    from megatron.training import argument_utils, yaml_arguments
    from megatron.training.models import GPTModelConfig

    args = _args_without_rng()
    args.yaml_cfg = "model.yaml" if source == "yaml" else None
    before = vars(args).copy()
    run_config.rng = RNGConfig(seed=987, inference_rng_tracker=enabled)
    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        inference_sampling_seed=777,
        inference_rng_tracker=not enabled,
    )
    native = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
    if source in ("native", "provided", "separate"):
        run_config.model = GPTModelConfig(
            transformer=transformer if source == "native" else native, vocab_size=128, seq_length=16
        )

    def from_args(legacy_args, **overrides):
        assert legacy_args is args
        assert overrides == {"inference_sampling_seed": 987, "inference_rng_tracker": enabled}
        transformer.inference_sampling_seed = overrides["inference_sampling_seed"]
        return transformer

    cli_factory = Mock(side_effect=from_args)
    yaml_factory = Mock(return_value=transformer)
    monkeypatch.setattr(argument_utils, "core_transformer_config_from_args", cli_factory)
    monkeypatch.setattr(yaml_arguments, "core_transformer_config_from_yaml", yaml_factory)
    result = argument_utils.get_transformer_config(
        args,
        transformer if source == "provided" else None,
        use_yaml=True,
        reuse_model_config=source != "separate",
    )
    assert result is transformer
    assert result.inference_rng_tracker is enabled
    assert result.inference_sampling_seed == (987 if source in ("args", "separate") else 777)
    assert cli_factory.call_count == (source in ("args", "separate"))
    assert yaml_factory.call_count == (source == "yaml")
    assert native.inference_rng_tracker is False
    assert vars(args) == before


@pytest.mark.parametrize("provider", ["gpt", "hybrid", "mamba"])
@pytest.mark.parametrize("source", ["native", "legacy", "other_provider"])
def test_inference_builder_respects_native_model_and_provider_override(
    monkeypatch, provider, source, run_config
):
    from megatron.core.transformer import TransformerConfig
    from megatron.inference import utils
    from megatron.training import argument_utils
    from megatron.training.models import GPTModelConfig, HybridModelConfig

    requested_cls = GPTModelConfig if provider == "gpt" else HybridModelConfig
    other_cls = HybridModelConfig if provider == "gpt" else GPTModelConfig

    def model(cls, seed):
        return cls(
            transformer=TransformerConfig(
                num_layers=2, hidden_size=32, num_attention_heads=4, inference_sampling_seed=seed
            ),
            vocab_size=128,
            seq_length=16,
        )

    native = model(requested_cls if source == "native" else other_cls, 777)
    run_config.model = None if source == "legacy" else native
    run_config.rng = RNGConfig(seed=987, inference_rng_tracker=True)
    built = model(requested_cls, 987)
    resolver = Mock(return_value=built.transformer)
    factory = Mock(return_value=built)
    monkeypatch.setattr(argument_utils, "get_transformer_config", resolver)
    monkeypatch.setattr(utils, "gpt_config_from_args", factory)
    monkeypatch.setattr(utils, "hybrid_config_from_args", factory)
    monkeypatch.setattr(utils, "GPTModelBuilder", lambda config: config)
    monkeypatch.setattr(utils, "HybridModelBuilder", lambda config: config)
    args = Namespace(model_provider="gpt")
    result = utils.get_model_builder(args, provider)
    assert result is (native if source == "native" else built)
    assert result.transformer.inference_rng_tracker is True
    assert result.transformer.inference_sampling_seed == (777 if source == "native" else 987)
    if source == "native":
        factory.assert_not_called()
        resolver.assert_not_called()
    else:
        resolver.assert_called_once_with(args, use_yaml=provider == "gpt", reuse_model_config=False)
        factory.assert_called_once_with(args, config=built.transformer)
        assert native.transformer.inference_rng_tracker is False


@pytest.mark.parametrize("model_type", ["gpt", "hybrid", "legacy"])
def test_inference_finalization_is_idempotent_without_training_configs(model_type):
    from megatron.core.transformer import TransformerConfig
    from megatron.training.config import (
        CheckpointConfig,
        InferenceConfigContainer,
        InferenceSetupConfig,
    )
    from megatron.training.models import GPTModelConfig, HybridModelConfig

    model = None
    if model_type != "legacy":
        cls = GPTModelConfig if model_type == "gpt" else HybridModelConfig
        model = cls(
            transformer=TransformerConfig(
                num_layers=2, hidden_size=32, num_attention_heads=4, inference_sampling_seed=777
            ),
            vocab_size=128,
            seq_length=16,
        )
    cfg = InferenceConfigContainer(
        model=model,
        checkpoint=CheckpointConfig(),
        inference=InferenceSetupConfig(),
        rng=RNGConfig(seed=987, inference_rng_tracker=True),
    )
    cfg.finalize()
    cfg.finalize()
    assert not hasattr(cfg, "optimizer")
    if model is not None:
        assert model.transformer.inference_rng_tracker is True
        assert model.transformer.inference_sampling_seed == 777
        cfg.rng.inference_rng_tracker = False
        cfg.finalize()
        assert model.transformer.inference_rng_tracker is False


def test_mimo_finalization_updates_specs_not_metadata(run_config):
    from megatron.core.models.mimo.config.base_configs import MimoModelConfig
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.spec_utils import ModuleSpec

    active = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    ignored = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    spec = ModuleSpec(
        module=Mock,
        params={"config": active, "metadata": SimpleNamespace(config=ignored)},
        metainfo={"config": ignored},
    )
    spec.submodules = {"shared": [spec, spec]}
    model = MimoModelConfig(
        language_model_spec=spec, module_to_grid_map={MIMO_LANGUAGE_MODULE_KEY: ignored}
    )
    run_config.rng.inference_rng_tracker = True
    run_config.finalize_model_config(model)
    assert active.inference_rng_tracker is True
    assert ignored.inference_rng_tracker is False
    run_config.finalize_model_config(SimpleNamespace(config=ignored))
    assert ignored.inference_rng_tracker is False


def test_tensorboard_records_owned_rng_values(monkeypatch, run_config):
    args = Namespace(iteration=19, seed=1)
    writer = Mock()
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "get_tensorboard_writer", lambda: writer)
    run_config.rng = RNGConfig(seed=321, te_rng_tracker=True)
    initialize.write_args_to_tensorboard()
    writer.add_text.assert_any_call("seed", "321", global_step=19)
    writer.add_text.assert_any_call("te_rng_tracker", "True", global_step=19)
    assert vars(args) == {"iteration": 19, "seed": 1}


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
