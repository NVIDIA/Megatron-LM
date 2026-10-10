# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Behavioral coverage for RNG policy after legacy argument construction."""

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import random as tensor_parallel_random
from megatron.training import arguments, checkpointing, global_vars, initialize
from megatron.training.config.common_config import RNGConfig
from tests.unit_tests.test_utilities import Utils


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
def test_tracker_cli_flags_populate_rng_config(enabled):
    from megatron.training.argument_utils import _default_config_from_args

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(["--te-rng-tracker", "--inference-rng-tracker"] if enabled else [])
    assert args.te_rng_tracker is enabled
    assert args.inference_rng_tracker is enabled
    config = _default_config_from_args(RNGConfig, args)
    assert asdict(config) == {
        "seed": 1234,
        "data_parallel_random_init": False,
        "te_rng_tracker": enabled,
        "inference_rng_tracker": enabled,
    }


@pytest.mark.parametrize("model_kind", ["gpt", "hybrid"])
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("te_tracker", [False, True])
@pytest.mark.parametrize("inference_tracker", [False, True])
def test_container_derives_tracker_fields_one_way(
    run_config, model_kind, inference, te_tracker, inference_tracker
):
    from megatron.core.transformer import TransformerConfig
    from megatron.training.config import (
        CheckpointConfig,
        InferenceConfigContainer,
        InferenceSetupConfig,
    )
    from megatron.training.models import GPTModelConfig, HybridModelConfig

    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        use_te_rng_tracker=not te_tracker,
        inference_rng_tracker=not inference_tracker,
    )
    model_cls = GPTModelConfig if model_kind == "gpt" else HybridModelConfig
    model = model_cls(transformer=transformer, vocab_size=128, seq_length=16)
    cfg = (
        InferenceConfigContainer(
            model=model, checkpoint=CheckpointConfig(), inference=InferenceSetupConfig()
        )
        if inference
        else run_config
    )
    cfg.model = model
    cfg.rng = RNGConfig(te_rng_tracker=te_tracker, inference_rng_tracker=inference_tracker)
    before = asdict(cfg.rng)
    for _ in range(2):
        cfg.validate()
        assert transformer.use_te_rng_tracker is te_tracker
        assert transformer.inference_rng_tracker is inference_tracker
        assert asdict(cfg.rng) == before


def test_native_cuda_graph_requirement_is_resolved_before_seeding(run_config):
    from megatron.core.transformer import TransformerConfig
    from megatron.training.models import GPTModelConfig

    transformer = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    transformer.cuda_graph_impl = "transformer_engine"
    run_config.model = GPTModelConfig(transformer=transformer, vocab_size=128, seq_length=16)
    run_config.rng = RNGConfig()
    run_config.validate()
    assert run_config.rng.te_rng_tracker is True
    assert transformer.use_te_rng_tracker is True
    run_config.validate()
    assert run_config.rng.te_rng_tracker is True


@pytest.mark.parametrize("yaml", [False, True])
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_shared_config_factories_use_run_rng_after_bootstrap(
    monkeypatch, run_config, yaml, registered, enabled
):
    import torch

    from megatron.core.transformer import TransformerConfig
    from megatron.training.argument_utils import core_transformer_config_from_args
    from megatron.training.yaml_arguments import core_transformer_config_from_yaml

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.params_dtype = torch.float32
    args.num_layers, args.hidden_size, args.num_attention_heads = 2, 32, 4
    args.te_rng_tracker = args.inference_rng_tracker = not enabled if registered else enabled
    if yaml:
        args.language_model = Namespace(
            **vars(TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4))
        )
        args.language_model.activation_func = "gelu"
        args.language_model.embedding_init_method = "xavier_uniform"
        args.model_parallel = Namespace()
    run_config.rng = RNGConfig(te_rng_tracker=enabled, inference_rng_tracker=enabled)
    if not registered:
        monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
    factory = core_transformer_config_from_yaml if yaml else core_transformer_config_from_args
    config = factory(args)
    assert config.use_te_rng_tracker is enabled
    assert config.inference_rng_tracker is enabled
    assert args.te_rng_tracker is (not enabled if registered else enabled)
    if registered:
        del args.te_rng_tracker, args.inference_rng_tracker
        config = factory(args)
        assert config.use_te_rng_tracker is enabled
        assert config.inference_rng_tracker is enabled


@pytest.mark.parametrize("te_tracker", [False, True])
@pytest.mark.parametrize("inference_tracker", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("skip", [False, True])
def test_initialization_uses_owner_and_preserves_deferred_seeding(
    monkeypatch, lazy, skip, te_tracker, inference_tracker, run_config
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
        seed=919,
        data_parallel_random_init=True,
        te_rng_tracker=te_tracker,
        inference_rng_tracker=inference_tracker,
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


def _forbid_global_process_groups(monkeypatch):
    """Make every global group, rank and world-size accessor raise, including by-name imports."""

    def forbidden(*args, **kwargs):
        pytest.fail("the global parallel state was read")

    for name in dir(parallel_state):
        if name.startswith("get_") and name.endswith(
            ("_rank", "_ranks", "_world_size", "_group", "_groups")
        ):
            monkeypatch.setattr(parallel_state, name, forbidden)
            if hasattr(tensor_parallel_random, name):
                monkeypatch.setattr(tensor_parallel_random, name, forbidden)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", forbidden)


@pytest.mark.parametrize("dp_random", [False, True])
def test_collection_seed_offsets_use_full_data_parallel_group(monkeypatch, dp_random):
    """With a collection, the offsets come from its pp and dp_gtp_remat groups, not the globals."""
    pp, dp, dp_gtp_remat = object(), object(), object()
    ranks = {pp: 2, dp: 5, dp_gtp_remat: 3}
    monkeypatch.setattr(initialize, "get_pg_rank", lambda group: ranks[group])
    _forbid_global_process_groups(monkeypatch)
    python_seed, numpy_seed, torch_seed, cuda_seed = Mock(), Mock(), Mock(), Mock()
    monkeypatch.setattr(initialize.random, "seed", python_seed)
    monkeypatch.setattr(initialize.np.random, "seed", numpy_seed)
    monkeypatch.setattr(initialize.torch, "manual_seed", torch_seed)
    monkeypatch.setattr(initialize.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(initialize.tensor_parallel, "model_parallel_cuda_manual_seed", cuda_seed)
    pg_collection = ProcessGroupCollection(pp=pp, dp=dp, dp_gtp_remat=dp_gtp_remat)
    initialize._set_random_seed(101, dp_random, pg_collection=pg_collection)
    expected_seed = 301 + (30 if dp_random else 0)
    for seeded in (python_seed, numpy_seed, torch_seed):
        seeded.assert_called_once_with(expected_seed)
    assert cuda_seed.call_args.args[0] == expected_seed
    assert cuda_seed.call_args.kwargs["pg_collection"] is pg_collection
    for rank_or_size in (
        "tp_rank",
        "ep_rank",
        "etp_rank",
        "gtp_remat_rank",
        "egtp_remat_rank",
        "gtp_remat_world_size",
        "egtp_remat_world_size",
    ):
        assert cuda_seed.call_args.kwargs[rank_or_size] is None


@pytest.mark.parametrize(
    "groups, dp_random",
    [({"dp_gtp_remat": object()}, False), ({"pp": object(), "dp": object()}, True)],
)
def test_collection_seed_requires_its_offset_groups(monkeypatch, groups, dp_random):
    """A collection without pp, or without dp_gtp_remat for DP-random init, is an error."""
    _forbid_global_process_groups(monkeypatch)
    with pytest.raises(ValueError, match="pg_collection must set pp"):
        initialize._set_random_seed(101, dp_random, pg_collection=ProcessGroupCollection(**groups))


def test_initialization_seeds_from_the_model_collection(monkeypatch, run_config):
    args = _args_without_rng()
    args.lazy_mpu_init = False
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
    seed = Mock()
    monkeypatch.setattr(initialize, "_set_random_seed", seed)
    pg_collection = ProcessGroupCollection()
    initialize.initialize_megatron(
        allow_no_cuda=True, skip_dependency_compilation=True, seed_pg_collection=pg_collection
    )
    assert seed.call_args.kwargs["pg_collection"] is pg_collection


@pytest.mark.parametrize("tp, pp, gtp", [(1, 2, 2), (2, 1, 2), (1, 1, 1)])
@pytest.mark.parametrize("dp_random", [False, True])
def test_collection_seeds_match_the_global_grid(monkeypatch, tp, pp, gtp, dp_random):
    """Seeding from the global grid's collection reproduces the seeds of the global path."""
    if Utils.world_size < 4 or Utils.world_size % (tp * pp * gtp):
        pytest.skip("needs a world size of at least 4 that the grid divides")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, gtp_remat_size=gtp
    )
    try:
        tensor_parallel_random.initialize_rng_tracker(force_reset=True)

        def tracker_states():
            tracker = tensor_parallel_random.get_cuda_rng_tracker()
            return {name: state.clone() for name, state in tracker.get_states().items()}

        initialize._set_random_seed(1234, dp_random)
        expected = tracker_states()

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=[
                "pp",
                "dp",
                "dp_gtp_remat",
                "tp",
                "ep",
                "expt_tp",
                "gtp_remat",
                "expt_gtp_remat",
            ]
        )
        # Under GTP-remat the replicate DP group leaves out the GTP peers, so a DP offset taken
        # from it would differ from the global path.
        assert pg_collection.dp_gtp_remat.size() == gtp * pg_collection.dp.size()
        with monkeypatch.context() as patch:
            _forbid_global_process_groups(patch)
            initialize._set_random_seed(1234, dp_random, pg_collection=pg_collection)
        actual = tracker_states()

        assert actual.keys() == expected.keys()
        for name, state in expected.items():
            assert torch.equal(actual[name], state), name
    finally:
        Utils.destroy_model_parallel()


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
