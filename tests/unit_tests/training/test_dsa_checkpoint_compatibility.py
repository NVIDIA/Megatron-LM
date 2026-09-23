# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint metadata compatibility for dense-to-DSA indexer initialization."""

from types import SimpleNamespace

import pytest

from megatron.training import checkpointing


@pytest.fixture
def checkpoint_args_pair(monkeypatch):
    backbone = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        add_position_embedding=False,
        gdp_num_householder=3,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    saved = SimpleNamespace(**backbone, experimental_attention_variant=None)
    runtime = SimpleNamespace(
        **backbone,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_simplified_use_learned_k=True,
        dsa_simplified_indexer_disable_main_input_norm=False,
        dsa_standard_indexer_use_main_input_norm=False,
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
        dsa_indexer_use_hadamard=False,
        dsa_train_indexer_only=True,
        dsa_train_main_only=False,
        no_load_optim=True,
        finetune=False,
        load="dense-checkpoint",
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)
    monkeypatch.setattr(checkpointing, "print_rank_0", lambda *_args: None)
    return saved, runtime


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("backend", ["reference", "torch-min-memory", "triton-min-memory", "cute"])
def test_dense_checkpoint_can_initialize_indexer_without_finetuning(
    monkeypatch, checkpoint_args_pair, legacy, backend
):
    saved, runtime = checkpoint_args_pair
    if legacy:
        del saved.experimental_attention_variant
    else:
        # New dense checkpoints serialize inert DSA defaults, not a saved indexer architecture.
        saved.dsa_indexer_mode = "standard"
        saved.dsa_simplified_use_learned_k = False
        saved.dsa_indexer_n_heads = 8
        saved.dsa_indexer_head_dim = 64
        saved.dsa_indexer_topk = 2048
    runtime.dsa_gqa_backend = runtime.dsa_min_memory_backend = backend
    saved.consumed_train_samples = 592
    saved.consumed_valid_samples = 32
    saved.skipped_train_samples = 16
    saved_before = vars(saved).copy()
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *_args, **_kwargs: ({"args": saved, "iteration": 37}, "checkpoint.pt", False, None),
    )

    restored, _ = checkpointing.load_args_from_checkpoint(runtime)
    checkpointing.check_checkpoint_args(saved)

    assert restored.iteration == 37
    assert not restored.finetune
    assert restored.no_load_optim
    assert restored.experimental_attention_variant == "dsa"
    assert restored.dsa_indexer_mode == "simplified"
    assert restored.dsa_simplified_use_learned_k
    assert restored.dsa_indexer_n_heads == 1
    assert restored.dsa_indexer_head_dim == 128
    assert restored.dsa_indexer_topk == 512
    assert restored.dsa_gqa_backend == restored.dsa_min_memory_backend == backend
    assert vars(saved) == saved_before


@pytest.mark.parametrize(
    "overrides",
    [
        {"no_load_optim": False},
        {"dsa_train_indexer_only": False},
        {"dsa_train_indexer_only": False, "dsa_train_main_only": True},
        {"experimental_attention_variant": "other"},
    ],
)
def test_dense_conversion_remains_restricted_to_fresh_indexer_training(
    checkpoint_args_pair, overrides
):
    saved, runtime = checkpoint_args_pair
    vars(runtime).update(overrides)
    with pytest.raises(AssertionError, match="experimental_attention_variant"):
        checkpointing.check_checkpoint_args(saved)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_layers", 4),
        ("hidden_size", 128),
        ("num_attention_heads", 8),
        ("add_position_embedding", True),
        ("gdp_num_householder", 4),
        ("tensor_model_parallel_size", 2),
        ("pipeline_model_parallel_size", 2),
    ],
)
def test_dense_indexer_initialization_still_checks_backbone(checkpoint_args_pair, field, value):
    saved, _ = checkpoint_args_pair
    setattr(saved, field, value)
    with pytest.raises(AssertionError, match=field):
        checkpointing.check_checkpoint_args(saved)


@pytest.mark.parametrize("no_load_optim", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("dsa_indexer_mode", "standard"),
        ("dsa_simplified_use_learned_k", False),
        ("dsa_indexer_head_dim", 64),
        ("dsa_indexer_topk", 1024),
        ("dsa_simplified_indexer_disable_main_input_norm", True),
    ],
)
def test_existing_dsa_checkpoint_still_checks_indexer_architecture(
    checkpoint_args_pair, no_load_optim, field, value
):
    _, runtime = checkpoint_args_pair
    runtime.no_load_optim = no_load_optim
    saved = SimpleNamespace(**vars(runtime))
    checkpointing.check_checkpoint_args(saved)
    setattr(saved, field, value)
    with pytest.raises(AssertionError, match=field):
        checkpointing.check_checkpoint_args(saved)


def test_other_attention_variant_is_not_a_dense_checkpoint(checkpoint_args_pair):
    saved, _ = checkpoint_args_pair
    saved.experimental_attention_variant = "other"
    with pytest.raises(AssertionError, match="experimental_attention_variant"):
        checkpointing.check_checkpoint_args(saved)
