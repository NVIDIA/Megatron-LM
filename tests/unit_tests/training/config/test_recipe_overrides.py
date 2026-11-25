# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Recipe overrides edit native config leaves without replacing Python objects."""

import pytest

from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.training.config.utils import apply_config_overrides


def test_overrides_preserve_native_objects_and_leave_finalization_to_startup(run_config):
    transformer = run_config.model.transformer
    activation = transformer.activation_func
    layout = PipelineParallelLayerLayout.from_str("Et*2L", pipeline_model_parallel_size=1)
    transformer.pipeline_model_parallel_layout = layout
    apply_config_overrides(
        run_config,
        [
            "train.train_iters=4",
            "train.train_iters=8",
            "optimizer.lr=3e-4",
            "train.manual_gc=true",
            "train.exit_interval=null",
            "profiling.profile_ranks=[2,4]",
        ],
    )
    assert run_config.train.train_iters == 8
    assert run_config.optimizer.lr == 3e-4
    assert run_config.train.manual_gc is True
    assert run_config.train.exit_interval is None
    assert run_config.profiling.profile_ranks == [2, 4]
    assert run_config.model.transformer is transformer
    assert transformer.activation_func is activation
    assert transformer.pipeline_model_parallel_layout is layout
    assert not run_config._finalized


@pytest.mark.parametrize("layout", ["Et*2|t*2L", "invalid", "[[E,t,t],[t,t,L]]", "null"])
def test_pipeline_layout_override_requires_recipe_construction(run_config, layout):
    transformer = run_config.model.transformer
    original = transformer.pipeline_model_parallel_layout
    with pytest.raises(ValueError, match="recipe factory"):
        apply_config_overrides(
            run_config, [f"model.transformer.pipeline_model_parallel_layout={layout}"]
        )
    assert transformer.pipeline_model_parallel_layout is original
    assert not run_config._finalized


@pytest.mark.parametrize(
    "override",
    [
        "train.typo=2",
        "train.train_iters.value=2",
        "_finalized=true",
        "train={}",
        "train.train_iters",
        "model.transformer.activation_func=silu",
    ],
)
def test_invalid_override_cannot_create_phantom_fields(run_config, override):
    with pytest.raises(ValueError):
        apply_config_overrides(run_config, [override])
    assert not hasattr(run_config.train, "typo")
    assert not run_config._finalized


def test_rank_local_failure_reaches_launcher_without_collective_teardown(run_config, monkeypatch):
    from megatron.training import api
    from scripts import run_recipe

    def fail_training(*args, **kwargs):
        raise RuntimeError("rank-local dataset failure")

    def fail_teardown():
        pytest.fail("Collective teardown can hide the rank-local exception")

    monkeypatch.setattr(run_recipe, "_load_callable", lambda path: lambda: run_config)
    monkeypatch.setattr(api, "pretrain", fail_training)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.destroy_process_group", fail_teardown)
    with pytest.raises(RuntimeError, match="rank-local dataset failure"):
        run_recipe.main(["--recipe", "test:recipe", "--forward-step", "test:step"])
