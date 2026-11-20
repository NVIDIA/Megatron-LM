"""Recipe overrides edit native config leaves without replacing Python objects."""

import pytest

from scripts.run_recipe import _apply_overrides


def test_overrides_preserve_native_objects_and_leave_finalization_to_startup(run_config):
    transformer = run_config.model.transformer
    activation = transformer.activation_func
    _apply_overrides(
        run_config,
        [
            "train.train_iters=4",
            "train.train_iters=8",
            "optimizer.lr=3e-4",
            "train.manual_gc=true",
            "train.exit_interval=null",
            "train.phase_transition_iterations=[2,4]",
        ],
    )
    assert run_config.train.train_iters == 8
    assert run_config.optimizer.lr == 3e-4
    assert run_config.train.manual_gc is True
    assert run_config.train.exit_interval is None
    assert run_config.train.phase_transition_iterations == [2, 4]
    assert run_config.model.transformer is transformer
    assert transformer.activation_func is activation
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
        _apply_overrides(run_config, [override])
    assert not hasattr(run_config.train, "typo")
    assert not run_config._finalized
