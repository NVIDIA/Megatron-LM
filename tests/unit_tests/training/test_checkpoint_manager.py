# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the pluggable checkpoint manager (CPU only, no distributed setup)."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from megatron.training import checkpoint_manager as cm
from megatron.training.config import CheckpointConfig

_CUSTOM_MODULE = "megatron.training._custom_checkpoint_manager_for_test"
_GROUP_NAMES = (
    "tp_group",
    "pp_group",
    "cp_group",
    "dp_group",
    "dp_cp_group",
    "dp_gtp_remat_group",
    "expt_dp_group",
)


class _GoodManager:
    """Implements the full CheckpointManager interface."""

    def __init__(self, checkpoint_config):
        self.checkpoint_config = checkpoint_config

    def save(self, ctx):
        pass

    def load(self, ctx):
        return 3, 4

    def finalize_async_saves(self, blocking=False, terminate=False):
        pass


class _NoFinalizeManager:
    """Lacks finalize_async_saves."""

    def __init__(self, checkpoint_config):
        pass

    def save(self, ctx):
        pass

    def load(self, ctx):
        return 0, 0


class _RequiredStateFinalizeManager(_GoodManager):
    """finalize_async_saves keeps a required extra argument the trainer never passes."""

    def finalize_async_saves(self, state, blocking=False, terminate=False):
        pass


class _BlockingOnlyFinalizeManager(_GoodManager):
    """finalize_async_saves lacks terminate, which the trainer passes at shutdown."""

    def finalize_async_saves(self, blocking=False):
        pass


class _TwoArgSaveManager(_GoodManager):
    """save needs two arguments, but the loop passes one context."""

    def save(self, ctx, callback_manager):
        pass


@pytest.fixture(autouse=True)
def _fresh_manager():
    cm.set_checkpoint_manager(None)
    yield
    cm.set_checkpoint_manager(None)


@pytest.fixture
def custom_module(monkeypatch):
    """An importable module under an allowlisted prefix that holds the manager classes."""
    module = types.ModuleType(_CUSTOM_MODULE)
    module.GoodManager = _GoodManager
    module.NoFinalizeManager = _NoFinalizeManager
    module.TwoArgSaveManager = _TwoArgSaveManager
    module.RequiredStateFinalizeManager = _RequiredStateFinalizeManager
    module.BlockingOnlyFinalizeManager = _BlockingOnlyFinalizeManager
    monkeypatch.setitem(sys.modules, _CUSTOM_MODULE, module)
    return module


def _config(custom_manager_class=None):
    return SimpleNamespace(custom_manager_class=custom_manager_class)


class TestCreateCheckpointManager:
    def test_default_when_no_custom_class(self):
        manager = cm.create_checkpoint_manager(_config())

        assert isinstance(manager, cm.DefaultCheckpointManager)

    def test_custom_class_is_built_with_the_config(self, custom_module):
        config = _config(f"{_CUSTOM_MODULE}.GoodManager")

        manager = cm.create_checkpoint_manager(config)

        assert isinstance(manager, _GoodManager)
        assert manager.checkpoint_config is config

    def test_class_outside_the_allowlist_is_rejected_before_import(self):
        # An allowlist bypass would instead surface as ImportError from the missing package.
        with pytest.raises(ValueError, match="allowlist"):
            cm.create_checkpoint_manager(_config("not_an_allowed_pkg.module.Manager"))

    def test_non_class_target_is_rejected_without_being_called(self, custom_module):
        called = []
        custom_module.side_effect_fn = lambda config: called.append(config)

        with pytest.raises(TypeError, match="must be a class"):
            cm.create_checkpoint_manager(_config(f"{_CUSTOM_MODULE}.side_effect_fn"))

        assert called == []

    def test_unsafe_target_is_rejected_before_import(self):
        from megatron.training.config.instantiate_utils import target_allowlist

        with (
            patch.object(target_allowlist, "is_allowed", return_value=True),
            patch("importlib.import_module") as import_module,
        ):
            with pytest.raises(ValueError, match="not allowed"):
                cm.create_checkpoint_manager(_config("torch.hub.load"))

        import_module.assert_not_called()

    @pytest.mark.parametrize(
        "target, error",
        [
            ("NoDotsInThisName", ValueError),
            (f"{_CUSTOM_MODULE}.DoesNotExist", AttributeError),
            ("megatron.training._missing_module_for_test.Manager", ImportError),
            (f"{_CUSTOM_MODULE}.NoFinalizeManager", TypeError),
            (f"{_CUSTOM_MODULE}.TwoArgSaveManager", TypeError),
            (f"{_CUSTOM_MODULE}.RequiredStateFinalizeManager", TypeError),
            (f"{_CUSTOM_MODULE}.BlockingOnlyFinalizeManager", TypeError),
        ],
    )
    def test_invalid_custom_class_is_rejected(self, custom_module, target, error):
        with pytest.raises(error):
            cm.create_checkpoint_manager(_config(target))


class TestManagerReportsCheckpointing:
    def test_default_manager_reports_itself(self):
        assert cm.manager_reports_checkpointing(cm.DefaultCheckpointManager(_config()))

    def test_custom_manager_is_reported_by_the_trainer_unless_it_opts_in(self):
        assert not cm.manager_reports_checkpointing(_GoodManager(_config()))

        class _SelfReporting(_GoodManager):
            reports_checkpointing = True

        assert cm.manager_reports_checkpointing(_SelfReporting(_config()))


class TestDefaultCheckpointManager:
    def test_save_forwards_every_context_field(self):
        groups = {name: object() for name in _GROUP_NAMES}
        ctx = cm.CheckpointSaveContext(
            iteration=7,
            model=[Mock()],
            optimizer=Mock(),
            opt_param_scheduler=Mock(),
            num_floating_point_operations_so_far=123,
            checkpointing_context={"save_strategy": object()},
            non_persistent_ckpt=True,
            train_data_iterator=Mock(),
            preprocess_common_state_dict_fn=Mock(),
            rng_state_key_prefix="language.",
            on_saved=Mock(),
            **groups,
        )

        with patch("megatron.training.checkpointing.save_checkpoint") as save_checkpoint:
            cm.DefaultCheckpointManager(_config()).save(ctx)

        save_checkpoint.assert_called_once_with(
            7,
            ctx.model,
            ctx.optimizer,
            ctx.opt_param_scheduler,
            123,
            ctx.checkpointing_context,
            non_persistent_ckpt=True,
            train_data_iterator=ctx.train_data_iterator,
            preprocess_common_state_dict_fn=ctx.preprocess_common_state_dict_fn,
            rng_state_key_prefix="language.",
            on_saved=ctx.on_saved,
            **groups,
        )

    def test_load_forwards_every_context_field_and_returns_the_result(self):
        groups = {
            name: object()
            for name in ("tp_group", "pp_group", "dp_cp_group", "dp_group", "expt_dp_group")
        }
        ctx = cm.CheckpointLoadContext(
            model=[Mock()],
            optimizer=Mock(),
            opt_param_scheduler=Mock(),
            checkpointing_context={"load_strategy": object()},
            strict=False,
            restore_training_state=True,
            record_finetune=True,
            checkpoint_group=object(),
            rng_state_key_prefix="encoder.",
            model_sharded_state_dict_modifier=Mock(),
            **groups,
        )

        with patch(
            "megatron.training.checkpointing.load_checkpoint", return_value=(5, 6)
        ) as load_checkpoint:
            result = cm.DefaultCheckpointManager(_config()).load(ctx)

        assert result == (5, 6)
        load_checkpoint.assert_called_once_with(
            ctx.model,
            ctx.optimizer,
            ctx.opt_param_scheduler,
            strict=False,
            checkpointing_context=ctx.checkpointing_context,
            skip_load_to_model_and_opt=False,
            checkpoint_group=ctx.checkpoint_group,
            rng_state_key_prefix="encoder.",
            model_sharded_state_dict_modifier=ctx.model_sharded_state_dict_modifier,
            restore_training_state=True,
            record_finetune=True,
            **groups,
        )

    def test_contexts_carry_full_train_state_and_config(self):
        train_state, config = Mock(), Mock()
        save_ctx = cm.CheckpointSaveContext(
            iteration=1,
            model=[],
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            train_state=train_state,
            config=config,
        )
        load_ctx = cm.CheckpointLoadContext(
            model=[],
            optimizer=None,
            opt_param_scheduler=None,
            train_state=train_state,
            config=config,
        )
        for ctx in (save_ctx, load_ctx):
            assert ctx.train_state is train_state
            assert ctx.config is config

    @pytest.mark.parametrize(
        "ckpt_format, fsdp2_ddp, expected",
        [("torch_dist", True, True), ("torch_dist", False, False), ("torch", True, False)],
    )
    def test_load_skips_load_state_dict_only_for_fsdp2_torch_dist(
        self, ckpt_format, fsdp2_ddp, expected
    ):
        from megatron.core.distributed import TorchFullyShardedDataParallelConfig

        ddp = TorchFullyShardedDataParallelConfig() if fsdp2_ddp else Mock()
        ctx = cm.CheckpointLoadContext(
            model=[], optimizer=None, opt_param_scheduler=None, config=SimpleNamespace(ddp=ddp)
        )
        config = SimpleNamespace(custom_manager_class=None, ckpt_format=ckpt_format)

        with patch(
            "megatron.training.checkpointing.load_checkpoint", return_value=(0, 0)
        ) as load_checkpoint:
            cm.DefaultCheckpointManager(config).load(ctx)

        assert load_checkpoint.call_args.kwargs["skip_load_to_model_and_opt"] is expected

    @pytest.mark.parametrize("blocking, terminate", [(False, False), (True, True)])
    def test_finalize_forwards_blocking_and_terminate(self, blocking, terminate):
        with patch("megatron.training.async_utils.maybe_finalize_async_save") as finalize:
            cm.DefaultCheckpointManager(_config()).finalize_async_saves(
                blocking=blocking, terminate=terminate
            )

        finalize.assert_called_once_with(blocking=blocking, terminate=terminate)


class TestGetCheckpointManager:
    @staticmethod
    def _run_config(checkpoint):
        return patch(
            "megatron.training.global_vars.get_run_config",
            return_value=SimpleNamespace(checkpoint=checkpoint),
        )

    def test_one_manager_per_run_config(self):
        config = _config()

        with self._run_config(config):
            first = cm.get_checkpoint_manager()
            second = cm.get_checkpoint_manager()

        assert first is second

    def test_manager_is_rebuilt_when_the_checkpoint_config_changes(self, custom_module):
        with self._run_config(_config()):
            default_manager = cm.get_checkpoint_manager()
        with self._run_config(_config(f"{_CUSTOM_MODULE}.GoodManager")):
            custom_manager = cm.get_checkpoint_manager()

        assert isinstance(default_manager, cm.DefaultCheckpointManager)
        assert isinstance(custom_manager, _GoodManager)

    def test_manager_is_rebuilt_after_teardown_even_with_the_same_config_object(self):
        from megatron.training.global_vars import unset_global_variables

        config = _config()

        with self._run_config(config):
            first = cm.get_checkpoint_manager()
            unset_global_variables()
            second = cm.get_checkpoint_manager()

        assert first is not second

    def test_installed_manager_is_returned_for_the_current_config(self):
        installed = _GoodManager(None)
        config = _config()

        with self._run_config(config):
            cm.set_checkpoint_manager(installed)
            result = cm.get_checkpoint_manager()

        assert result is installed


class TestCheckpointConfigValidation:
    def test_custom_manager_class_must_be_fully_qualified(self):
        with pytest.raises(AssertionError, match="fully qualified"):
            CheckpointConfig(custom_manager_class="NoDotsInThisName").validate()

    def test_fully_qualified_custom_manager_class_is_accepted(self):
        CheckpointConfig(custom_manager_class="mypackage.module.Manager").validate()
