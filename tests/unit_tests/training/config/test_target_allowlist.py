# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.training.config.instantiate_utils import (
    InstantiationException,
    TargetAllowlist,
    _resolve_target,
    target_allowlist,
)


class TestTargetAllowlistIsAllowed:
    """Tests for the TargetAllowlist.is_allowed() method."""

    def test_allows_megatron_training_targets(self):
        al = TargetAllowlist()
        assert al.is_allowed("megatron.training.config.training_config.TrainingConfig")
        assert al.is_allowed("megatron.training.config.container.PretrainConfigContainer")

    def test_allows_megatron_core_targets(self):
        al = TargetAllowlist()
        assert al.is_allowed("megatron.core.optimizer.OptimizerConfig")
        assert al.is_allowed(
            "megatron.core.distributed.distributed_data_parallel_config.DistributedDataParallelConfig"
        )

    def test_allows_torch_targets(self):
        al = TargetAllowlist()
        assert al.is_allowed("torch.float16")
        assert al.is_allowed("torch.bfloat16")
        assert al.is_allowed("torch.float32")

    def test_allows_transformers_targets(self):
        al = TargetAllowlist()
        assert al.is_allowed("transformers.GenerationConfig.from_dict")
        assert al.is_allowed("transformers.LlamaConfig.from_dict")

    def test_allows_signal_targets(self):
        al = TargetAllowlist()
        assert al.is_allowed("signal.Signals")

    def test_allows_exact_functools_partial(self):
        al = TargetAllowlist()
        assert al.is_allowed("functools.partial")

    def test_blocks_os_system(self):
        al = TargetAllowlist()
        assert not al.is_allowed("os.system")

    def test_blocks_subprocess(self):
        al = TargetAllowlist()
        assert not al.is_allowed("subprocess.call")
        assert not al.is_allowed("subprocess.Popen")

    def test_blocks_builtins(self):
        al = TargetAllowlist()
        assert not al.is_allowed("builtins.eval")
        assert not al.is_allowed("builtins.exec")
        assert not al.is_allowed("builtins.__import__")

    def test_blocks_shutil(self):
        al = TargetAllowlist()
        assert not al.is_allowed("shutil.rmtree")

    def test_blocks_importlib(self):
        al = TargetAllowlist()
        assert not al.is_allowed("importlib.import_module")

    def test_blocks_empty_string(self):
        al = TargetAllowlist()
        assert not al.is_allowed("")

    def test_blocks_partial_prefix_match(self):
        """Ensure prefix matching doesn't match partial module names."""
        al = TargetAllowlist()
        # "torchvision" starts with "torch" but not "torch."
        assert not al.is_allowed("torchvision.models.resnet50")


class TestTargetAllowlistAddRemove:
    """Tests for add/remove prefix and exact."""

    def test_add_prefix(self):
        al = TargetAllowlist()
        assert not al.is_allowed("custom_lib.MyClass")
        al.add_prefix("custom_lib.")
        assert al.is_allowed("custom_lib.MyClass")

    def test_add_prefix_requires_trailing_dot(self):
        al = TargetAllowlist()
        with pytest.raises(ValueError, match="Prefix must end with '.'"):
            al.add_prefix("custom_lib")

    def test_add_prefix_is_idempotent(self):
        al = TargetAllowlist()
        al.add_prefix("custom_lib.")
        al.add_prefix("custom_lib.")
        assert al.allowed_prefixes.count("custom_lib.") == 1

    def test_remove_prefix(self):
        al = TargetAllowlist()
        al.add_prefix("custom_lib.")
        assert al.is_allowed("custom_lib.MyClass")
        al.remove_prefix("custom_lib.")
        assert not al.is_allowed("custom_lib.MyClass")

    def test_remove_prefix_not_found_raises(self):
        al = TargetAllowlist()
        with pytest.raises(ValueError):
            al.remove_prefix("nonexistent.")

    def test_add_exact(self):
        al = TargetAllowlist()
        assert not al.is_allowed("os.getcwd")
        al.add_exact("os.getcwd")
        assert al.is_allowed("os.getcwd")
        # Other os.* targets still blocked
        assert not al.is_allowed("os.system")

    def test_remove_exact(self):
        al = TargetAllowlist()
        al.add_exact("os.getcwd")
        assert al.is_allowed("os.getcwd")
        al.remove_exact("os.getcwd")
        assert not al.is_allowed("os.getcwd")

    def test_remove_exact_nonexistent_is_noop(self):
        al = TargetAllowlist()
        al.remove_exact("nonexistent.target")  # Should not raise


class TestTargetAllowlistEnableDisable:
    """Tests for enable/disable and env var override."""

    def test_disable_allows_everything(self):
        al = TargetAllowlist()
        al.disable()
        assert al.is_allowed("os.system")
        assert al.is_allowed("subprocess.call")
        assert not al.enabled

    def test_enable_after_disable(self):
        al = TargetAllowlist()
        al.disable()
        assert al.is_allowed("os.system")
        al.enable()
        assert not al.is_allowed("os.system")
        assert al.enabled

    def test_properties(self):
        al = TargetAllowlist()
        assert isinstance(al.allowed_prefixes, tuple)
        assert isinstance(al.allowed_exact, frozenset)
        assert "functools.partial" in al.allowed_exact
        assert "megatron.training." in al.allowed_prefixes


class TestResolveTargetAllowlistEnforcement:
    """Tests that _resolve_target() enforces the allowlist."""

    def test_blocked_string_target_raises(self):
        with pytest.raises(InstantiationException, match="not in the allowlist"):
            _resolve_target("os.system", "", check_callable=True)

    def test_blocked_target_error_message_contains_target_name(self):
        with pytest.raises(InstantiationException, match="os.system"):
            _resolve_target("os.system", "", check_callable=True)

    def test_blocked_target_error_message_contains_prefixes(self):
        with pytest.raises(InstantiationException, match="megatron.training."):
            _resolve_target("os.system", "", check_callable=True)

    def test_blocked_target_error_message_contains_remediation(self):
        with pytest.raises(InstantiationException, match="add_prefix"):
            _resolve_target("os.system", "", check_callable=True)

    def test_blocked_target_error_message_includes_full_key(self):
        with pytest.raises(InstantiationException, match="full_key: my.config.key"):
            _resolve_target("os.system", "my.config.key", check_callable=True)

    def test_nonstring_target_bypasses_allowlist(self):
        """Already-resolved callables should not be blocked."""
        result = _resolve_target(int, "", check_callable=True)
        assert result is int

    def test_allowed_target_resolves(self):
        """Allowed targets should be resolved normally."""
        result = _resolve_target("functools.partial", "", check_callable=True)
        import functools

        assert result is functools.partial


class TestResolveTargetClassAllowlistEnforcement:
    """Tests that _resolve_target_class() in utils.py respects the allowlist."""

    def test_blocked_target_returns_none(self):
        from megatron.training.config.utils import _resolve_target_class

        result = _resolve_target_class("os.system")
        assert result is None

    def test_allowed_target_resolves(self):
        from megatron.training.config.utils import _resolve_target_class

        # This should resolve to the actual class
        result = _resolve_target_class("megatron.training.config.instantiate_utils.TargetAllowlist")
        assert result is TargetAllowlist


class TestModuleLevelSingleton:
    """Tests for the module-level target_allowlist singleton."""

    def test_singleton_is_enabled_by_default(self):
        assert target_allowlist.enabled

    def test_singleton_has_default_prefixes(self):
        assert "megatron.training." in target_allowlist.allowed_prefixes
        assert "megatron.core." in target_allowlist.allowed_prefixes
        assert "torch." in target_allowlist.allowed_prefixes
