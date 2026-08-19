# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""One provider type, assembled from per-operation owners."""

from types import SimpleNamespace

import pytest

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.ops import (
    BackendOptions,
    BackendSpecProvider,
    build_spec_provider,
    find_operation,
    get_backend,
    operations,
)
from megatron.core.ops.linear.contract import COLUMN_PARALLEL_LAYER_NORM_LINEAR
from megatron.core.ops.norm import LAYER_NORM, WrappedTorchNorm
from megatron.core.ops.norm import reference as reference_norm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear

_needs_te = pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required")


def _options(**kwargs):
    return BackendOptions(transformer_impl="local", **kwargs)


class TestOneProviderType:
    def test_every_preset_returns_the_same_type(self):
        """Nothing downstream can branch on which backend it got."""
        assert type(get_backend("local")) is BackendSpecProvider

    def test_overrides_do_not_change_the_type(self):
        provider = build_spec_provider(_options(operation_backends={"layer_norm": "torch"}))
        assert type(provider) is BackendSpecProvider

    def test_an_unowned_slot_names_itself(self):
        """A slot with no owner explains how to select one instead of returning nonsense."""
        provider = BackendSpecProvider({})
        with pytest.raises(NotImplementedError, match="norm.layer_norm"):
            provider.layer_norm()


class TestAssembly:
    def test_owner_takes_over_only_its_operation(self):
        provider = build_spec_provider(_options(operation_backends={"layer_norm": "torch"}))
        assert provider.layer_norm(rms_norm=False) is WrappedTorchNorm
        assert provider.column_parallel_linear() is ColumnParallelLinear

    def test_assembly_costs_nothing_at_call_time(self):
        """The owner's bound method is attached directly, with no forwarding wrapper."""
        owner = reference_norm.Norm()
        provider = BackendSpecProvider({LAYER_NORM: owner})
        assert provider.layer_norm.__func__ is reference_norm.Norm.layer_norm
        assert provider.layer_norm.__self__ is owner

    def test_derived_query_follows_the_override(self):
        """fuse_layernorm_and_linear() is derived, so it must see the new owner's answer."""

        class _FusedNormLinear:
            def column_parallel_layer_norm_linear(self):
                return _FusedNormLinear

        assert get_backend("local").fuse_layernorm_and_linear() is False

        provider = BackendSpecProvider({COLUMN_PARALLEL_LAYER_NORM_LINEAR: _FusedNormLinear()})
        assert provider.fuse_layernorm_and_linear() is True

    def test_owner_without_the_operation_is_rejected(self):
        with pytest.raises(ValueError, match="reference.Norm.*does not implement core_attention"):
            BackendSpecProvider({find_operation("core_attention"): reference_norm.Norm()})

    def test_repr_shows_who_owns_what(self):
        provider = BackendSpecProvider({LAYER_NORM: reference_norm.Norm()})
        assert "layer_norm=reference.Norm" in repr(provider)


class TestBackendNamesAreScopedToTheirOperation:
    def test_the_same_name_means_different_things_per_operation(self):
        """'transformer_engine' resolves within the operation's family, not globally."""
        from megatron.core.ops import backends_for

        assert "apex" in backends_for(find_operation("layer_norm"))
        assert "apex" not in backends_for(find_operation("core_attention"))

    def test_unknown_backend_lists_only_the_relevant_choices(self):
        with pytest.raises(ValueError, match="Unknown backend 'liger' for operation 'layer_norm'"):
            build_spec_provider(_options(operation_backends={"layer_norm": "liger"}))

    def test_a_backend_from_another_family_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown backend 'apex'"):
            build_spec_provider(_options(operation_backends={"core_attention": "apex"}))

    def test_unknown_operation_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="Unknown operation 'norm'"):
            build_spec_provider(_options(operation_backends={"norm": "torch"}))

    def test_operations_can_be_named_by_family(self):
        assert find_operation("norm.layer_norm") is LAYER_NORM

    def test_unknown_preset_is_rejected(self):
        with pytest.raises(ValueError, match="unknown transformer_impl='nope'"):
            get_backend("nope")


class TestOptions:
    def test_overrides_are_copied_from_the_caller(self):
        overrides = {"layer_norm": "torch"}
        options = _options(operation_backends=overrides)
        overrides["layer_norm"] = "apex"
        assert options.operation_backends["layer_norm"] == "torch"

    def test_config_field_is_read(self):
        config = SimpleNamespace(
            transformer_impl="local", op_backend_overrides={"layer_norm": "torch"}
        )
        options = BackendOptions.from_config(config)
        assert options.operation_backends["layer_norm"] == "torch"

    @_needs_te
    def test_transformer_impl_override_wins_over_the_config(self):
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
        from megatron.core.ops import get_backend_spec_provider

        config = SimpleNamespace(transformer_impl="local")
        provider = get_backend_spec_provider(config, transformer_impl="transformer_engine")
        assert provider.column_parallel_linear() is TEColumnParallelLinear


class TestConflictingSelectors:
    def test_two_settings_claiming_one_operation_is_an_error(self):
        options = _options(
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl="native",
            operation_backends={"vocab_parallel_cross_entropy": "megatron"},
        )
        with pytest.raises(ValueError, match="vocab_parallel_cross_entropy"):
            build_spec_provider(options)

    def test_unrelated_settings_do_not_conflict(self):
        options = _options(
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl="native",
            operation_backends={"layer_norm": "torch"},
        )
        assert build_spec_provider(options).layer_norm(rms_norm=False) is WrappedTorchNorm


def test_every_operation_is_declared_by_exactly_one_family():
    assert {op.family for op in operations()} == {"norm", "linear", "attention", "moe", "loss"}
