# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Operations from different backends compose into one provider."""

from types import SimpleNamespace

import pytest

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.ops import BackendOptions, Operation, build_spec_provider, compose, get_backend
from megatron.core.ops.norm.reference import TorchNormBackend, WrappedTorchNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear

_needs_te = pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required")


def _options(**kwargs):
    return BackendOptions(transformer_impl="local", **kwargs)


class TestCompose:
    def test_owner_takes_over_only_its_operation(self):
        base = get_backend("local")
        composed = compose(base, {Operation.LAYER_NORM: TorchNormBackend()})

        assert composed.layer_norm(rms_norm=False) is WrappedTorchNorm
        assert composed.column_parallel_linear() is ColumnParallelLinear

    def test_base_is_not_modified(self):
        base = get_backend("local")
        before = base.layer_norm(rms_norm=False)
        compose(base, {Operation.LAYER_NORM: TorchNormBackend()})
        assert base.layer_norm(rms_norm=False) is before

    def test_composition_costs_nothing_at_call_time(self):
        """The owner's bound method is attached directly, with no forwarding wrapper."""
        owner = TorchNormBackend()
        composed = compose(get_backend("local"), {Operation.LAYER_NORM: owner})
        assert composed.layer_norm.__func__ is TorchNormBackend.layer_norm
        assert composed.layer_norm.__self__ is owner

    def test_derived_query_follows_the_override(self):
        """fuse_layernorm_and_linear() is derived, so it must see the new owner's answer."""

        class _FusedNormLinear:
            def column_parallel_layer_norm_linear(self):
                return _FusedNormLinear

        base = get_backend("local")
        assert base.fuse_layernorm_and_linear() is False

        composed = compose(base, {Operation.COLUMN_PARALLEL_LAYER_NORM_LINEAR: _FusedNormLinear()})
        assert composed.column_parallel_layer_norm_linear() is _FusedNormLinear
        assert composed.fuse_layernorm_and_linear() is True

    def test_owner_without_the_operation_is_rejected(self):
        with pytest.raises(ValueError, match="does not implement core_attention"):
            compose(get_backend("local"), {Operation.CORE_ATTENTION: TorchNormBackend()})

    def test_no_owners_returns_the_base_unchanged(self):
        base = get_backend("local")
        assert compose(base, {}) is base


class TestOperationOverrides:
    def test_override_by_name(self):
        provider = build_spec_provider(_options(operation_backends={"layer_norm": "torch"}))
        assert provider.layer_norm(rms_norm=False) is WrappedTorchNorm

    @_needs_te
    def test_override_reaches_across_backends(self):
        """A Transformer Engine model can take one operation from somewhere else."""
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

        provider = build_spec_provider(
            BackendOptions(
                transformer_impl="transformer_engine", operation_backends={"layer_norm": "torch"}
            )
        )
        assert provider.layer_norm(rms_norm=False) is WrappedTorchNorm
        assert provider.column_parallel_linear() is TEColumnParallelLinear

    def test_unknown_operation_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="Unknown operation 'norm'"):
            _options(operation_backends={"norm": "torch"})

    def test_unknown_backend_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="Unknown backend 'liger' for operation 'layer_norm'"):
            build_spec_provider(_options(operation_backends={"layer_norm": "liger"}))

    def test_overrides_are_copied_from_the_caller(self):
        overrides = {"layer_norm": "torch"}
        options = _options(operation_backends=overrides)
        overrides["layer_norm"] = "apex"
        assert options.operation_backends[Operation.LAYER_NORM] == "torch"

    def test_config_field_is_read(self):
        config = SimpleNamespace(
            transformer_impl="local", op_backend_overrides={"layer_norm": "torch"}
        )
        options = BackendOptions.from_config(config)
        assert options.operation_backends[Operation.LAYER_NORM] == "torch"


class TestConflictingSelectors:
    def test_two_settings_claiming_one_operation_is_an_error(self):
        options = _options(
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl="native",
            operation_backends={"vocab_parallel_cross_entropy": "megatron_cross_entropy"},
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
