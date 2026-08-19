# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The one place a BackendSpecProvider is built."""

from __future__ import annotations

from typing import Callable, Mapping

from megatron.core.ops.operations import Operation
from megatron.core.ops.options import BackendOptions
from megatron.core.ops.spec_provider import BackendSpecProvider, compose

__all__ = ["available_backends", "build_spec_provider", "get_backend", "get_backend_spec_provider"]


def _local(options: BackendOptions) -> BackendSpecProvider:
    from megatron.core.ops.providers.local import LocalSpecProvider

    return LocalSpecProvider()


def _transformer_engine(options: BackendOptions) -> BackendSpecProvider:
    from megatron.core.ops.providers.transformer_engine import TESpecProvider

    return TESpecProvider()


def _inference_optimized(options: BackendOptions) -> BackendSpecProvider:
    from megatron.core.ops.providers.inference import InferenceSpecProvider

    return InferenceSpecProvider()


def _torch_norm(options: BackendOptions) -> object:
    from megatron.core.ops.norm.reference import TorchNormBackend

    return TorchNormBackend()


def _apex_norm(options: BackendOptions) -> object:
    from megatron.core.ops.norm.apex import ApexNormBackend

    return ApexNormBackend()


def _te_norm(options: BackendOptions) -> object:
    from megatron.core.ops.norm.transformer_engine import TENormBackend

    return TENormBackend()


def _megatron_cross_entropy(options: BackendOptions) -> object:
    from megatron.core.ops.loss.cross_entropy import MegatronCrossEntropyBackend

    return MegatronCrossEntropyBackend()


def _megatron_fused_cross_entropy(options: BackendOptions) -> object:
    from megatron.core.ops.loss.fused_cross_entropy import MegatronFusedCrossEntropyBackend

    return MegatronFusedCrossEntropyBackend()


def _te_cross_entropy(options: BackendOptions) -> object:
    from megatron.core.ops.loss.transformer_engine import TECrossEntropyBackend

    return TECrossEntropyBackend(cuda_graph_capturable=options.cuda_graph_impl == "full_iteration")


#: Every named backend, and how to build it. A backend that implements the whole protocol can
#: serve as a base; the rest own a single operation. Each factory takes the options so it can
#: bind anything it must decide up front, and nothing here is consulted after the model is built.
#:
#: Adding a backend means adding its module under megatron/core/ops/<operation>/ and one entry
#: here. No existing branch changes, and no call site changes.
_BACKENDS: Mapping[str, Callable[[BackendOptions], object]] = {
    # Complete backends, selectable through --transformer-impl.
    "local": _local,
    "transformer_engine": _transformer_engine,
    "inference_optimized": _inference_optimized,
    # Single-operation backends, selectable through --op-backend.
    "torch": _torch_norm,
    "apex": _apex_norm,
    "te_norm": _te_norm,
    "megatron_cross_entropy": _megatron_cross_entropy,
    "megatron_fused_cross_entropy": _megatron_fused_cross_entropy,
    "te_cross_entropy": _te_cross_entropy,
}

_BASE_BACKENDS = ("local", "transformer_engine", "inference_optimized")

#: How the pre-existing cross entropy settings map onto the operation they select.
_CROSS_ENTROPY_FUSION = {"native": "megatron_fused_cross_entropy", "te": "te_cross_entropy"}


def available_backends() -> tuple[str, ...]:
    """Every backend name that --op-backend accepts."""
    return tuple(_BACKENDS)


def _build_backend(
    name: str, options: BackendOptions, *, operation: Operation | None = None
) -> object:
    factory = _BACKENDS.get(name)
    if factory is None:
        where = f" for operation '{operation}'" if operation is not None else ""
        raise ValueError(
            f"Unknown backend '{name}'{where}. Available backends: {', '.join(_BACKENDS)}"
        )
    return factory(options)


def _base_provider(options: BackendOptions) -> BackendSpecProvider:
    """Select the backend that supplies every operation nothing else claims."""
    if options.transformer_impl not in _BASE_BACKENDS:
        raise ValueError(
            f"unknown transformer_impl='{options.transformer_impl}'. "
            f"Valid choices: {', '.join(_BASE_BACKENDS)}"
        )
    provider = _build_backend(options.transformer_impl, options)
    if options.use_kitchen:
        from megatron.core.ops.providers.kitchen import kitchen_provider

        provider = kitchen_provider(
            provider,
            use_kitchen_attention=options.use_kitchen_attention,
            kitchen_attention_backend=options.kitchen_attention_backend,
        )
    return provider  # type: ignore[return-value]


def _legacy_operation_backends(options: BackendOptions) -> dict[Operation, str]:
    """Translate settings that predate --op-backend into operation choices."""
    owners: dict[Operation, str] = {}
    if options.cross_entropy_loss_fusion:
        impl = options.cross_entropy_fusion_impl
        if impl not in _CROSS_ENTROPY_FUSION:
            raise ValueError(
                f"Unknown cross_entropy_fusion_impl='{impl}'. "
                f"Valid choices: {', '.join(_CROSS_ENTROPY_FUSION)}"
            )
        owners[Operation.VOCAB_PARALLEL_CROSS_ENTROPY] = _CROSS_ENTROPY_FUSION[impl]
    return owners


def _operation_owners(options: BackendOptions) -> dict[Operation, object]:
    """Resolve every operation an explicit selector claims, rejecting two claims on one slot."""
    legacy = _legacy_operation_backends(options)
    explicit = dict(options.operation_backends)

    contested = sorted(str(operation) for operation in set(legacy) & set(explicit))
    if contested:
        raise ValueError(
            "Two settings select a backend for the same operation: "
            + ", ".join(
                f"'{operation}' (from an existing setting and from --op-backend)"
                for operation in contested
            )
            + ". Keep only one of them."
        )

    return {
        operation: _build_backend(name, options, operation=operation)
        for operation, name in {**legacy, **explicit}.items()
    }


def build_spec_provider(options: BackendOptions) -> BackendSpecProvider:
    """Build the provider for ``options``: one base backend, then explicit operation owners.

    Selection order is fixed and total: the base backend answers everything, settings that
    predate --op-backend take over the operations they name, and --op-backend takes over last.
    Everything an optional dependency needs is checked here, while the model is being built.
    """
    return compose(_base_provider(options), _operation_owners(options))


def get_backend_spec_provider(
    config: object, *, transformer_impl: str | None = None
) -> BackendSpecProvider:
    """Build the provider a TransformerConfig asks for."""
    return build_spec_provider(
        BackendOptions.from_config(config, transformer_impl=transformer_impl)
    )


def get_backend(transformer_impl: str, **options) -> BackendSpecProvider:
    """Build the provider for a base backend name, with no config to read."""
    return build_spec_provider(BackendOptions(transformer_impl=transformer_impl, **options))
