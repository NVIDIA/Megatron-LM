# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Map Megatron trainer arguments to application-owned Resource attributes."""

import os
from importlib.metadata import PackageNotFoundError, version

import torch
from nemo.lens.resources import detect_gpu, detect_kubernetes, detect_slurm
from nemo.lens.resources.attributes import get_otel_resource_attributes
from nemo.lens.resources.slurm import derive_nv_dl_run_uuid
from nemo.lens.semconv import (
    NV_DL_LOCAL_RANK,
    NV_DL_PROVIDER_NAME,
    NV_DL_RANK,
    NV_DL_ROLE,
    NV_DL_RUN_UUID,
    NV_DL_SOFTWARE_CUDA,
    NV_DL_SOFTWARE_NCCL,
    NV_DL_SOFTWARE_TORCH,
    NV_DL_SOFTWARE_TRANSFORMER_ENGINE,
    NV_DL_TOPOLOGY_SIZE_DP,
    NV_DL_TOPOLOGY_SIZE_PP,
    NV_DL_TOPOLOGY_SIZE_TP,
    NV_DL_TRAINING_CONFIG_GLOBAL_BATCH_SIZE,
    NV_DL_TRAINING_CONFIG_MICRO_BATCH_SIZE,
    NV_DL_TRAINING_CONFIG_OPTIMIZER,
    NV_DL_TRAINING_CONFIG_RECOMPUTE_GRANULARITY,
    NV_DL_TRAINING_CONFIG_SEQUENCE_LENGTH,
    NV_DL_TRAINING_TARGET_TRAIN_ITERS,
    NV_DL_TRAINING_TARGET_TRAIN_SAMPLES,
    NV_DL_TRAINING_TARGET_TRAIN_TOKENS,
    NV_DL_WORLD_SIZE,
)
from nemo.lens.semconv.encoding import compose_attributes
from nemo.lens.semconv.resources import normalize_resource_attributes

_NV_MCORE_VERSION = 'nv.mcore.version'
_LOCAL_RESOURCE_ATTRIBUTES = frozenset({'host.name', 'process.pid', 'host.gpu.count'})

_ARGUMENT_ATTRIBUTES = (
    (NV_DL_RANK, 'rank'),
    (NV_DL_WORLD_SIZE, 'world_size'),
    (NV_DL_LOCAL_RANK, 'local_rank'),
    (NV_DL_TOPOLOGY_SIZE_TP, 'tensor_model_parallel_size'),
    (NV_DL_TOPOLOGY_SIZE_PP, 'pipeline_model_parallel_size'),
    (NV_DL_TOPOLOGY_SIZE_DP, 'data_parallel_size'),
)

_TRAINING_ARGUMENT_ATTRIBUTES = (
    (NV_DL_TRAINING_CONFIG_GLOBAL_BATCH_SIZE, 'global_batch_size'),
    (NV_DL_TRAINING_CONFIG_MICRO_BATCH_SIZE, 'micro_batch_size'),
    (NV_DL_TRAINING_CONFIG_SEQUENCE_LENGTH, 'seq_length'),
    (NV_DL_TRAINING_CONFIG_OPTIMIZER, 'optimizer'),
    (NV_DL_TRAINING_CONFIG_RECOMPUTE_GRANULARITY, 'recompute_granularity'),
    (NV_DL_TRAINING_TARGET_TRAIN_ITERS, 'train_iters'),
    (NV_DL_TRAINING_TARGET_TRAIN_SAMPLES, 'train_samples'),
)


def _set(attrs, key, value):
    if value is not None and value != '':
        attrs[key] = value


def _safe_product(*values):
    result = 1
    for value in values:
        if value is None:
            return None
        try:
            result *= int(value)
        except (TypeError, ValueError):
            return None
    return result


def _distribution_version(*package_names):
    for package_name in package_names:
        try:
            return version(package_name)
        except PackageNotFoundError:
            continue
        except Exception:
            # Package metadata is optional telemetry context and must not make
            # trainer startup fail when an installation has malformed metadata.
            continue
    return None


def _software_attributes():
    attrs = {}
    _set(attrs, NV_DL_SOFTWARE_TORCH, getattr(torch, '__version__', None))
    _set(attrs, NV_DL_SOFTWARE_CUDA, getattr(getattr(torch, 'version', None), 'cuda', None))
    try:
        nccl_version = torch.cuda.nccl.version()
    except Exception:
        pass
    else:
        if isinstance(nccl_version, (tuple, list)):
            nccl_version = '.'.join(str(part) for part in nccl_version)
        _set(attrs, NV_DL_SOFTWARE_NCCL, str(nccl_version))

    _set(
        attrs,
        NV_DL_SOFTWARE_TRANSFORMER_ENGINE,
        _distribution_version('transformer-engine', 'transformer_engine'),
    )
    try:
        from megatron.core.package_info import __version__ as mcore_version
    except Exception:
        pass
    else:
        _set(attrs, _NV_MCORE_VERSION, mcore_version)
    return attrs


def build_telemetry_resource_attrs(args, *, include_training: bool = True):
    """Return Megatron-owned Resource defaults derived from trainer arguments."""
    attrs = {NV_DL_PROVIDER_NAME: 'mcore'}
    for attribute, argument in _ARGUMENT_ATTRIBUTES:
        _set(attrs, attribute, getattr(args, argument, None))

    if not include_training:
        attrs.update(_software_attributes())
        return attrs

    for attribute, argument in _TRAINING_ARGUMENT_ATTRIBUTES:
        _set(attrs, attribute, getattr(args, argument, None))

    train_samples = attrs.get(NV_DL_TRAINING_TARGET_TRAIN_SAMPLES)
    if train_samples is None:
        train_samples = _safe_product(
            getattr(args, 'train_iters', None), getattr(args, 'global_batch_size', None)
        )
        _set(attrs, NV_DL_TRAINING_TARGET_TRAIN_SAMPLES, train_samples)

    train_tokens = getattr(args, 'train_tokens', None)
    if train_tokens is None:
        train_tokens = _safe_product(train_samples, getattr(args, 'seq_length', None))
    _set(attrs, NV_DL_TRAINING_TARGET_TRAIN_TOKENS, train_tokens)
    attrs.update(_software_attributes())
    return attrs


def compose_telemetry_resource_attrs(args, *, include_training: bool = True):
    """Resolve the application-owned Resource map for one trainer process.

    Inherited application identity replaces Megatron defaults. Slurm and
    Kubernetes detection then replace inherited values, matching Lens provider
    precedence. Process-local host fields are left for the trainer's provider,
    while inherited GPU identity wins over one local detection attempt.
    """
    current = normalize_resource_attributes(get_otel_resource_attributes())
    defaults = build_telemetry_resource_attrs(args, include_training=include_training)
    # Plain torchrun has no NVRx carrier. Derive its attempt identity here,
    # after worker spawn supplies TORCHELASTIC_RESTART_COUNT. A launcher-owned
    # UUID (including NVRx's cycle identity) retains precedence via current.
    _set(defaults, NV_DL_RUN_UUID, derive_nv_dl_run_uuid())
    defaults.update(detect_gpu(local_rank=getattr(args, 'local_rank', None)))
    service_name = (
        _nonempty_text(getattr(args, 'otel_service_name', None))
        or _nonempty_text(os.environ.get('OTEL_SERVICE_NAME'))
        or _nonempty_text(current.get('service.name'))
        or 'megatron-lm'
    )
    overrides = detect_slurm()
    overrides.update(detect_kubernetes())
    if include_training:
        overrides[NV_DL_ROLE] = 'trainer'
    overrides['service.name'] = service_name
    resolved = compose_attributes(current, defaults=defaults, overrides=overrides)

    for name in _LOCAL_RESOURCE_ATTRIBUTES:
        resolved.pop(name, None)

    return resolved


def _nonempty_text(value):
    return value if isinstance(value, str) and value.strip() else None
