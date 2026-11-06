# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fail closed on missing or changed state across level-two sleep."""
from fnmatch import filter as matching

from .c8_fingerprint import checksum


def capture(worker):
    """Capture every selected parameter and native restored buffer, by name."""
    model = worker.model_runner.model
    draft = worker.get_draft_model()
    parameters = worker._get_sleep_parameters(model)
    patterns = worker.model_config.sleep_preserve_parameter_names
    if not patterns or '*.bias_vl' not in patterns:
        raise RuntimeError('L2 sleep requires *.bias_vl preservation')
    if draft is not None and 'draft.*' not in patterns:
        raise RuntimeError('L2 sleep requires draft.* preservation with DSpark')
    names = set()
    for pattern in patterns:
        found = matching(parameters, pattern)
        if not found:
            raise RuntimeError(f'L2 preservation pattern has no tensor: {pattern}')
        names.update(found)
    tensors = {f'parameter:{name}': parameters[name] for name in sorted(names)}
    tensors.update((f'buffer:{name}', value) for name, value in model.named_buffers())
    if draft is not None:
        tensors.update((f'draft_buffer:{name}', value) for name, value in draft.named_buffers())
    return {name: checksum(tensor) for name, tensor in tensors.items()}


def verify(worker, expected):
    actual = capture(worker)
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise RuntimeError(f'L2 wake tensor set changed: missing={missing}, extra={extra}')
    changed = [name for name in expected if actual[name] != expected[name]]
    if changed:
        raise RuntimeError(f'L2 wake fingerprint mismatch: {changed}')


def verify_backups(worker, expected):
    """Require native copies for every selected CUDA parameter and every buffer."""
    parameters = worker._get_sleep_parameters(worker.model_runner.model)
    names = {key.removeprefix('parameter:') for key in expected if key.startswith('parameter:')}
    wanted = {name for name in names if not parameters[name].is_cpu}
    saved = worker._sleep_saved_parameters
    if set(saved) != wanted:
        raise RuntimeError(
            f'L2 parameter backups differ: expected={sorted(wanted)}, saved={sorted(saved)}'
        )
    for prefix, attr in (
        ('buffer:', '_sleep_saved_buffers'),
        ('draft_buffer:', '_sleep_saved_draft_buffers'),
    ):
        wanted = {key.removeprefix(prefix) for key in expected if key.startswith(prefix)}
        actual = getattr(worker, attr, {})
        if set(actual) != wanted:
            raise RuntimeError(
                f'L2 {prefix} backups differ: expected={sorted(wanted)}, saved={sorted(actual)}'
            )
