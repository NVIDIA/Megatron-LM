# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare gradients using the same TP partitions as distributed checkpoints."""

from contextlib import contextmanager
from dataclasses import replace

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory


def parameter_gradient_shards(module, cp_group=None, tp_group=None):
    """Return canonical gradient sections, reducing CP and replicated SP gradients.

    Checkpoint factories describe interleaved Q/K/V projection partitions too,
    so TP parity does not incorrectly concatenate a whole fused projection.
    Collect all gradients before checking values to avoid rank-asymmetric exits
    while peers are still waiting in a parameter collective.
    """
    gradients = {}
    parameters = list(module.named_parameters())
    present = torch.tensor(
        [parameter.grad is not None for _, parameter in parameters],
        device=parameters[0][1].device,
        dtype=torch.int32,
    )
    minimum, maximum = present.clone(), present.clone()
    torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    assert torch.equal(minimum, maximum), "parameter gradient participation differs across ranks"
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        gradient = param.grad.detach().float().clone()
        if cp_group is not None and cp_group.size() > 1:
            torch.distributed.all_reduce(gradient, group=cp_group)
        if (
            tp_group is not None
            and tp_group.size() > 1
            and getattr(param, "sequence_parallel", False)
        ):
            torch.distributed.all_reduce(gradient, group=tp_group)
        gradients[name] = gradient

    result = {}

    def collect(value):
        if isinstance(value, ShardedTensor):
            result[value.key] = (value.data.detach().clone(), value.global_offset)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    for name, value in module.sharded_state_dict().items():
        if name not in gradients:
            continue
        if isinstance(value, ShardedTensorFactory):
            collect(replace(value, data=gradients[name]).build())
        elif isinstance(value, ShardedTensor):
            collect(replace(value, data=gradients[name], dtype=torch.float32))
    return result


def assert_parameter_gradient_shards_close(reference, actual, *, atol, rtol):
    """Compare each TP partition against the corresponding serial section."""
    assert reference.keys() == actual.keys()
    for name, (gradient, offset) in actual.items():
        baseline, baseline_offset = reference[name]
        assert not any(baseline_offset), "reference must be unsharded"
        selection = tuple(slice(start, start + size) for start, size in zip(offset, gradient.shape))
        torch.testing.assert_close(
            gradient,
            baseline[selection],
            atol=atol,
            rtol=rtol,
            msg=lambda msg, param_name=name: f"Mismatch in parameter gradient {param_name}: {msg}",
        )


@contextmanager
def synchronized_assertions():
    """Propagate a numerical assertion to every rank before collective teardown."""
    error = None
    try:
        yield
    except AssertionError as caught:
        error = caught
    failed = torch.tensor(int(error is not None), device="cuda", dtype=torch.int32)
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if failed.item():
        if error is not None:
            raise error
        raise AssertionError("parallel parity comparison failed on another distributed rank")
