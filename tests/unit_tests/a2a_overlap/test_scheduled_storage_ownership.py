# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for schedule bindings around externally owned EP allocations."""

import pytest
import torch

from megatron.core.pipeline_parallel import combined_1f1b_tensor_release as release


@pytest.mark.parametrize("cross_stream", [False, True])
def test_external_storage_is_preserved_while_owned_storage_is_released(monkeypatch, cross_stream):
    # Isolate allocation ownership from CUDA transport. The same policy receives real CUDA
    # tensors in ScheduleNode; stream/event ordering has separate GPU regression coverage.
    def tensors(value):
        values = (value,) if isinstance(value, torch.Tensor) else value
        return iter(values or ())

    monkeypatch.setattr(release, "_iter_unique_cuda_tensors", tensors)
    owner = object()
    consumer = object() if cross_stream else owner
    manager = release.Combined1F1BTensorRelease()
    external = torch.arange(8)
    ordinary = torch.ones(8)
    output = torch.ones(4)
    expected = external.clone()
    manager.consume_inputs_and_publish_outputs(
        (), (external, ordinary), stream=owner, node="producer", release_consumed=False
    )
    manager.consume_inputs_and_publish_outputs(
        (external, ordinary),
        output,
        stream=consumer,
        node="consumer",
        release_consumed=True,
        preserve_storage=lambda value: value is external,
    )
    torch.testing.assert_close(external, expected)
    assert id(external) not in manager._owners
    assert id(ordinary) not in manager._owners
    assert manager._owners[id(output)].tensor is output
    if cross_stream:
        assert ordinary.untyped_storage().nbytes() > 0
        assert [entry.tensor for entry in manager._pending[owner]] == [ordinary]
        manager.drain(owner)
    assert ordinary.untyped_storage().nbytes() == 0
    assert external.untyped_storage().nbytes() == expected.untyped_storage().nbytes()
    assert not manager._pending
