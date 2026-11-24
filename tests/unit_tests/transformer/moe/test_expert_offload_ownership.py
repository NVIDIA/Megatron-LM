# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from https://github.com/NVIDIA-NeMo/nv-mistralai-megatron/pull/797
# Source revision: aeab73be76294f8930377aae1e853b4e930e8dd0
# License: BSD-3-Clause; original NVIDIA notices retained (see LICENSE).
"""CPU regressions for effective expert precision and saved-storage ownership."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.transformer.moe.experts as experts_module
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    ChunkOffloadHandler,
    OffloadTensorGroup,
    OffloadTensorPool,
    PipelineOffloadManager,
)
from megatron.core.transformer.moe.experts import TEGroupedMLP


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("override", [None, False, True])
@pytest.mark.parametrize("previous_policy", [False, True])
def test_fc1_offload_save_policy_uses_runtime_precision(
    monkeypatch, ambient, override, previous_policy
):
    """A BF16 invocation saves the whole input and restores prior module state."""
    observed = []

    class Linear:
        save_original_input = previous_policy

        def will_execute_quantized(self, *, is_context_quantized):
            assert is_context_quantized is ambient
            return ambient if override is None else override

        def __call__(self, hidden, counts):
            observed.append(self.save_original_input)
            return hidden * 2, None

    linear = Linear()
    owner = SimpleNamespace(offload_expert_fc1=True, linear_fc1=linear)
    monkeypatch.setattr(experts_module, "_quantization_autocast_active", lambda: ambient)
    monkeypatch.setattr(experts_module, "apply_module", lambda module: module)
    monkeypatch.setattr(
        experts_module,
        "set_save_original_input",
        lambda module: setattr(module, "save_original_input", True),
    )
    hidden = torch.ones(8, 4)
    result, bias = TEGroupedMLP._forward_fc1(owner, hidden, [4, 4])
    quantized = ambient if override is None else override
    assert observed == [previous_policy or not quantized]
    assert linear.save_original_input is previous_policy
    torch.testing.assert_close(result, hidden * 2)
    assert bias is None


def test_fc1_save_policy_does_not_leak_between_invocations(monkeypatch):
    """NVFP4 MTP/first-last BF16 calls may share a module with quantized calls."""
    ambient = False
    observed = []

    class Linear:
        save_original_input = False

        def will_execute_quantized(self, *, is_context_quantized):
            return is_context_quantized

        def __call__(self, hidden, counts):
            observed.append(self.save_original_input)
            return hidden, None

    linear = Linear()
    owner = SimpleNamespace(offload_expert_fc1=True, linear_fc1=linear)
    monkeypatch.setattr(experts_module, "_quantization_autocast_active", lambda: ambient)
    monkeypatch.setattr(experts_module, "apply_module", lambda module: module)
    monkeypatch.setattr(
        experts_module,
        "set_save_original_input",
        lambda module: setattr(module, "save_original_input", True),
    )
    for ambient in (False, True, False, True):
        TEGroupedMLP._forward_fc1(owner, torch.ones(8, 4), [4, 4])
        assert linear.save_original_input is False
    assert observed == [True, False, True, False]


def test_fc1_save_policy_restores_after_forward_failure(monkeypatch):
    class Linear:
        save_original_input = False

        def will_execute_quantized(self, *, is_context_quantized):
            return False

        def __call__(self, hidden, counts):
            assert self.save_original_input
            raise RuntimeError("test failure")

    linear = Linear()
    owner = SimpleNamespace(offload_expert_fc1=True, linear_fc1=linear)
    monkeypatch.setattr(experts_module, "_quantization_autocast_active", lambda: False)
    monkeypatch.setattr(experts_module, "apply_module", lambda module: module)
    monkeypatch.setattr(
        experts_module,
        "set_save_original_input",
        lambda module: setattr(module, "save_original_input", True),
    )
    with pytest.raises(RuntimeError, match="test failure"):
        TEGroupedMLP._forward_fc1(owner, torch.ones(8, 4), [4, 4])
    assert linear.save_original_input is False


def test_fc1_without_offload_does_not_query_quantization(monkeypatch):
    def forbidden_query():
        raise AssertionError("Offload-disabled execution must not query quantization")

    owner = SimpleNamespace(
        offload_expert_fc1=False, linear_fc1=lambda hidden, counts: (hidden, None)
    )
    monkeypatch.setattr(experts_module, "_quantization_autocast_active", forbidden_query)
    monkeypatch.setattr(experts_module, "apply_module", lambda module: module)
    hidden = torch.ones(8, 4)
    result, _ = TEGroupedMLP._forward_fc1(owner, hidden, [4, 4])
    assert result is hidden


@pytest.mark.parametrize(
    "alias_kind",
    ["slice", "detached", "strided", "byte_view", "wrapped_view", "unrelated", "copied"],
)
@pytest.mark.parametrize("offload", [False, True])
def test_bulk_offload_preserves_storage_of_uncopied_saved_views(
    monkeypatch, caplog, alias_kind, offload
):
    """A large released parent must outlive every small resident view in its group."""

    class StorageWrapper(torch.Tensor):
        def untyped_storage(self):
            raise RuntimeError("Use physical data tensors instead of wrapper storage")

        def get_data_tensors(self):
            return (self.as_subclass(torch.Tensor),)

    parent = torch.arange(64, dtype=torch.float32)
    aliases = {
        "slice": lambda: parent[2:6],
        "detached": lambda: parent[2:6].detach(),
        "strided": lambda: parent[::16],
        "byte_view": lambda: parent[2:4].view(torch.uint8),
        "wrapped_view": lambda: parent[2:6].as_subclass(StorageWrapper),
        "unrelated": lambda: torch.ones(4),
        "copied": lambda: parent[8:],
    }
    alias = aliases[alias_kind]()
    group = SimpleNamespace(
        _name="expert_fc1", _tensors={0: parent, 1: alias}, is_offload_margin=False
    )
    chunk = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    chunk._groups_to_offload = [group]
    monkeypatch.setattr(ChunkOffloadHandler, "_warned_retained_storage", False)
    chunk._groups_to_reload = []
    chunk.should_bulk_offload = lambda selected: offload
    chunk.tensor_need_offloading_checker = (
        lambda tensor: isinstance(tensor, torch.Tensor) and tensor.numel() >= 32
    )

    def copy_eligible(selected):
        for key, tensor in list(selected._tensors.items()):
            if chunk.tensor_need_offloading_checker(tensor):
                selected._tensors[key] = (tensor.device, tensor.clone(), False, None)

    chunk.bulk_offload_group = copy_eligible
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    chunk.bulk_offload("expert_fc1", [parent])
    retained = not offload or alias_kind not in ("unrelated", "copied")
    assert parent.untyped_storage().nbytes() == (256 if retained else 0)
    assert chunk._groups_to_offload == []
    assert chunk._groups_to_reload == [group]

    guarded = offload and alias_kind not in ("unrelated", "copied")
    assert group.retained_storage_bytes == (256 if guarded else 0)
    assert ("Activation offload retained" in caplog.text) is guarded
    if guarded:
        caplog.clear()
        # Another microbatch handler refreshes bytes without repeating the warning.
        later_chunk = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
        later_chunk._groups_to_reload = []
        later_chunk.should_bulk_offload = chunk.should_bulk_offload
        later_chunk.tensor_need_offloading_checker = chunk.tensor_need_offloading_checker
        later_chunk.bulk_offload_group = chunk.bulk_offload_group
        later = torch.ones(96)
        group._tensors = {0: later, 1: later[:4]}
        later_chunk._groups_to_offload = [group]
        later_chunk.bulk_offload("expert_fc1", [later, later])
        assert group.retained_storage_bytes == 384
        assert "Activation offload retained" not in caplog.text


@pytest.mark.parametrize("aliases", [0, 1, 2])
@pytest.mark.parametrize("retained_bytes", [0, 256])
def test_collect_offload_info_preserves_runtime_retained_storage_bytes(aliases, retained_bytes):
    """Discovery counts transfers without predicting which storage the guard retains."""
    parent = torch.ones(64)
    group = OffloadTensorGroup.__new__(OffloadTensorGroup)
    group._tensors = {0: parent, **{i + 1: parent[:4] for i in range(aliases)}}
    group._offload_info_collected = False
    group.total_offload_bytes = 0
    group.total_tensor_count = 0
    group.retained_storage_bytes = retained_bytes
    chunk = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    chunk.tensor_need_offloading_checker = lambda tensor: tensor.numel() >= 32
    chunk._collect_offload_info(group)
    assert group.total_offload_bytes == 256
    assert group.retained_storage_bytes == retained_bytes
    chunk._collect_offload_info(group)
    assert group.total_offload_bytes == 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for storage ownership")
@pytest.mark.parametrize("saved_alias", [False, True])
def test_real_offload_storage_guard_backward(saved_alias):
    """Real D2H/H2D and autograd preserve views and report retained GPU bytes."""
    PipelineOffloadManager.reset_instance()
    try:
        chunk = ChunkOffloadHandler(32, OffloadTensorPool(device="cpu", pin_memory=True))
        group = OffloadTensorGroup("expert_fc1")
        chunk.offload_groups = [group]
        chunk._groups_to_offload = [group]
        chunk._offloaded_group_index = 1
        x = torch.ones(65536, device="cuda", requires_grad=True)
        parent = x * 2
        with torch.autograd.graph.saved_tensors_hooks(chunk.tensor_push, chunk.tensor_pop):
            loss = parent.square().sum()
            if saved_alias:
                loss = loss + parent[:4].square().sum()
        chunk.d2h_stream.wait_stream(torch.cuda.current_stream())
        chunk.bulk_offload("expert_fc1", [parent])
        expected_bytes = 65536 * x.element_size() if saved_alias else 0
        assert parent.untyped_storage().nbytes() == expected_bytes
        assert group.retained_storage_bytes == expected_bytes
        loss.backward()
        expected_grad = torch.full_like(x, 8)
        if saved_alias:
            expected_grad[:4] = 16
        torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        PipelineOffloadManager.reset_instance()
