# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Selection without Liger installed, and GPU parity with the optional Liger backend."""

import builtins
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.backends import get_backend
from megatron.core.ops import _backends
from megatron.core.ops.loss import backends
from megatron.core.ops.loss.cross_entropy import vocab_parallel_cross_entropy
from tests.unit_tests.test_utilities import Utils

_ADAPTER = "megatron.core.ops.loss.liger_cross_entropy"
_LIBRARY = "liger_kernel.megatron"


@pytest.mark.parametrize("fusion,impl", [(False, "liger"), (True, "native")])
def test_unselected_never_imports_liger(monkeypatch, fusion, impl):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("liger_kernel") and name != _ADAPTER
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        _backends, "import_module", Mock(side_effect=AssertionError("unexpected import"))
    )
    target = backends.select_cross_entropy(fusion, impl)
    assert target is (backends._fused_ce if fusion else backends.unfused_cross_entropy)
    _backends.import_module.assert_not_called()


@pytest.mark.parametrize("backend", ["local", "transformer_engine", "inference_optimized"])
def test_provider_selects_liger_cross_entropy_independently(backend):
    pytest.importorskip(_LIBRARY)
    from megatron.core.ops.loss.liger_cross_entropy import liger_cross_entropy

    provider = get_backend(
        backend, cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="liger"
    )
    assert provider.vocab_parallel_cross_entropy() is liger_cross_entropy


@pytest.fixture(scope="module", params=[None, 1, 2, 4], ids=["local", "tp1", "tp2", "tp4"])
def ce_group(request):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    size = request.param or 1
    if Utils.world_size < size or Utils.world_size % size:
        pytest.skip(f"needs a world size divisible by TP={size}")
    with Utils.interleaved_group(size) as group:
        yield (None if request.param is None else group), group, Utils.rank % (
            Utils.world_size // size
        )


def _with_layout(tensor, layout):
    if layout == "transpose":
        return tensor.transpose(0, 1).contiguous().transpose(0, 1)
    if layout == "strided":
        storage = torch.empty(
            *tensor.shape[:-1], tensor.size(-1) * 2, device=tensor.device, dtype=tensor.dtype
        )
        view = storage[..., ::2]
        view.copy_(tensor)
        return view
    return tensor.contiguous()


@pytest.mark.parametrize(
    "dtype,layout,vocab_shard,all_ignored",
    [
        (torch.float32, "contiguous", 37, False),
        (torch.bfloat16, "contiguous", 1024, False),
        (torch.float32, "transpose", 37, False),
        (torch.bfloat16, "strided", 37, False),
        (torch.float32, "strided", 37, True),
    ],
)
def test_loss_and_gradients_match_references(
    ce_group, dtype, layout, vocab_shard, all_ignored, monkeypatch
):
    pytest.importorskip(_LIBRARY)
    target = backends.select_cross_entropy(True, "liger")
    group, reference_group, data_id = ce_group
    size, rank = reference_group.size(), reference_group.rank()
    generator = torch.Generator().manual_seed(912 + data_id)
    full = torch.randn(11, 3, vocab_shard * size, generator=generator).to("cuda", dtype)
    labels = torch.randint(vocab_shard * size, (11, 3), generator=generator).cuda()
    labels[0, 0], labels[0, 1] = 0, vocab_shard * size - 1
    labels[1::2] = -100
    if all_ignored:
        labels.fill_(-100)
    labels = _with_layout(labels, layout)
    upstream = _with_layout(torch.randn(11, 3, generator=generator).cuda(), "strided")
    upstream[0, 0] = 0
    start, end = rank * vocab_shard, (rank + 1) * vocab_shard
    logits = _with_layout(full[..., start:end].clone(), layout).detach().requires_grad_(True)
    native_logits = logits.detach().contiguous().clone().requires_grad_(True)
    reference_logits = full.float().detach().requires_grad_(True)
    reference = F.cross_entropy(
        reference_logits.reshape(-1, vocab_shard * size), labels.reshape(-1), reduction="none"
    ).view_as(labels)
    reference_grad = torch.autograd.grad(reference, reference_logits, upstream)[0][..., start:end]

    original_reduce = torch.distributed.all_reduce

    def checked_reduce(tensor, *args, **kwargs):
        assert group is not None and kwargs.get("group") is group
        return original_reduce(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.distributed, "all_reduce", checked_reduce)
        loss = target(logits, labels, group)
        torch.testing.assert_close(logits, full[..., start:end], atol=0, rtol=0)
        gradient = torch.autograd.grad(loss, logits, upstream)[0]
        torch.testing.assert_close(logits, full[..., start:end], atol=0, rtol=0)
    native = vocab_parallel_cross_entropy(
        native_logits, labels.contiguous(), tp_group=reference_group
    ) * (labels != -100)
    native_grad = torch.autograd.grad(native, native_logits, upstream)[0]
    assert loss.shape == labels.shape and loss.dtype == torch.float32
    assert gradient.dtype == dtype
    torch.testing.assert_close(loss, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(loss, native, atol=2e-6, rtol=2e-6)
    tolerance = dict(atol=2e-7, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-5, rtol=1e-2)
    torch.testing.assert_close(gradient, reference_grad.to(dtype), **tolerance)
    torch.testing.assert_close(gradient, native_grad, **tolerance)
