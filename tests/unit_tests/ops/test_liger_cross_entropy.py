# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Selection without Liger installed, and GPU parity with the optional Liger backend."""

import builtins
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
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


@pytest.mark.parametrize(
    "fusion,impl",
    [(False, impl) for impl in ("native", "te", "liger")] + [(True, "native"), (True, "te")],
)
@pytest.mark.parametrize("cuda_graph_impl", [None, "full_iteration"])
def test_unselected_never_imports_liger(monkeypatch, fusion, impl, cuda_graph_impl):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("liger_kernel") and name != _ADAPTER
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        _backends, "import_module", Mock(side_effect=AssertionError("unexpected import"))
    )
    monkeypatch.setattr(backends, "is_te_min_version", lambda version: True)
    target = backends.select_cross_entropy(fusion, impl, cuda_graph_impl)
    if not fusion:
        assert target is backends.unfused_cross_entropy
    elif impl == "native":
        assert target is backends._fused_ce
    else:
        assert target.func is backends.te_cross_entropy
        assert target.keywords == {"cuda_graph_capturable": cuda_graph_impl == "full_iteration"}
    _backends.import_module.assert_not_called()


@pytest.mark.parametrize("fusion", [False, True])
def test_unknown_backend_still_fails(monkeypatch, fusion):
    monkeypatch.setattr(
        _backends, "import_module", Mock(side_effect=AssertionError("unexpected import"))
    )
    with pytest.raises(ValueError, match="unknown cross_entropy_fusion_impl"):
        backends.select_cross_entropy(fusion, "not-a-backend")
    _backends.import_module.assert_not_called()


def test_selected_checks_export_without_version_floor(monkeypatch):
    library = ModuleType(_LIBRARY)
    library.LigerMegatronCrossEntropy = Mock()
    library.__version__ = "0.0.0"
    adapter = ModuleType(_ADAPTER)
    adapter.liger_cross_entropy = Mock()
    monkeypatch.setitem(sys.modules, _ADAPTER, adapter)
    importer = Mock(return_value=library)
    monkeypatch.setattr(_backends, "import_module", importer)
    monkeypatch.setattr(
        _backends, "installed_version", Mock(side_effect=AssertionError("no release floor"))
    )
    assert backends.select_cross_entropy(True, "liger") is adapter.liger_cross_entropy
    importer.assert_called_once_with(_LIBRARY)


@pytest.mark.parametrize("backend", ["local", "transformer_engine", "inference_optimized"])
def test_provider_selects_liger_cross_entropy_independently(monkeypatch, backend):
    library = ModuleType(_LIBRARY)
    library.LigerMegatronCrossEntropy = Mock()
    adapter = ModuleType(_ADAPTER)
    adapter.liger_cross_entropy = Mock()
    monkeypatch.setitem(sys.modules, _ADAPTER, adapter)
    monkeypatch.setattr(_backends, "import_module", Mock(return_value=library))
    provider = get_backend(
        backend, cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="liger"
    )
    assert provider.vocab_parallel_cross_entropy() is adapter.liger_cross_entropy


@pytest.mark.parametrize("error_type", [ModuleNotFoundError, ImportError, OSError, RuntimeError])
def test_selected_chains_import_failure(monkeypatch, error_type):
    cause = error_type("failed to load Liger")
    if isinstance(cause, ModuleNotFoundError):
        cause.name = "liger_kernel"
    importer = Mock(side_effect=cause)
    monkeypatch.setattr(_backends, "import_module", importer)
    with pytest.raises(ImportError, match="Liger vocab-parallel cross entropy") as caught:
        backends.select_cross_entropy(True, "liger")
    assert caught.value.__cause__ is cause
    importer.assert_called_once_with(_LIBRARY)
    if isinstance(cause, ModuleNotFoundError):
        assert isinstance(caught.value, ModuleNotFoundError)
        assert caught.value.name == "liger_kernel"


@pytest.mark.parametrize("export", ["missing", "none", "broken"])
def test_selected_requires_usable_export(monkeypatch, export):
    library = ModuleType(_LIBRARY)
    cause = RuntimeError("lazy export failed")
    if export == "none":
        library.LigerMegatronCrossEntropy = None
    elif export == "broken":

        def failed_getattr(name):
            raise cause

        library.__getattr__ = failed_getattr
    monkeypatch.setattr(_backends, "import_module", Mock(return_value=library))
    with pytest.raises(ImportError, match="LigerMegatronCrossEntropy") as caught:
        backends.select_cross_entropy(True, "liger")
    if export == "missing":
        assert isinstance(caught.value.__cause__, AttributeError)
    elif export == "broken":
        assert caught.value.__cause__ is cause


@pytest.fixture
def mocked_adapter(monkeypatch):
    """Load the real adapter with a fake library, without polluting its canonical import."""
    library = ModuleType(_LIBRARY)
    kernel = Mock()
    library.LigerMegatronCrossEntropy = Mock(return_value=kernel)
    monkeypatch.setitem(sys.modules, _LIBRARY, library)
    spec = importlib.util.spec_from_file_location(
        "_test_liger_ce_adapter", Path(backends.__file__).with_name("liger_cross_entropy.py")
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    library.LigerMegatronCrossEntropy.assert_called_once_with(
        ignore_index=-100, reduction="none", label_smoothing=0.0
    )
    return adapter, kernel


@pytest.mark.parametrize("return_argmax", [False, True])
def test_original_argmax_precedes_mutating_forward(mocked_adapter, monkeypatch, return_argmax):
    adapter, kernel = mocked_adapter
    logits = torch.tensor([[[1.0, 5.0, 5.0, 2.0], [9.0, 1.0, 0.0, 9.0]]], requires_grad=True)
    labels = torch.tensor([[1, 0]])
    original_predictions = logits.detach().argmax(dim=-1)
    expected_loss = torch.ones_like(labels, dtype=torch.bfloat16)

    def consume_logits(logits, labels, tp_group):
        assert tp_group is None
        with torch.no_grad():
            logits.zero_()
            logits[..., -1] = 1.0
        return expected_loss

    kernel.side_effect = consume_logits
    monkeypatch.setattr(
        torch.distributed, "all_reduce", Mock(side_effect=AssertionError("None is local"))
    )
    argmax = Mock(wraps=adapter._original_argmax)
    monkeypatch.setattr(adapter, "_original_argmax", argmax)
    assert "return_argmax" in inspect.signature(adapter.liger_cross_entropy).parameters
    result = adapter.liger_cross_entropy(logits, labels, None, return_argmax=return_argmax)
    loss = result[0] if return_argmax else result
    torch.testing.assert_close(loss, expected_loss.float())
    kernel.assert_called_once_with(logits, labels, tp_group=None)
    if return_argmax:
        predictions = result[1]
        assert torch.equal(predictions, original_predictions)
        assert not torch.equal(predictions, logits.argmax(dim=-1))
        assert predictions.dtype == torch.int64 and not predictions.requires_grad
        argmax.assert_called_once()
    else:
        argmax.assert_not_called()


@pytest.fixture(scope="module", params=[None, 1, 2, 4], ids=["local", "tp1", "tp2", "tp4"])
def ce_group(request):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    size = request.param or 1
    if Utils.world_size < size or Utils.world_size % size:
        pytest.skip(f"needs a world size divisible by TP={size}")
    Utils.initialize_distributed()
    # Interleave ranks and vary data between groups: WORLD communication and global
    # rank-based vocabulary offsets must not accidentally pass subgroup tests.
    num_groups = Utils.world_size // size
    group = None
    for offset in range(num_groups):
        ranks = list(range(offset, Utils.world_size, num_groups))
        candidate = torch.distributed.new_group(ranks)
        if Utils.rank in ranks:
            group = candidate
    yield (None if request.param is None else group), group, Utils.rank % num_groups
    torch.distributed.barrier()
    torch.distributed.destroy_process_group(group)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
@pytest.mark.parametrize("vocab_shard", [37, 1024])
@pytest.mark.parametrize("return_argmax", [False, True])
def test_loss_and_gradients_match_references(
    ce_group, dtype, layout, vocab_shard, return_argmax, monkeypatch
):
    pytest.importorskip(_LIBRARY)
    target = backends.select_cross_entropy(True, "liger")
    group, reference_group, data_id = ce_group
    size, rank = reference_group.size(), reference_group.rank()
    generator = torch.Generator().manual_seed(912 + data_id)
    full = torch.randn(11, 3, vocab_shard * size, generator=generator).to("cuda", dtype)
    labels = torch.randint(vocab_shard * size, (11, 3), generator=generator).cuda()
    labels[0, 0], labels[0, 1] = 0, vocab_shard * size - 1
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
        result = target(logits, labels, group, return_argmax=return_argmax)
        loss = result[0] if return_argmax else result
        gradient = torch.autograd.grad(loss, logits, upstream)[0]
    native = vocab_parallel_cross_entropy(
        native_logits, labels.contiguous(), tp_group=reference_group
    )
    native_grad = torch.autograd.grad(native, native_logits, upstream)[0]
    assert loss.shape == labels.shape and loss.dtype == torch.float32
    assert gradient.dtype == dtype
    torch.testing.assert_close(loss, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(loss, native, atol=2e-6, rtol=2e-6)
    tolerance = dict(atol=2e-7, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-5, rtol=1e-2)
    torch.testing.assert_close(gradient, reference_grad.to(dtype), **tolerance)
    torch.testing.assert_close(gradient, native_grad, **tolerance)
    if return_argmax:
        assert torch.equal(result[1], full.argmax(dim=-1))
        assert result[1].dtype == torch.int64 and not result[1].requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_global_argmax_ties_and_original_predictions(ce_group, dtype):
    pytest.importorskip(_LIBRARY)
    target = backends.select_cross_entropy(True, "liger")
    group, reference_group, data_id = ce_group
    size, rank, vocab_shard = reference_group.size(), reference_group.rank(), 17
    full = torch.zeros(4, 2, size * vocab_shard, device="cuda", dtype=dtype)
    full[0, :, 2] = full[0, :, -1] = 9
    full[1, :, -2:] = 7
    # Equal maxima on every shard, and a winner that differs between TP subgroups.
    full[2, :, 1::vocab_shard] = 8
    full[3, :, (data_id + 3) % full.size(-1)] = 6
    labels = full.argmax(dim=-1)
    logits = full[..., rank * vocab_shard : (rank + 1) * vocab_shard].clone().requires_grad_(True)
    loss, predictions = target(logits, labels, group, return_argmax=True)
    assert torch.equal(predictions, labels)
    assert not predictions.requires_grad and predictions.grad_fn is None
    loss.sum().backward()
    assert logits.grad.dtype == dtype


@pytest.mark.parametrize("all_ignored", [False, True])
def test_ignored_labels_match_pytorch_and_masked_native(ce_group, all_ignored):
    pytest.importorskip(_LIBRARY)
    target = backends.select_cross_entropy(True, "liger")
    group, reference_group, data_id = ce_group
    size, rank, shard = reference_group.size(), reference_group.rank(), 37
    generator = torch.Generator().manual_seed(817 + data_id)
    full = torch.randn(5, 2, shard * size, generator=generator).cuda()
    labels = torch.randint(shard * size, (5, 2), generator=generator).cuda()
    labels[::2] = -100
    if all_ignored:
        labels.fill_(-100)
    mask = labels != -100
    logits = full[..., rank * shard : (rank + 1) * shard].clone().requires_grad_(True)
    native_logits = logits.detach().clone().requires_grad_(True)
    ref_logits = full.clone().requires_grad_(True)
    loss = target(logits, labels, group)
    reference = F.cross_entropy(
        ref_logits.reshape(-1, shard * size), labels.reshape(-1), reduction="none"
    ).view_as(labels)
    native = vocab_parallel_cross_entropy(native_logits, labels, tp_group=reference_group)
    torch.testing.assert_close(loss, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(loss, native * mask, atol=2e-6, rtol=2e-6)
    loss.sum().backward()
    reference.sum().backward()
    (native * mask).sum().backward()
    torch.testing.assert_close(logits.grad, ref_logits.grad[..., rank * shard : (rank + 1) * shard])
    torch.testing.assert_close(logits.grad, native_logits.grad)
