# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native byte copies, bounded scheduling, overlap and failure propagation."""
import json
import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader
from megatron.lite.primitive.ckpt.row_reader import RowSliceReader


def _save(root, tensors):
    for name, value in tensors.items():
        save_file({name: value.contiguous()}, str(root / (name + '.safetensors')))
    (root / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {name: name + '.safetensors' for name in tensors}})
    )


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def test_row_prefetch_overlaps_and_writes_only_final_owned_buffers(tmp_path, monkeypatch):
    tensors = {
        'weight': torch.arange(512, dtype=torch.int64)
        .to(torch.uint8)
        .reshape(16, 32)
        .view(torch.float8_e4m3fn),
        'scale': torch.arange(16, dtype=torch.uint8).reshape(16, 1).view(torch.float8_e8m0fnu),
    }
    _save(tmp_path, tensors)
    destinations = {k: torch.empty((12, *t.shape[1:]), dtype=t.dtype) for k, t in tensors.items()}
    entered, release = threading.Event(), threading.Event()
    original = RowSliceReader._copy

    def blocked(*args):
        assert threading.current_thread() is not threading.main_thread()
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(RowSliceReader, '_copy', staticmethod(blocked))
    with RowSliceReader(tmp_path, workers=2) as reader:
        reader.prefetch_row_copies([(k, 3, 11, t) for k, t in destinations.items()], chunk_bytes=64)
        assert entered.wait(5)  # constructor returned while real I/O workers were blocked
        release.set()
        result = reader.get_row_tensors([(k, 3, 11) for k in tensors])
        for key, target in destinations.items():
            assert result[key] is target
            _same(target[:8], tensors[key][3:11])
            assert not target[8:].view(torch.uint8).any()
    assert not any(t.name.startswith('hf-row-read') for t in threading.enumerate())


def test_payload_prefetch_is_bounded_and_preserves_native_decode(tmp_path):
    tensors = {
        'expert.weight': torch.arange(256, dtype=torch.uint8).view(torch.int8).reshape(8, 32),
        'expert.scale': torch.arange(120, 136, dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .reshape(8, 2),
        'dense.weight': torch.arange(512, dtype=torch.float32).reshape(8, 64),
    }
    _save(tmp_path, tensors)
    with (
        SafeTensorReader(tmp_path) as reference,
        RowSliceReader(tmp_path, prefetch_bytes=512) as reader,
    ):
        reader.prefetch_tensors(['expert.weight', 'dense.weight'])
        assert reader._pending_bytes <= 512
        for name in ('expert.weight', 'dense.weight'):
            args = dict(target_shape=torch.Size((8, 64)), target_dtype=torch.bfloat16)
            _same(reader.get_tensor(name, **args), reference.get_tensor(name, **args))
            # A single native tensor larger than the window is admitted alone.
            assert reader._pending_bytes <= 512 or len(reader._pending) == 1
    assert reader._closed and not reader._pending


def test_prefetch_propagates_worker_errors_and_rejects_changed_ranges(tmp_path, monkeypatch):
    _save(tmp_path, {'x': torch.ones(8, 4)})
    target = torch.empty(8, 4)
    with RowSliceReader(tmp_path) as reader:
        reader.prefetch_row_copies([('x', 0, 8, target)])
        with pytest.raises(ValueError, match='range changed'):
            reader.get_row_tensors([('x', 1, 8)])
    with RowSliceReader(tmp_path) as reader:
        with pytest.raises(ValueError, match='native row destination'):
            reader.prefetch_row_copies([('x', 0, 8, target.double())])

    def fail(*_):
        raise OSError('injected read failure')

    monkeypatch.setattr(RowSliceReader, '_copy', staticmethod(fail))
    with pytest.raises(OSError, match='injected read failure'):
        with RowSliceReader(tmp_path) as reader:
            reader.prefetch_row_copies([('x', 0, 8, target)])
            reader.get_row_tensors([('x', 0, 8)])
    assert not any(t.name.startswith('hf-row-read') for t in threading.enumerate())


def test_strict_release_loader_consumes_prefetched_host_bytes(tmp_path):
    from megatron.lite.model.deepseek_v41.lite.checkpoint import load_hf_weights, prefetch_engram
    from megatron.lite.primitive.modules.engram import ShardedFrozenTable

    tensors = {
        'embed.weight': torch.arange(8).reshape(2, 4).bfloat16(),
        'layers.0.engram.embed.weight': torch.arange(256, dtype=torch.uint8)
        .reshape(8, 32)
        .view(torch.float8_e4m3fn),
        'layers.0.engram.embed.scale': torch.arange(8, dtype=torch.uint8)
        .reshape(8, 1)
        .view(torch.float8_e8m0fnu),
    }
    _save(tmp_path, tensors)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)

    def model(prefetch):
        module = torch.nn.Module()
        module.embed = torch.nn.Embedding(2, 4, dtype=torch.bfloat16)
        engram = torch.nn.Module()
        engram.embed = ShardedFrozenTable(8, 32, group, initialize=False)
        module.engrams = torch.nn.ModuleDict({'0': engram})
        module.layout = SimpleNamespace(has_embed=True, has_head=False)
        module.local_layer_range = (0, 1)
        module.impl = SimpleNamespace(prefetch_weights=prefetch)
        if prefetch:
            module._load_reader = RowSliceReader(tmp_path)
            prefetch_engram(module._load_reader, engram.embed, 0)
        return module

    a, b = model(False), model(True)
    cfg, ps = SimpleNamespace(n_routed_experts=1), SimpleNamespace(ep_size=1, ep_rank=0)
    for m in (a, b):
        load_hf_weights(m, str(tmp_path), cfg, ps)
    assert a.state_dict().keys() == b.state_dict().keys()
    for name, value in a.state_dict().items():
        _same(value, b.state_dict()[name])
    assert b._load_reader is None
