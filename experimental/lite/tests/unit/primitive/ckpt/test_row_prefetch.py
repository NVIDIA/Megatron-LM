# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native byte copies, bounded scheduling, overlap and failure propagation."""
import json
import threading

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
