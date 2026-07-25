# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import io
import pickle
from argparse import Namespace
from collections import OrderedDict
from pickle import UnpicklingError

import pytest
import torch

from megatron.core.safe_globals import SafeUnpickler
from megatron.core.utils import is_torch_min_version


class UnsafeClass:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"UnsafeClass(value={self.value})"


class TestSafeGlobals:
    def test_safe_globals(self, tmp_path_dist_ckpt):
        # create dummy checkpoint
        ckpt_path = tmp_path_dist_ckpt / "test_safe_globals.pt"
        dummy_obj = Namespace(dummy_value=0)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            torch.save(dummy_obj, ckpt_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        torch.load(ckpt_path)

    @pytest.mark.skipif(not is_torch_min_version("2.6a0"), reason="PyTorch 2.6 is required")
    def test_unsafe_globals(self, tmp_path_dist_ckpt):
        # create dummy checkpoint
        ckpt_path = tmp_path_dist_ckpt / "test_safe_globals.pt"
        dummy_obj = UnsafeClass(123)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            torch.save(dummy_obj, ckpt_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # expected error
        with pytest.raises(UnpicklingError):
            torch.load(ckpt_path)

        # add class to safe globals
        torch.serialization.add_safe_globals([UnsafeClass])
        torch.load(ckpt_path)


class TestSafeUnpickler:
    def test_safe_types(self):
        data = {"key": [1, 2.0, True, "s"], "od": OrderedDict(a=1)}
        raw = pickle.dumps(data)
        result = SafeUnpickler(io.BytesIO(raw)).load()
        assert result == data

    def test_unsafe_types(self):
        raw = pickle.dumps(UnsafeClass(123))
        with pytest.raises(pickle.UnpicklingError, match="Refusing to unpickle"):
            SafeUnpickler(io.BytesIO(raw)).load()


class TestSafePickleLoad:
    def test_safe_types(self):
        data = {"key": [1, 2.0, True, b"bytes"], "od": OrderedDict(a=1)}
        raw = io.BytesIO(pickle.dumps(data))
        from megatron.core.safe_globals import _safe_pickle_load

        result = _safe_pickle_load(raw, buffers=[])
        assert result == data

    def test_unsafe_class_rejected(self):
        raw = io.BytesIO(pickle.dumps(UnsafeClass(42)))
        from megatron.core.safe_globals import _safe_pickle_load

        with pytest.raises(pickle.UnpicklingError, match="Refusing to unpickle"):
            _safe_pickle_load(raw)


class TestSafeNumpyLoad:
    def test_npy_array(self, tmp_path):
        import numpy as np

        from megatron.core.safe_globals import safe_numpy_load

        arr = np.array([1, 2, 3], dtype=np.uint32)
        path = tmp_path / "arr.npy"
        np.save(str(path), arr)

        result = safe_numpy_load(str(path), allow_pickle=True)
        np.testing.assert_array_equal(result, arr)

    def test_npz_archive(self, tmp_path):
        import numpy as np

        from megatron.core.safe_globals import safe_numpy_load

        a = np.array([1.0, 2.0])
        b = np.array([3, 4], dtype=np.int32)
        path = tmp_path / "archive.npz"
        np.savez(str(path), a=a, b=b)

        result = safe_numpy_load(str(path))
        np.testing.assert_array_equal(result["a"], a)
        np.testing.assert_array_equal(result["b"], b)

    def test_pickle_load_is_patched_during_call(self, tmp_path):
        # Verify that pickle.load is replaced with _safe_pickle_load while
        # safe_numpy_load runs, and restored afterward.
        import pickle as _pickle

        import numpy as np

        from megatron.core.safe_globals import _safe_pickle_load, safe_numpy_load

        arr = np.array([0])
        path = tmp_path / "arr.npy"
        np.save(str(path), arr)

        seen = []

        original_safe = _safe_pickle_load

        def capturing_safe(file, **kwargs):
            seen.append(_pickle.load)
            return original_safe(file, **kwargs)

        import megatron.core.safe_globals as sg

        original = sg._safe_pickle_load
        sg._safe_pickle_load = capturing_safe
        try:
            safe_numpy_load(str(path))
        finally:
            sg._safe_pickle_load = original

        # pickle.load is restored after the call
        assert _pickle.load is not capturing_safe

    def test_thread_safety(self, tmp_path):
        # Concurrent calls must not see each other's patch or corrupt results.
        import threading

        import numpy as np

        from megatron.core.safe_globals import safe_numpy_load

        arrays = {i: np.arange(i, i + 4, dtype=np.float32) for i in range(8)}
        paths = {}
        for i, arr in arrays.items():
            p = tmp_path / f"arr_{i}.npy"
            np.save(str(p), arr)
            paths[i] = p

        results = {}
        errors = []

        def load(i):
            try:
                results[i] = safe_numpy_load(str(paths[i]))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=load, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for i, arr in arrays.items():
            np.testing.assert_array_equal(results[i], arr)
