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
