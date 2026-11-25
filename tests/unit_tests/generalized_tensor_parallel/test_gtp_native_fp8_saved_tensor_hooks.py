# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A native-FP8 GTP weight that autograd saved through saved-tensor hooks.

Under ``torch.autograd.graph.saved_tensors_hooks`` (fine-grained activation offloading) the
backward receives a detached copy of the saved weight. The GTP protocol calls on that copy
must act on the live parameter.

Run standalone with ``--confcutdir=tests/unit_tests/generalized_tensor_parallel``.
No collectives or process groups are needed, and no NCCL settings are changed.
"""

import os

import pytest
import torch


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    previous_device = torch.cuda.current_device()
    device = int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count()
    torch.cuda.set_device(device)
    try:
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("MXFP8 requires Blackwell")
        yield
    finally:
        torch.cuda.set_device(previous_device)


def test_saved_native_fp8_weight_acts_on_the_live_parameter(cuda_device, monkeypatch):
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        _gtp_reclass_native_fp8_shard,
    )

    quantizer = MXFP8Quantizer(TE_DType[torch.float8_e4m3fn], rowwise=True, columnwise=True)
    quantizer.internal = False
    weight = torch.nn.Parameter(
        quantizer(torch.randn(128, 128, device="cuda", dtype=torch.bfloat16))
    )
    _gtp_reclass_native_fp8_shard(weight)
    # The protocol calls must see state set after the weight was saved, not a snapshot.
    monkeypatch.setattr(weight, "all_gather_and_prefetch", lambda **kw: weight.next_w)
    monkeypatch.setattr(weight, "all_gather_and_prefetch_bwd", lambda **kw: weight.next_w)
    monkeypatch.setattr(weight, "wgrad_reduce_scatter", lambda grads, **kw: weight.next_w)
    monkeypatch.setattr(weight, "get_wgrad_tensor", lambda: weight.next_w)
    seen = []

    class SaveWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, param, inp):
            ctx.save_for_backward(param)
            return inp * 2

        @staticmethod
        def backward(ctx, grad):
            saved = ctx.saved_tensors[0]
            for alias in (saved, saved.detach()):
                assert alias.materialize_group_for_forward() is weight.next_w
                assert alias.materialize_group_for_backward() is weight.next_w
                assert alias.finalize_group_grads(None) is weight.next_w
                assert alias.grad_buffer() is weight.next_w
            seen.append(saved)
            return None, grad * 2

    inp = torch.ones(1, device="cuda", requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (t,), lambda packed: packed[0]):
        out = SaveWeight.apply(weight, inp)
    weight.next_w = object()
    out.sum().backward()
    assert len(seen) == 1
    torch.testing.assert_close(inp.grad, torch.full_like(inp, 2))
