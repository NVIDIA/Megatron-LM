# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression for multi-invocation FSDP units (deferred release + deferred
gradient reduction).

An FSDP unit may legitimately run several forward passes inside one
microbatch — e.g. a vision tower executed in image-boundary chunks. Under
activation recompute every chunk's checkpoint backward is its own inner
``autograd.backward``, so per-parameter gradient hooks fire once PER
invocation; without the deferral fixes this reduce-scattered per event
(racing bucket allocation and multiply-reducing) and released unsharded
parameter buffers while other invocations' kernels were still enqueued.

Run via:
    torchrun --nproc-per-node 2 -m pytest -q \\
        tests/unit_tests/distributed/test_mfsdp_multi_invocation.py
"""

import os

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_WORLD = int(os.getenv("WORLD_SIZE", "1"))

pytestmark = pytest.mark.skipif(_WORLD != 2, reason="two-rank MegatronFSDP regression")


@pytest.fixture(scope="module", autouse=True)
def _init_model_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


class _Stack(nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(depth))

    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.gelu(layer(x))
        return x


def _build(*, defer_flags, seed=1234):
    torch.manual_seed(seed)
    model = _Stack().cuda()
    if defer_flags:
        for submodule in model.modules():
            submodule._fsdp_defer_release = True
        for parameter in model.parameters():
            parameter._fsdp_defer_grad_reduce = True
    config = TransformerConfig(
        num_layers=1, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
    )
    ddp_config = DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy="optim_grads_params",
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        megatron_fsdp_main_params_dtype=None,
    )
    wrapped = FullyShardedDataParallel(
        config=config, ddp_config=ddp_config, module=model, fsdp_unit_modules=[nn.Linear]
    )
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-2)
    return wrapped, optimizer


def _data(rank_offset=0):
    torch.manual_seed(777 + torch.distributed.get_rank() + rank_offset)
    return torch.randn(8, 64, device="cuda")


def _train(wrapped, optimizer, *, num_chunks, use_checkpoint, microbatches=2):
    for microbatch in range(microbatches):
        if hasattr(wrapped, "zero_grad_buffer"):
            wrapped.zero_grad_buffer()
        data = _data(rank_offset=microbatch)
        if num_chunks == 1:
            out = wrapped(data)
        else:
            # Multi-invocation: the SAME wrapped module runs once per chunk;
            # with checkpointing each chunk's backward is its own inner
            # autograd.backward — the exact trigger of the original race.
            chunks = data.chunk(num_chunks)
            outs = []
            for piece in chunks:
                if use_checkpoint:
                    outs.append(checkpoint(wrapped, piece, use_reentrant=True))
                else:
                    outs.append(wrapped(piece))
            out = torch.cat(outs)
        out.square().mean().backward()
        if hasattr(wrapped, "finish_grad_sync"):
            wrapped.finish_grad_sync()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return out


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_multi_invocation_matches_single_call(use_checkpoint):
    reference, ref_opt = _build(defer_flags=False)
    multi, multi_opt = _build(defer_flags=True)

    out_ref = _train(reference, ref_opt, num_chunks=1, use_checkpoint=False)
    out_multi = _train(multi, multi_opt, num_chunks=4, use_checkpoint=use_checkpoint)

    torch.testing.assert_close(out_multi, out_ref, rtol=1e-5, atol=1e-6)
    for (name, p_ref), (_, p_multi) in zip(
        reference.named_parameters(), multi.named_parameters()
    ):
        torch.testing.assert_close(
            p_multi.to_local() if hasattr(p_multi, "to_local") else p_multi,
            p_ref.to_local() if hasattr(p_ref, "to_local") else p_ref,
            rtol=1e-5,
            atol=1e-6,
            msg=name,
        )
