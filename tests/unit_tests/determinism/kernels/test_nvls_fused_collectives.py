# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused NVLS reduce-scatter + residual add + RMSNorm + all-gather of fused TP communication.

``fused_multimem_rs_add_norm_ag`` launches one block per token: each block processes its tokens
one after another, so a block sized from the whole per-rank shard ran several tokens in series
with most of its threads idle. The launch geometry, and the launcher's refusal of inputs the
kernel would read or write out of bounds, are checked on the real launcher without a GPU. On an
NVLink multicast group, the kernel replays bit-exactly and gives the bits of the shard-sized
launch it replaced: the extra threads of a wider block only add zeros to the block-wide sum of
squares.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton import fused_collectives
from megatron.core.inference.communication.torch_symm_triton.kernel_config import (
    SYMM_MEM_KERNEL_CONFIG,
)
from tests.unit_tests.determinism.kernels.harness import bytes_equal
from tests.unit_tests.test_utilities import Utils

MAX_NUM_BLOCKS = SYMM_MEM_KERNEL_CONFIG.max_num_blocks
MAX_BLOCK_SIZE = SYMM_MEM_KERNEL_CONFIG.max_block_size
NUMEL_PER_THREAD = 8  # 128 bits of bf16


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class _LaunchRecorder:
    """Stands in for the Triton kernel: records the grid and launch arguments."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.launches.append(dict(grid=grid, kwargs=kwargs))


@pytest.mark.parametrize(
    "world_size,hidden,tokens",
    [
        (8, 4096, 64),
        (8, 2048, 64),
        (8, 3584, 256),
        (4, 4096, 32),
        (2, 2048, 16),
        (2, 8192, 16),
        (2, 16384, 64),
        (8, 4096, 4096),
    ],
)
def test_launch_runs_one_token_per_block(monkeypatch, world_size, hidden, tokens):
    """Each block covers one token's row, and up to max_num_blocks tokens per rank get a
    block each."""
    pytest.importorskip("triton")
    recorder = _LaunchRecorder()
    monkeypatch.setattr(fused_collectives, "_multimem_reduce_scatter_residual_add_kernel", recorder)
    handle = SimpleNamespace(world_size=world_size, rank=0, multicast_ptr=0, signal_pad_ptrs_dev=0)
    symm = torch.empty(tokens, hidden, dtype=torch.bfloat16)
    residual = torch.empty(tokens // world_size, hidden, dtype=torch.bfloat16)
    weight = torch.empty(hidden, dtype=torch.bfloat16)

    fused_collectives.fused_multimem_rs_add_norm_ag(residual, symm, handle, residual, weight, 1e-5)

    (launch,) = recorder.launches
    tokens_per_rank = tokens // world_size
    row_threads = math.ceil(hidden / NUMEL_PER_THREAD)
    block_size = launch["kwargs"]["BLOCK_SIZE"]
    assert launch["grid"] == (min(tokens_per_rank, MAX_NUM_BLOCKS), 1, 1)
    assert block_size == min(_next_power_of_2(row_threads), MAX_BLOCK_SIZE)
    assert launch["kwargs"]["num_warps"] == max(1, block_size // SYMM_MEM_KERNEL_CONFIG.warp_size)


@pytest.mark.parametrize(
    "case,error",
    [
        ("valid", None),
        ("tokens-not-a-multiple-of-world-size", "divisible by the world size"),
        ("hidden-not-a-multiple-of-8", "multiple of 8"),
        ("weights-not-bf16", "must be bfloat16"),
        ("weights-not-hidden-long", "RMSNorm weights, got"),
        ("residual-input-not-shard-sized", "residual input must hold"),
        ("residual-output-not-shard-sized", "residual output must hold"),
        ("input-not-contiguous", "input tensor must be contiguous"),
        ("residual-input-not-contiguous", "residual input must be contiguous"),
        ("residual-output-not-contiguous", "residual output must be contiguous"),
        ("weights-not-contiguous", "weights must be contiguous"),
    ],
)
def test_launcher_refuses_inputs_outside_the_contract(monkeypatch, case, error):
    """Inputs the kernel cannot process are refused before anything is launched: with a token
    count not divisible by the world size the last rank would reduce and normalize rows past the
    end and write them into every rank's buffer; with H % 8 != 0 every row after the first is
    misaligned; weights that are not H bf16 values are misdecoded or read out of bounds;
    residuals that are not this rank's tokens are read and written out of bounds; and a strided
    view with the right element count is addressed through its data pointer as dense rows, so
    the wrong elements are read and written."""
    pytest.importorskip("triton")
    recorder = _LaunchRecorder()
    monkeypatch.setattr(fused_collectives, "_multimem_reduce_scatter_residual_add_kernel", recorder)
    world_size, tokens, hidden = 4, 8, 1024
    weight_dtype, weight_len = torch.bfloat16, None
    residual_in_rows = residual_out_rows = None
    if case == "tokens-not-a-multiple-of-world-size":
        tokens = 10
    elif case == "hidden-not-a-multiple-of-8":
        hidden = 12  # tokens * hidden is still a multiple of 8
    elif case == "weights-not-bf16":
        weight_dtype = torch.float32
    elif case == "weights-not-hidden-long":
        weight_len = hidden + 8
    elif case == "residual-input-not-shard-sized":
        residual_in_rows = tokens // world_size + 1
    elif case == "residual-output-not-shard-sized":
        residual_out_rows = tokens // world_size + 1
    handle = SimpleNamespace(
        world_size=world_size, rank=world_size - 1, multicast_ptr=0, signal_pad_ptrs_dev=0
    )
    shard_rows = math.ceil(tokens / world_size)

    def tensor(*shape, dtype=torch.bfloat16, contiguous=True):
        if contiguous:
            return torch.empty(*shape, dtype=dtype)
        # Every other element of a twice-as-wide last dimension: the right element count.
        return torch.empty(*shape[:-1], 2 * shape[-1], dtype=dtype)[..., ::2]

    symm = tensor(tokens, 1, hidden, contiguous=case != "input-not-contiguous")
    residual_in = tensor(
        residual_in_rows or shard_rows,
        1,
        hidden,
        contiguous=case != "residual-input-not-contiguous",
    )
    residual_out = tensor(
        residual_out_rows or shard_rows,
        1,
        hidden,
        contiguous=case != "residual-output-not-contiguous",
    )
    weight = tensor(
        weight_len or hidden, dtype=weight_dtype, contiguous=case != "weights-not-contiguous"
    )

    def launch():
        fused_collectives.fused_multimem_rs_add_norm_ag(
            residual_out, symm, handle, residual_in, weight, 1e-5
        )

    if error is None:
        launch()
        assert len(recorder.launches) == 1
        return
    with pytest.raises(AssertionError, match=error):
        launch()
    assert not recorder.launches, f"{case}: the kernel was launched"


@pytest.fixture(scope="module")
def tp_symm_buffer():
    if not torch.cuda.is_available() or Utils.world_size < 2:
        pytest.skip("needs at least two GPUs")
    if Utils.world_size & (Utils.world_size - 1):
        pytest.skip("the symmetric-memory barrier needs a power-of-two world size")
    from megatron.core.inference.communication.torch_symm_triton.utils import is_device_nvls_capable

    if not is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device())):
        pytest.skip("NVLS multicast requires a Hopper+ GPU (SM >= 9)")
    from megatron.core import parallel_state
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=tp_group)
        if buf.symm_mem_hdl is None:
            pytest.skip(f"symmetric memory unavailable: {buf.init_failure_reason}")
        yield tp_group, buf
    finally:
        SymmetricMemoryManager.destroy("tp")
        Utils.destroy_model_parallel()


def _shard_sized_launch(residual_output, symm, handle, residual_input, weight, eps):
    """The same kernel with the launch geometry it had before: BLOCK_SIZE and the grid sized
    from the whole per-rank shard."""
    num_threads = math.ceil(symm.numel() // NUMEL_PER_THREAD / handle.world_size)
    if num_threads < MAX_BLOCK_SIZE:
        block_size, num_blocks = _next_power_of_2(num_threads), 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min(math.ceil(num_threads / MAX_BLOCK_SIZE), MAX_NUM_BLOCKS)
    hidden = symm.size(-1)
    fused_collectives._multimem_reduce_scatter_residual_add_kernel[(num_blocks, 1, 1)](
        residual_output.data_ptr(),
        residual_input.data_ptr(),
        weight.data_ptr(),
        handle.multicast_ptr,
        handle.signal_pad_ptrs_dev,
        symm.numel() // hidden,
        eps=eps,
        HIDDEN_SIZE=hidden,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=NUMEL_PER_THREAD,
        RANK=handle.rank,
        WORLD_SIZE=handle.world_size,
        num_warps=block_size // SYMM_MEM_KERNEL_CONFIG.warp_size,
    )


def _all_ranks_ok(ok: bool) -> bool:
    flag = torch.tensor([int(ok)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


@pytest.mark.parametrize("hidden", [2048, 3584, 4096, 16384])
def test_replays_and_matches_shard_sized_launch(tp_symm_buffer, hidden):
    """Residual output and normalized all-gathered output replay bit-exactly and equal the
    shard-sized launch's, with one or several tokens per block."""
    tp_group, buf = tp_symm_buffer
    world_size = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)
    eps = 1e-5
    weight_gen = torch.Generator(device="cuda").manual_seed(7)  # same norm weights on all ranks
    weight = (1 + 0.1 * torch.randn(hidden, device="cuda", generator=weight_gen)).bfloat16()
    for tokens_per_rank in (8, 64, 256):
        tokens = tokens_per_rank * world_size
        symm = buf.maybe_get_tensor([tokens, hidden], torch.bfloat16)
        assert symm["handle"] is not None
        gen = torch.Generator(device="cuda").manual_seed(100 + rank)
        partial = torch.randn(tokens, hidden, device="cuda", generator=gen).bfloat16()
        residual = torch.randn(tokens_per_rank, hidden, device="cuda", generator=gen).bfloat16()

        def run(launch):
            symm["tensor"].copy_(partial)
            out = residual.clone()
            launch(out, symm["tensor"], symm["handle"], out, weight, eps)
            return out, symm["tensor"].clone()

        expected = run(_shard_sized_launch)
        for replay in range(3):
            actual = run(fused_collectives.fused_multimem_rs_add_norm_ag)
            ok = all(bytes_equal(a, e) for a, e in zip(actual, expected))
            assert _all_ranks_ok(ok), (
                f"H={hidden}, {tokens_per_rank} tokens per rank, replay {replay}: output "
                "differs from the shard-sized launch"
            )
