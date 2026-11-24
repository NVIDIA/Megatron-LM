# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for combined Tensor Parallelism + Generalized Tensor Parallelism (TP+GTP).

Process group layout (world_size = tp_size x gtp_remat_size):

    rank = gtp_rank x tp_size + tp_rank

    TP  group: all ranks that share the same gtp_rank  (size = tp_size)
    GTP group: all ranks that share the same tp_rank   (size = gtp_remat_size)

Tests use (tp_size, gtp_remat_size) = (2, 2) on 4 ranks.
"""

import types

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.extensions.transformer_engine import _gtp_pre_init
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTP_CONFIG,
    GTPShardedParam,
    update_gtp_config,
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    _requires_multi_gpu,
    _run_distributed,
    cached_group,
)


def _build_groups(rank: int, world_size: int, tp_size: int, gtp_remat_size: int):
    """TP and GTP groups for rank = gtp_rank * tp_size + tp_rank.

    TP groups are contiguous blocks, GTP groups strided. Returns
    (tp_group, gtp_remat_group, tp_rank, gtp_rank).
    """
    assert tp_size * gtp_remat_size == world_size
    tp_groups = [cached_group(range(g * tp_size, (g + 1) * tp_size)) for g in range(gtp_remat_size)]
    gtp_groups = [cached_group(range(t, world_size, tp_size)) for t in range(tp_size)]
    tp_rank, gtp_rank = rank % tp_size, rank // tp_size
    return tp_groups[gtp_rank], gtp_groups[tp_rank], tp_rank, gtp_rank


def _gather(tensor, group, dim):
    shards = [torch.empty_like(tensor) for _ in range(group.size())]
    dist.all_gather(shards, tensor.contiguous(), group=group)
    return torch.cat(shards, dim=dim)


def _assert_close(actual, expected, label):
    """BF16 GEMM vs an FP32 reference: within 1% of the reference's magnitude."""
    expected = expected.float()
    tol = 1e-2 * expected.abs().max().item()
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=tol, msg=label)


def _worker_linear_fwd_bwd(rank, world_size, port, tp_size, gtp_remat_size, parallel_mode):
    """Output, dX and the GTP main_grad shard all equal an FP32 reference on the full weight.

    Every rank sees the same full input and output gradient, so the reduced wgrad (a gtp mean)
    is exactly this rank's rows of the TP-local weight gradient.
    """
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        wait_for_gtp_grad_reduction_on_current_stream,
    )

    torch.manual_seed(0)
    tp_group, gtp_remat_group, tp_rank, gtp_rank = _build_groups(
        rank, world_size, tp_size, gtp_remat_size
    )
    batch, rows = 16, 32  # rows of weight per rank
    column = parallel_mode == "column"
    in_f = 64 if column else tp_size * 64
    out_f = tp_size * gtp_remat_size * rows if column else gtp_remat_size * rows
    dtype = torch.bfloat16

    layer = _make_gtp_linear(
        in_f, out_f, gtp_remat_group, dtype, parallel_mode=parallel_mode, tp_group=tp_group
    )
    assert isinstance(layer.weight, GTPShardedParam) and layer.weight.shape[0] == rows
    layer.weight.main_grad = torch.zeros(layer.weight.shape, dtype=torch.float32, device="cuda")

    # TP-local weight (GTP shards gathered) and the full weight (TP shards gathered too).
    local_w = _gather(layer.weight.data, gtp_remat_group, dim=0).float()
    full_w = _gather(local_w, tp_group, dim=0 if column else 1)

    full_x = torch.randn(batch, in_f, dtype=dtype, device="cuda")
    dist.broadcast(full_x, src=0)
    # Column: every TP rank takes the full input; row: its own slice of in_features.
    local_in = in_f if column else in_f // tp_size
    inputs = slice(0, in_f) if column else slice(tp_rank * local_in, (tp_rank + 1) * local_in)
    x = full_x[:, inputs].clone().requires_grad_(True)
    out = layer(x)

    # Column: this rank's output slice; dX sums every TP rank's contribution (TE all-reduces).
    # Row: TP-reduced full output; dX is this rank's input slice of dY @ W.
    out_cols = out_f // tp_size if column else out_f
    full_dy = torch.randn(batch, out_cols * (tp_size if column else 1), dtype=dtype, device="cuda")
    dist.broadcast(full_dy, src=0)
    dy = full_dy[:, tp_rank * out_cols : (tp_rank + 1) * out_cols] if column else full_dy
    ref_out = full_x.float() @ (local_w if column else full_w).T
    ref_dx = (full_dy.float() @ full_w)[:, inputs]
    ref_wgrad = dy.float().T @ x.detach().float()  # [out_cols, local_in]: TP-local weight grad

    out.backward(dy)
    wait_for_gtp_grad_reduction_on_current_stream()
    _assert_close(out, ref_out, f"rank {rank}: output")
    _assert_close(x.grad, ref_dx, f"rank {rank}: dX")
    _assert_close(
        layer.weight.main_grad,
        ref_wgrad[gtp_rank * rows : (gtp_rank + 1) * rows],
        f"rank {rank}: main_grad",
    )


@pytest.mark.parametrize("parallel_mode", ["column", "row"])
def test_tp2_gtp2_linear_matches_reference(parallel_mode):
    """TP=2 x GTP=2 column/row te.Linear: forward, dX and wgrad values."""
    _requires_multi_gpu(4)
    _run_distributed(_worker_linear_fwd_bwd, 4, 2, 2, parallel_mode)


# ---------------------------------------------------------------------------
# TestTPGTPPaddingAlignment - GTP pre-shard must pad the *per-TP* slice so the weight
#    stays MXFP8-aligned AFTER TE's tp-split (padding the full out_features would let the
#    tp-split de-align it).
# ---------------------------------------------------------------------------


def _worker_pre_init_tp_padding(rank, world_size, port, tp_size, gtp_remat_size):
    """`_gtp_pre_init` pads ``output_size // out_split_size`` (the per-TP slice), so the shard
    TE hands each rank after its own tp-split is a multiple of ``pad_for_alignment`` (32 for MXFP8).

    tp2 x gtp2, pad=32, out_features=192:
      correct:   per-TP slice 96 -> pad 128 -> /gtp2 = 64 -> TE tp-split /2 = 64 (32-aligned).
      pre-fix:   pad full 192 -> /gtp2 = 96 -> TE tp-split /2 = 48 (NOT 32-aligned).
    """
    tp_group, gtp_remat_group, tp_rank, gtp_rank = _build_groups(
        rank, world_size, tp_size, gtp_remat_size
    )
    out_features = 192
    orig_pad = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=32)
    try:
        extra_kwargs = {}
        shard_out, gtp_ctx = _gtp_pre_init(
            types.SimpleNamespace(),
            out_features,
            gtp_remat_group,
            extra_kwargs,
            out_split_size=tp_size,
        )
        # shard_out is what TE receives as out_features; TE splits it again across tp_size.
        per_gpu = shard_out // tp_size
        assert per_gpu % 32 == 0, (
            f"rank {rank}: per-GPU shard {per_gpu} not 32-aligned -- TE tp-split de-aligned the "
            "GTP pad (pad the per-TP slice, not the full out_features)"
        )
        assert per_gpu == 64, f"rank {rank}: expected per-GPU 64, got {per_gpu}"
        # pad_length is measured on the per-TP slice: 128 - 96 = 32.
        gtp_remat_group_ctx, pad_length, logical = gtp_ctx
        assert pad_length == 32, f"rank {rank}: expected pad_length 32, got {pad_length}"
        assert gtp_remat_group_ctx.size() == gtp_remat_size and logical == out_features
    finally:
        update_gtp_config(pad_for_alignment=orig_pad)


class TestTPGTPPaddingAlignment:
    @pytest.mark.parametrize("tp_size,gtp_remat_size", [(2, 2)])
    def test_pre_init_pads_per_tp_slice(self, tp_size, gtp_remat_size):
        world_size = tp_size * gtp_remat_size
        _requires_multi_gpu(world_size)
        _run_distributed(_worker_pre_init_tp_padding, world_size, tp_size, gtp_remat_size)
