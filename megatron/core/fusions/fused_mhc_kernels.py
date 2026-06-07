# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused cuTile kernels for mHC (Manifold-Constrained Hyper-Connections).

Requires cuda.tile (cuTile) for optimal performance on supported GPUs
(compute capability 10.x+).  Reference (non-fused) implementations live in
``megatron.core.transformer.hyper_connection`` and are used when cuTile is
unavailable or when the ``use_fused_mhc`` config flag is False.

Four fused operations:
  - sinkhorn:     Sinkhorn-Knopp projection to doubly stochastic matrix
  - h_aggregate:  weighted n-stream -> 1-stream aggregation
  - h_post_bda:   fused H_res @ residual + H_post * (x + bias)
  - proj_rms:     fused projection + RMS normalization
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Check cuTile availability
# ---------------------------------------------------------------------------
_CUTILE_AVAILABLE = False
try:
    import cuda.tile as ct

    _CUTILE_AVAILABLE = True
except ImportError:
    pass


def is_cutile_available() -> bool:
    """Return True if cuTile fused kernels are available."""
    return _CUTILE_AVAILABLE


# ============================================================================
# CuTile implementations (only defined when cuda.tile is available)
# ============================================================================

if _CUTILE_AVAILABLE:
    ConstInt = ct.Constant[int]
    PAD_ZERO = ct.PaddingMode.ZERO
    LOG2E = 1.4426950408889634

    # -- Sinkhorn kernels ----------------------------------------------------

    @ct.kernel
    def _ct_sinkhorn_fwd_kernel(
        inp, out, M_init_out, eps, HC: ConstInt, NUM_ITERS: ConstInt, TILE_SIZE: ConstInt
    ):
        pid = ct.bid(0)
        logits = ct.load(inp, index=(pid, 0, 0), shape=(TILE_SIZE, HC, HC)).astype(ct.float32)
        row_max = ct.max(logits, axis=2, keepdims=True)
        M = ct.exp2((logits - row_max) * LOG2E)
        ct.store(
            M_init_out,
            index=(pid, 0, 0),
            tile=ct.reshape(M.astype(M_init_out.dtype), (TILE_SIZE, HC, HC)),
        )
        for _ in range(NUM_ITERS):
            row_sum = ct.sum(M, axis=2, keepdims=True)
            M = M / (row_sum + eps)
            col_sum = ct.sum(M, axis=1, keepdims=True)
            M = M / (col_sum + eps)
        ct.store(out, index=(pid, 0, 0), tile=ct.reshape(M.astype(out.dtype), (TILE_SIZE, HC, HC)))

    @ct.kernel
    def _ct_sinkhorn_bwd_kernel(
        grad_out,
        M_init,
        grad_inp,
        ws_M,
        ws_rs,
        ws_cs,
        eps,
        HC: ConstInt,
        NUM_ITERS: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        pid = ct.bid(0)
        M_base = pid * (2 * NUM_ITERS)
        v_base = pid * NUM_ITERS

        M = ct.load(M_init, index=(pid, 0, 0), shape=(TILE_SIZE, HC, HC)).astype(ct.float32)
        for t in range(NUM_ITERS):
            ct.store(ws_M, index=(M_base + 2 * t, 0, 0), tile=M)
            row_sum = ct.sum(M, axis=2, keepdims=True)
            ct.store(ws_rs, index=(v_base + t, 0, 0), tile=row_sum)
            M = M / (row_sum + eps)
            ct.store(ws_M, index=(M_base + 2 * t + 1, 0, 0), tile=M)
            col_sum = ct.sum(M, axis=1, keepdims=True)
            ct.store(ws_cs, index=(v_base + t, 0, 0), tile=col_sum)
            M = M / (col_sum + eps)

        grad = ct.load(grad_out, index=(pid, 0, 0), shape=(TILE_SIZE, HC, HC)).astype(ct.float32)
        for t_rev in range(NUM_ITERS):
            t = NUM_ITERS - 1 - t_rev
            col_s = ct.load(ws_cs, index=(v_base + t, 0, 0), shape=(TILE_SIZE, 1, HC))
            grad = grad / (col_s + eps)
            col_corr = ct.sum(grad * M, axis=1, keepdims=True)
            grad = grad - col_corr
            M = ct.load(ws_M, index=(M_base + 2 * t + 1, 0, 0), shape=(TILE_SIZE, HC, HC))
            row_s = ct.load(ws_rs, index=(v_base + t, 0, 0), shape=(TILE_SIZE, HC, 1))
            grad = grad / (row_s + eps)
            row_corr = ct.sum(grad * M, axis=2, keepdims=True)
            grad = grad - row_corr
            M = ct.load(ws_M, index=(M_base + 2 * t, 0, 0), shape=(TILE_SIZE, HC, HC))
        grad = grad * M
        ct.store(grad_inp, index=(pid, 0, 0), tile=grad.astype(grad_inp.dtype))

    def _cutile_sinkhorn_fwd(
        input_logits: Tensor, num_iterations: int, eps: float = 1e-8
    ) -> Tuple[Tensor, Tensor]:
        original_shape = input_logits.shape
        hc = original_shape[-1]
        N_batch = input_logits.numel() // (hc * hc)
        TILE_SIZE = math.gcd(N_batch, 128)
        dev = input_logits.device
        out = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
        M_init = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
        ct.launch(
            torch.cuda.current_stream(),
            (math.ceil(N_batch / TILE_SIZE), 1, 1),
            _ct_sinkhorn_fwd_kernel,
            (input_logits.view(N_batch, hc, hc), out, M_init, eps, hc, num_iterations, TILE_SIZE),
        )
        return out.view(original_shape), M_init.view(original_shape)

    def _cutile_sinkhorn_bwd(
        grad_output: Tensor, M_init: Tensor, num_iterations: int, eps: float = 1e-8
    ) -> Tensor:
        original_shape = grad_output.shape
        hc = original_shape[-1]
        N_batch = grad_output.numel() // (hc * hc)
        TILE_SIZE = math.gcd(N_batch, 128)
        dev = grad_output.device
        ws_M = torch.empty(N_batch * 2 * num_iterations, hc, hc, dtype=torch.float32, device=dev)
        ws_rs = torch.empty(N_batch * num_iterations, hc, 1, dtype=torch.float32, device=dev)
        ws_cs = torch.empty(N_batch * num_iterations, 1, hc, dtype=torch.float32, device=dev)
        grad_input = torch.empty(N_batch, hc, hc, dtype=grad_output.dtype, device=dev)
        ct.launch(
            torch.cuda.current_stream(),
            (math.ceil(N_batch / TILE_SIZE), 1, 1),
            _ct_sinkhorn_bwd_kernel,
            (
                grad_output.view(N_batch, hc, hc),
                M_init.view(N_batch, hc, hc),
                grad_input,
                ws_M,
                ws_rs,
                ws_cs,
                eps,
                hc,
                num_iterations,
                TILE_SIZE,
            ),
        )
        return grad_input.view(original_shape)

    # -- H_aggregate kernels -------------------------------------------------

    @ct.kernel
    def _ct_h_agg_fwd_kernel(x, h_pre, out, N: ConstInt, TILE_M: ConstInt, TILE_C: ConstInt):
        pid = ct.bid(0)
        num_tiles = ct.num_tiles(x, axis=2, shape=(TILE_M, N, TILE_C))
        h_tile = ct.load(h_pre, index=(pid, 0), shape=(TILE_M, N), padding_mode=PAD_ZERO)
        h_tile = ct.expand_dims(h_tile, axis=2)
        for j in range(num_tiles):
            x_tile = ct.load(x, index=(pid, 0, j), shape=(TILE_M, N, TILE_C), padding_mode=PAD_ZERO)
            acc = ct.sum(x_tile * h_tile, axis=1).astype(ct.float32)
            ct.store(out, index=(pid, j), tile=acc.astype(out.dtype))

    @ct.kernel
    def _ct_h_agg_bwd_kernel(go, x, h_pre, gx, gh, N: ConstInt, TILE_M: ConstInt, TILE_C: ConstInt):
        pid = ct.bid(0)
        num_c_tiles = ct.num_tiles(go, axis=1, shape=(TILE_M, TILE_C))
        h_tile = ct.load(h_pre, index=(pid, 0), shape=(TILE_M, N), padding_mode=PAD_ZERO)
        h_expanded = ct.expand_dims(h_tile, axis=2)
        gh_acc = ct.full((TILE_M, N), 0, dtype=ct.float32)
        for ct_idx in range(num_c_tiles):
            go_tile = ct.load(
                go, index=(pid, ct_idx), shape=(TILE_M, TILE_C), padding_mode=PAD_ZERO
            )
            go_expanded = ct.expand_dims(go_tile, axis=1)
            x_tile = ct.load(
                x, index=(pid, 0, ct_idx), shape=(TILE_M, N, TILE_C), padding_mode=PAD_ZERO
            )
            gx_tile = go_expanded * h_expanded
            ct.store(gx, index=(pid, 0, ct_idx), tile=gx_tile.astype(gx.dtype))
            gh_acc += ct.sum(go_expanded * x_tile, axis=2)
        ct.store(gh, index=(pid, 0), tile=gh_acc.astype(gh.dtype))

    def _cutile_h_aggregate_fwd(x: Tensor, h_pre: Tensor) -> Tensor:
        s, b, n, C = x.shape
        sb = s * b
        TILE_SIZE = math.gcd(sb, 4)
        TILE_C = math.gcd(C, 1024)
        out = torch.empty(sb, C, dtype=x.dtype, device=x.device)
        ct.launch(
            torch.cuda.current_stream(),
            (math.ceil(sb / TILE_SIZE),),
            _ct_h_agg_fwd_kernel,
            (x.view(sb, n, C), h_pre.view(sb, n), out, n, TILE_SIZE, TILE_C),
        )
        return out.view(s, b, C)

    def _cutile_h_aggregate_bwd(
        grad_output: Tensor, x: Tensor, h_pre: Tensor
    ) -> Tuple[Tensor, Tensor]:
        s, b, n, C = x.shape
        sb = s * b
        TILE_C = math.gcd(C, 1024)
        TILE_M = math.gcd(sb, 4)
        gx = torch.empty(sb, n, C, dtype=x.dtype, device=x.device)
        gh = torch.empty(sb, n, dtype=x.dtype, device=x.device)
        ct.launch(
            torch.cuda.current_stream(),
            (math.ceil(sb / TILE_M),),
            _ct_h_agg_bwd_kernel,
            (
                grad_output.view(sb, C),
                x.view(sb, n, C),
                h_pre.view(sb, n),
                gx,
                gh,
                n,
                TILE_M,
                TILE_C,
            ),
        )
        return gx.view(s, b, n, C), gh.view(s, b, n)

    # -- H_post BDA kernels --------------------------------------------------

    @ct.kernel
    def _ct_hpb_fwd_kernel(
        hr, orig, hp, x, out, N: ConstInt, TILE_C: ConstInt, TILE_SIZE: ConstInt
    ):
        pid = ct.bid(0)
        num_c_tiles = ct.num_tiles(x, axis=1, shape=(TILE_SIZE, TILE_C))
        hp_tile = ct.load(hp, index=(pid, 0), shape=(TILE_SIZE, N), padding_mode=PAD_ZERO)
        hp_2d = ct.reshape(hp_tile, (N, 1))
        hr_tile = ct.load(hr, index=(pid, 0, 0), shape=(TILE_SIZE, N, N), padding_mode=PAD_ZERO)
        hr_2d = ct.reshape(hr_tile, (N, N))
        for ct_idx in range(num_c_tiles):
            orig_tile = ct.load(
                orig, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            orig_2d = ct.reshape(orig_tile, (N, TILE_C))
            x_tile = ct.load(
                x, index=(pid, ct_idx), shape=(TILE_SIZE, TILE_C), padding_mode=PAD_ZERO
            )
            x_2d = ct.reshape(x_tile, (1, TILE_C))
            out_2d = hp_2d * x_2d
            for j in range(N):
                out_2d += ct.extract(hr_2d, (0, j), shape=(N, 1)) * ct.extract(
                    orig_2d, (j, 0), shape=(1, TILE_C)
                )
            ct.store(
                out,
                index=(pid, 0, ct_idx),
                tile=ct.reshape(out_2d, (TILE_SIZE, N, TILE_C)).astype(out.dtype),
            )

    @ct.kernel
    def _ct_hpb_fwd_bias_kernel(
        hr, orig, hp, x, bias, out, N: ConstInt, TILE_C: ConstInt, TILE_SIZE: ConstInt
    ):
        pid = ct.bid(0)
        num_c_tiles = ct.num_tiles(x, axis=1, shape=(TILE_SIZE, TILE_C))
        hp_tile = ct.load(hp, index=(pid, 0), shape=(TILE_SIZE, N), padding_mode=PAD_ZERO)
        hp_2d = ct.reshape(hp_tile, (N, 1))
        hr_tile = ct.load(hr, index=(pid, 0, 0), shape=(TILE_SIZE, N, N), padding_mode=PAD_ZERO)
        hr_2d = ct.reshape(hr_tile, (N, N))
        for ct_idx in range(num_c_tiles):
            orig_tile = ct.load(
                orig, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            orig_2d = ct.reshape(orig_tile, (N, TILE_C))
            x_tile = ct.load(
                x, index=(pid, ct_idx), shape=(TILE_SIZE, TILE_C), padding_mode=PAD_ZERO
            )
            bias_tile = ct.load(bias, index=(ct_idx,), shape=(TILE_C,), padding_mode=PAD_ZERO)
            xb_2d = ct.reshape(x_tile, (1, TILE_C)) + ct.reshape(bias_tile, (1, TILE_C))
            out_2d = hp_2d * xb_2d
            for j in range(N):
                out_2d += ct.extract(hr_2d, (0, j), shape=(N, 1)) * ct.extract(
                    orig_2d, (j, 0), shape=(1, TILE_C)
                )
            ct.store(
                out,
                index=(pid, 0, ct_idx),
                tile=ct.reshape(out_2d, (TILE_SIZE, N, TILE_C)).astype(out.dtype),
            )

    @ct.kernel
    def _ct_hpb_bwd_kernel(
        go,
        hr,
        orig,
        hp,
        x,
        g_hr,
        g_orig,
        g_hp,
        g_x,
        N: ConstInt,
        TILE_C: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        pid = ct.bid(0)
        num_c_tiles = ct.cdiv(go.shape[2], TILE_C)
        hp_tile = ct.load(hp, index=(pid, 0), shape=(TILE_SIZE, N))
        hp_2d = ct.reshape(hp_tile, (1, N))
        hr_tile = ct.load(hr, index=(pid, 0, 0), shape=(TILE_SIZE, N, N), padding_mode=PAD_ZERO)
        hr_2d = ct.reshape(hr_tile, (N, N))
        acc_g_hp_2d = ct.full((N, 1), 0, dtype=ct.float32)
        acc_g_hr_2d = ct.full((N, N), 0, dtype=ct.float32)
        for ct_idx in range(num_c_tiles):
            x_tile = ct.load(
                x, index=(pid, ct_idx), shape=(TILE_SIZE, TILE_C), padding_mode=PAD_ZERO
            )
            x_2d = ct.reshape(x_tile, (1, TILE_C))
            go_tile = ct.load(
                go, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            go_2d = ct.reshape(go_tile, (N, TILE_C))
            orig_tile = ct.load(
                orig, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            orig_2d = ct.reshape(orig_tile, (N, TILE_C))
            g_x_2d = ct.full((1, TILE_C), 0, dtype=hp.dtype)
            g_orig_2d = ct.full((N, TILE_C), 0, dtype=hp.dtype)
            for j in range(N):
                g_x_2d += ct.extract(hp_2d, (0, j), shape=(1, 1)).item() * ct.extract(
                    go_2d, (j, 0), shape=(1, TILE_C)
                )
                g_orig_2d += ct.extract(hr_2d, (j, 0), shape=(1, N)).reshape((N, 1)) * ct.extract(
                    go_2d, (j, 0), shape=(1, TILE_C)
                )
            acc_g_hp_2d += ct.sum(go_2d * x_2d, axis=1, keepdims=True)
            acc_g_hr_2d += ct.sum(
                ct.expand_dims(go_2d, axis=1) * ct.expand_dims(orig_2d, axis=0), axis=2
            )
            ct.store(
                g_x,
                index=(pid, ct_idx),
                tile=ct.reshape(g_x_2d, (TILE_SIZE, TILE_C)).astype(g_x.dtype),
            )
            ct.store(
                g_orig,
                index=(pid, 0, ct_idx),
                tile=ct.reshape(g_orig_2d, (TILE_SIZE, N, TILE_C)).astype(g_orig.dtype),
            )
        ct.store(
            g_hp, index=(pid, 0), tile=ct.reshape(acc_g_hp_2d, (TILE_SIZE, N)).astype(g_hp.dtype)
        )
        ct.store(
            g_hr,
            index=(pid, 0, 0),
            tile=ct.reshape(acc_g_hr_2d, (TILE_SIZE, N, N)).astype(g_hr.dtype),
        )

    @ct.kernel
    def _ct_hpb_bwd_bias_kernel(
        go,
        hr,
        orig,
        hp,
        x,
        bias,
        g_hr,
        g_orig,
        g_hp,
        g_x,
        N: ConstInt,
        TILE_C: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        pid = ct.bid(0)
        num_c_tiles = ct.cdiv(go.shape[2], TILE_C)
        hp_tile = ct.load(hp, index=(pid, 0), shape=(TILE_SIZE, N))
        hp_2d = ct.reshape(hp_tile, (1, N))
        hr_tile = ct.load(hr, index=(pid, 0, 0), shape=(TILE_SIZE, N, N), padding_mode=PAD_ZERO)
        hr_2d = ct.reshape(hr_tile, (N, N))
        acc_g_hp_2d = ct.full((N, 1), 0, dtype=ct.float32)
        acc_g_hr_2d = ct.full((N, N), 0, dtype=ct.float32)
        for ct_idx in range(num_c_tiles):
            x_tile = ct.load(
                x, index=(pid, ct_idx), shape=(TILE_SIZE, TILE_C), padding_mode=PAD_ZERO
            )
            bias_tile = ct.load(bias, index=(ct_idx,), shape=(TILE_C,), padding_mode=PAD_ZERO)
            xb_2d = ct.reshape(x_tile, (1, TILE_C)) + ct.reshape(bias_tile, (1, TILE_C))
            go_tile = ct.load(
                go, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            go_2d = ct.reshape(go_tile, (N, TILE_C))
            orig_tile = ct.load(
                orig, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            orig_2d = ct.reshape(orig_tile, (N, TILE_C))
            g_x_2d = ct.full((1, TILE_C), 0, dtype=hp.dtype)
            g_orig_2d = ct.full((N, TILE_C), 0, dtype=hp.dtype)
            for j in range(N):
                g_x_2d += ct.extract(hp_2d, (0, j), shape=(1, 1)).item() * ct.extract(
                    go_2d, (j, 0), shape=(1, TILE_C)
                )
                g_orig_2d += ct.extract(hr_2d, (j, 0), shape=(1, N)).reshape((N, 1)) * ct.extract(
                    go_2d, (j, 0), shape=(1, TILE_C)
                )
            acc_g_hp_2d += ct.sum(go_2d * xb_2d, axis=1, keepdims=True)
            acc_g_hr_2d += ct.sum(
                ct.expand_dims(go_2d, axis=1) * ct.expand_dims(orig_2d, axis=0), axis=2
            )
            ct.store(
                g_x,
                index=(pid, ct_idx),
                tile=ct.reshape(g_x_2d, (TILE_SIZE, TILE_C)).astype(g_x.dtype),
            )
            ct.store(
                g_orig,
                index=(pid, 0, ct_idx),
                tile=ct.reshape(g_orig_2d, (TILE_SIZE, N, TILE_C)).astype(g_orig.dtype),
            )
        ct.store(
            g_hp, index=(pid, 0), tile=ct.reshape(acc_g_hp_2d, (TILE_SIZE, N)).astype(g_hp.dtype)
        )
        ct.store(
            g_hr,
            index=(pid, 0, 0),
            tile=ct.reshape(acc_g_hr_2d, (TILE_SIZE, N, N)).astype(g_hr.dtype),
        )

    def _cutile_h_post_bda_fwd(
        h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        s, b, n, C = original_residual.shape
        sb = s * b
        TILE_C = math.gcd(C, 1024)
        TILE_SIZE = math.gcd(sb, 1)
        out = torch.empty(sb, n, C, dtype=h_res.dtype, device=h_res.device)
        grid = (math.ceil(sb / TILE_SIZE),)
        if bias is not None:
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _ct_hpb_fwd_bias_kernel,
                (
                    h_res.view(sb, n, n),
                    original_residual.view(sb, n, C),
                    h_post.view(sb, n),
                    x.view(sb, C),
                    bias,
                    out,
                    n,
                    TILE_C,
                    TILE_SIZE,
                ),
            )
        else:
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _ct_hpb_fwd_kernel,
                (
                    h_res.view(sb, n, n),
                    original_residual.view(sb, n, C),
                    h_post.view(sb, n),
                    x.view(sb, C),
                    out,
                    n,
                    TILE_C,
                    TILE_SIZE,
                ),
            )
        return out.view(s, b, n, C)

    def _cutile_h_post_bda_bwd(
        grad_output: Tensor,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        s, b, n, C = original_residual.shape
        sb = s * b
        TILE_C = math.gcd(C, 1024)
        TILE_SIZE = math.gcd(sb, 1)
        g_hr = torch.empty(sb, n, n, dtype=h_res.dtype, device=h_res.device)
        g_res = torch.empty(sb, n, C, dtype=h_res.dtype, device=h_res.device)
        g_hp = torch.empty(sb, n, dtype=h_res.dtype, device=h_res.device)
        g_x = torch.empty(sb, C, dtype=h_res.dtype, device=h_res.device)
        grid = (sb,)
        if bias is not None:
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _ct_hpb_bwd_bias_kernel,
                (
                    grad_output.view(sb, n, C),
                    h_res.view(sb, n, n),
                    original_residual.view(sb, n, C),
                    h_post.view(sb, n),
                    x.view(sb, C),
                    bias,
                    g_hr,
                    g_res,
                    g_hp,
                    g_x,
                    n,
                    TILE_C,
                    TILE_SIZE,
                ),
            )
        else:
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _ct_hpb_bwd_kernel,
                (
                    grad_output.view(sb, n, C),
                    h_res.view(sb, n, n),
                    original_residual.view(sb, n, C),
                    h_post.view(sb, n),
                    x.view(sb, C),
                    g_hr,
                    g_res,
                    g_hp,
                    g_x,
                    n,
                    TILE_C,
                    TILE_SIZE,
                ),
            )
        g_bias = g_x.sum(dim=0) if bias is not None else None
        return (
            g_hr.view(s, b, n, n),
            g_res.view(s, b, n, C),
            g_hp.view(s, b, n),
            g_x.view(s, b, C),
            g_bias,
        )

    # -- Proj RMS kernels ----------------------------------------------------

    @ct.function
    def _ct_rms_dnorm(a_tile, norm_tile, dr_tile, K):
        inv_norm = ct.where(norm_tile > 0, 1.0 / norm_tile, 0.0)
        inv_sqrt_k = 1.0 / ct.sqrt(K)
        eps = 1e-8
        u = norm_tile * inv_sqrt_k + eps
        coeff = -(1.0 / (u * u)) * inv_sqrt_k
        return dr_tile * coeff * a_tile * inv_norm

    @ct.kernel
    def _ct_proj_rms_fwd_kernel(
        A,
        B,
        PROJ,
        NORM,
        R,
        M: int,
        N: int,
        K: int,
        eps: float,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        TILE_K: ConstInt,
    ):
        tile_m_id = ct.bid(0)
        num_k_tiles = ct.cdiv(K, TILE_K)
        acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        sum_sq = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
        for tile_k_id in range(num_k_tiles):
            a_tile = ct.load(
                A, index=(tile_m_id, tile_k_id), shape=(TILE_M, TILE_K), padding_mode=PAD_ZERO
            )
            b_tile = ct.load(B, index=(0, tile_k_id), shape=(TILE_N, TILE_K), padding_mode=PAD_ZERO)
            acc = ct.mma(
                a_tile.astype(ct.tfloat32), b_tile.transpose().astype(ct.tfloat32), acc=acc
            )
            sum_sq += ct.sum(a_tile * a_tile, axis=1, keepdims=True)
        norm_tile = ct.sqrt(sum_sq)
        v = norm_tile / ct.sqrt(K) + eps
        r_tile = 1.0 / v
        ct.store(PROJ, index=(tile_m_id, 0), tile=acc.astype(PROJ.dtype))
        ct.store(NORM, index=(tile_m_id, 0), tile=norm_tile.astype(NORM.dtype))
        ct.store(R, index=(tile_m_id, 0), tile=r_tile.astype(R.dtype))

    @ct.kernel
    def _ct_proj_rms_bwd_kernel(
        A,
        B,
        NORM,
        DD,
        DR,
        DA,
        DB,
        M: int,
        N: int,
        K: int,
        TILE_SIZE_M: ConstInt,
        TILE_SIZE_N: ConstInt,
        TILE_SIZE_K: ConstInt,
    ):
        zero_pad = ct.PaddingMode.ZERO
        tile_k_id = ct.bid(0)
        NUM_M_TILES = ct.cdiv(M, TILE_SIZE_M)
        accumulator_db = ct.full((TILE_SIZE_K, TILE_SIZE_N), 0.0, dtype=ct.float32)
        for tile_m_id in range(NUM_M_TILES):
            accumulator_da = ct.full((TILE_SIZE_M, TILE_SIZE_K), 0.0, dtype=ct.float32)
            a_tile = ct.load(
                A,
                index=(tile_m_id, tile_k_id),
                shape=(TILE_SIZE_M, TILE_SIZE_K),
                padding_mode=zero_pad,
            )
            norm_tile = ct.load(
                NORM, index=(tile_m_id, 0), shape=(TILE_SIZE_M, 1), padding_mode=zero_pad
            )
            dr_tile = ct.load(
                DR, index=(tile_m_id, 0), shape=(TILE_SIZE_M, 1), padding_mode=zero_pad
            )
            accumulator_da = accumulator_da + _ct_rms_dnorm(a_tile, norm_tile, dr_tile, K)
            b_tile = ct.load(
                B, index=(0, tile_k_id), shape=(TILE_SIZE_N, TILE_SIZE_K), padding_mode=zero_pad
            )
            dd_tile = ct.load(
                DD, index=(tile_m_id, 0), shape=(TILE_SIZE_M, TILE_SIZE_N), padding_mode=zero_pad
            )
            dd_tile = ct.astype(dd_tile, ct.tfloat32)
            accumulator_da = ct.mma(dd_tile, b_tile.astype(ct.tfloat32), acc=accumulator_da)
            ct.store(DA, index=(tile_m_id, tile_k_id), tile=accumulator_da.astype(DA.dtype))
            accumulator_db = ct.mma(
                a_tile.transpose().astype(ct.tfloat32), dd_tile, acc=accumulator_db
            )
        ct.store(DB, index=(0, tile_k_id), tile=accumulator_db.transpose().astype(DB.dtype))

    @ct.kernel
    def _ct_proj_rms_bwd_small_k_kernel(
        A, B, NORM, DD, DR, DA, DB, M: int, N: int, K: int, TILE_N_SIZE: ConstInt
    ):
        zero_pad = ct.PaddingMode.ZERO
        TILE_DB_SIZE_M = 128
        TILE_DB_SIZE_K = 64
        NUM_M_TILES = ct.cdiv(M, TILE_DB_SIZE_M)
        NUM_K_TILES = ct.cdiv(K, TILE_DB_SIZE_K)
        if ct.bid(1) == 0:
            for tile_id in range(ct.bid(0), NUM_K_TILES, ct.num_blocks(0)):
                accumulator_db = ct.full((TILE_DB_SIZE_K, TILE_N_SIZE), 0.0, dtype=ct.float32)
                for m_tile in range(NUM_M_TILES):
                    a_tile = ct.load(
                        A,
                        index=(m_tile, tile_id),
                        shape=(TILE_DB_SIZE_M, TILE_DB_SIZE_K),
                        padding_mode=zero_pad,
                    )
                    dd_tile = ct.load(
                        DD,
                        index=(m_tile, 0),
                        shape=(TILE_DB_SIZE_M, TILE_N_SIZE),
                        padding_mode=zero_pad,
                    )
                    accumulator_db = ct.mma(
                        a_tile.transpose().astype(ct.tfloat32),
                        dd_tile.astype(ct.tfloat32),
                        acc=accumulator_db,
                    )
                ct.store(
                    DB,
                    index=(0, tile_id),
                    tile=accumulator_db.transpose().astype(DB.dtype),
                    allow_tma=False,
                )
        TILE_DA_SIZE_M = 128
        TILE_DA_SIZE_K = 256
        NUM_DA_TILES = ct.cdiv(M, TILE_DA_SIZE_M) * ct.cdiv(K, TILE_DA_SIZE_K)
        NUM_DA_K_TILES = ct.cdiv(K, TILE_DA_SIZE_K)
        if ct.bid(1) == 1:
            for tile_id in range(ct.bid(0), NUM_DA_TILES, ct.num_blocks(0)):
                b_tile_idx = tile_id % NUM_DA_K_TILES
                dd_tile_idx = tile_id // NUM_DA_K_TILES
                accumulator_da = ct.full((TILE_DA_SIZE_M, TILE_DA_SIZE_K), 0.0, dtype=ct.float32)
                a_tile = ct.load(
                    A,
                    index=(dd_tile_idx, b_tile_idx),
                    shape=(TILE_DA_SIZE_M, TILE_DA_SIZE_K),
                    padding_mode=zero_pad,
                )
                norm_tile = ct.load(
                    NORM, index=(dd_tile_idx, 0), shape=(TILE_DA_SIZE_M, 1), padding_mode=zero_pad
                )
                dr_tile = ct.load(
                    DR, index=(dd_tile_idx, 0), shape=(TILE_DA_SIZE_M, 1), padding_mode=zero_pad
                )
                accumulator_da = accumulator_da + _ct_rms_dnorm(
                    a_tile.astype(ct.float32), norm_tile, dr_tile, K
                )
                b_tile = ct.load(
                    B,
                    index=(0, b_tile_idx),
                    shape=(TILE_N_SIZE, TILE_DA_SIZE_K),
                    padding_mode=zero_pad,
                )
                dd_tile = ct.load(
                    DD,
                    index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, TILE_N_SIZE),
                    padding_mode=zero_pad,
                )
                accumulator_da = ct.mma(
                    dd_tile.astype(ct.tfloat32), b_tile.astype(ct.tfloat32), acc=accumulator_da
                )
                ct.store(DA, index=(dd_tile_idx, b_tile_idx), tile=accumulator_da.astype(DA.dtype))

    def _next_power_of_2(n: int) -> int:
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        n += 1
        return n

    def _cutile_proj_rms_fwd(
        x: Tensor, weight: Tensor, eps: float = 1e-8
    ) -> Tuple[Tensor, Tensor, Tensor]:
        M, K = x.shape
        N = weight.shape[0]
        TILE_M = 128
        TILE_N = _next_power_of_2(N)
        TILE_K = 128
        num_tiles_m = math.ceil(M / TILE_M)
        proj = torch.empty(M, N, dtype=x.dtype, device=x.device)
        norm = torch.empty(M, 1, dtype=x.dtype, device=x.device)
        r = torch.empty(M, 1, dtype=x.dtype, device=x.device)
        ct.launch(
            torch.cuda.current_stream(),
            (num_tiles_m,),
            _ct_proj_rms_fwd_kernel,
            (x, weight, proj, norm, r, M, N, K, eps, TILE_M, TILE_N, TILE_K),
        )
        return proj, norm, r

    def _cutile_proj_rms_bwd(
        grad_proj: Tensor,
        grad_r: Tensor,
        x: Tensor,
        weight: Tensor,
        norm: Tensor,
        eps: float = 1e-8,
    ) -> Tuple[Tensor, Tensor]:
        M, K = x.shape
        N = weight.shape[0]
        da = torch.empty_like(x)
        db = torch.empty_like(weight)
        TILE_SIZE_N = _next_power_of_2(N)
        assert TILE_SIZE_N <= 256, f"TILE_SIZE_N too large: {TILE_SIZE_N}"
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        if K >= 8192:
            TILE_SIZE_M, TILE_SIZE_K = 128, 128
            grid = (math.ceil(K / TILE_SIZE_K), 1)
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _ct_proj_rms_bwd_kernel,
                (
                    x,
                    weight,
                    norm,
                    grad_proj,
                    grad_r,
                    da,
                    db,
                    M,
                    N,
                    K,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    TILE_SIZE_K,
                ),
            )
        else:
            grid = (num_sms, 2, 1)
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _ct_proj_rms_bwd_small_k_kernel,
                (x, weight, norm, grad_proj, grad_r, da, db, M, N, K, TILE_SIZE_N),
            )
        return da, db


# ============================================================================
# Autograd Functions (cuTile only – guarded by _CUTILE_AVAILABLE)
# ============================================================================

if not _CUTILE_AVAILABLE:

    def _no_cutile_error(*_args, **_kwargs):
        raise RuntimeError(
            "Fused mHC kernels require cuda.tile (cuTile) which is not installed. "
            "Either install cuTile or set use_fused_mhc=False to use reference "
            "implementations."
        )

    fused_sinkhorn = _no_cutile_error
    fused_h_aggregate = _no_cutile_error
    fused_h_post_bda = _no_cutile_error
    fused_proj_rms = _no_cutile_error

else:

    class FusedSinkhornKnopp(torch.autograd.Function):
        """Fused Sinkhorn-Knopp projection to doubly stochastic matrix (cuTile)."""

        @staticmethod
        def forward(ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6):
            """cuTile fused Sinkhorn forward."""
            output, M_init = _cutile_sinkhorn_fwd(input_logits, num_iterations, eps)
            ctx.save_for_backward(M_init)
            ctx.num_iterations = num_iterations
            ctx.eps = eps
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """cuTile fused Sinkhorn backward."""
            (M_init,) = ctx.saved_tensors
            grad_input = _cutile_sinkhorn_bwd(grad_output, M_init, ctx.num_iterations, ctx.eps)
            return grad_input, None, None

    class FusedHAggregate(torch.autograd.Function):
        """Fused n-stream weighted aggregation (cuTile)."""

        @staticmethod
        def forward(ctx, x: Tensor, h_pre: Tensor):
            """cuTile fused h_aggregate forward."""
            output = _cutile_h_aggregate_fwd(x, h_pre)
            ctx.save_for_backward(x, h_pre)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """cuTile fused h_aggregate backward."""
            x, h_pre = ctx.saved_tensors
            return _cutile_h_aggregate_bwd(grad_output, x, h_pre)

    class FusedHPostBDA(torch.autograd.Function):
        """Fused: output = H_res @ orig_res + H_post * (x [+ bias]) (cuTile)."""

        @staticmethod
        def forward(
            ctx,
            h_res: Tensor,
            original_residual: Tensor,
            h_post: Tensor,
            x: Tensor,
            bias: Optional[Tensor],
        ):
            """cuTile fused h_post_bda forward."""
            output = _cutile_h_post_bda_fwd(h_res, original_residual, h_post, x, bias)
            if bias is not None:
                ctx.save_for_backward(h_res, original_residual, h_post, x, bias)
                ctx.has_bias = True
            else:
                ctx.save_for_backward(h_res, original_residual, h_post, x)
                ctx.has_bias = False
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """cuTile fused h_post_bda backward."""
            if ctx.has_bias:
                h_res, orig_res, h_post, x, bias = ctx.saved_tensors
            else:
                h_res, orig_res, h_post, x = ctx.saved_tensors
                bias = None
            return _cutile_h_post_bda_bwd(grad_output, h_res, orig_res, h_post, x, bias)

    class FusedProjRms(torch.autograd.Function):
        """Fused projection + RMS normalization (cuTile)."""

        @staticmethod
        def forward(ctx, x: Tensor, weight: Tensor, eps: float = 1e-6):
            """cuTile fused proj_rms forward."""
            proj, norm, r = _cutile_proj_rms_fwd(x, weight, eps)
            ctx.save_for_backward(x, weight, norm)
            ctx.eps = eps
            return proj, r

        @staticmethod
        def backward(ctx, grad_proj, grad_r):
            """cuTile fused proj_rms backward."""
            x, weight, norm = ctx.saved_tensors
            grad_x, grad_weight = _cutile_proj_rms_bwd(grad_proj, grad_r, x, weight, norm, ctx.eps)
            return grad_x, grad_weight, None

    # ========================================================================
    # Public API (only available when cuTile is installed)
    # ========================================================================

    def fused_sinkhorn(input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
        """Project logits to doubly stochastic matrix via Sinkhorn-Knopp.

        Args:
            input_logits: [..., n, n] raw logits
            num_iterations: Sinkhorn iterations
            eps: numerical stability

        Returns:
            [..., n, n] doubly stochastic matrix
        """
        return FusedSinkhornKnopp.apply(input_logits, num_iterations, eps)

    def fused_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
        """Weighted n-stream to 1-stream aggregation.

        Args:
            x: [s, b, n, C] n-stream hidden states
            h_pre: [s, b, n] aggregation weights

        Returns:
            [s, b, C] aggregated hidden states
        """
        return FusedHAggregate.apply(x, h_pre)

    def fused_h_post_bda(
        h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        """Fused H_res @ residual + H_post * (x + bias).

        Args:
            h_res: [s, b, n, n] residual mixing matrix
            original_residual: [s, b, n, C] n-stream residual
            h_post: [s, b, n] expansion weights
            x: [s, b, C] layer output
            bias: [C] or None

        Returns:
            [s, b, n, C] fused output
        """
        return FusedHPostBDA.apply(h_res, original_residual, h_post, x, bias)

    def fused_proj_rms(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
        """Fused projection + RMS normalization.

        Args:
            x: [M, K] input
            weight: [N, K] projection weight
            eps: stability epsilon

        Returns:
            proj: [M, N] = x @ weight^T
            r: [M, 1] = 1 / (||x|| / sqrt(K) + eps)
        """
        return FusedProjRms.apply(x, weight, eps)
