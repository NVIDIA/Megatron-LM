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
_CUTILE_EXPERIMENTAL_AVAILABLE = False
try:
    import cuda.tile as ct

    _CUTILE_AVAILABLE = True
    try:
        import cuda.tile_experimental as ct_experimental

        _CUTILE_EXPERIMENTAL_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


def is_cutile_available() -> bool:
    """Return True if cuTile fused kernels are available."""
    return _CUTILE_AVAILABLE


# ---------------------------------------------------------------------------
# Check Triton availability
# ---------------------------------------------------------------------------
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    pass


def is_triton_available() -> bool:
    """Return True if Triton is available for supported mHC kernels."""
    return _TRITON_AVAILABLE


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

    def _sinkhorn_autotune_tile_sizes(N_batch):
        """Generate autotune search space for sinkhorn kernels."""
        for ts in (1, 2, 4, 8, 16, 32, 64, 128):
            if ts <= N_batch:
                yield ts

    _sinkhorn_fwd_best_cfg: dict = {}
    _sinkhorn_bwd_best_cfg: dict = {}

    def _cutile_sinkhorn_fwd(
        input_logits: Tensor, num_iterations: int, eps: float = 1e-8
    ) -> Tuple[Tensor, Tensor]:
        original_shape = input_logits.shape
        hc = original_shape[-1]
        N_batch = input_logits.numel() // (hc * hc)
        dev = input_logits.device
        stream = torch.cuda.current_stream()
        out = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
        M_init = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
        inp = input_logits.view(N_batch, hc, hc)

        cache_key = (N_batch, hc, num_iterations)
        cached = _sinkhorn_fwd_best_cfg.get(cache_key)

        if cached is not None or not _CUTILE_EXPERIMENTAL_AVAILABLE:
            ts = cached if cached is not None else math.gcd(N_batch, 128)
            ct.launch(
                stream,
                (math.ceil(N_batch / ts), 1, 1),
                _ct_sinkhorn_fwd_kernel,
                (inp, out, M_init, eps, hc, num_iterations, ts),
            )
        else:
            from types import SimpleNamespace

            configs = [
                SimpleNamespace(TILE_SIZE=ts)
                for ts in _sinkhorn_autotune_tile_sizes(N_batch)
            ]
            tuned = ct_experimental.autotune_launch(
                stream,
                grid_fn=lambda cfg: (math.ceil(N_batch / cfg.TILE_SIZE), 1, 1),
                kernel=_ct_sinkhorn_fwd_kernel,
                args_fn=lambda cfg: (
                    inp, out, M_init, eps, hc, num_iterations, cfg.TILE_SIZE,
                ),
                search_space=configs,
            )
            best_ts = tuned.tuned_config.TILE_SIZE
            _sinkhorn_fwd_best_cfg[cache_key] = best_ts
            ct.launch(
                stream,
                (math.ceil(N_batch / best_ts), 1, 1),
                _ct_sinkhorn_fwd_kernel,
                (inp, out, M_init, eps, hc, num_iterations, best_ts),
            )

        return out.view(original_shape), M_init.view(original_shape)

    def _cutile_sinkhorn_bwd(
        grad_output: Tensor, M_init: Tensor, num_iterations: int, eps: float = 1e-8
    ) -> Tensor:
        original_shape = grad_output.shape
        hc = original_shape[-1]
        N_batch = grad_output.numel() // (hc * hc)
        dev = grad_output.device
        stream = torch.cuda.current_stream()
        grad_input = torch.empty(N_batch, hc, hc, dtype=grad_output.dtype, device=dev)
        go = grad_output.view(N_batch, hc, hc)
        mi = M_init.view(N_batch, hc, hc)

        cache_key = (N_batch, hc, num_iterations)
        cached = _sinkhorn_bwd_best_cfg.get(cache_key)

        def _alloc_and_launch(ts):
            ws_M = torch.empty(N_batch * 2 * num_iterations, hc, hc, dtype=torch.float32, device=dev)
            ws_rs = torch.empty(N_batch * num_iterations, hc, 1, dtype=torch.float32, device=dev)
            ws_cs = torch.empty(N_batch * num_iterations, 1, hc, dtype=torch.float32, device=dev)
            ct.launch(
                stream,
                (math.ceil(N_batch / ts), 1, 1),
                _ct_sinkhorn_bwd_kernel,
                (go, mi, grad_input, ws_M, ws_rs, ws_cs, eps, hc, num_iterations, ts),
            )

        if cached is not None or not _CUTILE_EXPERIMENTAL_AVAILABLE:
            ts = cached if cached is not None else math.gcd(N_batch, 128)
            _alloc_and_launch(ts)
        else:
            from types import SimpleNamespace

            configs = [
                SimpleNamespace(TILE_SIZE=ts)
                for ts in _sinkhorn_autotune_tile_sizes(N_batch)
            ]
            # Allocate workspace for largest tile size (all configs share same workspace shape).
            ws_M = torch.empty(N_batch * 2 * num_iterations, hc, hc, dtype=torch.float32, device=dev)
            ws_rs = torch.empty(N_batch * num_iterations, hc, 1, dtype=torch.float32, device=dev)
            ws_cs = torch.empty(N_batch * num_iterations, 1, hc, dtype=torch.float32, device=dev)
            tuned = ct_experimental.autotune_launch(
                stream,
                grid_fn=lambda cfg: (math.ceil(N_batch / cfg.TILE_SIZE), 1, 1),
                kernel=_ct_sinkhorn_bwd_kernel,
                args_fn=lambda cfg: (
                    go, mi, grad_input, ws_M, ws_rs, ws_cs, eps, hc, num_iterations,
                    cfg.TILE_SIZE,
                ),
                search_space=configs,
            )
            best_ts = tuned.tuned_config.TILE_SIZE
            _sinkhorn_bwd_best_cfg[cache_key] = best_ts
            # Re-launch with best config.
            _alloc_and_launch(best_ts)

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

    def _h_agg_autotune_configs(sb, C):
        """Generate autotune search space for h_aggregate kernels."""
        for tile_m in (1, 2, 4, 8):
            for tile_c in (32, 64, 128, 256, 512, 1024):
                if tile_m <= sb and tile_c <= C:
                    yield {"TILE_M": tile_m, "TILE_C": tile_c}

    _h_agg_fwd_best_cfg: dict = {}
    _h_agg_bwd_best_cfg: dict = {}

    def _cutile_h_aggregate_fwd(x: Tensor, h_pre: Tensor) -> Tensor:
        s, b, n, C = x.shape
        sb = s * b
        stream = torch.cuda.current_stream()
        out = torch.empty(sb, C, dtype=x.dtype, device=x.device)
        x_flat = x.view(sb, n, C)
        h_flat = h_pre.view(sb, n)

        cache_key = (sb, n, C)
        cached = _h_agg_fwd_best_cfg.get(cache_key)

        # Autotune disabled — causes cudaErrorLaunchFailure during training.
        if cached is not None:
            tm, tc = cached
        else:
            tm, tc = math.gcd(sb, 4), math.gcd(C, 1024)
        ct.launch(stream, (math.ceil(sb / tm),), _ct_h_agg_fwd_kernel,
                  (x_flat, h_flat, out, n, tm, tc))

        return out.view(s, b, C)

    def _cutile_h_aggregate_bwd(
        grad_output: Tensor, x: Tensor, h_pre: Tensor
    ) -> Tuple[Tensor, Tensor]:
        s, b, n, C = x.shape
        sb = s * b
        stream = torch.cuda.current_stream()
        gx = torch.empty(sb, n, C, dtype=x.dtype, device=x.device)
        gh = torch.empty(sb, n, dtype=h_pre.dtype, device=x.device)
        go_flat = grad_output.view(sb, C)
        x_flat = x.view(sb, n, C)
        h_flat = h_pre.view(sb, n)

        cache_key = (sb, n, C)
        cached = _h_agg_bwd_best_cfg.get(cache_key)

        # Autotune disabled — causes cudaErrorLaunchFailure during training.
        if cached is not None:
            tm, tc = cached
        else:
            tm, tc = math.gcd(sb, 4), math.gcd(C, 1024)
        ct.launch(stream, (math.ceil(sb / tm),), _ct_h_agg_bwd_kernel,
                  (go_flat, x_flat, h_flat, gx, gh, n, tm, tc))

        return gx.view(s, b, n, C), gh.view(s, b, n)

    # -- H_post BDA kernels --------------------------------------------------

    @ct.kernel
    def _ct_hpb_fwd_kernel(
        hr, orig, hp, x, out, N: ConstInt, TILE_C: ConstInt, TILE_SIZE: ConstInt
    ):
        pid = ct.bid(0)
        num_c_tiles = ct.num_tiles(x, axis=1, shape=(TILE_SIZE, TILE_C))
        hp_tile = ct.load(hp, index=(pid, 0), shape=(TILE_SIZE, N), padding_mode=PAD_ZERO)
        hp_exp = ct.expand_dims(hp_tile, axis=2)  # (TILE_SIZE, N, 1)
        hr_tile = ct.load(hr, index=(pid, 0, 0), shape=(TILE_SIZE, N, N), padding_mode=PAD_ZERO)
        for ct_idx in range(num_c_tiles):
            orig_tile = ct.load(
                orig, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            x_tile = ct.load(
                x, index=(pid, ct_idx), shape=(TILE_SIZE, TILE_C), padding_mode=PAD_ZERO
            )
            x_exp = ct.expand_dims(x_tile, axis=1)  # (TILE_SIZE, 1, TILE_C)
            out_tile = hp_exp * x_exp  # (TILE_SIZE, N, TILE_C)
            for j in range(N):
                hr_col = ct.extract(hr_tile, (0, 0, j), shape=(TILE_SIZE, N, 1))
                orig_row = ct.extract(orig_tile, (0, j, 0), shape=(TILE_SIZE, 1, TILE_C))
                out_tile = out_tile + hr_col * orig_row
            ct.store(out, index=(pid, 0, ct_idx), tile=out_tile.astype(out.dtype))

    @ct.kernel
    def _ct_hpb_fwd_bias_kernel(
        hr, orig, hp, x, bias, out, N: ConstInt, TILE_C: ConstInt, TILE_SIZE: ConstInt
    ):
        pid = ct.bid(0)
        num_c_tiles = ct.num_tiles(x, axis=1, shape=(TILE_SIZE, TILE_C))
        hp_tile = ct.load(hp, index=(pid, 0), shape=(TILE_SIZE, N), padding_mode=PAD_ZERO)
        hp_exp = ct.expand_dims(hp_tile, axis=2)  # (TILE_SIZE, N, 1)
        hr_tile = ct.load(hr, index=(pid, 0, 0), shape=(TILE_SIZE, N, N), padding_mode=PAD_ZERO)
        for ct_idx in range(num_c_tiles):
            orig_tile = ct.load(
                orig, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            x_tile = ct.load(
                x, index=(pid, ct_idx), shape=(TILE_SIZE, TILE_C), padding_mode=PAD_ZERO
            )
            bias_tile = ct.load(bias, index=(ct_idx,), shape=(TILE_C,), padding_mode=PAD_ZERO)
            xb_exp = ct.expand_dims(x_tile + bias_tile, axis=1)  # (TILE_SIZE, 1, TILE_C)
            out_tile = hp_exp * xb_exp  # (TILE_SIZE, N, TILE_C)
            for j in range(N):
                hr_col = ct.extract(hr_tile, (0, 0, j), shape=(TILE_SIZE, N, 1))
                orig_row = ct.extract(orig_tile, (0, j, 0), shape=(TILE_SIZE, 1, TILE_C))
                out_tile = out_tile + hr_col * orig_row
            ct.store(out, index=(pid, 0, ct_idx), tile=out_tile.astype(out.dtype))

    @ct.kernel
    def _ct_hpb_bwd_g_x_orig_kernel(
        go, hr, hp,
        g_orig, g_x,
        N: ConstInt, TILE_C: ConstInt, TILE_SIZE: ConstInt,
    ):
        """Compute g_x = hp @ go and g_orig = hr.T @ go.

        Grid: (ceil(sb / TILE_SIZE), ceil(C / TILE_C)).
        2D grid — no loop, no accumulators.
        """
        pid = ct.bid(0)
        ct_idx = ct.bid(1)
        hp_tile = ct.load(hp, index=(pid, 0), shape=(TILE_SIZE, N), padding_mode=PAD_ZERO)
        hr_tile = ct.load(hr, index=(pid, 0, 0), shape=(TILE_SIZE, N, N), padding_mode=PAD_ZERO)
        go_tile = ct.load(
            go, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
        )
        g_x_tile = ct.full((TILE_SIZE, 1, TILE_C), 0, dtype=hp.dtype)
        g_orig_tile = ct.full((TILE_SIZE, N, TILE_C), 0, dtype=hp.dtype)
        for j in range(N):
            hp_j = ct.extract(hp_tile, (0, j), shape=(TILE_SIZE, 1))
            hp_j_exp = ct.expand_dims(hp_j, axis=2)  # [TS, 1, 1]
            go_j = ct.extract(go_tile, (0, j, 0), shape=(TILE_SIZE, 1, TILE_C))
            g_x_tile = g_x_tile + hp_j_exp * go_j
            hr_row_j = ct.extract(hr_tile, (0, j, 0), shape=(TILE_SIZE, 1, N))
            g_orig_tile = g_orig_tile + ct.reshape(hr_row_j, (TILE_SIZE, N, 1)) * go_j
        ct.store(
            g_x, index=(pid, ct_idx),
            tile=ct.reshape(g_x_tile, (TILE_SIZE, TILE_C)).astype(g_x.dtype),
        )
        ct.store(g_orig, index=(pid, 0, ct_idx), tile=g_orig_tile.astype(g_orig.dtype))

    @ct.kernel
    def _ct_hpb_bwd_g_hp_hr_kernel(
        go, orig, x,
        g_hr, g_hp,
        N: ConstInt, TILE_C: ConstInt, TILE_SIZE: ConstInt,
    ):
        """Compute g_hp = sum(go * x) and g_hr = go @ orig.T (no bias).

        Grid: (ceil(sb / TILE_SIZE),).  Loops over C-tiles.
        """
        pid = ct.bid(0)
        num_c_tiles = ct.cdiv(go.shape[2], TILE_C)
        acc_g_hp = ct.full((TILE_SIZE, N, 1), 0, dtype=ct.float32)
        acc_g_hr = ct.full((TILE_SIZE, N, N), 0, dtype=ct.float32)
        for ct_idx in range(num_c_tiles):
            x_tile = ct.load(x, index=(pid, ct_idx), shape=(TILE_SIZE, TILE_C), padding_mode=PAD_ZERO)
            x_exp = ct.expand_dims(x_tile, axis=1)  # [TS, 1, TC]
            go_tile = ct.load(
                go, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            orig_tile = ct.load(
                orig, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            acc_g_hp = acc_g_hp + ct.sum(go_tile * x_exp, axis=2, keepdims=True)
            acc_g_hr = acc_g_hr + ct.sum(
                ct.expand_dims(go_tile, axis=2) * ct.expand_dims(orig_tile, axis=1), axis=3
            )
        ct.store(
            g_hp, index=(pid, 0),
            tile=ct.reshape(acc_g_hp, (TILE_SIZE, N)).astype(g_hp.dtype),
        )
        ct.store(g_hr, index=(pid, 0, 0), tile=acc_g_hr.astype(g_hr.dtype))

    @ct.kernel
    def _ct_hpb_bwd_g_hp_hr_bias_kernel(
        go, orig, x, bias,
        g_hr, g_hp,
        N: ConstInt, TILE_C: ConstInt, TILE_SIZE: ConstInt,
    ):
        """Compute g_hp = sum(go * (x+bias)) and g_hr = go @ orig.T (with bias).

        Grid: (ceil(sb / TILE_SIZE),).  Loops over C-tiles.
        """
        pid = ct.bid(0)
        num_c_tiles = ct.cdiv(go.shape[2], TILE_C)
        acc_g_hp = ct.full((TILE_SIZE, N, 1), 0, dtype=ct.float32)
        acc_g_hr = ct.full((TILE_SIZE, N, N), 0, dtype=ct.float32)
        for ct_idx in range(num_c_tiles):
            x_tile = ct.load(x, index=(pid, ct_idx), shape=(TILE_SIZE, TILE_C), padding_mode=PAD_ZERO)
            bias_tile = ct.load(bias, index=(ct_idx,), shape=(TILE_C,), padding_mode=PAD_ZERO)
            xb_exp = ct.expand_dims(x_tile + bias_tile, axis=1)  # [TS, 1, TC]
            go_tile = ct.load(
                go, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            orig_tile = ct.load(
                orig, index=(pid, 0, ct_idx), shape=(TILE_SIZE, N, TILE_C), padding_mode=PAD_ZERO
            )
            acc_g_hp = acc_g_hp + ct.sum(go_tile * xb_exp, axis=2, keepdims=True)
            acc_g_hr = acc_g_hr + ct.sum(
                ct.expand_dims(go_tile, axis=2) * ct.expand_dims(orig_tile, axis=1), axis=3
            )
        ct.store(
            g_hp, index=(pid, 0),
            tile=ct.reshape(acc_g_hp, (TILE_SIZE, N)).astype(g_hp.dtype),
        )
        ct.store(g_hr, index=(pid, 0, 0), tile=acc_g_hr.astype(g_hr.dtype))

    # -- H_post BDA autotune configs & caches --------------------------------

    def _hpb_autotune_configs(sb, C):
        """Generate TILE_SIZE × TILE_C search space for h_post_bda kernels."""
        for tile_size in (1, 2, 4, 8):
            for tile_c in (32, 64, 128, 256, 512, 1024):
                if tile_c <= C and tile_size <= sb:
                    yield {"TILE_SIZE": tile_size, "TILE_C": tile_c}

    _hpb_fwd_best_cfg: dict = {}
    _hpb_bwd_g_x_orig_best_cfg: dict = {}
    _hpb_bwd_g_hp_hr_best_cfg: dict = {}

    def _cutile_h_post_bda_fwd(
        h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        s, b, n, C = original_residual.shape
        sb = s * b
        stream = torch.cuda.current_stream()
        out = torch.empty(sb, n, C, dtype=h_res.dtype, device=h_res.device)
        hr_flat = h_res.view(sb, n, n)
        orig_flat = original_residual.view(sb, n, C)
        hp_flat = h_post.view(sb, n)
        x_flat = x.view(sb, C)

        cache_key = (sb, n, C, bias is not None)
        cached = _hpb_fwd_best_cfg.get(cache_key)
        kernel = _ct_hpb_fwd_bias_kernel if bias is not None else _ct_hpb_fwd_kernel

        # Autotune disabled — causes cudaErrorLaunchFailure during training.
        if cached is not None:
            ts, tc = cached
        else:
            ts, tc = 1, math.gcd(C, 1024)
        args = (hr_flat, orig_flat, hp_flat, x_flat)
        if bias is not None:
            args = args + (bias,)
        args = args + (out, n, tc, ts)
        ct.launch(stream, (math.ceil(sb / ts),), kernel, args)

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
        stream = torch.cuda.current_stream()
        g_hr = torch.empty(sb, n, n, dtype=h_res.dtype, device=h_res.device)
        g_res = torch.empty(sb, n, C, dtype=original_residual.dtype, device=h_res.device)
        g_hp = torch.empty(sb, n, dtype=h_post.dtype, device=h_res.device)
        g_x = torch.empty(sb, C, dtype=x.dtype, device=h_res.device)
        go_flat = grad_output.view(sb, n, C)
        hr_flat = h_res.view(sb, n, n)
        orig_flat = original_residual.view(sb, n, C)
        hp_flat = h_post.view(sb, n)
        x_flat = x.view(sb, C)

        # --- Kernel A: g_x, g_orig (2D grid, no loop) ---
        cache_key_a = ('hpb_bwd_g_x_orig', sb, n, C)
        cached_a = _hpb_bwd_g_x_orig_best_cfg.get(cache_key_a)

        if cached_a is not None or not _CUTILE_EXPERIMENTAL_AVAILABLE:
            if cached_a is not None:
                ts, tc = cached_a
            else:
                ts, tc = 1, math.gcd(C, 1024)
            ct.launch(
                stream, (math.ceil(sb / ts), math.ceil(C / tc)),
                _ct_hpb_bwd_g_x_orig_kernel,
                (go_flat, hr_flat, hp_flat, g_res, g_x, n, tc, ts),
            )
        else:
            from types import SimpleNamespace

            configs = [SimpleNamespace(**c) for c in _hpb_autotune_configs(sb, C)]
            tuned = ct_experimental.autotune_launch(
                stream,
                grid_fn=lambda cfg: (math.ceil(sb / cfg.TILE_SIZE), math.ceil(C / cfg.TILE_C)),
                kernel=_ct_hpb_bwd_g_x_orig_kernel,
                args_fn=lambda cfg: (
                    go_flat, hr_flat, hp_flat, g_res, g_x, n, cfg.TILE_C, cfg.TILE_SIZE,
                ),
                search_space=configs,
            )
            best = tuned.tuned_config
            _hpb_bwd_g_x_orig_best_cfg[cache_key_a] = (best.TILE_SIZE, best.TILE_C)
            ct.launch(
                stream, (math.ceil(sb / best.TILE_SIZE), math.ceil(C / best.TILE_C)),
                _ct_hpb_bwd_g_x_orig_kernel,
                (go_flat, hr_flat, hp_flat, g_res, g_x, n, best.TILE_C, best.TILE_SIZE),
            )

        # --- Kernel B: g_hp, g_hr (1D grid, loops C-tiles) ---
        cache_key_b = ('hpb_bwd_g_hp_hr', sb, n, C, bias is not None)
        cached_b = _hpb_bwd_g_hp_hr_best_cfg.get(cache_key_b)
        hp_hr_kernel = _ct_hpb_bwd_g_hp_hr_bias_kernel if bias is not None else _ct_hpb_bwd_g_hp_hr_kernel

        if cached_b is not None or not _CUTILE_EXPERIMENTAL_AVAILABLE:
            if cached_b is not None:
                ts, tc = cached_b
            else:
                ts, tc = 1, math.gcd(C, 1024)
            args = (go_flat, orig_flat, x_flat)
            if bias is not None:
                args = args + (bias,)
            args = args + (g_hr, g_hp, n, tc, ts)
            ct.launch(stream, (math.ceil(sb / ts),), hp_hr_kernel, args)
        else:
            from types import SimpleNamespace

            configs = [SimpleNamespace(**c) for c in _hpb_autotune_configs(sb, C)]

            def _hp_hr_args_fn(cfg):
                args = (go_flat, orig_flat, x_flat)
                if bias is not None:
                    args = args + (bias,)
                return args + (g_hr, g_hp, n, cfg.TILE_C, cfg.TILE_SIZE)

            tuned = ct_experimental.autotune_launch(
                stream,
                grid_fn=lambda cfg: (math.ceil(sb / cfg.TILE_SIZE),),
                kernel=hp_hr_kernel,
                args_fn=_hp_hr_args_fn,
                search_space=configs,
            )
            best = tuned.tuned_config
            _hpb_bwd_g_hp_hr_best_cfg[cache_key_b] = (best.TILE_SIZE, best.TILE_C)
            args = (go_flat, orig_flat, x_flat)
            if bias is not None:
                args = args + (bias,)
            args = args + (g_hr, g_hp, n, best.TILE_C, best.TILE_SIZE)
            ct.launch(stream, (math.ceil(sb / best.TILE_SIZE),), hp_hr_kernel, args)

        g_bias = g_x.sum(dim=0).to(dtype=bias.dtype) if bias is not None else None
        return (
            g_hr.view(s, b, n, n),
            g_res.view(s, b, n, C),
            g_hp.view(s, b, n),
            g_x.view(s, b, C),
            g_bias,
        )

    # -- Proj RMS kernels ----------------------------------------------------

    @ct.function
    def _ct_rms_dnorm(a_tile, norm_tile, dr_tile, K, eps=1e-6):
        inv_norm = ct.where(norm_tile > 0, 1.0 / norm_tile, 0.0)
        inv_sqrt_k = 1.0 / ct.sqrt(K)
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
        SPLIT_K: ConstInt,
    ):
        '''
        Grid: (num_tiles_m, num_tiles_k).  Fused matmul + norm + r: proj, norm, r in one pass over K.
        '''
        tile_m_id = ct.bid(0)
        split_k_id = ct.bid(1)
        num_m_tiles = ct.cdiv(M, TILE_M)
        num_k_tiles = ct.cdiv(K, TILE_K)
        num_k_tiles_per_split = ct.cdiv(num_k_tiles, SPLIT_K)
        tile_k_id_start = split_k_id * num_k_tiles_per_split
        tile_k_id_end = ct.minimum(tile_k_id_start + num_k_tiles_per_split, num_k_tiles)
        acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        sum_sq = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
        for tile_k_id in range(tile_k_id_start, tile_k_id_end):
            a_tile = ct.load(
                A, index=(tile_m_id, tile_k_id), shape=(TILE_M, TILE_K), padding_mode=PAD_ZERO
            )
            b_tile = ct.load(B, index=(0, tile_k_id), shape=(TILE_N, TILE_K), padding_mode=PAD_ZERO)
            acc = ct.mma(
                a_tile.astype(ct.tfloat32), b_tile.transpose().astype(ct.tfloat32), acc=acc
            )
            sum_sq += ct.sum(a_tile * a_tile, axis=1, keepdims=True)
        # norm_tile = ct.sqrt(sum_sq)

        bid_m_k = tile_m_id + split_k_id * num_m_tiles
        ct.store(PROJ, index=(bid_m_k, 0), tile=acc.astype(PROJ.dtype))
        ct.store(NORM, index=(bid_m_k, 0), tile=sum_sq.astype(NORM.dtype))
        # v = norm_tile / ct.sqrt(K) + eps
        # r_tile = 1.0 / v
        # ct.atomic_add(PROJ, proj_indices, acc.astype(PROJ.dtype))
        # ct.atomic_add(NORM, norm_indices, norm_tile.astype(NORM.dtype))
        # ct.store(R, index=(tile_m_id, 0), tile=r_tile.astype(R.dtype))

    # -- Sigmoid helper for cuTile kernels ------------------------------------

    @ct.function
    def _ct_sigmoid(x):
        """Sigmoid via exp2: σ(x) = 1 / (1 + 2^(-x * log2(e)))."""
        return 1.0 / (1.0 + ct.exp2(-x * LOG2E))

    # -- Reduce split-K + compute_h kernel ------------------------------------

    @ct.kernel
    def _ct_reduce_compute_h_kernel(
        Y_acc,
        R_acc,
        Bias,
        Alpha_pre,
        Alpha_post,
        Alpha_res,
        Y,
        R,
        PROJ_OUT,
        M: int,
        N: int,
        K: int,
        n: int,
        eps: float,
        TILE_SIZE_M: ConstInt,
        TILE_SIZE_N: ConstInt,
        SPLIT_K: ConstInt,
    ):
        """Reduce split-K partial proj/norm, compute r, and apply compute_h activations.

        Grid: (ceil(M / TILE_SIZE_M),).
        TILE_SIZE_N = next_power_of_2(N) so one tile covers the full N dimension.
        Alpha_{pre,post,res} are [1] tensors (scalar parameters).
        """
        bid_m = ct.bid(0)
        num_bid_m = ct.cdiv(M, TILE_SIZE_M)

        alpha_pre = ct.load(Alpha_pre, index=(0,), shape=(1,)).item()
        alpha_post = ct.load(Alpha_post, index=(0,), shape=(1,)).item()
        alpha_res = ct.load(Alpha_res, index=(0,), shape=(1,)).item()

        # 1. Reduce split-K partials
        y_accum = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)
        r_accum = ct.full((TILE_SIZE_M, 1), 0.0, dtype=ct.float32)

        for split_idx in ct.static_iter(range(SPLIT_K)):
            bid_m_k = bid_m + split_idx * num_bid_m
            y_tile = ct.load(
                Y_acc, index=(bid_m_k, 0), shape=(TILE_SIZE_M, TILE_SIZE_N),
                padding_mode=PAD_ZERO,
            )
            y_accum = y_accum + ct.astype(y_tile, ct.float32)

            r_tile = ct.load(
                R_acc, index=(bid_m_k, 0), shape=(TILE_SIZE_M, 1),
                padding_mode=PAD_ZERO,
            )
            r_accum = r_accum + ct.astype(r_tile, ct.float32)

        # Store reduced proj for backward
        ct.store(PROJ_OUT, index=(bid_m, 0), tile=y_accum.astype(PROJ_OUT.dtype))

        # 2. Compute r = norm / sqrt(K)
        denom = ct.full((TILE_SIZE_M, 1), K * 1.0, dtype=ct.float32)
        mean = ct.truediv(r_accum, denom)
        rstd = ct.rsqrt(mean)
        ones = ct.full((TILE_SIZE_M, 1), 1.0, dtype=ct.float32)
        r_val = ct.truediv(ones, rstd)  # norm / sqrt(K)

        ct.store(R, index=(bid_m, 0), tile=r_val.astype(R.dtype))

        # 3. Build per-column scale from 3 scalar alphas via masks
        offsets = ct.arange(TILE_SIZE_N, dtype=ct.int32)

        one = ct.full((TILE_SIZE_N,), 1.0, dtype=ct.float32)
        zero = ct.full((TILE_SIZE_N,), 0.0, dtype=ct.float32)

        mask_pre = ct.where(ct.less(offsets, n), one, zero)
        mask_post = ct.where(ct.less(offsets, 2 * n), one, zero) - mask_pre
        mask_res = one - mask_pre - mask_post

        scale = alpha_pre * mask_pre + alpha_post * mask_post + alpha_res * mask_res
        scale = ct.reshape(scale, (1, TILE_SIZE_N))

        mask_pre = ct.reshape(mask_pre, (1, TILE_SIZE_N))
        mask_post = ct.reshape(mask_post, (1, TILE_SIZE_N))
        mask_res = ct.reshape(mask_res, (1, TILE_SIZE_N))

        # 4. Apply compute_h: linear = proj * scale / (r + eps) + bias
        bias_tile = ct.load(
            Bias, index=(0, 0), shape=(1, TILE_SIZE_N), padding_mode=PAD_ZERO,
        )
        bias_tile = ct.astype(bias_tile, ct.float32)

        linear = y_accum * scale / (r_val + eps) + bias_tile

        # 5. Mask-based activations
        sig = _ct_sigmoid(linear)
        out = sig * mask_pre + 2.0 * sig * mask_post + linear * mask_res

        ct.store(Y, index=(bid_m, 0), tile=out.astype(Y.dtype))

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
        eps: float,
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
            accumulator_da = accumulator_da + _ct_rms_dnorm(a_tile, norm_tile, dr_tile, K, eps)
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
        A, B, NORM, DD, DR, DA, DB, M: int, N: int, K: int, eps: float, TILE_N_SIZE: ConstInt
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
                    a_tile.astype(ct.float32), norm_tile, dr_tile, K, eps
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

    def _proj_rms_fwd_autotune_configs(N):
        """Generate autotune search space for proj_rms forward kernel."""
        TILE_N = _next_power_of_2(N)
        tile_ms = (32, 64, 128)
        tile_ks = (32, 64, 128)
        split_ks = (1, 2, 4, 8, 16)
        for tile_m in tile_ms:
            for tile_k in tile_ks:
                for split_k in split_ks:
                        yield {"TILE_M": tile_m, "TILE_N": TILE_N, "TILE_K": tile_k, 'SPLIT_K': split_k}

    # Cache the best config across calls (keyed by M, N, K).
    _proj_rms_fwd_best_cfg: dict = {}

    def _cutile_proj_rms_fwd(
        x: Tensor, weight: Tensor, eps: float = 1e-8
    ) -> Tuple[Tensor, Tensor, Tensor]:
        M, K = x.shape
        N = weight.shape[0]
        TILE_N = _next_power_of_2(N)
        dev = x.device
        stream = torch.cuda.current_stream()

        cache_key = (M, N, K)
        cached = _proj_rms_fwd_best_cfg.get(cache_key)

        if cached is not None or not _CUTILE_EXPERIMENTAL_AVAILABLE:
            # Use cached best config, or fall back to default if no experimental.
            if cached is not None:
                tm, tn, tk, split_k = cached
            else:
                tm, tn, tk, split_k = 128, TILE_N, 128, 1

            # print(cached)
            # print('hahahahahah')
            # split_k = 1
            proj = torch.empty(split_k * M, N, dtype=x.dtype, device=dev)
            norm = torch.empty(split_k * M, 1, dtype=x.dtype, device=dev)
            r = torch.empty(split_k * M, 1, dtype=x.dtype, device=dev)

            ct.launch(
                stream,
                (math.ceil(M / tm), split_k),
                _ct_proj_rms_fwd_kernel,
                (x, weight, proj, norm, r, M, N, K, eps, tm, tn, tk, split_k),
            )
            proj = proj.view(split_k, M, N).sum(dim=0).to(dtype=x.dtype)
            norm = norm.view(split_k, M, 1).sum(dim=0).to(dtype=x.dtype)
            # r = r.view(split_k, M, 1).sum(dim=0)
        else:
            # Autotune on first call for this shape.
            from types import SimpleNamespace

            configs = [SimpleNamespace(**c) for c in _proj_rms_fwd_autotune_configs(N)]
            # filter out configs with TILE_K > K or TILE_M > M
            configs = [cfg for cfg in configs if cfg.TILE_K <= K and cfg.TILE_M <= M]
            # if no valid config, set a default config with TILE_M = M and TILE_K = K and SPLIT_K = 1
            # if len(configs) == 0:
            #     configs = [SimpleNamespace(TILE_M=M, TILE_N=TILE_N, TILE_K=K, SPLIT_K=1)]
            mx_split_k = max(cfg.SPLIT_K for cfg in configs)
            proj = torch.empty(mx_split_k * M, N, dtype=x.dtype, device=dev)
            norm = torch.empty(mx_split_k * M, 1, dtype=x.dtype, device=dev)
            r = torch.empty(mx_split_k * M, 1, dtype=x.dtype, device=dev)
            tuned = ct_experimental.autotune_launch(
                stream,
                grid_fn=lambda cfg: (math.ceil(M / cfg.TILE_M), cfg.SPLIT_K),
                kernel=_ct_proj_rms_fwd_kernel,
                args_fn=lambda cfg: (
                    x, weight, proj, norm, r, M, N, K, eps,
                    cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.SPLIT_K,
                ),
                search_space=configs,
            )
            best = tuned.tuned_config
            # print(best)
            # print('hahahahahah')
            # best.SPLIT_K = 1
            _proj_rms_fwd_best_cfg[cache_key] = (best.TILE_M, best.TILE_N, best.TILE_K, best.SPLIT_K)
            proj = torch.empty(best.SPLIT_K * M, N, dtype=x.dtype, device=dev)
            norm = torch.empty(best.SPLIT_K * M, 1, dtype=x.dtype, device=dev)
            r = torch.empty(best.SPLIT_K * M, 1, dtype=x.dtype, device=dev)
            # Re-launch with best config for correct output.
            ct.launch(
                stream,
                (math.ceil(M / best.TILE_M), best.SPLIT_K),
                _ct_proj_rms_fwd_kernel,
                (x, weight, proj, norm, r, M, N, K, eps,
                 best.TILE_M, best.TILE_N, best.TILE_K, best.SPLIT_K),
            )

            proj = proj.view(best.SPLIT_K, M, N).sum(dim=0)
            norm = norm.view(best.SPLIT_K, M, 1).sum(dim=0)
        norm = torch.sqrt(norm)
        # r = r.view(split_k, M, 1).sum(dim=0)
        r = 1.0 / (norm / math.sqrt(K) + eps)
        return proj, norm, r

    # -- Reduce + compute_h launcher ------------------------------------------

    def _reduce_compute_h_autotune_configs(M):
        """Generate autotune search space for reduce_compute_h kernel."""
        for tile_m in (1, 2, 4, 8, 16, 32, 64, 128):
            if tile_m <= M:
                yield tile_m

    _reduce_compute_h_best_cfg: dict = {}

    def _cutile_reduce_compute_h(
        proj_acc: Tensor,
        norm_acc: Tensor,
        bias: Tensor,
        alpha_pre: Tensor,
        alpha_post: Tensor,
        alpha_res: Tensor,
        n: int,
        M: int,
        N: int,
        K: int,
        eps: float,
        tile_n: int,
        split_k: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Launch reduce split-K + compute_h kernel.

        Returns:
            y_activated: [M, N] with h_pre | h_post | h_res activated
            r: [M, 1]  r = norm / sqrt(K)
            proj_reduced: [M, N] reduced projection (for backward)
        """
        dev = proj_acc.device
        stream = torch.cuda.current_stream()

        bias_2d = bias.unsqueeze(0).contiguous()  # [1, N]

        y_out = torch.empty(M, N, dtype=proj_acc.dtype, device=dev)
        r_out = torch.empty(M, 1, dtype=proj_acc.dtype, device=dev)
        proj_out = torch.empty(M, N, dtype=proj_acc.dtype, device=dev)

        cache_key = (M, N, K, n, split_k)
        cached = _reduce_compute_h_best_cfg.get(cache_key)

        def _make_args(tm):
            return (
                proj_acc, norm_acc, bias_2d,
                alpha_pre, alpha_post, alpha_res,
                y_out, r_out, proj_out,
                M, N, K, n, eps,
                tm, tile_n, split_k,
            )

        if cached is not None or not _CUTILE_EXPERIMENTAL_AVAILABLE:
            tm = cached if cached is not None else min(128, M)
            ct.launch(stream, (math.ceil(M / tm),), _ct_reduce_compute_h_kernel, _make_args(tm))
        else:
            from types import SimpleNamespace

            configs = [
                SimpleNamespace(TILE_M=tm)
                for tm in _reduce_compute_h_autotune_configs(M)
            ]
            tuned = ct_experimental.autotune_launch(
                stream,
                grid_fn=lambda cfg: (math.ceil(M / cfg.TILE_M),),
                kernel=_ct_reduce_compute_h_kernel,
                args_fn=lambda cfg: _make_args(cfg.TILE_M),
                search_space=configs,
            )
            best_tm = tuned.tuned_config.TILE_M
            _reduce_compute_h_best_cfg[cache_key] = best_tm
            ct.launch(
                stream, (math.ceil(M / best_tm),),
                _ct_reduce_compute_h_kernel, _make_args(best_tm),
            )

        return y_out, r_out, proj_out

    # -- Combined proj_rms + compute_h forward --------------------------------

    def _cutile_proj_rms_compute_h_fwd(
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
        alpha_pre: Tensor,
        alpha_post: Tensor,
        alpha_res: Tensor,
        n: int,
        eps: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Fused proj_rms + compute_h forward.

        Launches the existing _ct_proj_rms_fwd_kernel (split-K matmul + partial norm),
        then _ct_reduce_compute_h_kernel (reduce + r + activations).

        Returns:
            y_activated: [M, N] activated h values
            r: [M, 1] r = norm / sqrt(K)
            proj_reduced: [M, N] reduced projection (for backward)
        """
        M, K = x.shape
        N = weight.shape[0]
        TILE_N = _next_power_of_2(N)
        dev = x.device
        stream = torch.cuda.current_stream()

        cache_key = (M, N, K)
        cached = _proj_rms_fwd_best_cfg.get(cache_key)

        if cached is not None or not _CUTILE_EXPERIMENTAL_AVAILABLE:
            if cached is not None:
                tm, tn, tk, split_k = cached
            else:
                tm, tn, tk, split_k = 128, TILE_N, 128, 1

            proj_acc = torch.empty(split_k * M, N, dtype=x.dtype, device=dev)
            norm_acc = torch.empty(split_k * M, 1, dtype=x.dtype, device=dev)
            r_placeholder = torch.empty(split_k * M, 1, dtype=x.dtype, device=dev)

            ct.launch(
                stream,
                (math.ceil(M / tm), split_k),
                _ct_proj_rms_fwd_kernel,
                (x, weight, proj_acc, norm_acc, r_placeholder, M, N, K, eps, tm, tn, tk, split_k),
            )
        else:
            from types import SimpleNamespace

            configs = [SimpleNamespace(**c) for c in _proj_rms_fwd_autotune_configs(N)]
            configs = [cfg for cfg in configs if cfg.TILE_K <= K and cfg.TILE_M <= M]
            mx_split_k = max(cfg.SPLIT_K for cfg in configs)
            proj_acc = torch.empty(mx_split_k * M, N, dtype=x.dtype, device=dev)
            norm_acc = torch.empty(mx_split_k * M, 1, dtype=x.dtype, device=dev)
            r_placeholder = torch.empty(mx_split_k * M, 1, dtype=x.dtype, device=dev)
            tuned = ct_experimental.autotune_launch(
                stream,
                grid_fn=lambda cfg: (math.ceil(M / cfg.TILE_M), cfg.SPLIT_K),
                kernel=_ct_proj_rms_fwd_kernel,
                args_fn=lambda cfg: (
                    x, weight, proj_acc, norm_acc, r_placeholder, M, N, K, eps,
                    cfg.TILE_M, cfg.TILE_N, cfg.TILE_K, cfg.SPLIT_K,
                ),
                search_space=configs,
            )
            best = tuned.tuned_config
            _proj_rms_fwd_best_cfg[cache_key] = (best.TILE_M, best.TILE_N, best.TILE_K, best.SPLIT_K)
            tm, tn, tk, split_k = best.TILE_M, best.TILE_N, best.TILE_K, best.SPLIT_K

            proj_acc = torch.empty(split_k * M, N, dtype=x.dtype, device=dev)
            norm_acc = torch.empty(split_k * M, 1, dtype=x.dtype, device=dev)
            r_placeholder = torch.empty(split_k * M, 1, dtype=x.dtype, device=dev)
            ct.launch(
                stream,
                (math.ceil(M / tm), split_k),
                _ct_proj_rms_fwd_kernel,
                (x, weight, proj_acc, norm_acc, r_placeholder, M, N, K, eps, tm, tn, tk, split_k),
            )

        # Launch reduce + compute_h kernel
        y_activated, r, proj_reduced = _cutile_reduce_compute_h(
            proj_acc, norm_acc, bias,
            alpha_pre, alpha_post, alpha_res,
            n, M, N, K, eps,
            TILE_N, split_k,
        )
        return y_activated, r, proj_reduced

    def _proj_rms_bwd_autotune_configs(N):
        """Generate autotune search space for proj_rms backward kernel (K >= 8192 path)."""
        TILE_N = _next_power_of_2(N)
        tile_ms = (32, 64, 128)
        tile_ks = (32, 64, 128, 256)
        for tile_m in tile_ms:
            for tile_k in tile_ks:
                yield {"TILE_SIZE_M": tile_m, "TILE_SIZE_N": TILE_N, "TILE_SIZE_K": tile_k}

    _proj_rms_bwd_best_cfg: dict = {}

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
        stream = torch.cuda.current_stream()

        if K >= 8192:
            cache_key = (M, N, K)
            cached = _proj_rms_bwd_best_cfg.get(cache_key)

            if cached is not None or not _CUTILE_EXPERIMENTAL_AVAILABLE:
                if cached is not None:
                    tm, tn, tk = cached
                else:
                    tm, tn, tk = 128, TILE_SIZE_N, 128
                ct.launch(
                    stream,
                    (math.ceil(K / tk), 1),
                    _ct_proj_rms_bwd_kernel,
                    (x, weight, norm, grad_proj, grad_r, da, db, M, N, K, eps, tm, tn, tk),
                )
            else:
                from types import SimpleNamespace

                configs = [SimpleNamespace(**c) for c in _proj_rms_bwd_autotune_configs(N)]
                tuned = ct_experimental.autotune_launch(
                    stream,
                    grid_fn=lambda cfg: (math.ceil(K / cfg.TILE_SIZE_K), 1),
                    kernel=_ct_proj_rms_bwd_kernel,
                    args_fn=lambda cfg: (
                        x, weight, norm, grad_proj, grad_r, da, db, M, N, K, eps,
                        cfg.TILE_SIZE_M, cfg.TILE_SIZE_N, cfg.TILE_SIZE_K,
                    ),
                    search_space=configs,
                )
                best = tuned.tuned_config
                _proj_rms_bwd_best_cfg[cache_key] = (
                    best.TILE_SIZE_M, best.TILE_SIZE_N, best.TILE_SIZE_K,
                )
                ct.launch(
                    stream,
                    (math.ceil(K / best.TILE_SIZE_K), 1),
                    _ct_proj_rms_bwd_kernel,
                    (x, weight, norm, grad_proj, grad_r, da, db, M, N, K, eps,
                     best.TILE_SIZE_M, best.TILE_SIZE_N, best.TILE_SIZE_K),
                )
        else:
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            grid = (num_sms, 2, 1)
            ct.launch(
                stream,
                grid,
                _ct_proj_rms_bwd_small_k_kernel,
                (x, weight, norm, grad_proj, grad_r, da, db, M, N, K, eps, TILE_SIZE_N),
            )
        return da, db

    # -- Fused compute_h + proj_rms backward kernels ----------------------------

    @ct.kernel
    def _ct_fused_grad_h_proj_kernel(
        GRAD_Y,         # [M, N]
        Y_ACT,          # [M, N]
        PROJ,           # [M, N]
        R,              # [M, 1]
        GRAD_R_EXT,     # [M, 1]
        Alpha_pre,      # [1]
        Alpha_post,     # [1]
        Alpha_res,      # [1]
        GRAD_H,         # [M, TILE_SIZE_N] output
        GRAD_PROJ,      # [M, TILE_SIZE_N] output
        GRAD_R_TOTAL,   # [M, 1] output
        M: int,
        N: int,
        n: int,
        eps: float,
        TILE_SIZE_M: ConstInt,
        TILE_SIZE_N: ConstInt,
        HAS_GRAD_R_EXT: ConstInt,
    ):
        """Precompute grad_h, grad_proj, and grad_r_total for downstream backward kernels.

        Grid: (ceil(M / TILE_SIZE_M),).
        """
        tile_m_id = ct.bid(0)

        alpha_pre = ct.load(Alpha_pre, index=(0,), shape=(1,)).item()
        alpha_post = ct.load(Alpha_post, index=(0,), shape=(1,)).item()
        alpha_res = ct.load(Alpha_res, index=(0,), shape=(1,)).item()

        offsets = ct.arange(TILE_SIZE_N, dtype=ct.int32)
        one = ct.full((TILE_SIZE_N,), 1.0, dtype=ct.float32)
        zero = ct.full((TILE_SIZE_N,), 0.0, dtype=ct.float32)
        mask_pre = ct.where(ct.less(offsets, n), one, zero)
        mask_post = ct.where(ct.less(offsets, 2 * n), one, zero) - mask_pre
        mask_res = one - mask_pre - mask_post

        scale = alpha_pre * mask_pre + alpha_post * mask_post + alpha_res * mask_res
        scale_2d = ct.reshape(scale, (1, TILE_SIZE_N))
        mask_pre_2d = ct.reshape(mask_pre, (1, TILE_SIZE_N))
        mask_post_2d = ct.reshape(mask_post, (1, TILE_SIZE_N))
        mask_res_2d = ct.reshape(mask_res, (1, TILE_SIZE_N))

        gy_tile = ct.load(
            GRAD_Y, index=(tile_m_id, 0), shape=(TILE_SIZE_M, TILE_SIZE_N),
            padding_mode=PAD_ZERO,
        )
        gy_tile = ct.astype(gy_tile, ct.float32)
        y_tile = ct.load(
            Y_ACT, index=(tile_m_id, 0), shape=(TILE_SIZE_M, TILE_SIZE_N),
            padding_mode=PAD_ZERO,
        )
        y_tile = ct.astype(y_tile, ct.float32)
        proj_tile = ct.load(
            PROJ, index=(tile_m_id, 0), shape=(TILE_SIZE_M, TILE_SIZE_N),
            padding_mode=PAD_ZERO,
        )
        proj_tile = ct.astype(proj_tile, ct.float32)
        r_tile = ct.load(
            R, index=(tile_m_id, 0), shape=(TILE_SIZE_M, 1),
            padding_mode=PAD_ZERO,
        )
        r_tile = ct.astype(r_tile, ct.float32)

        # Activation backward → grad_h
        gh_pre = gy_tile * y_tile * (1.0 - y_tile) * mask_pre_2d
        half_y = y_tile * 0.5
        gh_post = gy_tile * half_y * (1.0 - half_y) * 2.0 * mask_post_2d
        gh_res = gy_tile * mask_res_2d
        grad_h = gh_pre + gh_post + gh_res

        r_eps = r_tile + eps
        inv_r_eps = 1.0 / r_eps
        grad_proj_tile = grad_h * scale_2d * inv_r_eps

        # grad_r_total: reduction over N → [TILE_M, 1]
        grad_r_from_h = ct.sum(
            grad_h * proj_tile * scale_2d * (-inv_r_eps * inv_r_eps),
            axis=1, keepdims=True,
        )
        if HAS_GRAD_R_EXT:
            grad_r_ext_tile = ct.load(
                GRAD_R_EXT, index=(tile_m_id, 0), shape=(TILE_SIZE_M, 1),
                padding_mode=PAD_ZERO,
            )
            grad_r_ext_tile = ct.astype(grad_r_ext_tile, ct.float32)
        else:
            grad_r_ext_tile = ct.full((TILE_SIZE_M, 1), 0.0, dtype=ct.float32)
        grad_r_total = grad_r_from_h + grad_r_ext_tile

        ct.store(GRAD_H, index=(tile_m_id, 0), tile=grad_h.astype(GRAD_H.dtype))
        ct.store(GRAD_PROJ, index=(tile_m_id, 0), tile=grad_proj_tile.astype(GRAD_PROJ.dtype))
        ct.store(GRAD_R_TOTAL, index=(tile_m_id, 0), tile=grad_r_total.astype(GRAD_R_TOTAL.dtype))

    @ct.kernel
    def _ct_fused_grad_x_weight_kernel(
        X,              # [M, K]
        WEIGHT,         # [N, K]
        GRAD_PROJ,      # [M, TILE_SIZE_N] precomputed
        GRAD_R_TOTAL,   # [M, 1] precomputed
        R,              # [M, 1]
        GRAD_X,         # [M, K] output
        GRAD_WEIGHT,    # [N, K] output
        M: int,
        N: int,
        K: int,
        TILE_SIZE_M: ConstInt,
        TILE_SIZE_N: ConstInt,
        TILE_SIZE_K: ConstInt,
    ):
        """Compute grad_x and grad_weight simultaneously.

        Grid: (ceil(K / TILE_SIZE_K),).
        Each block handles one K-tile and loops over all M-tiles.
        Per M-tile: computes and stores grad_x, accumulates grad_weight.
        """
        tile_k_id = ct.bid(0)
        NUM_M_TILES = ct.cdiv(M, TILE_SIZE_M)

        # Load weight tile once — only depends on K-tile
        weight_tile = ct.load(
            WEIGHT, index=(0, tile_k_id), shape=(TILE_SIZE_N, TILE_SIZE_K),
            padding_mode=PAD_ZERO,
        )

        acc_grad_weight = ct.full((TILE_SIZE_K, TILE_SIZE_N), 0.0, dtype=ct.float32)

        for tile_m_id in range(NUM_M_TILES):
            grad_proj_tile = ct.load(
                GRAD_PROJ, index=(tile_m_id, 0), shape=(TILE_SIZE_M, TILE_SIZE_N),
                padding_mode=PAD_ZERO,
            )
            x_tile = ct.load(
                X, index=(tile_m_id, tile_k_id), shape=(TILE_SIZE_M, TILE_SIZE_K),
                padding_mode=PAD_ZERO,
            )
            grad_r_total = ct.load(
                GRAD_R_TOTAL, index=(tile_m_id, 0), shape=(TILE_SIZE_M, 1),
                padding_mode=PAD_ZERO,
            )
            r_tile = ct.load(
                R, index=(tile_m_id, 0), shape=(TILE_SIZE_M, 1),
                padding_mode=PAD_ZERO,
            )
            r_tile = ct.astype(r_tile, ct.float32)

            # grad_x = grad_proj @ weight + grad_r_total * x / (r * K)
            inv_rK = 1.0 / (r_tile * K)
            acc_grad_x = (grad_r_total * inv_rK) * ct.astype(x_tile, ct.float32)
            acc_grad_x = ct.mma(
                grad_proj_tile.astype(ct.tfloat32),
                weight_tile.astype(ct.tfloat32),
                acc=acc_grad_x,
            )
            ct.store(GRAD_X, index=(tile_m_id, tile_k_id), tile=acc_grad_x.astype(GRAD_X.dtype))

            # Accumulate grad_weight += x.T @ grad_proj
            acc_grad_weight = ct.mma(
                x_tile.transpose().astype(ct.tfloat32),
                grad_proj_tile.astype(ct.tfloat32),
                acc=acc_grad_weight,
            )

        ct.store(GRAD_WEIGHT, index=(0, tile_k_id), tile=acc_grad_weight.transpose().astype(GRAD_WEIGHT.dtype))

    @ct.kernel
    def _ct_scalar_grads_partials_kernel(
        GRAD_H,           # [M, TILE_SIZE_N] precomputed
        PROJ,           # [M, N]
        R,              # [M, 1]
        GRAD_ALPHA_PRE_PARTIALS,   # [num_m_blocks, 1] output
        GRAD_ALPHA_POST_PARTIALS,  # [num_m_blocks, 1] output
        GRAD_ALPHA_RES_PARTIALS,   # [num_m_blocks, 1] output
        GRAD_BIAS_PARTIALS,        # [num_m_blocks, TILE_SIZE_N] output
        M: int,
        N: int,
        n: int,
        eps: float,
        TILE_SIZE_M: ConstInt,
        TILE_SIZE_N: ConstInt,
    ):
        """Compute per-M-tile scalar-gradient partials.

        Grid: (ceil(M / TILE_SIZE_M),).  Each block processes one M-tile.
        """
        bid_m = ct.bid(0)

        offsets = ct.arange(TILE_SIZE_N, dtype=ct.int32)
        one = ct.full((TILE_SIZE_N,), 1.0, dtype=ct.float32)
        zero = ct.full((TILE_SIZE_N,), 0.0, dtype=ct.float32)
        mask_pre = ct.where(ct.less(offsets, n), one, zero)
        mask_post = ct.where(ct.less(offsets, 2 * n), one, zero) - mask_pre
        mask_res = one - mask_pre - mask_post

        mask_pre_2d = ct.reshape(mask_pre, (1, TILE_SIZE_N))
        mask_post_2d = ct.reshape(mask_post, (1, TILE_SIZE_N))
        mask_res_2d = ct.reshape(mask_res, (1, TILE_SIZE_N))

        grad_h = ct.load(
            GRAD_H, index=(bid_m, 0), shape=(TILE_SIZE_M, TILE_SIZE_N),
            padding_mode=PAD_ZERO,
        )
        proj_tile = ct.load(
            PROJ, index=(bid_m, 0), shape=(TILE_SIZE_M, TILE_SIZE_N),
            padding_mode=PAD_ZERO,
        )
        proj_tile = ct.astype(proj_tile, ct.float32)
        r_tile = ct.load(
            R, index=(bid_m, 0), shape=(TILE_SIZE_M, 1),
            padding_mode=PAD_ZERO,
        )
        r_tile = ct.astype(r_tile, ct.float32)

        r_eps = r_tile + eps
        inv_r_eps = 1.0 / r_eps

        ga_all = grad_h * proj_tile * inv_r_eps
        ga_pre = ct.reshape(ct.sum(ga_all * mask_pre_2d), (1, 1))
        ga_post = ct.reshape(ct.sum(ga_all * mask_post_2d), (1, 1))
        ga_res = ct.reshape(ct.sum(ga_all * mask_res_2d), (1, 1))
        partial_gb = ct.sum(grad_h, axis=0, keepdims=False)
        ct.store(
            GRAD_ALPHA_PRE_PARTIALS, index=(bid_m, 0),
            tile=ga_pre.astype(GRAD_ALPHA_PRE_PARTIALS.dtype),
        )
        ct.store(
            GRAD_ALPHA_POST_PARTIALS, index=(bid_m, 0),
            tile=ga_post.astype(GRAD_ALPHA_POST_PARTIALS.dtype),
        )
        ct.store(
            GRAD_ALPHA_RES_PARTIALS, index=(bid_m, 0),
            tile=ga_res.astype(GRAD_ALPHA_RES_PARTIALS.dtype),
        )
        ct.store(
            GRAD_BIAS_PARTIALS, index=(bid_m, 0),
            tile=ct.reshape(partial_gb, (1, TILE_SIZE_N)).astype(GRAD_BIAS_PARTIALS.dtype),
        )

    @ct.kernel
    def _ct_scalar_grads_reduce_kernel(
        GRAD_ALPHA_PRE_PARTIALS,   # [num_m_blocks, 1]
        GRAD_ALPHA_POST_PARTIALS,  # [num_m_blocks, 1]
        GRAD_ALPHA_RES_PARTIALS,   # [num_m_blocks, 1]
        GRAD_BIAS_PARTIALS,        # [num_m_blocks, TILE_SIZE_N]
        GRAD_ALPHA_PRE,            # [1, 1] output
        GRAD_ALPHA_POST,           # [1, 1] output
        GRAD_ALPHA_RES,            # [1, 1] output
        GRAD_BIAS,                 # [1, TILE_SIZE_N] output
        NUM_M_BLOCKS: int,
        TILE_SIZE_N: ConstInt,
    ):
        """Reduce scalar-gradient partials and write final dtype outputs."""
        acc_pre = ct.full((1, 1), 0.0, dtype=ct.float32)
        acc_post = ct.full((1, 1), 0.0, dtype=ct.float32)
        acc_res = ct.full((1, 1), 0.0, dtype=ct.float32)
        acc_bias = ct.full((1, TILE_SIZE_N), 0.0, dtype=ct.float32)

        for bid_m in range(NUM_M_BLOCKS):
            acc_pre += ct.load(
                GRAD_ALPHA_PRE_PARTIALS, index=(bid_m, 0), shape=(1, 1),
                padding_mode=PAD_ZERO,
            ).astype(ct.float32)
            acc_post += ct.load(
                GRAD_ALPHA_POST_PARTIALS, index=(bid_m, 0), shape=(1, 1),
                padding_mode=PAD_ZERO,
            ).astype(ct.float32)
            acc_res += ct.load(
                GRAD_ALPHA_RES_PARTIALS, index=(bid_m, 0), shape=(1, 1),
                padding_mode=PAD_ZERO,
            ).astype(ct.float32)
            acc_bias += ct.load(
                GRAD_BIAS_PARTIALS, index=(bid_m, 0), shape=(1, TILE_SIZE_N),
                padding_mode=PAD_ZERO,
            ).astype(ct.float32)

        ct.store(GRAD_ALPHA_PRE, index=(0, 0), tile=acc_pre.astype(GRAD_ALPHA_PRE.dtype))
        ct.store(GRAD_ALPHA_POST, index=(0, 0), tile=acc_post.astype(GRAD_ALPHA_POST.dtype))
        ct.store(GRAD_ALPHA_RES, index=(0, 0), tile=acc_res.astype(GRAD_ALPHA_RES.dtype))
        ct.store(GRAD_BIAS, index=(0, 0), tile=acc_bias.astype(GRAD_BIAS.dtype))

    @ct.kernel
    def _ct_fused_compute_h_proj_rms_bwd_small_k_kernel(
        X,              # [M, K]
        WEIGHT,         # [N, K]
        GRAD_Y,         # [M, N]
        Y_ACT,          # [M, N]
        PROJ,           # [M, N]
        R,              # [M, 1]
        GRAD_R_EXT,     # [M, 1]
        Alpha_pre,      # [1]
        Alpha_post,     # [1]
        Alpha_res,      # [1]
        GRAD_X,         # [M, K] output
        GRAD_WEIGHT,    # [N, K] output
        M: int,
        N: int,
        K: int,
        n: int,
        eps: float,
        TILE_N_SIZE: ConstInt,
        HAS_GRAD_R_EXT: ConstInt,
    ):
        """Fused backward (small K path) with work-stealing.

        Grid: (num_sms, 2).
        bid(1)==0: grad_weight via work-stealing over K-tiles, loops M.
        bid(1)==1: grad_x via work-stealing over (M×K) tiles.
        Scalar gradients are computed by the separate partial/reduce kernels.
        """
        zero_pad = ct.PaddingMode.ZERO

        alpha_pre = ct.load(Alpha_pre, index=(0,), shape=(1,)).item()
        alpha_post = ct.load(Alpha_post, index=(0,), shape=(1,)).item()
        alpha_res = ct.load(Alpha_res, index=(0,), shape=(1,)).item()

        # Build masks and scale
        offsets = ct.arange(TILE_N_SIZE, dtype=ct.int32)
        one = ct.full((TILE_N_SIZE,), 1.0, dtype=ct.float32)
        zero_v = ct.full((TILE_N_SIZE,), 0.0, dtype=ct.float32)
        mask_pre = ct.where(ct.less(offsets, n), one, zero_v)
        mask_post = ct.where(ct.less(offsets, 2 * n), one, zero_v) - mask_pre
        mask_res = one - mask_pre - mask_post

        scale = alpha_pre * mask_pre + alpha_post * mask_post + alpha_res * mask_res
        scale_2d = ct.reshape(scale, (1, TILE_N_SIZE))
        mask_pre_2d = ct.reshape(mask_pre, (1, TILE_N_SIZE))
        mask_post_2d = ct.reshape(mask_post, (1, TILE_N_SIZE))
        mask_res_2d = ct.reshape(mask_res, (1, TILE_N_SIZE))

        TILE_DB_SIZE_M = 128
        TILE_DB_SIZE_K = 64
        NUM_M_TILES = ct.cdiv(M, TILE_DB_SIZE_M)
        NUM_K_TILES = ct.cdiv(K, TILE_DB_SIZE_K)

        if ct.bid(1) == 0:
            # --- grad_weight path ---
            for tile_id in range(ct.bid(0), NUM_K_TILES, ct.num_blocks(0)):
                accumulator_db = ct.full((TILE_DB_SIZE_K, TILE_N_SIZE), 0.0, dtype=ct.float32)
                for m_tile in range(NUM_M_TILES):
                    x_tile = ct.load(
                        X, index=(m_tile, tile_id),
                        shape=(TILE_DB_SIZE_M, TILE_DB_SIZE_K), padding_mode=zero_pad,
                    )
                    # Compute grad_proj on-the-fly
                    gy_tile = ct.load(
                        GRAD_Y, index=(m_tile, 0),
                        shape=(TILE_DB_SIZE_M, TILE_N_SIZE), padding_mode=zero_pad,
                    )
                    gy_tile = ct.astype(gy_tile, ct.float32)
                    y_tile = ct.load(
                        Y_ACT, index=(m_tile, 0),
                        shape=(TILE_DB_SIZE_M, TILE_N_SIZE), padding_mode=zero_pad,
                    )
                    y_tile = ct.astype(y_tile, ct.float32)
                    r_tile = ct.load(
                        R, index=(m_tile, 0),
                        shape=(TILE_DB_SIZE_M, 1), padding_mode=zero_pad,
                    )
                    r_tile = ct.astype(r_tile, ct.float32)

                    gh_pre = gy_tile * y_tile * (1.0 - y_tile) * mask_pre_2d
                    half_y = y_tile * 0.5
                    gh_post = gy_tile * half_y * (1.0 - half_y) * 2.0 * mask_post_2d
                    gh_res = gy_tile * mask_res_2d
                    grad_h = gh_pre + gh_post + gh_res
                    r_eps = r_tile + eps
                    inv_r_eps = 1.0 / r_eps
                    grad_proj_tile = grad_h * scale_2d * inv_r_eps

                    accumulator_db = ct.mma(
                        x_tile.transpose().astype(ct.tfloat32),
                        grad_proj_tile.astype(ct.tfloat32),
                        acc=accumulator_db,
                    )

                ct.store(
                    GRAD_WEIGHT, index=(0, tile_id),
                    tile=accumulator_db.transpose().astype(GRAD_WEIGHT.dtype),
                    allow_tma=False,
                )

        TILE_DA_SIZE_M = 128
        TILE_DA_SIZE_K = 256
        NUM_DA_TILES = ct.cdiv(M, TILE_DA_SIZE_M) * ct.cdiv(K, TILE_DA_SIZE_K)
        NUM_DA_K_TILES = ct.cdiv(K, TILE_DA_SIZE_K)

        if ct.bid(1) == 1:
            # --- grad_x path ---
            for tile_id in range(ct.bid(0), NUM_DA_TILES, ct.num_blocks(0)):
                b_tile_idx = tile_id % NUM_DA_K_TILES
                dd_tile_idx = tile_id // NUM_DA_K_TILES

                # Compute grad_proj on-the-fly
                gy_tile = ct.load(
                    GRAD_Y, index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, TILE_N_SIZE), padding_mode=zero_pad,
                )
                gy_tile = ct.astype(gy_tile, ct.float32)
                y_tile = ct.load(
                    Y_ACT, index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, TILE_N_SIZE), padding_mode=zero_pad,
                )
                y_tile = ct.astype(y_tile, ct.float32)
                proj_tile = ct.load(
                    PROJ, index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, TILE_N_SIZE), padding_mode=zero_pad,
                )
                proj_tile = ct.astype(proj_tile, ct.float32)
                r_tile = ct.load(
                    R, index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, 1), padding_mode=zero_pad,
                )
                r_tile = ct.astype(r_tile, ct.float32)

                gh_pre = gy_tile * y_tile * (1.0 - y_tile) * mask_pre_2d
                half_y = y_tile * 0.5
                gh_post = gy_tile * half_y * (1.0 - half_y) * 2.0 * mask_post_2d
                gh_res = gy_tile * mask_res_2d
                grad_h = gh_pre + gh_post + gh_res
                r_eps = r_tile + eps
                inv_r_eps = 1.0 / r_eps
                grad_proj_tile = grad_h * scale_2d * inv_r_eps

                # Simplified rms_dnorm: grad_r_total * x / (r * K)
                grad_r_from_h = ct.sum(
                    grad_h * proj_tile * scale_2d * (-inv_r_eps * inv_r_eps),
                    axis=1, keepdims=True,
                )
                if HAS_GRAD_R_EXT:
                    grad_r_ext_tile = ct.load(
                        GRAD_R_EXT, index=(dd_tile_idx, 0),
                        shape=(TILE_DA_SIZE_M, 1), padding_mode=zero_pad,
                    )
                    grad_r_ext_tile = ct.astype(grad_r_ext_tile, ct.float32)
                else:
                    grad_r_ext_tile = ct.full((TILE_DA_SIZE_M, 1), 0.0, dtype=ct.float32)
                grad_r_total = grad_r_from_h + grad_r_ext_tile

                x_tile = ct.load(
                    X, index=(dd_tile_idx, b_tile_idx),
                    shape=(TILE_DA_SIZE_M, TILE_DA_SIZE_K), padding_mode=zero_pad,
                )
                inv_rK = 1.0 / (r_tile * K)
                accumulator_da = (grad_r_total * inv_rK) * ct.astype(x_tile, ct.float32)

                weight_tile = ct.load(
                    WEIGHT, index=(0, b_tile_idx),
                    shape=(TILE_N_SIZE, TILE_DA_SIZE_K), padding_mode=zero_pad,
                )
                accumulator_da = ct.mma(
                    grad_proj_tile.astype(ct.tfloat32),
                    weight_tile.astype(ct.tfloat32),
                    acc=accumulator_da,
                )
                ct.store(GRAD_X, index=(dd_tile_idx, b_tile_idx), tile=accumulator_da.astype(GRAD_X.dtype))

    def _fused_grad_x_weight_autotune_configs(N):
        """Autotune search space for fused grad_x + grad_weight kernel."""
        TILE_N = _next_power_of_2(N)
        tile_ms = (32, 64, 128)
        tile_ks = (32, 64, 128, 256)
        for tile_m in tile_ms:
            for tile_k in tile_ks:
                yield {"TILE_SIZE_M": tile_m, "TILE_SIZE_N": TILE_N, "TILE_SIZE_K": tile_k}

    _fused_grad_x_weight_best_cfg: dict = {}

    def _cutile_fused_compute_h_proj_rms_bwd(
        x: Tensor,
        weight: Tensor,
        grad_y: Tensor,
        y_activated: Tensor,
        proj: Tensor,
        r: Tensor,
        grad_r_ext: Tensor,
        alpha_pre: Tensor,
        alpha_post: Tensor,
        alpha_res: Tensor,
        bias: Tensor,
        n: int,
        eps: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Fused compute_h + proj_rms backward.

        Returns:
            grad_x: [M, K]
            grad_weight: [N, K]
            grad_alpha_pre: [1]
            grad_alpha_post: [1]
            grad_alpha_res: [1]
            grad_bias: [N]
        """
        M, K = x.shape
        N = weight.shape[0]
        TILE_N = _next_power_of_2(N)
        assert TILE_N <= 256, f"TILE_SIZE_N too large: {TILE_N}"
        dev = x.device
        stream = torch.cuda.current_stream()

        grad_x = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)
        has_grad_r_ext = grad_r_ext is not None
        has_grad_r_ext_flag = int(has_grad_r_ext)
        grad_r_ext_arg = grad_r_ext if has_grad_r_ext else r

        # 0. Precompute grad_h, grad_proj, grad_r_total
        grad_h_buf = torch.empty(M, TILE_N, dtype=torch.float32, device=dev)
        grad_proj_buf = torch.empty(M, TILE_N, dtype=torch.float32, device=dev)
        grad_r_total_buf = torch.empty(M, 1, dtype=torch.float32, device=dev)

        tile_m_precomp = min(128, M)
        ct.launch(
            stream,
            (math.ceil(M / tile_m_precomp),),
            _ct_fused_grad_h_proj_kernel,
            (
                grad_y, y_activated, proj, r, grad_r_ext_arg,
                alpha_pre, alpha_post, alpha_res,
                grad_h_buf, grad_proj_buf, grad_r_total_buf,
                M, N, n, eps,
                tile_m_precomp, TILE_N, has_grad_r_ext_flag,
            ),
        )

        if K >= 8192:
            # 1. Fused grad_x + grad_weight kernel — 1D grid (K-tiles), loops M
            cache_key = ('grad_x_weight', M, N, K)
            cached = _fused_grad_x_weight_best_cfg.get(cache_key)

            if cached is not None or not _CUTILE_EXPERIMENTAL_AVAILABLE:
                if cached is not None:
                    tm, tn, tk = cached
                else:
                    tm, tn, tk = 128, TILE_N, 128
                ct.launch(
                    stream,
                    (math.ceil(K / tk),),
                    _ct_fused_grad_x_weight_kernel,
                    (
                        x, weight, grad_proj_buf, grad_r_total_buf, r,
                        grad_x, grad_weight,
                        M, N, K,
                        tm, tn, tk,
                    ),
                )
            else:
                from types import SimpleNamespace

                configs = [SimpleNamespace(**c) for c in _fused_grad_x_weight_autotune_configs(N)]
                tuned = ct_experimental.autotune_launch(
                    stream,
                    grid_fn=lambda cfg: (math.ceil(K / cfg.TILE_SIZE_K),),
                    kernel=_ct_fused_grad_x_weight_kernel,
                    args_fn=lambda cfg: (
                        x, weight, grad_proj_buf, grad_r_total_buf, r,
                        grad_x, grad_weight,
                        M, N, K,
                        cfg.TILE_SIZE_M, cfg.TILE_SIZE_N, cfg.TILE_SIZE_K,
                    ),
                    search_space=configs,
                )
                best = tuned.tuned_config
                _fused_grad_x_weight_best_cfg[cache_key] = (
                    best.TILE_SIZE_M, best.TILE_SIZE_N, best.TILE_SIZE_K,
                )
                ct.launch(
                    stream,
                    (math.ceil(K / best.TILE_SIZE_K),),
                    _ct_fused_grad_x_weight_kernel,
                    (
                        x, weight, grad_proj_buf, grad_r_total_buf, r,
                        grad_x, grad_weight,
                        M, N, K,
                        best.TILE_SIZE_M, best.TILE_SIZE_N, best.TILE_SIZE_K,
                    ),
                )
        else:
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            ct.launch(
                stream,
                (num_sms, 2, 1),
                _ct_fused_compute_h_proj_rms_bwd_small_k_kernel,
                (
                    x, weight, grad_y, y_activated, proj, r, grad_r_ext_arg,
                    alpha_pre, alpha_post, alpha_res,
                    grad_x, grad_weight,
                    M, N, K, n, eps,
                    TILE_N, has_grad_r_ext_flag,
                ),
            )

        # 2. Separate lightweight kernel for scalar gradients (grad_alpha, grad_bias)
        tile_m_scalar = min(128, M)
        num_m_blocks = math.ceil(M / tile_m_scalar)
        grad_alpha_pre_partials = torch.empty(num_m_blocks, 1, dtype=torch.float32, device=dev)
        grad_alpha_post_partials = torch.empty(num_m_blocks, 1, dtype=torch.float32, device=dev)
        grad_alpha_res_partials = torch.empty(num_m_blocks, 1, dtype=torch.float32, device=dev)
        grad_bias_partials = torch.empty(num_m_blocks, TILE_N, dtype=torch.float32, device=dev)
        grad_alpha_pre = torch.empty(1, 1, dtype=alpha_pre.dtype, device=dev)
        grad_alpha_post = torch.empty(1, 1, dtype=alpha_post.dtype, device=dev)
        grad_alpha_res = torch.empty(1, 1, dtype=alpha_res.dtype, device=dev)
        grad_bias = torch.empty(1, TILE_N, dtype=bias.dtype, device=dev)

        ct.launch(
            stream,
            (num_m_blocks,),
            _ct_scalar_grads_partials_kernel,
            (
                grad_h_buf, proj, r,
                grad_alpha_pre_partials, grad_alpha_post_partials, grad_alpha_res_partials,
                grad_bias_partials,
                M, N, n, eps,
                tile_m_scalar, TILE_N,
            ),
        )
        ct.launch(
            stream,
            (1,),
            _ct_scalar_grads_reduce_kernel,
            (
                grad_alpha_pre_partials, grad_alpha_post_partials, grad_alpha_res_partials,
                grad_bias_partials,
                grad_alpha_pre, grad_alpha_post, grad_alpha_res, grad_bias,
                num_m_blocks, TILE_N,
            ),
        )

        return (
            grad_x,
            grad_weight,
            grad_alpha_pre.view_as(alpha_pre),
            grad_alpha_post.view_as(alpha_post),
            grad_alpha_res.view_as(alpha_res),
            grad_bias.view(-1)[:N],
        )


# ============================================================================
# Unified public dispatch
# ============================================================================
# The public fused API chooses the fastest validated backend per operation:
#
#   sinkhorn fwd/bwd:       Triton -> cuTile -> torch
#   h_post_bda fwd/bwd:     Triton -> cuTile -> torch
#   h_aggregate fwd:        Triton -> cuTile -> torch
#   h_aggregate bwd:        cuTile -> torch
#   proj_rms/proj_rms_compute_h: cuTile -> torch
#
# Runtime CUDA launch failures are intentionally not swallowed; after such an
# error the CUDA context may not be safely reusable for fallback work.
# ============================================================================

from megatron.core.transformer.hyper_connection import (
    native_fused_add_3 as fused_add_3,
    native_h_aggregate,
    native_h_post_bda,
    native_proj_rms,
    native_sinkhorn,
)


def _get_triton_sinkhorn():
    if not _TRITON_AVAILABLE:
        return None
    try:
        from megatron.core.fusions.triton_mhc_kernels import triton_fused_sinkhorn

        return triton_fused_sinkhorn
    except ImportError:
        return None


def _get_triton_h_aggregate_fwd():
    if not _TRITON_AVAILABLE:
        return None
    try:
        from megatron.core.fusions.triton_mhc_kernels import _triton_h_aggregate_fwd

        return _triton_h_aggregate_fwd
    except ImportError:
        return None


def _get_triton_h_post_bda_fwd():
    if not _TRITON_AVAILABLE:
        return None
    try:
        from megatron.core.fusions.triton_mhc_kernels import _triton_h_post_bda_fwd

        return _triton_h_post_bda_fwd
    except ImportError:
        return None


def _get_triton_h_post_bda_bwd():
    if not _TRITON_AVAILABLE:
        return None
    try:
        from megatron.core.fusions.triton_mhc_kernels import _triton_h_post_bda_bwd

        return _triton_h_post_bda_bwd
    except ImportError:
        return None


def _torch_h_aggregate_bwd(
    grad_output: Tensor, x: Tensor, h_pre: Tensor
) -> Tuple[Tensor, Tensor]:
    grad_x = grad_output.unsqueeze(2) * h_pre.unsqueeze(-1)
    grad_h = torch.sum(grad_output.unsqueeze(2) * x, dim=-1)
    return grad_x.to(dtype=x.dtype), grad_h.to(dtype=h_pre.dtype)


def _torch_h_post_bda_bwd(
    grad_output: Tensor,
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    s, b, n, C = original_residual.shape
    sb = s * b
    go = grad_output.reshape(sb, n, C)
    hr = h_res.reshape(sb, n, n)
    orig = original_residual.reshape(sb, n, C)
    hp = h_post.reshape(sb, n)
    x_flat = x.reshape(sb, C)

    g_hr = torch.bmm(go, orig.transpose(1, 2)).view(s, b, n, n)
    g_res = torch.bmm(hr.transpose(1, 2), go).view(s, b, n, C)
    g_x = torch.sum(go * hp.unsqueeze(-1), dim=1).view(s, b, C)
    xb = x_flat if bias is None else x_flat + bias.view(1, C)
    g_hp = torch.sum(go * xb.unsqueeze(1), dim=2).view(s, b, n)
    g_bias = g_x.reshape(sb, C).sum(dim=0).to(dtype=bias.dtype) if bias is not None else None
    return (
        g_hr.to(dtype=h_res.dtype),
        g_res.to(dtype=original_residual.dtype),
        g_hp.to(dtype=h_post.dtype),
        g_x.to(dtype=x.dtype),
        g_bias,
    )


def _torch_proj_rms_compute_h(
    x: Tensor,
    weight: Tensor,
    alpha_pre: Tensor,
    alpha_post: Tensor,
    alpha_res: Tensor,
    bias: Tensor,
    n: int,
    eps: float,
) -> Tuple[Tensor, Tensor]:
    proj = torch.matmul(x, weight.t())
    r = x.float().norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
    alpha = torch.cat(
        [
            alpha_pre.expand(n),
            alpha_post.expand(n),
            alpha_res.expand(weight.shape[0] - 2 * n),
        ],
        dim=-1,
    )
    h = proj * alpha.unsqueeze(0) / (r + eps) + bias.unsqueeze(0)
    h_pre = h[..., :n].sigmoid()
    h_post = h[..., n : 2 * n].sigmoid() * 2
    y_activated = torch.cat([h_pre, h_post, h[..., 2 * n :]], dim=-1)
    return y_activated, r.to(dtype=x.dtype)


if _CUTILE_AVAILABLE:

    class CutileSinkhornKnopp(torch.autograd.Function):
        """cuTile Sinkhorn-Knopp projection fallback."""

        @staticmethod
        def forward(ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6):
            output, M_init = _cutile_sinkhorn_fwd(input_logits, num_iterations, eps)
            ctx.save_for_backward(M_init)
            ctx.num_iterations = num_iterations
            ctx.eps = eps
            return output

        @staticmethod
        def backward(ctx, grad_output):
            (M_init,) = ctx.saved_tensors
            grad_input = _cutile_sinkhorn_bwd(grad_output, M_init, ctx.num_iterations, ctx.eps)
            return grad_input, None, None

    class CutileHAggregate(torch.autograd.Function):
        """cuTile n-stream weighted aggregation."""

        @staticmethod
        def forward(ctx, x: Tensor, h_pre: Tensor):
            output = _cutile_h_aggregate_fwd(x, h_pre)
            ctx.save_for_backward(x, h_pre)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            x, h_pre = ctx.saved_tensors
            return _cutile_h_aggregate_bwd(grad_output, x, h_pre)

    class CutileProjRms(torch.autograd.Function):
        """cuTile projection + RMS normalization."""

        @staticmethod
        def forward(ctx, x: Tensor, weight: Tensor, eps: float = 1e-6):
            proj, norm, r = _cutile_proj_rms_fwd(x, weight, eps)
            ctx.save_for_backward(x, weight, norm)
            ctx.eps = eps
            return proj, r

        @staticmethod
        def backward(ctx, grad_proj, grad_r):
            x, weight, norm = ctx.saved_tensors
            grad_x, grad_weight = _cutile_proj_rms_bwd(grad_proj, grad_r, x, weight, norm, ctx.eps)
            return grad_x, grad_weight, None

    class CutileProjRmsComputeH(torch.autograd.Function):
        """cuTile projection + RMS norm + compute_h activations."""

        @staticmethod
        def forward(
            ctx,
            x: Tensor,
            weight: Tensor,
            alpha_pre: Tensor,
            alpha_post: Tensor,
            alpha_res: Tensor,
            bias: Tensor,
            n: int,
            eps: float = 1e-6,
        ):
            y_activated, r, proj_reduced = _cutile_proj_rms_compute_h_fwd(
                x, weight, bias, alpha_pre, alpha_post, alpha_res, n, eps,
            )
            ctx.save_for_backward(
                x, weight, y_activated, proj_reduced, r,
                alpha_pre, alpha_post, alpha_res, bias,
            )
            ctx.n = n
            ctx.eps = eps
            return y_activated, r

        @staticmethod
        def backward(ctx, grad_y, grad_r_ext):
            (
                x, weight, y_activated, proj, r,
                alpha_pre, alpha_post, alpha_res, bias_param,
            ) = ctx.saved_tensors

            grad_x, grad_weight, grad_ap, grad_apo, grad_ar, grad_bias = (
                _cutile_fused_compute_h_proj_rms_bwd(
                    x, weight, grad_y, y_activated, proj, r, grad_r_ext,
                    alpha_pre, alpha_post, alpha_res,
                    bias_param, ctx.n, ctx.eps,
                )
            )

            return grad_x, grad_weight, grad_ap, grad_apo, grad_ar, grad_bias, None, None


class FusedHAggregate(torch.autograd.Function):
    """H_aggregate with Triton/cuTile/torch forward and cuTile/torch backward."""

    @staticmethod
    def forward(ctx, x: Tensor, h_pre: Tensor):
        triton_fwd = _get_triton_h_aggregate_fwd()
        if triton_fwd is not None:
            output = triton_fwd(x, h_pre)
        elif _CUTILE_AVAILABLE:
            output = _cutile_h_aggregate_fwd(x, h_pre)
        else:
            output = native_h_aggregate(x, h_pre)
        ctx.save_for_backward(x, h_pre)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, h_pre = ctx.saved_tensors
        if _CUTILE_AVAILABLE:
            return _cutile_h_aggregate_bwd(grad_output, x, h_pre)
        return _torch_h_aggregate_bwd(grad_output, x, h_pre)


class FusedHPostBDA(torch.autograd.Function):
    """H_post_bda with Triton/cuTile/torch forward and backward."""

    @staticmethod
    def forward(
        ctx,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Optional[Tensor],
    ):
        triton_fwd = _get_triton_h_post_bda_fwd()
        if triton_fwd is not None:
            output = triton_fwd(h_res, original_residual, h_post, x, bias)
        elif _CUTILE_AVAILABLE:
            output = _cutile_h_post_bda_fwd(h_res, original_residual, h_post, x, bias)
        else:
            output = native_h_post_bda(h_res, original_residual, h_post, x, bias)
        if bias is not None:
            ctx.save_for_backward(h_res, original_residual, h_post, x, bias)
            ctx.has_bias = True
        else:
            ctx.save_for_backward(h_res, original_residual, h_post, x)
            ctx.has_bias = False
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            h_res, orig_res, h_post, x, bias = ctx.saved_tensors
        else:
            h_res, orig_res, h_post, x = ctx.saved_tensors
            bias = None

        triton_bwd = _get_triton_h_post_bda_bwd()
        if triton_bwd is not None:
            return triton_bwd(grad_output, h_res, orig_res, h_post, x, bias)
        if _CUTILE_AVAILABLE:
            return _cutile_h_post_bda_bwd(grad_output, h_res, orig_res, h_post, x, bias)
        return _torch_h_post_bda_bwd(grad_output, h_res, orig_res, h_post, x, bias)


def fused_sinkhorn(input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
    """Project logits to a doubly stochastic matrix using Triton, cuTile, then torch."""
    triton_sinkhorn = _get_triton_sinkhorn()
    if triton_sinkhorn is not None:
        return triton_sinkhorn(input_logits, num_iterations, eps)
    if _CUTILE_AVAILABLE:
        return CutileSinkhornKnopp.apply(input_logits, num_iterations, eps)
    return native_sinkhorn(input_logits, num_iterations, eps)


def fused_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
    """Weighted n-stream to 1-stream aggregation using Triton/cuTile/torch."""
    if _TRITON_AVAILABLE or _CUTILE_AVAILABLE:
        return FusedHAggregate.apply(x, h_pre)
    return native_h_aggregate(x, h_pre)


def fused_h_post_bda(
    h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
) -> Tensor:
    """Fused H_res @ residual + H_post * (x + bias)."""
    if _TRITON_AVAILABLE or _CUTILE_AVAILABLE:
        return FusedHPostBDA.apply(h_res, original_residual, h_post, x, bias)
    return native_h_post_bda(h_res, original_residual, h_post, x, bias)


def fused_proj_rms(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """Projection + RMS normalization using cuTile, then torch."""
    if _CUTILE_AVAILABLE:
        return CutileProjRms.apply(x, weight, eps)
    return native_proj_rms(x, weight, eps)


def fused_proj_rms_compute_h(
    x: Tensor,
    weight: Tensor,
    alpha_pre: Tensor,
    alpha_post: Tensor,
    alpha_res: Tensor,
    bias: Tensor,
    n: int,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Projection + RMS norm + compute_h activations using cuTile, then torch."""
    if _CUTILE_AVAILABLE:
        return CutileProjRmsComputeH.apply(
            x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps
        )
    return _torch_proj_rms_compute_h(x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps)
