"""
FlexAttention CP parity tests: cp=1 vs cp={2,4,8} output accuracy.

Verifies that _forward_mot produces numerically equivalent results regardless
of context-parallel degree, including when token counts are not divisible by
cp_size (padding on the last rank).

Usage
-----
# smoke test (single GPU, cp=1 path only)
python  test_flex_attention_cp_parity.py --smoke

# cp=2 parity + padding
torchrun --nproc_per_node=2 test_flex_attention_cp_parity.py

# cp=4 parity + padding
torchrun --nproc_per_node=4 test_flex_attention_cp_parity.py

# cp=8 parity + padding
torchrun --nproc_per_node=8 test_flex_attention_cp_parity.py
"""

import argparse
import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "bagel-package"))
sys.path.insert(0, os.path.join(_ROOT, "bagel-package", "bagel"))

from torch.nn.attention.flex_attention import create_block_mask

from megatron.core.transformer.transformer_config import TransformerConfig




from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams  # noqa: E402
from megatron.core.models.bagel.flex_attention import FlexAttention               # noqa: E402


# =============================================================================
# Per-cp-size test parameters
#
# Constraints:
#   nh % cp_size == 0          (Ulysses head split)
#   u_par % cp_size == 0       (parity case: no padding, clean split)
#   g_par % cp_size == 0
#   u_pad % cp_size != 0       (padding case: last rank gets fewer tokens)
#   g_pad % cp_size != 0
#   u_pad // cp_size >= 1      (every rank gets ≥1 real und token)
#   g_pad // cp_size >= 1      (every rank gets ≥1 real gen token)
# =============================================================================

# cp_size -> SimpleNamespace(u_par, g_par, u_pad, g_pad, u_und, u_gen, nh, hd)
CONFIGS = {
    2: SimpleNamespace(
        # Parity (no padding): Lund=4, Lgen=2 — both ranks fully utilised
        u_par=8,  g_par=4,
        # Padding: Lund=3, Lgen=2 — rank 1: 1 und-pad, 1 gen-pad
        u_pad=5,  g_pad=3,
        # Single-branch: und-only Lund=4; gen-only Lgen=2
        u_und=8,  u_gen=4,
        nh=4,  hd=32,   # HIDDEN=128, H_cp=2
    ),
    4: SimpleNamespace(
        # Parity (no padding): Lund=2, Lgen=2
        u_par=8,  g_par=8,
        # Padding: Lund=3, Lgen=3 — rank 3: 1 und-pad, 1 gen-pad
        u_pad=11, g_pad=11,
        # Single-branch: und-only Lund=2; gen-only Lgen=2
        u_und=8,  u_gen=8,
        nh=4,  hd=32,   # HIDDEN=128, H_cp=1
    ),
    8: SimpleNamespace(
        # Parity (no padding): Lund=2, Lgen=2
        u_par=16, g_par=16,
        # Padding: Lund=2, Lgen=2 — rank 7: 1 und-pad, 1 gen-pad
        u_pad=15, g_pad=15,
        # Single-branch: und-only Lund=2; gen-only Lgen=2
        u_und=16, u_gen=16,
        nh=8,  hd=32,   # HIDDEN=256, H_cp=1
    ),
}


# =============================================================================
# Minimal ProcessGroupCollection-compatible object
# =============================================================================

class _PGC:
    def __init__(self, tp, cp=None):
        self.tp = tp
        if cp is not None:
            self.cp = cp


# =============================================================================
# Helpers
# =============================================================================

def _make_config(nh: int, hd: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=nh * hd,
        num_attention_heads=nh,
        num_query_groups=nh,
        kv_channels=hd,
        add_bias_linear=False,
        add_qkv_bias=False,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
    )


def _block_mask(seq_len: int, device: str):
    """Full-attention BlockMask over seq_len tokens."""
    return create_block_mask(
        lambda b, h, q, kv: q >= 0,
        B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len,
        device=device,
    )


def _make_qkv(seq_len: int, nh: int, hd: int, device: str, seed: int):
    torch.manual_seed(seed)
    q = torch.randn(seq_len, 1, nh, hd, dtype=torch.bfloat16, device=device)
    k = torch.randn(seq_len, 1, nh, hd, dtype=torch.bfloat16, device=device)
    v = torch.randn(seq_len, 1, nh, hd, dtype=torch.bfloat16, device=device)
    return q, k, v


def _build_local_qkv(q_full, k_full, v_full,
                     u, g, nh, hd, cp_size, rank,
                     noisy_padding=False, seed=0):
    """
    Slice + pad full [u+g, 1, nh, hd] q/k/v into per-rank compact layout
    [Lund+Lgen, 1, nh, hd].

    noisy_padding=True fills pad slots with random values instead of zeros,
    allowing tests to confirm padding is never read.

    Returns (q_local, k_local, v_local, Lund, Lgen, actual_lund, actual_lgen,
             und_idx_full, gen_idx_full, local_und_idx, local_gen_idx).
    """
    device = q_full.device
    Lund = math.ceil(u / cp_size)
    Lgen = math.ceil(g / cp_size)
    actual_lund = min(Lund, max(0, u - rank * Lund))
    actual_lgen = min(Lgen, max(0, g - rank * Lgen))

    und_idx_full = torch.arange(u,     device=device)
    gen_idx_full = torch.arange(u, u+g, device=device)
    local_und_idx = und_idx_full[rank * Lund : rank * Lund + actual_lund]
    local_gen_idx = gen_idx_full[rank * Lgen : rank * Lgen + actual_lgen]

    def _slice_pad(x, start, actual, padded, noisy):
        chunk = x[start : start + actual]
        n_pad = padded - actual
        if n_pad > 0:
            if noisy:
                torch.manual_seed(seed + start + 9999)
                fill = torch.randn(n_pad, 1, nh, hd, dtype=x.dtype, device=device)
            else:
                fill = x.new_zeros(n_pad, 1, nh, hd)
            chunk = torch.cat([chunk, fill], dim=0)
        return chunk

    def _local(x, noisy):
        und = _slice_pad(x, rank * Lund,     actual_lund, Lund, noisy)
        gen = _slice_pad(x, u + rank * Lgen, actual_lgen, Lgen, noisy)
        return torch.cat([und, gen], dim=0)

    return (
        _local(q_full, noisy_padding),
        _local(k_full, noisy_padding),
        _local(v_full, noisy_padding),
        Lund, Lgen, actual_lund, actual_lgen,
        und_idx_full, gen_idx_full, local_und_idx, local_gen_idx,
    )


# =============================================================================
# Smoke test (cp=1, single process)
# =============================================================================

def test_cp1_smoke(cfg):
    """Verify _forward_mot runs for cp=1 and produces finite output."""
    device = "cuda"
    nh, hd = cfg.nh, cfg.hd
    u, g   = cfg.u_par, cfg.g_par
    config = _make_config(nh, hd)

    class _TrivialPG:
        def size(self): return 1

    pgc = _PGC(tp=_TrivialPG(), cp=None)
    fa = FlexAttention(config=config, layer_number=1,
                       attn_mask_type=None, attention_type="self",
                       pg_collection=pgc).to(device)

    und_idx = torch.arange(u, device=device)
    gen_idx = torch.arange(u, u+g, device=device)
    psp = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx, packed_gen_token_indexes=gen_idx,
        local_und_token_indexes=und_idx,  local_gen_token_indexes=gen_idx,
        padded_und_seqlen=u, padded_gen_seqlen=g,
    )
    bm = _block_mask(u + g, device)
    q, k, v = _make_qkv(u+g, nh, hd, device, seed=42)

    with torch.no_grad():
        out = fa._forward_mot(q, k, v, psp, bm)

    hidden = nh * hd
    assert out.shape == (u+g, 1, hidden), f"Expected ({u+g},1,{hidden}), got {out.shape}"
    assert torch.all(torch.isfinite(out))
    print(f"  [cp=1 smoke] PASS  shape={out.shape}")


# =============================================================================
# Generic parity test: cp=1 baseline vs cp=N Ulysses
# =============================================================================

def run_parity_test(u, g, nh, hd, tp_group, cp_group, seed=42):
    """
    Forward + gradient parity: cp=1 (independent per rank) vs cp=N (Ulysses A2A).

    Token counts u, g must be divisible by cp_size for a clean split with no
    padding.  This isolates correctness of the A2A communication itself.
    """
    rank    = dist.get_rank()
    device  = "cuda"
    cp_size = cp_group.size()
    config  = _make_config(nh, hd)
    bm      = _block_mask(u + g, device)

    q_full, k_full, v_full = _make_qkv(u+g, nh, hd, device, seed=seed)

    (q_local, k_local, v_local,
     Lund, Lgen, actual_lund, actual_lgen,
     und_idx_full, gen_idx_full,
     local_und_idx, local_gen_idx) = _build_local_qkv(
        q_full, k_full, v_full, u, g, nh, hd, cp_size, rank, seed=seed
    )

    psp_cp1 = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx_full, packed_gen_token_indexes=gen_idx_full,
        local_und_token_indexes=und_idx_full,  local_gen_token_indexes=gen_idx_full,
        padded_und_seqlen=u, padded_gen_seqlen=g,
    )
    psp_cpN = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx_full, packed_gen_token_indexes=gen_idx_full,
        local_und_token_indexes=local_und_idx, local_gen_token_indexes=local_gen_idx,
        padded_und_seqlen=Lund, padded_gen_seqlen=Lgen,
    )

    pgc_cp1 = _PGC(tp=tp_group, cp=None)
    pgc_cpN = _PGC(tp=tp_group, cp=cp_group)
    fa_cp1  = FlexAttention(config=config, layer_number=1, attn_mask_type=None,
                            attention_type="self", pg_collection=pgc_cp1).to(device)
    fa_cpN  = FlexAttention(config=config, layer_number=1, attn_mask_type=None,
                            attention_type="self", pg_collection=pgc_cpN).to(device)

    # ── Forward ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        out_cp1 = fa_cp1._forward_mot(q_full,  k_full,  v_full,  psp_cp1, bm)
        out_cpN = fa_cpN._forward_mot(q_local, k_local, v_local, psp_cpN, bm)

    dist.barrier()

    und_ref = out_cp1[rank * Lund : rank * Lund + actual_lund]
    gen_ref = out_cp1[u + rank * Lgen : u + rank * Lgen + actual_lgen]
    und_got = out_cpN[:actual_lund]
    gen_got = out_cpN[Lund : Lund + actual_lgen]

    atol = rtol = 1e-2
    torch.testing.assert_close(und_got, und_ref, atol=atol, rtol=rtol,
        msg=lambda m: f"cp={cp_size} rank={rank} UND fwd: {m}")
    torch.testing.assert_close(gen_got, gen_ref, atol=atol, rtol=rtol,
        msg=lambda m: f"cp={cp_size} rank={rank} GEN fwd: {m}")

    und_err = (und_got - und_ref).abs().max().item()
    gen_err = (gen_got - gen_ref).abs().max().item()
    print(f"  [cp={cp_size} parity fwd  rank={rank:2d}] PASS  "
          f"Lund={Lund} Lgen={Lgen}  "
          f"und_err={und_err:.4f}  gen_err={gen_err:.4f}")

    # ── Gradient ──────────────────────────────────────────────────────────────
    q_full2, k_full2, v_full2 = _make_qkv(u+g, nh, hd, device, seed=seed+1)
    q_full2 = q_full2.requires_grad_(True)

    out_cp1_g = fa_cp1._forward_mot(q_full2, k_full2, v_full2, psp_cp1, bm)
    out_cp1_g.sum().backward()
    grad_cp1 = q_full2.grad.clone()

    (q_loc2, k_loc2, v_loc2, *_) = _build_local_qkv(
        q_full2.detach(), k_full2, v_full2, u, g, nh, hd, cp_size, rank, seed=seed+1
    )
    q_und_leaf = q_loc2[:Lund].detach().requires_grad_(True)
    q_gen_leaf = q_loc2[Lund:].detach().requires_grad_(True)
    q_loc2_leaf = torch.cat([q_und_leaf, q_gen_leaf], dim=0)

    out_cpN_g = fa_cpN._forward_mot(q_loc2_leaf, k_loc2, v_loc2, psp_cpN, bm)
    (out_cpN_g[:actual_lund].sum() + out_cpN_g[Lund:Lund+actual_lgen].sum()).backward()

    dist.barrier()

    grad_und_ref = grad_cp1[rank * Lund : rank * Lund + actual_lund]
    grad_gen_ref = grad_cp1[u + rank * Lgen : u + rank * Lgen + actual_lgen]
    grad_und_got = q_und_leaf.grad[:actual_lund]
    grad_gen_got = q_gen_leaf.grad[:actual_lgen]

    atol = rtol = 2e-2
    torch.testing.assert_close(grad_und_got, grad_und_ref, atol=atol, rtol=rtol,
        msg=lambda m: f"cp={cp_size} rank={rank} UND grad: {m}")
    torch.testing.assert_close(grad_gen_got, grad_gen_ref, atol=atol, rtol=rtol,
        msg=lambda m: f"cp={cp_size} rank={rank} GEN grad: {m}")

    und_g_err = (grad_und_got - grad_und_ref).abs().max().item()
    gen_g_err = (grad_gen_got - grad_gen_ref).abs().max().item()
    print(f"  [cp={cp_size} parity grad rank={rank:2d}] PASS  "
          f"und_err={und_g_err:.4f}  gen_err={gen_g_err:.4f}")


# =============================================================================
# Generic padding test
# =============================================================================

def run_padding_test(u, g, nh, hd, tp_group, cp_group, seed=77):
    """
    Three checks for cp=N when u % cp_size != 0 or g % cp_size != 0.

    A — Forward parity  : real-token outputs match cp=1 baseline.
    B — Padding isolation: replacing zero-padding with random noise must not
                           change the real-token outputs (exact equality).
    C — Zero gradient   : backward through non-padded outputs leaves grad=0
                          at all padding positions.
    """
    rank    = dist.get_rank()
    device  = "cuda"
    cp_size = cp_group.size()
    config  = _make_config(nh, hd)
    bm      = _block_mask(u + g, device)

    q_full, k_full, v_full = _make_qkv(u+g, nh, hd, device, seed=seed)

    (q_local, k_local, v_local,
     Lund, Lgen, actual_lund, actual_lgen,
     und_idx_full, gen_idx_full,
     local_und_idx, local_gen_idx) = _build_local_qkv(
        q_full, k_full, v_full, u, g, nh, hd, cp_size, rank,
        noisy_padding=False, seed=seed,
    )
    n_pad_und = Lund - actual_lund
    n_pad_gen = Lgen - actual_lgen

    psp_cp1 = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx_full, packed_gen_token_indexes=gen_idx_full,
        local_und_token_indexes=und_idx_full,  local_gen_token_indexes=gen_idx_full,
        padded_und_seqlen=u, padded_gen_seqlen=g,
    )
    psp_cpN = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx_full, packed_gen_token_indexes=gen_idx_full,
        local_und_token_indexes=local_und_idx, local_gen_token_indexes=local_gen_idx,
        padded_und_seqlen=Lund, padded_gen_seqlen=Lgen,
    )

    pgc_cp1 = _PGC(tp=tp_group, cp=None)
    pgc_cpN = _PGC(tp=tp_group, cp=cp_group)
    fa_cp1  = FlexAttention(config=config, layer_number=1, attn_mask_type=None,
                            attention_type="self", pg_collection=pgc_cp1).to(device)
    fa_cpN  = FlexAttention(config=config, layer_number=1, attn_mask_type=None,
                            attention_type="self", pg_collection=pgc_cpN).to(device)

    # ── Check A: forward parity ───────────────────────────────────────────────
    with torch.no_grad():
        out_cp1 = fa_cp1._forward_mot(q_full,  k_full,  v_full,  psp_cp1, bm)
        out_cpN = fa_cpN._forward_mot(q_local, k_local, v_local, psp_cpN, bm)

    dist.barrier()

    und_ref = out_cp1[rank * Lund : rank * Lund + actual_lund]
    gen_ref = out_cp1[u + rank * Lgen : u + rank * Lgen + actual_lgen]
    und_got = out_cpN[:actual_lund]
    gen_got = out_cpN[Lund : Lund + actual_lgen]

    atol = rtol = 1e-2
    torch.testing.assert_close(und_got, und_ref, atol=atol, rtol=rtol,
        msg=lambda m: f"[pad/A] cp={cp_size} rank={rank} UND: {m}")
    torch.testing.assert_close(gen_got, gen_ref, atol=atol, rtol=rtol,
        msg=lambda m: f"[pad/A] cp={cp_size} rank={rank} GEN: {m}")

    und_err = (und_got - und_ref).abs().max().item() if actual_lund > 0 else 0.0
    gen_err = (gen_got - gen_ref).abs().max().item() if actual_lgen > 0 else 0.0
    print(f"  [cp={cp_size} pad/A fwd   rank={rank:2d}] PASS  "
          f"n_pad=({n_pad_und},{n_pad_gen})  "
          f"und_err={und_err:.4f}  gen_err={gen_err:.4f}")

    # ── Check B: padding values are never read ────────────────────────────────
    (q_noisy, k_noisy, v_noisy, *_) = _build_local_qkv(
        q_full, k_full, v_full, u, g, nh, hd, cp_size, rank,
        noisy_padding=True, seed=seed,
    )
    with torch.no_grad():
        out_noisy = fa_cpN._forward_mot(q_noisy, k_noisy, v_noisy, psp_cpN, bm)

    dist.barrier()

    torch.testing.assert_close(
        out_cpN[:actual_lund], out_noisy[:actual_lund], atol=0, rtol=0,
        msg=lambda m: f"[pad/B] cp={cp_size} rank={rank} UND: noisy padding leaked: {m}",
    )
    torch.testing.assert_close(
        out_cpN[Lund : Lund + actual_lgen], out_noisy[Lund : Lund + actual_lgen],
        atol=0, rtol=0,
        msg=lambda m: f"[pad/B] cp={cp_size} rank={rank} GEN: noisy padding leaked: {m}",
    )
    print(f"  [cp={cp_size} pad/B noise rank={rank:2d}] PASS  "
          f"padding values do not reach real tokens")

    # ── Check C: gradient is zero at padding positions ────────────────────────
    q_und_leaf = q_local[:Lund].detach().requires_grad_(True)
    q_gen_leaf = q_local[Lund:].detach().requires_grad_(True)
    q_leaf = torch.cat([q_und_leaf, q_gen_leaf], dim=0)

    out_g = fa_cpN._forward_mot(q_leaf, k_local, v_local, psp_cpN, bm)
    (out_g[:actual_lund].sum() + out_g[Lund : Lund + actual_lgen].sum()).backward()

    dist.barrier()

    if n_pad_und > 0:
        pad_grad = q_und_leaf.grad[actual_lund:]
        assert torch.all(pad_grad == 0), (
            f"cp={cp_size} rank={rank}: und-pad grad != 0, "
            f"max={pad_grad.abs().max().item():.4e}"
        )
    if n_pad_gen > 0:
        pad_grad = q_gen_leaf.grad[actual_lgen:]
        assert torch.all(pad_grad == 0), (
            f"cp={cp_size} rank={rank}: gen-pad grad != 0, "
            f"max={pad_grad.abs().max().item():.4e}"
        )

    print(f"  [cp={cp_size} pad/C grad  rank={rank:2d}] PASS  "
          f"grad=0 at {n_pad_und} und-pad and {n_pad_gen} gen-pad slots")


# =============================================================================
# Single-branch test: und-only (G=0) or gen-only (U=0)
# =============================================================================

def run_single_branch_test(branch, u, g, nh, hd, tp_group, cp_group, seed=99):
    """
    Forward parity + gradient flow for single-branch inputs.

    branch='und'  U>0, G=0 — only understanding tokens are present.
    branch='gen'  U=0, G>0 — only generation tokens are present.

    Checks:
      A — Forward parity: cp=N output matches cp=1 baseline for the active
          branch tokens on every rank.
      B — Gradient flow : backward through active-branch outputs produces
          non-zero gradient in the corresponding q leaf.
    """
    assert branch in ("und", "gen"), f"unknown branch {branch!r}"
    rank    = dist.get_rank()
    device  = "cuda"
    cp_size = cp_group.size()
    config  = _make_config(nh, hd)

    total = u + g          # u+0 = u  or  0+g = g
    bm    = _block_mask(total, device)

    q_full, k_full, v_full = _make_qkv(total, nh, hd, device, seed=seed)

    (q_local, k_local, v_local,
     Lund, Lgen, actual_lund, actual_lgen,
     und_idx_full, gen_idx_full,
     local_und_idx, local_gen_idx) = _build_local_qkv(
        q_full, k_full, v_full, u, g, nh, hd, cp_size, rank, seed=seed
    )

    # cp=1 baseline sees the full (single-branch) sequence
    psp_cp1 = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx_full, packed_gen_token_indexes=gen_idx_full,
        local_und_token_indexes=und_idx_full,  local_gen_token_indexes=gen_idx_full,
        padded_und_seqlen=u, padded_gen_seqlen=g,
    )
    psp_cpN = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx_full, packed_gen_token_indexes=gen_idx_full,
        local_und_token_indexes=local_und_idx, local_gen_token_indexes=local_gen_idx,
        padded_und_seqlen=Lund, padded_gen_seqlen=Lgen,
    )

    pgc_cp1 = _PGC(tp=tp_group, cp=None)
    pgc_cpN = _PGC(tp=tp_group, cp=cp_group)
    fa_cp1  = FlexAttention(config=config, layer_number=1, attn_mask_type=None,
                            attention_type="self", pg_collection=pgc_cp1).to(device)
    fa_cpN  = FlexAttention(config=config, layer_number=1, attn_mask_type=None,
                            attention_type="self", pg_collection=pgc_cpN).to(device)

    # ── Check A: forward parity ───────────────────────────────────────────────
    with torch.no_grad():
        out_cp1 = fa_cp1._forward_mot(q_full,  k_full,  v_full,  psp_cp1, bm)
        out_cpN = fa_cpN._forward_mot(q_local, k_local, v_local, psp_cpN, bm)

    dist.barrier()

    # Expected output shapes: und-only → [Lund,1,H]; gen-only → [Lgen,1,H]
    hidden = nh * hd
    if branch == "und":
        assert out_cpN.shape == (Lund, 1, hidden), \
            f"und-only output shape: expected ({Lund},1,{hidden}), got {out_cpN.shape}"
        if actual_lund > 0:
            und_ref = out_cp1[rank * Lund : rank * Lund + actual_lund]
            und_got = out_cpN[:actual_lund]
            torch.testing.assert_close(und_got, und_ref, atol=1e-2, rtol=1e-2,
                msg=lambda m: f"[und-only] cp={cp_size} rank={rank} UND fwd: {m}")
            err = (und_got - und_ref).abs().max().item()
        else:
            err = 0.0
    else:  # gen
        assert out_cpN.shape == (Lgen, 1, hidden), \
            f"gen-only output shape: expected ({Lgen},1,{hidden}), got {out_cpN.shape}"
        if actual_lgen > 0:
            # For U=0, Lund=0 → gen tokens start at index 0 in cp=1 output
            gen_ref = out_cp1[rank * Lgen : rank * Lgen + actual_lgen]
            gen_got = out_cpN[:actual_lgen]   # Lund=0, so gen starts at 0
            torch.testing.assert_close(gen_got, gen_ref, atol=1e-2, rtol=1e-2,
                msg=lambda m: f"[gen-only] cp={cp_size} rank={rank} GEN fwd: {m}")
            err = (gen_got - gen_ref).abs().max().item()
        else:
            err = 0.0

    print(f"  [cp={cp_size} {branch}-only fwd  rank={rank:2d}] PASS  "
          f"Lund={Lund} Lgen={Lgen}  err={err:.4f}")

    # ── Check B: gradient flows into the active branch ────────────────────────
    q_und_leaf = q_local[:Lund].detach().requires_grad_(True)
    q_gen_leaf = q_local[Lund:].detach().requires_grad_(True)
    q_leaf     = torch.cat([q_und_leaf, q_gen_leaf], dim=0)

    out_g = fa_cpN._forward_mot(q_leaf, k_local, v_local, psp_cpN, bm)

    if branch == "und" and actual_lund > 0:
        out_g[:actual_lund].sum().backward()
        grad_sum = q_und_leaf.grad[:actual_lund].abs().sum().item()
        assert grad_sum > 0, \
            f"[und-only] cp={cp_size} rank={rank}: und grad is zero (expected non-zero)"
    elif branch == "gen" and actual_lgen > 0:
        # Lund=0, gen starts at index 0
        out_g[:actual_lgen].sum().backward()
        grad_sum = q_gen_leaf.grad[:actual_lgen].abs().sum().item()
        assert grad_sum > 0, \
            f"[gen-only] cp={cp_size} rank={rank}: gen grad is zero (expected non-zero)"
    else:
        grad_sum = 0.0  # rank has no real tokens (shouldn't happen with chosen params)

    dist.barrier()
    print(f"  [cp={cp_size} {branch}-only grad rank={rank:2d}] PASS  "
          f"grad_sum={grad_sum:.4f}")


# =============================================================================
# MoT vs non-MoT comparison test
# =============================================================================

def run_mot_vs_nonmot_test(u, g, nh, hd, tp_group, cp_group, seed=42):
    """
    Compare _forward_mot (type-balanced CP, cp=N) against the standard non-MoT
    forward path (full sequence, single process, no type splitting).

    The non-MoT path calls fa.forward(...) without packed_seq_params: it runs
    compiled_flex_attention over all U+G tokens on each rank independently.

    The MoT path calls fa._forward_mot(...) with MoTPackedSeqParams: tokens are
    partitioned by type (und/gen) across cp ranks, Ulysses A2A gathers heads,
    attention is computed over all U+G global positions, then A2A_inv reassembles.

    For full attention (no causal mask) both paths compute the identical function
    so their outputs must agree within floating-point tolerance.

    Covers:
      - Forward parity: und and gen real-token outputs match on every rank.
      - Shape check: _forward_mot output shape == [Lund+Lgen, 1, hidden].
    """
    rank    = dist.get_rank()
    device  = "cuda"
    cp_size = cp_group.size()
    config  = _make_config(nh, hd)
    hidden  = nh * hd

    # Full block mask over all U+G tokens (full attention, no causal)
    bm = _block_mask(u + g, device)

    q_full, k_full, v_full = _make_qkv(u + g, nh, hd, device, seed=seed)

    (q_local, k_local, v_local,
     Lund, Lgen, actual_lund, actual_lgen,
     und_idx_full, gen_idx_full,
     local_und_idx, local_gen_idx) = _build_local_qkv(
        q_full, k_full, v_full, u, g, nh, hd, cp_size, rank, seed=seed
    )

    psp_cpN = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx_full, packed_gen_token_indexes=gen_idx_full,
        local_und_token_indexes=local_und_idx, local_gen_token_indexes=local_gen_idx,
        padded_und_seqlen=Lund, padded_gen_seqlen=Lgen,
    )

    # Non-MoT reference: standard forward, full sequence, no CP
    pgc_ref = _PGC(tp=tp_group, cp=None)
    fa_ref  = FlexAttention(config=config, layer_number=1, attn_mask_type=None,
                            attention_type="self", pg_collection=pgc_ref).to(device)

    # MoT path: type-balanced CP with Ulysses A2A
    pgc_mot = _PGC(tp=tp_group, cp=cp_group)
    fa_mot  = FlexAttention(config=config, layer_number=1, attn_mask_type=None,
                            attention_type="self", pg_collection=pgc_mot).to(device)

    with torch.no_grad():
        # Standard path: fa.forward without packed_seq_params → no MoT dispatch
        out_ref = fa_ref.forward(q_full, k_full, v_full,
                                 attention_mask=bm, packed_seq_params=None)
        # MoT path: type-balanced Ulysses A2A
        out_mot = fa_mot._forward_mot(q_local, k_local, v_local, psp_cpN, bm)

    dist.barrier()

    # Shape check
    assert out_mot.shape == (Lund + Lgen, 1, hidden), (
        f"cp={cp_size} rank={rank}: unexpected shape {out_mot.shape}, "
        f"expected ({Lund+Lgen}, 1, {hidden})"
    )

    # Compare real und tokens
    und_ref = out_ref[rank * Lund : rank * Lund + actual_lund]
    gen_ref = out_ref[u + rank * Lgen : u + rank * Lgen + actual_lgen]
    und_got = out_mot[:actual_lund]
    gen_got = out_mot[Lund : Lund + actual_lgen]

    atol = rtol = 1e-2
    if actual_lund > 0:
        torch.testing.assert_close(und_got, und_ref, atol=atol, rtol=rtol,
            msg=lambda m: f"[mot_vs_nonmot] cp={cp_size} rank={rank} UND: {m}")
    if actual_lgen > 0:
        torch.testing.assert_close(gen_got, gen_ref, atol=atol, rtol=rtol,
            msg=lambda m: f"[mot_vs_nonmot] cp={cp_size} rank={rank} GEN: {m}")

    und_err = (und_got - und_ref).abs().max().item() if actual_lund > 0 else 0.0
    gen_err = (gen_got - gen_ref).abs().max().item() if actual_lgen > 0 else 0.0
    print(f"  [cp={cp_size} mot_vs_nonmot rank={rank:2d}] PASS  "
          f"Lund={Lund} Lgen={Lgen}  und_err={und_err:.4f}  gen_err={gen_err:.4f}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29500",
                                rank=0, world_size=1)
        test_cp1_smoke(CONFIGS[2])
        dist.destroy_process_group()
        print("\nSmoke test passed.")
        return

    # ── Distributed (torchrun) ────────────────────────────────────────────────
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    assert world_size in CONFIGS, (
        f"No config for world_size={world_size}. "
        f"Supported: {sorted(CONFIGS.keys())}"
    )
    cfg = CONFIGS[world_size]

    # Per-rank trivial tp group — ALL ranks must call new_group for each group.
    tp_groups = [dist.new_group(ranks=[r]) for r in range(world_size)]
    tp_group  = tp_groups[rank]

    # cp group spanning all ranks
    cp_group = dist.new_group(ranks=list(range(world_size)))

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  cp={world_size}  nh={cfg.nh}  hd={cfg.hd}  HIDDEN={cfg.nh*cfg.hd}")
        print(f"  Parity test    : U={cfg.u_par}  G={cfg.g_par}"
              f"  (Lund={math.ceil(cfg.u_par/world_size)}"
              f"  Lgen={math.ceil(cfg.g_par/world_size)})")
        print(f"  Padding test   : U={cfg.u_pad}  G={cfg.g_pad}"
              f"  (Lund={math.ceil(cfg.u_pad/world_size)}"
              f"  Lgen={math.ceil(cfg.g_pad/world_size)})")
        print(f"  und-only test  : U={cfg.u_und}  G=0"
              f"  (Lund={math.ceil(cfg.u_und/world_size)})")
        print(f"  gen-only test  : U=0   G={cfg.u_gen}"
              f"  (Lgen={math.ceil(cfg.u_gen/world_size)})")
        print(f"  mot vs nonmot  : U={cfg.u_par}  G={cfg.g_par}"
              f"  (Lund={math.ceil(cfg.u_par/world_size)}"
              f"  Lgen={math.ceil(cfg.g_par/world_size)})")
        print(f"{'='*60}")

    dist.barrier()

    # ── Parity tests (no padding) ─────────────────────────────────────────────
    if rank == 0:
        print("\n--- Parity (no padding) ---")
    dist.barrier()

    run_parity_test(cfg.u_par, cfg.g_par, cfg.nh, cfg.hd, tp_group, cp_group)
    dist.barrier()

    # ── Padding tests ─────────────────────────────────────────────────────────
    if rank == 0:
        print("\n--- Padding ---")
    dist.barrier()

    run_padding_test(cfg.u_pad, cfg.g_pad, cfg.nh, cfg.hd, tp_group, cp_group)
    dist.barrier()

    # ── Single-branch tests ────────────────────────────────────────────────────
    if rank == 0:
        print("\n--- Single-branch: und-only (G=0) ---")
    dist.barrier()

    run_single_branch_test("und", cfg.u_und, 0, cfg.nh, cfg.hd, tp_group, cp_group)
    dist.barrier()

    if rank == 0:
        print("\n--- Single-branch: gen-only (U=0) ---")
    dist.barrier()

    run_single_branch_test("gen", 0, cfg.u_gen, cfg.nh, cfg.hd, tp_group, cp_group)
    dist.barrier()

    # ── MoT vs non-MoT comparison ──────────────────────────────────────────────
    if rank == 0:
        print("\n--- MoT (type-balanced CP) vs non-MoT (global-ordered, no CP) ---")
    dist.barrier()

    run_mot_vs_nonmot_test(cfg.u_par, cfg.g_par, cfg.nh, cfg.hd, tp_group, cp_group)
    dist.barrier()

    if rank == 0:
        print(f"\nAll cp={world_size} tests passed.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
