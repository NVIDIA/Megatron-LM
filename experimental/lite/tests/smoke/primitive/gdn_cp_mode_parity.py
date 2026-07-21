# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 GatedDeltaNet CP-mode precision parity proxy.

Matrix (baseline = CP off / full sequence, cp_size=1):
  SBHD: gdn_cp_mode in {"headwise"(default), "chunkwise"} x CP in {cp2/cp4}
  packed THD: gdn_cp_mode="chunkwise" (the RL path) vs CP-off dense reference

``headwise`` is bitwise-exact vs CP-off; ``chunkwise`` runs the FLA ring
(cross-rank chunk-state reassociation) so it is expected at the bf16 floor, not bitwise.
The packed THD arm exercises the packing-aware reshuffle the old ``sharded`` copy broke.

Design
------
The primitive input is SBHD ``x = [seq, batch, hidden]``; CP shards along the
seq dim (dim 0) with the Megatron **zigzag** layout. Each rank feeds its zigzag
shard; the module internally head-parallelises / reconstructs the sequence and
returns the rank-local zigzag shard of the output.

- ``headwise`` (default): head-parallel all-to-all. Each rank ends up with the
  full sequence for ``1/cp`` of the heads (plus the matching conv1d/A_log/dt_bias
  slices), runs the ordinary full-sequence recurrence, then all-to-alls back. Heads
  are independent -> **bitwise identical** to the CP-off reference, memory sharded.
  on every rank, then zigzag-slice the output. Also bitwise identical to CP-off, but
  every rank materialises the full sequence for all heads (worst memory).

Reference = a cp_size=1 module (identical weights) run on the FULL sequence on
rank 0. For every rank we compare its CP output to the zigzag slice of the
reference output (forward), and likewise for the input gradient; weight grads are
CP-all-reduced then compared to the reference weight grads. Both modes are expected
to be bitwise-exact (max_abs == 0) vs the reference.

Run under: torchrun --nproc_per_node={2,4} gdn_cp_mode_parity.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist

# --- proxy geometry: REAL Qwen3.5 GDN head config (16 k / 32 v heads, dk=dv=128,
# conv=4). Head counts are kept real (not truncated) because headwise CP shards
# heads across ranks, so num_k_heads must stay divisible by cp_size (>=4 here). ---
HIDDEN = 256
NUM_K_HEADS = 16
K_HEAD_DIM = 128
NUM_V_HEADS = 32
V_HEAD_DIM = 128
CONV_KERNEL = 4
RMS_EPS = 1e-6
SEQ = 2048          # divisible by 2*cp_size for cp in {2,4} and by FLA chunk 64
BATCH = 1
DTYPE = torch.bfloat16
SEED = 20260714


def _make_ps(cp_size, cp_rank, cp_group):
    from megatron.lite.primitive.parallel.state import ParallelState

    return ParallelState(cp_group=cp_group, cp_size=cp_size, cp_rank=cp_rank)


def _make_gdn(cp_size, cp_rank, cp_group, cp_mode):
    from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet

    return GatedDeltaNet(
        hidden_size=HIDDEN,
        linear_num_key_heads=NUM_K_HEADS,
        linear_key_head_dim=K_HEAD_DIM,
        linear_num_value_heads=NUM_V_HEADS,
        linear_value_head_dim=V_HEAD_DIM,
        linear_conv_kernel_dim=CONV_KERNEL,
        rms_norm_eps=RMS_EPS,
        ps=_make_ps(cp_size, cp_rank, cp_group),
        deterministic=True,
        cp_mode=cp_mode,
    )


def _randomize_and_broadcast(module):
    """Give the module non-degenerate weights, identical across ranks (src=0)."""
    torch.manual_seed(SEED)
    with torch.no_grad():
        for name, p in module.named_parameters():
            # A_log defaults to 0 and dt_bias to 1 -> gating is trivial; perturb so
            # the delta-rule recurrence is genuinely exercised.
            if name.endswith("A_log"):
                p.copy_(torch.randn_like(p) * 0.5 - 1.0)
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn_like(p) * 0.1)
            else:
                p.mul_(1.0)  # keep nn default init (already random for lin/conv)
    for p in module.parameters():
        dist.broadcast(p.data, src=0)
    for b in module.buffers():
        dist.broadcast(b.data, src=0)


def _diff(a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    max_abs = d.max().item()
    scale = b.abs().max().clamp_min(1e-8)
    max_rel = (d.max() / scale).item()
    return max_abs, max_rel, scale.item()


def run_matrix(cp_size, cp_rank, cp_group, cp1_group, device, rank, world):
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp

    results = {}
    # full-sequence input, identical on every rank
    torch.manual_seed(SEED + 1)
    full_x = torch.randn(SEQ, BATCH, HIDDEN, device=device, dtype=DTYPE)
    dist.broadcast(full_x, src=0)
    torch.manual_seed(SEED + 2)
    ref_cotangent = torch.randn(SEQ, BATCH, HIDDEN, device=device, dtype=DTYPE)
    dist.broadcast(ref_cotangent, src=0)

    # ----- reference (cp off, cp1) is built per-mode on rank 0 with identical weights -----
    ref_out_full = None
    ref_in_grad_full = None
    ref_wgrads = None

    # headwise is bitwise-exact vs CP-off (heads independent); chunkwise
    # runs the FLA ring so it reassociates in bf16 and is expected at the bf16 floor,
    # not bitwise (still orders of magnitude below the old packed-path O(1) corruption).
    for cp_mode in ("headwise", "chunkwise"):
        cp_mod = _make_gdn(cp_size, cp_rank, cp_group, cp_mode).to(device=device, dtype=DTYPE)
        _randomize_and_broadcast(cp_mod)

        # reference module (cp1) with identical weights, on rank 0
        ref_mod = None
        if rank == 0:
            ref_mod = _make_gdn(1, 0, cp1_group, cp_mode).to(device=device, dtype=DTYPE)
            ref_mod.load_state_dict(cp_mod.state_dict())

        # ---- forward ----
        local_x = (
            zigzag_slice_for_cp(full_x, cp_rank, cp_size, seq_dim=0)
            .detach()
            .requires_grad_(True)
        )
        cp_out = cp_mod(local_x)  # [S_local, B, H]

        # Compare each rank's local out vs the zigzag slice of the reference out.
        if rank == 0:
            ref_x = full_x.detach().requires_grad_(True)
            with torch.enable_grad():
                ref_out = ref_mod(ref_x)
            # backward on reference with the zigzag-sliced cotangent (full)
            ref_out.backward(ref_cotangent)
            ref_in_grad_full = ref_x.grad.detach().clone()
            ref_wgrads = {n: (p.grad.detach().clone() if p.grad is not None else None)
                          for n, p in ref_mod.named_parameters()}
            ref_out_full = ref_out.detach().clone()

        # broadcast ref_out_full so every rank can slice its own expected shard
        if ref_out_full is None:
            ref_out_full = torch.empty(SEQ, BATCH, HIDDEN, device=device, dtype=DTYPE)
        dist.broadcast(ref_out_full, src=0)
        expected_local = zigzag_slice_for_cp(ref_out_full, cp_rank, cp_size, seq_dim=0)
        f_abs, f_rel, f_scale = _diff(cp_out.detach(), expected_local)

        # ---- backward ---- (cotangent = zigzag slice of the shared full cotangent)
        local_cot = zigzag_slice_for_cp(ref_cotangent, cp_rank, cp_size, seq_dim=0)
        cp_out.backward(local_cot)
        cp_in_grad = local_x.grad.detach().clone()

        # broadcast ref input grad; compare zigzag slice
        if ref_in_grad_full is None:
            ref_in_grad_full = torch.empty(SEQ, BATCH, HIDDEN, device=device, dtype=DTYPE)
        dist.broadcast(ref_in_grad_full, src=0)
        expected_in_grad = zigzag_slice_for_cp(ref_in_grad_full, cp_rank, cp_size, seq_dim=0)
        g_abs, g_rel, g_scale = _diff(cp_in_grad, expected_in_grad)

        # weight grads: CP-all-reduce (sum) then compare on rank 0
        for n, p in cp_mod.named_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=cp_group)
        w_abs = w_rel = 0.0
        worst_w = None
        if rank == 0:
            for n, p in cp_mod.named_parameters():
                rg = ref_wgrads.get(n)
                if rg is None:
                    rg = torch.zeros_like(p.grad)
                a, r, _ = _diff(p.grad, rg)
                if a > w_abs:
                    w_abs, worst_w = a, n
                w_rel = max(w_rel, r)

        results[cp_mode] = dict(
            f_abs=f_abs, f_rel=f_rel, f_scale=f_scale,
            g_abs=g_abs, g_rel=g_rel, g_scale=g_scale,
            w_abs=w_abs, w_rel=w_rel, worst_w=worst_w,
        )
        if cp_rank == 0:
            print(
                f"GDN_CP_PARITY mode={cp_mode} cp={cp_size} seq={SEQ} "
                f"fwd[max_abs={f_abs:.3e} max_rel={f_rel:.3e} scale={f_scale:.3e}] "
                f"in_grad[max_abs={g_abs:.3e} max_rel={g_rel:.3e} scale={g_scale:.3e}] "
                f"w_grad[max_abs={w_abs:.3e} max_rel={w_rel:.3e} worst={worst_w}]",
                flush=True,
            )
        del cp_mod, ref_mod
        torch.cuda.empty_cache()
        dist.barrier()

    return results


# Packed THD lengths (each divisible by 2*cp for cp in {2,4}); a sequence deliberately
# spans the contiguous CP-rank boundary so the packing-aware reshuffle is exercised.
PACKED_LENS = [1024, 512, 512]  # total 2048


def run_packed_matrix(cp_size, cp_rank, cp_group, cp1_group, device, rank, world):
    """Packed THD chunkwise parity: CP chunkwise vs CP-off dense reference.

    This is the path RL actually runs (``impl_cfg.use_thd=True``) and the one the old
    ``sharded`` copy corrupted. Reference = a cp1 module on the FULL packed sequence;
    each CP rank is fed its zigzag shard + the *global* cu_seqlens. A correct
    packing-aware reshuffle keeps the packed chunkwise output at the bf16 ring floor vs
    the reference (was ~O(1) / 220x-ppo_kl before the fix).
    """
    from megatron.lite.primitive.parallel.thd import split_packed_to_cp_local
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    lens = torch.tensor(PACKED_LENS, dtype=torch.int32, device=device)
    cu = torch.zeros(len(PACKED_LENS) + 1, dtype=torch.int32, device=device)
    torch.cumsum(lens, dim=0, out=cu[1:])
    total = int(cu[-1].item())
    max_seqlen = int(lens.max().item())

    def _psp(cp_sz, cp_rk, grp):
        extra = {}
        if cp_sz > 1:
            extra["local_cp_size"] = cp_sz
            extra["cp_group"] = grp
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            cp_rank=cp_rk,
            **extra,
        )

    torch.manual_seed(SEED + 5)
    full_x = torch.randn(total, BATCH, HIDDEN, device=device, dtype=DTYPE)
    dist.broadcast(full_x, src=0)

    cp_mod = _make_gdn(cp_size, cp_rank, cp_group, "chunkwise").to(device=device, dtype=DTYPE)
    _randomize_and_broadcast(cp_mod)

    ref_out_full = None
    if rank == 0:
        ref_mod = _make_gdn(1, 0, cp1_group, "chunkwise").to(device=device, dtype=DTYPE)
        ref_mod.load_state_dict(cp_mod.state_dict())
        with torch.no_grad():
            ref_out_full = ref_mod(full_x, packed_seq_params=_psp(1, 0, cp1_group)).detach().clone()
        del ref_mod
    if ref_out_full is None:
        ref_out_full = torch.empty(total, BATCH, HIDDEN, device=device, dtype=DTYPE)
    dist.broadcast(ref_out_full, src=0)

    local_x = split_packed_to_cp_local(
        full_x, cu_seqlens_padded=cu, cp_size=cp_size, cp_rank=cp_rank, dim=0
    ).contiguous()
    with torch.no_grad():
        cp_out = cp_mod(local_x, packed_seq_params=_psp(cp_size, cp_rank, cp_group))

    expected_local = split_packed_to_cp_local(
        ref_out_full, cu_seqlens_padded=cu, cp_size=cp_size, cp_rank=cp_rank, dim=0
    )
    f_abs, f_rel, f_scale = _diff(cp_out.detach(), expected_local)
    if cp_rank == 0:
        print(
            f"GDN_CP_PACKED_PARITY mode=chunkwise cp={cp_size} lens={PACKED_LENS} "
            f"fwd[max_abs={f_abs:.3e} max_rel={f_rel:.3e} scale={f_scale:.3e}]",
            flush=True,
        )
    del cp_mod
    torch.cuda.empty_cache()
    dist.barrier()


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    cp_size = world
    cp_rank = rank
    cp_group = dist.new_group(list(range(world)))
    cp1_group = dist.new_group([0])

    if rank == 0:
        import megatron.lite.primitive.modules.gated_delta_net as g

        print(f"GDN_CP_ENV world={world} HAS_FLA={g._HAS_FLA}", flush=True)

    run_matrix(cp_size, cp_rank, cp_group, cp1_group, device, rank, world)
    run_packed_matrix(cp_size, cp_rank, cp_group, cp1_group, device, rank, world)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("GDN_CP_PARITY_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("GDN_CP_PARITY_ERROR", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
