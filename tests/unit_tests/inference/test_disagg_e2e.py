# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end functional test of the prefill->decode KV transfer.

Drives the KV reshard + transport + import directly
(send_request_kv_resharded + post_recv/finish) on real GPUs over the NCCL
transport, and asserts the decode side reconstructs the exact global KV:
prefill TP2 {0,1} -> decode TP1 {2} (3 procs, >=3 GPUs), exercising the
hetero TP2->TP1 head merge. A second variant adds hybrid Mamba boundary
snapshots and asserts the decode assembles the exact global snapshot state
through the same hetero remap.

The LLM forward is stubbed (a fake context whose export returns this rank's
shard of a known global KV/snapshot state and whose import records the
assembled tensors); everything else is the real transport.
"""

import os

import pytest
import torch

mp = torch.multiprocessing

L, H, BC, BS, HD = 4, 8, 2, 4, 6  # global model + block dims
PROMPT = list(range(BC * BS))  # block_count = BC
# Hybrid variant: global Mamba dims + one snapshot per block.
NHEADS, HEADDIM, DSTATE, NGROUPS, DCONV, ML = 8, 4, 2, 2, 3, 4
D_INNER = NHEADS * HEADDIM
CONV_DIM = D_INNER + 2 * NGROUPS * DSTATE
SNAP_HASHES = [111, 222]


def _global_kv_staging():
    """Global KV in export/import staging layout [BC,2,L,BS,H,HD],
    value = layer*100 + head (deterministic, identical on all ranks)."""
    g = torch.zeros(BC, 2, L, BS, H, HD)
    for l in range(L):
        for h in range(H):
            g[:, :, l, :, h, :] = l * 100 + h
    return g


def _global_snapshots():
    """Global snapshot staging: conv (n, ML, CONV_DIM, DCONV) and ssm
    (n, ML, NHEADS, HEADDIM, DSTATE), distinct value per element."""
    n = len(SNAP_HASHES)
    conv = torch.arange(n * ML * CONV_DIM * DCONV, dtype=torch.float32).reshape(
        n, ML, CONV_DIM, DCONV
    )
    ssm = (
        torch.arange(n * ML * NHEADS * HEADDIM * DSTATE, dtype=torch.float32).reshape(
            n, ML, NHEADS, HEADDIM, DSTATE
        )
        + 100_000.0
    )
    return conv, ssm


def _shard_snapshots(conv_g, ssm_g, mlay):
    """Shard the global snapshot staging to one rank ([x|B|C] conv bands,
    head-sharded ssm)."""
    s, e = mlay.layer_range()
    r, tp = mlay.tp_rank, mlay.tp_size
    di_l = D_INNER // tp
    g_l = (NGROUPS // tp) * DSTATE
    gsz = NGROUPS * DSTATE
    x = conv_g[:, s:e, 0:D_INNER][:, :, r * di_l : (r + 1) * di_l]
    b = conv_g[:, s:e, D_INNER : D_INNER + gsz][:, :, r * g_l : (r + 1) * g_l]
    c = conv_g[:, s:e, D_INNER + gsz : D_INNER + 2 * gsz][:, :, r * g_l : (r + 1) * g_l]
    conv_l = torch.cat([x, b, c], dim=2).contiguous()
    nh_l = NHEADS // tp
    ssm_l = ssm_g[:, s:e, r * nh_l : (r + 1) * nh_l].contiguous()
    return conv_l, ssm_l


class _FakeSlotAllocator:
    """Just the pool tensors the decode recv path reads shapes/dtypes off."""

    def __init__(self, mlay, device):
        self.conv_states = torch.zeros(
            mlay.num_layers, 8, mlay.conv_dim_local, DCONV, device=device
        )
        self.ssm_states = torch.zeros(
            mlay.num_layers, 8, mlay.nheads_local, HEADDIM, DSTATE, device=device
        )


class _FakeCtx:
    def __init__(self, layout, device="cpu", mamba_layout=None):
        self.cache_mla_latent = False
        self.is_hybrid_model = mamba_layout is not None
        self.block_size_tokens = BS
        self._layout = layout
        self._mamba_layout = mamba_layout
        self._device = device
        h0, h1 = layout.head_range()
        # memory_buffer only used (on decode) for derive_decode_schema shape
        self.memory_buffer = torch.zeros(2, L, BC, BS, h1 - h0, HD, device=device)
        self.imported = None
        self.imported_snapshots = None
        self._g = _global_kv_staging().to(device)
        self.mamba_slot_allocator = (
            _FakeSlotAllocator(mamba_layout, device) if mamba_layout is not None else None
        )

    def export_request_kv(self, request_id):
        h0, h1 = self._layout.head_range()
        snapshots = None
        if self._mamba_layout is not None:
            conv_g, ssm_g = _global_snapshots()
            conv_l, ssm_l = _shard_snapshots(
                conv_g.to(self._device), ssm_g.to(self._device), self._mamba_layout
            )
            snapshots = {
                "block_hashes": list(SNAP_HASHES),
                "conv_states_tensor": conv_l,
                "ssm_states_tensor": ssm_l,
            }
        return {
            "staging_tensor": self._g[:, :, :, :, h0:h1, :].clone(),
            "mamba_snapshots": snapshots,
        }

    def import_request_kv(self, staging, block_hashes, mamba_snapshots=None):
        self.imported = staging
        self.imported_snapshots = mamba_snapshots
        return {"block_ids": list(range(staging.shape[0])), "block_hashes": block_hashes}


class _FakeEng:
    def __init__(self, layout, device="cpu", mamba_layout=None):
        self.context = _FakeCtx(layout, device=device, mamba_layout=mamba_layout)


def _make_layout(rank, world):
    from megatron.core.inference.disaggregation.kv_reshard import KVShardLayout

    if world == 3:
        # hetero: prefill TP2 {0,1} -> decode TP1 {2} (exercises head merge)
        if rank in (0, 1):
            return ("prefill", "prefill", KVShardLayout(L, H, 2, rank, 1, 0, rank))
        return ("decode", "d0", KVShardLayout(L, H, 1, 0, 1, 0, rank))
    # world == 2: prefill TP1 {0} -> decode TP1 {1} (identity reshard; full path)
    if rank == 0:
        return ("prefill", "prefill", KVShardLayout(L, H, 1, 0, 1, 0, 0))
    return ("decode", "d0", KVShardLayout(L, H, 1, 0, 1, 0, 1))


def _make_mamba_layout(rank, world):
    from megatron.core.inference.disaggregation.mamba_reshard import (
        MambaShardLayout,
        MambaStateDims,
    )

    dims = MambaStateDims(
        nheads=NHEADS, headdim=HEADDIM, d_state=DSTATE, ngroups=NGROUPS, d_conv=DCONV
    )
    if rank in (0, 1):  # prefill TP2
        return MambaShardLayout(
            global_rank=rank, tp_size=2, tp_rank=rank, layer_start=0, num_layers=ML, dims=dims
        )
    return MambaShardLayout(  # decode TP1
        global_rank=rank, tp_size=1, tp_rank=0, layer_start=0, num_layers=ML, dims=dims
    )


def _worker(rank, world, port, q, hybrid=False):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        from megatron.core.inference.disaggregation import kv_transfer_push as H
        from megatron.core.inference.disaggregation.transfer_backends.nccl import (
            NcclTransportBackend,
        )

        role, _replica, layout = _make_layout(rank, world)
        # The full prefill/decode layout lists (what a handshake would exchange,
        # made explicit; _make_layout is deterministic per rank).
        all_layouts = [_make_layout(r, world) for r in range(world)]
        src_layouts = [lay for (rl, _, lay) in all_layouts if rl == "prefill"]
        dst_layouts = [lay for (rl, _, lay) in all_layouts if rl == "decode"]
        mamba_layout = _make_mamba_layout(rank, world) if hybrid else None
        src_mamba = [_make_mamba_layout(r, world) for r in (0, 1)] if hybrid else []
        dst_mamba = [_make_mamba_layout(2, world)] if hybrid else []

        backend = NcclTransportBackend()
        backend.init()
        eng = _FakeEng(layout, device=device, mamba_layout=mamba_layout)

        if role == "prefill":
            h = H.send_request_kv_resharded(
                layout,
                src_layouts,
                dst_layouts,
                backend=backend,
                payload=eng.context.export_request_kv(7),
                src_mamba_layouts=src_mamba,
                dst_mamba_layouts=dst_mamba,
            )
            h.wait()
            q.put((f"prefill{rank}", "prefill"))
        else:
            handoff = {"snapshot_hashes": list(SNAP_HASHES)} if hybrid else None
            recv = H.post_recv_request_kv_resharded(
                eng,
                layout,
                src_layouts,
                dst_layouts,
                PROMPT,
                backend=backend,
                handoff=handoff,
                src_mamba_layouts=src_mamba,
                dst_mamba_layouts=dst_mamba,
            )
            res = recv.finish(eng) if recv is not None else None
            imported = eng.context.imported
            expected = _global_kv_staging().to(imported.device)
            ok = res is not None and torch.equal(imported, expected)
            if hybrid:
                snaps = eng.context.imported_snapshots
                conv_g, ssm_g = _global_snapshots()
                want_conv, want_ssm = _shard_snapshots(
                    conv_g.to(device), ssm_g.to(device), mamba_layout
                )
                ok = ok and snaps is not None and snaps["block_hashes"] == SNAP_HASHES
                ok = ok and torch.equal(snaps["conv_states_tensor"], want_conv)
                ok = ok and torch.equal(snaps["ssm_states_tensor"], want_ssm)
            q.put(("decode", (7, bool(ok))))
        # Rendezvous before any rank tears down the group (rank 0 hosts the
        # store); reached only on the success path.
        dist.barrier()
    except Exception:
        import traceback

        q.put((f"rank{rank}-ERROR", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_e2e(port, world=3, hybrid=False):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, port, q, hybrid)) for r in range(world)]
    for p in procs:
        p.start()
    out = {}
    for _ in range(world):
        k, v = q.get(timeout=180)
        out[k] = v
    for p in procs:
        p.join(timeout=60)
    return out


def _assert_ok(out):
    errs = {k: v for k, v in out.items() if "ERROR" in k}
    assert not errs, errs
    prefills = {k: v for k, v in out.items() if k.startswith("prefill")}
    assert prefills and all(v == "prefill" for v in prefills.values()), out
    assert out["decode"] == (7, True), out  # request 7, exact global KV reconstructed


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 3),
    reason="requires >=3 CUDA devices (prefill TP2 {0,1} + decode TP1 {2})",
)
def test_disagg_e2e_tp2_to_tp1_nccl():
    _assert_ok(_run_e2e(29601))


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 3),
    reason="requires >=3 CUDA devices (prefill TP2 {0,1} + decode TP1 {2})",
)
def test_disagg_e2e_tp2_to_tp1_nccl_hybrid_snapshots():
    _assert_ok(_run_e2e(29611, hybrid=True))
