# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end functional test of the prefill->decode KV transfer.

Drives the KV reshard + transport + import directly
(send_request_kv_resharded + post_recv/finish) on real GPUs over the NCCL
transport, and asserts the decode side reconstructs the exact global KV:
prefill TP2 {0,1} -> decode TP1 {2} (3 procs, >=3 GPUs), exercising the hetero
TP2->TP1 head merge.

The LLM forward is stubbed (a fake context whose export returns this
rank's head-shard of a known global KV and whose import records the
assembled tensor); everything else is the real transport.
"""

import os

import pytest
import torch

mp = torch.multiprocessing

L, H, BC, BS, HD = 4, 8, 2, 4, 6           # global model + block dims
PROMPT = list(range(BC * BS))               # block_count = BC


def _global_kv_staging():
    """Global KV in export/import staging layout [BC,2,L,BS,H,HD],
    value = layer*100 + head (deterministic, identical on all ranks)."""
    g = torch.zeros(BC, 2, L, BS, H, HD)
    for l in range(L):
        for h in range(H):
            g[:, :, l, :, h, :] = l * 100 + h
    return g


class _FakeCtx:
    def __init__(self, layout, device="cpu"):
        self.cache_mla_latent = False
        self.is_hybrid_model = False
        self.block_size_tokens = BS
        self._layout = layout
        self._device = device
        h0, h1 = layout.head_range()
        # memory_buffer only used (on decode) for derive_decode_schema shape
        self.memory_buffer = torch.zeros(2, L, BC, BS, h1 - h0, HD, device=device)
        self.imported = None
        self._g = _global_kv_staging().to(device)

    def export_request_kv(self, request_id):
        h0, h1 = self._layout.head_range()
        return {
            "layout": "std_attn_v1",
            "block_count": BC,
            "block_size_tokens": BS,
            "num_layers": L,
            "num_heads_per_partition": h1 - h0,
            "hidden_per_head": HD,
            "block_hashes": [],
            "staging_tensor": self._g[:, :, :, :, h0:h1, :].clone(),
        }

    def import_request_kv(self, payload):
        self.imported = payload["staging_tensor"]
        return {"block_ids": list(range(payload["block_count"])), "ok": True}


class _FakeEng:
    def __init__(self, layout, device="cpu"):
        self.context = _FakeCtx(layout, device=device)


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


def _worker(rank, world, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        from megatron.core.inference.disaggregation import kv_transfer as H
        from megatron.core.inference.disaggregation.transfer_backends.nccl import (
            NcclTransportBackend,
        )

        role, _replica, layout = _make_layout(rank, world)
        # The full prefill/decode layout lists (what a handshake would exchange,
        # made explicit -- _make_layout is deterministic per rank).
        all_layouts = [_make_layout(r, world) for r in range(world)]
        src_layouts = [lay for (rl, _, lay) in all_layouts if rl == "prefill"]
        dst_layouts = [lay for (rl, _, lay) in all_layouts if rl == "decode"]

        backend = NcclTransportBackend()
        backend.init()
        eng = _FakeEng(layout, device=device)

        if role == "prefill":
            h = H.send_request_kv_resharded(
                eng, 7, layout, src_layouts, dst_layouts, backend=backend,
            )
            if h is not None:
                h.wait()
            q.put((f"prefill{rank}", "prefill"))
        else:
            recv = H.post_recv_request_kv_resharded(
                eng, layout, src_layouts, dst_layouts, PROMPT, backend=backend,
            )
            res = recv.finish(eng) if recv is not None else None
            imported = eng.context.imported
            expected = _global_kv_staging().to(imported.device)
            ok = res is not None and torch.equal(imported, expected)
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


def _run_e2e(port, world=3):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, port, q)) for r in range(world)]
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
