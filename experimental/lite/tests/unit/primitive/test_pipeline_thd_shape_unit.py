# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit tests for THD dynamic-batch P2P shapes in the pipeline layer.

Under THD each micro-batch packs a different token count, so the inter-stage recv
buffer is a different shape per mb; a fixed buffer sized from the first mb truncates
later, larger ones (size mismatch -> abort/hang). Megatron's dynamic shape exchange
sizes the recv buffer from the exact ``size()`` the sender puts on the wire. Two
tiers: ``_1f1b_schedule`` (recv<->mb off-by-one guard at every stage position) and
a real 2-process gloo PP2 proof (NEW sizes-from-wire green; OLD fixed-buffer red).
"""

from __future__ import annotations

import multiprocessing
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.primitive.parallel import pipeline as pl


def _make_ps(pp_size: int, pp_rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        pp_size=pp_size, pp_rank=pp_rank,
        pp_is_first=(pp_rank == 0), pp_is_last=(pp_rank == pp_size - 1),
        pp_prev_rank=pp_rank - 1, pp_next_rank=pp_rank + 1, pp_group=None,
    )


# --- _1f1b_schedule: recv<->microbatch mapping under variable shapes ---
class _MockModel(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.weight = torch.nn.Parameter(torch.ones(hidden, hidden))
        self._input_tensor = None

    def set_input_tensor(self, t):
        self._input_tensor = t


def _run_schedule(pp_size, pp_rank, seq_lens, hidden=8):
    """Run the real _1f1b_schedule for one rank with _send_recv_pipeline mocked to
    play the peer (recv tensor of the peer-sent shape, in transfer order); records
    recv shapes to prove each recv maps to the right mb."""
    ps = _make_ps(pp_size, pp_rank)
    num_mb = len(seq_lens)
    fwd_shapes = [(int(s), 1, hidden) for s in seq_lens]
    model = _MockModel(hidden)
    recorded_fwd, recorded_bwd = [], []
    fwd_q, bwd_q = list(fwd_shapes), list(fwd_shapes)  # both arrive in mb order

    def fake_srp(send_fwd, send_bwd, recv_fwd, recv_bwd, ps_, tensor_shape,
                 *, fwd_recv_buf=None, bwd_recv_buf=None, batch_p2p=True,
                 clone_recv=False, dynamic_shape=False):
        assert dynamic_shape, "1F1B schedule must use dynamic shape exchange"
        assert fwd_recv_buf is None and bwd_recv_buf is None, "dynamic recv must not pre-size"
        fwd_out = bwd_out = None
        if recv_fwd:
            shp = fwd_q.pop(0); recorded_fwd.append(shp)
            fwd_out = torch.ones(shp, requires_grad=True)
        if recv_bwd:
            shp = bwd_q.pop(0); recorded_bwd.append(shp)
            bwd_out = torch.ones(shp)
        return fwd_out, bwd_out

    def forward_step_fn(m, batch):
        s = int(batch["S"])
        if ps.pp_is_first:
            base = torch.ones(s, 1, hidden, requires_grad=True)
        else:
            inp = unwrap_model(m)._input_tensor
            assert inp is not None, "middle/last stage forwarded with a None input"
            assert tuple(inp.shape) == (s, 1, hidden), (
                f"input shape {tuple(inp.shape)} != {(s, 1, hidden)} for mb S={s}")
            base = inp
        hidden_t = base * m.weight.sum()
        out = {"hidden_states": hidden_t}
        if ps.pp_is_last:
            out["loss"] = hidden_t.float().sum()
        return out

    batches = [{"S": s} for s in seq_lens]
    orig_srp, pl._send_recv_pipeline = pl._send_recv_pipeline, fake_srp
    try:
        pl._1f1b_schedule(
            forward_step_fn, model, iter([(b, None) for b in batches]),
            num_mb, SimpleNamespace(num_microbatches=num_mb), ps, fwd_shapes[0])
    finally:
        pl._send_recv_pipeline = orig_srp
    return recorded_fwd, recorded_bwd, fwd_shapes


VARLEN = [5, 9, 3, 7]  # deliberately non-uniform, ascending & descending mix


@pytest.mark.parametrize("pp_rank", [1, 2])
def test_middle_stage_recv_shapes_match_each_microbatch(pp_rank):
    # PP4 interior stage: every fwd AND bwd recv sized for its own micro-batch.
    rf, rb, fs = _run_schedule(4, pp_rank, VARLEN)
    assert rf == fs and rb == fs, (rf, rb, fs)


def test_last_stage_recv_shapes_match_each_microbatch():
    rf, rb, fs = _run_schedule(4, 3, VARLEN)  # last: every fwd input, no bwd recv
    assert rf == fs and rb == [], (rf, rb, fs)


def test_first_stage_recv_shapes_match_each_microbatch():
    rf, rb, fs = _run_schedule(4, 0, VARLEN)  # first: no fwd input, one bwd grad/mb
    assert rf == [] and rb == fs, (rf, rb, fs)


def test_pp2_recv_shapes_match_each_microbatch():
    rf, rb, fs = _run_schedule(2, 0, VARLEN)  # first stage
    assert rf == [] and rb == fs, (rf, rb, fs)
    rf2, rb2, _ = _run_schedule(2, 1, VARLEN)  # last stage
    assert rf2 == fs and rb2 == [], (rf2, rb2)


# --- Tier-A: real gloo 2-process PP2, real _send_recv_pipeline, variable seq ---
# NEW sizes the recv buffer from the wire so the S=7 mb (> first S=5) round-trips
# bitwise; OLD's fixed S=5 buffer overflows on S=7 and gloo aborts the receiver
# (spawn failure = red). Direct isend/irecv (gloo has no batch_isend_irecv); fp32.
_PP_H = 8
_PP_VARLEN = [5, 3, 7]  # 7 > first(5) is the overflow case


def _pp_hidden(mb_idx: int, seqlen: int) -> torch.Tensor:  # deterministic [S,1,H]
    gen = torch.Generator().manual_seed(20260715 + mb_idx)
    return torch.randn(seqlen, 1, _PP_H, generator=gen)


def _dynamic_shape_worker(rank, world, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(world))
    pl._PIPELINE_TENSOR_DTYPE = torch.float32  # gloo-safe; sizing is dtype-agnostic
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        ps = _make_ps(2, rank)
        detail = []
        for i, s in enumerate(_PP_VARLEN):
            if rank == 0:
                pl._send_recv_pipeline(_pp_hidden(i, s), None, False, False, ps,
                                       (1, 1, _PP_H), batch_p2p=False, dynamic_shape=True)
            else:
                fwd_buf, _ = pl._send_recv_pipeline(None, None, True, False, ps,
                                                    (1, 1, _PP_H), batch_p2p=False, dynamic_shape=True)
                expect = _pp_hidden(i, s)
                detail.append((s, tuple(fwd_buf.shape) == (s, 1, _PP_H),
                               bool(torch.equal(fwd_buf.detach().float(), expect.float()))))
        results.append((rank, detail))
    finally:
        dist.destroy_process_group()


def _fixed_buffer_worker(rank, world, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(world))
    pl._PIPELINE_TENSOR_DTYPE = torch.float32
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        ps = _make_ps(2, rank)
        first_len = _PP_VARLEN[0]  # fixed recv buffer sized from the FIRST mb (S=5)
        if rank == 0:
            pl._send_recv_pipeline(_pp_hidden(2, _PP_VARLEN[2]), None, False, False, ps,
                                   (first_len, 1, _PP_H), batch_p2p=False, dynamic_shape=False)
        else:
            fixed_buf = torch.empty(first_len, 1, _PP_H, dtype=torch.float32)  # only fits S=5
            pl._send_recv_pipeline(None, None, True, False, ps, (first_len, 1, _PP_H),
                                   fwd_recv_buf=fixed_buf, batch_p2p=False, dynamic_shape=False)
        results.append((rank, "no_abort"))  # reaching here without abort is the bug
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_dynamic_shape_variable_len_recv_gloo():  # NEW: recv sized from wire, 0 mismatch
    import torch.multiprocessing as mp
    results = multiprocessing.Manager().list()
    mp.spawn(_dynamic_shape_worker, args=(2, 29663, results), nprocs=2, join=True)
    by_rank = dict(results)
    assert set(by_rank) == {0, 1}, f"missing ranks: {list(by_rank)}"
    recv_detail = by_rank[1]  # last stage recorded every fwd recv
    assert len(recv_detail) == len(_PP_VARLEN), recv_detail
    for s, shape_ok, bitwise_ok in recv_detail:
        assert shape_ok, f"S={s}: recv buffer not sized from the wire"
        assert bitwise_ok, f"S={s}: recv hidden not bitwise-equal to sent hidden"


@pytest.mark.distributed
def test_fixed_buffer_truncates_variable_len_gloo():  # OLD: fixed buffer overflows -> abort
    import torch.multiprocessing as mp
    results = multiprocessing.Manager().list()
    with pytest.raises(Exception):  # gloo aborts the receiver -> spawn failure
        mp.spawn(_fixed_buffer_worker, args=(2, 29664, results), nprocs=2, join=True)
