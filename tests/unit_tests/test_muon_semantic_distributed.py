# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Four-process CPU/Gloo correctness tests for physical TP/GTP projection layouts.

Run: python -m unittest tests.unit_tests.test_muon_semantic_distributed -v
"""

import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.muon_layout import MuonProjectionLayout
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon


def optimizer(p, pg=None, mode='blockwise'):
    return TensorParallelMuon(
        [p],
        lr=0.003,
        momentum=0.93,
        nesterov=True,
        weight_decay=0.1,
        split_qkv=True,
        split_qkv_per_head=True,
        fp32_matmul_prec='highest',
        adamw_betas=(0.8, 0.97),
        adamw_eps=1e-7,
        pg_collection=pg,
        tp_mode=mode,
    )


def worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo',
        init_method=f'file://{store}',
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    gtp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    tp_rank, grank = rank % 2, rank // 2
    pg = SimpleNamespace(
        tp=tp_groups[grank],
        expt_tp=tp_groups[grank],
        gtp_remat=gtp_groups[tp_rank],
        expt_gtp_remat=gtp_groups[tp_rank],
    )
    layouts = [
        # TP cuts a Muon matrix; GTP padding crosses the subsequent TP range.
        (MuonProjectionLayout((5, 3, 4, 5, 3, 4), (False, True, False, False, True, False)), 4),
        (
            MuonProjectionLayout.gdn(
                ['query', 'key', 'value', 'z', 'beta', 'alpha'], (4, 4, 6, 6, 2, 2), 2, 3
            ),
            8,
        ),
        (
            MuonProjectionLayout.gdn(
                ['query', 'key', 'value', 'z', 'f', 'b', 'w'], (4, 4, 6, 6, 4, 4, 6), 2, 3
            ),
            2,
        ),
        # Replicated MLA down projection is covered separately below.
    ]
    for index, (layout, padding) in enumerate(layouts):
        torch.manual_seed(700 + index + (tp_rank if layout.tp_local else 0))
        full = torch.randn(sum(layout.splits), 8)
        reference = torch.nn.Parameter(full.clone())
        reference.muon_layout = layout
        refopt = optimizer(reference)
        logical_tp_rows = full.shape[0] if layout.tp_local else full.shape[0] // 2
        local_rows = (logical_tp_rows + padding) // 2
        assert local_rows * 2 == logical_tp_rows + padding

        def shard(t):
            if not layout.tp_local:
                t = t.narrow(0, tp_rank * logical_tp_rows, logical_tp_rows)
            t = torch.nn.functional.pad(t, (0, 0, 0, padding))
            return t.narrow(0, grank * local_rows, local_rows).clone()

        p = torch.nn.Parameter(shard(full))
        p.muon_layout = layout
        p.tensor_model_parallel, p.partition_dim = True, 0
        p.is_gtp_weight_remat, p.pad_length = True, padding
        opt = optimizer(p, pg)
        for step in range(4):
            torch.manual_seed(900 + index * 10 + step + (100 * tp_rank if layout.tp_local else 0))
            grad = torch.randn_like(full)
            p.grad, reference.grad = shard(grad), grad.clone()
            opt.step()
            refopt.step()
            torch.testing.assert_close(p, shard(reference.detach()), atol=3e-6, rtol=3e-6)
            # Gate moments must never include padding or next-TP-rank control rows.
            for state_name in ('gate_exp_avg', 'gate_exp_avg_sq'):
                torch.testing.assert_close(
                    opt.state[p][state_name],
                    shard(refopt.state[reference][state_name]),
                    atol=2e-6,
                    rtol=2e-6,
                )
    # SwiGLU's physical [gate_local, up_local] layout is not contiguous global TP.
    torch.manual_seed(81)
    full = torch.randn(24, 8)
    for mode in ('blockwise', 'duplicated', 'distributed'):
        local = torch.cat([full[:12].chunk(2)[tp_rank], full[12:].chunk(2)[tp_rank]])
        p = torch.nn.Parameter(local.clone())
        p.muon_layout = MuonProjectionLayout.matrices((6, 6), tp_local=True, tp_partitioned=True)
        p.tensor_model_parallel, p.partition_dim = True, 0
        opt = optimizer(p, pg, mode)
        ref = torch.nn.Parameter(local.clone() if mode == 'blockwise' else full.clone())
        ref.muon_layout = MuonProjectionLayout.matrices((ref.shape[0] // 2,) * 2)
        refopt = optimizer(ref)
        for step in range(3):
            torch.manual_seed(101 + step)
            grad = torch.randn_like(full)
            local_grad = torch.cat([grad[:12].chunk(2)[tp_rank], grad[12:].chunk(2)[tp_rank]])
            p.grad = local_grad.clone()
            ref.grad = local_grad.clone() if mode == 'blockwise' else grad.clone()
            opt.step()
            refopt.step()
            expected = (
                ref
                if mode == 'blockwise'
                else torch.cat([ref[:12].chunk(2)[tp_rank], ref[12:].chunk(2)[tp_rank]])
            )
            torch.testing.assert_close(p, expected, atol=5e-6, rtol=5e-6)
    # Fused MLA stores [q_latent_rank, kv_combined_rank] on each TP rank.
    # Gathering these buffers produces interleaved Q/KV blocks, not global Q/KV.
    torch.manual_seed(135)
    full = torch.randn(24, 8)

    def fused_shard(t):
        return torch.cat([t[:8].chunk(2)[tp_rank], t[8:].chunk(2)[tp_rank]])

    for padding in (None, 2):

        def shard(t):
            local = fused_shard(t)
            if padding is None:
                return local
            local = torch.nn.functional.pad(local, (0, 0, 0, padding))
            return local.chunk(2, dim=0)[grank].clone()

        p = torch.nn.Parameter(shard(full))
        p.muon_layout = MuonProjectionLayout.matrices((8, 12, 4), tp_reorder_splits=(8, 16))
        p.tensor_model_parallel, p.partition_dim = True, 0
        if padding is not None:
            p.is_gtp_weight_remat, p.pad_length = True, padding
        opt = optimizer(p, pg)
        ref = torch.nn.Parameter(full.clone())
        ref.muon_layout = MuonProjectionLayout.matrices((8, 12, 4))
        refopt = optimizer(ref)
        for step in range(3):
            torch.manual_seed(155 + step)
            grad = torch.randn_like(full)
            p.grad, ref.grad = shard(grad), grad.clone()
            opt.step()
            refopt.step()
            torch.testing.assert_close(p, shard(ref), atol=5e-6, rtol=5e-6)
    # A replicated MLA down projection must not all-gather duplicate weights.
    p = torch.nn.Parameter(torch.randn(16, 8))
    p.muon_layout = MuonProjectionLayout.matrices((12, 4))
    p.tensor_model_parallel, p.partition_dim = False, -1
    opt = optimizer(p, pg)
    q = torch.nn.Parameter(p.detach().clone())
    q.muon_layout = p.muon_layout
    refopt = optimizer(q)
    p.grad = torch.randn_like(p)
    q.grad = p.grad.clone()
    opt.step()
    refopt.step()
    torch.testing.assert_close(p, q)
    dist.destroy_process_group()


@unittest.skipUnless(HAVE_EMERGING_OPTIMIZERS, 'emerging_optimizers is not installed')
class TestDistributedMuonSemantics(unittest.TestCase):
    def test_tp_gtp_padding_and_interleaved_layouts(self):
        if dist.is_initialized() or torch.cuda.is_initialized():
            self.skipTest("Run this standalone CPU/Gloo test with python -m unittest")
        with tempfile.TemporaryDirectory() as path:
            mp.start_processes(
                worker,
                args=(os.path.join(path, 'store'),),
                nprocs=4,
                join=True,
                start_method='fork',
            )


if __name__ == '__main__':
    unittest.main()
