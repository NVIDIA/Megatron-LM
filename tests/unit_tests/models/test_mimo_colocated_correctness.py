# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""TransformerBlock correctness test for colocated MIMO communication (9 checks x 3 iters).

Run:  uv run python -m torch.distributed.run --nproc_per_node=8 \
        -m pytest tests/unit_tests/models/test_mimo_colocated_correctness.py -v -s
"""
import os

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.mimo.comm.colocated_communicator import ColocatedBridgeCommunicator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.pipeline_parallel.test_bridge_communicator import (
    _avg_params,
    _get_pg_collection_from_grid,
    _shard_and_copy_,
)
from tests.unit_tests.test_utilities import Utils

_active_grids: list = []

H, NHEADS, SEQ, GBS = 1024, 8, 8, 8


def _make_block(num_layers, dtype, pg):
    torch.manual_seed(12345)
    model_parallel_cuda_manual_seed(
        123, tp_rank=pg.tp.rank(), ep_rank=dist.get_rank(), etp_rank=dist.get_rank()
    )
    cfg = TransformerConfig(
        num_layers=num_layers,
        hidden_size=H,
        num_attention_heads=NHEADS,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        context_parallel_size=pg.cp.size(),
    )
    block = (
        TransformerBlock(cfg, get_gpt_layer_with_transformer_engine_spec(), pg_collection=pg)
        .cuda()
        .to(dtype)
    )
    with torch.no_grad():
        for m in block.modules():
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.zero_()
    return block


def _grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    g = HyperCommGrid(
        shape=[tp, cp, pp, dp],
        dim_names=["tp", "cp", "pp", "dp"],
        rank_offset=offset,
        backend="nccl",
    )
    for d in ["tp", "cp", "pp", "dp"]:
        g.create_pg([d])
    _active_grids.append(g)
    return g


def _cmp_wt(ref, tp_blk, tp_sz, tp_rk, atol, tag):
    for n, tp_p in tp_blk.state_dict().items():
        rp = ref.state_dict()[n]
        if not (torch.is_tensor(rp) and torch.is_tensor(tp_p)):
            continue
        if rp.shape == tp_p.shape:
            exp = rp
        elif tp_p.shape[0] * tp_sz == rp.shape[0]:
            exp = rp.chunk(tp_sz, dim=0)[tp_rk]
        elif rp.ndim > 1 and tp_p.shape[1] * tp_sz == rp.shape[1]:
            exp = rp.chunk(tp_sz, dim=1)[tp_rk]
        else:
            continue
        torch.testing.assert_close(tp_p, exp, atol=atol, rtol=0, msg=f"{tag} {n}")


def _cmp_grad(ref, tp_blk, tp_sz, tp_rk, atol, tag):
    for (rn, rp), (_, tp_p) in zip(ref.named_parameters(), tp_blk.named_parameters()):
        if rp.grad is None or tp_p.grad is None:
            continue
        if rp.grad.shape == tp_p.grad.shape:
            exp = rp.grad
        elif tp_p.grad.shape[0] * tp_sz == rp.grad.shape[0]:
            exp = rp.grad.chunk(tp_sz, dim=0)[tp_rk]
        elif rp.grad.ndim > 1 and tp_p.grad.shape[1] * tp_sz == rp.grad.shape[1]:
            exp = rp.grad.chunk(tp_sz, dim=1)[tp_rk]
        else:
            continue
        torch.testing.assert_close(tp_p.grad, exp, atol=atol, rtol=0, msg=f"{tag} {rn}")


class TestColocatedCorrectness:
    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def teardown_method(self):
        torch.use_deterministic_algorithms(False)
        for g in _active_grids:
            g.destroy()
        _active_grids.clear()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"), reason="Requires PyTorch 2.3+"
    )
    @pytest.mark.parametrize(
        "enc_tp,enc_dp,llm_tp,llm_dp",
        [(2, 4, 4, 2), (4, 2, 2, 4), (4, 2, 4, 2)],
        ids=["fan_in", "fan_out", "equal"],
    )
    def test_correctness(self, enc_tp, enc_dp, llm_tp, llm_dp):
        for k, v in {
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "NVTE_FLASH_ATTN": "0",
            "NVTE_FUSED_ATTN": "0",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }.items():
            os.environ[k] = v
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rank, dtype, lr = dist.get_rank(), torch.float32, 1e-6
        Utils.initialize_model_parallel(1, create_gloo_process_groups=False)

        ref_g = _grid(tp=1, dp=8)
        enc_g = _grid(tp=enc_tp, dp=enc_dp)
        llm_g = _grid(tp=llm_tp, dp=llm_dp)
        ref_pg, enc_pg, llm_pg = [_get_pg_collection_from_grid(g) for g in (ref_g, enc_g, llm_g)]

        ref_enc = _make_block(1, dtype, ref_pg)
        _avg_params(ref_enc, ref_g.get_pg("dp"))
        ref_llm = _make_block(2, dtype, ref_pg)
        _avg_params(ref_llm, ref_g.get_pg("dp"))
        col_enc = _make_block(1, dtype, enc_pg)
        _shard_and_copy_(ref_enc, col_enc, enc_tp, enc_pg.tp.rank())
        col_llm = _make_block(2, dtype, llm_pg)
        _shard_and_copy_(ref_llm, col_llm, llm_tp, llm_pg.tp.rank())

        comm = ColocatedBridgeCommunicator(
            src_grid=enc_g,
            dest_grid=llm_g,
            src_module_name="encoder",
            dest_module_name="llm",
            dim_mapping={'s': 0, 'b': 1, 'h': 2},
        )
        ref_opt = torch.optim.SGD(list(ref_enc.parameters()) + list(ref_llm.parameters()), lr=lr)
        col_opt = torch.optim.SGD(list(col_enc.parameters()) + list(col_llm.parameters()), lr=lr)
        e_di, l_di = enc_g.get_pg("dp").rank(), llm_g.get_pg("dp").rank()
        be, bl = GBS // enc_dp, GBS // llm_dp
        dist.barrier()

        for it in range(3):
            t = f"[iter={it} rank={rank}]"
            torch.manual_seed(100 + it)
            torch.cuda.manual_seed(100 + it)
            gi = torch.randn(SEQ, GBS, H, device="cuda", dtype=dtype)

            # Reference
            ri = gi.clone().detach().requires_grad_(True)
            reo = ref_enc(hidden_states=ri, attention_mask=None)
            rlo = ref_llm(hidden_states=reo, attention_mask=None)
            rlo.sum().backward()

            # Colocated
            ci = gi[:, e_di * be : (e_di + 1) * be, :].clone().detach().requires_grad_(True)
            eo = col_enc(hidden_states=ci, attention_mask=None)
            co = comm.communicate(eo)
            lo = col_llm(hidden_states=co, attention_mask=None)
            cl = lo.sum()
            cl.backward()

            # 1-Enc activations  2-Post-comm  3-LLM activations  4-Loss  5-Input grads
            torch.testing.assert_close(
                eo,
                reo[:, e_di * be : (e_di + 1) * be, :].detach(),
                atol=5e-4,
                rtol=0,
                msg=f"{t} enc_out",
            )
            torch.testing.assert_close(
                co.detach(),
                reo[:, l_di * bl : (l_di + 1) * bl, :].detach(),
                atol=5e-4,
                rtol=0,
                msg=f"{t} post_comm",
            )
            rls = rlo[:, l_di * bl : (l_di + 1) * bl, :]
            torch.testing.assert_close(
                lo.detach(), rls.detach(), atol=5e-3, rtol=0, msg=f"{t} llm_out"
            )
            torch.testing.assert_close(
                cl.detach(), rls.detach().sum(), atol=1e-1, rtol=0, msg=f"{t} loss"
            )
            # Tight check: catches a class of bugs where fan-out backward drops
            # cross-rank gradient contributions (zero-pad-without-gather regression).
            # Observed numerical noise floor is ~2e-7; the old zero-pad bug
            # produced ~1e-5 disagreement on samples handled by sibling llm_dp ranks.
            torch.testing.assert_close(
                ci.grad,
                ri.grad[:, e_di * be : (e_di + 1) * be, :],
                atol=5e-6,
                rtol=1e-3,
                msg=f"{t} input_grad",
            )

            # 6-Enc param grads  7-LLM param grads (after DP all-reduce SUM)
            for p in col_enc.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=enc_g.get_pg("dp"))
            _cmp_grad(ref_enc, col_enc, enc_tp, enc_pg.tp.rank(), 5e-3, f"{t} enc_pgrad")
            for p in col_llm.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=llm_g.get_pg("dp"))
            _cmp_grad(ref_llm, col_llm, llm_tp, llm_pg.tp.rank(), 5e-3, f"{t} llm_pgrad")

            # Optimizer step
            ref_opt.step()
            ref_opt.zero_grad()
            col_opt.step()
            col_opt.zero_grad()

            # 8-Enc weights  9-LLM weights
            _cmp_wt(ref_enc, col_enc, enc_tp, enc_pg.tp.rank(), 1e-5, f"{t} enc_wt")
            _cmp_wt(ref_llm, col_llm, llm_tp, llm_pg.tp.rank(), 1e-5, f"{t} llm_wt")
            dist.barrier()
