# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Correctness test for colocated MIMO communication under mean-CE + DDP.

Uses real Megatron ``DistributedDataParallel`` wrapping, mean cross-entropy
loss (the realistic training target — the implicit ``1/local_B_llm`` factor
in the backward chain is exactly the case that exposes encoder-DDP scaling
bugs), and compares encoder param grads against a single-GPU reference
running the full batch. Each rank runs the reference independently on
identical (DP-averaged) weights so the reference is the same on every rank.

Run:  uv run python -m torch.distributed.run --nproc_per_node=8 \\
        -m pytest tests/unit_tests/models/test_mimo_colocated_correctness.py -v -s
"""
import os
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from packaging import version

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.mimo.comm.colocated_communicator import ColocatedBridgeCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.pipeline_parallel.test_bridge_communicator import (
    _avg_params,
    _shard_and_copy_,
)
from tests.unit_tests.test_utilities import Utils

_active_grids: list = []
_active_comms: list = []

H, NHEADS, SEQ, GBS, VOCAB = 256, 8, 8, 8, 128


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
    return block, cfg


def _grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    # Include ep/expt_dp so we can build the full ProcessGroupCollection that
    # Megatron DDP expects. All non-DP/TP dims are size 1 in this test.
    g = HyperCommGrid(
        shape=[tp, cp, pp, dp, 1, 1],
        dim_names=["tp", "cp", "pp", "dp", "ep", "expt_dp"],
        rank_offset=offset,
        backend="nccl",
    )
    for d in ["tp", "cp", "pp", "dp", "ep", "expt_dp"]:
        g.create_pg([d])
    g.create_pg(["dp", "cp"])
    _active_grids.append(g)
    return g


def _pg_from_grid(grid):
    pg = ProcessGroupCollection()
    pg.tp = grid.get_pg("tp")
    pg.cp = grid.get_pg("cp")
    pg.pp = grid.get_pg("pp")
    pg.dp = grid.get_pg("dp")
    pg.dp_cp = grid.get_pg(["dp", "cp"])
    pg.ep = grid.get_pg("ep")
    pg.expt_dp = grid.get_pg("expt_dp")
    return pg


def _cmp_main_grad(ref, parallel, tp_sz, tp_rk, atol, tag):
    """Compare parallel ``param.main_grad`` (post-DDP-sync) to ref ``param.grad``.

    Both modules have matching parameter names; shapes differ by TP sharding.
    """
    ref_params = dict(ref.named_parameters())
    for name, tp_p in parallel.named_parameters():
        if name not in ref_params:
            continue
        rp = ref_params[name]
        if rp.grad is None or tp_p.main_grad is None:
            continue
        tp_grad = tp_p.main_grad.to(rp.grad.dtype)
        if rp.grad.shape == tp_grad.shape:
            exp = rp.grad
        elif tp_grad.shape[0] * tp_sz == rp.grad.shape[0]:
            exp = rp.grad.chunk(tp_sz, dim=0)[tp_rk]
        elif rp.grad.ndim > 1 and tp_grad.shape[1] * tp_sz == rp.grad.shape[1]:
            exp = rp.grad.chunk(tp_sz, dim=1)[tp_rk]
        else:
            continue
        torch.testing.assert_close(tp_grad, exp, atol=atol, rtol=0, msg=f"{tag} {name}")


class TestColocatedMeanCECorrectness:
    """End-to-end mean-CE + Megatron DDP gradient correctness.

    Catches the encoder DDP mis-scaling that toy sum-loss tests hide: mean-CE
    divides per-sample grads by ``local_B_llm``; the encoder's DDP must divide
    by ``llm_dp`` (not ``enc_dp``) so the combined divisor is the full batch.
    """

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def teardown_method(self):
        torch.use_deterministic_algorithms(False)
        for c in _active_comms:
            c.destroy()
        _active_comms.clear()
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
    def test_mean_ce_encoder_grads_match_reference(
        self, enc_tp, enc_dp, llm_tp, llm_dp
    ):
        # Determinism — the test asserts bitwise-close grads, so we need the
        # attention kernels and TE paths to be reproducible.
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

        dtype = torch.float32
        Utils.initialize_model_parallel(1, create_gloo_process_groups=False)

        # ── Reference (each rank runs identical TP=1 model on full batch) ──
        ref_g = _grid(tp=1, dp=8)
        ref_pg = _pg_from_grid(ref_g)
        ref_enc, _ = _make_block(1, dtype, ref_pg)
        _avg_params(ref_enc, ref_g.get_pg("dp"))
        ref_llm, _ = _make_block(2, dtype, ref_pg)
        _avg_params(ref_llm, ref_g.get_pg("dp"))
        # Classifier head — keep TP=1 in all paths so the reference grad is
        # directly comparable without head-side sharding.
        torch.manual_seed(7777)
        ref_head = torch.nn.Linear(H, VOCAB, bias=False).cuda().to(dtype)
        _avg_params(ref_head, ref_g.get_pg("dp"))

        # ── Parallel encoder and LLM, same initial weights as reference ──
        enc_g = _grid(tp=enc_tp, dp=enc_dp)
        llm_g = _grid(tp=llm_tp, dp=llm_dp)
        enc_pg = _pg_from_grid(enc_g)
        llm_pg = _pg_from_grid(llm_g)
        col_enc, enc_cfg = _make_block(1, dtype, enc_pg)
        _shard_and_copy_(ref_enc, col_enc, enc_tp, enc_pg.tp.rank())
        col_llm, llm_cfg = _make_block(2, dtype, llm_pg)
        _shard_and_copy_(ref_llm, col_llm, llm_tp, llm_pg.tp.rank())
        # Head on parallel path matches the reference head exactly.
        col_head = torch.nn.Linear(H, VOCAB, bias=False).cuda().to(dtype)
        col_head.weight.data.copy_(ref_head.weight.data)

        comm = ColocatedBridgeCommunicator(
            src_grid=enc_g,
            dest_grid=llm_g,
            src_module_name="encoder",
            dest_module_name="llm",
            dim_mapping={'s': 0, 'b': 1, 'h': 2},
        )
        _active_comms.append(comm)

        # ── Wrap parallel encoder and LLM with Megatron DDP ──
        # overlap_grad_reduce=False + use_distributed_optimizer=False → grads
        # land in each param's ``main_grad`` after ``finish_grad_sync()``
        # without reduce-scatter sharding, making per-param comparison easy.
        base_ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=False, use_distributed_optimizer=False, bucket_size=10_000_000
        )
        enc_ddp_config = replace(
            base_ddp_config, gradient_reduce_div_factor=llm_g.get_pg("dp").size()
        )
        col_enc_ddp = DistributedDataParallel(
            config=enc_cfg,
            ddp_config=enc_ddp_config,
            module=col_enc,
            pg_collection=enc_pg,
        )
        col_llm_ddp = DistributedDataParallel(
            config=llm_cfg,
            ddp_config=base_ddp_config,
            module=col_llm,
            pg_collection=llm_pg,
        )

        # ── Shared input / labels, identical on every rank ──
        torch.manual_seed(42)
        full_input = torch.randn(SEQ, GBS, H, device='cuda', dtype=dtype)
        full_labels = torch.randint(0, VOCAB, (SEQ, GBS), device='cuda')

        # ── Reference mean-CE forward + backward on FULL batch ──
        ref_input = full_input.clone().detach().requires_grad_(True)
        ref_enc_out = ref_enc(hidden_states=ref_input, attention_mask=None)
        ref_llm_out = ref_llm(hidden_states=ref_enc_out, attention_mask=None)
        ref_logits = ref_head(ref_llm_out)  # [S, B, VOCAB]
        ref_loss = F.cross_entropy(
            ref_logits.reshape(-1, VOCAB), full_labels.reshape(-1), reduction='mean'
        )
        ref_loss.backward()

        # ── Parallel forward + backward on DP slice with mean-CE ──
        enc_dp_idx = enc_g.get_pg("dp").rank()
        llm_dp_idx = llm_g.get_pg("dp").rank()
        b_enc, b_llm = GBS // enc_dp, GBS // llm_dp

        col_enc_ddp.zero_grad_buffer()
        col_llm_ddp.zero_grad_buffer()
        col_head.zero_grad()

        col_input = (
            full_input[:, enc_dp_idx * b_enc : (enc_dp_idx + 1) * b_enc, :]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        col_enc_out = col_enc_ddp(hidden_states=col_input, attention_mask=None)
        col_bridged = comm.communicate(col_enc_out)
        col_llm_out = col_llm_ddp(hidden_states=col_bridged, attention_mask=None)
        col_logits = col_head(col_llm_out)
        col_labels_slice = full_labels[:, llm_dp_idx * b_llm : (llm_dp_idx + 1) * b_llm]
        col_loss = F.cross_entropy(
            col_logits.reshape(-1, VOCAB), col_labels_slice.reshape(-1), reduction='mean'
        )
        col_loss.backward()

        # ── Trigger DDP all-reduce so ``main_grad`` holds the synced grad ──
        col_enc_ddp.finish_grad_sync()
        col_llm_ddp.finish_grad_sync()

        # ── Encoder grad check (the test that actually fails without the
        # gradient_reduce_div_factor fix) ──
        _cmp_main_grad(ref_enc, col_enc, enc_tp, enc_pg.tp.rank(), 5e-4, "enc_pgrad")

        # ── LLM grad check (sanity: the LLM path should already be correct) ──
        _cmp_main_grad(ref_llm, col_llm, llm_tp, llm_pg.tp.rank(), 5e-4, "llm_pgrad")

        dist.barrier()
