# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for THD format with CUDA Graph support.

Padding helpers and dataclass round-trip (any GPU count, fast):
    torchrun --nproc_per_node 1 -m pytest -xvs \
        tests/unit_tests/transformer/test_thd_cuda_graph.py \
        -k "Pad or Decompose"

End-to-end no-graph vs graph bitwise loss/grad_norm match for
Moonlight-16B and Qwen3-8B with TP2_CP2_PP2_EP4_ETP1 + sequence packing
(requires 8 GPUs, slow ~5 min per run, 4 runs total):
    pytest -xvs tests/unit_tests/transformer/test_thd_cuda_graph.py::TestE2EBitwise

The E2E test directly subprocesses `torchrun pretrain_gpt.py` -- the same
command exercised by test_moonlight_qwen3_bitwise.sh -- with both
cuda_graph_impl=none and cuda_graph_impl=transformer_engine, then compares
the per-iteration loss / grad_norm lines. They must be exactly equal.
"""

import os
import re
import subprocess

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams, pad_thd_for_cuda_graph
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils

os.environ.setdefault('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


# =============================================================================
# Helpers (shared by the lightweight unit tests)
# =============================================================================

def _make_cu(seqlens, device="cuda"):
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    for i, s in enumerate(seqlens):
        cu[i + 1] = cu[i] + s
    return cu


def _make_psp(seqlens):
    cu = _make_cu(seqlens)
    return PackedSeqParams(
        qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu.clone(),
        cu_seqlens_q_padded=cu.clone(), cu_seqlens_kv_padded=cu.clone(),
        max_seqlen_q=max(seqlens), max_seqlen_kv=max(seqlens))


def _build_layer(H, nh, nkv, ffn, max_seqlen, max_num_seqs, tp=1, sp=False):
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    config = TransformerConfig(
        num_layers=1, hidden_size=H, num_attention_heads=nh,
        num_query_groups=nkv, ffn_hidden_size=ffn,
        max_seqlen_per_dp_cp_rank=max_seqlen,
        thd_cuda_graph_max_num_seqs=max_num_seqs,
        tensor_model_parallel_size=tp, sequence_parallel=sp, bf16=True)
    model_parallel_cuda_manual_seed(42)
    return TransformerLayer(
        config, get_gpt_layer_with_transformer_engine_spec().submodules,
        layer_number=1).cuda().bfloat16()


# =============================================================================
# 1. pad_thd_for_cuda_graph correctness
# =============================================================================

class TestPadThdForCudaGraph:

    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_shapes_and_data_preservation(self):
        """Shapes are static; original data intact; padding zero-filled."""
        seqlens, max_seqlen, max_num_seqs = [100, 50, 30], 256, 8
        total_T = sum(seqlens)
        tokens = torch.arange(total_T, device="cuda").unsqueeze(0).float()
        p_tok, p_lab, p_loss, p_pos, p_params, p_mask = pad_thd_for_cuda_graph(
            tokens, tokens.clone(), torch.ones(1, total_T, device="cuda"),
            torch.arange(total_T, device="cuda").unsqueeze(0),
            _make_psp(seqlens), max_seqlen, max_num_seqs)
        for t in (p_tok, p_lab, p_loss, p_pos):
            assert t.shape == (1, max_seqlen)
        for cu in (p_params.cu_seqlens_q, p_params.cu_seqlens_kv,
                   p_params.cu_seqlens_q_padded, p_params.cu_seqlens_kv_padded):
            assert cu.shape[0] == max_num_seqs + 1
        assert p_mask.shape == (1, max_seqlen) and p_mask.dtype == torch.bool
        assert torch.equal(p_tok[0, :total_T], tokens[0])
        assert (p_tok[0, total_T:] == 0).all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_padding_mask_boundary(self):
        """False at real positions, True at padding (MoE aux-loss contract)."""
        seqlens, total_T, max_seqlen = [60, 40], 100, 128
        _, _, _, _, _, m = pad_thd_for_cuda_graph(
            torch.ones(1, total_T, device="cuda"), None, None, None,
            _make_psp(seqlens), max_seqlen, 4)
        assert not m[0, :total_T].any() and m[0, total_T:].all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cu_seqlens_fill_value(self):
        """Padded entries repeat last cumulative sum (prevents OOB reads)."""
        seqlens, total_T = [50, 30], 80
        _, _, _, _, p, _ = pad_thd_for_cuda_graph(
            torch.ones(1, total_T, device="cuda"), None, None, None,
            _make_psp(seqlens), 128, 32)
        assert p.cu_seqlens_q[0] == 0 and p.cu_seqlens_q[2] == 80
        assert (p.cu_seqlens_q[3:] == 80).all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_none_inputs(self):
        """Non-pre_process PP: mask from cu_seqlens when all tensors None."""
        seqlens, total_T, max_seqlen = [50, 30], 80, 128
        _, _, _, _, _, mask = pad_thd_for_cuda_graph(
            None, None, None, None, _make_psp(seqlens), max_seqlen, 4)
        assert mask.shape == (1, max_seqlen)
        assert not mask[0, :total_T].any() and mask[0, total_T:].all()


# =============================================================================
# 2. PackedSeqParams decompose / reconstruct
# =============================================================================

class TestDecomposeReconstruct:

    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_round_trip(self):
        """Decompose then reconstruct preserves cu_seqlens values."""
        psp = _make_psp([100, 50, 30])
        orig = {k: getattr(psp, k).clone() for k in
                ('cu_seqlens_q', 'cu_seqlens_kv', 'cu_seqlens_q_padded', 'cu_seqlens_kv_padded')}
        layer = _build_layer(256, 4, 4, 1024, 128, 8)
        kw = {'packed_seq_params': psp, 'other': 'kept'}
        TransformerLayer._decompose_packed_seq_params_to_kwargs(kw)
        assert 'packed_seq_params' not in kw and 'cu_seqlens_q' in kw
        layer._reconstruct_packed_seq_params_from_kwargs(kw)
        r = kw['packed_seq_params']
        assert r.qkv_format == 'thd' and r.max_seqlen_q == 128
        for k, v in orig.items():
            assert torch.equal(getattr(r, k), v)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_noop_without_packed_seq_params(self):
        """No-ops on non-THD kwargs (SBHD path)."""
        layer = _build_layer(256, 4, 4, 1024, 128, 8)
        kw = {'hidden_states': torch.randn(10, 1, 256, device="cuda")}
        keys = set(kw.keys())
        TransformerLayer._decompose_packed_seq_params_to_kwargs(kw)
        assert set(kw.keys()) == keys
        layer._reconstruct_packed_seq_params_from_kwargs(kw)
        assert set(kw.keys()) == keys


# =============================================================================
# 3. E2E no-graph vs graph bitwise loss/grad_norm match
#    Subprocess-launches `torchrun pretrain_gpt.py` -- same recipe as
#    test_moonlight_qwen3_bitwise.sh -- and asserts the per-iteration
#    metric strings are byte-identical between the two runs.
# =============================================================================

# Common args shared across both models (matches test_moonlight_qwen3_bitwise.sh).
_MEGATRON_DIR = os.environ.get(
    'MEGATRON_DIR',
    '/lustre/fsw/coreai_devtech_all/haocheny/migrate_to_TE_0415/Megatron-LM')
_MOONLIGHT_LOAD = os.environ.get(
    'MOONLIGHT_CKPT',
    '/lustre/fsw/coreai_devtech_all/haocheny/mcore_models/Moonlight-16B-A3B-Instruct')

_SFT_JSON = (
    '{"mode":"distribution","type":"lognormal",'
    '"min_seq_len":128,"max_seq_len":2048,"mean_seq_len":1024,"lognormal_sigma":0.8}'
)

_TRAIN_ITERS = 5

_COMMON_ARGS = [
    "--seq-length", "2048", "--max-position-embeddings", "8192",
    "--micro-batch-size", "1", "--global-batch-size", "4",
    "--train-iters", str(_TRAIN_ITERS),
    "--lr", "1e-5", "--min-lr", "1e-6", "--lr-decay-style", "cosine",
    "--lr-warmup-iters", "1",
    "--weight-decay", "0.01", "--clip-grad", "1.0",
    "--seed", "1234", "--te-rng-tracker", "--bf16",
    "--tensor-model-parallel-size", "2", "--pipeline-model-parallel-size", "2",
    "--context-parallel-size", "2",
    "--swiglu", "--disable-bias-linear", "--sequence-parallel",
    "--sft", "--mock-data",
    "--tokenizer-type", "NullTokenizer",
    "--sft-mock-dataset-config-json", _SFT_JSON,
    "--sequence-packing-scheduler", "dp_balanced",
    "--max-seqlen-per-dp-cp-rank", "1024",
    "--calculate-per-token-loss",
    "--transformer-impl", "transformer_engine",
    "--attention-dropout", "0", "--hidden-dropout", "0",
    "--no-bias-swiglu-fusion", "--no-gradient-accumulation-fusion",
    "--no-save-optim", "--no-save-rng",
    "--save-interval", "999999", "--eval-interval", "999999", "--eval-iters", "1",
    "--log-interval", "1", "--no-check-for-nan-in-loss-and-grad", "--deterministic-mode",
    "--thd-cuda-graph-max-num-seqs", "32",
]

_MOONLIGHT_ARGS = _COMMON_ARGS + [
    "--num-layers", "27", "--hidden-size", "2048",
    "--ffn-hidden-size", "11264", "--num-attention-heads", "16",
    "--decoder-first-pipeline-num-layers", "13",
    "--decoder-last-pipeline-num-layers", "14",
    "--expert-model-parallel-size", "4", "--expert-tensor-parallel-size", "1",
    "--multi-latent-attention",
    "--kv-lora-rank", "512", "--qk-head-dim", "128",
    "--qk-pos-emb-head-dim", "64", "--v-head-dim", "128",
    "--num-experts", "64", "--moe-ffn-hidden-size", "1408",
    "--moe-router-topk", "6",
    "--moe-shared-expert-intermediate-size", "2816",
    "--moe-layer-freq", "([0]+[1]*26)",
    "--moe-token-dispatcher-type", "alltoall",
    "--moe-router-score-function", "sigmoid",
    "--moe-router-topk-scaling-factor", "2.446",
    "--moe-router-load-balancing-type", "aux_loss",
    "--moe-aux-loss-coeff", "0.001",
    "--normalization", "RMSNorm", "--norm-epsilon", "1e-5",
    "--rotary-base", "50000",
    "--vocab-size", "163840",
    "--load", _MOONLIGHT_LOAD,
    "--no-load-optim", "--no-load-rng",
]

_QWEN3_ARGS = _COMMON_ARGS + [
    "--num-layers", "36", "--hidden-size", "4096",
    "--ffn-hidden-size", "12288", "--num-attention-heads", "32",
    "--group-query-attention", "--num-query-groups", "8",
    "--max-position-embeddings", "40960",
    "--normalization", "RMSNorm", "--norm-epsilon", "1e-6",
    "--rotary-base", "1000000",
    "--untie-embeddings-and-output-weights",
    "--vocab-size", "151936",
    "--moe-token-dispatcher-type", "alltoall",
]


def _run_pretrain(model_args, cuda_graph_args, master_port):
    """Subprocess-launch `torchrun pretrain_gpt.py` once and capture stdout."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _MEGATRON_DIR + ":" + env.get("PYTHONPATH", "")
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["NCCL_ALGO"] = "^NVLS"
    # Strip any inherited torchrun env so this subprocess starts a fresh group.
    for k in list(env.keys()):
        if k.startswith(("TORCHELASTIC_", "MASTER_", "RANK", "LOCAL_RANK",
                         "WORLD_SIZE", "GROUP_RANK", "LOCAL_WORLD_SIZE")):
            env.pop(k, None)
    # Clear pytest-conftest env vars that disable TE attention backends
    # (set by tests/unit_tests/conftest.py::set_env). Pretrain needs at
    # least one of fused/flash attention to build the model.
    env.pop("NVTE_FLASH_ATTN", None)
    env.pop("NVTE_FUSED_ATTN", None)

    cmd = [
        "torchrun", "--nproc_per_node", "8", "--nnodes", "1",
        "--master_addr", "localhost", "--master_port", str(master_port),
        "pretrain_gpt.py",
    ] + model_args + cuda_graph_args

    result = subprocess.run(
        cmd, cwd=_MEGATRON_DIR, env=env, capture_output=True, text=True,
        timeout=900,
    )
    return result


_ITER_START_RE = re.compile(r"iteration\s+(\d+)/\s*\d+ \|")


def _extract_metrics(stdout):
    """Extract deterministic per-iteration fields from a training log.

    Captured torchrun stdout interleaves writes from multiple ranks at the byte
    level (no newline between rank-0's iter line and rank-7's "Number of
    parameters" line, e.g.). So we cannot rely on full-line matching: we locate
    each `iteration N/M |` marker and pull the deterministic fields by name
    from a small window after it. Wall-clock `elapsed time per iteration`
    is intentionally excluded.
    """
    results = []
    for m in _ITER_START_RE.finditer(stdout):
        window = stdout[m.start():m.start() + 800]
        lr = re.search(r"learning rate:\s*(\S+)", window)
        lm_loss = re.search(r"lm loss:\s*(\S+)", window)
        grad_norm = re.search(r"grad norm:\s*(\S+)", window)
        lb_loss = re.search(r"load_balancing_loss:\s*(\S+)", window)
        if not (lr and lm_loss and grad_norm):
            continue
        parts = [
            f"iter={m.group(1)}",
            f"lr={lr.group(1)}",
            f"lm_loss={lm_loss.group(1)}",
        ]
        if lb_loss:
            parts.append(f"lb_loss={lb_loss.group(1)}")
        parts.append(f"grad_norm={grad_norm.group(1)}")
        results.append(" | ".join(parts))
    return results


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
@pytest.mark.parametrize(
    "model_name,model_args,base_port",
    [
        ("moonlight", _MOONLIGHT_ARGS, 29660),
        ("qwen3",     _QWEN3_ARGS,     29662),
    ],
)
class TestE2EBitwise:
    """End-to-end bitwise comparison: pretrain_gpt.py noGraph vs cudaGraph.

    Each test launches `torchrun pretrain_gpt.py` twice -- once without CUDA
    graphs and once with `cuda_graph_impl=transformer_engine cuda_graph_scope=attn`
    -- using the exact same args as test_moonlight_qwen3_bitwise.sh.
    Asserts the per-iteration `lm loss / load_balancing_loss / grad norm`
    lines are byte-identical.

    Slow (~5 min per model). Marked `internal` so CI can opt-in.
    """

    def test_no_graph_vs_graph(self, model_name, model_args, base_port):
        # No graph baseline.
        r1 = _run_pretrain(model_args, cuda_graph_args=[], master_port=base_port)
        assert r1.returncode == 0, (
            f"[{model_name}] noGraph pretrain failed (rc={r1.returncode})\n"
            f"--- stdout (tail) ---\n{r1.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{r1.stderr[-2000:]}")
        metrics_eager = _extract_metrics(r1.stdout)
        assert len(metrics_eager) == _TRAIN_ITERS, (
            f"[{model_name}] noGraph: expected {_TRAIN_ITERS} metric lines, "
            f"got {len(metrics_eager)}\n"
            f"--- stdout (tail) ---\n{r1.stdout[-2000:]}")

        # CUDA graph capture.
        r2 = _run_pretrain(
            model_args,
            cuda_graph_args=[
                "--cuda-graph-impl", "transformer_engine",
                "--cuda-graph-scope", "attn",
            ],
            master_port=base_port + 1,
        )
        assert r2.returncode == 0, (
            f"[{model_name}] cudaGraph pretrain failed (rc={r2.returncode})\n"
            f"--- stdout (tail) ---\n{r2.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{r2.stderr[-2000:]}")
        metrics_graph = _extract_metrics(r2.stdout)
        assert len(metrics_graph) == _TRAIN_ITERS, (
            f"[{model_name}] cudaGraph: expected {_TRAIN_ITERS} metric lines, "
            f"got {len(metrics_graph)}\n"
            f"--- stdout (tail) ---\n{r2.stdout[-2000:]}")

        # Bitwise compare per iteration.
        for i, (a, b) in enumerate(zip(metrics_eager, metrics_graph)):
            assert a == b, (
                f"[{model_name}] iter {i+1} differs:\n"
                f"  eager: {a}\n"
                f"  graph: {b}")
