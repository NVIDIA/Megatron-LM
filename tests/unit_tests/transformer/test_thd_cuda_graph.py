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

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import torch

from megatron.core.packed_seq_params import (
    PackedSeqParams,
    pad_sequence_for_thd,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils

# cuDNN 9.22 has no *deterministic* fused-attention backend for THD packed-sequence inputs
# (including the MLA asymmetric qk/v dims used here, e.g. Moonlight qk=192/v=128). With the
# strict NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 every backend is disabled and attention raises
# "No dot product attention backend". Allow non-deterministic algos: the no-graph vs graph
# runs are still bitwise identical because shapes, seed and CUBLAS_WORKSPACE_CONFIG are fixed
# (the selected algo is deterministic in practice).
os.environ.setdefault('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1')
os.environ.setdefault('NVTE_CUTEDSL_FUSED_GROUPED_MLP', '1')
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
        qkv_format='thd',
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu.clone(),
        cu_seqlens_q_padded=cu.clone(),
        cu_seqlens_kv_padded=cu.clone(),
        max_seqlen_q=max(seqlens),
        max_seqlen_kv=max(seqlens),
    )


def _build_layer(H, nh, nkv, ffn, max_seqlen, max_num_seqs, tp=1, sp=False):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

    config = TransformerConfig(
        num_layers=1,
        hidden_size=H,
        num_attention_heads=nh,
        num_query_groups=nkv,
        ffn_hidden_size=ffn,
        max_seqlen_per_dp_cp_rank=max_seqlen,
        thd_max_num_seqs=max_num_seqs,
        tensor_model_parallel_size=tp,
        sequence_parallel=sp,
        bf16=True,
    )
    model_parallel_cuda_manual_seed(42)
    return (
        TransformerLayer(
            config, get_gpt_layer_with_transformer_engine_spec().submodules, layer_number=1
        )
        .cuda()
        .bfloat16()
    )


# =============================================================================
# 1. pad_sequence_for_thd correctness
# =============================================================================


class TestPadSequenceForThd:

    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_generic_alignment_preserves_cu_seqlens(self):
        """Generic THD padding aligns token tensors while preserving sequence metadata."""
        seqlens, total_T = [50, 30], 80
        psp = _make_psp(seqlens)
        orig = psp.cu_seqlens_q.clone()
        p_tok, _, _, _, p, mask = pad_sequence_for_thd(
            torch.ones(1, total_T, device="cuda"), None, None, None, psp, alignment=64
        )
        assert p_tok.shape == (1, 128)
        assert torch.equal(p.cu_seqlens_q, orig)
        assert mask.shape == (1, 128)
        assert not mask[0, :total_T].any() and mask[0, total_T:].all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_shapes_and_data_preservation(self):
        """Shapes are static; original data intact; padding zero-filled."""
        seqlens, max_seqlen, max_num_seqs = [100, 50, 30], 256, 8
        total_T = sum(seqlens)
        tokens = torch.arange(total_T, device="cuda").unsqueeze(0).float()
        p_tok, p_lab, p_loss, p_pos, p_params, p_mask = pad_sequence_for_thd(
            tokens,
            tokens.clone(),
            torch.ones(1, total_T, device="cuda"),
            torch.arange(total_T, device="cuda").unsqueeze(0),
            _make_psp(seqlens),
            target_len=max_seqlen,
            max_num_seqs=max_num_seqs,
        )
        for t in (p_tok, p_lab, p_loss, p_pos):
            assert t.shape == (1, max_seqlen)
        for cu in (
            p_params.cu_seqlens_q,
            p_params.cu_seqlens_kv,
            p_params.cu_seqlens_q_padded,
            p_params.cu_seqlens_kv_padded,
        ):
            assert cu.shape[0] == max_num_seqs + 1
        assert p_mask.shape == (1, max_seqlen) and p_mask.dtype == torch.bool
        assert torch.equal(p_tok[0, :total_T], tokens[0])
        assert (p_tok[0, total_T:] == 0).all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_padding_mask_boundary(self):
        """False at real positions, True at padding (MoE aux-loss contract)."""
        seqlens, total_T, max_seqlen = [60, 40], 100, 128
        _, _, _, _, _, m = pad_sequence_for_thd(
            torch.ones(1, total_T, device="cuda"),
            None,
            None,
            None,
            _make_psp(seqlens),
            target_len=max_seqlen,
            max_num_seqs=4,
        )
        assert not m[0, :total_T].any() and m[0, total_T:].all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cu_seqlens_fill_value(self):
        """Padded entries repeat last cumulative sum (prevents OOB reads)."""
        seqlens, total_T = [50, 30], 80
        _, _, _, _, p, _ = pad_sequence_for_thd(
            torch.ones(1, total_T, device="cuda"),
            None,
            None,
            None,
            _make_psp(seqlens),
            target_len=128,
            max_num_seqs=32,
        )
        assert p.cu_seqlens_q[0] == 0 and p.cu_seqlens_q[2] == 80
        assert (p.cu_seqlens_q[3:] == 80).all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_none_inputs(self):
        """Non-pre_process PP: mask from cu_seqlens when all tensors None."""
        seqlens, total_T, max_seqlen = [50, 30], 80, 128
        _, _, _, _, _, mask = pad_sequence_for_thd(
            None, None, None, None, _make_psp(seqlens), target_len=max_seqlen, max_num_seqs=4
        )
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
        orig = {
            k: getattr(psp, k).clone()
            for k in (
                'cu_seqlens_q',
                'cu_seqlens_kv',
                'cu_seqlens_q_padded',
                'cu_seqlens_kv_padded',
            )
        }
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


class TestChunkWiseStaticInputs:

    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_transformer_block_static_inputs_use_thd_tensors(self):
        """Chunk-wise THD capture uses packed-sequence tensors instead of attention_mask."""
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )

        max_seqlen = 128
        max_num_seqs = 8
        config = TransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=1024,
            max_seqlen_per_dp_cp_rank=max_seqlen,
            thd_max_num_seqs=max_num_seqs,
            sequence_packing_scheduler="dp_balanced",
            moe_token_dispatcher_type="alltoall",
            cuda_graph_impl="transformer_engine",
            cuda_graph_granularity="chunk",
            cuda_graph_modules=[],
            bf16=True,
        )

        model_parallel_cuda_manual_seed(42)
        block = (
            TransformerBlock(config, get_gpt_layer_with_transformer_engine_spec())
            .cuda()
            .bfloat16()
        )
        static_inputs = block.get_layer_static_inputs(seq_length=max_seqlen, micro_batch_size=4)

        assert static_inputs["hidden_states"].shape == (max_seqlen, 1, 256)
        assert static_inputs["hidden_states"].dtype == torch.bfloat16
        assert "attention_mask" not in static_inputs
        for key in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
        ):
            assert static_inputs[key].shape == (max_num_seqs + 1,)
            assert static_inputs[key].dtype == torch.int32
        assert static_inputs["padding_mask"].shape == (1, max_seqlen)
        assert static_inputs["padding_mask"].dtype == torch.bool

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_chunk_replay_uses_pipeline_input_tensor_when_hidden_is_none(self, monkeypatch):
        """PP non-first stages pass hidden_states=None; chunk replay must use input_tensor."""
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.transformer.module import GraphableMegatronModule

        max_seqlen = 128
        max_num_seqs = 8
        config = TransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=1024,
            max_seqlen_per_dp_cp_rank=max_seqlen,
            thd_max_num_seqs=max_num_seqs,
            sequence_packing_scheduler="dp_balanced",
            moe_token_dispatcher_type="alltoall",
            cuda_graph_impl="transformer_engine",
            cuda_graph_granularity="chunk",
            cuda_graph_modules=[],
            bf16=True,
        )

        model_parallel_cuda_manual_seed(42)
        block = (
            TransformerBlock(
                config,
                get_gpt_layer_with_transformer_engine_spec(),
                pre_process=False,
            )
            .cuda()
            .bfloat16()
        )
        runtime_hidden = torch.randn(max_seqlen, 1, 256, device="cuda", dtype=torch.bfloat16)
        block.set_input_tensor(runtime_hidden)
        psp = _make_psp([64, 32])
        padding_mask = torch.zeros(1, max_seqlen, dtype=torch.bool, device="cuda")
        captured = {}

        def fake_base_replay(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return kwargs["hidden_states"]

        monkeypatch.setattr(GraphableMegatronModule, "_te_cuda_graph_replay", fake_base_replay)

        out = block._te_cuda_graph_replay(
            hidden_states=None,
            attention_mask=None,
            inference_context=None,
            packed_seq_params=psp,
            padding_mask=padding_mask,
        )

        assert out is runtime_hidden
        assert captured["kwargs"]["hidden_states"] is runtime_hidden
        assert "packed_seq_params" not in captured["kwargs"]
        assert captured["kwargs"]["cu_seqlens_q"] is psp.cu_seqlens_q
        assert captured["kwargs"]["padding_mask"] is padding_mask


# =============================================================================
# 3. E2E no-graph vs graph bitwise loss/grad_norm match
#    Subprocess-launches `torchrun pretrain_gpt.py` -- same recipe as
#    test_moonlight_qwen3_bitwise.sh -- and asserts the per-iteration
#    metric strings are byte-identical between the two runs.
# =============================================================================

# Common args shared across both models.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_SFT_JSON = (
    '{"mode":"distribution","type":"lognormal",'
    '"min_seq_len":128,"max_seq_len":2048,"mean_seq_len":1024,"lognormal_sigma":0.8}'
)

_TRAIN_ITERS = 5

_COMMON_ARGS = [
    "--seq-length",
    "2048",
    "--max-position-embeddings",
    "8192",
    "--micro-batch-size",
    "1",
    "--global-batch-size",
    "4",
    "--train-iters",
    str(_TRAIN_ITERS),
    "--lr",
    "1e-5",
    "--min-lr",
    "1e-6",
    "--lr-decay-style",
    "cosine",
    "--lr-warmup-iters",
    "1",
    "--weight-decay",
    "0.01",
    "--clip-grad",
    "1.0",
    "--seed",
    "1234",
    "--te-rng-tracker",
    "--bf16",
    "--tensor-model-parallel-size",
    "2",
    "--pipeline-model-parallel-size",
    "2",
    "--context-parallel-size",
    "2",
    "--swiglu",
    "--disable-bias-linear",
    "--sequence-parallel",
    "--sft",
    "--mock-data",
    "--tokenizer-type",
    "NullTokenizer",
    "--sft-mock-dataset-config-json",
    _SFT_JSON,
    "--sequence-packing-scheduler",
    "dp_balanced",
    "--max-seqlen-per-dp-cp-rank",
    "1024",
    "--pad-packed-seq-alignment",
    "--calculate-per-token-loss",
    "--transformer-impl",
    "transformer_engine",
    "--attention-dropout",
    "0",
    "--hidden-dropout",
    "0",
    "--no-bias-swiglu-fusion",
    "--no-gradient-accumulation-fusion",
    "--no-save-optim",
    "--no-save-rng",
    "--save-interval",
    "999999",
    "--eval-interval",
    "999999",
    "--eval-iters",
    "1",
    "--log-interval",
    "1",
    "--no-check-for-nan-in-loss-and-grad",
    "--deterministic-mode",
    "--thd-max-num-seqs",
    "32",
]

_MOONLIGHT_ARGS = _COMMON_ARGS + [
    "--num-layers",
    "27",
    "--hidden-size",
    "2048",
    "--ffn-hidden-size",
    "11264",
    "--num-attention-heads",
    "16",
    "--decoder-first-pipeline-num-layers",
    "13",
    "--decoder-last-pipeline-num-layers",
    "14",
    "--expert-model-parallel-size",
    "4",
    "--expert-tensor-parallel-size",
    "1",
    "--multi-latent-attention",
    "--kv-lora-rank",
    "512",
    "--qk-head-dim",
    "128",
    "--qk-pos-emb-head-dim",
    "64",
    "--v-head-dim",
    "128",
    "--num-experts",
    "64",
    "--moe-ffn-hidden-size",
    "1408",
    "--moe-router-topk",
    "6",
    "--moe-shared-expert-intermediate-size",
    "2816",
    "--moe-layer-freq",
    "([0]+[1]*26)",
    "--moe-token-dispatcher-type",
    "flex",
    "--moe-flex-dispatcher-backend",
    "hybridep",
    "--moe-router-score-function",
    "sigmoid",
    "--moe-router-topk-scaling-factor",
    "2.446",
    "--moe-router-load-balancing-type",
    "aux_loss",
    "--moe-aux-loss-coeff",
    "0.001",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-5",
    "--rotary-base",
    "50000",
    "--vocab-size",
    "163840",
]

_QWEN3_ARGS = _COMMON_ARGS + [
    "--num-layers",
    "36",
    "--hidden-size",
    "4096",
    "--ffn-hidden-size",
    "12288",
    "--num-attention-heads",
    "32",
    "--group-query-attention",
    "--num-query-groups",
    "8",
    "--max-position-embeddings",
    "40960",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-6",
    "--rotary-base",
    "1000000",
    "--untie-embeddings-and-output-weights",
    "--vocab-size",
    "151936",
    "--moe-token-dispatcher-type",
    "flex",
    "--moe-flex-dispatcher-backend",
    "hybridep",
]


def _run_pretrain(model_args, cuda_graph_args, master_port):
    """Subprocess-launch `torchrun pretrain_gpt.py` once and capture stdout."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    env["PATH"] = "/usr/bin:/usr/local/bin:" + env.get("PATH", "")
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # See module-level note: cuDNN 9.22 has no deterministic THD fused-attention backend for
    # the MLA dims used here, so =0 leaves no usable backend. =1 still yields bitwise-identical
    # no-graph vs graph metrics (shapes/seed/CUBLAS_WORKSPACE_CONFIG are fixed).
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"
    env["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["NCCL_ALGO"] = "^NVLS"
    # Strip any inherited torchrun env so this subprocess starts a fresh group.
    for k in list(env.keys()):
        if k.startswith(
            (
                "TORCHELASTIC_",
                "MASTER_",
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "GROUP_RANK",
                "LOCAL_WORLD_SIZE",
            )
        ):
            env.pop(k, None)
    # Clear pytest-conftest env vars that disable TE attention backends
    # (set by tests/unit_tests/conftest.py::set_env). Pretrain needs at
    # least one of fused/flash attention to build the model.
    env.pop("NVTE_FLASH_ATTN", None)
    env.pop("NVTE_FUSED_ATTN", None)

    cmd = (
        [
            "torchrun",
            "--nproc_per_node",
            "8",
            "--nnodes",
            "1",
            "--master_addr",
            "localhost",
            "--master_port",
            str(master_port),
            "pretrain_gpt.py",
        ]
        + model_args
        + cuda_graph_args
    )

    result = subprocess.run(
        cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=900
    )
    return result


def _replace_arg(args, flag, value):
    args = list(args)
    try:
        idx = args.index(flag)
    except ValueError:
        args.extend([flag, value])
    else:
        args[idx + 1] = value
    return args


def _remove_arg_pair(args, flag):
    args = list(args)
    while flag in args:
        idx = args.index(flag)
        del args[idx : idx + 2]
    return args


def _reduced_moonlight_vpp_hybridep_args():
    sft_json = json.dumps(
        {
            "mode": "distribution",
            "type": "lognormal",
            "min_seq_len": 128,
            "max_seq_len": 512,
            "mean_seq_len": 256,
            "lognormal_sigma": 0.8,
        }
    )
    args = list(_COMMON_ARGS)
    args = _replace_arg(args, "--seq-length", "512")
    args = _replace_arg(args, "--attention-dropout", "0.1")
    args = _replace_arg(args, "--hidden-dropout", "0.1")
    args = _replace_arg(args, "--lr-warmup-iters", "0")
    args = _replace_arg(args, "--sft-mock-dataset-config-json", sft_json)
    args = _remove_arg_pair(args, "--global-batch-size")
    args = _remove_arg_pair(args, "--train-iters")
    args.extend(
        [
            "--step-batch-size-schedule",
            "0:4 2048:6",
            "--train-samples",
            "10",
            "--lr-decay-samples",
            "10",
            "--eval-global-batch-size",
            "4",
            "--num-layers",
            "4",
            "--hidden-size",
            "512",
            "--ffn-hidden-size",
            "2048",
            "--num-attention-heads",
            "8",
            "--num-layers-per-virtual-pipeline-stage",
            "1",
            "--expert-model-parallel-size",
            "4",
            "--expert-tensor-parallel-size",
            "1",
            "--multi-latent-attention",
            "--kv-lora-rank",
            "32",
            "--qk-head-dim",
            "64",
            "--qk-pos-emb-head-dim",
            "64",
            "--v-head-dim",
            "64",
            "--num-experts",
            "8",
            "--moe-ffn-hidden-size",
            "512",
            "--moe-router-topk",
            "2",
            "--moe-shared-expert-intermediate-size",
            "512",
            "--moe-layer-freq",
            "([0]+[1]*3)",
            "--moe-token-dispatcher-type",
            "flex",
            "--moe-flex-dispatcher-backend",
            "hybridep",
            "--moe-router-dtype",
            "fp32",
            "--moe-router-load-balancing-type",
            "aux_loss",
            "--moe-aux-loss-coeff",
            "0.001",
            "--moe-expert-capacity-factor",
            "1.0",
            "--moe-pad-expert-input-to-capacity",
            "--normalization",
            "RMSNorm",
            "--norm-epsilon",
            "1e-5",
            "--vocab-size",
            "8192",
        ]
    )
    return args


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
        window = stdout[m.start() : m.start() + 800]
        lr = re.search(r"learning rate:\s*(\S+)", window)
        lm_loss = re.search(r"lm loss:\s*(\S+)", window)
        grad_norm = re.search(r"grad norm:\s*(\S+)", window)
        lb_loss = re.search(r"load_balancing_loss:\s*(\S+)", window)
        if not (lr and lm_loss and grad_norm):
            continue
        parts = [f"iter={m.group(1)}", f"lr={lr.group(1)}", f"lm_loss={lm_loss.group(1)}"]
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
    [("moonlight", _MOONLIGHT_ARGS, 29660), ("qwen3", _QWEN3_ARGS, 29662)],
)
class TestE2EBitwise:
    """End-to-end bitwise comparison: pretrain_gpt.py noGraph vs cudaGraph.

    Each test launches `torchrun pretrain_gpt.py` twice -- once without CUDA
    graphs and once with `cuda_graph_impl=transformer_engine cuda_graph_modules=attn`
    -- using the same model/test settings as test_moonlight_qwen3_bitwise.sh.
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
            f"--- stderr (tail) ---\n{r1.stderr[-2000:]}"
        )
        metrics_eager = _extract_metrics(r1.stdout)
        assert len(metrics_eager) == _TRAIN_ITERS, (
            f"[{model_name}] noGraph: expected {_TRAIN_ITERS} metric lines, "
            f"got {len(metrics_eager)}\n"
            f"--- stdout (tail) ---\n{r1.stdout[-2000:]}"
        )

        # CUDA graph capture.
        r2 = _run_pretrain(
            model_args,
            cuda_graph_args=[
                "--cuda-graph-impl",
                "transformer_engine",
                "--cuda-graph-modules",
                "attn",
            ],
            master_port=base_port + 1,
        )
        assert r2.returncode == 0, (
            f"[{model_name}] cudaGraph pretrain failed (rc={r2.returncode})\n"
            f"--- stdout (tail) ---\n{r2.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{r2.stderr[-2000:]}"
        )
        metrics_graph = _extract_metrics(r2.stdout)
        assert len(metrics_graph) == _TRAIN_ITERS, (
            f"[{model_name}] cudaGraph: expected {_TRAIN_ITERS} metric lines, "
            f"got {len(metrics_graph)}\n"
            f"--- stdout (tail) ---\n{r2.stdout[-2000:]}"
        )

        # Bitwise compare per iteration.
        for i, (a, b) in enumerate(zip(metrics_eager, metrics_graph)):
            assert a == b, f"[{model_name}] iter {i+1} differs:\n" f"  eager: {a}\n" f"  graph: {b}"



def _moonlight_hybridep_paged_stash_args():
    """Full Moonlight + HybridEP + paged stash (validated on GB200 / SM100).

    ``_MOONLIGHT_ARGS`` already selects the flex / HybridEP dispatcher; this
    appends the sync-free grouped-MLP + paged-stash flags exercised by the
    chunk-wise local capture under paged stash.
    """
    return _MOONLIGHT_ARGS + [
        "--moe-grouped-gemm",
        "--use-transformer-engine-op-fuser",
        "--moe-mlp-glu-interleave-size",
        "32",
        "--moe-expert-rank-capacity-factor",
        "4.0",
        "--moe-paged-stash",
        "--moe-paged-stash-buffer-size-factor-cuda",
        "4.0",
        "--moe-paged-stash-buffer-size-factor-cpu",
        "0.0",
    ]


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
@pytest.mark.skipif(not HAVE_HYBRIDEP, reason="HybridEP is not available")
class TestE2ELocalChunkHybridEP:
    """SM-aware end-to-end bitwise check for the local F/B-split chunk graph.

    The MoE path is selected by GPU compute capability so each platform runs the
    configuration it was validated on:
      * SM100 / Blackwell -> HybridEP + paged stash (slot count auto-inferred)
      * SM90  / Hopper     -> HybridEP + drop-and-pad

    Compares no-graph vs ``--cuda-graph-impl local --cuda-graph-granularity chunk``
    and asserts the per-iteration metrics are bitwise identical.
    """

    def test_no_graph_vs_local_chunk_graph(self):
        major = torch.cuda.get_device_capability()[0]
        graph_args = [
            "--cuda-graph-impl",
            "local",
            "--cuda-graph-granularity",
            "chunk",
            "--cuda-graph-dynamic-microbatches",
        ]
        if major >= 10:  # Blackwell / SM100: paged stash, slot count auto-inferred
            model_args = _moonlight_hybridep_paged_stash_args()
            label = "sm100-local-chunk-hybridep-paged-stash"
        elif major == 9:  # Hopper / SM90: drop-and-pad
            model_args = _reduced_moonlight_vpp_hybridep_args()
            label = "sm90-local-chunk-hybridep-drop-pad"
            graph_args += ["--cuda-graph-num-microbatch-slots", "4"]
        else:
            pytest.skip(
                f"SM{major}0 not covered by this e2e; expect SM90 (drop-and-pad) "
                "or SM100 (paged-stash)."
            )

        r1 = _run_pretrain(model_args, cuda_graph_args=[], master_port=29672)
        assert r1.returncode == 0, (
            f"[{label}] noGraph pretrain failed (rc={r1.returncode})\n"
            f"--- stdout (tail) ---\n{r1.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{r1.stderr[-2000:]}"
        )
        r2 = _run_pretrain(model_args, cuda_graph_args=graph_args, master_port=29673)
        assert r2.returncode == 0, (
            f"[{label}] localChunk pretrain failed (rc={r2.returncode})\n"
            f"--- stdout (tail) ---\n{r2.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{r2.stderr[-2000:]}"
        )

        metrics_eager = _extract_metrics(r1.stdout)
        metrics_graph = _extract_metrics(r2.stdout)
        assert len(metrics_eager) >= 1, (
            f"[{label}] noGraph produced no metric lines\n"
            f"--- stdout (tail) ---\n{r1.stdout[-2000:]}"
        )
        assert metrics_eager == metrics_graph, (
            f"[{label}] metrics differ:\n  eager: {metrics_eager}\n  graph: {metrics_graph}"
        )
