# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 HF<->lite checkpoint NUMERIC round-trip + ground-truth smoke.

The existing export unit tests (tests/model/test_qwen35_export.py) and the
save/load/export coverage smoke only assert NAMES / DTYPE / FINITENESS, never
numeric fidelity — so a transform that is self-consistent on the round-trip but
wrong vs real HF would pass silently. This smoke closes that blind spot:

  test_qwen35_save_load_numeric_roundtrip
    build A(seed1) -> save_hf -> build B(seed2) -> load_hf(B) -> assert every
    param allclose(A, B). Catches ASYMMETRIC load/export bugs (load != inverse
    of export). Vocab embed/head rows >= vocab_size are unused TP-padding (random
    in A, zeroed on reload) and are excluded; the real model has vocab%128==0 so
    no padding exists.

  test_qwen35_real_hf_load_export_matches_original  [opt-in: QWEN35_HF_DIR]
    GROUND TRUTH against real Qwen3.5 safetensors: load real HF -> export ->
    compare to the ORIGINAL safetensors tensor-by-tensor. Catches HF-file-level
    mismatches that native round-trip cannot see. It does NOT prove the loaded
    native layout is semantically correct if load/export are mutually inverse but
    both wrong vs forward semantics. Needs an 80GB GPU; loads only the first 8
    decoder layers to bound memory/time.

Run the standard numeric case with:
  experimental/lite/tests/run_tests.sh \
    experimental/lite/tests/primitive/test_qwen35_hf_numeric_roundtrip_smoke.py
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.gpus(1),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
]


def _qwen35_tiny_cfg():
    pytest.importorskip("fla", reason="qwen3_5 needs the FLA / GatedDeltaNet stack.")
    pytest.importorskip(
        "transformer_engine.pytorch", reason="qwen3_5 smoke needs real Transformer Engine."
    )
    from megatron.lite.model.qwen3_5.config import Qwen35Config

    return Qwen35Config(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["full_attention", "linear_attention"],
        partial_rotary_factor=1.0,
        max_position_embeddings=4096,
    )


def _build(cfg, seed):
    from megatron.lite.model.qwen3_5.lite import protocol
    from megatron.lite.runtime.contracts.config import ParallelConfig

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parallel = ParallelConfig(tp=1, pp=1, cp=1, ep=1, etp=1, vpp=1)
    impl = protocol.ImplConfig(
        parallel=parallel, optimizer="dist_opt", use_deepep=False, deterministic=True
    )
    bundle = protocol.build_model(cfg, impl_cfg=impl)
    return bundle, protocol


def _named(chunks):
    out = {}
    for i, chunk in enumerate(chunks):
        for name, param in chunk.named_parameters():
            out[f"{i}.{name}"] = param.detach().float().cpu().clone()
    return out


@pytest.fixture(scope="module", autouse=True)
def _single_rank_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for qwen3.5 numeric round-trip smoke.")
    created = False
    if not dist.is_initialized():
        for k, v in {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29597",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
        }.items():
            os.environ.setdefault(k, v)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        created = True
    yield
    if created and dist.is_initialized():
        dist.destroy_process_group()


def test_qwen35_save_load_numeric_roundtrip(tmp_path):
    if dist.get_world_size() != 1:
        pytest.skip("numeric round-trip smoke runs single-rank (tp1).")
    cfg = _qwen35_tiny_cfg()
    a, proto = _build(cfg, seed=1)
    b, _ = _build(cfg, seed=2)

    before = _named(a.chunks)
    b_before = _named(b.chunks)
    assert sum(1 for k in before if not torch.equal(before[k], b_before[k])) > 0

    out_dir = str(tmp_path / "hf")
    os.makedirs(out_dir, exist_ok=True)
    proto.save_hf_weights(a.chunks, out_dir, cfg, a.parallel_state)
    proto.load_hf_weights(b.chunks[0], out_dir, cfg, b.parallel_state)

    pa, pb = _named(a.chunks), _named(b.chunks)
    vocab = cfg.vocab_size
    mism = []
    for k in pa:
        ta, tb = pa[k], pb[k]
        if k.endswith("embed.embedding.weight") or k.endswith("head.col.linear.weight"):
            assert ta.shape[0] >= vocab
            ta, tb = ta[:vocab], tb[:vocab]  # exclude unused TP-padding rows
        # atol/rtol=1e-3 are bf16-cast round-trip tolerances.
        if not torch.allclose(ta, tb, atol=1e-3, rtol=1e-3):
            mism.append(f"{k} (max_abs_diff={(ta - tb).abs().max().item()})")
    assert not mism, "HF save->load not numeric round-trip:\n" + "\n".join(mism)


@pytest.mark.skipif(
    not os.environ.get("QWEN35_HF_DIR"),
    reason="set QWEN35_HF_DIR to a real Qwen3.5 HF checkpoint to run the ground-truth check.",
)
@pytest.mark.optional
def test_qwen35_real_hf_load_export_matches_original(tmp_path):
    if dist.get_world_size() != 1:
        pytest.skip("ground-truth smoke runs single-rank (tp1).")
    import json

    from safetensors import safe_open

    from megatron.lite.model.qwen3_5.config import Qwen35Config
    from megatron.lite.model.qwen3_5.lite import protocol
    from megatron.lite.runtime.contracts.config import ParallelConfig

    model_dir = os.environ["QWEN35_HF_DIR"]
    cfg = Qwen35Config.from_hf(model_dir)
    n = min(8, cfg.num_hidden_layers)
    cfg.num_hidden_layers = n
    cfg.layer_types = cfg.layer_types[:n]
    cfg.num_nextn_predict_layers = 0
    # Ensure both attention branches are actually exercised in the truncated model.
    assert "full_attention" in cfg.layer_types
    assert "linear_attention" in cfg.layer_types

    parallel = ParallelConfig(tp=1, pp=1, cp=1, ep=1, etp=1, vpp=1)
    impl = protocol.ImplConfig(
        parallel=parallel, optimizer="dist_opt", use_deepep=False, deterministic=True
    )
    bundle = protocol.build_model(cfg, impl_cfg=impl)
    protocol.load_hf_weights(bundle.chunks[0], model_dir, cfg, bundle.parallel_state)

    out_dir = str(tmp_path / "hf_export")
    os.makedirs(out_dir, exist_ok=True)
    protocol.save_hf_weights(bundle.chunks, out_dir, cfg, bundle.parallel_state)

    orig_map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))[
        "weight_map"
    ]

    def _orig(key):
        with safe_open(os.path.join(model_dir, orig_map[key]), framework="pt") as fh:
            return fh.get_tensor(key).float()

    exported = {}
    for fn in os.listdir(out_dir):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(out_dir, fn), framework="pt") as fh:
                for key in fh.keys():
                    exported[key] = fh.get_tensor(key).float()

    # Derive the expected comparison set from the ORIGINAL weight map (not from
    # `exported`): otherwise a tensor the exporter silently drops would never be
    # compared and a missing-export bug would pass undetected.
    #
    # Scope to what this truncated text-only build actually produces: the first
    # `n` decoder layers (prefix `model.language_model.layers.{i}.`) plus the
    # global embed/norm/head tensors. The original checkpoint also carries the
    # MTP head (`mtp.*`, disabled here via num_nextn_predict_layers=0) and the
    # vision tower (`model.visual.*`, not built on the text path); both are
    # intentionally NOT built/exported, so they are excluded from expected_keys.
    global_keys = (
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    )
    layer_prefixes = tuple(f"model.language_model.layers.{i}." for i in range(n))
    expected_keys = {
        k for k in orig_map if k.startswith(layer_prefixes) or k in global_keys
    }
    assert expected_keys, "no comparable tensors found in original weight map."

    missing = expected_keys - set(exported)
    assert not missing, "export missing tensors: " + ", ".join(sorted(missing))

    mism = []
    for k in sorted(expected_keys):
        o, e = _orig(k), exported[k]
        if o.shape != e.shape:
            mism.append(f"{k} shape {tuple(o.shape)} != {tuple(e.shape)}")
        elif not torch.allclose(o, e, atol=2e-2, rtol=2e-2):  # bf16-cast export tolerance
            mism.append(f"{k} max_abs_diff={(o - e).abs().max().item()}")
    assert not mism, "lite export does not match original HF safetensors:\n" + "\n".join(mism)
