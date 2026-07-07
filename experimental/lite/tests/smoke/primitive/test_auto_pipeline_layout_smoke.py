# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pipeline-layout smoke: real models build through the layout and train on PP>1.

Builds {qwen3_5, qwen3_moe, kimi_k2, glm5, deepseek_v4} with ``num_hidden_layers=9``
on pp=4 (not divisible) and asserts the layout drives a real model, not just the
layout math: each builds, trains a finite-loss step over the uneven split, and HF
export reconstructs every decoder layer. This exercises the auto layout mode on real
models; the custom ``pp_layout`` string mode and the exact per-stage splits (incl.
the real 43/61/78 counts) are pinned in the deterministic unit tests.

Topology follows the save/load/export smoke's validated dist_opt capability:
tp2/ep2 for TP-capable models, tp1/ep2 for glm5 / deepseek_v4 (native lite TP=1).
CP=1: cp>1 on these tiny proxy seqs breaks TE attention backend selection
(orthogonal to layout; covered by the dedicated CP smokes).

Run: torchrun --nproc_per_node=8 -m pytest --mlite-smoke, selecting per-env subsets
with -k (qwen3_5 on the qwen3.5 site; the rest on the DSA overlay). Models also
gate themselves with importorskip.
"""
from __future__ import annotations

import os
import re
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.primitive.deterministic import set_deterministic
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu, pytest.mark.distributed]

# 9 decoders over pp=4 is non-divisible; 9 is small enough to stay fast yet leaves no
# empty stage even for the MTP model (deepseek_v4 adds a slot on the last stage).
_NUM_LAYERS = 9
_PP = 4

# GLM5 / DeepSeek-V4 native lite support TP=1 only (matches the save/load smoke).
_TP1_ONLY = {"glm5", "deepseek_v4"}


def _require_te() -> None:
    te = pytest.importorskip(
        "transformer_engine.pytorch",
        reason="auto pipeline-layout smoke requires real Transformer Engine.",
    )
    assert hasattr(te, "Linear"), "smoke requires real Transformer Engine Linear."


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        optimizer="adam",
        lr=1.0e-3,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        clip_grad=1.0,
        offload_fraction=0.0,
    )


def _random_packed_batch(vocab_size: int) -> PackedBatch:
    return PackedBatch(
        input_ids=torch.randint(0, vocab_size, (2048,), device="cuda"),
        labels=torch.randint(0, vocab_size, (2048,), device="cuda"),
        seq_lens=torch.full((1,), 2048, dtype=torch.int64, device="cuda"),
    )


# ──────────────────────────────────────────────────────────────────────────
# Tiny non-divisible model configs (mirror the save/load smoke, num_hidden_layers=9).
# ──────────────────────────────────────────────────────────────────────────
def _qwen3_5():
    pytest.importorskip("fla", reason="qwen3_5 needs the FLA / GatedDeltaNet stack.")
    _require_te()
    from megatron.lite.model.qwen3_5.config import Qwen35Config
    from megatron.lite.model.qwen3_5.lite import protocol

    cfg = Qwen35Config(
        num_hidden_layers=_NUM_LAYERS,
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
        layer_types=(["full_attention", "linear_attention"] * ((_NUM_LAYERS + 1) // 2))[
            :_NUM_LAYERS
        ],
        partial_rotary_factor=1.0,
        max_position_embeddings=4096,
    )
    return cfg, protocol


def _qwen3_moe():
    _require_te()
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite import protocol

    cfg = Qwen3MoEConfig(
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        max_position_embeddings=4096,
        layer_types=["full_attention"] * _NUM_LAYERS,
    )
    return cfg, protocol


def _kimi_k2():
    _require_te()
    from megatron.lite.model.kimi_k2.config import KimiK2Config
    from megatron.lite.model.kimi_k2.lite import protocol

    cfg = KimiK2Config(
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        intermediate_size=96,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    return cfg, protocol


def _glm5():
    pytest.importorskip("cudnn", reason="glm5 fused DSA needs the cudnn DSA stack.")
    _require_te()
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import protocol

    cfg = Glm5Config(
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=128,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        vocab_size=32,
        max_position_embeddings=4096,
        initializer_range=0.002,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=512,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )
    return cfg, protocol


def _deepseek_v4():
    pytest.importorskip("cudnn", reason="deepseek_v4 fused DSA needs the cudnn DSA stack.")
    _require_te()
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite import protocol

    cfg = DeepseekV4Config(
        vocab_size=64,
        hidden_size=128,
        moe_intermediate_size=16,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=64,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        max_position_embeddings=4096,
        compress_ratios=[4, 4],
        sliding_window=128,
        num_hash_layers=2,
        hc_mult=2,
        index_head_dim=64,
        index_n_heads=8,
        index_topk=512,
        # Real MTP: an extra nextn layer on the last stage exercises the
        # MTP-aware branch of the auto layout balancing.
        num_nextn_predict_layers=1,
        rms_norm_eps=1e-6,
    )
    return cfg, protocol


MODELS = {
    "qwen3_5": _qwen3_5,
    "qwen3_moe": _qwen3_moe,
    "kimi_k2": _kimi_k2,
    "glm5": _glm5,
    "deepseek_v4": _deepseek_v4,
}


# ──────────────────────────────────────────────────────────────────────────
# Distributed harness (mirrors the save/load/export smoke).
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for auto pipeline-layout smoke.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 8:
        pytest.skip("Megatron Lite smoke tests are capped at single-node 8 GPUs.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        timeout_s = int(os.environ.get("MLITE_DIST_TIMEOUT_S", "180"))
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(seconds=timeout_s)
        )
        created_pg = True
    yield
    try:
        from megatron.core import parallel_state as mpu

        if mpu.is_initialized():
            mpu.destroy_model_parallel()
    finally:
        if created_pg and dist.is_initialized():
            dist.destroy_process_group()


# Process groups lite's init_parallel creates per build, recorded so the autouse
# teardown can free them (mcore's destroy_model_parallel frees only mcore's groups).
_BUILT_PARALLEL_STATES: list = []
_PS_GROUP_ATTRS = (
    "tp_group", "ep_group", "etp_group", "cp_group", "pp_group", "pp_cpu_group",
    "dp_group", "dp_cp_group", "tp_ep_group", "ep_dp_group",
)


@pytest.fixture(autouse=True)
def _reset_parallel_state_between_tests():
    yield
    import gc

    from megatron.core import parallel_state as mpu

    if mpu.is_initialized():
        mpu.destroy_model_parallel()
    # Finalize the just-built model first so DDP releases its group references, then
    # explicitly destroy the process groups lite's init_parallel created for this
    # test. Without this, lite's ~10 fresh NCCL/gloo groups per build leak across the
    # ~6 builds in one torchrun process and make a later PP P2P / export collective
    # fail nondeterministically (mcore's destroy_model_parallel frees only its own).
    gc.collect()
    for ps in _BUILT_PARALLEL_STATES:
        for attr in _PS_GROUP_ATTRS:
            group = getattr(ps, attr, None)
            if group is not None:
                try:
                    dist.destroy_process_group(group)
                except Exception:
                    pass
    _BUILT_PARALLEL_STATES.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _proxy_topology(model_name: str, pp_layout=None) -> ParallelConfig:
    # PP is held at 4 (the layout axis under test); the rest follow the save/load
    # smoke's validated dist_opt capability. CP=1 — see module docstring.
    # pp_layout=None -> auto mode; a string -> custom mode (advanced explicit layout).
    if model_name in _TP1_ONLY:  # glm5 / deepseek_v4: native lite is TP=1 only.
        return ParallelConfig(tp=1, ep=2, etp=1, pp=_PP, cp=1, pp_layout=pp_layout)
    return ParallelConfig(tp=2, ep=2, etp=1, pp=_PP, cp=1, pp_layout=pp_layout)


def _build_handle(model_name: str, *, seed: int, pp_layout=None):
    from types import SimpleNamespace

    from megatron.lite.runtime.contracts.handle import ModelHandle

    cfg, protocol = MODELS[model_name]()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parallel = _proxy_topology(model_name, pp_layout=pp_layout)
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer="dist_opt",
        optimizer_config=_optimizer_config(),
        use_deepep=False,
        deterministic=True,
    )
    bundle = protocol.build_model(cfg, impl_cfg=impl_cfg)
    chunks = bundle.chunks
    extras = dict(bundle.extras)
    extras.update(
        {
            "model_chunks": chunks,
            "forward_step": bundle.forward_step,
            "finalize_grads": bundle.finalize_grads,
            "protocol": protocol,
        }
    )
    handle = ModelHandle(
        model=chunks,
        optimizer=bundle.optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=parallel),
        _extras=extras,
    )
    _BUILT_PARALLEL_STATES.append(bundle.parallel_state)
    return handle, cfg


def _local_layer_indices(handle) -> list[int]:
    chunk = unwrap_model(handle._extras["model_chunks"][0])
    return list(chunk.layer_indices)


# Decoder layer ids in exported HF names: matches both `model.layers.N.` (HF-rooted,
# most models) and bare `layers.N.` (deepseek_v4-flash release naming).
_HF_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _hf_decoder_layer_indices(names) -> set[int]:
    """Decoder layer ids referenced by exported HF weight names (``...layers.N...``)."""
    return {int(m.group(1)) for n in names if (m := _HF_LAYER_RE.search(n))}


@pytest.mark.parametrize("model_name", list(MODELS))
def test_uneven_pp_builds_trains_and_exports(model_name):
    """A non-divisible layer count builds an uneven PP split, trains a step, and
    exports every layer. Build + train + export share one model build (one heavy
    distributed build per test keeps the builds in one torchrun process bounded).

    - build: the non-divisible count produces a valid balanced uneven split (not
      "not divisible"); this stage owns a sorted, in-range, contiguous decoder run.
    - train: one real step over the uneven pipeline yields a finite loss (proves PP
      P2P across stages of unequal depth actually runs the built model).
    - export: HF export all-gathers the per-stage shards (keyed on global layer
      names) back to a complete state — every decoder layer 0..N-1 reconstructed,
      every tensor finite. Regression guard for the deepseek_v4 local-vs-global
      layer-key bug; a dropped/duplicated stage would silently corrupt exports.
    """
    if dist.get_world_size() != 8:
        pytest.skip("auto pipeline-layout proxy smoke requires exactly 8 GPUs.")

    set_deterministic(2026)

    handle, cfg = _build_handle(model_name, seed=4242)
    ps = handle._parallel_state
    assert ps.pp_size == _PP
    assert cfg.num_hidden_layers == _NUM_LAYERS

    local = _local_layer_indices(handle)
    assert local == sorted(local) and len(set(local)) == len(local), local
    assert local and all(0 <= i < _NUM_LAYERS for i in local), local
    assert local == list(range(local[0], local[0] + len(local))), local

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    batch = _random_packed_batch(cfg.vocab_size)
    runtime.zero_grad(handle)
    result = runtime.forward_backward(handle, iter([batch]), None, num_microbatches=1)
    runtime.optimizer_step(handle)
    loss = result.model_output.loss
    assert loss is not None and torch.isfinite(loss).all(), (
        f"{model_name}: non-finite loss {loss} on uneven PP layout"
    )

    # Export across the uneven PP split (every rank materializes the full HF state).
    protocol = handle._extras["protocol"]
    chunks = handle._extras["model_chunks"]
    exported = list(protocol.export_hf_weights(chunks, cfg, ps))
    present = _hf_decoder_layer_indices([name for name, _ in exported])
    expected = set(range(cfg.num_hidden_layers))
    assert expected.issubset(present), (
        f"{model_name}: uneven-PP export dropped decoder layers "
        f"{sorted(expected - present)} (have {sorted(present)}); PP gather lost a stage"
    )
    for name, tensor in exported:
        assert torch.isfinite(tensor.float()).all(), (
            f"{model_name}: non-finite exported tensor {name} from uneven PP gather"
        )
    if dist.get_rank() == 0:
        print(
            "NON_SKIP_UNEVEN_PP_BUILD_TRAIN_EXPORT "
            f"model={model_name} num_layers={cfg.num_hidden_layers} "
            f"split={local} exported_decoder_layers={len(present & expected)}"
        )


# Custom-string mode (``ParallelConfig.pp_layout``) is covered by the deterministic
# unit tests (test_parallel_unit.py: the explicit string is honored verbatim, differs
# from the auto split, and a >pp-stage string raises). It shares the build path with
# the auto matrix above (only ``ps.pp_layout`` differs), so it is not given a separate
# GPU build here: keeping this smoke to one heavy distributed build per model avoids a
# trailing build tripping NCCL P2P comm-init timeouts from process-group churn.
