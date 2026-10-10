# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build DeepSeek-V4 with MXFP4 bindings and run packed CP2 inference."""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.optional,
    pytest.mark.gpus(2, min_architecture="blackwell"),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
]


@pytest.fixture(scope="module")
def cp_world():
    # Runtime process groups are cached across model builds. Keep the world alive
    # across both precision cases so cached subgroups remain valid.
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("run with torchrun on two Blackwell GPUs")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    try:
        yield
    finally:
        if created_group:
            dist.destroy_process_group()


@pytest.mark.parametrize("precision", ("fast", "exact"))
def test_csa_protocol_cp2_matches_upstream_and_bypasses_training(precision, cp_world):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite import protocol
    from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention
    from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig

    _check_protocol(
        protocol,
        DeepseekV4Config,
        CompressedSparseAttention,
        PackedBatch,
        ParallelConfig,
        precision,
    )


def _check_protocol(protocol, config_type, attention_type, batch_type, parallel_type, precision):
    selector = {"backend": "reference", "precision": "fast"}
    if precision == "exact":

        def read(variable):
            value = os.environ.get(variable, "").strip()
            if not value:
                pytest.skip(f"{variable} is not set")
            return json.loads(value if value[0] in "[{" else Path(value).read_text())

        spec = next(
            (
                s
                for s in read("LITETOPK_TEST_SELECTORS")
                if s["native_format"] == "mxfp4" and s["precision"] == "exact"
            ),
            None,
        )
        if spec is None:
            pytest.skip("an exact MXFP4 selector is required")
        exact = read("LITETOPK_TEST_EXACT_TOPK")
        for entry in reversed(exact.pop("pythonpath", [])):
            sys.path.insert(0, entry)
        selector = {
            "backend": "litetopk",
            "precision": "exact",
            "litetopk": spec["litetopk"],
            "exact_topk": exact,
        }

    cfg = config_type(
        vocab_size=64,
        hidden_size=128,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        max_position_embeddings=1024,
        compress_ratios=[4, 128],
        sliding_window=128,
        num_hash_layers=0,
        hc_mult=2,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        num_nextn_predict_layers=0,
    )

    def build(spec):
        return protocol.build_model(
            cfg,
            impl_cfg=protocol.ImplConfig(
                parallel=parallel_type(cp=2), optimizer=None, mtp_enable=False, indexer_topk=spec
            ),
        )

    torch.manual_seed(20261004)
    unbound_bundle = build(None)
    bound_bundle = build(selector)
    unbound, bound = unbound_bundle.chunks[0].eval(), bound_bundle.chunks[0].eval()
    bound.load_state_dict(unbound.state_dict())
    installation = bound_bundle.extras["indexer_topk"]
    assert "indexer_topk" not in unbound_bundle.extras
    assert len(installation.bindings) == 1
    attentions = [m for m in bound.modules() if isinstance(m, attention_type)]
    assert [m._indexer_topk for m in attentions] == [installation.bindings[0], None]

    # All visible compressed keys fit in top-k. The sequence boundary crosses rank 0.
    lengths = [384, 640]
    tokens = torch.randint(0, cfg.vocab_size, (2, sum(lengths)), device="cuda")
    batch = batch_type(
        input_ids=tokens[0],
        labels=tokens[1],
        seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        loss_mask=torch.ones(sum(lengths), device="cuda"),
    )
    with torch.no_grad():
        expected = protocol._forward_step(unbound, batch)
        actual = protocol._forward_step(bound, batch)
    assert torch.isfinite(actual["loss"])
    assert torch.isfinite(actual["log_probs"]).all()
    torch.testing.assert_close(actual["log_probs"], expected["log_probs"], atol=2e-2, rtol=2e-2)
    stats = installation.bindings[0].stats
    assert (stats.calls, stats.rows, stats.reference_rows, stats.litetopk_rows) == (1, 512, 512, 0)

    bound.train()
    with torch.no_grad():
        protocol._forward_step(bound, batch)
    assert installation.bindings[0].stats.calls == 1
    installation.release()
