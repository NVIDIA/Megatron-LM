# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay the kernel boundaries reconciled by the full main-to-dev integration."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer
from megatron.core.transformer.experimental_attention_variant.dsa import _DSAWeightsProjection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_ddp_fp32_gradient_reduction_replays(use_distributed_optimizer, overlap):
    """Replay both all-reduce and reduce-scatter with native DDP gradient hooks."""
    if Utils.world_size < 2:
        pytest.skip("requires at least two data-parallel ranks")
    Utils.initialize_model_parallel()
    try:
        seeded()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        module = torch.nn.Linear(128, 128, bias=False).cuda().bfloat16()
        ddp = DistributedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=overlap,
                use_distributed_optimizer=use_distributed_optimizer,
            ),
            module=module,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        inputs = torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16)

        def reduce_gradients(value):
            ddp.zero_grad_buffer()
            ddp(value).float().square().mean().backward()
            ddp.finish_grad_sync()
            assert module.weight.main_grad.dtype == torch.float32
            return module.weight.main_grad.detach().clone()

        assert_replays_bit_exact(
            reduce_gradients,
            (inputs,),
            replays=3,
            backward=False,
            contention=True,
            what="DDP FP32 gradient buffer reduction",
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("rank", range(4))
def test_balanced_route_metadata_replays(rank, monkeypatch):
    """Rebuild, rather than cache-hit, integer bincount/sort route metadata on every replay."""
    group = SimpleNamespace(size=lambda: 4, rank=lambda: rank)
    for name in ("_LAST_PLAN", "_SEEN_CU", "_ZZ_PACK_OK"):
        monkeypatch.setattr(cp_balanced_indexer, name, {})
    boundaries = torch.tensor([0, 8192, 8192, 12288, 16384], device="cuda", dtype=torch.int32)

    def build(cu):
        cp_balanced_indexer._LAST_PLAN.clear()
        packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
        cp_balanced_indexer.prebuild_balanced_layouts(packed, cp_group=group)
        plan = packed._dsa_cp_balance_layout_cache[("zigzag", rank)]
        assert sum(plan["disp_in_splits"]) == 4096
        assert sum(plan["disp_out_splits"]) == 4096
        return {
            **{
                key: plan[key]
                for key in (
                    "gather_idx",
                    "inv_idx",
                    "disp_send_rows",
                    "disp_recv_rows",
                    "cmb_send_rows",
                    "cmb_recv_rows",
                )
            },
            "counts": torch.tensor(plan["disp_out_splits"], device=cu.device),
        }

    assert_replays_bit_exact(
        build,
        (boundaries,),
        replays=3,
        backward=False,
        contention=True,
        what="balanced route metadata (not balanced indexer compute)",
    )


@pytest.mark.parametrize("te_gemm_supported", [None, False], ids=["dispatch", "fp32_fallback"])
def test_dsa_weights_projection_replays(te_gemm_supported):
    """Replay BF16 operands with FP32 output plus input and weight gradients."""
    seeded()
    inputs = torch.randn(4096, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def project(value, projection):
        linear = SimpleNamespace(fuse_wgrad_accumulation=False)
        indexer = SimpleNamespace(_weights_proj_te_gemm_supported=te_gemm_supported)
        result = _DSAWeightsProjection.apply(value, projection, linear, indexer)
        assert result.dtype == torch.float32
        return result

    assert_replays_bit_exact(
        project,
        (inputs, weight),
        replays=3,
        backward=True,
        contention=True,
        what="DSA weights FP32 projection",
    )
