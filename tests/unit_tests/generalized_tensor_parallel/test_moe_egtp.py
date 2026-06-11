# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration tests for EGTP + MoE correctness.

Test groups
-----------
TestMoEEGTPCorrectness  - EGTP MoE loss trajectory matches baseline (no-EGTP) over 10
                          training steps using MXFP8 and Nemotron3-Super MoE hyperparameters.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.experimental.gtp import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.17", allow_module_level=True)

from transformer_engine.pytorch import fp8_autocast

from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.experimental.gtp import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _requires_mxfp8,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

# ---------------------------------------------------------------------------
# MoE EGTP correctness: per-step loss trajectory EP=4 baseline vs EP=2+EGTP=2
# ---------------------------------------------------------------------------


def _worker_moe_egtp_correctness(rank, world_size, port):
    """Verify EP=2+EGTP=2 MoE produces the same per-step loss as an EP=4 no-EGTP baseline.

    Phase 1 — EP=4, EGTP=1:
        All 4 ranks form one EP group; each rank holds 2 full expert weights (8 total).
        All ranks receive the same MoE-layer input; alltoall dispatch routes each token
        to its assigned expert rank, so each rank computes a different token subset.
        Gradients are local to each expert's rank.  Weight update:
            param.data -= lr * param.grad

    Phase 2 — EP=2, EGTP=2:
        Two EP groups of 2 ranks, each EGTP-sharded over 2 ranks.  Expert weights
        are sharded along dim 0 within each EGTP group (shard = full_dim0 / egtp_size).
        After backward, wgrad reduce-scatter sums each shard's identical wgrad:
            main_grad[rank_i] = egtp_size * dW[shard_i]
        The optimizer divides by egtp_size:
            param.data -= (lr / egtp_size) * param.main_grad

    Weight sharing (test-only):
        To ensure both phases start from identical expert weights, an all-gather
        collects the full 8-expert table from the EP=4 group (where each rank holds
        only 2 experts) onto every rank.  Phase 2 then slices each rank's local
        experts and EGTP shard from that global table.

    Nemotron3-Super Proxy MoE hyperparameters (scaled for unit-test speed):
        hidden=4096, ffn_hidden_size=2688, num_experts=8, topk=2
    MXFP8 alignment with EGTP=2:
        2688/2=1344, 1344%16=0 (fc1 shard); 4096/2=2048, 2048%16=0 (fc2 shard)
    """
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Nemotron3-Super MoE hyperparameters (num_experts scaled from 512 to 8 for test speed).
    HIDDEN = 4096
    FFN_HIDDEN = 2688
    NUM_EXPERTS = 8
    TOPK = 2
    SEQ = 32
    BATCH = 1
    LR = 0.01
    STEPS = 10
    dtype = torch.bfloat16
    recipe = MXFP8BlockScaling()

    def make_config():
        return TransformerConfig(
            num_attention_heads=32,
            num_layers=1,
            hidden_size=HIDDEN,
            num_moe_experts=NUM_EXPERTS,
            moe_router_topk=TOPK,
            moe_ffn_hidden_size=FFN_HIDDEN,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.0,
            add_bias_linear=False,
            params_dtype=dtype,
            hidden_dropout=0.0,
            bias_dropout_fusion=False,
            fp8='e4m3',
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    def make_moe_layer(config, pg_collection):
        moe_spec = get_moe_module_spec(use_te=True, num_experts=NUM_EXPERTS, moe_grouped_gemm=True)
        return moe_spec(config, layer_number=1, pg_collection=pg_collection)

    def run_step(layer, x):
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            output, _ = layer(x)
        return output.mean()

    # -------------------------------------------------------------------------
    # Phase 1: Baseline — EP=4, EGTP=1 (DP=1)
    # -------------------------------------------------------------------------
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=4,
        expert_gtp_remat_size=1,
    )
    model_parallel_cuda_manual_seed(42)

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['ep'])
    ep_group = pg_collection.ep
    num_local_experts_baseline = NUM_EXPERTS // 4  # = 2

    config = make_config()
    layer = make_moe_layer(config, None)  # MoELayer uses get_default_pg_collection()
    layer.cuda()

    # Verify baseline has no GTP sharding (EGTP=1 should leave plain parameters).
    assert not any(
        isinstance(p, GTPShardedParam) for p in layer.parameters()
    ), "Baseline EP=4 layer should have no GTPShardedParam (EGTP=1)"

    # Synchronize non-expert weights from rank 0; expert weights are rank-local.
    for name, p in layer.named_parameters():
        if 'linear_fc1.weight' not in name and 'linear_fc2.weight' not in name:
            dist.broadcast(p.data, src=0)

    # Collect the full expert weight table so Phase 2 can restore identical init weights.
    # EP=4: each rank holds 2 experts; all-gather gives every rank the complete [8, dim, ...] table.
    local_fc1 = torch.stack(
        [
            dict(layer.named_parameters())[f'experts.linear_fc1.weight{i}'].data
            for i in range(num_local_experts_baseline)
        ]
    )  # [2, FFN_HIDDEN, HIDDEN]
    global_fc1 = torch.zeros(NUM_EXPERTS, FFN_HIDDEN, HIDDEN, dtype=dtype, device='cuda')
    dist.all_gather_into_tensor(global_fc1, local_fc1, group=ep_group)

    local_fc2 = torch.stack(
        [
            dict(layer.named_parameters())[f'experts.linear_fc2.weight{i}'].data
            for i in range(num_local_experts_baseline)
        ]
    )  # [2, HIDDEN, FFN_HIDDEN]
    global_fc2 = torch.zeros(NUM_EXPERTS, HIDDEN, FFN_HIDDEN, dtype=dtype, device='cuda')
    dist.all_gather_into_tensor(global_fc2, local_fc2, group=ep_group)

    # Save non-expert param values (router, norms, etc.) from rank 0.
    non_expert_weights = {}
    for name, p in layer.named_parameters():
        if 'linear_fc1.weight' not in name and 'linear_fc2.weight' not in name:
            non_expert_weights[name] = p.data.clone()

    baseline_losses = []
    for step in range(STEPS):
        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)

        loss = run_step(layer, x)
        if rank == 0:
            baseline_losses.append(loss.item())

        loss.backward()
        with torch.no_grad():
            for p in layer.parameters():
                if p.grad is not None:
                    p.data.sub_(LR * p.grad)
                    p.grad.zero_()

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}
    FP8GlobalStateManager.reset()

    # -------------------------------------------------------------------------
    # Phase 2: EP=2, EGTP=2 (DP=1 effective)
    # -------------------------------------------------------------------------
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        expert_gtp_remat_size=2,
    )
    model_parallel_cuda_manual_seed(42)

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['expt_gtp'])
    egtp_group = pg_collection.expt_gtp
    egtp_size = egtp_group.size()
    egtp_rank = egtp_group.rank()
    ep_rank_egtp = dist.get_rank(ps.get_expert_model_parallel_group())
    num_local_experts_egtp = NUM_EXPERTS // 2  # = 4

    config = make_config()
    # Build full pg_collection for MoELayer: default groups + expt_gtp for EGTP sharding.
    moe_pg = get_default_pg_collection()
    moe_pg.expt_gtp = egtp_group
    layer_egtp = make_moe_layer(config, moe_pg)
    layer_egtp.cuda()

    # Verify EGTP is truly active: expert weight params must be GTPShardedParam instances.
    egtp_params = [p for p in layer_egtp.parameters() if isinstance(p, GTPShardedParam)]
    assert len(egtp_params) > 0, "EGTP is not active: no GTPShardedParam found in EP=2+EGTP=2 layer"

    # Restore weights from saved global tables.
    # Expert local index j → global expert id = ep_rank_egtp * num_local_experts_egtp + j.
    fc1_shard = FFN_HIDDEN // egtp_size  # 2688/2 = 1344
    fc2_shard = HIDDEN // egtp_size  # 4096/2 = 2048
    for name, p in layer_egtp.named_parameters():
        if 'linear_fc1.weight' in name:
            j = int(name.rsplit('weight', 1)[1])
            gid = ep_rank_egtp * num_local_experts_egtp + j
            p.data.copy_(global_fc1[gid, egtp_rank * fc1_shard : (egtp_rank + 1) * fc1_shard])
        elif 'linear_fc2.weight' in name:
            j = int(name.rsplit('weight', 1)[1])
            gid = ep_rank_egtp * num_local_experts_egtp + j
            p.data.copy_(global_fc2[gid, egtp_rank * fc2_shard : (egtp_rank + 1) * fc2_shard])
        elif name in non_expert_weights:
            p.data.copy_(non_expert_weights[name])

    # Pre-allocate main_grad for EGTP params (required before the first backward).
    for p in layer_egtp.parameters():
        if isinstance(p, GTPShardedParam):
            p.main_grad = torch.zeros(p.shape, dtype=dtype, device='cuda')

    egtp_losses = []
    for step in range(STEPS):
        for p in layer_egtp.parameters():
            if isinstance(p, GTPShardedParam):
                p.main_grad.zero_()

        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)

        loss = run_step(layer_egtp, x)
        if rank == 0:
            egtp_losses.append(loss.item())

        loss.backward()

        # After RS, main_grad = egtp_size * dW_shard.  Divide by egtp_size to match baseline.
        with torch.no_grad():
            for p in layer_egtp.parameters():
                if isinstance(p, GTPShardedParam):
                    p.data.sub_((LR / egtp_size) * p.main_grad)
                elif p.grad is not None:
                    p.data.sub_(LR * p.grad)
                    p.grad.zero_()

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()
    GTPShardedParam._chain_state = {}

    # -------------------------------------------------------------------------
    # Compare per-step loss trajectories on rank 0
    # -------------------------------------------------------------------------
    if rank == 0:
        assert len(baseline_losses) == STEPS
        assert len(egtp_losses) == STEPS
        for step, (lb, le) in enumerate(zip(baseline_losses, egtp_losses)):
            print(f"Step {step:2d}: baseline={lb:.6f}  egtp={le:.6f}", flush=True)
        torch.testing.assert_close(
            torch.tensor(egtp_losses), torch.tensor(baseline_losses), atol=1e-5, rtol=1e-5
        )


class TestMoEEGTPCorrectness:
    def test_moe_egtp_loss_trajectory_matches_baseline(self):
        """EP=2+EGTP=2 MoE per-step losses must match EP=4 baseline: atol=1e-5, rtol=1e-5; MXFP8"""
        _requires_mxfp8()
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_moe_egtp_correctness, 4)
