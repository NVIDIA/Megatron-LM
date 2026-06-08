# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration tests for GTP + Mamba correctness.

Test groups
-----------
TestMambaGTPCorrectness  - GTP Mamba loss trajectory matches baseline (no-GTP) over 10
                           training steps using MXFP8 and Nemotron3-Super Mamba hyperparameters.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.experimental.gtp import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.17", allow_module_level=True)

from transformer_engine.pytorch import fp8_autocast

from megatron.experimental.gtp import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _requires_mxfp8,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

# ---------------------------------------------------------------------------
# Mamba GTP correctness: per-step loss trajectory baseline vs GTP=4
# ---------------------------------------------------------------------------


def _worker_mamba_gtp_correctness(rank, world_size, port):
    """Verify GTP Mamba produces the same per-step loss as a no-GTP baseline.

    Phase 1 — GTP=1, DP=4:
        All 4 ranks hold the full model and process identical inputs.  Gradients
        are identical across ranks (no all-reduce needed).  Weight update:
            param.data -= lr * param.grad

    Phase 2 — GTP=4, DP=1:
        Weights sharded across 4 ranks.  After backward, wgrad reduce-scatter
        sums each shard's identical wgrad over all ranks, so:
            main_grad[rank_i] = gtp_size * dW[shard_i]
        The optimizer divides by gtp_size to recover the per-element gradient:
            param.data -= (lr / gtp_size) * param.main_grad

    Both phases use identical initial weights (synced from rank 0 in phase 1,
    restored as shards in phase 2) and identical step-by-step inputs.  The
    per-step loss trajectories must agree within 0.1% relative error.
    """
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    from megatron.core import parallel_state as ps
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
    from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Nemotron3-Super Proxy Mamba hyperparameters.
    # in_proj_out = 2*8192 + 2*8*128 + 128 = 18560; 18560/4 = 4640, 4640%16 = 0 (MXFP8-aligned).
    HIDDEN = 4096
    NHEADS = 128       # mamba_num_heads; d_inner = nheads * headdim = 128 * 64 = 8192
    NGROUPS = 8        # mamba_num_groups (default)
    D_STATE = 128      # mamba_state_dim (default)
    NUM_LAYERS = 2
    SEQ = 32
    BATCH = 1
    LR = 0.01
    STEPS = 10
    dtype = torch.bfloat16
    recipe = MXFP8BlockScaling()

    def make_config():
        return TransformerConfig(
            num_attention_heads=32,
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN,
            mamba_num_heads=NHEADS,
            mamba_head_dim=64,
            mamba_state_dim=D_STATE,
            mamba_num_groups=NGROUPS,
            use_mamba_mem_eff_path=True,
            params_dtype=dtype,
            hidden_dropout=0.0,
            bias_dropout_fusion=False,
            fp8='e4m3',
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    def make_mamba_stack(config, pg_collection):
        submodules = MambaLayerSubmodules(
            mixer=ModuleSpec(
                module=MambaMixer,
                submodules=MambaMixerSubmodules(
                    in_proj=TELayerNormColumnParallelLinear,
                    out_proj=TERowParallelLinear,
                ),
            ),
            mamba_bda=get_bias_dropout_add,
        )
        return torch.nn.ModuleList(
            [
                MambaLayer(config, submodules, layer_number=i + 1, pg_collection=pg_collection)
                for i in range(NUM_LAYERS)
            ]
        )

    def run_step(layers, x):
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            for layer in layers:
                x = layer(x)
        return x.mean()

    # -------------------------------------------------------------------------
    # Phase 1: Baseline — GTP=1 (DP=4)
    # -------------------------------------------------------------------------
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
    )
    model_parallel_cuda_manual_seed(42)

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp'])
    config = make_config()
    layers = make_mamba_stack(config, pg_collection)
    for layer in layers:
        layer.cuda()

    # Verify baseline has no GTP sharding (gtp_remat_size=1 should leave plain parameters).
    assert not any(
        isinstance(p, GTPShardedParam) for p in layers.parameters()
    ), "Baseline GTP=1 stack should have no GTPShardedParam"

    # Synchronize weights from rank 0 across all DP ranks.
    for p in layers.parameters():
        dist.broadcast(p.data, src=0)

    # Save initial weights; will be used to initialize the GTP model identically.
    saved_weights = {n: p.data.clone() for n, p in layers.named_parameters()}

    baseline_losses = []
    for step in range(STEPS):
        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)

        loss = run_step(layers, x)
        if rank == 0:
            baseline_losses.append(loss.item())

        loss.backward()
        with torch.no_grad():
            for p in layers.parameters():
                if p.grad is not None:
                    p.data.sub_(LR * p.grad)
                    p.grad.zero_()

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}
    FP8GlobalStateManager.reset()

    # -------------------------------------------------------------------------
    # Phase 2: GTP=4 (DP=1)
    # -------------------------------------------------------------------------
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=4
    )
    model_parallel_cuda_manual_seed(42)

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp'])
    config = make_config()
    layers_gtp = make_mamba_stack(config, pg_collection)
    for layer in layers_gtp:
        layer.cuda()

    gtp_group = ps.get_generalized_tensor_parallel_remat_group()
    gtp_size = gtp_group.size()
    gtp_rank = gtp_group.rank()

    # Verify GTP is truly active: at least one param must be a GTPShardedParam.
    gtp_params = [p for p in layers_gtp.parameters() if isinstance(p, GTPShardedParam)]
    assert len(gtp_params) > 0, "GTP is not active: no GTPShardedParam found in GTP=4 Mamba stack"

    # Restore initial weights: GTP params get the matching shard, others get the full tensor.
    for name, p in layers_gtp.named_parameters():
        full = saved_weights[name]
        if isinstance(p, GTPShardedParam):
            shard_size = p.shape[0]
            p.data.copy_(full[gtp_rank * shard_size: (gtp_rank + 1) * shard_size])
        else:
            p.data.copy_(full)

    # Pre-allocate main_grad for GTP params (required before the first backward).
    for p in layers_gtp.parameters():
        if isinstance(p, GTPShardedParam):
            p.main_grad = torch.zeros(p.shape, dtype=dtype, device='cuda')

    gtp_losses = []
    for step in range(STEPS):
        for p in layers_gtp.parameters():
            if isinstance(p, GTPShardedParam):
                p.main_grad.zero_()

        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)

        loss = run_step(layers_gtp, x)
        if rank == 0:
            gtp_losses.append(loss.item())

        loss.backward()

        # After RS, main_grad = gtp_size * dW_shard (sum over ranks, all ranks hold the same
        # full wgrad after all-gathering the weight in fwd).  Divide by gtp_size so the weight
        # update is equivalent to the baseline.
        with torch.no_grad():
            for p in layers_gtp.parameters():
                if isinstance(p, GTPShardedParam):
                    p.data.sub_((LR / gtp_size) * p.main_grad)
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
        assert len(gtp_losses) == STEPS
        for step, (lb, lg) in enumerate(zip(baseline_losses, gtp_losses)):
            print(f"Step {step:2d}: baseline={lb:.6f}  gtp={lg:.6f}", flush=True)
        torch.testing.assert_close(
            torch.tensor(gtp_losses), torch.tensor(baseline_losses), atol=1e-5, rtol=1e-5
        )


class TestMambaGTPCorrectness:
    def test_mamba_gtp_loss_trajectory_matches_baseline(self):
        """GTP Mamba per-step losses must match no-GTP baseline (atol=1e-5, rtol=1e-5; MXFP8, Nemotron3-Super)."""
        _requires_mxfp8()
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_mamba_gtp_correctness, 4)
