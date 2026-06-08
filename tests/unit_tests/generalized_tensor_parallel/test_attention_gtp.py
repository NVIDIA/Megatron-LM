# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration tests for GTP + Attention (TransformerLayer) correctness.

Test groups
-----------
TestAttentionGTPCorrectness  - GTP TransformerLayer loss trajectory matches baseline (no-GTP)
                               over 10 training steps using MXFP8 and Nemotron3-Super proxy
                               hyperparameters.
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
# Attention GTP correctness: per-step loss trajectory baseline vs GTP=4
# ---------------------------------------------------------------------------


def _worker_attention_gtp_correctness(rank, world_size, port):
    """Verify GTP TransformerLayer produces the same per-step loss as a no-GTP baseline.

    Phase 1 — GTP=1, DP=4:
        All 4 ranks hold the full model and process identical inputs.  Gradients
        are identical across ranks (no all-reduce needed).  Weight update:
            param.data -= lr * param.grad

    Phase 2 — GTP=4, DP=1:
        All linear weights (QKV proj, output proj, MLP fc1/fc2) sharded across
        4 ranks.  After backward, wgrad reduce-scatter sums each shard's wgrad:
            main_grad[rank_i] = gtp_size * dW[shard_i]
        The optimizer divides by gtp_size to recover the per-element gradient:
            param.data -= (lr / gtp_size) * param.main_grad

    Both phases use identical initial weights (synced from rank 0 in Phase 1,
    restored as shards in Phase 2) and identical step-by-step inputs.

    Nemotron3-Super proxy hyperparameters:
        hidden=4096, num_heads=32 (head_dim=128), ffn_hidden_size=16384 (=4xhidden)
    MXFP8 alignment with GTP=4:
        QKV shard: 3x4096/4=3072, 3072%32=0 ✓; proj shard: 4096/4=1024, 1024%32=0 ✓
        fc1 shard: 16384/4=4096, 4096%32=0 ✓; fc2 shard: 4096/4=1024, 1024%32=0 ✓
    """
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    HIDDEN = 4096
    NUM_HEADS = 32       # head_dim = HIDDEN / NUM_HEADS = 128
    FFN_HIDDEN = 16384   # = 4 x HIDDEN (default GPT FFN ratio)
    NUM_LAYERS = 2
    SEQ = 32
    BATCH = 1
    LR = 0.01
    STEPS = 10
    dtype = torch.bfloat16
    recipe = MXFP8BlockScaling()

    def make_config():
        return TransformerConfig(
            num_attention_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN,
            ffn_hidden_size=FFN_HIDDEN,
            add_bias_linear=False,
            params_dtype=dtype,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bias_dropout_fusion=False,
            fp8='e4m3',
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    def make_transformer_stack(config, pg_collection):
        spec = get_gpt_layer_with_transformer_engine_spec()
        return torch.nn.ModuleList([
            spec.module(config, spec.submodules, layer_number=i + 1, pg_collection=pg_collection)
            for i in range(NUM_LAYERS)
        ])

    def run_step(layers, x):
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            for layer in layers:
                x, _ = layer(x, attention_mask=None)
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
    layers = make_transformer_stack(config, pg_collection)
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
    layers_gtp = make_transformer_stack(config, pg_collection)
    for layer in layers_gtp:
        layer.cuda()

    gtp_group = ps.get_generalized_tensor_parallel_remat_group()
    gtp_size = gtp_group.size()
    gtp_rank = gtp_group.rank()

    # Verify GTP is truly active: linear weights must be GTPShardedParam instances.
    gtp_params = [p for p in layers_gtp.parameters() if isinstance(p, GTPShardedParam)]
    assert len(gtp_params) > 0, (
        "GTP is not active: no GTPShardedParam found in GTP=4 transformer stack"
    )

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

        # After RS, main_grad = gtp_size * dW_shard.  Divide by gtp_size to match baseline.
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


class TestAttentionGTPCorrectness:
    def test_attention_gtp_loss_trajectory_matches_baseline(self):
        """GTP TransformerLayer per-step losses must match no-GTP baseline (atol=1e-5, rtol=1e-5; MXFP8, Nemotron3-Super proxy)."""
        _requires_mxfp8()
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_attention_gtp_correctness, 4)
