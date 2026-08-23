# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration test for GTP correctness.

Validates that GTP run as a first-class parallelism axis
(world_size = TP * GTP * CP * DP) produces the same per-step loss as a no-GTP
baseline. This is the end-to-end proof that the standalone-GTP rank grid built
in parallel_state trains correctly.

Mirrors TestAttentionGTPCorrectness. With world=4 and gtp_remat_size=4, GTP
yields dp_replicate=1 and a single shard group [0,1,2,3], so the loss must match
the GTP_remat_size=1 baseline.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from transformer_engine.pytorch import fp8_autocast

from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401  (autouse, module-scoped: initializes the dist PG); noqa: F401  (autouse)
    _assert_loss_trajectories_match,
    _restore_gtp_shards_and_init_main_grad,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


def _worker_gtp_loss_correctness(rank, world_size, port):
    """Baseline (GTP_remat_size=1, DP=4) vs GTP_remat_size=4 (world=TP1*GTP4*CP1*DP1)."""
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    HIDDEN = 4096
    NUM_HEADS = 32
    FFN_HIDDEN = 16384
    NUM_LAYERS = 2
    SEQ = 32
    BATCH = 1
    LR = 0.01
    STEPS = 10
    dtype = torch.bfloat16

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
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    def make_transformer_stack(config, pg_collection):
        spec = get_gpt_layer_with_transformer_engine_spec()
        return torch.nn.ModuleList(
            [
                spec.module(
                    config, spec.submodules, layer_number=i + 1, pg_collection=pg_collection
                )
                for i in range(NUM_LAYERS)
            ]
        )

    def run_step(layers, x):
        with fp8_autocast(enabled=False):
            for layer in layers:
                x, _ = layer(x, attention_mask=None)
        return x.mean()

    # ---- Phase 1: Baseline — GTP_remat=1 (DP=4) ----
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
    )
    model_parallel_cuda_manual_seed(42)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['tp', 'cp', 'gtp_remat']
    )
    config = make_config()
    layers = make_transformer_stack(config, pg_collection)
    for layer in layers:
        layer.cuda()
    for p in layers.parameters():
        dist.broadcast(p.data, src=0)
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

    # ---- Phase 2: GTP_remat=4 (world = TP1 * GTP4 * CP1 * DP1) ----
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        gtp_remat_size=4,  # standalone-axis GTP_remat under test
    )
    model_parallel_cuda_manual_seed(42)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['tp', 'cp', 'gtp_remat']
    )
    config = make_config()
    layers_gtp = make_transformer_stack(config, pg_collection)
    for layer in layers_gtp:
        layer.cuda()

    gtp_remat_group = ps.get_gtp_weight_remat_group()
    gtp_remat_size = gtp_remat_group.size()
    gtp_rank = gtp_remat_group.rank()
    assert gtp_remat_size == 4, f"GTP shard group size should be 4, got {gtp_remat_size}"

    gtp_params = [p for p in layers_gtp.parameters() if isinstance(p, GTPShardedParam)]
    assert len(gtp_params) > 0, "GTP not active: no GTPShardedParam found"

    _restore_gtp_shards_and_init_main_grad(layers_gtp, saved_weights, gtp_rank, dtype)

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
        with torch.no_grad():
            for p in layers_gtp.parameters():
                if isinstance(p, GTPShardedParam):
                    p.data.sub_((LR / gtp_remat_size) * p.main_grad)
                elif p.grad is not None:
                    p.data.sub_(LR * p.grad)
                    p.grad.zero_()

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()
    GTPShardedParam._chain_state = {}

    if rank == 0:
        _assert_loss_trajectories_match(baseline_losses, gtp_losses, STEPS)


class TestGTPLossCorrectness:
    def test_gtp_loss_trajectory_matches_baseline(self):
        """GTP_remat_size=4 per-step losses must match no-GTP baseline (atol=1e-5, rtol=1e-5)."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_gtp_loss_correctness, 4)
