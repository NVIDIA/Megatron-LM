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

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from transformer_engine.pytorch import fp8_autocast, fp8_model_init

from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _assert_loss_trajectories_match,
    _requires_mxfp8,
    _restore_gtp_shards_and_init_main_grad,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

# ---------------------------------------------------------------------------
# Mamba GTP_remat correctness: per-step loss trajectory baseline vs GTP_remat=4
# ---------------------------------------------------------------------------


def _worker_mamba_gtp_correctness(rank, world_size, port):
    """Verify GTP Mamba produces the same per-step loss as a no-GTP baseline.

    Phase 1 — GTP_remat_size=1, DP=4:
        All 4 ranks hold the full model and process identical inputs.  Gradients
        are identical across ranks (no all-reduce needed).  Weight update:
            param.data -= lr * param.grad

    Phase 2 — GTP_remat_size=4, DP=1:
        Weights sharded across 4 ranks.  After backward, wgrad reduce-scatter
        sums each shard's identical wgrad over all ranks, so:
            main_grad[rank_i] = gtp_remat_size * dW[shard_i]
        The optimizer divides by gtp_remat_size to recover the per-element gradient:
            param.data -= (lr / gtp_remat_size) * param.main_grad

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
    NHEADS = 128  # mamba_num_heads; d_inner = nheads * headdim = 128 * 64 = 8192
    NGROUPS = 8  # mamba_num_groups (default)
    D_STATE = 128  # mamba_state_dim (default)
    NUM_LAYERS = 2
    SEQ = 32
    BATCH = 1
    LR = 0.01
    STEPS = 10
    dtype = torch.bfloat16
    recipe = MXFP8BlockScaling()  # native-FP8 Phase 3 (Phases 1-2 run in BF16)

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
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    def make_mamba_stack(config, pg_collection):
        submodules = MambaLayerSubmodules(
            mixer=ModuleSpec(
                module=MambaMixer,
                submodules=MambaMixerSubmodules(
                    in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
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
        with fp8_autocast(enabled=False):
            for layer in layers:
                x = layer(x)
        return x.mean()

    # -------------------------------------------------------------------------
    # Phase 1: Baseline — GTP_remat=1 (DP=4)
    # -------------------------------------------------------------------------
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
    )
    model_parallel_cuda_manual_seed(42)

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['tp', 'cp', 'gtp_remat']
    )
    config = make_config()
    layers = make_mamba_stack(config, pg_collection)
    for layer in layers:
        layer.cuda()

    # Verify baseline has no GTP_remat sharding (gtp_remat_size=1 should leave plain parameters).
    assert not any(
        isinstance(p, GTPShardedParam) for p in layers.parameters()
    ), "Baseline GTP_remat_size=1 stack should have no GTPShardedParam"

    # Synchronize weights from rank 0 across all DP ranks.
    for p in layers.parameters():
        dist.broadcast(p.data, src=0)

    # Save initial weights; will be used to initialize the GTP_remat model identically.
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
    # Phase 2: GTP_remat=4 (DP=1)
    # -------------------------------------------------------------------------
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=4
    )
    model_parallel_cuda_manual_seed(42)

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['tp', 'cp', 'gtp_remat']
    )
    config = make_config()
    layers_gtp = make_mamba_stack(config, pg_collection)
    for layer in layers_gtp:
        layer.cuda()

    gtp_remat_group = ps.get_gtp_weight_remat_group()
    gtp_remat_size = gtp_remat_group.size()
    gtp_rank = gtp_remat_group.rank()

    # Verify GTP_remat is truly active: at least one param must be a GTPShardedParam.
    gtp_params = [p for p in layers_gtp.parameters() if isinstance(p, GTPShardedParam)]
    assert (
        len(gtp_params) > 0
    ), "GTP is not active: no GTPShardedParam found in GTP_remat_size=4 Mamba stack"

    # Restore initial weights into shards and pre-allocate main_grad for the backward.
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

        # After RS, main_grad = gtp_remat_size * dW_shard (sum over ranks, all ranks hold the same
        # full wgrad after all-gathering the weight in fwd).  Divide by gtp_remat_size so the weight
        # update is equivalent to the baseline.
        with torch.no_grad():
            for p in layers_gtp.parameters():
                if isinstance(p, GTPShardedParam):
                    p.data.sub_((LR / gtp_remat_size) * p.main_grad)
                elif p.grad is not None:
                    p.data.sub_(LR * p.grad)
                    p.grad.zero_()

    ps.destroy_model_parallel()
    GTPShardedParam._chain_state = {}
    FP8GlobalStateManager.reset()

    # -------------------------------------------------------------------------
    # Phase 3: GTP_remat=4 (DP=1), NATIVE MXFP8 weights (--fp8-param-gather path).
    # in_proj/out_proj are built under fp8_model_init -> native MXFP8 GTP shards. FP8 params can't
    # be updated in place, so this leg keeps an FP32 master and re-quantizes each step via
    # gtp_native_fp8_load_context (same copy_ mechanism as checkpoint load). Loss must track the
    # BF16 baseline within MXFP8 noise; a frozen/miswired FP8 weight flattens or diverges it.
    # -------------------------------------------------------------------------
    from megatron.core.fp8_utils import is_float8tensor
    from megatron.core.tensor_parallel.gtp_api import gtp_native_fp8_load_context, is_gtp_param

    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=4
    )
    model_parallel_cuda_manual_seed(42)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['tp', 'cp', 'gtp_remat']
    )
    config = make_config()
    with fp8_model_init(enabled=True, recipe=recipe):
        layers_fp8 = make_mamba_stack(config, pg_collection)
    for layer in layers_fp8:
        layer.cuda()

    # Verify native-FP8 GTP is truly active: some weight must be a native FP8 GTP shard.
    native_fp8 = [p for p in layers_fp8.parameters() if is_gtp_param(p) and is_float8tensor(p)]
    assert len(native_fp8) > 0, "No native-FP8 GTP weight found in fp8_model_init mamba stack"

    # Init: FP32 master per param = the gtp_rank shard of the saved baseline weights (padded to the
    # native-FP8 shard size); FP8 params re-quantized from their master via the load context.
    masters = {}
    with torch.no_grad(), gtp_native_fp8_load_context(layers_fp8):
        for name, p in layers_fp8.named_parameters():
            full = saved_weights[name]
            if is_gtp_param(p):
                shard = p.shape[0]  # native-FP8 shard may include GTP alignment pad rows
                aligned = shard * gtp_remat_size
                if full.shape[0] < aligned:
                    full = torch.nn.functional.pad(full, (0, 0, 0, aligned - full.shape[0]))
                m = full[gtp_rank * shard : (gtp_rank + 1) * shard].float().clone()
            else:
                m = full.float().clone()
            masters[name] = m
            p.copy_(m.to(dtype))  # BF16->FP8 (inside the load context) or plain BF16 copy
    for p in layers_fp8.parameters():
        if is_gtp_param(p):
            p.main_grad = torch.zeros(p.shape, dtype=dtype, device='cuda')

    fp8_losses = []
    for step in range(STEPS):
        for p in layers_fp8.parameters():
            if is_gtp_param(p):
                p.main_grad.zero_()

        torch.manual_seed(step)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device='cuda')
        dist.broadcast(x, src=0)

        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            y = x
            for layer in layers_fp8:
                y = layer(y)
        loss = y.mean()
        if rank == 0:
            fp8_losses.append(loss.item())
        loss.backward()

        # Update FP32 masters (same math as bf16 Phase 2), then re-quantize FP8 shards from them.
        with torch.no_grad():
            for name, p in layers_fp8.named_parameters():
                if is_gtp_param(p):
                    masters[name].sub_((LR / gtp_remat_size) * p.main_grad.float())
                elif p.grad is not None:
                    masters[name].sub_(LR * p.grad.float())
                    p.grad.zero_()
            with gtp_native_fp8_load_context(layers_fp8):
                for name, p in layers_fp8.named_parameters():
                    p.copy_(masters[name].to(dtype))

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()
    GTPShardedParam._chain_state = {}
    FP8GlobalStateManager.reset()

    # -------------------------------------------------------------------------
    # Compare per-step loss trajectories on rank 0
    # -------------------------------------------------------------------------
    if rank == 0:
        _assert_loss_trajectories_match(baseline_losses, gtp_losses, STEPS)
        # Native-FP8 leg tracks the baseline within MXFP8 noise (looser than the bf16-vs-bf16 tol).
        import torch as _torch

        diff = (_torch.tensor(fp8_losses) - _torch.tensor(baseline_losses)).abs().max().item()
        assert diff < 0.2, (
            f"Native-FP8 GTP mamba loss diverges from BF16 baseline "
            f"(max per-step |diff|={diff:.4f}; fp8: {fp8_losses[0]:.3f}->{fp8_losses[-1]:.3f}, "
            f"bf16: {baseline_losses[0]:.3f}->{baseline_losses[-1]:.3f})."
        )


class TestMambaGTPCorrectness:
    def test_mamba_gtp_loss_trajectory_matches_baseline(self):
        """GTP Mamba per-step losses must match the no-GTP baseline for BOTH bf16 (Phase 2) and
        native-MXFP8 (Phase 3) weight-remat, within bf16 reduction / mxfp8 quantization noise."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _requires_mxfp8()  # Phase 3 builds native-FP8 mamba weights (fp8_model_init)
        _run_distributed(_worker_mamba_gtp_correctness, 4)
