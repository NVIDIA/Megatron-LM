# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration test for GTP correctness with local partial CUDA graphs.

This is the local-CUDA-graph counterpart of ``test_gtp_loss_correctness.py``.
It compares an eager GTP=1 baseline with a GTP=4 run that captures only
attention and verifies the complete loss trajectory across graph capture and replay.
"""

import copy
import gc

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import (
    HAVE_GTP,
    GTPChain,
    classify_gtp_remat_chains,
    wait_for_gtp_grad_reduction_on_current_stream,
)

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.17", allow_module_level=True)

from transformer_engine.pytorch import fp8_autocast

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module
from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _assert_loss_trajectories_match,
    _restore_gtp_shards_and_init_main_grad,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


def _worker_gtp_partial_cg_loss_correctness(rank, world_size, port):
    """Compare eager GTP=1 with GTP=4 and local attention CUDA graphs."""
    del world_size, port

    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import (
        initialize_rng_tracker,
        model_parallel_cuda_manual_seed,
    )
    from megatron.core.transformer.cuda_graphs import (
        _CudagraphGlobalRecord,
        create_cudagraphs,
        delete_cuda_graphs,
    )
    from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
    from megatron.core.transformer.transformer_config import TransformerConfig

    hidden = 4096
    num_heads = 32
    ffn_hidden = 16384
    # Four layers force parameters with matching scheduling domains/shapes to
    # reuse the two-slot wgrad ring across independently replayed graphs.
    num_layers = 4
    sequence_length = 32
    batch_size = 1
    learning_rate = 0.01
    steps = 10
    dtype = torch.bfloat16

    def make_config(*, partial_cg=False):
        return TransformerConfig(
            num_attention_heads=num_heads,
            num_layers=num_layers,
            hidden_size=hidden,
            ffn_hidden_size=ffn_hidden,
            add_bias_linear=False,
            params_dtype=dtype,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bias_dropout_fusion=False,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            gtp_weight_remat_size=4 if partial_cg else 1,
            cuda_graph_impl="local" if partial_cg else "none",
            cuda_graph_modules=["attn"] if partial_cg else [],
            cuda_graph_warmup_steps=2,
        )

    def make_attention_stack(config, pg_collection):
        spec = copy.deepcopy(get_gpt_layer_with_transformer_engine_spec())
        spec.submodules.pre_mlp_layernorm = IdentityOp
        spec.submodules.mlp = IdentityOp
        spec.submodules.mlp_bda = IdentityFuncOp
        return torch.nn.ModuleList(
            [
                spec.module(
                    config, spec.submodules, layer_number=i + 1, pg_collection=pg_collection
                )
                for i in range(num_layers)
            ]
        )

    def run_step(layers, x):
        with fp8_autocast(enabled=False):
            for layer in layers:
                x, _ = layer(x, attention_mask=None)
        return x.mean()

    # Baseline: eager GTP=1 (DP=4).
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=1
    )
    model_parallel_cuda_manual_seed(42)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp", "cp", "gtp_remat"]
    )
    baseline_config = make_config()
    baseline = make_attention_stack(baseline_config, pg_collection).cuda()
    for param in baseline.parameters():
        dist.broadcast(param.data, src=0)
    saved_weights = {name: param.data.clone() for name, param in baseline.named_parameters()}

    baseline_losses = []
    for step in range(steps):
        torch.manual_seed(step)
        x = torch.randn(
            sequence_length, batch_size, hidden, dtype=dtype, device="cuda"
        )
        dist.broadcast(x, src=0)
        x.requires_grad_()
        loss = run_step(baseline, x)
        if rank == 0:
            baseline_losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            for param in baseline.parameters():
                if param.grad is not None:
                    param.data.sub_(learning_rate * param.grad)
                    param.grad.zero_()

    del baseline, loss, x
    ps.destroy_model_parallel()
    gtp_module.reset_gtp_state()

    # Optimized path: GTP=4 with attention-only local CUDA graphs.
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        gtp_remat_size=4,
    )
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(42)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp", "cp", "gtp_remat"]
    )
    partial_cg_config = make_config(partial_cg=True)
    partial_cg = make_attention_stack(partial_cg_config, pg_collection).cuda()
    classify_gtp_remat_chains(
        partial_cg,
        cuda_graph_modules=partial_cg_config.cuda_graph_modules,
        cuda_graph_impl=partial_cg_config.cuda_graph_impl,
    )

    gtp_group = ps.get_gtp_weight_remat_group()
    gtp_size = gtp_group.size()
    gtp_rank = gtp_group.rank()
    assert gtp_size == 4
    gtp_params = [
        param for param in partial_cg.parameters() if isinstance(param, GTPShardedParam)
    ]
    assert gtp_params, "GTP not active: no GTPShardedParam found"
    assert all(param.chain_id == GTPChain.GRAPHED.value for param in gtp_params)
    _restore_gtp_shards_and_init_main_grad(partial_cg, saved_weights, gtp_rank, dtype)
    # Production captures after DDP has mapped every parameter into a main-grad
    # buffer and initialized the fused-accumulation marker. Mirror those invariants
    # without pulling the full DDP stack into this focused GTP/CUDA-graph test.
    for param in partial_cg.parameters():
        if not hasattr(param, "main_grad"):
            param.main_grad = torch.zeros_like(param)
        param.grad_added_to_main_grad = False

    original_cross_cg_overlap = gtp_module.GTP_CONFIG.cross_cg_overlap
    gtp_module.GTP_CONFIG.cross_cg_overlap = True
    partial_cg_losses = []
    try:
        for step in range(steps):
            for param in partial_cg.parameters():
                param.main_grad.zero_()
                param.grad = None
                # DDP resets this before every local-CG training iteration.
                param.grad_added_to_main_grad = False
            torch.manual_seed(step)
            x = torch.randn(
                sequence_length,
                batch_size,
                hidden,
                dtype=dtype,
                device="cuda",
            )
            dist.broadcast(x, src=0)
            x.requires_grad_()
            loss = run_step(partial_cg, x)
            if rank == 0:
                partial_cg_losses.append(loss.item())
            loss.backward()
            wait_for_gtp_grad_reduction_on_current_stream()

            if step == 0:
                create_cudagraphs()
                assert _CudagraphGlobalRecord.cudagraph_created
                assert all(
                    layer.cudagraph_manager.cudagraph_runners[0].gtp_remat
                    for layer in partial_cg
                )

            with torch.no_grad():
                for param in partial_cg.parameters():
                    if isinstance(param, GTPShardedParam):
                        param.data.sub_((learning_rate / gtp_size) * param.main_grad)
                    else:
                        grad = param.grad if param.grad is not None else param.main_grad
                        param.data.sub_(learning_rate * grad)
                        param.grad = None
        del loss, x
    finally:
        torch.cuda.synchronize()
        for layer in partial_cg:
            for runner in layer.cudagraph_manager.cudagraph_runners:
                if runner.fwd_graph is not None:
                    runner.fwd_graph.reset()
                if runner.bwd_graph is not None:
                    runner.bwd_graph.reset()
        delete_cuda_graphs()
        for layer in partial_cg:
            layer.cudagraph_manager.cudagraph_runners.clear()
        gc.collect()
        gtp_module.GTP_CONFIG.cross_cg_overlap = original_cross_cg_overlap
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()
        gtp_module.reset_gtp_state()

    if rank == 0:
        _assert_loss_trajectories_match(
            baseline_losses,
            partial_cg_losses,
            steps,
            label="gtp_remat_partial_cg",
        )


class TestGTPPartialCGCorrectness:
    def test_gtp_partial_cg_loss_trajectory_matches_baseline(self):
        """GTP local-CG losses must match the eager no-GTP baseline."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires at least 4 CUDA devices")
        _run_distributed(_worker_gtp_partial_cg_loss_correctness, 4)
