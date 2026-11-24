# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Train through real TE graph capture/replay with recurrence offloading."""

from dataclasses import replace
from functools import partial

import pytest
import torch

from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.extensions.te_cuda_graphs import TECudaGraphMixin
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.tensor_parallel.random import initialize_rng_tracker
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.transformer_layer_mixins import TECudaGraphTransformerLayerMixin
from megatron.core.utils import is_te_min_version, is_torch_min_version, unwrap_model
from tests.unit_tests.ssm.test_gdn_offload_training import (
    _build_training_model,
    _run_microbatches,
    _step_snapshot,
    _training_batches,
    _training_config,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.skipif(
    not is_torch_min_version("2.9.0a0") or not is_te_min_version("2.14.0"),
    reason="CUDA graph offloading requires PyTorch >= 2.9 and TE >= 2.14.",
)
@pytest.mark.parametrize(
    "variant,fused,recompute_norm,whole_layer,delay,fraction,tp,pp,mixed",
    [
        pytest.param("gdn", False, False, False, False, 1.0, 1, 1, True, id="mixed-attn"),
        pytest.param("gdn", True, True, False, True, 1.0, 1, 1, False, id="fused-norm-delay"),
        pytest.param("gdn", False, False, True, False, 1.0, 1, 1, False, id="gdn-layer"),
        pytest.param("gdn2", False, True, True, True, 1.0, 1, 1, False, id="gdn2-layer"),
        pytest.param("gdn2", False, False, True, False, 0.0, 1, 1, False, id="resident"),
        pytest.param("gdn2", False, False, False, False, 1.0, 2, 1, False, id="tp-sp"),
        pytest.param("gdn2", False, False, True, True, 1.0, 1, 2, False, id="pp"),
    ],
)
def test_gdn_offload_with_te_graphs(
    variant: str,
    fused: bool,
    recompute_norm: bool,
    whole_layer: bool,
    delay: bool,
    fraction: float,
    tp: int,
    pp: int,
    mixed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare complete optimizer states after captured backward on changing inputs."""
    if Utils.world_size % (tp * pp):
        pytest.skip("This topology requires two GPUs.")
    if variant == "gdn2":
        pytest.importorskip("fla.ops.gdn2")
    if fused:
        pytest.importorskip("causal_conv1d", minversion="1.6.1")
        monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    Utils.initialize_model_parallel(tp, pp)
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    init_num_microbatches_calculator(
        rank=torch.distributed.get_rank(),
        global_batch_size=4 * (Utils.world_size // (tp * pp)),
        micro_batch_size=1,
        data_parallel_size=Utils.world_size // (tp * pp),
        decrease_batch_size_if_needed=False,
    )
    helper = None
    replayed_layers = set()
    replay = TECudaGraphMixin._te_cuda_graph_replay

    def record_replay(layer, *args, **kwargs):
        replayed_layers.add(id(layer))
        return replay(layer, *args, **kwargs)

    monkeypatch.setattr(TECudaGraphMixin, "_te_cuda_graph_replay", record_replay)
    replay_args = TECudaGraphTransformerLayerMixin._get_te_cuda_graph_replay_args
    mixed_scopes = set()

    def record_replay_args(layer, *args, **kwargs):
        inputs, metadata = replay_args(layer, *args, **kwargs)
        if mixed and layer.config.fine_grained_activation_offloading:
            recurrence = getattr(layer.self_attention, "offload_gdn_core_attention", False)
            assert ("cuda_graph_stream" in metadata) == (not recurrence)
            assert ("cuda_graph_event" in metadata) == (not recurrence)
            mixed_scopes.add("recurrence" if recurrence else "attention")
        return inputs, metadata

    monkeypatch.setattr(
        TECudaGraphTransformerLayerMixin, "_get_te_cuda_graph_replay_args", record_replay_args
    )
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        references = []
        for enabled in (False, True):
            off_interface.reset_instance()
            replayed_layers.clear()
            config = replace(
                _training_config(
                    tp,
                    tp > 1,
                    1,
                    pp,
                    None,
                    fraction,
                    recompute_norm,
                    fused_pre_gdr=fused,
                    variant=variant,
                ),
                fine_grained_activation_offloading=enabled,
                offload_modules=(
                    (["gdn_core_attn", "core_attn"] if mixed else ["gdn_core_attn"])
                    if enabled
                    else []
                ),
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=[] if whole_layer else [CudaGraphModule.attn],
                cuda_graph_warmup_steps=3,
                use_te_rng_tracker=True,
                delay_offload_until_cuda_graph=delay,
            )
            # Match native training: DDP's retained AccumulateGrad nodes must use a side stream.
            current_stream = torch.cuda.current_stream()
            init_stream = torch.cuda.Stream()
            init_stream.wait_stream(current_stream)
            with torch.cuda.stream(init_stream):
                models, optimizer = _build_training_model(config, groups)
            current_stream.wait_stream(init_stream)
            manager = PipelineOffloadManager.get_instance()
            pool_checks = []

            def finalize(*args, **kwargs):
                finalize_model_grads(*args, **kwargs)
                in_use = manager.cpu_tensor_pool.get_pool_status()["global_stats"]["current_in_use"]
                assert in_use == 0
                pool_checks.append(in_use)

            config.finalize_model_grads_func = partial(finalize, pg_collection=groups)
            schedule = get_forward_backward_func(pp_size=pp, vp_size=None)
            generator = torch.Generator().manual_seed(19 + groups.dp.rank())
            states = []
            for step in range(6):
                if step == 3:
                    helper = TECudaGraphHelper(
                        models, config, 128, 1, optimizers=[optimizer], pg_collection=groups
                    )
                    helper.create_cudagraphs()
                    assert helper.graphs_created()
                    assert all(
                        len(layer.te_cuda_graphs) == helper.num_microbatches
                        for layer in unwrap_model(models[0]).decoder.layers
                    )
                batches = _training_batches(generator, groups, 4)
                losses = _run_microbatches(schedule, models, optimizer, groups, batches)
                states.append(_step_snapshot(models, optimizer, losses))
            assert len(replayed_layers) == len(unwrap_model(models[0]).decoder.layers)
            if enabled and mixed:
                assert mixed_scopes == {"recurrence", "attention"}
            assert len(pool_checks) == 6
            if enabled:
                assert (manager.offload_summary_total_bytes > 0) == (fraction > 0)
                if fused:
                    torch.testing.assert_close(states, references, rtol=1e-5, atol=1e-8)
                else:
                    assert not any(diff(references, states))
            else:
                references = states
            helper.delete_cuda_graphs()
            helper = None
            del models, optimizer, states
            torch.cuda.synchronize()
    finally:
        if helper is not None and helper.graphs_created():
            helper.delete_cuda_graphs()
        torch.cuda.synchronize()
        off_interface.reset_instance()
        destroy_num_microbatches_calculator()
        initialize_rng_tracker(force_reset=True)
        Utils.destroy_model_parallel()
