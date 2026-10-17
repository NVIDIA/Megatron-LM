# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Train through real TE graph capture/replay with recurrence offloading."""

from dataclasses import replace
from functools import partial

import pytest
import torch

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace, diff
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.tensor_parallel.random import initialize_rng_tracker
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.utils import unwrap_model
from tests.unit_tests.ssm.test_gdn_offload_training import (
    _build_training_model,
    _forward_step,
    _snapshot_leaf,
    _training_config,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (1, 2)], ids=["tp1", "tp2-sp", "pp2"])
@pytest.mark.parametrize(
    "variant,fused,recompute_norm,whole_layer,delay",
    [
        ("gdn", False, False, False, False),
        ("gdn", True, True, False, True),
        ("gdn", True, False, True, False),
        ("gdn", False, True, True, True),
        ("gdn2", False, False, True, False),
        ("gdn2", False, True, False, True),
    ],
)
def test_gdn_offload_with_te_graphs(
    variant: str,
    fused: bool,
    recompute_norm: bool,
    whole_layer: bool,
    delay: bool,
    tp: int,
    pp: int,
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
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        references = []
        for enabled in (False, True):
            off_interface.reset_instance()
            config = replace(
                _training_config(
                    tp,
                    tp > 1,
                    1,
                    pp,
                    None,
                    1.0,
                    recompute_norm,
                    fused_pre_gdr=fused,
                    variant=variant,
                ),
                fine_grained_activation_offloading=enabled,
                offload_modules=["gdn_core_attn"] if enabled else [],
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
                        len(layer.cuda_graphs) == helper.num_microbatches
                        for layer in unwrap_model(models[0]).decoder.layers
                    )
                batches = []
                for _ in range(4):
                    tokens = torch.randint(0, 512, (1, 129), generator=generator).cuda()
                    batches.append(
                        {
                            "tokens": tokens[:, :-1].contiguous(),
                            "labels": tokens[:, 1:].contiguous(),
                            "position_ids": torch.arange(128, device="cuda").unsqueeze(0),
                        }
                    )
                optimizer.zero_grad()
                for model in models:
                    model.zero_grad_buffer()
                losses = schedule(
                    forward_step_func=_forward_step,
                    data_iterator=[iter(batches)],
                    model=models,
                    num_microbatches=4,
                    seq_length=128,
                    micro_batch_size=1,
                    forward_only=False,
                    p2p_communicator=P2PCommunicator(groups.pp, config),
                    pg_collection=groups,
                )
                gradients = {
                    name: p.main_grad.detach().clone() for name, p in models[0].named_parameters()
                }
                success, grad_norm, _ = optimizer.step()
                assert success and grad_norm > 0
                states.append(
                    {
                        "losses": losses,
                        "grads": gradients,
                        "grad_norm": grad_norm,
                        "weights": {
                            name: p.detach().clone() for name, p in models[0].named_parameters()
                        },
                        "optimizer": dict_list_map_outplace(_snapshot_leaf, optimizer.state_dict()),
                    }
                )
            assert len(pool_checks) == 6
            if enabled:
                assert manager.offload_summary_total_bytes > 0
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
