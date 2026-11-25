# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TE mHC graph parity through real DDP and pipeline schedules.

The default runs PP1. Run the identical checks through PP2/VPP2 with:

    MHC_CG_PP_SIZE=2 MHC_CG_VP_SIZE=2 torchrun --nproc-per-node=4 -m pytest \
        tests/unit_tests/transformer/test_mhc_te_pipeline_training.py

Combine TP/SP or fixed CP with pipeline and ordinary expert parallelism on eight GPUs:

    MHC_CG_PP_SIZE=2 MHC_CG_VP_SIZE=2 MHC_CG_TP_SIZE=2 MHC_CG_EP_SIZE=2 \
        torchrun --nproc-per-node=8 -m pytest -k hybrid_moe_mtp \
        tests/unit_tests/transformer/test_mhc_te_pipeline_training.py
    MHC_CG_PP_SIZE=2 MHC_CG_CP_SIZE=2 MHC_CG_EP_SIZE=2 \
        torchrun --nproc-per-node=8 -m pytest -k hybrid_moe_mtp \
        tests/unit_tests/transformer/test_mhc_te_pipeline_training.py

TP > 1 enables SP; expert TP equals TP. CP uses the official fixed-SBHD zigzag
batch split. Different DP replicas consume different deterministic token batches.

After three eager training warmups, capture real TE callables, then execute five
changed-input replay steps with synchronized gradients and optimizer updates.

The attention graph split cases add selective mHC recompute and compare against
eager training without recompute, including nonzero hidden dropout.
"""

from collections import Counter

import pytest
import torch

from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.tensor_parallel import random as tensor_parallel_random
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import (
    TECudaGraphHelper,
    _te_cuda_graph_layer_is_graphable,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_model_config, is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_mhc_full_iteration_training import (
    _NUM_MICROBATCHES,
    _SEQ_LENGTH,
    _TRAIN_STEPS,
    _WARMUP_STEPS,
    _assert_training_close,
    _build_models,
    _copy_parameters,
    _forward_microbatch,
    _initialize_test_parallelism,
    _named_parameters,
    _restore_rng,
    _snapshot_rng,
    _training_step,
    _update_and_snapshot,
)


def _unwrap(model):
    module = model.module
    return module.module if isinstance(module, Float16Module) else module


def _run_te_pipeline_training(models, groups, pp_size, use_graph, recompute_rebuilds=None):
    parameters = _named_parameters(models)
    optimizer = torch.optim.SGD(list(parameters.values()), lr=0.02, momentum=0.1)
    forward_backward = get_forward_backward_func()
    p2p_communicator = (
        P2PCommunicator(pp_group=groups.pp, config=get_model_config(models[0]))
        if pp_size > 1
        else None
    )
    gradients_seen = set()
    gradient_hooks = [
        parameter.register_hook(lambda gradient, name=name: gradients_seen.add(name))
        for name, parameter in parameters.items()
    ]
    helper = None
    replay_calls = Counter()
    replay_microbatches = {}
    iteration = -1

    def forward_step(data_iterator, model):
        return _forward_microbatch(data_iterator, model, groups, pp_size)

    def capture():
        config = _unwrap(models[0]).config
        graph_helper = TECudaGraphHelper(
            models, config, _SEQ_LENGTH, 1, optimizers=[optimizer], pg_collection=groups
        )
        expected_layers, expected_mtp = [], []
        for model in models:
            module = _unwrap(model)
            decoder_layers = [
                layer
                for layer in module.decoder.layers
                if _te_cuda_graph_layer_is_graphable(layer, config)
            ]
            expected_layers.extend(decoder_layers)
            expected_mtp.extend([False] * len(decoder_layers))
            for depth in getattr(getattr(module, "mtp", None), "layers", []):
                nested_layers = [
                    layer
                    for layer in getattr(depth.mtp_model_layer, "layers", (depth.mtp_model_layer,))
                    if _te_cuda_graph_layer_is_graphable(layer, config)
                ]
                expected_layers.extend(nested_layers)
                expected_mtp.extend([True] * len(nested_layers))
        assert graph_helper.flattened_callables == expected_layers and expected_layers
        assert graph_helper.flattened_callables_is_mtp == expected_mtp
        assert len({id(layer) for layer in expected_layers}) == len(expected_layers)
        graph_helper.create_cudagraphs()
        assert graph_helper.graphs_created()
        # DDP with overlap_param_gather=False has no manual parameter-gather
        # hooks. Its actual gradient accumulation/reduction hooks remain active.
        assert all(not model.use_forward_hook for model in models)
        for layer in expected_layers:
            assert len(layer.te_cuda_graphs) == graph_helper.num_microbatches
            original_graphs = list(layer.te_cuda_graphs)
            counted_graphs = []
            for graph_index, graph in enumerate(original_graphs):

                def counted(*args, _layer=layer, _index=graph_index, _graph=graph, **kwargs):
                    microbatch = _layer.te_cuda_graph_current_microbatch
                    assert _index == microbatch % len(_layer.te_cuda_graphs)
                    replay_calls[(iteration, id(_layer))] += 1
                    replay_microbatches.setdefault((iteration, id(_layer)), []).append(microbatch)
                    return _graph(*args, **kwargs)

                counted.reset = graph.reset
                counted_graphs.append(counted)
            layer.te_cuda_graphs = counted_graphs
        return graph_helper

    history = []
    try:
        for iteration in range(_TRAIN_STEPS):
            if use_graph and iteration == _WARMUP_STEPS:
                helper = capture()
            # TE helper sets is_graph_capturing during capture, which makes
            # MCore DDP leave parameter grads available to TE. With gradient
            # accumulation fusion disabled, TE returns those grads through
            # autograd on every replay, so the leaf hooks must run each step.
            gradients_seen.clear()
            losses = _training_step(
                forward_backward,
                forward_step,
                models,
                groups,
                p2p_communicator,
                optimizer,
                iteration,
            )
            if recompute_rebuilds is not None:
                recompute_rebuilds.append(_count_recompute_rebuilds.count)
                _count_recompute_rebuilds.count = 0
            assert gradients_seen == parameters.keys(), (
                f"Parameters without a gradient in iteration {iteration}: "
                f"{parameters.keys() - gradients_seen}"
            )
            history.append(_update_and_snapshot(parameters, optimizer, losses))
            if helper is not None:
                for layer in helper.flattened_callables:
                    key = (iteration, id(layer))
                    assert replay_calls[key] == _NUM_MICROBATCHES
                    assert replay_microbatches[key] == list(range(_NUM_MICROBATCHES))
        if use_graph:
            assert helper is not None and helper.graphs_created()
            assert sum(replay_calls.values()) == (
                (_TRAIN_STEPS - _WARMUP_STEPS) * _NUM_MICROBATCHES * len(helper.flattened_callables)
            )
        return history
    finally:
        for hook in gradient_hooks:
            hook.remove()
        if helper is not None and helper.graphs_created():
            helper.delete_cuda_graphs()


class _count_recompute_rebuilds:
    """Count output-discarding checkpoint rebuilds, i.e. selective recompute regions run."""

    count = 0

    def __init__(self, monkeypatch) -> None:
        rebuild = tensor_parallel_random._RecomputeRegion.rebuild

        def counted_rebuild(region):
            if region.state in (
                tensor_parallel_random._RegionState.RECORDED,
                tensor_parallel_random._RegionState.RELEASED,
            ):
                _count_recompute_rebuilds.count += 1
            return rebuild(region)

        _count_recompute_rebuilds.count = 0
        monkeypatch.setattr(tensor_parallel_random._RecomputeRegion, "rebuild", counted_rebuild)


def _train_eager_reference_and_candidate(model_kind, hidden_dropout, candidate_kwargs, monkeypatch):
    """Train eager mHC without recompute and a candidate from the same initial state.

    Attention dropout stays zero: hidden dropout is drawn eagerly in both runs, so they
    consume identical random streams when recomputation replays its forward masks.
    """
    parallel_sizes = _initialize_test_parallelism(model_kind, torch.bfloat16)
    pp_size = parallel_sizes["pp_size"]
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    # Parity needs IEEE FP32 GEMMs on both sides; CI images enable TF32 by default.
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234, te_rng_tracker=True, force_reset_rng=True)
        common = dict(
            dropout=hidden_dropout,
            attention_dropout=0.0,
            te_graph_modules=[CudaGraphModule.attn],
            use_te_rng_tracker=True,
            **parallel_sizes,
        )
        reference_models, groups = _build_models(model_kind, torch.bfloat16, "none", **common)
        candidate_models, _ = _build_models(
            model_kind, torch.bfloat16, **candidate_kwargs, **common
        )
        _copy_parameters(reference_models, candidate_models)
        initial_rng = _snapshot_rng()
        expected = _run_te_pipeline_training(reference_models, groups, pp_size, False)
        _restore_rng(initial_rng)
        rebuilds = []
        _count_recompute_rebuilds(monkeypatch)
        actual = _run_te_pipeline_training(
            candidate_models,
            groups,
            pp_size,
            candidate_kwargs["graph_impl"] == "transformer_engine",
            recompute_rebuilds=rebuilds,
        )
        # Every training step, including each graph replay step, recomputes mHC regions.
        assert len(rebuilds) == len(actual) and all(rebuilds), rebuilds
        _assert_training_close(actual, expected, torch.bfloat16)
        return candidate_models, groups, actual
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not is_te_min_version("2.10.0"), reason="TE graph reset requires TE >= 2.10")
@pytest.mark.parametrize(
    "model_kind,hidden_dropout",
    [("gpt_dense", 0.1), ("gpt_moe", 0.0), ("hybrid_dense", 0.0), ("hybrid_moe_mtp", 0.1)],
)
def test_mhc_te_attention_graph_split_recompute_training_parity(
    model_kind, hidden_dropout, monkeypatch
):
    """Attention-only graphs with selective mHC recompute match eager mHC training."""
    _train_eager_reference_and_candidate(
        model_kind,
        hidden_dropout,
        dict(graph_impl="transformer_engine", mhc_recompute=True, mhc_attention_graph_split=True),
        monkeypatch,
    )


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("model_kind", ["gpt_dense", "hybrid_dense"])
def test_mhc_recompute_replays_dropout_with_graph_safe_rng(model_kind, monkeypatch):
    """Eager selective mHC recompute under a graph-safe RNG tracker replays its masks."""
    _train_eager_reference_and_candidate(
        model_kind, 0.1, dict(graph_impl="none", mhc_recompute=True), monkeypatch
    )


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not is_te_min_version("2.10.0"), reason="TE graph reset requires TE >= 2.10")
@pytest.mark.parametrize("model_kind", ["gpt_moe", "hybrid_dense", "hybrid_moe_mtp"])
def test_mhc_te_pipeline_training_parity(model_kind):
    """Real TE graph replay matches eager through the selected PP/VPP schedule."""
    parallel_sizes = _initialize_test_parallelism(model_kind, torch.bfloat16)
    pp_size = parallel_sizes["pp_size"]
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    # Parity needs IEEE FP32 GEMMs on both sides; CI images enable TF32 by default.
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234, te_rng_tracker=True, force_reset_rng=True)
        scopes = [CudaGraphModule.attn]
        scopes.append(CudaGraphModule.moe_router if "moe" in model_kind else CudaGraphModule.mlp)
        eager_models, groups = _build_models(
            model_kind,
            torch.bfloat16,
            "none",
            dropout=0.0,
            te_graph_modules=scopes,
            **parallel_sizes,
        )
        graph_models, _ = _build_models(
            model_kind,
            torch.bfloat16,
            "transformer_engine",
            dropout=0.0,
            te_graph_modules=scopes,
            **parallel_sizes,
        )
        _copy_parameters(eager_models, graph_models)
        graph_parameters = _named_parameters(graph_models)
        initial_keys = [set(model.state_dict()) for model in graph_models]
        initial_parameters = {
            name: parameter.detach().cpu().clone() for name, parameter in graph_parameters.items()
        }
        initial_rng = _snapshot_rng()
        expected = _run_te_pipeline_training(eager_models, groups, pp_size, False)
        _restore_rng(initial_rng)
        actual = _run_te_pipeline_training(graph_models, groups, pp_size, True)
        _assert_training_close(actual, expected, torch.bfloat16)
        assert [set(model.state_dict()) for model in graph_models] == initial_keys
        assert any(
            not torch.equal(value, actual[-1]["parameters"][name])
            for name, value in initial_parameters.items()
        ), "The test must execute optimizer updates"
        for fragment in ("hyper_connection", ".mtp."):
            names = [name for name in graph_parameters if fragment in name]
            if fragment == "hyper_connection" or (
                "mtp" in model_kind and groups.pp.rank() == pp_size - 1
            ):
                assert names, f"Missing {fragment} parameters"
                assert any(actual[-1]["gradients"][name].count_nonzero() for name in names)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()
