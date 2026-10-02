# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _update_router_expert_bias_with_quantile,
    finalize_model_grads,
    reset_model_temporary_tensors,
)
from megatron.core.extensions.transformer_engine import fused_topk_with_score_function_supports_qb
from megatron.core.transformer.moe.moe_utils import (
    get_updated_expert_bias_with_quantile,
    topk_routing_with_score_function,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _QBFinalizeModel(torch.nn.Module):
    """Minimal model wrapper that runs the production gradient finalizer."""

    def __init__(self, router: torch.nn.Module, config: TransformerConfig):
        super().__init__()
        self.router = router
        self.config = config
        self.ddp_config = DistributedDataParallelConfig()
        self.finish_grad_sync_calls = 0

    def finish_grad_sync(self, force_all_reduce: bool = False):
        del force_all_reduce
        self.finish_grad_sync_calls += 1


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=1,
        moe_router_score_function="sigmoid",
        moe_router_load_balancing_type="quantile_balancing",
        moe_router_quantile_balancing_estimation_scope="global_batch",
        moe_aux_loss_coeff=0.0,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_qb_global_batch_config():
    config = _config()
    assert config.moe_router_quantile_balancing_estimation_scope == "global_batch"


def test_qb_accepts_singleton_load_balancing_list():
    config = _config(
        moe_router_load_balancing_type=["quantile_balancing"], moe_aux_loss_coeff=[0.0]
    )
    assert config.moe_router_load_balancing_type == "quantile_balancing"
    assert config.moe_aux_loss_coeff == 0.0


def test_qb_cli_exposes_global_batch_scope_and_histogram_bins():
    from megatron.training.arguments import add_megatron_arguments

    parser = argparse.ArgumentParser()
    add_megatron_arguments(parser)
    args = parser.parse_args(
        [
            "--moe-router-load-balancing-type",
            "quantile_balancing",
            "--moe-router-quantile-balancing-estimation-scope",
            "global_batch",
            "--moe-router-qb-num-bins",
            "257",
        ]
    )

    assert args.moe_router_load_balancing_type == ["quantile_balancing"]
    assert args.moe_router_quantile_balancing_estimation_scope == "global_batch"
    assert args.moe_router_qb_num_bins == 257


def test_qb_main_accepts_micro_batch_scope():
    config = _config(moe_router_quantile_balancing_estimation_scope="micro_batch")
    assert config.moe_router_quantile_balancing_estimation_scope == "micro_batch"


def test_qb_main_defaults_to_micro_batch_scope():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=1,
        moe_router_score_function="sigmoid",
        moe_router_load_balancing_type="quantile_balancing",
        moe_aux_loss_coeff=0.0,
    )
    assert config.moe_router_quantile_balancing_estimation_scope == "micro_batch"


def test_qb_rejects_nonzero_aux_loss():
    with pytest.raises(ValueError, match="moe_aux_loss_coeff=0"):
        _config(moe_aux_loss_coeff=0.01)


def test_qb_treats_negative_capacity_as_disabled():
    config = _config(moe_expert_capacity_factor=-1.0)
    assert config.moe_expert_capacity_factor is None


def test_qb_negative_capacity_preserves_sinkhorn_validation():
    with pytest.raises(ValueError, match="moe_expert_capacity_factor only works"):
        _config(moe_router_load_balancing_type="sinkhorn", moe_expert_capacity_factor=-1.0)


def test_qb_rejects_active_expert_capacity():
    with pytest.raises(ValueError, match="does not support per-expert token dropping"):
        _config(moe_expert_capacity_factor=1.0)


@pytest.mark.parametrize(
    "overrides, error_match",
    [
        ({"moe_router_score_function": "softmax"}, "requires moe_router_score_function"),
        ({"moe_router_enable_expert_bias": True}, "do not also enable"),
        ({"moe_router_num_groups": 2, "moe_router_group_topk": 1}, "does not support group"),
        ({"moe_enable_routing_replay": True}, "does not support routing replay"),
        ({"moe_expert_rank_capacity_factor": 1.0}, "expert-rank capacity requires"),
        ({"moe_router_topk": 4}, "requires.*0 < moe_router_topk"),
        ({"moe_router_qb_num_bins": 1}, "must be greater than one"),
    ],
)
def test_qb_rejects_incompatible_config(overrides, error_match):
    """Reject QB combinations that cannot produce the required K3 histogram."""
    with pytest.raises(ValueError, match=error_match):
        _config(**overrides)


def test_qb_histogram_recovery():
    histogram = torch.tensor([[0, 2, 2, 0], [2, 2, 0, 0]], dtype=torch.int32)
    bin_bounds = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    expert_bias = torch.zeros(2, dtype=torch.float32)

    updated_bias, updated_bounds = get_updated_expert_bias_with_quantile(
        histogram, bin_bounds, expert_bias, topk=1
    )

    torch.testing.assert_close(updated_bias, torch.tensor([0.25, -0.25]))
    torch.testing.assert_close(updated_bounds, torch.tensor([-1.25, 1.25]))


@pytest.mark.parametrize(
    "overrides, error_match",
    [
        ({"qb_bin_bounds": None}, "must be provided together"),
        ({"expert_bias": None}, "requires an expert bias"),
        ({"score_function": "softmax"}, "requires score_function='sigmoid'"),
        ({"use_pre_softmax": True}, "does not use pre-softmax"),
        ({"topk": 4}, "requires topk < num_experts"),
        ({"num_groups": 2}, "does not support group-limited routing"),
        ({"router_replay": object()}, "does not support router replay"),
        ({"qb_histogram": torch.zeros(3, 8, dtype=torch.int32)}, "must have shape"),
        ({"qb_histogram": torch.zeros(4, 8, dtype=torch.int64)}, "dtype torch.int32"),
        ({"qb_bin_bounds": torch.zeros(2, dtype=torch.float64)}, "must be an FP32 tensor"),
    ],
)
def test_qb_routing_rejects_invalid_histogram_inputs(overrides, error_match):
    """Validate QB-specific routing inputs before either fused or unfused dispatch."""
    kwargs = dict(
        logits=torch.zeros(2, 4),
        topk=1,
        score_function="sigmoid",
        expert_bias=torch.zeros(4),
        qb_histogram=torch.zeros(4, 8, dtype=torch.int32),
        qb_bin_bounds=torch.tensor([-1.0, 1.0], dtype=torch.float32),
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=error_match):
        topk_routing_with_score_function(**kwargs)


def test_qb_empty_histogram_preserves_bias_and_bounds():
    histogram = torch.zeros(2, 4, dtype=torch.int32)
    bin_bounds = torch.tensor([-2.0, 3.0], dtype=torch.float32)
    expert_bias = torch.tensor([0.25, -0.25], dtype=torch.float32)

    updated_bias, updated_bounds = get_updated_expert_bias_with_quantile(
        histogram, bin_bounds, expert_bias, topk=1
    )

    torch.testing.assert_close(updated_bias, expert_bias)
    torch.testing.assert_close(updated_bounds, bin_bounds)


def test_qb_uses_pooled_global_batch_quantile_not_mean_microbatch_quantile():
    bounds = torch.tensor([0.0, 10.0], dtype=torch.float32)
    bias = torch.zeros(2, dtype=torch.float32)
    mb1 = torch.zeros(2, 10, dtype=torch.int32)
    mb2 = torch.zeros_like(mb1)
    mb1[0, 0], mb1[0, 9], mb1[1, 0] = 3, 1, 4
    mb2[0, 9], mb2[1, 0] = 4, 4

    pooled_bias, _ = get_updated_expert_bias_with_quantile(mb1 + mb2, bounds, bias, topk=1)
    mb1_bias, _ = get_updated_expert_bias_with_quantile(mb1, bounds, bias, topk=1)
    mb2_bias, _ = get_updated_expert_bias_with_quantile(mb2, bounds, bias, topk=1)

    assert not torch.allclose(pooled_bias, (mb1_bias + mb2_bias) / 2)


def test_qb_unfused_histogram_accumulates_microbatches():
    logits = torch.tensor([[-1.0, 0.5, 1.5], [2.0, -0.5, 0.25]], dtype=torch.float32)
    bias = torch.tensor([-0.2, 0.1, 0.0], dtype=torch.float32)
    bounds = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    histogram = torch.zeros(3, 8, dtype=torch.int32)

    scores = torch.sigmoid(logits)
    topk_result = torch.topk(scores + bias, 2, dim=1, sorted=True)
    cutoff = topk_result.values[:, -1:]
    expected_bins = torch.floor((cutoff - scores - bounds[0]) * (8 / (bounds[1] - bounds[0])))
    expected_bins = expected_bins.to(torch.int64).clamp_(0, 7)
    expected = torch.zeros_like(histogram)
    offsets = torch.arange(3, dtype=torch.int64) * 8
    flat_indices = (expected_bins + offsets).reshape(-1)
    expected.view(-1).scatter_add_(
        0, flat_indices, torch.ones_like(flat_indices, dtype=torch.int32)
    )

    outputs = []
    for _ in range(2):
        outputs.append(
            topk_routing_with_score_function(
                logits,
                topk=1,
                score_function="sigmoid",
                expert_bias=bias,
                fused=False,
                qb_histogram=histogram,
                qb_bin_bounds=bounds,
            )
        )

    torch.testing.assert_close(histogram, expected * 2)
    selected = topk_result.indices[:, :1]
    expected_map = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, selected, True)
    torch.testing.assert_close(outputs[0][1], expected_map)
    torch.testing.assert_close(outputs[1][1], expected_map)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_topk_with_score_function_supports_qb,
    reason="requires the Transformer Engine QB fused-router API",
)
def test_qb_fused_atomic_matches_unfused_histogram():
    torch.manual_seed(1234)
    logits = torch.randn(37, 64, device="cuda", dtype=torch.float32)
    bias = torch.linspace(-0.1, 0.1, 64, device="cuda", dtype=torch.float32)
    bounds = torch.tensor([-1.1, 1.1], device="cuda", dtype=torch.float32)
    unfused_histogram = torch.zeros(64, 128, device="cuda", dtype=torch.int32)
    fused_histogram = torch.zeros_like(unfused_histogram)

    unfused_probs, unfused_map = topk_routing_with_score_function(
        logits,
        topk=8,
        score_function="sigmoid",
        expert_bias=bias,
        fused=False,
        qb_histogram=unfused_histogram,
        qb_bin_bounds=bounds,
    )
    fused_probs, fused_map = topk_routing_with_score_function(
        logits,
        topk=8,
        score_function="sigmoid",
        expert_bias=bias,
        fused=True,
        qb_histogram=fused_histogram,
        qb_bin_bounds=bounds,
    )

    torch.testing.assert_close(fused_probs, unfused_probs)
    torch.testing.assert_close(fused_map, unfused_map)
    torch.testing.assert_close(fused_histogram, unfused_histogram)


def test_qb_finalize_updates_once_and_reset_preserves_buffers():
    router = torch.nn.Module()
    router.register_buffer("expert_bias", torch.zeros(2, dtype=torch.float32))
    router.register_buffer(
        "qb_histogram",
        torch.tensor([[0, 2, 2, 0], [2, 2, 0, 0]], dtype=torch.int32),
        persistent=False,
    )
    router.register_buffer("qb_bin_bounds", torch.tensor([-1.0, 1.0], dtype=torch.float32))
    router.frozen_expert_bias = False
    model = torch.nn.Module()
    model.router = router
    config = SimpleNamespace(
        moe_router_topk=1,
        moe_router_enable_expert_bias=False,
        moe_router_load_balancing_type="quantile_balancing",
    )
    histogram_ptr = router.qb_histogram.data_ptr()
    bounds_ptr = router.qb_bin_bounds.data_ptr()

    _update_router_expert_bias_with_quantile([model], config, reduction_groups=())

    torch.testing.assert_close(router.expert_bias, torch.tensor([0.25, -0.25]))
    reset_model_temporary_tensors(config, [model])
    assert torch.count_nonzero(router.qb_histogram) == 0
    assert router.qb_histogram.data_ptr() == histogram_ptr
    assert router.qb_bin_bounds.data_ptr() == bounds_ptr


def test_qb_finalize_without_active_router_is_a_noop():
    model = torch.nn.Module()
    config = SimpleNamespace(moe_router_topk=1)

    _update_router_expert_bias_with_quantile([model], config, reduction_groups=())


def test_qb_router_maintains_float32_bias_and_bounds():
    from megatron.core.transformer.moe.router import TopKRouter

    router = SimpleNamespace(
        expert_bias=torch.zeros(4, dtype=torch.bfloat16),
        qb_bin_bounds=torch.tensor([-1.0, 1.0], dtype=torch.bfloat16),
    )

    TopKRouter._maintain_float32_expert_bias(router)

    assert router.expert_bias.dtype == torch.float32
    assert router.qb_bin_bounds.dtype == torch.float32


def test_qb_paged_stash_retry_discards_failed_attempt_histogram():
    from megatron.core.transformer.moe.paged_stash import PagedStashRunner

    histogram = torch.ones(3, 7, dtype=torch.int32)
    histogram_ptr = histogram.data_ptr()
    runner = PagedStashRunner.__new__(PagedStashRunner)
    runner.moe_layers = [SimpleNamespace(router=SimpleNamespace(qb_histogram=histogram))]

    runner._reset_qb_histograms()

    assert torch.count_nonzero(histogram) == 0
    assert histogram.data_ptr() == histogram_ptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.internal
@pytest.mark.parametrize(
    "fused",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not fused_topk_with_score_function_supports_qb,
                reason="requires the Transformer Engine QB fused-router API",
            ),
        ),
    ],
)
def test_qb_mcore_router_accumulates_microbatches_and_finalizes(fused, monkeypatch):
    """Exercise unfused and fused QB through a real MoE router and finalizer."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.training.initialize import _set_random_seed
    from tests.unit_tests.test_utilities import Utils

    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(1, 1)
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    config = _config(
        hidden_size=16,
        ffn_hidden_size=32,
        moe_router_topk=1,
        moe_router_fusion=fused,
        moe_token_dispatcher_type="alltoall",
        moe_router_qb_num_bins=128,
        params_dtype=torch.float32,
        add_bias_linear=False,
    )
    submodules = get_submodules(
        get_gpt_layer_local_submodules(config.num_moe_experts, moe_grouped_gemm=False).mlp
    )
    assert isinstance(submodules, MoESubmodules)
    layer = MoELayer(config, submodules).cuda()
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[:, 0].copy_(torch.tensor([3.0, 1.0, -0.5, -2.0], device="cuda"))

    for direction in (1.0, -1.0):
        hidden_states = torch.zeros(64, 1, config.hidden_size, device="cuda")
        hidden_states[:, 0, 0] = direction * torch.linspace(-2.0, 2.0, 64, device="cuda")
        output, _ = layer(hidden_states)
        output.float().square().mean().backward()

    router = layer.router
    assert router.expert_bias is not None
    assert "expert_bias" in router.state_dict()
    assert not hasattr(router, "qb_bias")
    torch.testing.assert_close(
        router.qb_histogram.sum(dim=1),
        torch.full((config.num_moe_experts,), 128, dtype=torch.int64, device="cuda"),
    )
    histogram_ptr = router.qb_histogram.data_ptr()
    bounds_ptr = router.qb_bin_bounds.data_ptr()
    old_bias = router.expert_bias.clone()
    marked_bounds = []
    monkeypatch.setitem(
        _update_router_expert_bias_with_quantile.__globals__,
        "mark_qb_bin_bounds_validated",
        marked_bounds.append,
    )
    _update_router_expert_bias_with_quantile(
        [layer], config, reduction_groups=(torch.distributed.group.WORLD,)
    )

    assert len(marked_bounds) == 1
    assert marked_bounds[0] is router.qb_bin_bounds
    assert not torch.equal(router.expert_bias, old_bias)
    torch.testing.assert_close(
        router.expert_bias.mean(), torch.zeros((), device="cuda"), atol=1e-7, rtol=0
    )
    reset_model_temporary_tensors(config, [layer])
    assert torch.count_nonzero(router.qb_histogram) == 0
    assert router.qb_histogram.data_ptr() == histogram_ptr
    assert router.qb_bin_bounds.data_ptr() == bounds_ptr
    Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.internal
def test_qb_router_histogram_gating_and_activation_recompute():
    """Accumulate once under recompute, but not in eval, frozen, or no-grad forwards."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
    from megatron.core.tensor_parallel.random import checkpoint
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.training.initialize import _set_random_seed
    from tests.unit_tests.test_utilities import Utils

    Utils.destroy_model_parallel()
    try:
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = _config(
            hidden_size=16,
            ffn_hidden_size=32,
            moe_router_topk=2,
            moe_router_fusion=False,
            moe_token_dispatcher_type="alltoall",
            moe_router_qb_num_bins=128,
            params_dtype=torch.float32,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(config.num_moe_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        router = MoELayer(config, submodules).cuda().router
        hidden_states = torch.randn(32, 1, config.hidden_size, device="cuda", requires_grad=True)

        router.eval()
        router(hidden_states)
        assert torch.count_nonzero(router.qb_histogram) == 0

        router.train()
        router.frozen_expert_bias = True
        router(hidden_states)
        assert torch.count_nonzero(router.qb_histogram) == 0

        router.frozen_expert_bias = False
        with torch.no_grad():
            router(hidden_states)
        assert torch.count_nonzero(router.qb_histogram) == 0

        routing_probs, _ = checkpoint(router, False, hidden_states)
        assert torch.count_nonzero(router.qb_histogram) == 0
        routing_probs.square().sum().backward()
        torch.testing.assert_close(
            router.qb_histogram.sum(dim=1),
            torch.full(
                (config.num_moe_experts,),
                hidden_states.shape[0] * hidden_states.shape[1],
                dtype=torch.int64,
                device="cuda",
            ),
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.internal
@pytest.mark.parametrize(
    "fused",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not fused_topk_with_score_function_supports_qb,
                reason="requires the Transformer Engine QB fused-router API",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "tp_size,ep_size,dense_dp_size,expert_dp_size", [(1, 4, 8, 2), (1, 8, 8, 1), (4, 2, 2, 1)]
)
def test_qb_world8_ep_topologies_finalize_model_grads(
    fused, tp_size, ep_size, dense_dp_size, expert_dp_size
):
    """Validate QB reduction and finalization across eight-rank EP topologies."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.training.initialize import _set_random_seed
    from tests.unit_tests.test_utilities import Utils

    # torchrun exports WORLD_SIZE before MCore initializes the process group. Guard on the
    # environment here so a normal single-GPU unit-test run skips before trying to construct an
    # impossible EP4/EP8 topology, while the intended eight-rank launch reaches initialization.
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        pytest.skip("requires a world size of 8")

    Utils.destroy_model_parallel()
    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=tp_size,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        # Establish that MCore generated exactly the topology under test. In particular, EP is
        # folded into the router's dense-data-parallel group, while expert data parallelism is
        # world_size / (TP * EP). The production QB reduction group spans TP x dense-DP x CP and
        # must therefore contain all eight ranks for every parameterized topology.
        assert dist.get_world_size() == 8
        assert parallel_state.get_tensor_model_parallel_world_size() == tp_size
        assert parallel_state.get_expert_model_parallel_world_size() == ep_size
        assert parallel_state.get_data_parallel_world_size() == dense_dp_size
        assert parallel_state.get_expert_data_parallel_world_size() == expert_dp_size
        tp_dp_cp_group = parallel_state.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )
        assert tp_dp_cp_group.size() == 8

        config = _config(
            hidden_size=16,
            ffn_hidden_size=32,
            num_moe_experts=8,
            moe_router_topk=1,
            moe_router_fusion=fused,
            moe_token_dispatcher_type="alltoall",
            moe_router_qb_num_bins=128,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=tp_size,
            # MCore requires sequence parallelism whenever training combines MoE with TP. The
            # TP1 cases deliberately leave it off; TP4/EP2 turns it on to exercise the supported
            # expert-tensor-parallel execution path rather than bypassing MoELayer's safety check.
            sequence_parallel=tp_size > 1,
            params_dtype=torch.float32,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(config.num_moe_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        layer = MoELayer(config, submodules).cuda()
        with torch.no_grad():
            layer.router.weight.zero_()
            layer.router.weight[:, 0].copy_(
                torch.tensor([3.0, 2.0, 1.0, 0.25, -0.25, -1.0, -2.0, -3.0], device="cuda")
            )

        dense_dp_rank = parallel_state.get_data_parallel_rank()
        token_axis = torch.linspace(-2.0, 2.0, 32, device="cuda")
        for microbatch in range(2):
            hidden_states = torch.zeros(32, 1, config.hidden_size, device="cuda")
            hidden_states[:, 0, 0] = (
                (1.0 if microbatch == 0 else -1.0) * token_axis
                + 0.35 * dense_dp_rank
                + 0.2 * microbatch
            )
            output, _ = layer(hidden_states)
            output.float().square().mean().backward()

        # Every token contributes one margin sample to every expert's histogram, independent of
        # top-k. Two 32-token microbatches must therefore leave exactly 64 samples per expert in
        # each rank-local accumulator. This also proves accumulation happened across microbatches
        # instead of replacing the first microbatch's statistics.
        router = layer.router
        local_histogram = router.qb_histogram.clone()
        torch.testing.assert_close(
            local_histogram.sum(dim=1),
            torch.full((config.num_moe_experts,), 64, dtype=torch.int64, device="cuda"),
        )
        gathered_histograms = [torch.empty_like(local_histogram) for _ in range(8)]
        dist.all_gather(gathered_histograms, local_histogram)

        # Rank-dependent inputs deliberately produce at least two distinct local histograms. This
        # prevents a false-positive where final biases agree merely because every rank started from
        # identical statistics, without exercising the distributed histogram reduction.
        assert any(
            not torch.equal(gathered_histograms[0], histogram)
            for histogram in gathered_histograms[1:]
        )

        # Build an independent oracle from an explicit all-reduce of the pre-finalization local
        # histogram, then apply the same pure quantile update math. finalize_model_grads must
        # reproduce both the resulting expert bias and the adaptively updated bin bounds.
        expected_histogram = local_histogram.clone()
        dist.all_reduce(expected_histogram, group=tp_dp_cp_group)
        expected_bias, expected_bounds = get_updated_expert_bias_with_quantile(
            expected_histogram,
            router.qb_bin_bounds.clone(),
            router.expert_bias.clone(),
            config.moe_router_topk,
        )
        histogram_ptr = router.qb_histogram.data_ptr()
        bounds_ptr = router.qb_bin_bounds.data_ptr()
        model = _QBFinalizeModel(router, config)

        finalize_model_grads([model])

        torch.testing.assert_close(router.expert_bias, expected_bias)
        torch.testing.assert_close(router.qb_bin_bounds, expected_bounds)

        # Verify global agreement explicitly on WORLD, rather than checking only each rank against
        # its locally computed oracle. This catches a wrong reduction group that could otherwise
        # leave different EP or TP subgroups internally self-consistent.
        gathered_biases = [torch.empty_like(router.expert_bias) for _ in range(8)]
        gathered_bounds = [torch.empty_like(router.qb_bin_bounds) for _ in range(8)]
        dist.all_gather(gathered_biases, router.expert_bias)
        dist.all_gather(gathered_bounds, router.qb_bin_bounds)
        for bias, bounds in zip(gathered_biases[1:], gathered_bounds[1:]):
            torch.testing.assert_close(bias, gathered_biases[0])
            torch.testing.assert_close(bounds, gathered_bounds[0])

        # Finalization consumes the global-batch histogram exactly once and clears it in place.
        # Stable storage addresses are required by full-iteration CUDA graph capture; the persistent
        # bin-bound buffer must likewise be updated without replacement. The wrapper also proves the
        # normal gradient synchronization stage still executes exactly once.
        assert torch.count_nonzero(router.qb_histogram) == 0
        assert router.qb_histogram.data_ptr() == histogram_ptr
        assert router.qb_bin_bounds.data_ptr() == bounds_ptr
        assert model.finish_grad_sync_calls == 1
    finally:
        Utils.destroy_model_parallel()
