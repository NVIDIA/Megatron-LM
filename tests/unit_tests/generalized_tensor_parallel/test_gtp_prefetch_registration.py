# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit prefetch membership does not change sharding, grouping or capture scope."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel.gtp_api import register_prefetch_chain
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _make_gtp_linear,
    _torchrun_dist_init,
    reset_gtp_globals,
)


@pytest.fixture(autouse=True)
def restore_capture_config():
    """Keep classification tests independent of earlier graph tests."""
    gtp.set_cuda_graph_modules(None, cuda_graph_impl="none")
    yield
    gtp.set_cuda_graph_modules(None, cuda_graph_impl="none")


def _weight(group=None, grouped=False):
    weight = gtp.GTPShardedParam(torch.zeros(4, device="cuda"))
    weight.group = group
    weight.is_routed_expert = grouped
    return weight


def _model(**weights):
    return SimpleNamespace(named_parameters=lambda: iter(weights.items()))


def test_empty_selection_and_replicated_weight_rejection():
    register_prefetch_chain("optional", [])
    weight = _weight()
    replicated = torch.nn.Parameter(torch.zeros(4, device="cuda"))
    with pytest.raises(TypeError, match="already GTP-sharded"):
        register_prefetch_chain("dense", [weight, replicated])
    assert not hasattr(weight, "_gtp_prefetch_chain_name")
    gtp.classify_gtp_remat_chains(_model(sharded=weight, replicated=replicated))
    assert weight.chain_id == gtp.GTPChain.UNGRAPHED.value
    assert not hasattr(replicated, "chain_id")


@pytest.mark.parametrize("name", ["", "bad name", "a/b", None])
def test_invalid_name(name):
    with pytest.raises(ValueError, match="name"):
        register_prefetch_chain(name, [])


def test_conflicting_membership_is_atomic():
    first, second = _weight(), _weight()
    register_prefetch_chain("one", [first])
    register_prefetch_chain("one", [first])  # Idempotent before classification.
    with pytest.raises(ValueError, match="already belongs"):
        register_prefetch_chain("two", [second, first])
    assert not hasattr(second, "_gtp_prefetch_chain_name")


@pytest.mark.parametrize(
    "state", ["_gtp_chain_classified", "prefetch_initialized", "_recompute_initialized"]
)
def test_registration_rejects_initialized_weights(state):
    weight = _weight()
    setattr(weight, state, True)
    with pytest.raises(RuntimeError, match="before classification"):
        register_prefetch_chain("late", [weight])


def test_grouped_projection_is_indivisible():
    first, second = _weight(grouped=True), _weight(grouped=True)
    first.weight_list = [first, second]
    with pytest.raises(ValueError, match="leading weight"):
        register_prefetch_chain("fc1", [second])
    register_prefetch_chain("fc1", [first])
    assert second._gtp_prefetch_chain_name == "fc1"
    with pytest.raises(ValueError, match="already belongs"):
        register_prefetch_chain("fc2", [first, second])


@pytest.mark.parametrize("different_group", [False, True])
def test_reject_incompatible_members(different_group):
    first = _weight()
    second = _weight(group=object()) if different_group else _weight(grouped=True)
    if not different_group:
        second.weight_list = [second]
    with pytest.raises(ValueError, match="one weight kind and one process group"):
        register_prefetch_chain("mixed", [first, second])
    assert not hasattr(first, "_gtp_prefetch_chain_name")


def test_separate_registrations_cannot_merge_process_groups():
    first, second = _weight(), _weight(group=object())
    register_prefetch_chain("same", [first])
    register_prefetch_chain("same", [second])
    with pytest.raises(ValueError, match="mixes weight kinds or process groups"):
        gtp.classify_gtp_remat_chains(_model(first=first, second=second))


@pytest.mark.parametrize("impl,scope", [("none", []), ("local", ["moe"]), ("full_iteration", [])])
def test_capture_filter_and_shared_streams(impl, scope):
    group = object()
    dense = _weight(group)
    fc1, fc2 = _weight(group, True), _weight(group, True)
    fc1.weight_list, fc2.weight_list = [fc1], [fc2]
    register_prefetch_chain("dense", [dense])
    register_prefetch_chain("fc1", [fc1])
    register_prefetch_chain("fc2", [fc2])
    model = _model(
        **{
            "decoder.layers.0.self_attention.linear_proj.weight": dense,
            "decoder.layers.0.mlp.experts.linear_fc1.weight0": fc1,
            "decoder.layers.0.mlp.experts.linear_fc2.weight0": fc2,
        }
    )
    gtp.classify_gtp_remat_chains(model, cuda_graph_impl=impl, cuda_graph_modules=scope)
    assert len({w.chain_id for w in (dense, fc1, fc2)}) == 3
    assert gtp._chain_is_grouped(fc1.chain_id)
    assert not gtp._chain_is_grouped(dense.chain_id)
    assert gtp._stream_key(fc1.chain_id, group) == gtp._stream_key(fc2.chain_id, group)
    assert gtp._stream_key(dense.chain_id, group) != gtp._stream_key(fc1.chain_id, group)
    for weight, captured in (
        (dense, impl == "full_iteration"),
        (fc1, impl != "none"),
        (fc2, impl != "none"),
    ):
        assert gtp._chain_is_graphed(weight.chain_id) == captured
        assert (weight.chain_id in gtp.get_graphed_chain_ids()) == captured
    registered_ids = {dense.chain_id, fc1.chain_id, fc2.chain_id}
    gtp.reset_gtp_state()
    assert registered_ids.isdisjoint(gtp.get_graphed_chain_ids())


def test_same_logical_chain_splits_at_capture_boundary():
    attention, embedding = _weight(), _weight()
    register_prefetch_chain("dense", [attention, embedding])
    gtp.classify_gtp_remat_chains(
        _model(
            **{
                "decoder.layers.0.self_attention.linear_proj.weight": attention,
                "embedding.word_embeddings.weight": embedding,
            }
        ),
        cuda_graph_impl="local",
        cuda_graph_modules=["attn"],
    )
    assert attention.chain_id != embedding.chain_id
    assert gtp._chain_is_graphed(attention.chain_id)
    assert not gtp._chain_is_graphed(embedding.chain_id)
    assert not embedding._need_weight_prefetch_bwd


def test_registered_dense_chains_follow_execution_not_registration_order():
    group = torch.distributed.group.WORLD
    layers = [_make_gtp_linear(32, 32, group) for _ in range(4)]
    model = torch.nn.ModuleList(layers)
    # Execution alternates chains; list order must not override observed order.
    register_prefetch_chain("a", [layers[2].weight, layers[0].weight])
    register_prefetch_chain("b", [layers[3].weight, layers[1].weight])
    gtp.classify_gtp_remat_chains(model)
    x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
    full_weights = []
    for layer in layers:
        weight = torch.empty(
            layer.weight._unsharded_shape_padded, device="cuda", dtype=torch.bfloat16
        )
        torch.distributed.all_gather_into_tensor(weight, layer.weight.detach(), group=group)
        full_weights.append(weight[:32])
    for _ in range(3):
        for layer, weight in zip(layers, full_weights):
            output = layer(x, is_first_microbatch=True)
            torch.testing.assert_close(output, torch.nn.functional.linear(x, weight))
    for i in (0, 1):
        assert layers[i].weight.next_w is layers[i + 2].weight
        assert layers[i + 2].weight.prev_w is layers[i].weight
    assert gtp.get_ag_stream(layers[0].weight.chain_id, group) is gtp.get_ag_stream(
        layers[1].weight.chain_id, group
    )


def test_registered_chain_buffers_do_not_alias():
    group = torch.distributed.group.WORLD
    first, second = _weight(group), _weight(group)
    register_prefetch_chain("a", [first])
    register_prefetch_chain("b", [second])
    gtp.classify_gtp_remat_chains(_model(first=first, second=second))
    assert gtp._stream_key(first.chain_id, None) == gtp._stream_key(second.chain_id, None)
    for recompute in (False, True):
        assert first._get_cache_key(
            torch.bfloat16, True, False, recompute
        ) != second._get_cache_key(torch.bfloat16, True, False, recompute)
