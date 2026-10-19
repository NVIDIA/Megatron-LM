# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.paged_stash import PagedStashRunner
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher, _HybridEPManager


@pytest.mark.parametrize(
    ("expert_rank_capacity_factor", "expected_num_permuted_tokens"), [(None, None), (1.5, 128)]
)
def test_combine_only_releases_dynamic_token_count(
    monkeypatch, expert_rank_capacity_factor, expected_num_permuted_tokens
):
    manager = object.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(moe_permute_fusion_into_hybridep=False)
    manager.handle = object()
    manager.num_permuted_tokens = 128
    manager.pad_multiple = None
    manager.drop_and_pad = False
    manager.moe_expert_rank_capacity_factor = expert_rank_capacity_factor
    manager._original_num_tokens = None
    manager._padded_num_tokens = None

    def fake_hybrid_ep_combine(**kwargs):
        assert kwargs["handle"] is manager.handle
        assert kwargs["num_permuted_tokens"] == 128
        return kwargs["x"]

    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_combine", fake_hybrid_ep_combine
    )

    hidden_states = torch.empty(4, 8)
    assert manager.combine(hidden_states) is hidden_states
    assert manager.handle is None
    assert manager.num_permuted_tokens == expected_num_permuted_tokens


class _FakeHybridEP:
    """Stand-in for hybrid_ep_dispatch/combine that records the budget of each dispatch.

    Microbatches alternate between two per-expert token counts. A static budget smaller than
    the routed token count drops tokens and sets the overflow flag, as HybridEP does.
    """

    def __init__(self, counts):
        self.counts = counts
        self.budgets = []

    def dispatch(self, **kwargs):
        budget = kwargs["num_permuted_tokens"]
        counts = self.counts[len(self.budgets) % len(self.counts)]
        self.budgets.append(budget)
        num_routed = int(counts.sum())
        num_rows = num_routed if budget is None else budget
        overflow = budget is not None and num_routed > budget
        hidden = torch.zeros(num_rows, kwargs["x"].shape[-1])
        return hidden, torch.ones(num_rows), None, counts, (torch.tensor([int(overflow)]),)

    def install(self, monkeypatch):
        monkeypatch.setattr(
            "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_dispatch", self.dispatch
        )
        monkeypatch.setattr(
            "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_combine",
            lambda **kwargs: kwargs["x"],
        )


# One microbatch over the 256-token budget (4 tokens x top-2 x 1.5, aligned to 256), one under.
_COUNTS = [torch.tensor([200, 100]), torch.tensor([50, 60])]
_STATIC_BUDGET = 256


def _rank_capacity_manager():
    manager = object.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(
        moe_router_topk=2,
        moe_expert_rank_capacity_factor=1.5,
        moe_hybridep_pad_variable_tokens=False,
        moe_hybridep_routing_map_mode="bool",
        moe_permute_fusion_into_hybridep=False,
        moe_flex_dispatcher_num_sms=32,
        moe_hybridep_num_blocks_permute=None,
        moe_hybridep_num_blocks_unpermute=None,
        moe_hybridep_num_sms_preprocessing=32,
        moe_use_grouped_tensor=True,
        use_transformer_engine_op_fuser=False,
        fp8=None,
        fp4=None,
    )
    manager.group = None
    manager.num_experts = manager.num_local_experts = 2
    manager.router_topk = 2
    manager.capacity_factor = None
    manager.drop_and_pad = False
    manager.dense_routing_supported = False
    manager.use_dense_routing_for_bool_map = False
    manager.moe_expert_rank_capacity_factor = 1.5
    manager._saved_num_permuted_tokens = None
    manager.num_permuted_tokens = None
    manager.pad_multiple = None
    manager.handle = None
    manager.over_budget = torch.zeros(1, dtype=torch.bool)
    manager._original_num_tokens = None
    manager._padded_num_tokens = None
    return manager


def _moe_forward(manager, run_setup_metadata=True):
    """One MoE-layer forward; skipping setup_metadata mimics a moe_preprocess graph replay."""
    if run_setup_metadata:
        manager.setup_metadata(torch.ones(4, 2, dtype=torch.bool), torch.ones(4, 2))
    dispatched = manager.dispatch(torch.ones(4, 8))
    tokens_per_expert = manager.get_number_of_tokens_per_expert().clone()
    manager.combine(dispatched)
    return tokens_per_expert


def _paged_stash_runner(manager, forward_backward_func=None, cuda_graph_modules=None):
    dispatcher = object.__new__(MoEFlexTokenDispatcher)
    dispatcher.config = manager.config
    dispatcher._comm_manager = manager
    runner = PagedStashRunner.__new__(PagedStashRunner)
    runner.config = SimpleNamespace(
        moe_paged_stash=True,
        cuda_graph_impl="none" if cuda_graph_modules is None else "transformer_engine",
        cuda_graph_modules=cuda_graph_modules or [],
    )
    runner.moe_layers = [SimpleNamespace(token_dispatcher=dispatcher)]
    runner._configs_to_sync_moe_paged_stash = [runner.config]
    runner.stash_manager = SimpleNamespace(
        overflow=None, host_spill=None, release_stash_buffers=lambda: None
    )
    runner.model = []
    runner.optimizer = None
    runner.copy_main_params = False
    runner.forward_backward_func = forward_backward_func
    runner._required_recv_capacity = None
    runner._required_capacity_factor = None
    runner._te_graph_capture_finished = False
    runner._te_graph_runtime_num_microbatches = None
    # The over-budget flag of this rank stands in for the cross-rank all-reduce.
    runner.check_moe_overflow = lambda: (0, int(manager.over_budget.any()), 0)
    return runner


@pytest.mark.parametrize("is_training", [True, False])
def test_dropless_rerun_dispatches_first_microbatch_dynamically(monkeypatch, is_training):
    fake = _FakeHybridEP(_COUNTS)
    fake.install(monkeypatch)
    manager = _rank_capacity_manager()
    for _ in _COUNTS:
        _moe_forward(manager)
    assert fake.budgets == [_STATIC_BUDGET, _STATIC_BUDGET]
    assert manager.over_budget.item()

    _paged_stash_runner(manager).prepare_for_rerun(is_training=is_training)

    fake.budgets.clear()
    for counts in _COUNTS:
        tokens_per_expert = _moe_forward(manager)
        # The failed attempt's budget would drop tokens again, and its stale expert counts
        # would split the dispatched rows at the wrong boundaries.
        assert fake.budgets[-1] is None
        torch.testing.assert_close(tokens_per_expert, counts)
        assert manager.num_permuted_tokens is None
    assert not manager.over_budget.any()


def test_paged_stash_rerun_restores_static_budget(monkeypatch):
    fake = _FakeHybridEP(_COUNTS)
    fake.install(monkeypatch)
    manager = _rank_capacity_manager()

    def forward_backward_func(**kwargs):
        for _ in range(kwargs["num_microbatches"]):
            _moe_forward(manager)

    runner = _paged_stash_runner(manager, forward_backward_func)
    runner(model=[None], data_iterator=None, num_microbatches=2, seq_length=4, forward_only=False)

    assert fake.budgets == [_STATIC_BUDGET, _STATIC_BUDGET, None, None]
    assert manager.moe_expert_rank_capacity_factor == 1.5
    # A replay that skips setup_metadata() dispatches with the static budget again and
    # records over-budget for the next PagedStashRunner check.
    assert manager.num_permuted_tokens == _STATIC_BUDGET
    manager.over_budget.zero_()
    _moe_forward(manager, run_setup_metadata=False)
    assert fake.budgets[-1] == _STATIC_BUDGET
    assert manager.over_budget.item()


def test_captured_te_whole_moe_graph_still_raises_without_rerun(monkeypatch):
    fake = _FakeHybridEP(_COUNTS)
    fake.install(monkeypatch)
    manager = _rank_capacity_manager()

    def forward_backward_func(**kwargs):
        for _ in range(kwargs["num_microbatches"]):
            _moe_forward(manager)

    runner = _paged_stash_runner(manager, forward_backward_func, cuda_graph_modules=[])
    runner.mark_te_graph_captured(num_microbatches=2)
    with pytest.raises(RuntimeError, match="Dynamic fallback is not supported"):
        runner(
            model=[None], data_iterator=None, num_microbatches=2, seq_length=4, forward_only=False
        )

    assert fake.budgets == [_STATIC_BUDGET, _STATIC_BUDGET]
    assert manager.moe_expert_rank_capacity_factor == 1.5
    assert manager.num_permuted_tokens == _STATIC_BUDGET


def test_rerun_keeps_static_per_expert_capacity():
    manager = _rank_capacity_manager()
    manager.drop_and_pad = True
    manager.num_permuted_tokens = 64
    manager.tokens_per_expert = torch.tensor([32, 32])

    manager.disable_expert_rank_capacity()
    assert manager.moe_expert_rank_capacity_factor is None
    assert manager.num_permuted_tokens == 64
    torch.testing.assert_close(manager.tokens_per_expert, torch.tensor([32, 32]))

    manager.restore_expert_rank_capacity()
    assert manager.moe_expert_rank_capacity_factor == 1.5
    assert manager.num_permuted_tokens == 64
