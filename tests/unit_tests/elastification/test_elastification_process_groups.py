# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Flextron training uses the process groups of its model and run, not the global grid.

Each test either makes every global parallel_state accessor fail, or gives the code groups whose
layout differs from the global grid, so a read of the global grid cannot pass unnoticed.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.elastification import flextron_utils
from megatron.elastification import loss_func as loss_func_module
from megatron.elastification import pretrain_hybrid_flex
from megatron.elastification.flextron_elasticity_hooks import (
    FlextronGroupedMLPElasticityManager,
    FlextronMambaElasticityManager,
)
from megatron.elastification.flextron_utils import (
    FlextronModelManager,
    inject_flextron_forward_logic,
)
from tests.unit_tests.test_utilities import Utils


class _FakeGroup:
    """Process-group stand-in with a rank, a size and the global ranks it spans."""

    def __init__(self, rank, size, first_rank=0):
        self._rank = rank
        self.ranks = list(range(first_rank, first_rank + size))

    def rank(self):
        return self._rank

    def size(self):
        return len(self.ranks)


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _forbid_global_process_groups(monkeypatch):
    """Fail the test on any read of the global groups, ranks, sizes or pipeline stages."""

    def fail(*args, **kwargs):
        pytest.fail("Flextron code must not read the global parallel state")

    for name, value in list(vars(parallel_state).items()):
        if name.startswith(("get_", "is_pipeline_")) and callable(value):
            monkeypatch.setattr(parallel_state, name, fail)
    monkeypatch.setattr(
        ProcessGroupCollection,
        "use_mpu_process_groups",
        Mock(side_effect=AssertionError("supplied groups must not be rebuilt from globals")),
    )


def _use_fake_groups(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)


def _use_global_fake_groups(monkeypatch, **groups):
    """Make the global fallback build a collection with ``groups``."""
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        ProcessGroupCollection,
        "use_mpu_process_groups",
        staticmethod(lambda required_pgs=None: ProcessGroupCollection(**groups)),
    )


# --------------------------------------------------------------------------- loss function


@pytest.fixture
def collectives(monkeypatch):
    """Record the groups of the loss collectives instead of running them on the fake groups."""
    calls = SimpleNamespace(all_reduce=[], all_gather=[])
    _use_fake_groups(monkeypatch)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, group=None, **kwargs: calls.all_reduce.append(group),
    )

    def all_gather(tensor_list, tensor, group=None, **kwargs):
        calls.all_gather.append(group)
        for gathered in tensor_list:
            gathered.copy_(tensor)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: group.size())
    return calls


def _loss_args(**overrides):
    args = dict(
        router_beta=1.0,
        loss_alpha=1.0,
        freeze_router=False,
        export_kd_teacher_load=False,
        budget_list=[1.0, 0.5],
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_sequence_parallel_loss_uses_the_supplied_tensor_parallel_group(monkeypatch, collectives):
    _forbid_global_process_groups(monkeypatch)
    monkeypatch.setattr(loss_func_module, "get_args", _loss_args)
    tp_group = _FakeGroup(rank=1, size=2)
    # The second half of the mask belongs to TP rank 1 of 2.
    loss_mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])

    loss = loss_func_module._mask_loss(
        (torch.tensor([[2.0, 3.0]]), False, True), loss_mask, tp_group=tp_group
    )

    assert loss.item() == pytest.approx(5.0)
    assert collectives.all_reduce == [tp_group]


def test_omitted_group_warns_and_uses_the_global_group(monkeypatch, collectives):
    monkeypatch.setattr(loss_func_module, "get_args", _loss_args)
    global_tp_group = _FakeGroup(rank=0, size=2)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_group", lambda *args, **kwargs: global_tp_group
    )

    with pytest.warns(
        ProcessGroupFallbackWarning, match="_mask_loss was called without `tp_group`"
    ) as record:
        loss_func_module._mask_loss((torch.tensor([[1.0, 2.0]]), True, False), torch.ones(1, 2))

    message = str(record[0].message)
    assert "deprecated since Megatron Core 0.21" in message
    assert "removed in 0.23" in message
    assert collectives.all_reduce == [global_tp_group]


def _kd_model(pg_collection):
    def compute_kd_loss(student_loss, loss_reduction_fn):
        split_loss = loss_reduction_fn((torch.tensor([[4.0, 4.0]]), True, False))
        return {
            "kd_loss": split_loss,
            "logits_loss": split_loss,
            "intermediate_loss": torch.tensor(0.0),
        }

    return SimpleNamespace(
        pg_collection=pg_collection, training=True, compute_kd_loss=compute_kd_loss
    )


def test_distillation_report_is_gathered_over_the_model_data_parallel_group(
    monkeypatch, collectives
):
    _forbid_global_process_groups(monkeypatch)
    monkeypatch.setattr(
        loss_func_module, "get_args", lambda: _loss_args(export_kd_teacher_load=True)
    )
    tp_group = _FakeGroup(rank=0, size=2)
    dp_group = _FakeGroup(rank=1, size=4)
    model = _kd_model(ProcessGroupCollection(tp=tp_group, dp_gtp_remat=dp_group))
    output = (torch.tensor([[1.0, 2.0]]), (torch.tensor([0.5]), {}))

    _, num_tokens, report = loss_func_module.loss_func(
        torch.ones(1, 2), output, model, selected_budget=0.5
    )

    assert collectives.all_reduce == [tp_group]
    assert collectives.all_gather == [dp_group, dp_group]
    # Every rank of the group drew the selected budget, so the budget's token count is the
    # local count divided by the size of the data-parallel group.
    budget_report = report["logits distillation loss 0.500"]
    assert budget_report[0].item() == pytest.approx(8.0)
    assert budget_report[1].item() == pytest.approx(num_tokens.item() / dp_group.size())


def test_distillation_rejects_a_model_without_a_data_parallel_group(monkeypatch, collectives):
    _forbid_global_process_groups(monkeypatch)
    monkeypatch.setattr(
        loss_func_module, "get_args", lambda: _loss_args(export_kd_teacher_load=True)
    )
    model = _kd_model(ProcessGroupCollection(tp=_FakeGroup(rank=0, size=2)))
    output = (torch.tensor([[1.0, 2.0]]), (torch.tensor([0.5]), {}))

    with pytest.raises(ValueError, match="pg_collection.dp_gtp_remat"):
        loss_func_module.loss_func(torch.ones(1, 2), output, model, selected_budget=0.5)
    assert collectives.all_gather == []


# --------------------------------------------------------------------------- model manager


def test_router_is_built_and_named_on_the_model_groups(monkeypatch):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    router_cls = Mock(return_value=object())
    monkeypatch.setattr(flextron_utils, "FlextronRouter", router_cls)
    pg_collection = ProcessGroupCollection(pp=_FakeGroup(rank=3, size=4))
    model = SimpleNamespace(pg_collection=pg_collection)
    manager = FlextronModelManager.__new__(FlextronModelManager)
    manager.model = model
    manager.config = SimpleNamespace(enable_router=True)

    manager.setup_router()

    router_cls.assert_called_once_with(config=manager.config, pg_collection=pg_collection)
    assert model.router_pp3 is router_cls.return_value
    assert model.router is router_cls.return_value


@pytest.mark.parametrize("pp_rank,returns_budget_loss", [(1, True), (0, False)])
def test_forward_returns_the_budget_loss_on_the_last_stage_of_the_model(
    monkeypatch, pp_rank, returns_budget_loss
):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    model = SimpleNamespace(
        config=SimpleNamespace(flextron=True, is_flex_eval=False),
        pg_collection=ProcessGroupCollection(pp=_FakeGroup(rank=pp_rank, size=2)),
        _flextron_manager=SimpleNamespace(
            router=object(),
            process_router_output=lambda budget: ({}, lambda kwargs, budget: "budget-loss"),
            update_hook_elasticity_params=lambda kwargs: None,
        ),
    )
    model.forward = lambda **kwargs: "lm-loss"
    inject_flextron_forward_logic(model)

    result = model.forward(None, None, None, labels="labels", budget=0.5)

    assert result == (("lm-loss", "budget-loss") if returns_budget_loss else "lm-loss")


# --------------------------------------------------------------------------- training script


class _HostValue:
    """Batch entry that stays on the host when get_batch moves it to the GPU."""

    def cuda(self, non_blocking=False):
        return self


@pytest.fixture
def flex_get_batch(monkeypatch):
    """pretrain_hybrid_flex.get_batch with stubbed TP and CP helpers that record their groups."""
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    args = SimpleNamespace(
        context_parallel_size=1,
        sft=False,
        dataloader_inter_document_masking=False,
        dynamic_context_parallel=False,
        create_attention_mask_in_dataloader=False,
        micro_batch_size=1,
        seq_length=8,
        pipeline_model_parallel_size=2,
    )
    config = SimpleNamespace(
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        virtual_pipeline_model_parallel_size=None,
    )
    monkeypatch.setattr(pretrain_hybrid_flex, "get_args", lambda: args)
    monkeypatch.setattr(
        pretrain_hybrid_flex, "core_transformer_config_from_args", lambda args: config
    )
    tp_broadcast = Mock(
        side_effect=lambda batch, **kwargs: {
            **dict.fromkeys(pretrain_hybrid_flex.BATCH_KEYS),
            **batch,
        }
    )
    monkeypatch.setattr(pretrain_hybrid_flex, "get_batch_on_this_tp_rank", tp_broadcast)
    cp_shard = Mock(side_effect=lambda batch, **kwargs: batch)
    monkeypatch.setattr(pretrain_hybrid_flex, "get_batch_on_this_cp_rank", cp_shard)
    return SimpleNamespace(args=args, config=config, tp_broadcast=tp_broadcast, cp_shard=cp_shard)


def _flex_groups(tp_rank, pp_rank, pp_size):
    """Groups whose global ranks differ from any global grid of the test's world."""
    return ProcessGroupCollection(
        tp=_FakeGroup(tp_rank, 2, first_rank=5),
        pp=_FakeGroup(pp_rank, pp_size, first_rank=11),
        cp=_FakeGroup(0, 1, first_rank=5),
        dynamic_cp={1: _FakeGroup(0, 1), 2: _FakeGroup(1, 2)},
    )


@pytest.mark.parametrize(
    "tp_rank,pp_rank,pp_size,vp_stage,vp_size",
    [(0, 0, 2, None, None), (1, 1, 2, None, None), (0, 1, 2, 1, 2), (1, 0, 4, 0, 2)],
)
def test_flex_get_batch_broadcasts_and_shards_over_the_supplied_groups(
    flex_get_batch, tp_rank, pp_rank, pp_size, vp_stage, vp_size
):
    entry = flex_get_batch
    entry.config.virtual_pipeline_model_parallel_size = vp_size
    pgs = _flex_groups(tp_rank, pp_rank, pp_size)
    sample = {"tokens": _HostValue()}
    data_iterator = iter([sample]) if tp_rank == 0 else None

    tokens, *_ = pretrain_hybrid_flex.get_batch(data_iterator, vp_stage=vp_stage, pg_collection=pgs)

    assert tokens is (sample["tokens"] if tp_rank == 0 else None)
    kwargs = entry.tp_broadcast.call_args.kwargs
    assert kwargs["broadcast_group"] is pgs.tp
    assert kwargs["broadcast_src_rank"] == 5
    assert kwargs["tp_rank"] == tp_rank
    assert kwargs["is_pipeline_first_stage"] is (pp_rank == 0)
    assert kwargs["is_pipeline_last_stage"] is (pp_rank == pp_size - 1)
    cp_kwargs = entry.cp_shard.call_args.kwargs
    assert cp_kwargs["cp_group"] is pgs.cp
    # The dynamic CP groups come from the collection too.
    assert cp_kwargs["dynamic_cp_group_func"](group_size=2) is pgs.dynamic_cp[2]


@pytest.mark.parametrize(
    "pp_rank,pp_size,vp_stage,vp_size", [(1, 3, None, None), (1, 2, 0, 2), (0, 2, 1, 2)]
)
def test_flex_get_batch_skips_stages_of_the_supplied_pipeline_without_data(
    flex_get_batch, pp_rank, pp_size, vp_stage, vp_size
):
    flex_get_batch.config.virtual_pipeline_model_parallel_size = vp_size
    data_iterator = iter([{"tokens": _HostValue()}])

    batch = pretrain_hybrid_flex.get_batch(
        data_iterator, vp_stage=vp_stage, pg_collection=_flex_groups(0, pp_rank, pp_size)
    )

    assert batch == (None,) * 7
    flex_get_batch.tp_broadcast.assert_not_called()
    assert next(data_iterator) is not None


@pytest.mark.parametrize("missing", ["tp", "pp", "cp", "dynamic_cp"])
def test_flex_get_batch_rejects_a_collection_without_its_groups(flex_get_batch, missing):
    flex_get_batch.args.dynamic_context_parallel = missing == "dynamic_cp"
    groups = {name: _FakeGroup(0, 1) for name in ("tp", "pp", "cp")}
    groups["dynamic_cp"] = {1: _FakeGroup(0, 1)}
    groups[missing] = None

    with pytest.raises(ValueError, match=f"pg_collection.{missing}"):
        pretrain_hybrid_flex.get_batch(
            iter([{"tokens": _HostValue()}]), pg_collection=ProcessGroupCollection(**groups)
        )
    flex_get_batch.tp_broadcast.assert_not_called()


def test_flex_get_batch_without_groups_warns_and_uses_the_global_groups(
    flex_get_batch, monkeypatch
):
    global_groups = dict(
        tp=_FakeGroup(0, 2, first_rank=3), pp=_FakeGroup(0, 2), cp=_FakeGroup(0, 1)
    )
    _use_global_fake_groups(monkeypatch, **global_groups)

    with pytest.warns(
        ProcessGroupFallbackWarning,
        match="pretrain_hybrid_flex.get_batch was called without `pg_collection`",
    ):
        pretrain_hybrid_flex.get_batch(iter([{"tokens": _HostValue()}]))

    assert flex_get_batch.tp_broadcast.call_args.kwargs["broadcast_group"] is global_groups["tp"]


@pytest.fixture
def budget_draw(monkeypatch):
    """get_grad_acc_based_random_choice with 8 samples per step and micro-batches of 1."""
    monkeypatch.setattr(pretrain_hybrid_flex, "get_current_global_batch_size", lambda: 8)
    monkeypatch.setattr(pretrain_hybrid_flex, "get_micro_batch_size", lambda: 1)
    monkeypatch.setattr(
        pretrain_hybrid_flex,
        "get_train_state",
        lambda: SimpleNamespace(iteration=1, resume_iteration=0),
    )
    monkeypatch.setattr(pretrain_hybrid_flex, "_global_choice_counter", 0)
    return pretrain_hybrid_flex.get_grad_acc_based_random_choice


def test_budget_draw_counts_accumulation_steps_over_the_model_data_parallel_group(
    monkeypatch, budget_draw
):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    pg_collection = ProcessGroupCollection(dp_gtp_remat=_FakeGroup(rank=1, size=2))

    choice = budget_draw(SimpleNamespace(), pg_collection=pg_collection)

    # 8 samples per step over 2 data-parallel ranks is 4 accumulation steps.
    assert choice == np.random.RandomState(42 + 4 * 1 * 10).uniform(0, 1)
    assert pretrain_hybrid_flex._global_choice_counter == 1


def test_budget_draw_without_groups_warns(monkeypatch, budget_draw):
    _use_fake_groups(monkeypatch)
    _use_global_fake_groups(monkeypatch, dp_gtp_remat=_FakeGroup(rank=0, size=4))

    with pytest.warns(
        ProcessGroupFallbackWarning,
        match="get_grad_acc_based_random_choice was called without `pg_collection`",
    ):
        choice = budget_draw(SimpleNamespace())

    assert choice == np.random.RandomState(42 + 2 * 1 * 10).uniform(0, 1)


@pytest.mark.parametrize(
    "tp_rank,pp_rank,pp_size,built",
    [(0, 0, 2, True), (0, 1, 2, True), (1, 0, 2, False), (0, 1, 3, False), (0, 0, 1, True)],
)
def test_datasets_are_built_on_the_ranks_of_the_supplied_groups(
    monkeypatch, tp_rank, pp_rank, pp_size, built
):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    monkeypatch.setattr(
        pretrain_hybrid_flex,
        "get_args",
        lambda: SimpleNamespace(virtual_pipeline_model_parallel_size=None),
    )
    pgs = ProcessGroupCollection(tp=_FakeGroup(tp_rank, 2), pp=_FakeGroup(pp_rank, pp_size))

    assert pretrain_hybrid_flex.is_dataset_built_on_rank(pg_collection=pgs) is built


@pytest.mark.parametrize("missing", ["tp", "pp"])
def test_dataset_predicate_rejects_a_collection_without_its_groups(monkeypatch, missing):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    monkeypatch.setattr(
        pretrain_hybrid_flex,
        "get_args",
        lambda: SimpleNamespace(virtual_pipeline_model_parallel_size=None),
    )
    groups = {"tp": _FakeGroup(1, 2), "pp": _FakeGroup(0, 2)}
    groups[missing] = None

    with pytest.raises(ValueError, match=f"pg_collection.{missing}"):
        pretrain_hybrid_flex.is_dataset_built_on_rank(
            pg_collection=ProcessGroupCollection(**groups)
        )


def test_dataset_provider_passes_the_run_groups_to_the_builder(monkeypatch):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    monkeypatch.setattr(
        pretrain_hybrid_flex,
        "get_args",
        lambda: SimpleNamespace(
            sft=False, mock_data=True, virtual_pipeline_model_parallel_size=None
        ),
    )
    monkeypatch.setattr(
        pretrain_hybrid_flex, "core_gpt_dataset_config_from_args", lambda args: None
    )
    monkeypatch.setattr(pretrain_hybrid_flex, "print_rank_0", lambda *args, **kwargs: None)
    builder = Mock()
    builder.return_value.build.return_value = ("train", "valid", "test")
    monkeypatch.setattr(pretrain_hybrid_flex, "BlendedMegatronDatasetBuilder", builder)
    pgs = ProcessGroupCollection(tp=_FakeGroup(1, 2), pp=_FakeGroup(0, 2))

    datasets = pretrain_hybrid_flex.train_valid_test_datasets_provider([1, 1, 1], pg_collection=pgs)

    assert datasets == ("train", "valid", "test")
    is_built_on_rank = builder.call_args.args[2]
    # TP rank 1 of the run's tensor-parallel group does not build the datasets.
    assert is_built_on_rank() is False


def test_forward_step_passes_the_model_groups(monkeypatch):
    pg_collection = ProcessGroupCollection(tp=_FakeGroup(0, 1))
    get_batch = Mock(return_value=(None,) * 7)
    draw = Mock(side_effect=[0.5, 1.0])
    monkeypatch.setattr(pretrain_hybrid_flex, "get_batch", get_batch)
    monkeypatch.setattr(pretrain_hybrid_flex, "get_grad_acc_based_random_choice", draw)
    monkeypatch.setattr(pretrain_hybrid_flex, "_logged_params_norm", True)
    monkeypatch.setattr(pretrain_hybrid_flex, "get_timers", lambda: Mock())
    monkeypatch.setattr(
        pretrain_hybrid_flex,
        "get_args",
        lambda: SimpleNamespace(
            original_model_sample_prob=0.0, budget_list=[1.0], budget_probs=None
        ),
    )
    model = Mock(return_value="output")
    model.pg_collection = pg_collection

    output, _ = pretrain_hybrid_flex.forward_step(iter([]), model)

    assert output == "output"
    assert get_batch.call_args.kwargs["pg_collection"] is pg_collection
    assert [call.kwargs["pg_collection"] for call in draw.call_args_list] == [pg_collection] * 2


# --------------------------------------------------------------------------- GPU tests


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _ShardedGroupedMLP(nn.Module):
    """Grouped-MLP stand-in whose FC1 output holds this rank's shard of the FFN channels."""

    def __init__(self, hidden_size, ffn_hidden_size, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.hidden_size = hidden_size
        shard_size = ffn_hidden_size // tp_group.size()
        self.captured = None

        class _FC1(nn.Module):
            def forward(_self, x):
                return x.new_ones(*x.shape[:-1], shard_size), None

        self.linear_fc1 = _FC1()

    def forward(self, hidden_states):
        intermediate, _ = self.linear_fc1(hidden_states)
        self.captured = intermediate.detach().clone()
        return hidden_states, None


@requires_cuda
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_grouped_mlp_mask_is_split_over_the_module_group(monkeypatch, tp_rank):
    Utils.initialize_model_parallel()
    try:
        hidden_size, ffn_hidden_size = 16, 128
        config = SimpleNamespace(
            flextron=True,
            soft_mask=True,
            flex_hetero_ffn=False,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            emb_int_list=[hidden_size],
            mlp_int_list=[ffn_hidden_size, ffn_hidden_size // 2],
            hybrid_layer_pattern="E",
            layernorm_epsilon=1e-5,
        )
        module = _ShardedGroupedMLP(hidden_size, ffn_hidden_size, _FakeGroup(tp_rank, 2)).cuda()
        manager = FlextronGroupedMLPElasticityManager(config)
        manager.attach_hooks(module)
        # Full embedding, half of the FFN channels: only TP rank 0 of 2 keeps its shard.
        manager.set_elasticity_params(
            router_emb=(torch.tensor([1.0], device="cuda"), hidden_size),
            router_mlp=(torch.tensor([0.0, 1.0], device="cuda"), ffn_hidden_size // 2),
        )
        with monkeypatch.context() as patch:
            _forbid_global_process_groups(patch)
            module(torch.ones(2, hidden_size, dtype=torch.bfloat16, device="cuda"))
        manager.detach_hooks()

        expected = 1.0 if tp_rank == 0 else 0.0
        assert module.captured.shape[-1] == ffn_hidden_size // 2
        assert (module.captured == expected).all()
    finally:
        Utils.destroy_model_parallel()


@requires_cuda
def test_mamba_masks_use_the_mixer_tensor_parallel_group(monkeypatch):
    Utils.initialize_model_parallel()
    try:
        tp_size, nheads, headdim, ngroups, d_state = 2, 8, 4, 2, 4
        config = SimpleNamespace(
            flextron=True, hidden_size=16, emb_int_list=[16], mamba_int_list=[nheads, nheads // 2]
        )
        mixer = SimpleNamespace(
            tp_group=_FakeGroup(0, tp_size),
            nheads=nheads,
            headdim=headdim,
            d_state=d_state,
            nheads_local_tp=nheads // tp_size,
            d_inner_local_tp=nheads * headdim // tp_size,
            ngroups_local_tp=ngroups // tp_size,
        )
        manager = FlextronMambaElasticityManager(config)
        manager.mamba_mixer = mixer
        with monkeypatch.context() as patch:
            _forbid_global_process_groups(patch)
            manager._init_mamba_masks()

        # Half of the 8 heads is 2 of the 4 heads on each of the 2 TP ranks: 2 heads of z and
        # x, all of B and C, and 2 dt entries stay active.
        half_budget = manager.in_proj_mask_list[manager.mamba_masks_lookup[nheads // 2]]
        local_bc = 2 * mixer.ngroups_local_tp * d_state
        assert half_budget.sum().item() == 2 * (2 * headdim) + local_bc + 2
    finally:
        Utils.destroy_model_parallel()


def _strided_group(world_size):
    """Pair rank r with rank r + world_size // 2, unlike any group of the default global grid."""
    rank = torch.distributed.get_rank()
    half = world_size // 2
    own_group = None
    for first in range(half):
        ranks = [first, first + half]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            own_group = group
    return own_group


@requires_cuda
def test_router_seeds_and_shards_over_the_supplied_groups(monkeypatch):
    world_size = Utils.world_size
    if world_size < 4 or world_size % 2:
        pytest.skip("needs an even world size of at least 4")
    import megatron.training as megatron_training
    from megatron.elastification.router import hybrid_flex_router
    from tests.unit_tests.elastification.test_hybrid_flex_router import _router_config

    Utils.initialize_model_parallel()
    try:
        monkeypatch.setattr(
            hybrid_flex_router.FlextronRouter, "_sync_router_weights", lambda self: None
        )
        monkeypatch.setattr(hybrid_flex_router, "get_current_global_batch_size", lambda: 8)
        monkeypatch.setattr(hybrid_flex_router, "get_micro_batch_size", lambda: 1)
        monkeypatch.setattr(
            megatron_training,
            "get_args",
            lambda: SimpleNamespace(train_iters=10, train_samples=10, global_batch_size=8),
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        # A tensor-parallel communicator with the ranks of the global one, but not the same
        # object, and a data-parallel group of 2 ranks on a grid whose global data-parallel
        # group has all of them.
        pg_collection.tp, _ = torch.distributed.new_subgroups_by_enumeration(
            [[rank] for rank in range(world_size)]
        )
        dp_group = _strided_group(world_size)
        pg_collection.dp_gtp_remat = dp_group

        router = hybrid_flex_router.FlextronRouter(_router_config(), pg_collection=pg_collection)

        assert router.dp_size == 2
        assert router.dp_rank == dp_group.rank()
        assert router.grad_accumulation_steps == 4
        tp_attribute = "_tp_group" if hybrid_flex_router.HAVE_TE else "tp_group"
        for gate in (router.gate_mlp, router.gate_emb, router.gate_mamba, router.gate_moe_expert):
            for linear in (gate[0], gate[2]):
                assert getattr(linear, tp_attribute) is pg_collection.tp
    finally:
        Utils.destroy_model_parallel()


@requires_cuda
def test_router_without_groups_warns(monkeypatch):
    import megatron.training as megatron_training
    from megatron.elastification.router import hybrid_flex_router
    from tests.unit_tests.elastification.test_hybrid_flex_router import _router_config

    Utils.initialize_model_parallel()
    try:
        monkeypatch.setattr(
            hybrid_flex_router.FlextronRouter, "_sync_router_weights", lambda self: None
        )
        monkeypatch.setattr(hybrid_flex_router, "get_current_global_batch_size", lambda: 1)
        monkeypatch.setattr(hybrid_flex_router, "get_micro_batch_size", lambda: 1)
        monkeypatch.setattr(
            megatron_training,
            "get_args",
            lambda: SimpleNamespace(train_iters=10, train_samples=10, global_batch_size=1),
        )

        with pytest.warns(
            ProcessGroupFallbackWarning, match="FlextronRouter was called without `pg_collection`"
        ):
            router = hybrid_flex_router.FlextronRouter(_router_config())

        assert router.dp_size == parallel_state.get_data_parallel_world_size()
    finally:
        Utils.destroy_model_parallel()
