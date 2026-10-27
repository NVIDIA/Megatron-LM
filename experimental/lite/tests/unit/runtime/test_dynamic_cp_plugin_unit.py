# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.runtime.contracts.data import (
    ForwardResult,
    ModelOutputs,
    PackedBatch,
)
from megatron.lite.runtime.contracts.handle import ModelHandle
from megatron.lite.runtime.contracts.loss import LossContext, split_loss_context


class _Group:
    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


def test_dynamic_cp_keeps_r3_routes_and_masks_with_the_selected_samples():
    from megatron.lite.runtime.backends.mlite.dynamic_cp import (
        _batch_samples,
        _select_batch,
    )

    batch = PackedBatch(
        input_ids=torch.arange(5),
        labels=torch.arange(5),
        seq_lens=torch.tensor([3, 2]),
        routed_experts=torch.nested.as_nested_tensor(
            [torch.tensor([[[10, 11]], [[12, 13]]]), torch.tensor([[[20, 21]]])],
            layout=torch.jagged,
        ),
        r3_replay_mask=torch.tensor([True, True, False, True, False]),
    )

    selected = _select_batch(_batch_samples(batch), [1, 0], cp_size=1, leader=True)

    assert torch.equal(
        selected.r3_replay_mask, torch.tensor([True, False, True, True, False])
    )
    assert getattr(selected.routed_experts, "is_nested", False)
    routes = list(selected.routed_experts.unbind())
    assert torch.equal(routes[0], torch.tensor([[[20, 21]]]))
    assert torch.equal(routes[1], torch.tensor([[[10, 11]], [[12, 13]]]))


def test_dynamic_cp_allows_pipeline_parallel_but_not_virtual_pipeline_parallel():
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    pool, singleton, cp2 = _Group(4), _Group(1), _Group(2)
    handle = ModelHandle(
        model=object(),
        parallel_state=SimpleNamespace(
            dp_size=4, dp_rank=0, dp_group=pool, dp_cp_group=pool, cp_size=1, pp_size=2
        ),
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=2, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 8},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: cp2, 4: pool},
    )

    plugin.initialize(handle)

    handle.config.parallel.vpp = 2
    with pytest.raises(NotImplementedError, match="VPP=1"):
        DynamicCPPlugin(
            {"max_seqlen_per_dp_cp_rank": 8},
            create_groups=lambda _ps, _minimum, _parallel: {
                1: singleton,
                2: cp2,
                4: pool,
            },
        ).initialize(handle)


def test_dynamic_cp_builds_separate_dp_cp_groups_for_each_pipeline_stage(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import _create_groups

    created = []
    module = types.ModuleType("megatron.core.parallel_state")

    def create_dynamic_dp_cp_groups(rank, ranks, **_kwargs):
        created.append((rank, ranks))
        return {2: _Group(2)} if rank in ranks else {}

    module.create_dynamic_dp_cp_groups = create_dynamic_dp_cp_groups
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", module)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 4)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(
        torch.distributed, "new_group", lambda ranks: _Group(len(ranks))
    )
    physical = _Group(4)

    groups = _create_groups(
        SimpleNamespace(dp_size=2, dp_cp_group=physical),
        1,
        SimpleNamespace(tp=1, cp=2, pp=2),
    )

    assert created == [(4, [0, 1, 2, 3]), (4, [4, 5, 6, 7])]
    assert groups[4] is physical
    assert groups[1].size() == 1


@pytest.mark.parametrize("action", ["record", "replay"])
def test_dynamic_cp_forwards_r2_r3_router_replay_to_the_runtime(monkeypatch, action):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    class Prepared:
        data = iter(())
        count = 1
        loss = None
        forward = staticmethod(lambda model, batch: None)
        pre_forward = None
        binding = SimpleNamespace(bind=lambda _size: None)

        def finish(self, require_complete):
            assert require_complete

    handle = SimpleNamespace(
        _model=object(), _extras={"forward_step": lambda model, batch: None}
    )
    plugin = DynamicCPPlugin({"max_seqlen_per_dp_cp_rank": 8})
    monkeypatch.setattr(plugin, "_prepare", lambda *_args: Prepared())
    replay = {"action": action}

    def original_forward_backward(*args, **kwargs):
        assert kwargs["router_replay"] is replay
        return "forwarded"

    assert (
        plugin.wrap_forward_backward(original_forward_backward)(
            handle, object(), None, router_replay=replay
        )
        == "forwarded"
    )


def test_dynamic_cp_rebinds_each_prefetched_pipeline_microbatch(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _MixedCPScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)
    pool, singleton = _Group(2), _Group(1)
    ps = SimpleNamespace(
        dp_size=2,
        dp_rank=0,
        dp_group=pool,
        dp_cp_group=pool,
        cp_size=1,
        cp_rank=0,
        cp_group=singleton,
        pp_size=2,
    )
    seen = []
    aux_scales = []
    handle = ModelHandle(
        model=object(),
        parallel_state=ps,
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=2, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={
            "forward_step": lambda _model, _batch: seen.append(ps.cp_size),
            "_dcp_original_pre_forward": lambda scale: aux_scales.append(float(scale)),
        },
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 4},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
    )
    plugin.initialize(handle)
    prepared = plugin._prepare(
        handle,
        PackedBatch(
            input_ids=torch.arange(12),
            labels=torch.arange(12),
            seq_lens=torch.tensor([2, 8, 2]),
        ),
        None,
        1,
    )
    handle._extras["_dcp_original_forward"] = handle._extras["forward_step"]

    # PP schedules prefetch all batches before running the first forward.
    prefetched = [split_loss_context(item)[0] for item in prepared.data]
    for batch in prefetched:
        prepared.pre_forward(torch.tensor(0.5))
        prepared.forward(handle._model, batch)
    prepared.finish(require_complete=True)

    assert seen == [1, 2]
    assert aux_scales == [0.5, 1.0]
    assert (ps.cp_size, ps.cp_group) == (1, singleton)


def test_qat_router_replay_uses_each_prefetched_microbatch_cp_group(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _MixedCPScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)
    pool, singleton = _Group(2), _Group(1)
    ps = SimpleNamespace(
        dp_size=2,
        dp_rank=0,
        dp_group=pool,
        dp_cp_group=pool,
        cp_size=1,
        cp_rank=0,
        cp_group=singleton,
        pp_size=2,
    )
    seen = []

    class Protocol:
        def pack_routed_experts(self, _model, _batch, routed):
            seen.append(ps.cp_size)
            return routed

    handle = ModelHandle(
        model=object(),
        parallel_state=ps,
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=2, vpp=1),
            impl_cfg={"use_thd": True, "qat": {"enabled": True}},
        ),
        _extras={"forward_step": lambda *_args: {}, "protocol": Protocol()},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 4},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
    )
    plugin.initialize(handle)
    routes = torch.nested.as_nested_tensor(
        [
            torch.ones(1, 1, 1, dtype=torch.long),
            torch.ones(7, 1, 1, dtype=torch.long),
            torch.ones(1, 1, 1, dtype=torch.long),
        ],
        layout=torch.jagged,
    )
    batch = PackedBatch(
        input_ids=torch.arange(12),
        labels=torch.arange(12),
        seq_lens=torch.tensor([2, 8, 2]),
        routed_experts=routes,
        r3_replay_mask=torch.ones(12, dtype=torch.bool),
    )

    def pipeline_like_forward_backward(
        inner_handle, data, _loss, *, num_microbatches, **_kwargs
    ):
        prefetched = [
            split_loss_context(next(data))[0] for _ in range(num_microbatches)
        ]
        for selected in prefetched:
            inner_handle._extras["protocol"].pack_routed_experts(
                inner_handle._model, selected, selected.routed_experts
            )
            inner_handle._extras["forward_step"](inner_handle._model, selected)

    plugin.wrap_forward_backward(pipeline_like_forward_backward)(
        handle, batch, None, router_replay={"action": "replay"}
    )

    assert seen == [1, 2]
    assert handle._extras["protocol"].__class__ is Protocol


def test_dynamic_cp_router_replay_rejects_fused_router(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    model = torch.nn.Module()
    model.router = torch.nn.Module()
    model.router.moe_router_fusion = True
    handle = SimpleNamespace(
        _model=model,
        _extras={"forward_step": lambda *_args: {}},
        config=SimpleNamespace(impl_cfg={"qat": {"enabled": True}}),
    )
    plugin = DynamicCPPlugin({"max_seqlen_per_dp_cp_rank": 8})
    monkeypatch.setattr(
        plugin, "_prepare", lambda *_args: pytest.fail("must fail before prepare")
    )

    with pytest.raises(NotImplementedError, match="moe_router_fusion=False"):
        plugin.wrap_forward_backward(lambda *_args, **_kwargs: None)(
            handle, object(), None, router_replay={"action": "replay"}
        )


def test_dynamic_cp_split_merge_preserves_jagged_tensordict_samples():
    TensorDict = pytest.importorskip("tensordict").TensorDict
    NonTensorData = pytest.importorskip("tensordict.tensorclass").NonTensorData
    from megatron.lite.runtime.backends.mlite.dynamic_cp import (
        _merge_source,
        _split_source,
    )

    source = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(
                [torch.arange(3), torch.arange(5)], layout=torch.jagged
            ),
            "loss_mask": torch.nested.as_nested_tensor(
                [torch.ones(3), torch.ones(5)], layout=torch.jagged
            ),
            "temperature": torch.tensor([0.5, 0.75]),
            "metadata": ["first", "second"],
            "pad_mode": ["no_padding", "no_padding"],
            "scalar_control": NonTensorData(data=torch.tensor(3), batch_size=[2]),
            "vector_control": NonTensorData(data=torch.tensor([3, 4]), batch_size=[2]),
        },
        batch_size=[2],
    )

    samples = _split_source(source, count=2)
    merged = _merge_source([samples[1], samples[0]], torch.device("cpu"))

    assert merged.batch_size == torch.Size([2])
    assert merged["input_ids"].is_nested
    assert [row.tolist() for row in merged["input_ids"].unbind()] == [
        [0, 1, 2, 3, 4],
        [0, 1, 2],
    ]
    assert [row.tolist() for row in merged["loss_mask"].unbind()] == [
        [1.0] * 5,
        [1.0] * 3,
    ]
    assert torch.equal(merged["temperature"], torch.tensor([0.75, 0.5]))
    assert list(merged["metadata"]) == ["second", "first"]
    assert isinstance(merged.get("pad_mode"), NonTensorData)
    assert merged.get("pad_mode").data == "no_padding"
    assert isinstance(merged.get("scalar_control"), NonTensorData)
    assert torch.equal(merged.get("scalar_control").data, torch.tensor(3))
    assert isinstance(merged.get("vector_control"), NonTensorData)
    assert torch.equal(merged.get("vector_control").data, torch.tensor([3, 4]))


def test_dynamic_cp_merge_rejects_mismatched_tensordict_keys():
    TensorDict = pytest.importorskip("tensordict").TensorDict
    from megatron.lite.runtime.backends.mlite.dynamic_cp import (
        _merge_source,
        _split_source,
    )

    common = {
        "input_ids": torch.nested.as_nested_tensor(
            [torch.arange(3)], layout=torch.jagged
        ),
        "temperature": torch.tensor([0.5]),
    }
    left = TensorDict(common, batch_size=[1])
    right = TensorDict({**common, "extra": torch.tensor([1])}, batch_size=[1])
    parts = [_split_source(left, 1)[0], _split_source(right, 1)[0]]

    with pytest.raises(ValueError, match="incompatible nested source keys"):
        _merge_source(parts, torch.device("cpu"))


def test_dynamic_cp_exposes_logical_dp_one_without_replacing_physical_dp(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    physical_dp_group = _Group(4, rank=2)
    pool_group = _Group(4, rank=2)
    logical_dp_group = _Group(1)
    cp2_group = _Group(2)
    ps = SimpleNamespace(
        dp_size=4,
        dp_rank=2,
        dp_group=physical_dp_group,
        dp_cp_group=pool_group,
        dp_cp_size=4,
        cp_size=1,
        pp_size=1,
    )
    handle = ModelHandle(
        model=object(),
        parallel_state=ps,
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 8},
        create_groups=lambda _ps, _minimum, _parallel: {
            1: logical_dp_group,
            2: cp2_group,
            4: pool_group,
        },
    )

    plugin.initialize(handle)

    assert handle.dp_size == 1
    assert handle.dp_rank == 0
    assert handle.dp_group is logical_dp_group
    assert handle.metric_group is pool_group
    assert ps.dp_size == 4
    assert ps.dp_rank == 2
    assert ps.dp_group is physical_dp_group


def test_dynamic_cp_enables_transformer_engine_batched_p2p(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    monkeypatch.setenv("NVTE_BATCH_MHA_P2P_COMM", "0")
    pool, singleton, cp2 = _Group(4), _Group(1), _Group(2)
    ps = SimpleNamespace(
        dp_size=4,
        dp_rank=0,
        dp_group=pool,
        dp_cp_group=pool,
        cp_size=1,
        pp_size=1,
    )
    handle = ModelHandle(
        model=object(),
        parallel_state=ps,
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 8},
        create_groups=lambda _ps, _minimum, _parallel: {
            1: singleton,
            2: cp2,
            4: pool,
        },
    )

    plugin.initialize(handle)

    assert __import__("os").environ["NVTE_BATCH_MHA_P2P_COMM"] == "1"


def test_install_wraps_only_the_target_runtime_instance():
    from megatron.lite.runtime.backends.mlite.dynamic_cp import install

    class Runtime:
        def build_model(self):
            return "handle"

        def forward_backward(self, *args, **kwargs):
            return args, kwargs

    enabled = Runtime()
    disabled = Runtime()

    install(enabled, {"enabled": True, "max_seqlen_per_dp_cp_rank": 8})

    assert "forward_backward" in enabled.__dict__
    assert "build_model" in enabled.__dict__
    assert "forward_backward" not in disabled.__dict__
    assert "build_model" not in disabled.__dict__


def test_dynamic_cp_plugin_is_explicitly_disabled_by_default():
    from megatron.lite.runtime.backends.mlite.dynamic_cp import install

    class Runtime:
        def build_model(self):
            return "handle"

        def forward_backward(self, *args, **kwargs):
            return args, kwargs

    runtime = Runtime()
    install(runtime, {"max_seqlen_per_dp_cp_rank": 8})

    assert "forward_backward" not in runtime.__dict__
    assert "build_model" not in runtime.__dict__


def test_disabled_runtime_does_not_import_dynamic_cp(monkeypatch):
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "megatron.lite.runtime.backends.mlite.dynamic_cp":
            pytest.fail("disabled runtime must not import the Dynamic CP plugin")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    runtime = MegatronLiteRuntime("", MegatronLiteConfig(impl_cfg={"use_thd": True}))

    assert "forward_backward" not in runtime.__dict__
    assert "build_model" not in runtime.__dict__


def test_runtime_config_installs_dynamic_cp_sidecar():
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    runtime = MegatronLiteRuntime(
        "",
        MegatronLiteConfig(
            impl_cfg={
                "use_thd": True,
                "runtime_plugins": {
                    "dynamic_context_parallel": {
                        "enabled": True,
                        "max_seqlen_per_dp_cp_rank": 8,
                    }
                },
            }
        ),
    )

    assert "forward_backward" in runtime.__dict__
    assert "build_model" in runtime.__dict__


def test_runtime_plugins_must_be_a_mapping():
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    with pytest.raises(TypeError, match="runtime_plugins must be a mapping"):
        MegatronLiteRuntime(
            "", MegatronLiteConfig(impl_cfg={"runtime_plugins": object()})
        )


class _CrossCutScheduler:
    def __init__(self, **kwargs):
        assert kwargs["dp_size"] == 1
        assert kwargs["cp_size"] == 2
        assert kwargs["max_seqlen_per_dp_cp_rank"] == 3

    def get_groups_and_subsamples(self, sample_lengths):
        # Length 5 must cross the CP=2 cut when one rank has capacity 3.
        assert sample_lengths == [(0, 5), (1, 2)]
        return [[[0, 1], [0, 1]]]


class _SplitScheduler:
    def __init__(self, **_kwargs):
        pass

    def get_groups_and_subsamples(self, sample_lengths):
        assert sample_lengths == [(0, 1), (1, 1)]
        return [[[0], [1]]]


class _MixedCPScheduler:
    def __init__(self, **_kwargs):
        pass

    def get_groups_and_subsamples(self, sample_lengths):
        assert sample_lengths == [(0, 2), (1, 8), (2, 2)]
        return [[[0], [2]], [[1], [1]]]


class _CP4OnlyScheduler:
    def __init__(self, **_kwargs):
        pass

    def get_groups_and_subsamples(self, sample_lengths):
        assert sample_lengths == [(0, 16)]
        return [[[0], [0], [0], [0]]]


def _install_fake_scheduler(monkeypatch):
    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _CrossCutScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)


def test_mixed_cp_plan_reports_each_sample_group_and_histogram(monkeypatch, capsys):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _MixedCPScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)
    pool, singleton = _Group(2), _Group(1)
    ps = SimpleNamespace(
        dp_size=2,
        dp_rank=0,
        dp_group=pool,
        dp_cp_group=pool,
        cp_size=1,
        cp_rank=0,
        cp_group=singleton,
        pp_size=1,
    )
    handle = ModelHandle(
        model=object(),
        parallel_state=ps,
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={"forward_step": lambda *_args: {}},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 4},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
    )
    plugin.initialize(handle)
    prepared = plugin._prepare(
        handle,
        PackedBatch(
            input_ids=torch.arange(12),
            labels=torch.arange(12),
            seq_lens=torch.tensor([2, 8, 2]),
        ),
        None,
        1,
    )

    local_cp_sizes = []
    for item in prepared.data:
        selected, _context = split_loss_context(item)
        local_cp_sizes.append(selected.extras["_mlite_dcp_local_cp_size"])
    prepared.finish(require_complete=True)

    assert local_cp_sizes == [1, 2]
    assert capsys.readouterr().out.splitlines() == [
        "MLITE_DYNAMIC_CP_PLAN step=0 cp_size_space=[1,2] "
        'cp_size_histogram={"1":2,"2":1} '
        'groups=[{"cp_size":1,"ranks":[0],"sample_ids":[0]},'
        '{"cp_size":1,"ranks":[1],"sample_ids":[2]},'
        '{"cp_size":2,"ranks":[0,1],"sample_ids":[1]}] global_num_tokens=None'
    ]


def test_mixed_cp_rebinds_context_parallel_modules_and_restores_physical_group(
    monkeypatch,
):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _MixedCPScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)

    pool, singleton = _Group(2), _Group(1)
    group_ranks = {id(pool): [0, 2], id(singleton): [0]}
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        lambda group: group_ranks[id(group)],
    )

    class ContextParallelAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cp_group = pool
            self.cp_global_ranks = [0, 2]
            self.cp_stream = object()
            self.cp_comm_type = "p2p"

        def set_context_parallel_group(
            self, cp_group, cp_global_ranks, cp_stream, cp_comm_type="p2p"
        ):
            self.cp_group = cp_group
            self.cp_global_ranks = cp_global_ranks
            self.cp_stream = cp_stream
            self.cp_comm_type = cp_comm_type

    model = torch.nn.Module()
    model.attention = ContextParallelAttention()
    ps = SimpleNamespace(
        dp_size=1,
        dp_rank=0,
        dp_group=singleton,
        dp_cp_group=pool,
        cp_size=2,
        cp_rank=0,
        cp_group=pool,
        cp_global_ranks=[0, 2],
        pp_size=1,
    )
    handle = ModelHandle(
        model=model,
        parallel_state=ps,
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=2, pp=1, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={"forward_step": lambda *_args: {}},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 4},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
    )
    plugin.initialize(handle)
    prepared = plugin._prepare(
        handle,
        PackedBatch(
            input_ids=torch.arange(12),
            labels=torch.arange(12),
            seq_lens=torch.tensor([2, 8, 2]),
        ),
        None,
        1,
    )

    next(prepared.data)
    assert (ps.cp_size, ps.cp_group, ps.cp_global_ranks) == (1, singleton, [0])
    assert model.attention.cp_group is singleton
    assert model.attention.cp_global_ranks == [0]

    next(prepared.data)
    assert (ps.cp_size, ps.cp_group, ps.cp_global_ranks) == (2, pool, [0, 2])
    assert model.attention.cp_group is pool
    assert model.attention.cp_global_ranks == [0, 2]

    prepared.finish(require_complete=True)
    assert (ps.cp_size, ps.cp_group, ps.cp_global_ranks) == (2, pool, [0, 2])
    assert model.attention.cp_group is pool
    assert model.attention.cp_global_ranks == [0, 2]


def test_mixed_cp_uses_pool_global_token_count_for_loss_normalization(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _MixedCPScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)

    def run_rank(rank):
        pool, singleton = _Group(2, rank), _Group(1)

        def sum_uneven_owned_tokens(value, *, group):
            assert group is pool
            owned = 7 if rank == 0 else 1
            peer = 1 if rank == 0 else 7
            assert torch.equal(value, torch.tensor(owned, dtype=torch.int64))
            value.add_(peer)

        monkeypatch.setattr(torch.distributed, "all_reduce", sum_uneven_owned_tokens)
        ps = SimpleNamespace(
            dp_size=2,
            dp_rank=rank,
            dp_group=pool,
            dp_cp_group=pool,
            cp_size=1,
            cp_rank=0,
            cp_group=singleton,
            pp_size=1,
        )
        handle = ModelHandle(
            model=object(),
            parallel_state=ps,
            config=SimpleNamespace(
                parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
                impl_cfg={"use_thd": True},
            ),
            _extras={"forward_step": lambda *_args: {}},
        )
        plugin = DynamicCPPlugin(
            {"max_seqlen_per_dp_cp_rank": 4},
            create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
        )
        plugin.initialize(handle)
        prepared = plugin._prepare(
            handle,
            iter(
                [
                    (
                        PackedBatch(
                            input_ids=torch.arange(12),
                            labels=torch.arange(12),
                            seq_lens=torch.tensor([2, 8, 2]),
                            loss_mask=torch.tensor(
                                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
                            ),
                        ),
                        LossContext(loss_scale=1 / 4, source_batch=torch.zeros(3, 1)),
                    )
                ]
            ),
            lambda _output, selected, context: (
                selected.loss_mask.sum(dtype=torch.float32) * context.loss_scale,
                {},
            ),
            1,
        )
        losses = []
        token_counts = []
        for item in prepared.data:
            selected, context = split_loss_context(item)
            loss, _metrics = prepared.loss({}, selected, context)
            losses.append(loss)
            token_counts.append(selected.extras["_mlite_dcp_global_num_tokens"])
        prepared.finish(require_complete=True)
        return sum(losses) / len(losses), token_counts

    rank0_loss, rank0_tokens = run_rank(0)
    rank1_loss, rank1_tokens = run_rank(1)
    dcp_global_loss = (rank0_loss + rank1_loss) / 2
    baseline_tokens = torch.tensor(7, dtype=torch.int64) + torch.tensor(
        1, dtype=torch.int64
    )
    baseline_global_loss = torch.tensor(8.0) / baseline_tokens

    assert rank0_tokens == rank1_tokens == [8, 8]
    assert torch.equal(baseline_tokens, torch.tensor(8, dtype=torch.int64))
    assert torch.equal(dcp_global_loss, baseline_global_loss)


def test_runtime_collector_reports_application_loss_not_backward_scaled_loss(
    monkeypatch,
):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _SplitScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, group: None)

    def gather(output, records, *, group):
        output[:] = [
            records,
            [
                {
                    "sample_ids": [1],
                    "model_output": {"values": [torch.tensor([2.0])]},
                    "loss": 1.0,
                    "metrics": {},
                }
            ],
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    pool, singleton = _Group(2), _Group(1)
    handle = ModelHandle(
        model=object(),
        parallel_state=SimpleNamespace(
            dp_size=2,
            dp_rank=0,
            dp_group=pool,
            dp_cp_group=pool,
            cp_size=1,
            cp_rank=0,
            cp_group=singleton,
            pp_size=1,
        ),
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={"forward_step": lambda *_args: {}},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 1},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
    )
    plugin.initialize(handle)
    collector = []

    def loss_fn(_output, _batch, _context):
        return torch.tensor(3.0), {}

    loss_fn.runtime_output_collector = collector
    loss_fn.runtime_output_extractor = lambda output: output
    loss_fn.runtime_output_loss_scale = 1 / 3
    prepared = plugin._prepare(
        handle,
        iter(
            [
                (
                    PackedBatch(
                        input_ids=torch.tensor([1]),
                        labels=torch.tensor([1]),
                        seq_lens=torch.tensor([1]),
                        loss_mask=torch.ones(1),
                    ),
                    LossContext(loss_scale=0.25, source_batch=torch.tensor([[0.0]])),
                ),
                (
                    PackedBatch(
                        input_ids=torch.tensor([2]),
                        labels=torch.tensor([2]),
                        seq_lens=torch.tensor([1]),
                        loss_mask=torch.ones(1),
                    ),
                    LossContext(loss_scale=0.25, source_batch=torch.tensor([[0.0]])),
                ),
            ]
        ),
        loss_fn,
        2,
    )

    batch, context = split_loss_context(next(prepared.data))
    scaled_loss, _metrics = prepared.loss(
        {"values": torch.tensor([[1.0]])}, batch, context
    )

    # correction=8, schedule_scale=1/2, replica_scale=2: all are non-unit.
    assert torch.equal(scaled_loss, torch.tensor(24.0))
    prepared.finish(require_complete=True)
    assert [record["loss"] for record in collector] == [1.0, 1.0]


@pytest.mark.parametrize("with_context", [False, True])
def test_dynamic_cp_missing_loss_mask_fails_before_normalization_can_degrade(
    monkeypatch, with_context
):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _SplitScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)
    pool, singleton = _Group(2), _Group(1)
    handle = ModelHandle(
        model=object(),
        parallel_state=SimpleNamespace(
            dp_size=2,
            dp_rank=0,
            dp_group=pool,
            dp_cp_group=pool,
            cp_size=1,
            cp_rank=0,
            cp_group=singleton,
            pp_size=1,
        ),
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={"forward_step": lambda *_args: {}},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 1},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
    )
    plugin.initialize(handle)

    with pytest.raises(
        ValueError,
        match="requires loss_mask on every sample for pool-global loss normalization",
    ):
        plugin._prepare(
            handle,
            iter(
                [
                    (
                        PackedBatch(
                            input_ids=torch.tensor([1, 2]),
                            labels=torch.tensor([1, 2]),
                            seq_lens=torch.tensor([1, 1]),
                        ),
                        (
                            LossContext(
                                loss_scale=0.5,
                                source_batch=torch.tensor([[0.0], [0.0]]),
                            )
                            if with_context
                            else None
                        ),
                    )
                ]
            ),
            lambda *_args: (torch.tensor(1.0), {}),
            1,
        )


def test_required_cp_size_coverage_fails_loudly(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _CP4OnlyScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)
    pool, singleton, cp2 = _Group(4), _Group(1), _Group(2)
    ps = SimpleNamespace(
        dp_size=4,
        dp_rank=0,
        dp_group=pool,
        dp_cp_group=pool,
        cp_size=1,
        cp_rank=0,
        cp_group=singleton,
        pp_size=1,
    )
    handle = ModelHandle(
        model=object(),
        parallel_state=ps,
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={"forward_step": lambda *_args: {}},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 4, "require_full_cp_size_coverage": True},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: cp2, 4: pool},
    )
    plugin.initialize(handle)

    with pytest.raises(
        RuntimeError,
        match=r"did not cover required cp_size values \[1, 2\]; expected \[1, 2, 4\]",
    ):
        plugin._prepare(
            handle,
            PackedBatch(
                input_ids=torch.arange(16),
                labels=torch.arange(16),
                seq_lens=torch.tensor([16]),
            ),
            None,
            1,
        )


def test_logical_dp_loss_compensates_physical_pool_average(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _SplitScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)
    pool, singleton = _Group(2), _Group(1)
    ps = SimpleNamespace(
        dp_size=2,
        dp_rank=0,
        dp_group=pool,
        dp_cp_group=pool,
        cp_size=1,
        cp_rank=0,
        cp_group=singleton,
        pp_size=1,
    )
    handle = ModelHandle(
        model=object(),
        parallel_state=ps,
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={"forward_step": lambda *_args: {}},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 1},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
    )
    plugin.initialize(handle)

    def sum_pool_tokens(value, *, group):
        assert group is pool
        value.add_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", sum_pool_tokens)
    prepared = plugin._prepare(
        handle,
        iter(
            [
                (
                    PackedBatch(
                        input_ids=torch.tensor([1, 2]),
                        labels=torch.tensor([1, 2]),
                        seq_lens=torch.tensor([1, 1]),
                        loss_mask=torch.ones(2),
                    ),
                    LossContext(source_batch=torch.tensor([[0.0], [0.0]])),
                )
            ]
        ),
        lambda *_args: (torch.tensor(1.0), {}),
        1,
    )
    selected, context = next(prepared.data)
    loss, _ = prepared.loss({}, selected, context)

    # Two one-token leaders contribute one normalized global loss after the
    # physical-pool DDP average; neither rank silently keeps a half-loss.
    assert torch.equal(loss, torch.tensor(1.0))
    prepared.finish(require_complete=True)


def test_restore_outputs_merges_metrics_from_every_distinct_leader(monkeypatch):
    from megatron.lite.runtime.backends.mlite import dynamic_cp

    pool = _Group(2)
    rank0_records = [
        {
            "sample_ids": [0],
            "model_output": {"values": [torch.tensor([10.0])]},
            "loss": 1.0,
            "metrics": {"score": 1.0},
        }
    ]
    rank1_records = [
        {
            "sample_ids": [1],
            "model_output": {"values": [torch.tensor([30.0])]},
            "loss": 3.0,
            "metrics": {"score": 3.0},
        }
    ]

    def gather(output, records, *, group):
        assert group is pool
        assert records is rank0_records
        output[:] = [rank0_records, rank1_records]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    collector = []
    dynamic_cp._restore_outputs(
        collector,
        rank0_records,
        pool=pool,
        input_groups=[[0, 1]],
        device=torch.device("cpu"),
    )

    assert len(collector) == 1
    scores = collector[0]["metrics"]["score"]
    assert scores == [1.0, 3.0]
    assert sum(scores) / len(scores) == 2.0
    assert sum(scores) / len(scores) not in scores
    assert sum(item.get("loss", 0.0) for item in collector) == 4.0


def test_restore_outputs_preserves_metric_aggregator_across_leaders(monkeypatch):
    from megatron.lite.runtime.backends.mlite import dynamic_cp

    class Metric:
        def __init__(self, values):
            self.values = list(values)

        def init_list(self):
            return Metric([])

        def append(self, value):
            self.values.extend(value.values)

    pool = _Group(2)
    rank0_records = [
        {
            "sample_ids": [0],
            "model_output": {},
            "loss": 0.0,
            "metrics": {"score": Metric([1.0])},
        }
    ]
    rank1_records = [
        {
            "sample_ids": [1],
            "model_output": {},
            "loss": 0.0,
            "metrics": {"score": Metric([3.0])},
        }
    ]
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, _records, group: output.__setitem__(
            slice(None), [rank0_records, rank1_records]
        ),
    )
    collector = []

    dynamic_cp._restore_outputs(
        collector,
        rank0_records,
        pool=pool,
        input_groups=[[0, 1]],
        device=torch.device("cpu"),
    )

    assert collector[0]["metrics"]["score"].values == [1.0, 3.0]


def test_restore_outputs_rejects_mismatched_model_output_keys(monkeypatch):
    from megatron.lite.runtime.backends.mlite import dynamic_cp

    pool = _Group(2)
    rank0_records = [
        {
            "sample_ids": [0],
            "model_output": {"values": [torch.tensor([10.0])]},
            "loss": 0.0,
            "metrics": {},
        }
    ]
    rank1_records = [
        {
            "sample_ids": [1],
            "model_output": {
                "values": [torch.tensor([30.0])],
                "logits": [torch.tensor([3.0])],
            },
            "loss": 0.0,
            "metrics": {},
        }
    ]
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, _records, group: output.__setitem__(
            slice(None), [rank0_records, rank1_records]
        ),
    )

    with pytest.raises(RuntimeError, match="incompatible model output keys"):
        dynamic_cp._restore_outputs(
            [],
            rank0_records,
            pool=pool,
            input_groups=[[0, 1]],
            device=torch.device("cpu"),
        )


def _loss_fn(kind: str, collector: list[dict]):
    def calculate(output, _batch, context):
        values = output["values"]
        source = context.source_batch
        if kind == "sft":
            loss = ((values - source) ** 2).sum()
        else:
            loss = -(values * source).sum()
        if not getattr(calculate, "runtime_collects_outputs", False):
            collector.append(
                {
                    "model_output": {"values": values.detach()},
                    "loss": float(loss.detach()),
                    "metrics": {"kind": kind},
                }
            )
        return loss, {"kind": kind}

    calculate.runtime_output_collector = collector
    calculate.runtime_output_extractor = lambda output: {"values": output["values"]}
    return calculate


def _run_loop(
    handle, data, loss_fn, *, num_microbatches=1, forward_only=False, **_kwargs
):
    loss = None
    for _ in range(num_microbatches):
        batch, context = split_loss_context(next(data))
        output = handle._extras["forward_step"](handle._model, batch)
        loss, _ = loss_fn(output, batch, context)
        if not forward_only:
            loss.backward()
    return ForwardResult(model_output=ModelOutputs(loss=loss.detach()))


@pytest.mark.parametrize(
    "kind,source", [("sft", [[1.0], [3.0]]), ("rl", [[2.0], [-1.0]])]
)
def test_sft_and_rl_true_loss_match_disabled_reference_across_cp_cut(
    monkeypatch, kind, source
):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    _install_fake_scheduler(monkeypatch)
    pool = _Group(2)
    singleton = _Group(1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, records, group: output.__setitem__(slice(None), [records, []]),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, group: None)

    batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        labels=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        seq_lens=torch.tensor([5, 2]),
        loss_mask=torch.ones(7),
    )
    source_batch = torch.tensor(source)

    def make_handle(weight):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(weight)
        ps = SimpleNamespace(
            dp_size=2,
            dp_rank=0,
            dp_group=pool,
            dp_cp_group=pool,
            cp_size=1,
            cp_rank=0,
            cp_group=singleton,
            pp_size=1,
        )

        def forward(module, selected):
            offsets = selected.cu_seqlens.tolist()
            rows = [
                selected.input_ids[start:end].float().mean().reshape(1)
                for start, end in zip(offsets[:-1], offsets[1:], strict=True)
            ]
            return {"values": module(torch.stack(rows))}

        return ModelHandle(
            model=model,
            parallel_state=ps,
            config=SimpleNamespace(
                parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
                impl_cfg={"use_thd": True},
            ),
            _extras={"forward_step": forward},
        )

    baseline_handle = make_handle(0.25)
    baseline_records = []
    baseline_result = _run_loop(
        baseline_handle,
        iter([(batch, LossContext(loss_scale=1 / 7, source_batch=source_batch))]),
        _loss_fn(kind, baseline_records),
    )

    dcp_handle = make_handle(0.25)
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 3},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
    )
    plugin.initialize(dcp_handle)
    dcp_records = []
    dcp_result = plugin.wrap_forward_backward(_run_loop)(
        dcp_handle,
        iter([(batch, LossContext(loss_scale=1 / 7, source_batch=source_batch))]),
        _loss_fn(kind, dcp_records),
    )

    assert torch.equal(dcp_result.model_output.loss, baseline_result.model_output.loss)
    assert torch.equal(
        dcp_handle._model.weight.grad, baseline_handle._model.weight.grad
    )
    assert torch.equal(
        dcp_records[0]["model_output"]["values"].values(),
        baseline_records[0]["model_output"]["values"].reshape(-1),
    )
