# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for the deepepv2 settings: the TransformerConfig fields, the shared token bound and
the arguments that reach the DeepEP v2 buffer.

A recording fake stands in for DeepEP's ElasticBuffer, so these tests need neither a GPU nor
DeepEP. They patch module attributes and ``torch.distributed.all_reduce`` through ``monkeypatch``
only, and re-import nothing.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from megatron.core import _rank_utils
from megatron.core.transformer import transformer_config
from megatron.core.transformer.moe import fused_a2a, token_dispatcher
from megatron.core.transformer.transformer_config import TransformerConfig

pytestmark = pytest.mark.internal

# The real check, saved before the autouse fixture replaces it with a recorder.
check_uniform_num_max_tokens_per_rank = fused_a2a._check_uniform_num_max_tokens_per_rank

# Keyword names of ElasticBuffer.get_buffer_size_hint and ElasticBuffer.__init__ that exist in
# both DeepEP 2.1 (deep_ep/buffers/elastic.py:381 and :228 at 01dc3aaa) and DeepEP 2.5
# (deep_ep/buffers/ep.py:374 and :242 at f6aa38fe). The fake rejects any other keyword.
_HINT_PARAMETERS = frozenset(
    "num_max_tokens_per_rank hidden num_topk use_fp8_dispatch allow_hybrid_mode "
    "allow_multiple_reduction".split()
)
_INIT_PARAMETERS = frozenset(
    "num_bytes num_max_tokens_per_rank hidden num_topk use_fp8_dispatch deterministic "
    "allow_hybrid_mode allow_multiple_reduction prefer_overlap_with_compute sl_idx "
    "num_allocated_qps num_cpu_timeout_secs num_gpu_timeout_secs explicitly_destroy".split()
)
# The QP and SM counts the fake resolves when asked for 0 (automatic).
AUTO_NUM_QPS = 65
AUTO_NUM_SMS = 24


class RecordingBuffer:
    """Records the arguments of the size hint, the constructor and dispatch, in call order."""

    calls = []

    def __init__(self, group, **kwargs):
        assert set(kwargs) <= _INIT_PARAMETERS, set(kwargs) - _INIT_PARAMETERS
        RecordingBuffer.calls.append(("init", kwargs))
        self.group = group
        self.init_kwargs = kwargs
        self.num_bytes = kwargs["num_bytes"]
        self.num_max_tokens_per_rank = kwargs["num_max_tokens_per_rank"]
        self.num_allocated_qps = kwargs.get("num_allocated_qps", 0) or AUTO_NUM_QPS
        self.dispatch_kwargs = []

    @staticmethod
    def get_buffer_size_hint(group, **kwargs):
        assert set(kwargs) <= _HINT_PARAMETERS, set(kwargs) - _HINT_PARAMETERS
        RecordingBuffer.calls.append(("hint", kwargs))
        return kwargs["num_max_tokens_per_rank"] * kwargs["hidden"]

    def capture(self):
        return None

    def dispatch(self, x, **kwargs):
        self.dispatch_kwargs.append(kwargs)
        handle = SimpleNamespace(
            num_sms=kwargs["num_sms"] or AUTO_NUM_SMS, num_recv_tokens_per_expert_list=[x.shape[0]]
        )
        return x.clone(), kwargs["topk_idx"], kwargs["topk_weights"], handle, None

    def combine(self, x, **kwargs):
        return x.clone(), None, None


def calls_of(kind):
    return [kwargs for name, kwargs in RecordingBuffer.calls if name == kind]


@pytest.fixture(autouse=True)
def fake_deep_ep_v2(monkeypatch):
    """Install the recording buffer, start without a cached buffer and record the bound check."""
    monkeypatch.setattr(RecordingBuffer, "calls", [])
    monkeypatch.setattr(fused_a2a, "ElasticBuffer", RecordingBuffer, raising=False)
    monkeypatch.setattr(fused_a2a, "_elastic_buffer", None)
    monkeypatch.setattr(fused_a2a, "_elastic_buffer_settings", None)
    monkeypatch.setattr(fused_a2a, "_elastic_buffer_log_num_sms", False)
    # log_single_rank logs on rank 0 only, and CI runs this file on several ranks.
    monkeypatch.setattr(_rank_utils, "safe_get_rank", lambda: 0)
    # The real check runs a CUDA all-reduce; it has its own tests below.
    monkeypatch.setattr(
        fused_a2a,
        "_check_uniform_num_max_tokens_per_rank",
        lambda group, bound: RecordingBuffer.calls.append(
            ("check", dict(group=group, bound=bound))
        ),
    )


def make_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="deepepv2",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestDeepEPV2ConfigFields:
    def test_defaults_keep_the_current_behaviour(self):
        config = make_config()
        assert config.moe_deepepv2_num_max_tokens_per_rank is None
        assert config.moe_deepepv2_pad_uneven_dispatch_inputs is False
        assert config.moe_deepepv2_num_allocated_qps == 0
        assert config.moe_deepepv2_allow_hybrid_mode is True
        assert config.moe_deepepv2_deterministic is True

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(moe_deepepv2_num_max_tokens_per_rank=4096),
            dict(moe_deepepv2_pad_uneven_dispatch_inputs=True),
            dict(moe_deepepv2_num_allocated_qps=2),
            dict(moe_deepepv2_allow_hybrid_mode=False),
            dict(moe_deepepv2_deterministic=False),
        ],
    )
    def test_accepts_valid_settings(self, overrides):
        config = make_config(**overrides)
        for name, value in overrides.items():
            assert getattr(config, name) == value

    @pytest.mark.parametrize(
        "overrides,match",
        [
            (dict(moe_deepepv2_num_max_tokens_per_rank=0), "None or a positive integer, got 0"),
            (dict(moe_deepepv2_num_max_tokens_per_rank=-8), "None or a positive integer, got -8"),
            (dict(moe_deepepv2_num_allocated_qps=-1), r"0 \(automatic\) or a positive integer"),
            (
                dict(
                    moe_deepepv2_num_max_tokens_per_rank=4096,
                    moe_deepepv2_pad_uneven_dispatch_inputs=True,
                ),
                "not both",
            ),
            (
                dict(moe_flex_dispatcher_backend="deepep", moe_deepepv2_num_max_tokens_per_rank=8),
                "moe_flex_dispatcher_backend='deepepv2'",
            ),
            (
                dict(
                    moe_flex_dispatcher_backend="hybridep",
                    moe_deepepv2_pad_uneven_dispatch_inputs=True,
                ),
                "moe_flex_dispatcher_backend='deepepv2'",
            ),
            (
                dict(moe_token_dispatcher_type="alltoall", moe_deepepv2_num_max_tokens_per_rank=8),
                "moe_token_dispatcher_type='flex'",
            ),
        ],
    )
    def test_rejects_invalid_settings(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            make_config(**overrides)

    @pytest.mark.parametrize(
        "overrides,raises",
        [
            (dict(), True),
            (dict(moe_deepepv2_num_max_tokens_per_rank=4096), False),
            (dict(moe_deepepv2_pad_uneven_dispatch_inputs=True), False),
        ],
    )
    def test_sequence_packing_needs_a_bound_or_padding(self, monkeypatch, overrides, raises):
        # Sequence packing checks the Transformer Engine version first; this host may have none.
        monkeypatch.setattr(transformer_config, "is_te_min_version", lambda *args, **kwargs: True)
        packing = dict(sequence_packing_scheduler="dp_balanced", max_seqlen_per_dp_cp_rank=64)
        if raises:
            with pytest.raises(ValueError, match="sequence_packing with deepepv2 requires"):
                make_config(**packing, **overrides)
        else:
            make_config(**packing, **overrides)

    def test_deterministic_mode_requires_deepepv2_determinism(self):
        assert make_config(deterministic_mode=True).moe_deepepv2_deterministic is True
        with pytest.raises(ValueError, match="requires moe_deepepv2_deterministic=True"):
            make_config(deterministic_mode=True, moe_deepepv2_deterministic=False)

    def test_pad_to_capacity_error_names_deepepv2(self):
        with pytest.raises(ValueError, match="the deepepv2 backend does not support"):
            make_config(moe_expert_capacity_factor=1.0, moe_pad_expert_input_to_capacity=True)

    def test_flex_dispatcher_rejects_tp_ep_of_one_with_value_error(self):
        # The process groups are None, so TP and EP are both 1; a ValueError survives python -O.
        pg_collection = SimpleNamespace(ep=None, expt_tp=None, tp_ep=None)
        with pytest.raises(ValueError, match="DeepEP v2 dispatcher requires TPxEP > 1"):
            token_dispatcher.MoEFlexTokenDispatcher(
                num_local_experts=4,
                local_expert_indices=[0, 1, 2, 3],
                config=make_config(),
                pg_collection=pg_collection,
            )


class TestGetElasticBuffer:
    group = object()

    def test_first_build_checks_the_bound_then_passes_the_settings(self):
        buffer = fused_a2a.get_elastic_buffer(
            self.group, num_max_tokens_per_rank=8, hidden=16, num_topk=2
        )

        assert [name for name, _ in RecordingBuffer.calls] == ["check", "hint", "init"]
        assert calls_of("check") == [dict(group=self.group, bound=8)]
        assert calls_of("hint") == [
            dict(num_max_tokens_per_rank=8, hidden=16, num_topk=2, allow_hybrid_mode=True)
        ]
        # deterministic=True is what deepepv2 always passed; the others are DeepEP's defaults.
        assert buffer.init_kwargs == dict(
            num_bytes=8 * 16,
            num_max_tokens_per_rank=8,
            hidden=16,
            num_topk=2,
            deterministic=True,
            allow_hybrid_mode=True,
            num_allocated_qps=0,
        )

    def test_cached_buffer_is_reused_without_a_collective(self):
        first = fused_a2a.get_elastic_buffer(self.group, 8, 16, 2)
        second = fused_a2a.get_elastic_buffer(self.group, 8, 16, 2)

        assert second is first
        assert len(calls_of("check")) == 1 and len(calls_of("init")) == 1

    @pytest.mark.parametrize(
        "setting",
        [dict(allow_hybrid_mode=False), dict(num_allocated_qps=2), dict(deterministic=False)],
    )
    def test_rebuilds_when_a_setting_changes(self, setting):
        first = fused_a2a.get_elastic_buffer(self.group, 8, 16, 2)
        second = fused_a2a.get_elastic_buffer(self.group, 8, 16, 2, **setting)
        third = fused_a2a.get_elastic_buffer(self.group, 8, 16, 2, **setting)

        assert second is not first and third is second
        name, value = next(iter(setting.items()))
        assert second.init_kwargs[name] == value
        hybrid = setting.get("allow_hybrid_mode", True)
        assert [hint["allow_hybrid_mode"] for hint in calls_of("hint")] == [True, hybrid, hybrid]
        # The group did not change, so the bound is not checked again.
        assert len(calls_of("check")) == 1

    def test_rebuilds_only_when_the_bound_grows(self):
        first = fused_a2a.get_elastic_buffer(self.group, 8, 16, 2)
        assert fused_a2a.get_elastic_buffer(self.group, 6, 16, 2) is first
        grown = fused_a2a.get_elastic_buffer(self.group, 12, 16, 2)

        assert grown is not first and grown.num_max_tokens_per_rank == 12

    def test_a_new_group_is_checked_again(self):
        other_group = object()
        fused_a2a.get_elastic_buffer(self.group, 8, 16, 2)
        buffer = fused_a2a.get_elastic_buffer(other_group, 8, 16, 2)

        assert buffer.group is other_group
        assert calls_of("check") == [
            dict(group=self.group, bound=8),
            dict(group=other_group, bound=8),
        ]

    def test_logs_the_resolved_qp_count_at_each_build(self, caplog):
        with caplog.at_level(logging.INFO, logger=fused_a2a.__name__):
            fused_a2a.get_elastic_buffer(self.group, 8, 16, 2)
            fused_a2a.get_elastic_buffer(self.group, 8, 16, 2)
            fused_a2a.get_elastic_buffer(self.group, 8, 16, 2, num_allocated_qps=2)

        messages = [r.getMessage() for r in caplog.records if "DeepEP v2 buffer" in r.getMessage()]
        assert len(messages) == 2
        assert f"num_allocated_qps={AUTO_NUM_QPS} (requested 0" in messages[0]
        assert "num_allocated_qps=2 (requested 2" in messages[1]


class TestUniformBoundCheck:
    """The real check, with a fake all-reduce that stands in for the other ranks."""

    group = object()

    def run_check(self, monkeypatch, local_bound, other_bounds):
        def fake_all_reduce(tensor, op=None, group=None):
            assert op == torch.distributed.ReduceOp.MAX and group is self.group
            assert tensor.dtype == torch.int64 and tensor.tolist() == [local_bound, -local_bound]
            bounds = [local_bound] + other_bounds
            tensor.copy_(torch.tensor([max(bounds), max(-b for b in bounds)]))

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        check_uniform_num_max_tokens_per_rank(self.group, local_bound, device="cpu")

    def test_equal_bounds_pass(self, monkeypatch):
        self.run_check(monkeypatch, 8, [8, 8, 8])

    @pytest.mark.parametrize("local_bound,other_bounds", [(8, [12, 8, 4]), (8, [8, 8, 9])])
    def test_unequal_bounds_raise_on_every_rank(self, monkeypatch, local_bound, other_bounds):
        low, high = min([local_bound] + other_bounds), max([local_bound] + other_bounds)
        with pytest.raises(ValueError, match=f"values from {low} to {high}") as error:
            self.run_check(monkeypatch, local_bound, other_bounds)
        assert "moe_deepepv2_num_max_tokens_per_rank" in str(error.value)
        assert "moe_deepepv2_pad_uneven_dispatch_inputs=True" in str(error.value)


def deepepv2_dispatch(
    buffer,
    x,
    token_indices,
    token_probs,
    num_experts,
    num_max_tokens_per_rank,
    expert_alignment=1,
    num_sms=0,
    async_finish=False,
    allocate_on_comm_stream=False,
):
    """fused_a2a.deepepv2_dispatch, which exists only when DeepEP v2 is installed."""
    return fused_a2a.DeepepV2Dispatch.apply(
        buffer,
        x.contiguous(),
        token_indices,
        token_probs,
        num_experts,
        num_max_tokens_per_rank,
        expert_alignment,
        num_sms,
        async_finish,
        allocate_on_comm_stream,
    )


class TestDeepepV2ManagerBound:
    group = object()

    def make_manager(self, monkeypatch, **overrides):
        monkeypatch.setattr(token_dispatcher, "deepepv2_dispatch", deepepv2_dispatch)
        monkeypatch.setattr(token_dispatcher, "deepepv2_combine", lambda *args, **kwargs: None)
        manager = token_dispatcher._DeepepV2Manager(
            self.group,
            num_local_experts=2,
            router_topk=2,
            num_experts=4,
            config=make_config(moe_router_dtype="fp32", **overrides),
        )
        return manager

    def dispatch(self, manager, num_tokens):
        manager.token_indices = torch.zeros((num_tokens, 2), dtype=torch.int64)
        manager.token_probs = torch.full((num_tokens, 2), 0.5)
        manager.dispatch(torch.ones((num_tokens, 16)))
        return manager.buffer

    def no_manager_all_reduce(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("the manager must not all-reduce without the pad option")

        monkeypatch.setattr(torch.distributed, "all_reduce", fail)

    def test_default_passes_the_local_count(self, monkeypatch):
        self.no_manager_all_reduce(monkeypatch)
        buffer = self.dispatch(self.make_manager(monkeypatch), 6)

        assert calls_of("check") == [dict(group=self.group, bound=6)]
        assert buffer.num_max_tokens_per_rank == 6
        assert buffer.dispatch_kwargs[0]["num_max_tokens_per_rank"] == 6

    def test_fixed_bound_reaches_the_buffer_and_every_dispatch(self, monkeypatch):
        self.no_manager_all_reduce(monkeypatch)
        manager = self.make_manager(monkeypatch, moe_deepepv2_num_max_tokens_per_rank=32)
        buffer = self.dispatch(manager, 6)
        assert self.dispatch(manager, 20) is buffer

        assert calls_of("check") == [dict(group=self.group, bound=32)]
        assert calls_of("hint")[0]["num_max_tokens_per_rank"] == 32
        assert buffer.num_max_tokens_per_rank == 32
        assert [kw["num_max_tokens_per_rank"] for kw in buffer.dispatch_kwargs] == [32, 32]

    def test_fixed_bound_rejects_more_tokens_before_communicating(self, monkeypatch):
        self.no_manager_all_reduce(monkeypatch)
        manager = self.make_manager(monkeypatch, moe_deepepv2_num_max_tokens_per_rank=4)

        with pytest.raises(ValueError, match="got 6 tokens on this rank, more than"):
            self.dispatch(manager, 6)
        assert RecordingBuffer.calls == []

    def test_pad_uses_the_group_maximum(self, monkeypatch):
        sent = []

        def fake_all_reduce(tensor, op=None, group=None):
            assert op == torch.distributed.ReduceOp.MAX and group is self.group
            sent.append(tensor.item())
            tensor.fill_(40 if len(sent) == 1 else 50)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        manager = self.make_manager(monkeypatch, moe_deepepv2_pad_uneven_dispatch_inputs=True)
        first = self.dispatch(manager, 6)
        second = self.dispatch(manager, 9)

        assert sent == [6, 9]
        assert calls_of("check") == [dict(group=self.group, bound=40)]
        assert first.num_max_tokens_per_rank == 40 and second.num_max_tokens_per_rank == 50
        assert first.dispatch_kwargs[0]["num_max_tokens_per_rank"] == 40
        assert second.dispatch_kwargs[0]["num_max_tokens_per_rank"] == 50

    def test_config_settings_reach_the_hint_and_the_buffer(self, monkeypatch):
        manager = self.make_manager(
            monkeypatch,
            moe_deepepv2_allow_hybrid_mode=False,
            moe_deepepv2_num_allocated_qps=2,
            moe_deepepv2_deterministic=False,
        )
        buffer = self.dispatch(manager, 6)

        assert calls_of("hint")[0]["allow_hybrid_mode"] is False
        assert buffer.init_kwargs["allow_hybrid_mode"] is False
        assert buffer.init_kwargs["num_allocated_qps"] == 2
        assert buffer.init_kwargs["deterministic"] is False

    def test_logs_the_resolved_sm_count_once_per_build(self, monkeypatch, caplog):
        manager = self.make_manager(monkeypatch)
        with caplog.at_level(logging.INFO, logger=fused_a2a.__name__):
            self.dispatch(manager, 6)
            self.dispatch(manager, 6)
            self.dispatch(manager, 8)  # the bound grows, so the buffer is rebuilt

        messages = [r.getMessage() for r in caplog.records if "SMs" in r.getMessage()]
        assert (
            messages
            == [f"DeepEP v2 dispatch uses {AUTO_NUM_SMS} SMs (requested 0; 0 is automatic)."] * 2
        )
