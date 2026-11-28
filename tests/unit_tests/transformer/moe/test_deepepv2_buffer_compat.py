# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for the DeepEP v2 buffer handling in fused_a2a.

DeepEP 2.5 renamed ElasticBuffer to EPBuffer, keeps ``topk_idx`` by reference in the dispatch
handle, and rounds explicit SM counts up to even. The fake buffers below mimic DeepEP 2.1 and 2.5,
so these tests need neither a GPU nor DeepEP. They patch module attributes, ``sys.modules``
entries and ``torch.Tensor.clone`` through ``monkeypatch`` only and re-import nothing, so nothing
is left behind except the buffer classes cached by ``_dispatch_copies_topk_idx``.
"""

import importlib.util
import inspect
import sys
import types
import warnings
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe import fused_a2a, token_dispatcher

pytestmark = pytest.mark.internal

# The SM count the fakes choose when asked for 0 (automatic).
AUTO_NUM_SMS = 24

# dispatch/combine parameter names in DeepEP 2.1 (deep_ep/buffers/elastic.py:855 and :1046 at
# 01dc3aaa) and 2.5 (deep_ep/buffers/ep.py:562 and :770 at f6aa38fe). The fakes reject any other
# keyword, so a keyword that one version lacks fails here.
_DISPATCH_PARAMETERS = frozenset(
    "x topk_idx topk_weights cumulative_local_expert_recv_stats num_experts num_max_tokens_per_rank "
    "expert_alignment num_sms num_qps previous_event async_with_compute_stream "
    "allocate_on_comm_stream handle do_handle_copy do_cpu_sync do_expand do_zero_padding "
    "use_tma_aligned_col_major_sf".split()
)
_COMBINE_PARAMETERS = frozenset(
    "x handle topk_weights bias num_sms num_qps previous_event async_with_compute_stream "
    "allocate_on_comm_stream".split()
)


class FakeEvent:
    def __init__(self, owner):
        self.owner = owner

    def current_stream_wait(self):
        self.owner.waited.append(self)


class FakeHandle:
    """The EPHandle fields Megatron reads, plus a version check on ``topk_idx`` like DeepEP 2.5."""

    def __init__(self, topk_idx, num_sms, num_tokens, check_version):
        self._topk_idx = topk_idx
        self._topk_idx_version = topk_idx._version if check_version else None
        self.num_sms = num_sms
        self.num_recv_tokens_per_expert_list = [num_tokens]

    @property
    def topk_idx(self):
        unchanged = self._topk_idx_version in (None, self._topk_idx._version)
        assert unchanged, "fake EPHandle: topk_idx changed while the handle was alive"
        return self._topk_idx


class _FakeBuffer:
    round_num_sms_to_even = False
    parameters = {}

    def _check_keywords(self, method, keywords):
        unknown = set(keywords) - self.parameters[method]
        assert not unknown, f"{type(self).__name__}.{method} has no parameter {sorted(unknown)}"

    def __init__(self, group=None, **kwargs):
        self.group = group
        self.init_kwargs = kwargs
        self.num_bytes = kwargs.get("num_bytes")
        self.num_max_tokens_per_rank = kwargs.get("num_max_tokens_per_rank")
        self.calls = []
        self.handle_copy_requests = []
        self.order = []
        self.captured = []
        self.previous_events = []
        self.waited = []

    @staticmethod
    def get_buffer_size_hint(group, num_max_tokens_per_rank, hidden, num_topk):
        return num_max_tokens_per_rank * hidden * num_topk

    def capture(self):
        event = FakeEvent(self)
        self.order.append("capture")
        self.captured.append(event)
        return event

    def _num_sms(self, num_sms, automatic):
        if num_sms == 0:
            return automatic()
        return num_sms + num_sms % 2 if self.round_num_sms_to_even else num_sms

    def _dispatch(self, x, topk_idx, topk_weights, num_sms, handle, keep_copy, kwargs):
        self._check_keywords("dispatch", kwargs)
        self.previous_events.append(kwargs.get("previous_event"))

        def automatic():
            # DeepEP derives the automatic count from num_experts, which a cached dispatch
            # (the backward of combine) is not given, so it cannot choose a count there.
            assert (
                kwargs.get("num_experts") is not None
            ), "fake: automatic SM count needs num_experts"
            return AUTO_NUM_SMS

        num_sms = self._num_sms(num_sms, automatic)
        self.calls.append(("dispatch", num_sms))
        if handle is not None:
            handle.topk_idx  # DeepEP reads the cached handle's topk_idx.
            return x.clone(), None, None, handle, FakeEvent(self)
        self.dispatched_topk_idx = topk_idx
        stored = topk_idx.clone() if keep_copy else topk_idx
        handle = FakeHandle(stored, num_sms, x.shape[0], check_version=not keep_copy)
        return x.clone(), topk_idx.clone(), topk_weights.clone(), handle, FakeEvent(self)

    def combine(self, x, handle, topk_weights=None, num_sms=0, **kwargs):
        self._check_keywords("combine", kwargs)
        self.previous_events.append(kwargs.get("previous_event"))
        num_sms = self._num_sms(num_sms, lambda: handle.num_sms)
        self.calls.append(("combine", num_sms))
        handle.topk_idx  # DeepEP passes handle.topk_idx to the combine kernel.
        weights = None if topk_weights is None else topk_weights.clone()
        return x.clone(), weights, FakeEvent(self)


class FakeElasticBuffer(_FakeBuffer):
    """DeepEP 2.1: the handle gets a clone of topk_idx; SM counts are used as given."""

    parameters = {
        "dispatch": _DISPATCH_PARAMETERS | {"previous_event_before_epilogue"},
        "combine": _COMBINE_PARAMETERS | {"previous_event_before_epilogue"},
    }

    def dispatch(
        self,
        x,
        topk_idx=None,
        topk_weights=None,
        num_sms=0,
        handle=None,
        do_handle_copy=True,
        **kwargs,
    ):
        self.handle_copy_requests.append(do_handle_copy)
        return self._dispatch(x, topk_idx, topk_weights, num_sms, handle, do_handle_copy, kwargs)


class FakeEPBuffer(_FakeBuffer):
    """DeepEP 2.5: the handle keeps topk_idx by reference; odd SM counts round up to even."""

    round_num_sms_to_even = True
    parameters = {
        "dispatch": _DISPATCH_PARAMETERS | {"defer_epilogue"},
        "combine": _COMBINE_PARAMETERS | {"defer_epilogue"},
    }

    def dispatch(
        self,
        x,
        topk_idx=None,
        topk_weights=None,
        num_sms=0,
        handle=None,
        do_handle_copy=False,
        **kwargs,
    ):
        self.handle_copy_requests.append(do_handle_copy)
        assert not do_handle_copy, "fake EPBuffer rejects do_handle_copy=True"
        return self._dispatch(x, topk_idx, topk_weights, num_sms, handle, False, kwargs)


def _fake_deep_ep(**attributes):
    module = types.ModuleType("deep_ep")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def _round_trip(buffer, token_indices, num_sms, after_dispatch=None, async_comm=False):
    """Dispatch, combine and backpropagate through both, as _DeepepV2Manager does."""
    x = torch.randn(token_indices.shape[0], 8, requires_grad=True)
    token_probs = torch.rand(token_indices.shape, requires_grad=True)
    recv_x, _, recv_probs, _, handle = fused_a2a.DeepepV2Dispatch.apply(
        buffer,
        x,
        token_indices,
        token_probs,
        4,
        token_indices.shape[0],
        1,
        num_sms,
        async_comm,
        async_comm,
    )
    if after_dispatch is not None:
        after_dispatch()
    combined_x, _ = fused_a2a.DeepepV2Combine.apply(
        buffer, recv_x, handle, num_sms, async_comm, async_comm
    )
    (combined_x.sum() + recv_probs.sum()).backward()
    assert x.grad is not None and token_probs.grad is not None
    return handle


def _token_indices():
    return torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])


class TestDeepEPV2BufferImport:
    def test_prefers_epbuffer(self, monkeypatch):
        fake = _fake_deep_ep(EPBuffer=FakeEPBuffer, ElasticBuffer=FakeElasticBuffer)
        monkeypatch.setitem(sys.modules, "deep_ep", fake)
        assert fused_a2a._import_deep_ep_v2_buffer() is FakeEPBuffer

    def test_falls_back_to_elasticbuffer(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "deep_ep", _fake_deep_ep(ElasticBuffer=FakeElasticBuffer))
        assert fused_a2a._import_deep_ep_v2_buffer() is FakeElasticBuffer

    @pytest.mark.parametrize(
        "deep_ep",
        [_fake_deep_ep(Buffer=object, HybridEPBuffer=object), None],
        ids=["no-v2", "absent"],
    )
    def test_no_v2_buffer(self, monkeypatch, deep_ep):
        # None in sys.modules makes `from deep_ep import ...` raise ImportError.
        monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)
        assert fused_a2a._import_deep_ep_v2_buffer() is None

    def test_warns_when_deep_ep_has_no_buffer_class(self, monkeypatch):
        fake = _fake_deep_ep(HybridEPBuffer=object, __version__="9.9.0")
        monkeypatch.setitem(sys.modules, "deep_ep", fake)
        with pytest.warns(
            UserWarning, match=r"deep_ep 9\.9\.0 .*none of Buffer, ElasticBuffer or EPBuffer"
        ):
            fused_a2a._warn_if_no_deep_ep_buffer(have_deep_ep=False, have_deep_ep_v2=False)

    @pytest.mark.parametrize(
        "deep_ep, have_deep_ep, have_deep_ep_v2",
        [(None, False, False), (_fake_deep_ep(), False, True), (_fake_deep_ep(), True, False)],
        ids=["absent", "v2-available", "v1-available"],
    )
    def test_no_warning_otherwise(self, monkeypatch, deep_ep, have_deep_ep, have_deep_ep_v2):
        monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fused_a2a._warn_if_no_deep_ep_buffer(have_deep_ep, have_deep_ep_v2)

    @pytest.mark.parametrize("buffer_cls", [FakeEPBuffer, FakeElasticBuffer])
    def test_get_elastic_buffer_builds_installed_class(self, monkeypatch, buffer_cls):
        group = object()
        monkeypatch.setattr(fused_a2a, "DeepEPV2Buffer", buffer_cls)
        # A cached buffer on the same group skips the CUDA communicator warm-up.
        stale = buffer_cls(group, num_bytes=0, num_max_tokens_per_rank=0)
        monkeypatch.setattr(fused_a2a, "_elastic_buffer", stale)

        buffer = fused_a2a.get_elastic_buffer(
            group, num_max_tokens_per_rank=8, hidden=16, num_topk=2
        )

        assert type(buffer) is buffer_cls and buffer is not stale
        assert buffer.group is group
        assert buffer.init_kwargs == dict(
            num_bytes=8 * 16 * 2,
            num_max_tokens_per_rank=8,
            hidden=16,
            num_topk=2,
            deterministic=True,
        )


class TestDeepEPV2ImportErrors:
    config = SimpleNamespace(
        moe_router_dtype="fp32",
        moe_expert_capacity_factor=None,
        moe_permute_fusion=False,
        moe_flex_dispatcher_num_sms=None,
    )

    def test_deepepv2_names_both_buffer_classes(self, monkeypatch):
        monkeypatch.setattr(token_dispatcher, "deepepv2_dispatch", None)
        with pytest.raises(ImportError) as excinfo:
            token_dispatcher._DeepepV2Manager(None, 1, 2, 4, self.config)
        message = str(excinfo.value)
        assert "deep_ep.EPBuffer (DeepEP 2.5)" in message
        assert "deep_ep.ElasticBuffer (DeepEP 2.1)" in message

    def test_deepep_points_deepep_2_5_users_to_deepepv2(self, monkeypatch):
        monkeypatch.setattr(token_dispatcher, "fused_dispatch", None)
        with pytest.raises(ImportError) as excinfo:
            token_dispatcher._DeepepManager(None, 1, 2, 4, self.config)
        assert "--moe-flex-dispatcher-backend deepepv2" in str(excinfo.value)


class TestDeepEPV2Handle:
    def test_detects_which_dispatch_copies_topk_idx(self):
        class NoCopyParameter:
            def dispatch(self, x, topk_idx=None):
                pass

        assert fused_a2a._dispatch_copies_topk_idx(FakeElasticBuffer)
        assert not fused_a2a._dispatch_copies_topk_idx(FakeEPBuffer)
        assert not fused_a2a._dispatch_copies_topk_idx(NoCopyParameter)

    def test_elasticbuffer_gets_the_callers_tensor(self):
        buffer = FakeElasticBuffer()
        token_indices = _token_indices()
        _round_trip(buffer, token_indices, num_sms=0)
        # DeepEP 2.1 copies topk_idx itself, so Megatron adds no copy.
        assert buffer.dispatched_topk_idx is token_indices
        # Megatron leaves do_handle_copy at its default, so DeepEP 2.1 keeps copying.
        assert buffer.handle_copy_requests == [True, True]

    def test_epbuffer_handle_keeps_its_own_topk_idx(self):
        buffer = FakeEPBuffer()
        token_indices = _token_indices()
        original = token_indices.clone()
        # Change the caller's tensor in place while the handle is alive (through backward).
        handle = _round_trip(
            buffer, token_indices, num_sms=0, after_dispatch=lambda: token_indices.add_(1)
        )
        assert buffer.dispatched_topk_idx is not token_indices
        assert torch.equal(handle.topk_idx, original)
        assert buffer.handle_copy_requests == [False, False]

    def test_epbuffer_copy_is_queued_before_the_captured_event(self, monkeypatch):
        buffer = FakeEPBuffer()
        token_indices = _token_indices()
        clone = torch.Tensor.clone

        def recording_clone(tensor, *args, **kwargs):
            if tensor is token_indices:
                buffer.order.append("clone token_indices")
            return clone(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "clone", recording_clone)
        _round_trip(buffer, token_indices, num_sms=0, async_comm=True)
        # The communication stream waits only for work queued before the captured event.
        assert buffer.order[:2] == ["clone token_indices", "capture"]

    @pytest.mark.parametrize("buffer_cls", [FakeEPBuffer, FakeElasticBuffer])
    def test_async_calls_wait_on_captured_events(self, buffer_cls):
        buffer = buffer_cls()
        _round_trip(buffer, _token_indices(), num_sms=0, async_comm=True)
        # Each of the four calls gets its own captured event and its returned event is waited on.
        assert len(buffer.captured) == 4
        assert buffer.previous_events == buffer.captured
        assert len(buffer.waited) == 4

    @pytest.mark.parametrize(
        "buffer_cls, num_sms, expected",
        [
            (FakeElasticBuffer, 0, AUTO_NUM_SMS),
            (FakeElasticBuffer, 21, 21),
            (FakeEPBuffer, 0, AUTO_NUM_SMS),
            (FakeEPBuffer, 20, 20),
            (FakeEPBuffer, 21, 22),
        ],
    )
    def test_each_backward_uses_its_forward_sm_count(self, buffer_cls, num_sms, expected):
        buffer = buffer_cls()
        _round_trip(buffer, _token_indices(), num_sms=num_sms)
        # Forward dispatch and combine, then the backward of combine (a cached dispatch) and
        # the backward of dispatch (a combine).
        assert buffer.calls == [
            ("dispatch", expected),
            ("combine", expected),
            ("dispatch", expected),
            ("combine", expected),
        ]


def _load_private_fused_a2a(monkeypatch, deep_ep):
    """Execute a private copy of fused_a2a against `deep_ep`; the copy never enters sys.modules."""
    monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)
    spec = importlib.util.spec_from_file_location("_fused_a2a_private_copy", fused_a2a.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "_fused_a2a_private_copy" not in sys.modules
    return module


class TestFusedA2AModuleImport:
    def test_deep_ep_2_5_enables_deepepv2(self, monkeypatch):
        module = _load_private_fused_a2a(monkeypatch, _fake_deep_ep(EPBuffer=FakeEPBuffer))
        assert module.HAVE_DEEP_EP_V2 and module.DeepEPV2Buffer is FakeEPBuffer
        assert callable(module.deepepv2_dispatch) and callable(module.deepepv2_combine)

    def test_warns_at_import_without_buffer_classes(self, monkeypatch):
        with pytest.warns(UserWarning, match="none of Buffer, ElasticBuffer or EPBuffer"):
            module = _load_private_fused_a2a(monkeypatch, _fake_deep_ep(__version__="9.9.0"))
        assert not module.HAVE_DEEP_EP and not module.HAVE_DEEP_EP_V2


class TestRealDeepEP:
    """Check the installed deep_ep against what fused_a2a relies on, without a GPU or a buffer."""

    def test_installed_buffer_matches_what_fused_a2a_relies_on(self):
        deep_ep = pytest.importorskip("deep_ep")
        if hasattr(deep_ep, "EPBuffer"):
            buffer_cls, fake, copies = deep_ep.EPBuffer, FakeEPBuffer, False
        elif hasattr(deep_ep, "ElasticBuffer"):
            buffer_cls, fake, copies = deep_ep.ElasticBuffer, FakeElasticBuffer, True
        else:
            pytest.skip("the installed deep_ep has neither EPBuffer nor ElasticBuffer")
        assert fused_a2a.DeepEPV2Buffer is buffer_cls
        # The fakes reject keywords outside these sets, so the real class must accept them all.
        init_keywords = set(
            "num_bytes num_max_tokens_per_rank hidden num_topk deterministic".split()
        )
        size_hint_keywords = {"num_max_tokens_per_rank", "hidden", "num_topk"}
        checked = [
            *fake.parameters.items(),
            ("__init__", init_keywords),
            ("get_buffer_size_hint", size_hint_keywords),
        ]
        for method, keywords in checked:
            missing = keywords - inspect.signature(getattr(buffer_cls, method)).parameters.keys()
            assert not missing, f"{buffer_cls.__name__}.{method} has no parameter {sorted(missing)}"
        assert callable(getattr(buffer_cls, "capture", None))
        # DeepEP 2.1 copies topk_idx into the dispatch handle itself; for 2.5, Megatron copies it.
        assert fused_a2a._dispatch_copies_topk_idx(buffer_cls) is copies
