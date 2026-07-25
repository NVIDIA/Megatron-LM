# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for ``inject_flextron_forward_logic``.

These tests pin two invariants the dev-branch divergence hunt uncovered:

1. When a Flextron manager is attached and a budget is passed in kwargs,
   ``update_hook_elasticity_params`` must be called *before* the original
   forward runs — otherwise all hooks fire with ``current_router_emb=None``
   and silently no-op.
2. When no Flextron manager is present, ``flextron_kwargs`` is cleared
   before the original forward so no unexpected keyword args leak through.
"""

from types import SimpleNamespace

import pytest

from megatron.elastification.flextron_utils import inject_flextron_forward_logic


class _CallLog:
    """Record the order in which stub methods are invoked."""

    def __init__(self):
        self.events = []

    def note(self, name, **payload):
        self.events.append((name, payload))


def _make_original_forward(log):
    def _fwd(
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        decoder_input=None,
        labels=None,
        inference_context=None,
        runtime_gather_output=None,
        inference_params=None,
    ):
        log.note(
            "original_forward",
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return "forward-result"

    return _fwd


def _make_manager(log, *, router_present=True, router_output_kwargs=None, loss_func=None):
    router_output_kwargs = (
        router_output_kwargs
        if router_output_kwargs is not None
        else {"router_emb": (object(), object())}
    )

    def process_router_output(budget_item):
        log.note("process_router_output", budget_item=budget_item)
        return router_output_kwargs, loss_func

    def update_hook_elasticity_params(flextron_kwargs):
        log.note("update_hook_elasticity_params", flextron_kwargs=flextron_kwargs)

    return SimpleNamespace(
        router=object() if router_present else None,
        process_router_output=process_router_output,
        update_hook_elasticity_params=update_hook_elasticity_params,
    )


def _attach_forward(model_cls, original):
    """Return a model whose .forward is `original` and install the wrapper.

    `original` is stored as an instance attribute so Python does not auto-bind
    it as a method — we want the stub called as a plain function by
    ``flextron_forward``.
    """
    model = model_cls()
    model.forward = original
    inject_flextron_forward_logic(model)
    return model


class _StubModelNoManager:
    """Minimal model with no ._flextron_manager attribute at all."""

    def __init__(self):
        self.config = SimpleNamespace()


class _StubModelWithManager:
    def __init__(self):
        self.config = SimpleNamespace(flextron=True, is_flex_eval=False)
        self._flextron_manager = None  # filled in by test


class TestForwardReplacement:
    def test_forward_is_replaced(self):
        log = _CallLog()
        original = _make_original_forward(log)
        model = _StubModelNoManager()
        model.forward = original
        before = model.forward
        inject_flextron_forward_logic(model)
        # model.forward is now a bound method wrapper, not the raw stub.
        assert model.forward is not before


class TestNoManager:
    def test_without_manager_original_is_called_directly(self):
        log = _CallLog()
        original = _make_original_forward(log)
        model = _attach_forward(_StubModelNoManager, original)

        result = model.forward(
            input_ids="ids",
            position_ids="pos",
            attention_mask="mask",
            budget=0.697,  # should be swallowed, not leaked through
        )

        assert result == "forward-result"
        # Only original_forward was called; no router / hook-update step.
        names = [e[0] for e in log.events]
        assert names == ["original_forward"]
        # budget kwarg was cleared before reaching original_forward.
        assert "budget" not in log.events[0][1]

    def test_manager_with_no_router_skips_router_logic(self):
        log = _CallLog()
        original = _make_original_forward(log)
        model = _attach_forward(_StubModelWithManager, original)
        model._flextron_manager = _make_manager(log, router_present=False)

        model.forward(input_ids="ids", position_ids="pos", attention_mask="mask", budget=0.5)

        names = [e[0] for e in log.events]
        assert names == ["original_forward"]


class TestManagerOrdering:
    def test_budget_kwarg_triggers_router_and_hooks_before_forward(self):
        """Core invariant: update_hook_elasticity_params runs *before* original_forward."""
        log = _CallLog()
        original = _make_original_forward(log)
        model = _attach_forward(_StubModelWithManager, original)
        model._flextron_manager = _make_manager(log)

        model.forward(input_ids="ids", position_ids="pos", attention_mask="mask", budget=0.697)

        names = [e[0] for e in log.events]
        # Exact expected sequence.
        assert names == [
            "process_router_output",
            "update_hook_elasticity_params",
            "original_forward",
        ]
        # Sanity: the budget actually used is the one passed in kwargs.
        assert log.events[0][1]["budget_item"] == 0.697

    def test_loss_func_invoked_when_returned(self):
        log = _CallLog()
        original = _make_original_forward(log)
        model = _attach_forward(_StubModelWithManager, original)

        def _loss_func(kwargs, budget_item):
            log.note("loss_func", budget_item=budget_item)
            return "budget-loss"

        model._flextron_manager = _make_manager(log, loss_func=_loss_func)

        model.forward(input_ids="ids", position_ids="pos", attention_mask="mask", budget=0.697)

        names = [e[0] for e in log.events]
        # loss_func must run after router output and before the hook update.
        assert names == [
            "process_router_output",
            "loss_func",
            "update_hook_elasticity_params",
            "original_forward",
        ]


class TestOverrideSelectedBudget:
    def test_override_non_one_sets_budget_from_override(self):
        log = _CallLog()
        original = _make_original_forward(log)
        model = _attach_forward(_StubModelWithManager, original)
        model._flextron_manager = _make_manager(log)
        model.config.is_flex_eval = True
        model.config.override_selected_budget = [0.577]

        # No budget kwarg on the caller side — override should supply it.
        model.forward(input_ids="ids", position_ids="pos", attention_mask="mask")

        names = [e[0] for e in log.events]
        assert names == [
            "process_router_output",
            "update_hook_elasticity_params",
            "original_forward",
        ]
        assert log.events[0][1]["budget_item"] == 0.577

    def test_override_without_flex_eval_raises(self):
        log = _CallLog()
        original = _make_original_forward(log)
        model = _attach_forward(_StubModelWithManager, original)
        model._flextron_manager = _make_manager(log)
        model.config.is_flex_eval = False
        model.config.override_selected_budget = [0.577]

        with pytest.raises(AssertionError):
            model.forward(input_ids="ids", position_ids="pos", attention_mask="mask")
