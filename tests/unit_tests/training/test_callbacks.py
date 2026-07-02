# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Unit tests for the training callback system."""

from unittest.mock import Mock

import pytest

from megatron.training.callbacks import (
    VALID_EVENTS,
    Callback,
    CallbackContext,
    CallbackManager,
    normalize_callbacks,
    should_fire,
)


class TestCallbackContext:
    """Unit tests for CallbackContext."""

    def test_required_fields_are_accessible(self):
        """Required fields (model, user_state) are accessible."""
        mock_model = [Mock()]
        user_state = {"key": "value"}

        ctx = CallbackContext(model=mock_model, user_state=user_state)

        assert ctx.model is mock_model
        assert ctx.user_state == {"key": "value"}

    def test_optional_fields_default_to_none(self):
        """Optional fields default to None when not provided."""
        ctx = CallbackContext(model=[Mock()], user_state={})

        assert ctx.optimizer is None
        assert ctx.scheduler is None
        assert ctx.loss_dict is None
        assert ctx.grad_norm is None
        assert ctx.skipped_iter is None
        assert ctx.total_loss_dict is None

    def test_optional_fields_are_accessible_when_provided(self):
        """Optional fields are accessible when explicitly provided."""
        mock_optimizer = Mock()
        mock_scheduler = Mock()
        mock_loss_dict = {"loss": Mock()}

        ctx = CallbackContext(
            model=[Mock()],
            user_state={},
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            loss_dict=mock_loss_dict,
            grad_norm=1.5,
            skipped_iter=False,
        )

        assert ctx.optimizer is mock_optimizer
        assert ctx.scheduler is mock_scheduler
        assert ctx.loss_dict is mock_loss_dict
        assert ctx.grad_norm == 1.5
        assert ctx.skipped_iter is False

    def test_user_state_defaults_to_empty_dict(self):
        """user_state defaults to empty dict if not provided."""
        ctx = CallbackContext(model=[Mock()])
        assert ctx.user_state == {}

    def test_user_state_is_mutable(self):
        """user_state can be modified by callbacks."""
        ctx = CallbackContext(model=[Mock()], user_state={})

        ctx.user_state["new_key"] = "new_value"
        assert ctx.user_state["new_key"] == "new_value"


class TestCallback:
    """Unit tests for the Callback base class."""

    def test_all_methods_are_no_ops(self):
        """All base class methods are no-ops (don't raise)."""
        callback = Callback()
        mock_context = Mock(spec=CallbackContext)

        # None of these should raise
        callback.on_data_init_start(mock_context)
        callback.on_train_start(mock_context)
        callback.on_train_step_start(mock_context)
        callback.on_train_step_end(mock_context)
        callback.on_train_end(mock_context)
        callback.on_eval_start(mock_context)
        callback.on_eval_step_start(mock_context)
        callback.on_eval_step_end(mock_context)
        callback.on_eval_end(mock_context)
        callback.on_test_start(mock_context)
        callback.on_test_step_start(mock_context)
        callback.on_test_step_end(mock_context)
        callback.on_test_end(mock_context)

    def test_subclass_can_override_methods(self):
        """Subclasses can override specific methods."""
        call_log = []

        class MyCallback(Callback):
            def on_train_start(self, context):
                call_log.append("train_start")

            def on_eval_end(self, context):
                call_log.append("eval_end")

        callback = MyCallback()
        mock_context = Mock(spec=CallbackContext)

        callback.on_train_start(mock_context)
        callback.on_eval_end(mock_context)
        callback.on_train_end(mock_context)  # Not overridden, should be no-op

        assert call_log == ["train_start", "eval_end"]


class TestCallbackManagerRegistration:
    """Unit tests for CallbackManager registration methods."""

    def test_register_valid_event(self):
        """register() accepts valid event names."""
        manager = CallbackManager()
        fn = Mock()

        manager.register("on_train_start", fn)

        assert manager.has_callbacks("on_train_start")
        assert fn in manager.list_callbacks("on_train_start")

    def test_register_invalid_event_raises(self):
        """register() raises ValueError for unknown events."""
        manager = CallbackManager()

        with pytest.raises(ValueError, match="Unknown event"):
            manager.register("on_invalid_event", Mock())

    def test_register_multiple_callbacks_same_event(self):
        """Multiple callbacks can be registered for the same event."""
        manager = CallbackManager()
        fn1, fn2, fn3 = Mock(), Mock(), Mock()

        manager.register("on_train_start", fn1)
        manager.register("on_train_start", fn2)
        manager.register("on_train_start", fn3)

        callbacks = manager.list_callbacks("on_train_start")
        assert callbacks == [fn1, fn2, fn3]

    def test_add_single_callback(self):
        """add() registers overridden methods from a single Callback instance."""

        class MyCallback(Callback):
            def on_train_start(self, context):
                pass

            def on_eval_end(self, context):
                pass

        manager = CallbackManager()
        manager.add(MyCallback())

        assert manager.has_callbacks("on_train_start")
        assert manager.has_callbacks("on_eval_end")
        assert manager.has_callbacks("on_eval_end")

    def test_add_list_of_callbacks(self):
        """add() accepts a list of Callback instances."""

        class CallbackA(Callback):
            def on_train_start(self, context):
                pass

        class CallbackB(Callback):
            def on_eval_end(self, context):
                pass

        manager = CallbackManager()
        manager.add([CallbackA(), CallbackB()])

        assert manager.has_callbacks("on_train_start")
        assert manager.has_callbacks("on_eval_end")
        assert manager.has_callbacks("on_train_start")
        assert manager.has_callbacks("on_eval_end")

    def test_add_ignores_base_class_methods(self):
        """add() does not register methods that aren't overridden."""

        class EmptyCallback(Callback):
            pass  # No overrides

        manager = CallbackManager()
        manager.add(EmptyCallback())

        for event in VALID_EVENTS:
            assert not manager.has_callbacks(event)

    def test_add_only_registers_overridden_methods(self):
        """add() only registers methods that differ from base class."""

        class PartialCallback(Callback):
            def on_train_start(self, context):
                pass

            # All other methods are inherited (not overridden)

        manager = CallbackManager()
        manager.add(PartialCallback())

        assert manager.has_callbacks("on_train_start")
        assert not manager.has_callbacks("on_train_end")
        assert not manager.has_callbacks("on_eval_start")
        assert not manager.has_callbacks("on_eval_end")
        assert not manager.has_callbacks("on_test_start")
        assert not manager.has_callbacks("on_test_end")


class TestCallbackManagerFire:
    """Unit tests for CallbackManager.fire() execution."""

    def test_fire_calls_registered_callbacks(self):
        """fire() invokes all callbacks for an event."""
        manager = CallbackManager()
        fn1, fn2 = Mock(), Mock()
        manager.register("on_train_start", fn1)
        manager.register("on_train_start", fn2)

        context = Mock(spec=CallbackContext)
        manager.fire("on_train_start", context)

        fn1.assert_called_once_with(context)
        fn2.assert_called_once_with(context)

    def test_fire_respects_registration_order(self):
        """Callbacks fire in the order they were registered."""
        call_order = []
        manager = CallbackManager()

        manager.register("on_train_start", lambda ctx: call_order.append(1))
        manager.register("on_train_start", lambda ctx: call_order.append(2))
        manager.register("on_train_start", lambda ctx: call_order.append(3))

        manager.fire("on_train_start", Mock(spec=CallbackContext))

        assert call_order == [1, 2, 3]

    def test_fire_respects_mixed_registration_order(self):
        """Order is preserved when mixing add() and register()."""
        call_order = []

        class FirstCallback(Callback):
            def on_train_start(self, context):
                call_order.append("class_first")

        class LastCallback(Callback):
            def on_train_start(self, context):
                call_order.append("class_last")

        manager = CallbackManager()
        manager.add(FirstCallback())
        manager.register("on_train_start", lambda ctx: call_order.append("fn_middle"))
        manager.add(LastCallback())

        manager.fire("on_train_start", Mock(spec=CallbackContext))

        assert call_order == ["class_first", "fn_middle", "class_last"]

    def test_fire_does_nothing_when_no_callbacks(self):
        """fire() is a no-op when no callbacks are registered."""
        manager = CallbackManager()

        # Should not raise
        manager.fire("on_train_start", Mock(spec=CallbackContext))

    def test_fire_only_fires_requested_event(self):
        """fire() only invokes callbacks for the specified event."""
        manager = CallbackManager()
        train_fn = Mock()
        eval_fn = Mock()

        manager.register("on_train_start", train_fn)
        manager.register("on_eval_start", eval_fn)

        manager.fire("on_train_start", Mock(spec=CallbackContext))

        train_fn.assert_called_once()
        eval_fn.assert_not_called()

    def test_exception_propagates(self):
        """Exceptions from callbacks propagate to caller."""
        manager = CallbackManager()
        manager.register("on_train_start", lambda ctx: 1 / 0)

        with pytest.raises(ZeroDivisionError):
            manager.fire("on_train_start", Mock(spec=CallbackContext))

    def test_exception_stops_subsequent_callbacks(self):
        """When a callback raises, subsequent callbacks are not called."""
        manager = CallbackManager()
        first_fn = Mock()
        second_fn = Mock()

        manager.register("on_train_start", first_fn)
        manager.register("on_train_start", lambda ctx: 1 / 0)
        manager.register("on_train_start", second_fn)

        with pytest.raises(ZeroDivisionError):
            manager.fire("on_train_start", Mock(spec=CallbackContext))

        first_fn.assert_called_once()
        second_fn.assert_not_called()


class TestCallbackManagerIntrospection:
    """Unit tests for CallbackManager introspection methods."""

    def test_events_property_returns_valid_events(self):
        """events property returns the set of valid event names."""
        manager = CallbackManager()

        assert manager.events == VALID_EVENTS
        assert "on_train_start" in manager.events
        assert "on_eval_end" in manager.events

    def test_list_callbacks_returns_registered(self):
        """list_callbacks() returns all registered callbacks for an event."""
        manager = CallbackManager()
        fn1, fn2 = Mock(), Mock()

        manager.register("on_train_start", fn1)
        manager.register("on_train_start", fn2)

        callbacks = manager.list_callbacks("on_train_start")
        assert callbacks == [fn1, fn2]

    def test_list_callbacks_returns_copy(self):
        """list_callbacks() returns a copy, not the internal list."""
        manager = CallbackManager()
        fn = Mock()
        manager.register("on_train_start", fn)

        callbacks = manager.list_callbacks("on_train_start")
        callbacks.append(Mock())  # Modify the returned list

        # Internal list should be unchanged
        assert len(manager.list_callbacks("on_train_start")) == 1

    def test_list_callbacks_invalid_event_raises(self):
        """list_callbacks() raises ValueError for unknown events."""
        manager = CallbackManager()

        with pytest.raises(ValueError, match="Unknown event"):
            manager.list_callbacks("invalid_event")

    def test_has_callbacks_returns_false_when_empty(self):
        """has_callbacks() returns False when no callbacks registered."""
        manager = CallbackManager()

        assert not manager.has_callbacks("on_train_start")

    def test_has_callbacks_returns_true_when_registered(self):
        """has_callbacks() returns True when callbacks are registered."""
        manager = CallbackManager()
        manager.register("on_train_start", Mock())

        assert manager.has_callbacks("on_train_start")

    def test_has_callbacks_returns_false_for_invalid_event(self):
        """has_callbacks() returns False for unknown events (no error)."""
        manager = CallbackManager()

        # Returns False rather than raising - defensive behavior
        assert not manager.has_callbacks("invalid_event")


class TestUserStatePersistence:
    """Test that user_state persists across callback invocations."""

    def test_user_state_persists_across_fires(self):
        """Same user_state dict is available across multiple fire() calls."""
        manager = CallbackManager()

        def increment_counter(ctx):
            ctx.user_state["counter"] = ctx.user_state.get("counter", 0) + 1

        manager.register("on_train_step_end", increment_counter)

        # Simulate what framework does - same user_state dict each time
        persistent_state = {}
        for _ in range(5):
            ctx = CallbackContext(model=[Mock()], user_state=persistent_state)
            manager.fire("on_train_step_end", ctx)

        assert persistent_state["counter"] == 5

    def test_user_state_shared_across_different_events(self):
        """user_state is shared across different event types."""
        manager = CallbackManager()

        def write_start_time(ctx):
            ctx.user_state["start_time"] = 100

        def read_start_time(ctx):
            ctx.user_state["elapsed"] = 200 - ctx.user_state["start_time"]

        manager.register("on_train_start", write_start_time)
        manager.register("on_train_end", read_start_time)

        persistent_state = {}

        ctx1 = CallbackContext(model=[Mock()], user_state=persistent_state)
        manager.fire("on_train_start", ctx1)

        ctx2 = CallbackContext(model=[Mock()], user_state=persistent_state)
        manager.fire("on_train_end", ctx2)

        assert persistent_state["start_time"] == 100
        assert persistent_state["elapsed"] == 100


class TestNormalizeCallbacks:
    """Unit tests for the normalize_callbacks() helper function."""

    def test_none_returns_none(self):
        """None input returns None."""
        assert normalize_callbacks(None) is None

    def test_callback_manager_returns_same_instance(self):
        """CallbackManager input returns the same instance."""
        manager = CallbackManager()
        manager.register("on_train_start", Mock())

        result = normalize_callbacks(manager)

        assert result is manager

    def test_list_creates_new_manager(self):
        """List of Callbacks creates a new CallbackManager."""

        class MyCallback(Callback):
            def on_train_start(self, context):
                pass

        callbacks = [MyCallback()]
        result = normalize_callbacks(callbacks)

        assert isinstance(result, CallbackManager)
        assert result.has_callbacks("on_train_start")

    def test_list_with_multiple_callbacks(self):
        """List with multiple Callbacks registers all of them."""

        class CallbackA(Callback):
            def on_train_start(self, context):
                pass

        class CallbackB(Callback):
            def on_eval_end(self, context):
                pass

        result = normalize_callbacks([CallbackA(), CallbackB()])

        assert result.has_callbacks("on_train_start")
        assert result.has_callbacks("on_eval_end")

    def test_empty_list_returns_empty_manager(self):
        """Empty list returns a CallbackManager with no callbacks."""
        result = normalize_callbacks([])

        assert isinstance(result, CallbackManager)
        for event in VALID_EVENTS:
            assert not result.has_callbacks(event)


class TestValidEvents:
    """Tests for the VALID_EVENTS constant."""

    def test_valid_events_contains_expected_events(self):
        """VALID_EVENTS contains all expected event names."""
        expected = {
            "on_data_init_start",
            "on_train_start",
            "on_train_step_start",
            "on_train_step_end",
            "on_train_end",
            "on_eval_start",
            "on_eval_step_start",
            "on_eval_step_end",
            "on_eval_end",
            "on_test_start",
            "on_test_step_start",
            "on_test_step_end",
            "on_test_end",
        }
        assert VALID_EVENTS == expected

    def test_valid_events_is_frozenset(self):
        """VALID_EVENTS is immutable (frozenset)."""
        assert isinstance(VALID_EVENTS, frozenset)


class TestShouldFire:
    """Tests for the should_fire helper function."""

    def test_returns_false_when_manager_is_none(self):
        """Returns False when callback_manager is None."""
        assert should_fire(None, "on_train_start") is False

    def test_returns_false_when_no_callbacks_registered(self):
        """Returns False when manager has no callbacks for the event."""
        manager = CallbackManager()
        assert should_fire(manager, "on_train_start") is False

    def test_returns_true_when_callbacks_registered(self):
        """Returns True when manager has callbacks for the event."""
        manager = CallbackManager()
        manager.register("on_train_start", lambda ctx: None)
        assert should_fire(manager, "on_train_start") is True

    def test_returns_false_for_different_event(self):
        """Returns False when callbacks are registered for a different event."""
        manager = CallbackManager()
        manager.register("on_train_end", lambda ctx: None)
        assert should_fire(manager, "on_train_start") is False

    def test_works_with_class_based_callbacks(self):
        """Works correctly with class-based Callback instances."""

        class MyCallback(Callback):
            def on_eval_end(self, context):
                pass

        manager = CallbackManager()
        manager.add(MyCallback())

        assert should_fire(manager, "on_eval_end") is True
        assert should_fire(manager, "on_train_start") is False
