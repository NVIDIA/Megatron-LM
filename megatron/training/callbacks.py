# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Training callbacks for Megatron.

This module provides a lightweight callback system for injecting custom logic
into the training loop without modifying framework code.

Two registration patterns are supported:

1. Class-based: Subclass `Callback` and override event methods
   ```python
   class MyCallback(Callback):
       def on_train_start(self, context):
           print("Training started!")

   pretrain(config, forward_step_func, callbacks=[MyCallback()])
   ```

2. Functional: Register functions directly with `CallbackManager`
   ```python
   manager = CallbackManager()
   manager.register("on_train_step_end", my_logging_fn)
   pretrain(config, forward_step_func, callbacks=manager)
   ```

Both patterns can be mixed. Callbacks fire in registration order.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch
    from megatron.core.optimizer import MegatronOptimizer
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.transformer import MegatronModule


logger: logging.Logger = logging.getLogger(__name__)


VALID_EVENTS: frozenset[str] = frozenset(
    {
        "on_setup_start",
        "on_data_init_start",
        "on_train_start",
        "on_train_step_start",
        "on_train_step_end",
        "on_log",
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
)


@dataclass
class CallbackContext:
    """Context passed to callbacks.

    Contains framework objects and a persistent user_state dict.
    Modifying framework objects is at the user's own risk.

    Attributes:
        model: List of model chunks.
        user_state: Mutable dict for storing user data across callback invocations.
        optimizer: Optimizer instance. Available during training events only.
        scheduler: Learning rate scheduler. Available during training events only.
        loss_dict: Reduced losses from training step. Available in on_train_step_end.
        grad_norm: Gradient norm. Available in on_train_step_end if computed.
        skipped_iter: Whether the iteration was skipped. Available in on_train_step_end.
        total_loss_dict: Aggregated eval losses. Available in on_eval_end.
        timers_to_log: Mutable timer-name list; append to include extra timers. In on_log.
        log_fragments: Mutable list; append strings to extend the stdout log line. In on_log.

    Field Availability by Event:
        All events: model (None in on_setup_start), user_state
        Training events: optimizer, scheduler
        on_data_init_start: optimizer, scheduler
        on_train_step_end: loss_dict, grad_norm, skipped_iter
        on_eval_end, on_test_end: total_loss_dict
        on_log: timers_to_log, log_fragments
    """

    # Always available (model is None in on_setup_start, before it is built)
    model: list[MegatronModule] | None
    user_state: dict = field(default_factory=dict)

    # Training events only
    optimizer: MegatronOptimizer | None = None
    scheduler: OptimizerParamScheduler | None = None

    # on_train_step_end
    loss_dict: dict[str, torch.Tensor] | None = None
    grad_norm: float | None = None
    skipped_iter: bool | None = None

    # on_eval_end
    total_loss_dict: dict[str, torch.Tensor] | None = None

    # on_log
    timers_to_log: list[str] | None = None
    log_fragments: list[str] | None = None


class Callback:
    """Base class for organizing callbacks.

    Subclass and override methods for events you want to handle.
    All methods are no-ops by default.

    Example:
        ```python
        class MyCallback(Callback):
            def on_train_start(self, context):
                context.user_state['start_time'] = time.time()

            def on_train_end(self, context):
                elapsed = time.time() - context.user_state['start_time']
                print(f"Training took {elapsed:.2f}s")

        pretrain(config, forward_step_func, callbacks=[MyCallback()])
        ```
    """

    def on_setup_start(self, context: CallbackContext) -> None:
        """Called before the model/optimizer are built (context.model is None)."""
        pass

    def on_data_init_start(self, context: CallbackContext) -> None:
        """Called after model/optimizer/checkpoint are ready, before dataset files are opened.

        This is the correct place to run JIT warmup with mock data and to log
        MLPerf init_stop/run_start markers, ensuring no real dataset I/O occurs
        before run_start is recorded.
        """
        pass

    def on_train_start(self, context: CallbackContext) -> None:
        """Called after model.train(), before training loop begins."""
        pass

    def on_train_step_start(self, context: CallbackContext) -> None:
        """Called at the top of each outer iteration, before the step."""
        pass

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Called once per outer iteration after the step (empty loss_dict when skip_train)."""
        pass

    def on_log(self, context: CallbackContext) -> None:
        """Called during periodic logging (respects --log-interval)."""
        pass

    def on_train_end(self, context: CallbackContext) -> None:
        """Called after training loop completes."""
        pass

    def on_eval_start(self, context: CallbackContext) -> None:
        """Called before the evaluation loop begins."""
        pass

    def on_eval_step_start(self, context: CallbackContext) -> None:
        """Called at the start of each evaluation step."""
        pass

    def on_eval_step_end(self, context: CallbackContext) -> None:
        """Called after each evaluation step completes."""
        pass

    def on_eval_end(self, context: CallbackContext) -> None:
        """Called after evaluation completes."""
        pass

    def on_test_start(self, context: CallbackContext) -> None:
        """Called before the test loop begins."""
        pass

    def on_test_step_start(self, context: CallbackContext) -> None:
        """Called at the start of each test step."""
        pass

    def on_test_step_end(self, context: CallbackContext) -> None:
        """Called after each test step completes."""
        pass

    def on_test_end(self, context: CallbackContext) -> None:
        """Called after the test loop completes."""
        pass


class CallbackManager:
    """Manages registration and execution of training callbacks.

    Supports two registration patterns:

    1. Class-based: Use add() with Callback subclass instances
       ```python
       manager.add(MyCallback())
       manager.add([CallbackA(), CallbackB()])
       ```

    2. Functional: Use register() with event name and callable
       ```python
       manager.register("on_train_start", my_function)
       ```

    Both patterns can be mixed. Callbacks fire in registration order.

    The manager also owns a `user_state` dictionary that persists across all
    callback invocations, allowing callbacks to share state.

    Example:
        ```python
        manager = CallbackManager()
        manager.add(MyCallback())
        manager.register("on_eval_end", lambda ctx: print("Eval done!"))
        pretrain(config, forward_step_func, callbacks=manager)
        ```
    """

    def __init__(self) -> None:
        """Initialize the callback manager with empty callback lists and user state."""
        self._callbacks: dict[str, list[Callable[[CallbackContext], None]]] = {event: [] for event in VALID_EVENTS}
        self._active_events: set[str] = set()
        self._user_state: dict = {}

    @property
    def user_state(self) -> dict:
        """Mutable dictionary for storing user data across callback invocations."""
        return self._user_state

    def add(self, callback: Callback | list[Callback]) -> None:
        """Register one or more Callback instances.

        Scans for methods that override the Callback base class and
        registers them to their corresponding events.

        Args:
            callback: Single Callback instance or list of Callback instances.

        Example:
            ```python
            manager.add(MyCallback())
            manager.add([TimingCallback(), LoggingCallback()])
            ```
        """
        callbacks = callback if isinstance(callback, list) else [callback]

        for cb in callbacks:
            for event_name in VALID_EVENTS:
                method = getattr(cb, event_name, None)
                base_method = getattr(Callback, event_name, None)
                if method is not None and method.__func__ is not base_method:
                    self._callbacks[event_name].append(method)
                    self._active_events.add(event_name)

    def register(self, event_name: str, fn: Callable[[CallbackContext], None]) -> None:
        """Register a callback function for a specific event.

        Args:
            event_name: Event to register for. Valid events:
                - "on_setup_start"
                - "on_data_init_start"
                - "on_train_start"
                - "on_train_step_start"
                - "on_train_step_end"
                - "on_log"
                - "on_train_end"
                - "on_eval_start"
                - "on_eval_step_start"
                - "on_eval_step_end"
                - "on_eval_end"
                - "on_test_start"
                - "on_test_step_start"
                - "on_test_step_end"
                - "on_test_end"
            fn: Callback function with signature (CallbackContext) -> None.

        Raises:
            ValueError: If event_name is not valid.

        Example:
            ```python
            manager.register("on_train_step_end", my_logging_fn)
            ```
        """
        if event_name not in VALID_EVENTS:
            raise ValueError(f"Unknown event '{event_name}'. Valid events: {VALID_EVENTS}")
        self._callbacks[event_name].append(fn)
        self._active_events.add(event_name)

    @property
    def events(self) -> frozenset[str]:
        """Set of valid event names for registration."""
        return VALID_EVENTS

    def list_callbacks(self, event_name: str) -> list[Callable[[CallbackContext], None]]:
        """Return list of callbacks registered for an event.

        Args:
            event_name: Name of the event.

        Returns:
            List of registered callables (in execution order).

        Raises:
            ValueError: If event_name is not valid.
        """
        if event_name not in VALID_EVENTS:
            raise ValueError(f"Unknown event '{event_name}'. Valid events: {VALID_EVENTS}")
        return list(self._callbacks[event_name])

    def has_callbacks(self, event_name: str) -> bool:
        """Check if any callbacks are registered for an event.

        Args:
            event_name: Name of the event.

        Returns:
            True if at least one callback is registered for the event.
        """
        return event_name in self._active_events

    def fire(self, event_name: str, context: CallbackContext) -> None:
        """Execute all callbacks for an event.

        Exceptions from callbacks propagate to the caller.

        Args:
            event_name: Name of the event to fire.
            context: CallbackContext to pass to callbacks.
        """
        for fn in self._callbacks[event_name]:
            fn(context)


def normalize_callbacks(
    callbacks: list[Callback] | CallbackManager | None,
) -> CallbackManager | None:
    """Normalize callbacks argument to a CallbackManager.

    This helper is used internally by pretrain() to accept multiple input formats.

    Args:
        callbacks: Either a list of Callback instances, a CallbackManager,
            or None.

    Returns:
        A CallbackManager instance, or None if callbacks was None.
    """
    if callbacks is None:
        return None
    if isinstance(callbacks, CallbackManager):
        return callbacks
    # It's a list of Callback instances
    manager = CallbackManager()
    manager.add(callbacks)
    return manager


def should_fire(callback_manager: CallbackManager | None, event_name: str) -> bool:
    """Check if callbacks should be fired for an event.

    Combines the None check and has_callbacks check into a single call.

    Args:
        callback_manager: The callback manager instance, or None.
        event_name: Name of the event to check.

    Returns:
        True if callback_manager exists and has callbacks for the event.
    """
    return callback_manager is not None and callback_manager.has_callbacks(event_name)
