# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared runtime state for Megatron-FSDP modules."""

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from .indexed_order import IndexedOrder

if TYPE_CHECKING:
    from .module import FsdpModule


class FsdpContext:
    """Runtime stream and prefetch state shared by FSDP roots constructed together."""

    allgather_stream: torch.cuda.Stream
    reduce_scatter_stream: torch.cuda.Stream
    # HFSDP/HSDP need explicit last-microbatch state. First-microbatch state is
    # unnecessary because it can be detected when ``model_weight``, after syncing
    # from ``main_weight``, has placements different from ``Placements.optimizer``.
    is_last_microbatch: bool
    # Static orders used to drive all-gather prefetch. We may want to switch to
    # capturing runtime order if static module order proves too fragile. Each
    # FsdpModule tracks its own materialized state via ``FsdpModule._unshard_event``.
    forward_order: IndexedOrder["FsdpModule"]
    backward_order: IndexedOrder["FsdpModule"]

    def __init__(self, device: torch.device) -> None:
        """Create rank-local runtime state for FSDP modules on ``device``.

        Args:
            device: Device on which this context schedules communication.
        """
        self.is_last_microbatch = True
        self.forward_order = IndexedOrder()
        self.backward_order = IndexedOrder()
        # Construction-only; empty after finalization. Registration establishes
        # that each module carries the FsdpModule mixin without requiring a
        # runtime import from this module back to ``module.py``.
        self._registered_modules: list[nn.Module] = []
        self._is_finalized = False
        with torch.cuda.device(device):
            self.allgather_stream = torch.cuda.Stream()
            self.reduce_scatter_stream = torch.cuda.Stream()

    def register_module(self, module: nn.Module) -> None:
        """Register a module constructed in this context."""
        if self._is_finalized:
            raise RuntimeError("Cannot register an FSDP module after its context is finalized.")
        self._registered_modules.append(module)

    def finalize(self) -> None:
        """Finalize roots, names, and cross-root prefetch orders."""
        if self._is_finalized:
            raise RuntimeError("FSDP context is already finalized.")

        registered_modules = set(self._registered_modules)
        children: set[nn.Module] = set()
        for module in self._registered_modules:
            _collect_fsdp_children(module, registered_modules, children)
        # FsdpModules that are not descendants of any other FsdpModule.
        roots = [module for module in self._registered_modules if module not in children]

        for root in roots:
            root_fsdp_module = cast("FsdpModule", root)
            root_fsdp_module._is_root = True
            for name, module in root.named_modules():
                if module not in registered_modules:
                    continue
                fsdp_module = cast("FsdpModule", module)
                fsdp_module._name = name
                self.forward_order.append(fsdp_module)

        for root in reversed(roots):
            _collect_backward_order(root, registered_modules, self.backward_order)

        self._registered_modules.clear()
        self._is_finalized = True

    def ensure_finalized(self) -> None:
        """Raise if construction has not completed for this context."""
        if not self._is_finalized:
            raise RuntimeError(
                "FSDP context is not finalized. Exit fully_shard_context before running forward."
            )

    def current_stream(self) -> torch.cuda.Stream:
        """Current stream on this context's device."""
        return torch.cuda.current_stream(self.allgather_stream.device)

    def register_post_backward_final_callback(self) -> None:
        """Register this root context's final callback for the current backward.

        Root ``post_backward()`` means only that root-owned parameters have
        accumulated gradients; it may run before descendant reductions, or not
        run at all when the root owns no trainable parameters. Waiting at
        autograd completion orders consumers after every descendant reduction.
        """

        def post_backward_final_callback() -> None:
            self.current_stream().wait_stream(self.reduce_scatter_stream)

        torch.autograd.Variable._execution_engine.queue_callback(post_backward_final_callback)


def _collect_backward_order(
    module: nn.Module,
    registered_modules: set[nn.Module],
    order: IndexedOrder["FsdpModule"],
) -> None:
    """Collect one root's static backward prefetch order."""
    if module in registered_modules:
        order.append(cast("FsdpModule", module))

    for child in reversed(list(module.children())):
        _collect_backward_order(child, registered_modules, order)


def _collect_fsdp_children(
    module: nn.Module, registered_modules: set[nn.Module], children: set[nn.Module]
) -> None:
    """Collect the nearest registered MFSDP descendants of ``module``."""
    for child in module.children():
        if child in registered_modules:
            children.add(child)
        else:
            _collect_fsdp_children(child, registered_modules, children)
