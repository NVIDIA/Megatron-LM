"""Runtime ABC and registry.

Runtime API tiers
-----------------
L1 — **Pretrain Ready** (9 abstract methods, must implement):
    build_model, save_checkpoint, load_checkpoint,
    train_mode, eval_mode,
    forward_backward, zero_grad, optimizer_step, lr_scheduler_step

L2 — **RL Ready** (+ export_weights):
    Enables RL frameworks to extract weights for the inference engine.

L3 — **RL Best** (+ to):
    Enables offloading model/optimizer/grad between training and rollout
    phases to free GPU memory for the inference engine.

A new backend only needs to implement L1 to work for pretraining.
Override ``export_weights`` and/or ``to`` to unlock higher tiers.
Check ``runtime.tier`` to see which level a backend supports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch

    from megatron.lite.runtime.contracts.data import ForwardResult
    from megatron.lite.runtime.contracts.handle import ModelHandle


class Runtime(ABC):
    """Base class for all runtime implementations.

    MegatronLiteRuntime and custom impls subclass this.
    """

    # ── L1: Pretrain Ready (必须实现) ────────────────────────────

    # Model lifecycle

    @abstractmethod
    def build_model(self, hf_path: str | None = None, cfg: Any = None, **kwargs) -> ModelHandle:
        """Build model state for this runtime.

        Implementations should default to the ``hf_path`` / ``backend_cfg``
        captured by ``create_runtime(RuntimeConfig(...))``. Passing arguments to
        ``build_model`` is supported as an advanced override path, but public
        examples should prefer ``handle = rt.build_model()``.
        """
        ...

    @abstractmethod
    def save_checkpoint(self, handle: ModelHandle, path: str, **kwargs) -> None: ...

    @abstractmethod
    def load_checkpoint(self, handle: ModelHandle, path: str, **kwargs) -> int: ...

    # Mode switching

    @abstractmethod
    def train_mode(self, handle: ModelHandle) -> Any: ...

    @abstractmethod
    def eval_mode(self, handle: ModelHandle) -> Any: ...

    # Training atoms

    @abstractmethod
    def forward_backward(
        self,
        handle: ModelHandle,
        data: Any,
        loss_fn: Callable | None,
        *,
        num_microbatches: int = 1,
        forward_only: bool = False,
    ) -> ForwardResult:
        """Forward + backward pass over data.

        Args:
            num_microbatches: Number of microbatches to accumulate inside one
                logical training step.
            loss_fn: Optional external loss function with signature::

                loss_fn(model_output: dict, batch) -> (loss: Tensor, metrics: dict)

            When ``loss_fn`` is None, the model computes loss internally
            (standard pretrain/SFT path).  When provided, ``model_output``
            is the dict returned by the model's forward (logits, log_probs, etc.)
            and ``batch`` is the current microbatch.
        """
        ...

    @abstractmethod
    def zero_grad(self, handle: ModelHandle) -> None: ...

    @abstractmethod
    def optimizer_step(self, handle: ModelHandle) -> tuple[bool, float, int | None]:
        """Run optimizer step.

        Returns:
            (update_successful, grad_norm, num_zeros_in_grad)
        """
        ...

    @abstractmethod
    def lr_scheduler_step(self, handle: ModelHandle) -> float | list[float]: ...

    # ── Parallel state queries ───────────────────────────────────

    def is_mp_src_rank_with_outputs(self, handle: ModelHandle) -> bool:
        """True if this rank is PP-last, TP-0, CP-0 (has full output)."""
        return True  # default: no parallelism, every rank has outputs

    # ── L2: RL Ready (覆盖即解锁) ───────────────────────────────

    def export_weights(self, handle: ModelHandle, **kwargs) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over (name, tensor) pairs for HF-compatible weight export.

        Required by RL frameworks to send weights to the inference engine.
        Override to unlock **RL Ready** tier.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement export_weights. "
            "Implement it to unlock the RL Ready tier."
        )

    # ── L3: RL Best (覆盖即解锁) ────────────────────────────────

    def to(
        self,
        handle: ModelHandle,
        device: str,
        *,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
    ) -> None:
        """Move model / optimizer / gradients to *device*.

        Enables offloading between training and rollout phases so the
        inference engine can reclaim GPU memory.
        Override to unlock **RL Best** tier.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to(). "
            "Implement it to unlock the RL Best tier."
        )

    # ── Tier introspection ──────────────────────────────────────

    @property
    def tier(self) -> Literal["pretrain", "rl_ready", "rl_best"]:
        """Report the highest API tier this runtime supports."""
        cls = type(self)
        has_export = cls.export_weights is not Runtime.export_weights
        has_to = cls.to is not Runtime.to
        if has_export and has_to:
            return "rl_best"
        if has_export:
            return "rl_ready"
        return "pretrain"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RUNTIME_REGISTRY: dict[str, str] = {
    "bridge": "megatron.lite.runtime.backends.bridge",
    "mbridge": "megatron.lite.runtime.backends.mbridge",
    "mlite": "megatron.lite.runtime.backends.mlite",
}
