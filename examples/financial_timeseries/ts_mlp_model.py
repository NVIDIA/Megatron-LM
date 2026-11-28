# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A minimal, runnable Megatron model for debugging the financial time-series training loop.

``TSMlpModel`` is the simplest thing that fits the data contract from ``preprocessing.py`` and
plugs into Megatron's ``pretrain()`` loop. It is intentionally NOT a sequence model: each of the
L context patches is embedded and mapped to its H-horizon forecasts independently. That makes it
trivially causal (a position never sees future patches) and ideal for shaking out tensor shapes,
the loss and the Megatron plumbing. Swap ``self.body`` for attention once the
loop is green.

What makes it "a Megatron model" (vs a plain nn.Module):
  * It subclasses ``MegatronModule`` and is built from a ``TransformerConfig``. Megatron's
    wrappers (Float16Module, DDP) and the checkpoint code read ``model.config`` and call
    ``model.state_dict_for_save_checkpoint()`` -- both come for free from ``MegatronModule``.
  * It implements ``set_input_tensor``. Megatron's pipeline schedule calls this on every model
    chunk before ``forward``. With pipeline-parallel size 1 the value is ``None`` and unused.

It uses plain ``nn.Linear`` / ``nn.Embedding`` (no tensor-parallel layers), so it does not need
``torch.distributed`` to be initialised to be *constructed* -- which is why the standalone
``debug_smoke_test.py`` can build and train it without any Megatron launch machinery.

forward inputs (supplied by pretrain_ts.forward_step):
    patches      (B, L, P, F) float
    inst_id      (B,)         int
    patch_tod    (B, L)       int
forward output (consumed by pretrain_ts.loss_func):
    {"return": (B, L, H), "volshare": (B, L, H)}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

# Recent Megatron-Core attaches a ProcessGroupCollection to every model (the training loop reads
# model.pg_collection). Older versions don't have this module; import defensively so the model
# still works on both.
try:
    from megatron.core.process_groups_config import ProcessGroupCollection
except ImportError:
    ProcessGroupCollection = None


class TSMlpModel(MegatronModule):
    """Minimal patch-level MLP that satisfies the financial time-series tensor contract.

    Each patch token is embedded independently, summed with a learned instrument
    embedding and a time-of-day embedding, passed through a per-token MLP and
    projected to one return and one volume-share forecast per horizon. There is no
    attention, so the model cannot use its context window; it exists as a fast
    end-to-end check of the data pipeline and marks where a real architecture goes.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        n_instruments: int,
        n_features: int,
        patch_len: int,
        context_len: int,
        n_horizons: int,
        n_tod: int,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection=None,
        vp_stage=None,
    ) -> None:
        """Build the embeddings, MLP body and forecast heads.

        Args:
            config: Megatron transformer config; only ``hidden_size`` and ``num_layers`` are used.
            n_instruments: Number of distinct instruments (size of the instrument embedding).
            n_features: Number of per-row input features in each patch.
            patch_len: Number of rows per patch.
            context_len: Number of patches per window (kept for reference; unused by the MLP).
            n_horizons: Number of forecast horizons, i.e. output channels per head.
            n_tod: Number of time-of-day buckets (size of the time-of-day embedding).
            pre_process: Whether this pipeline stage holds the input embeddings.
            post_process: Whether this pipeline stage holds the output heads.
            pg_collection: Optional ``ProcessGroupCollection``; built from the global
                parallel state when omitted.
            vp_stage: Virtual pipeline stage index, passed through by the framework.
        """
        super().__init__(config=config)
        # pre_process / post_process matter only for pipeline parallelism (which stage holds the
        # embedding vs the head). With PP=1 both are True and this single stage holds everything.
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.vp_stage = vp_stage

        # The training loop reads model.pg_collection. Use the one the framework passes; otherwise
        # build the default from the initialized global parallel state (mpu), exactly like mcore's
        # own TransformerBlock does. On older Megatron without this concept, leave it as-is.
        if (
            pg_collection is None
            and ProcessGroupCollection is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        hidden = config.hidden_size
        self.n_features = n_features
        self.patch_len = patch_len
        self.context_len = context_len
        self.n_horizons = n_horizons

        in_dim = patch_len * n_features  # flattened (P, F) per patch token
        self.patch_embed = nn.Linear(in_dim, hidden)
        self.inst_embed = nn.Embedding(n_instruments, hidden)  # learned per-instrument vector
        self.tod_embed = nn.Embedding(n_tod, hidden)  # time-of-day calendar embedding

        body = []
        for _ in range(max(1, config.num_layers)):  # depth driven by --num-layers
            body += [nn.Linear(hidden, hidden), nn.GELU()]
        self.body = nn.Sequential(*body)

        # two heads: point forecast of patch log-return and of volume-share, per horizon
        self.head_return = nn.Linear(hidden, n_horizons)
        self.head_volshare = nn.Linear(hidden, n_horizons)

    def set_input_tensor(self, input_tensor):
        """Required by Megatron's pipeline schedule. For PP=1 this is None and unused."""
        if isinstance(input_tensor, (list, tuple)):
            input_tensor = input_tensor[0]
        self.input_tensor = input_tensor

    def forward(
        self, patches: torch.Tensor, inst_id: torch.Tensor, patch_tod: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forecast returns and volume shares for every patch in the window.

        Args:
            patches: Input patches of shape ``(B, L, P, F)``.
            inst_id: Instrument ids of shape ``(B,)``.
            patch_tod: Time-of-day bucket per patch, shape ``(B, L)``.

        Returns:
            Dict with ``"return"`` and ``"volshare"`` tensors, each of shape ``(B, L, H)``.
        """
        B, L, P, F = patches.shape
        x = patches.reshape(B, L, P * F)  # (B, L, P*F)
        x = x.to(self.patch_embed.weight.dtype)  # match the (bf16/fp16) parameter dtype

        h = self.patch_embed(x)  # (B, L, hidden)
        h = h + self.inst_embed(inst_id).unsqueeze(1)  # broadcast instrument vec over L
        # guard against an out-of-range time-of-day index (e.g. if time_units_per_day was off)
        patch_tod = patch_tod.clamp_max(self.tod_embed.num_embeddings - 1)
        h = h + self.tod_embed(patch_tod)  # (B, L, hidden)
        h = self.body(h)  # (B, L, hidden)

        return {
            "return": self.head_return(h),  # (B, L, H)
            "volshare": self.head_volshare(h),  # (B, L, H)
        }
