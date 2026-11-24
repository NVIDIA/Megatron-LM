# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adaptive-delta region updates from ELR."""

from collections.abc import Iterable, MutableMapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.utils import unwrap_model


class AdaptiveDeltaGate(MegatronModule):
    """Token- and channel-dependent update shared by all passes of one region.

    The stack supplies ``loop_input = previous_state + region_input`` to both the
    body and this gate. The ELR update is beta = sigmoid(W loop_input + b) *
    softplus(gamma), followed by state + beta * (candidate - state). W, b, and
    gamma start at zero, giving beta = log(2) / 2. This is not a bounded convex
    interpolation, an nGPT normalization, or loop-conditioned modulation.
    """

    def __init__(self, config: TransformerConfig, pg_collection: ProcessGroupCollection) -> None:
        super().__init__(config=config)
        self.tp_group = pg_collection.tp
        device = (
            torch.device("cpu")
            if config.use_cpu_initialization
            else torch.device("cuda", torch.cuda.current_device())
        )
        self.beta_proj = nn.Linear(
            config.hidden_size, config.hidden_size, device=device, dtype=config.params_dtype
        )
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        self.base_alpha_raw = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(config.hidden_size, device=device, dtype=torch.float32))
        )
        # Hidden channels are complete on every TP rank. Under SP, replicated
        # gate parameters see disjoint tokens and need the same TP gradient sum
        # used by other sequence-parallel replicated parameters.
        for parameter in self.parameters():
            parameter.sequence_parallel = config.sequence_parallel

    def forward(
        self,
        previous_state: Tensor,
        candidate: Tensor,
        loop_input: Tensor,
        *,
        update_scale: Tensor | float = 1.0,
    ) -> Tensor:
        """Apply the update with an optional explicit loop-dependent exit factor."""
        beta = torch.sigmoid(
            self.beta_proj(loop_input.to(self.beta_proj.weight.dtype)).float()
        ) * F.softplus(self.base_alpha_raw.float())
        return (
            previous_state.float()
            + update_scale * beta * (candidate.float() - previous_state.float())
        ).to(previous_state.dtype)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Save replicated parameters; explicitly initialize them on ungated imports."""
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if (metadata or {}).get("initialize_recurrence_gates", False):
            return {
                key: LocalNonpersistentObject(torch.zeros_like(shard.data))
                for key, shard in state.items()
            }
        return state


def validate_ungated_checkpoint_keys(keys: Iterable[str]) -> None:
    """Prevent an import request from discarding trained or partially saved gates."""
    if any("recurrence_gates" in key.split(".") for key in keys):
        raise ValueError(
            "The checkpoint already contains recurrence gates. Remove "
            "--init-recurrence-gates to resume it without reinitializing learned gates."
        )


def validate_recurrence_gate_checkpoint(
    model: nn.Module, checkpoint_keys: Iterable[str], source_pattern: str | None = None
) -> None:
    """Do not discard trained gates or silently retarget them to different regions."""
    if not any("recurrence_gates" in key.split(".") for key in checkpoint_keys):
        return
    model = unwrap_model(model)
    stacks = [module for module in model.modules() if getattr(module, "recurrence_gates", None)]
    if not stacks:
        raise ValueError("The checkpoint contains recurrence gates; select a gated model spec")
    if source_pattern is not None and len(stacks) == 1:
        from megatron.core.models.hybrid.recurrence import parse_recurrence_pattern

        _, source_ranges = parse_recurrence_pattern(source_pattern.split("/")[0])
        if source_ranges != stacks[0].recurrent_layer_ranges:
            raise ValueError(
                "Checkpoint recurrence boundaries differ; preserve the trained gate regions"
            )


def initialize_recurrence_gate_state(model: nn.Module, state_dict: MutableMapping) -> None:
    """Add only new gate keys to an ungated ordinary state dict, in place.

    The caller must explicitly opt into this conversion and start a new optimizer.
    Existing model entries and their metadata are retained. The subsequent strict
    load still detects missing, unexpected, or shape-incompatible backbone weights.
    """
    model = unwrap_model(model)
    validate_ungated_checkpoint_keys(state_dict)
    found = False
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveDeltaGate):
            found = True
            prefix = f"{name}." if name else ""
            for key, value in module.state_dict().items():
                state_dict[prefix + key] = torch.zeros_like(value)
    if not found:
        raise ValueError("--init-recurrence-gates requires an adaptive-delta gated model spec")
