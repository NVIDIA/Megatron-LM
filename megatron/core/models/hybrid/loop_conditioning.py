# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""ELR loop-budget embeddings and layer-independent entry, exit, and branch modulation."""

import math
from collections.abc import Iterable, MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject
from megatron.core.models.hybrid.recurrence import validate_loop_count
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.utils import unwrap_model


@dataclass(frozen=True)
class LoopConditioningConfig:
    """Dimensions and the fixed reference budget for ELR conditioning."""

    conditioning_dim: int = 256
    """Width of the learned conditioning vector and each scalar's sinusoidal embedding."""

    rank: int = 256
    """Rank of each layer's gated low-rank input correction."""

    max_loops: int = 4
    """Reference budget K in k/K. Fixed across distribution changes; not a hard depth limit."""

    def __post_init__(self) -> None:
        for name in ("conditioning_dim", "rank", "max_loops"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"loop conditioning {name} must be a positive integer")
        if self.conditioning_dim % 2:
            raise ValueError("loop conditioning conditioning_dim must be even")


def _linear(
    config: TransformerConfig, in_features: int, out_features: int, *, zero=False, bias=True
) -> nn.Linear:
    device = (
        torch.device("cpu")
        if config.use_cpu_initialization
        else torch.device("cuda", torch.cuda.current_device())
    )
    # As with duplicated TELinear, use the CUDA stream shared across TP ranks.
    # On CPU, Megatron seeds the default generator identically across TP ranks.
    rng_context = nullcontext()
    if device.type == "cuda":
        tracker = get_cuda_rng_tracker()
        if tracker.is_initialized():
            rng_context = tracker.fork(get_data_parallel_rng_tracker_name())
    with rng_context, torch.no_grad():
        layer = nn.Linear(
            in_features, out_features, bias=bias, device=device, dtype=config.params_dtype
        )
        if zero:
            layer.weight.zero_()
        else:
            layer.weight.normal_(mean=0, std=config.init_method_std)
        if layer.bias is not None:
            layer.bias.zero_()
    return layer


def _project(layer: nn.Linear, value: Tensor) -> Tensor:
    return layer(value.to(layer.weight.dtype))


def _mark_sequence_parallel(module: nn.Module, config: TransformerConfig) -> None:
    for parameter in module.parameters():
        parameter.sequence_parallel = config.sequence_parallel


class _ConditioningModule(MegatronModule):
    """Replicated conditioning parameters with explicit gate-only checkpoint import."""

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Use freshly initialized conditioning tensors when importing an unconditioned model."""
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if (metadata or {}).get("initialize_loop_conditioning", False):
            return {key: LocalNonpersistentObject(shard.data) for key, shard in state.items()}
        return state


class LayerLoopConditioning(_ConditioningModule):
    """FiLM, gated low-rank correction, and residual-branch scaling (ELR Eq. 4)."""

    def __init__(
        self,
        config: TransformerConfig,
        conditioning_config: LoopConditioningConfig,
        pg_collection: ProcessGroupCollection,
        *,
        mixer: bool,
        mlp: bool,
    ) -> None:
        super().__init__(config)
        self.tp_group = pg_collection.tp
        self.mixer = BranchLoopConditioning(config, conditioning_config) if mixer else None
        self.mlp = BranchLoopConditioning(config, conditioning_config) if mlp else None
        _mark_sequence_parallel(self, config)


class BranchLoopConditioning(nn.Module):
    """Modulate one normalized branch input and its output, without changing the residual."""

    def __init__(
        self, config: TransformerConfig, conditioning_config: LoopConditioningConfig
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        dim, rank = conditioning_config.conditioning_dim, conditioning_config.rank
        self.input_scale = _linear(config, dim, hidden, zero=True)
        self.input_shift = _linear(config, dim, hidden, zero=True)
        self.low_rank_gate = _linear(config, dim, rank, zero=True)
        self.output_scale = _linear(config, dim, hidden, zero=True)
        self.down = _linear(config, hidden, rank, bias=False)
        self.up = _linear(config, rank, hidden, bias=False)

    def input(self, normalized: Tensor, condition: Tensor) -> Tensor:
        """Scale and shift the normalized input, then add a gated low-rank correction."""
        scale = _project(self.input_scale, condition).float()
        shift = _project(self.input_shift, condition).float()
        value = ((1 + scale) * normalized.float() + shift).to(normalized.dtype)
        correction = _project(
            self.up, _project(self.low_rank_gate, condition) * _project(self.down, value)
        )
        return (value.float() + correction.float()).to(normalized.dtype)

    def output(self, output_with_bias, condition: Tensor):
        """Scale the branch output and any separate bias before residual addition."""
        scale = 1 + _project(self.output_scale, condition).float()
        if isinstance(output_with_bias, Tensor):
            return (output_with_bias.float() * scale).to(output_with_bias.dtype)
        return tuple(
            None if value is None else (value.float() * scale).to(value.dtype)
            for value in output_with_bias
        )


class RegionLoopConditioning(nn.Module):
    """Later-pass RMSNorm/re-injection and an index-conditioned adaptive-delta exit."""

    def __init__(
        self, config: TransformerConfig, conditioning_config: LoopConditioningConfig
    ) -> None:
        super().__init__()
        dim, hidden = conditioning_config.conditioning_dim, config.hidden_size
        self.entry_scale = _linear(config, dim, hidden, zero=True)
        self.entry_shift = _linear(config, dim, hidden, zero=True)
        self.reinjection = _linear(config, dim, hidden, zero=True)
        self.exit_scale = _linear(config, dim, hidden, zero=True)
        self.eps = config.layernorm_epsilon

    def entry(
        self, combined: Tensor, region_input: Tensor, condition: Tensor, loop_index: int
    ) -> Tensor:
        """Keep the first pass unchanged; normalize and re-inject the input on later passes."""
        scale = _project(self.entry_scale, condition).float()
        shift = _project(self.entry_shift, condition).float()
        reinjection = _project(self.reinjection, condition).float()
        if loop_index == 0:
            # Retain zero-gradient edges for K=1 so DDP never waits for unused
            # entry-head parameters. No entry modulation is applied to pass one.
            return combined + ((scale + shift + reinjection).sum() * 0).to(combined.dtype)
        normalized = F.rms_norm(combined.float(), (combined.shape[-1],), eps=self.eps)
        return ((1 + scale) * normalized + shift + reinjection * region_input.float()).to(
            combined.dtype
        )

    def exit(self, condition: Tensor, loop_index: int) -> Tensor:
        """Return the learned update scale plus a unit offset on the first pass only."""
        return _project(self.exit_scale, condition).float() + (1 if loop_index == 0 else 0)


class LoopConditioner(_ConditioningModule):
    """Encode progress i/k and budget k/K, sharing the encoder across all regions/layers.

    Each scalar has a conventional sin/cos embedding with exponentially spaced
    frequencies from 1 to approximately 1/10,000 cycles per unit. ELR specifies
    sinusoidal embeddings but not their frequency convention. All modulation
    heads, unlike the encoder and low-rank bases, are initialized to zero.
    """

    def __init__(
        self,
        config: TransformerConfig,
        conditioning_config: LoopConditioningConfig,
        pg_collection: ProcessGroupCollection,
        num_regions: int,
    ) -> None:
        super().__init__(config)
        self.tp_group = pg_collection.tp
        self.conditioning_config = conditioning_config
        dim = conditioning_config.conditioning_dim
        self.encoder_in = _linear(config, 2 * dim, dim)
        self.encoder_out = _linear(config, dim, dim)
        device = self.encoder_in.weight.device
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(dim // 2, device=device).float() / (dim // 2)
        )
        self.register_buffer("frequencies", mark_keep_in_fp32(frequencies), persistent=False)
        self.register_buffer(
            "reference_max_loops", torch.tensor(conditioning_config.max_loops, device=device)
        )
        self.regions = nn.ModuleList(
            [RegionLoopConditioning(config, conditioning_config) for _ in range(num_regions)]
        )
        _mark_sequence_parallel(self, config)

    def forward(self, loop_index: int, n_loops: int) -> Tensor:
        """Encode progress and budget for the zero-based pass index."""
        validate_loop_count(n_loops)
        if (
            isinstance(loop_index, bool)
            or not isinstance(loop_index, int)
            or not 0 <= loop_index < n_loops
        ):
            raise ValueError("loop_index must identify a pass within n_loops")
        # Python scalars become kernel arguments, avoiding a per-pass host-to-device
        # copy of the progress/budget pair.
        phases = torch.stack(
            (
                self.frequencies * (2 * math.pi * (loop_index + 1) / n_loops),
                self.frequencies * (2 * math.pi * n_loops / self.conditioning_config.max_loops),
            )
        )
        embedding = torch.cat((phases.sin(), phases.cos()), dim=-1).flatten()
        return _project(self.encoder_out, F.silu(_project(self.encoder_in, embedding)))

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Buffer casts replace Tensor objects and lose custom keep-in-FP32
        # markers. Recreate this nonpersistent constant in FP32 on the new device.
        half = self.conditioning_config.conditioning_dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=self.encoder_in.weight.device).float()
            / half
        )
        self.frequencies = mark_keep_in_fp32(frequencies)
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "reference_max_loops"
        if key in state_dict and int(state_dict[key].item()) != self.conditioning_config.max_loops:
            raise ValueError("Checkpoint loop-conditioning reference budget differs from max_loops")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def validate_unconditioned_checkpoint_keys(keys: Iterable[str]) -> None:
    """Reject checkpoints with conditioning tensors before initializing new ones."""
    if any("loop_conditioning" in key.split(".") for key in keys):
        raise ValueError(
            "Checkpoint already contains loop conditioning; remove --init-loop-conditioning"
        )


def validate_loop_conditioning_checkpoint(model: nn.Module, keys: Iterable[str]) -> None:
    """Reject architecture choices that would discard trained conditioning parameters."""
    if any("loop_conditioning" in key.split(".") for key in keys):
        if not any(isinstance(module, LoopConditioner) for module in unwrap_model(model).modules()):
            raise ValueError(
                "Checkpoint contains loop conditioning; select a conditioned model spec"
            )


def initialize_loop_conditioning_state(model: nn.Module, state_dict: MutableMapping) -> None:
    """Import into a freshly constructed conditioned model without relaxing backbone checks."""
    model = unwrap_model(model)
    validate_unconditioned_checkpoint_keys(state_dict)
    found = False
    for name, module in model.named_modules():
        if isinstance(module, (LoopConditioner, LayerLoopConditioning)):
            found = True
            for key, value in module.state_dict().items():
                state_dict[f"{name}.{key}" if name else key] = value
        for old, new in getattr(module, "loop_conditioning_norm_key_map", {}).items():
            old_prefix = f"{name}.{old}" if name else old
            new_prefix = f"{name}.{new}" if name else new
            for key in list(state_dict):
                if key.startswith(old_prefix):
                    target = new_prefix + key[len(old_prefix) :]
                    if target in state_dict:
                        raise ValueError(f"Conflicting normalization keys: {key} and {target}")
                    state_dict[target] = state_dict.pop(key)
            # A separate TE norm has its own auxiliary state; the old fused
            # projection did not have this module. Never initialize its weights.
            norm = module.get_submodule(new.rstrip("."))
            extra_key = new_prefix + "_extra_state"
            if extra_key not in state_dict and "_extra_state" in norm.state_dict():
                state_dict[extra_key] = norm.state_dict()["_extra_state"]
    if not found:
        raise ValueError("--init-loop-conditioning requires a loop-conditioned model spec")
