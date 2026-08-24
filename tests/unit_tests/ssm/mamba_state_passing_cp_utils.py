# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for Mamba state-passing context-parallel unit tests."""

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_block import MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.transformer import TransformerConfig

# Relative RMS error tolerance used for BF16 comparisons against a
# full-sequence reference.
DEFAULT_ERROR_RATIO = 3e-3

# Load-balancing modes exercised by the state-passing CP tests.
STATE_PASSING_CP_MODES = ("permute_p2p", "permute_a2a", "virtual")


def balanced_cp_chunk_ids(rank: int, cp_size: int) -> tuple[int, int]:
    """Return the front/back chunk indices Megatron's balanced CP assigns to ``rank``."""
    assert cp_size > 0 and 0 <= rank < cp_size
    return rank, 2 * cp_size - 1 - rank


def select_balanced_cp_shard(
    tensor: torch.Tensor, rank: int, cp_size: int, *, sequence_dim: int = 0
) -> torch.Tensor:
    """Select Megatron's front/back balanced CP shard without packing the batch axis."""
    assert tensor.shape[sequence_dim] % (2 * cp_size) == 0
    chunks = torch.chunk(tensor, 2 * cp_size, dim=sequence_dim)
    front, back = balanced_cp_chunk_ids(rank, cp_size)
    return torch.cat((chunks[front], chunks[back]), dim=sequence_dim).contiguous()


def select_contiguous_cp_shard(
    tensor: torch.Tensor, rank: int, cp_size: int, *, sequence_dim: int = 0
) -> torch.Tensor:
    """Select the two causally adjacent chunks owned by a contiguous CP rank."""
    assert tensor.shape[sequence_dim] % (2 * cp_size) == 0
    chunks = torch.chunk(tensor, 2 * cp_size, dim=sequence_dim)
    return torch.cat((chunks[2 * rank], chunks[2 * rank + 1]), dim=sequence_dim).contiguous()


def select_state_passing_cp_shard(
    tensor: torch.Tensor,
    rank: int,
    cp_size: int,
    *,
    virtual: bool,
    batch_dim: int = 0,
    sequence_dim: int = 1,
) -> torch.Tensor:
    """Select a contiguous shard, or pack balanced front/back chunks on the batch axis.

    The state-passing kernels consume either a contiguous causal shard or, in
    ``virtual`` mode, an interleaved view where each balanced front/back chunk
    becomes an independent virtual batch entry.
    """
    assert cp_size > 0 and 0 <= rank < cp_size
    assert tensor.shape[sequence_dim] % cp_size == 0
    if not virtual:
        local_length = tensor.shape[sequence_dim] // cp_size
        return tensor.narrow(sequence_dim, rank * local_length, local_length).contiguous()

    assert batch_dim != sequence_dim
    moved = tensor.movedim((batch_dim, sequence_dim), (0, 1))
    assert moved.shape[1] % (2 * cp_size) == 0
    chunks = torch.chunk(moved, 2 * cp_size, dim=1)
    front, back = balanced_cp_chunk_ids(rank, cp_size)
    packed = torch.stack((chunks[front], chunks[back]), dim=1).flatten(0, 1).contiguous()
    return packed.movedim((0, 1), (batch_dim, sequence_dim))


def relative_rms_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the RMS error of ``actual`` relative to the RMS magnitude of ``expected``.

    A relative criterion is used instead of elementwise ``allclose`` because the
    state-passing path reassociates the SSD scan across CP boundaries, which
    changes BF16 rounding without changing the mathematical result.
    """
    actual_float = actual.float()
    expected_float = expected.float()
    assert torch.isfinite(actual_float).all(), "state-passing result contains non-finite values"
    assert torch.isfinite(expected_float).all(), "reference contains non-finite values"
    difference = actual_float - expected_float
    if difference.abs().max().item() == 0.0:
        return 0.0
    expected_rms = expected_float.square().mean().sqrt().item()
    return difference.square().mean().sqrt().item() / (expected_rms + 1e-8)


def assert_all_close_rms(
    checks: dict[str, tuple[torch.Tensor, torch.Tensor]],
    group: dist.ProcessGroup,
    ratio: float = DEFAULT_ERROR_RATIO,
) -> None:
    """Compare every (actual, expected) pair and fail identically on all CP ranks.

    Every comparison is evaluated before anything is asserted, and the verdict is
    all-reduced. Asserting per comparison would let one rank leave the test while
    the others are still inside a collective, which deadlocks the whole run
    instead of reporting a failure.
    """
    errors = {name: relative_rms_error(*pair) for name, pair in checks.items()}
    names = list(errors)
    local = torch.tensor([errors[name] for name in names], device=torch.cuda.current_device())
    dist.all_reduce(local, op=dist.ReduceOp.MAX, group=group)
    failures = [
        f"{name}: relative RMS error {error:.3e} exceeds {ratio:.3e}"
        for name, error in zip(names, local.tolist())
        if not error < ratio
    ]
    assert not failures, "state-passing CP mismatch on at least one rank:\n" + "\n".join(failures)


def broadcast_module_parameters(module: torch.nn.Module, group: dist.ProcessGroup) -> None:
    """Make every CP rank hold the same weights.

    The CP path combines activations produced from every rank's parameters, so a
    per-rank difference in initialization would invalidate any comparison against
    a rank-local full-sequence reference.
    """
    source = dist.get_global_rank(group, 0)
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=source, group=group)


def collect_parameter_grads(
    module: torch.nn.Module, reduce_group: dist.ProcessGroup | None = None
) -> dict[str, torch.Tensor]:
    """Collect parameter grads as FP32, optionally summing replicated grads across CP."""
    gradients = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float().clone()
        if reduce_group is not None:
            dist.all_reduce(gradient, group=reduce_group)
        gradients[name] = gradient
    return gradients


@dataclass(frozen=True)
class MambaModelShape:
    """TP-local Mamba mixer shape used to build test models."""

    hidden_size: int = 2688
    nheads: int = 64
    head_dim: int = 64
    state_dim: int = 128
    ngroups: int = 8
    chunk_size: int = 128
    d_conv: int = 4

    @property
    def d_inner(self) -> int:
        """Inner (gated) width of the mixer."""
        return self.nheads * self.head_dim

    @property
    def conv_dim(self) -> int:
        """Number of channels the depthwise causal convolution operates on."""
        return self.d_inner + 2 * self.ngroups * self.state_dim

    @property
    def projected_width(self) -> int:
        """Width of the ``in_proj`` output (z, x, B, C, dt)."""
        return 2 * self.d_inner + 2 * self.ngroups * self.state_dim + self.nheads


def build_mamba_config(
    shape: MambaModelShape,
    cp_size: int,
    *,
    load_balancing: str,
    use_state_passing_cp: bool = True,
    cuda_graph_impl: str = "none",
    cuda_graph_warmup_steps: int = 2,
) -> TransformerConfig:
    """Build a single-layer BF16 Mamba config for state-passing CP tests."""
    return TransformerConfig(
        hidden_size=shape.hidden_size,
        num_layers=1,
        num_attention_heads=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        mamba_num_heads=shape.nheads,
        mamba_head_dim=shape.head_dim,
        mamba_state_dim=shape.state_dim,
        mamba_num_groups=shape.ngroups,
        use_cpu_initialization=True,
        use_mamba_mem_eff_path=True,
        use_mamba_state_passing_cp=use_state_passing_cp,
        mamba_state_passing_cp_load_balancing=load_balancing,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_scope=[],
        cuda_graph_warmup_steps=cuda_graph_warmup_steps,
        # Both CUDA Graph backends require the TE RNG tracker: the default
        # tracker refuses to be captured.
        use_te_rng_tracker=cuda_graph_impl != "none",
        hidden_dropout=0.0,
    )


def _mamba_layer_spec():
    assert isinstance(mamba_stack_spec.submodules, MambaStackSubmodules)
    spec = mamba_stack_spec.submodules.mamba_layer
    assert isinstance(spec.submodules, MambaLayerSubmodules)
    return spec


def build_mamba_mixer(
    shape: MambaModelShape, cp_group: dist.ProcessGroup, tp_group: dist.ProcessGroup, **kwargs
) -> MambaMixer:
    """Build a ``MambaMixer`` bound to ``cp_group`` and move it to the current device."""
    config = build_mamba_config(shape, cp_group.size(), **kwargs)
    mixer_spec = _mamba_layer_spec().submodules.mixer
    assert isinstance(mixer_spec.submodules, MambaMixerSubmodules)
    mixer = MambaMixer(
        config,
        mixer_spec.submodules,
        shape.hidden_size,
        d_conv=shape.d_conv,
        chunk_size=shape.chunk_size,
        layer_number=1,
        pg_collection=ProcessGroupCollection(tp=tp_group, cp=cp_group),
    ).to(torch.cuda.current_device())
    mixer.train()
    assert mixer.conv1d_weight.shape[0] == shape.conv_dim
    assert mixer.in_proj.weight.shape[0] == shape.projected_width
    return mixer


def build_mamba_layer(
    shape: MambaModelShape, cp_group: dist.ProcessGroup, tp_group: dist.ProcessGroup, **kwargs
) -> tuple[MambaLayer, TransformerConfig]:
    """Build a ``MambaLayer`` bound to ``cp_group`` and move it to the current device."""
    config = build_mamba_config(shape, cp_group.size(), **kwargs)
    spec = _mamba_layer_spec()
    mixer_spec = replace(
        spec.submodules.mixer,
        params={
            **spec.submodules.mixer.params,
            "d_conv": shape.d_conv,
            "chunk_size": shape.chunk_size,
        },
    )
    layer = MambaLayer(
        config,
        replace(spec.submodules, mixer=mixer_spec),
        layer_number=1,
        pg_collection=ProcessGroupCollection(tp=tp_group, cp=cp_group),
    ).to(torch.cuda.current_device())
    layer.train()
    return layer, config


def set_state_passing_cp_mode(mixer: MambaMixer, mode: str | None) -> None:
    """Switch a built mixer between the A2A CP path (``None``) and a state-passing mode."""
    if mode is None:
        mixer.use_mamba_state_passing_cp = False
        return
    assert mode in STATE_PASSING_CP_MODES
    mixer.use_mamba_state_passing_cp = True
    mixer.mamba_state_passing_cp_load_balancing = mode
    mixer.config.mamba_state_passing_cp_load_balancing = mode
