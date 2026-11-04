# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dense-row references for the shared-prefix layer-level numerical tests.

A shared-prefix layout stores each prompt once and appends its completions:
``[prompt, completion_1, ..., completion_G]`` per root, roots concatenated, then optional
topology padding. A conventional (dense) batch runs every row ``[prompt, completion_g]`` on
its own. The tests build both token orders from one canonical problem, run the module under
test in each order, and compare the results per dense row:

* outputs are compared row by row;
* input gradients are compared after summing the dense prompt copies, which is the gradient
  the single shared prompt row must receive;
* parameter gradients are compared directly.

All differences are whole-tensor relative L2 norms.
"""

import os
from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor

from megatron.core.shared_prefix_layout import SharedPrefixForestLayout, SharedPrefixLayout


def round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to a multiple of ``multiple``."""
    return (value + multiple - 1) // multiple * multiple


def rel_l2(candidate: Tensor, reference: Tensor) -> float:
    """Relative L2 distance in float64."""
    reference = reference.double()
    denominator = reference.norm().item()
    numerator = (candidate.double() - reference).norm().item()
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def max_row_error(candidate: Tensor, reference: Tensor) -> float:
    """Largest per-token error, relative to the reference RMS token norm.

    Defects confined to a few positions (for example the last positions of one branch) are
    diluted in a whole-tensor norm; this metric keeps them visible.
    """
    candidate = candidate.double().reshape(candidate.shape[0], -1)
    reference = reference.double().reshape(reference.shape[0], -1)
    scale = reference.pow(2).sum(-1).mean().sqrt().item()
    if scale == 0.0:
        return 0.0
    return (candidate - reference).norm(dim=-1).max().item() / scale


def grads_rel_l2(candidate: dict[str, Tensor], reference: dict[str, Tensor]) -> float:
    """Relative L2 distance over the concatenation of all reference gradients."""
    numerator = denominator = 0.0
    for name, ref in reference.items():
        diff = candidate[name].double() - ref.double()
        numerator += diff.pow(2).sum().item()
        denominator += ref.double().pow(2).sum().item()
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return (numerator / denominator) ** 0.5


@dataclass(frozen=True)
class DenseRow:
    """One conventional row ``[prompt, completion]`` and where it lives in the star."""

    root: int
    prefix_len: int
    logical_len: int
    physical_len: int
    prompt_offset: int
    completion_offset: int

    @property
    def dense_len(self) -> int:
        """Return the row length, including its ordinary per-sequence padding."""
        return self.prefix_len + self.physical_len

    def star_indices(self) -> Tensor:
        """Physical star index of every token of this dense row."""
        return torch.cat(
            [
                torch.arange(self.prompt_offset, self.prompt_offset + self.prefix_len),
                torch.arange(self.completion_offset, self.completion_offset + self.physical_len),
            ]
        )


@dataclass(frozen=True)
class SharedPrefixProblem:
    """Prompt/completion geometry shared by the dense and the shared token orders.

    ``roots`` holds ``(prefix_len, logical_completion_lens)`` per root. With
    ``padding_multiple`` set, each dense row ``prompt + completion`` is padded to that multiple
    (ordinary per-sequence padding, owned by its branch). The star is then padded to
    ``topology_multiple`` plus ``extra_padding`` trailing topology-only tokens.
    """

    roots: tuple[tuple[int, tuple[int, ...]], ...]
    padding_multiple: int | None = None
    topology_multiple: int = 1
    extra_padding: int = 0

    @cached_property
    def rows(self) -> tuple[DenseRow, ...]:
        """Dense rows in root-major, completion-minor order."""
        rows = []
        offset = 0
        for root, (prefix_len, logical_lens) in enumerate(self.roots):
            prompt_offset = offset
            offset += prefix_len
            for logical_len in logical_lens:
                physical_len = logical_len
                if self.padding_multiple is not None:
                    physical_len = (
                        round_up(prefix_len + logical_len, self.padding_multiple) - prefix_len
                    )
                rows.append(
                    DenseRow(root, prefix_len, logical_len, physical_len, prompt_offset, offset)
                )
                offset += physical_len
        return tuple(rows)

    @property
    def total_len(self) -> int:
        """Return the unpadded star length."""
        return sum(prefix_len for prefix_len, _ in self.roots) + sum(
            row.physical_len for row in self.rows
        )

    @property
    def physical_len(self) -> int:
        """Return the star length including topology padding."""
        return round_up(self.total_len, self.topology_multiple) + self.extra_padding

    @property
    def max_dense_len(self) -> int:
        """Return the longest dense row."""
        return max(row.dense_len for row in self.rows)

    def layout(self, forest: bool) -> SharedPrefixLayout | SharedPrefixForestLayout:
        """Build the layout under test; ``forest=False`` requires exactly one root."""
        stars = []
        for root, (prefix_len, logical_lens) in enumerate(self.roots):
            physical = tuple(row.physical_len for row in self.rows if row.root == root)
            if self.padding_multiple is None:
                stars.append(SharedPrefixLayout(prefix_len, physical))
            else:
                stars.append(
                    SharedPrefixLayout(
                        prefix_len,
                        physical,
                        logical_completion_lens=tuple(logical_lens),
                        padding_multiple=self.padding_multiple,
                    )
                )
        if forest:
            return SharedPrefixForestLayout(tuple(stars))
        assert len(stars) == 1, "a plain star layout has exactly one root"
        return stars[0]

    def token_keys(self) -> tuple[Tensor, list[Tensor], int]:
        """Canonical token identities for MoE routing replay, for model-level callers.

        Returns ``(star_keys, row_keys, num_keys)``. A prompt token has one key shared by every
        dense copy; a completion token (including its ordinary padding) has a key of its own.
        Topology-only padding uses the last key.
        """
        keys = torch.full((self.physical_len,), -1, dtype=torch.long)
        next_key = 0
        for row in self.rows:
            for start, length in (
                (row.prompt_offset, row.prefix_len),
                (row.completion_offset, row.physical_len),
            ):
                if keys[start] < 0:
                    keys[start : start + length] = torch.arange(next_key, next_key + length)
                    next_key += length
        keys[keys < 0] = next_key
        row_keys = [keys.index_select(0, row.star_indices()) for row in self.rows]
        return keys, row_keys, next_key + 1


class LayerProblemData:
    """Random activations and cotangents for one layer, in both token orders.

    Hidden rows are drawn once per canonical token, so every dense copy of a prompt sees the same
    input. Each dense row draws its own output cotangent; the shared prompt row receives the sum
    of its copies, which makes the two scalar losses identical functions of the parameters.
    """

    def __init__(self, problem: SharedPrefixProblem, hidden_size: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.problem = problem
        physical_len = problem.physical_len
        canonical = torch.randn(physical_len, hidden_size, generator=generator, dtype=torch.float64)
        canonical[problem.total_len :] = 0
        self.star_input = canonical.unsqueeze(1)
        self.row_indices = [row.star_indices() for row in problem.rows]
        self.row_cotangents = [
            torch.randn(row.dense_len, hidden_size, generator=generator, dtype=torch.float64)
            for row in problem.rows
        ]
        star_cotangent = torch.zeros(physical_len, hidden_size, dtype=torch.float64)
        for indices, cotangent in zip(self.row_indices, self.row_cotangents):
            star_cotangent.index_add_(0, indices, cotangent)
        self.star_cotangent = star_cotangent.unsqueeze(1)

        rows, max_len = len(problem.rows), problem.max_dense_len
        self.dense_input = torch.zeros(max_len, rows, hidden_size, dtype=torch.float64)
        self.dense_cotangent = torch.zeros_like(self.dense_input)
        for column, (indices, cotangent) in enumerate(zip(self.row_indices, self.row_cotangents)):
            self.dense_input[: indices.numel(), column] = canonical.index_select(0, indices)
            self.dense_cotangent[: indices.numel(), column] = cotangent

    def dense_rows(self, dense: Tensor) -> list[Tensor]:
        """Split a ``[max_len, rows, hidden]`` dense tensor into its real rows."""
        return [dense[: indices.numel(), column] for column, indices in enumerate(self.row_indices)]

    def unpack_dense_rows(self, packed: Tensor) -> Tensor:
        """Inverse of concatenating the real dense rows: ``[sum(len), 1, h]`` -> dense batch.

        Kept here for model-level callers.
        """
        dense = packed.new_zeros(self.dense_input.shape)
        start = 0
        for column, indices in enumerate(self.row_indices):
            dense[: indices.numel(), column] = packed[start : start + indices.numel(), 0]
            start += indices.numel()
        return dense

    def star_rows(self, star: Tensor) -> list[Tensor]:
        """Gather every dense row from a ``[physical_len, 1, hidden]`` star tensor."""
        flat = star[:, 0]
        return [flat.index_select(0, indices.to(flat.device)) for indices in self.row_indices]

    def dense_input_grad_on_star(self, dense_grad: Tensor) -> Tensor:
        """Scatter-add dense input gradients onto star tokens (prompt copies summed)."""
        flat = torch.zeros(
            self.problem.physical_len,
            dense_grad.shape[-1],
            dtype=torch.float64,
            device=dense_grad.device,
        )
        for indices, grad in zip(self.row_indices, self.dense_rows(dense_grad)):
            flat.index_add_(0, indices.to(flat.device), grad.double())
        return flat[: self.problem.total_len]


def run_layer(module: torch.nn.Module, forward, inputs: Tensor, cotangent: Tensor) -> dict:
    """One forward/backward of ``forward(inputs)`` against a fixed output cotangent."""
    module.zero_grad(set_to_none=True)
    inputs = inputs.detach().clone().requires_grad_(True)
    output = forward(inputs)
    if isinstance(output, tuple):
        output = output[0]
    (output.float() * cotangent.float()).sum().backward()
    return {
        "output": output.detach().float(),
        "input_grad": inputs.grad.detach().float(),
        "param_grads": {
            name: param.grad.detach().double().clone()
            for name, param in module.named_parameters()
            if param.grad is not None
        },
    }


def canonical_dense_run(data: LayerProblemData, run: dict) -> dict:
    """Express a dense-batch layer run per dense row (outputs) and per star token (input grads)."""
    return {
        "output": torch.cat(data.dense_rows(run["output"])),
        "input_grad": data.dense_input_grad_on_star(run["input_grad"]),
        "param_grads": run["param_grads"],
    }


def canonical_star_run(data: LayerProblemData, run: dict) -> dict:
    """Express a shared-star layer run in the same canonical form as ``canonical_dense_run``."""
    return {
        "output": torch.cat(data.star_rows(run["output"])),
        "input_grad": run["input_grad"][: data.problem.total_len, 0].double(),
        "param_grads": run["param_grads"],
    }


def compare_canonical(candidate: dict, reference: dict) -> dict[str, float]:
    """Relative errors of a canonical layer run against a canonical reference run."""
    return {
        "output": rel_l2(candidate["output"], reference["output"]),
        "input_grad": rel_l2(candidate["input_grad"], reference["input_grad"]),
        "input_grad_max_row": max_row_error(candidate["input_grad"], reference["input_grad"]),
        "params": grads_rel_l2(candidate["param_grads"], reference["param_grads"]),
    }


def round_params_to(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """Make every parameter exactly representable in ``dtype`` (in place).

    A high-precision reference and a low-precision copy then evaluate the same function, so
    their difference measures arithmetic rounding only.
    """
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(param.to(dtype).to(param.dtype))


# Every differentiable shared-prefix Mamba backend, selected with NRL_SP_MAMBA_IMPL;
# ragged_state_fork is the default at every topology. The ``*_training`` variants run these same
# paths in training mode, and packed_fused is a single-root diagnostic oracle.
MAMBA_BACKENDS = ("ragged_state_fork", "state_fork", "replay_prefix", "packed_recurrence")


def build_local_mamba_layer(
    dtype: torch.dtype,
    *,
    hidden_size: int = 256,
    num_heads: int = 8,
    head_dim: int = 64,
    num_groups: int = 2,
    state_dim: int = 64,
    d_has_hdim: bool = False,
):
    """A MambaLayer whose projections are local (torch) linears.

    Transformer Engine runs FP32 GEMMs in TF32 (about 3e-4 relative error), which hides
    defects in the 1e-4..1e-3 range. Local ``ColumnParallelLinear``/``RowParallelLinear`` use
    ``torch.matmul``, which is IEEE FP32 while ``torch.backends.cuda.matmul.allow_tf32`` is
    False. The shared-prefix code paths are the same for both projection implementations.
    """
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
    from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        mamba_num_heads=num_heads,
        mamba_head_dim=head_dim,
        mamba_num_groups=num_groups,
        mamba_state_dim=state_dim,
        normalization="RMSNorm",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        use_cpu_initialization=True,
    )
    submodules = MambaLayerSubmodules(
        norm=WrappedTorchNorm,
        mixer=ModuleSpec(
            module=MambaMixer,
            params={"D_has_hdim": d_has_hdim},
            submodules=MambaMixerSubmodules(
                in_proj=ColumnParallelLinear, out_proj=RowParallelLinear
            ),
        ),
        mamba_bda=get_bias_dropout_add,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    layer = MambaLayer(config, submodules, pg_collection=pg_collection).cuda()
    # Like Float16Module, keep every parameter (including the norm and A_log) in ``dtype``.
    return layer.to(dtype).train()


def low_precision_copy(layer, dtype: torch.dtype, **build_kwargs):
    """A ``dtype`` MambaLayer holding exactly the (``dtype``-representable) weights of ``layer``."""
    copy = build_local_mamba_layer(dtype, **build_kwargs)
    copy.load_state_dict(layer.state_dict())
    return copy


def shared_mamba_layer_forward(layer, hidden_states: Tensor, layout, backend: str) -> Tensor:
    """Run one MambaLayer through the shared-prefix backend selected by ``NRL_SP_MAMBA_IMPL``.

    An unknown backend name fails (``_shared_prefix_mamba_impl`` raises ValueError) instead of
    silently running the default backend.
    """
    from megatron.core.ssm.mamba_shared_prefix import forward_mamba_layer_shared_prefix

    previous = os.environ.get("NRL_SP_MAMBA_IMPL")
    os.environ["NRL_SP_MAMBA_IMPL"] = backend
    try:
        return forward_mamba_layer_shared_prefix(layer, hidden_states, layout)
    finally:
        if previous is None:
            os.environ.pop("NRL_SP_MAMBA_IMPL", None)
        else:
            os.environ["NRL_SP_MAMBA_IMPL"] = previous
