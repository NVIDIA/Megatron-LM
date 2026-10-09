# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dense-row references for the shared-prefix numerical tests.

A shared-prefix layout stores each prompt once and appends its completions:
``[prompt, completion_1, ..., completion_G]`` per root, roots concatenated, then optional
topology padding. A conventional (dense) batch runs every row ``[prompt, completion_g]`` on
its own. The tests build both token orders from one canonical problem, run the module under
test in each order, and compare the results per dense row:

* outputs and logits are compared row by row;
* input gradients are compared after summing the dense prompt copies, which is the gradient
  the single shared prompt row must receive;
* parameter gradients and MoE expert-bias token counts are compared directly.

All differences are whole-tensor relative L2 norms.
"""

import os
from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor

from megatron.core.models.hybrid.shared_prefix_layout import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
)


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
        """Canonical token identities for routing replay.

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
        """Inverse of concatenating the real dense rows: ``[sum(len), 1, h]`` -> dense batch."""
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


# Every shared-prefix Mamba execution path. ``star_cp1`` is the TP1/CP1 single-root path the
# Hybrid stack dispatches plain layouts to; the others are selected with NRL_SP_MAMBA_IMPL.
MAMBA_BACKENDS = (
    "star_cp1",
    "state_fork",
    "replay_prefix",
    "packed_recurrence",
    "ragged_state_fork",
)


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
    """Run one MambaLayer through the named shared-prefix backend.

    Backends removed from the implementation are skipped, not failed, so the matrix keeps
    covering whatever remains.
    """
    import pytest

    from megatron.core.models.hybrid import shared_prefix

    if backend == "star_cp1":
        if isinstance(layout, SharedPrefixForestLayout):
            pytest.skip("the TP1/CP1 single-star path executes one root")
        forward = getattr(shared_prefix, "_forward_mamba_layer_shared_prefix", None)
        if forward is None:
            pytest.skip("the TP1/CP1 single-star Mamba path no longer exists")
        return forward(layer, hidden_states, layout)
    previous = os.environ.get("NRL_SP_MAMBA_IMPL")
    os.environ["NRL_SP_MAMBA_IMPL"] = backend
    try:
        return shared_prefix._forward_mamba_layer_shared_prefix_cp(layer, hidden_states, layout)
    except ValueError as error:
        if "NRL_SP_MAMBA_IMPL" in str(error):
            pytest.skip(f"shared-prefix Mamba backend {backend!r} no longer exists")
        raise
    finally:
        if previous is None:
            os.environ.pop("NRL_SP_MAMBA_IMPL", None)
        else:
            os.environ["NRL_SP_MAMBA_IMPL"] = previous


# ----------------------------------------------------------------------------- model level
# With Megatron's default init std (0.02) attention is nearly uniform, so wrong RoPE positions
# change the logits by less than BF16 rounding (measured: an off-by-one completion position moved
# the error from 0.729% to 0.730%). At 0.1 attention is peaked and position errors are visible.
ROPE_SENSITIVE_INIT_STD = 0.1


def build_hybrid_model(
    pattern: str, dtype: torch.dtype, *, position_embedding_type: str = "rope", **overrides
):
    """A tiny Mamba/attention/MoE HybridModel on the current TP/CP groups.

    ``attention_backend=auto`` lets Transformer Engine choose FlashAttention or cuDNN for
    16-bit dense rows and the unfused kernel for an FP32 reference. Callers must clear the NVTE
    attention environment variables first (see ``clear_attention_env``).
    """
    from megatron.core import parallel_state
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend

    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    cp_size = parallel_state.get_context_parallel_world_size()
    config = dict(
        num_layers=len(pattern.split("/")[0]),
        hidden_size=256,
        num_attention_heads=8,
        num_query_groups=4,
        kv_channels=32,
        ffn_hidden_size=512,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        normalization="RMSNorm",
        add_bias_linear=False,
        mamba_state_dim=32,
        mamba_head_dim=32,
        mamba_num_groups=4,
        mamba_num_heads=16,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=256,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False,
        moe_router_dtype="fp32",
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        attention_backend=AttnBackend.auto,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        sequence_parallel=tp_size > 1,
        init_method_std=ROPE_SENSITIVE_INIT_STD,
    )
    config.update(overrides)
    model = HybridModel(
        config=TransformerConfig(**config),
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=2048,
        max_sequence_length=8192,
        hybrid_layer_pattern=pattern,
        position_embedding_type=position_embedding_type,
        share_embeddings_and_output_weights=False,
    )
    return model.cuda().train()


def clear_attention_env(monkeypatch) -> None:
    """Let ``attention_backend`` own the process-wide NVTE attention switches for this test."""
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(name, raising=False)


def copy_params(source: torch.nn.Module, target: torch.nn.Module) -> None:
    """Copy every parameter of ``source`` into the same-named parameter of ``target``."""
    target_params = dict(target.named_parameters())
    with torch.no_grad():
        for name, param in source.named_parameters():
            target_params[name].copy_(param)


class ReplayedRouting:
    """Fix every MoE top-k decision to a per-token table in every execution order.

    Natural routing lets rounding noise flip near-tied top-k choices, which inflates every
    comparison (including dense against dense) and makes token counts differ for reasons
    unrelated to shared-prefix execution. Each canonical token (a shared prompt token, or one
    branch token) is assigned fixed experts through ``RouterReplay``; router logits, scores and
    their gradients still run. The routers are instrumented directly, not through
    ``moe_enable_routing_replay``.
    """

    def __init__(self, model: torch.nn.Module, num_keys: int, seed: int):
        from megatron.core.transformer.moe.router import TopKRouter
        from megatron.core.transformer.moe.router_replay import RouterReplay

        config = model.config
        generator = torch.Generator().manual_seed(seed)
        scores = torch.rand(num_keys, config.num_moe_experts, generator=generator)
        self.table = scores.argsort(dim=1)[:, : config.moe_router_topk].contiguous()
        self.routers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, TopKRouter)
        ]
        self.replays = []
        for _, router in self.routers:
            replay = RouterReplay()
            router.router_replay = replay
            self.replays.append(replay)

    def set(self, decoder_keys: Tensor, mtp_keys: Tensor | None = None) -> None:
        """Replay the experts of ``decoder_keys`` (and ``mtp_keys`` in MTP routers) next."""
        from megatron.core.transformer.moe.router_replay import RouterReplayAction

        for (name, _), replay in zip(self.routers, self.replays):
            keys = mtp_keys if mtp_keys is not None and name.startswith("mtp.") else decoder_keys
            replay.clear_indices()
            replay.set_target_indices(self.table.index_select(0, keys.cpu()).cuda())
            replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    def close(self) -> None:
        """Detach the replay instrumentation from the routers."""
        from megatron.core.transformer.moe.router_replay import RouterReplay

        for (_, router), replay in zip(self.routers, self.replays):
            router.router_replay = None
            if replay in RouterReplay.global_router_replay_instances:
                RouterReplay.global_router_replay_instances.remove(replay)


@dataclass
class ModelRun:
    """Completion logits per dense row, reduced gradients and world-summed expert counts."""

    logits: list[Tensor]
    grads: dict[str, Tensor]
    counts: list[Tensor]


class TokenProblem:
    """Token IDs and RL-style logit cotangents for a ``SharedPrefixProblem``.

    A row predicts its completion from positions ``P - 1 .. P + L - 2``; only those logits get
    a cotangent. The shared prompt's last row therefore receives the sum of the G cotangents.
    """

    def __init__(self, problem: SharedPrefixProblem, vocab_size: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.problem = problem
        star_ids = torch.zeros(problem.physical_len, dtype=torch.long)
        self.star_loss_mask = torch.zeros(problem.physical_len)
        prompts = {row.prompt_offset: row.prefix_len for row in problem.rows}
        for start, length in sorted(prompts.items()):
            star_ids[start : start + length] = torch.randint(
                1, vocab_size, (length,), generator=generator
            )
        for row in problem.rows:
            start = row.completion_offset
            star_ids[start : start + row.logical_len] = torch.randint(
                1, vocab_size, (row.logical_len,), generator=generator
            )
            self.star_loss_mask[start : start + row.logical_len] = 1
        self.star_ids = star_ids
        self.row_indices = [row.star_indices() for row in problem.rows]
        self.row_ids = [star_ids.index_select(0, indices) for indices in self.row_indices]
        self.row_loss_masks = [
            self.star_loss_mask.index_select(0, indices) for indices in self.row_indices
        ]
        self.predictors = [
            torch.arange(row.prefix_len - 1, row.prefix_len + row.logical_len - 1)
            for row in problem.rows
        ]
        self.row_cotangents = []
        for row, predictors in zip(problem.rows, self.predictors):
            cotangent = torch.zeros(row.dense_len, vocab_size)
            cotangent[predictors] = torch.randn(predictors.numel(), vocab_size, generator=generator)
            self.row_cotangents.append(cotangent)
        self.star_cotangent = torch.zeros(problem.physical_len, vocab_size)
        for indices, cotangent in zip(self.row_indices, self.row_cotangents):
            self.star_cotangent.index_add_(0, indices, cotangent)
        self.star_keys, self.row_keys, self.num_keys = problem.token_keys()


def _parallel_state():
    from megatron.core import parallel_state

    return (
        parallel_state.get_tensor_model_parallel_group(),
        parallel_state.get_tensor_model_parallel_rank(),
        parallel_state.get_tensor_model_parallel_world_size(),
        parallel_state.get_context_parallel_group(),
        parallel_state.get_context_parallel_rank(),
        parallel_state.get_context_parallel_world_size(),
    )


def _cp_local(length: int) -> Tensor:
    _, _, _, _, cp_rank, cp_size = _parallel_state()
    if cp_size == 1:
        return torch.arange(length)
    return SharedPrefixLayout.cp_local_indices(length, cp_size, cp_rank, "cpu")


def _sequence_shard(values: Tensor) -> Tensor:
    """This rank's sequence-parallel shard of CP-local values (routers see SP shards)."""
    _, tp_rank, tp_size, _, _, _ = _parallel_state()
    return values.chunk(tp_size)[tp_rank]


def _vocab_shard(values: Tensor) -> Tensor:
    _, tp_rank, tp_size, _, _, _ = _parallel_state()
    width = values.shape[-1] // tp_size
    return values[..., tp_rank * width : (tp_rank + 1) * width]


def _gather_logits(local: Tensor, local_indices: Tensor, length: int) -> Tensor:
    """``[1, length/CP, V/TP]`` local logits -> ``[length, V]`` in global token order."""
    tp_group, _, tp_size, cp_group, _, cp_size = _parallel_state()
    local = local[0].contiguous()
    if tp_size > 1:
        shards = [torch.empty_like(local) for _ in range(tp_size)]
        torch.distributed.all_gather(shards, local, group=tp_group)
        local = torch.cat(shards, dim=-1)
    if cp_size == 1:
        return local
    parts = [torch.empty_like(local) for _ in range(cp_size)]
    torch.distributed.all_gather(parts, local.contiguous(), group=cp_group)
    indices = [torch.empty_like(local_indices.cuda()) for _ in range(cp_size)]
    torch.distributed.all_gather(indices, local_indices.cuda(), group=cp_group)
    output = torch.empty(length, local.shape[-1], dtype=local.dtype, device=local.device)
    for part, index in zip(parts, indices):
        output[index] = part
    return output


def _reduced_grads(model: torch.nn.Module) -> dict[str, Tensor]:
    """Gradients summed over CP, and over TP for TP-replicated (sequence-parallel) parameters."""
    tp_group, _, _, cp_group, _, _ = _parallel_state()
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            grad = torch.zeros_like(param, dtype=torch.float64)
        else:
            grad = param.grad.detach().double().clone()
        if not getattr(param, "tensor_model_parallel", False):
            torch.distributed.all_reduce(grad, group=tp_group)
        torch.distributed.all_reduce(grad, group=cp_group)
        grads[name] = grad
    return grads


def _routers_with_counts(model: torch.nn.Module) -> list:
    from megatron.core.transformer.moe.router import TopKRouter

    return [
        module
        for module in model.modules()
        if isinstance(module, TopKRouter) and module.local_tokens_per_expert is not None
    ]


def _summed_counts(model: torch.nn.Module) -> list[Tensor]:
    tp_group, _, _, cp_group, _, _ = _parallel_state()
    counts = []
    for router in _routers_with_counts(model):
        count = router.local_tokens_per_expert.detach().clone()
        torch.distributed.all_reduce(count, group=tp_group)
        torch.distributed.all_reduce(count, group=cp_group)
        counts.append(count.cpu())
    return counts


def _start_run(model: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=True)
    for router in _routers_with_counts(model):
        router.local_tokens_per_expert.zero_()


def run_dense_rows(
    model: torch.nn.Module, tokens: TokenProblem, routing: ReplayedRouting | None = None
) -> ModelRun:
    """Run every dense row as its own ordinary forward/backward (CP zigzag, TP/SP)."""
    _start_run(model)
    with_mtp = bool(getattr(model, "mtp_process", False))
    logits = []
    for row, ids, loss_mask, cotangent, keys, predictors in zip(
        tokens.problem.rows,
        tokens.row_ids,
        tokens.row_loss_masks,
        tokens.row_cotangents,
        tokens.row_keys,
        tokens.predictors,
    ):
        local = _cp_local(row.dense_len)
        if routing is not None:
            routing.set(_sequence_shard(keys.index_select(0, local)))
        extra = {"loss_mask": loss_mask[local][None].cuda()} if with_mtp else {}
        output = model(
            input_ids=ids[local][None].cuda(),
            position_ids=local[None].cuda(),
            attention_mask=None,
            **extra,
        )
        loss = (output.float() * _vocab_shard(cotangent[local])[None].cuda()).sum()
        loss.backward()
        full = _gather_logits(output.detach().float(), local, row.dense_len)
        logits.append(full[predictors.cuda()].cpu())
    return ModelRun(logits, _reduced_grads(model), _summed_counts(model))


def run_shared(
    model: torch.nn.Module, tokens: TokenProblem, layout, routing: ReplayedRouting | None = None
) -> ModelRun:
    """Run the whole problem as one shared-prefix forward/backward."""
    _start_run(model)
    problem = tokens.problem
    with_mtp = bool(getattr(model, "mtp_process", False))
    local = _cp_local(problem.physical_len)
    if routing is not None:
        # The MTP heads run on the dense branches, packed branch-major in CP-local order.
        mtp_keys = torch.cat(
            [keys.index_select(0, _cp_local(keys.numel())) for keys in tokens.row_keys]
        )
        routing.set(
            _sequence_shard(tokens.star_keys.index_select(0, local)), _sequence_shard(mtp_keys)
        )
    extra = {"loss_mask": tokens.star_loss_mask[local][None].cuda()} if with_mtp else {}
    output = model(
        input_ids=tokens.star_ids[local][None].cuda(),
        position_ids=None,
        attention_mask=None,
        shared_prefix_layout=layout,
        **extra,
    )
    loss = (output.float() * _vocab_shard(tokens.star_cotangent[local])[None].cuda()).sum()
    loss.backward()
    full = _gather_logits(output.detach().float(), local, problem.physical_len)
    logits = [
        full[indices[predictors].cuda()].cpu()
        for indices, predictors in zip(tokens.row_indices, tokens.predictors)
    ]
    return ModelRun(logits, _reduced_grads(model), _summed_counts(model))


def model_grads_rel_l2(
    candidate: dict[str, Tensor], reference: dict[str, Tensor], model, prefix: str = ""
) -> float:
    """Gradient relative L2 over parameters named ``prefix*``, each TP shard counted once."""
    tp_group, tp_rank, _, _, _, _ = _parallel_state()
    totals = torch.zeros(2, dtype=torch.float64, device="cuda")
    for name, param in model.named_parameters():
        if not name.startswith(prefix):
            continue
        if not getattr(param, "tensor_model_parallel", False) and tp_rank != 0:
            continue
        diff = (candidate[name] - reference[name]).pow(2).sum()
        totals += torch.stack([diff, reference[name].pow(2).sum()]).to(totals.device)
    torch.distributed.all_reduce(totals, group=tp_group)
    numerator, denominator = totals.tolist()
    return (numerator / denominator) ** 0.5 if denominator > 0 else 0.0


def compare_model_runs(candidate: ModelRun, reference: ModelRun, model) -> dict[str, float]:
    """Completion-logit and whole-gradient relative errors of two model runs.

    Models with MTP heads also report the MTP-head gradients alone (``mtp_grads``).
    """
    errors = {
        "logits": rel_l2(torch.cat(candidate.logits), torch.cat(reference.logits)),
        "grads": model_grads_rel_l2(candidate.grads, reference.grads, model),
    }
    if getattr(model, "mtp_process", False):
        errors["mtp_grads"] = model_grads_rel_l2(
            candidate.grads, reference.grads, model, prefix="mtp."
        )
    return errors
