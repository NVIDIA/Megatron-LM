# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-level dense-row references for the shared-prefix HybridModel tests.

They extend the layer-level problems in ``tests/unit_tests/ssm/shared_prefix_test_utils.py`` to
whole models: a tiny HybridModel runs the dense rows and the shared layout, with MoE routing
optionally replayed, and the runs are compared per dense row.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from megatron.core.shared_prefix_layout import SharedPrefixLayout
from tests.unit_tests.ssm.shared_prefix_test_utils import SharedPrefixProblem, rel_l2

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
    model: torch.nn.Module,
    tokens: TokenProblem,
    routing: ReplayedRouting | None = None,
    *,
    mask_sequence_padding: bool = False,
    **forward_kwargs,
) -> ModelRun:
    """Run every dense row as its own ordinary forward/backward (CP zigzag, TP/SP).

    ``mask_sequence_padding`` passes each row's per-sequence padding as ``padding_mask``.
    ``forward_kwargs`` are passed to every ``model(...)`` call unchanged.
    """
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
        if mask_sequence_padding:
            padding = torch.arange(row.dense_len) >= row.prefix_len + row.logical_len
            extra["padding_mask"] = padding[local][None].cuda()
        output = model(
            input_ids=ids[local][None].cuda(),
            position_ids=local[None].cuda(),
            attention_mask=None,
            **extra,
            **forward_kwargs,
        )
        loss = (output.float() * _vocab_shard(cotangent[local])[None].cuda()).sum()
        loss.backward()
        full = _gather_logits(output.detach().float(), local, row.dense_len)
        logits.append(full[predictors.cuda()].cpu())
    return ModelRun(logits, _reduced_grads(model), _summed_counts(model))


def run_shared(
    model: torch.nn.Module,
    tokens: TokenProblem,
    layout,
    routing: ReplayedRouting | None = None,
    **forward_kwargs,
) -> ModelRun:
    """Run the whole problem as one shared-prefix forward/backward.

    ``forward_kwargs`` are passed to the ``model(...)`` call unchanged.
    """
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
        **forward_kwargs,
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
