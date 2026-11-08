# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 lite model protocol for the Megatron Lite runtime.

Data path per microbatch (``_forward_step``): documents are packed without
padding; the Magi-MSA runtime key is built once from host ``cu_seqlens`` and
cached on ``batch.extras`` so recompute reuses it, then the embedding (first
stage) and the labels / loss mask (last stage) are dispatched with it. Labels
are the next token within each document; the last token of every document and
tokens with ``loss_mask == 0`` get label ``-100``.

Scope: TP = ETP = 1 (the MSA v1 kernels fix the head counts); CP, PP, EP, DP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.primitive.modules.attention.msa import (
    MsaV1Numerics,
    build_msa_runtime_key,
    msa_dispatch,
    msa_position_ids,
    rotary_freqs_for_positions,
)
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.parallel.thd import parallel_state_from_model
from megatron.lite.primitive.quantization import QATSpec, apply_qat_to_chunks, normalize_qat_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch

_MSA_KEY = "_mlite_msa_runtime_key"
IGNORE_INDEX = -100


def EXPERT_CLASSIFIER(name: str) -> bool:
    return ".experts." in name and "shared" not in name


def PLACEMENT_FN(param_name: str) -> list:
    from megatron.lite.model.minimax_m3.lite.checkpoint import PLACEMENT_FN as placement_fn

    return placement_fn(param_name)


def is_expert_param(name: str) -> bool:
    return EXPERT_CLASSIFIER(name)


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"
    optimizer_config: OptimizerConfig | None = None
    use_deepep: bool = False
    # Deterministic kernels (Magi-MSA ordered reductions, deterministic
    # embedding / MoE index ops); requires torch deterministic algorithms.
    deterministic: bool = False
    # Magi dispatch chunk; None = tokens // cp // 2.
    msa_chunk_size: int | None = None
    qat: QATSpec | dict | None = None


def build_model_config(source: str | Path | dict, **overrides) -> MiniMaxM3Config:
    if isinstance(source, dict):
        return MiniMaxM3Config._from_hf_dict(source, **overrides)
    return MiniMaxM3Config.from_hf(str(source), **overrides)


def _validate_parallel_scope(p: ParallelConfig) -> None:
    etp = 1 if p.etp is None else p.etp
    if p.tp > 1 or etp > 1:
        raise NotImplementedError(f"MiniMax-M3 supports tp=etp=1 only, got tp={p.tp} etp={etp}")
    if p.vpp != 1:
        raise NotImplementedError("MiniMax-M3 does not support VPP")


def msa_numerics(config: MiniMaxM3Config) -> MsaV1Numerics:
    return MsaV1Numerics(
        head_dim=config.head_dim,
        index_dim=config.sparse_index_dim,
        block_size=config.sparse_block_size,
        topk_blocks=config.sparse_topk_blocks,
        init_blocks=config.sparse_init_block,
        local_blocks=config.sparse_local_block,
        kl_loss_coeff=config.sparse_kl_loss_coeff,
        kl_temperature=config.sparse_kl_temperature,
    )


def _shift_within_documents(
    labels: torch.Tensor, loss_mask: torch.Tensor | None, cu_seqlens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token labels per document; last token of each document and masked tokens -> -100."""
    total = labels.numel()
    device = labels.device
    ends = cu_seqlens[1:].to(device=device, dtype=torch.long) - 1
    nxt = torch.arange(1, total + 1, device=device)
    nxt[ends] = ends
    mask = torch.ones(total, device=device) if loss_mask is None else loss_mask.reshape(-1).float()
    shifted_labels = labels.reshape(-1)[nxt].clone()
    shifted_mask = mask[nxt].clone()
    shifted_mask[ends] = 0
    shifted_labels[shifted_mask == 0] = IGNORE_INDEX
    return shifted_labels, shifted_mask


def _aux_loss_scale() -> float:
    """The runtime's auxiliary-loss gradient scale (``1 / num_microbatches``), which the
    pre-forward hook sets before every microbatch. The MSA KL gradient is injected in backward,
    outside the loss scalar, so its coefficient takes this scale explicitly."""
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

    scale = MoEAuxLossAutoScaler.main_loss_backward_scale
    return 1.0 if scale is None else float(scale)


def get_msa_runtime_key(model: nn.Module, batch: PackedBatch) -> Any:
    """Build (once per microbatch) and cache the Magi-MSA runtime key on ``batch.extras``."""
    key = batch.extras.get(_MSA_KEY)
    if key is not None:
        return key
    ps = parallel_state_from_model(model) or ParallelState()
    config: MiniMaxM3Config = unwrap_model(model).config
    impl = unwrap_model(model).impl_options
    seq_lens = batch.seq_lens.detach().to("cpu", torch.int64)
    cu_seqlens = torch.zeros(seq_lens.numel() + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seq_lens, 0).to(torch.int32)
    total = int(cu_seqlens[-1])
    key = build_msa_runtime_key(
        cu_seqlens,
        ps.cp_group,
        numerics=msa_numerics(config),
        chunk_size=impl["msa_chunk_size"] or total // ps.cp_size // 2,
        tp_size=ps.tp_size,
        grad_scale=_aux_loss_scale(),
        deterministic=impl["deterministic"],
        dense_heads=(
            (config.num_attention_heads, config.num_key_value_heads)
            if config.has_full_attention_layers
            else None
        ),
    )
    batch.extras[_MSA_KEY] = key
    batch.extras["_mlite_msa_cu_seqlens"] = cu_seqlens
    return key


def prepare_forward_kwargs(model: nn.Module, batch: PackedBatch) -> dict[str, Any]:
    config: MiniMaxM3Config = unwrap_model(model).config
    key = get_msa_runtime_key(model, batch)
    cu_seqlens = batch.extras["_mlite_msa_cu_seqlens"]
    kwargs: dict[str, Any] = {
        "runtime_key": key,
        "rotary_freqs": rotary_freqs_for_positions(
            msa_position_ids(key), rotary_dim=config.rotary_dim, base=config.rope_theta
        ),
        "dispatch": lambda t: msa_dispatch(t, key),
    }
    if unwrap_model(model).pre_process:
        kwargs["input_ids"] = batch.input_ids.reshape(1, -1)
    if unwrap_model(model).post_process and batch.labels is not None:
        labels, mask = _shift_within_documents(batch.labels, batch.loss_mask, cu_seqlens)
        kwargs["labels"] = msa_dispatch(labels, key)
        kwargs["loss_mask"] = msa_dispatch(mask, key)
    return kwargs


def _forward_step(model: nn.Module, batch: PackedBatch) -> dict:
    out = model(**prepare_forward_kwargs(model, batch))
    if "loss_sum" in out:
        # Token mean of this rank's share; DP x CP gradients are averaged by the optimizer.
        out["loss"] = out["loss_sum"] / out["num_tokens"].clamp(min=1)
    return out


def _make_aux_loss_hook():
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

    def hook(scale: torch.Tensor) -> None:
        MoEAuxLossAutoScaler.set_loss_scale(scale)

    return hook


def _build_dist_opt_optimizer(chunks, model_cfg, impl_cfg: ImplConfig, ps: ParallelState):
    from megatron.lite.primitive.optimizers.megatron_wrap import build_dist_opt_training_optimizer

    return build_dist_opt_training_optimizer(
        chunks,
        model_cfg=model_cfg,
        impl_cfg=impl_cfg,
        ps=ps,
        model_name="minimax_m3",
        is_expert=is_expert_param,
        deterministic=impl_cfg.deterministic,
    )


def build_chunk(model_cfg: MiniMaxM3Config, impl_cfg: ImplConfig, ps: ParallelState) -> nn.Module:
    """Build this rank's model chunk (bf16, on GPU) without optimizer or QAT."""
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model

    chunk = MiniMaxM3Model(
        model_cfg, ps, deterministic=impl_cfg.deterministic, use_deepep=impl_cfg.use_deepep
    )
    chunk = chunk.to(torch.bfloat16).cuda()
    chunk.impl_options = {
        "deterministic": impl_cfg.deterministic,
        "msa_chunk_size": impl_cfg.msa_chunk_size,
    }
    return chunk


def build_model(model_cfg: MiniMaxM3Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    p = impl_cfg.parallel
    _validate_parallel_scope(p)
    ps = init_parallel(p)
    chunks = [build_chunk(model_cfg, impl_cfg, ps)]

    # Parametrize before optimizer construction so it captures the BF16 master.
    apply_qat_to_chunks(chunks, normalize_qat_spec(impl_cfg.qat))

    optimizer = None
    finalize_grads = None
    optimizer_backend = "none"
    if impl_cfg.optimizer == "dist_opt":
        optimizer, finalize_grads = _build_dist_opt_optimizer(chunks, model_cfg, impl_cfg, ps)
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        attach_model_sharded_state_dict(
            chunks, ps, get_placements=PLACEMENT_FN, is_expert=is_expert_param
        )
        register_training_hooks(chunks, optimizer)
        optimizer_backend = "dist_opt"
    elif impl_cfg.optimizer is not None:
        raise ValueError(f"Unknown minimax_m3 lite optimizer: {impl_cfg.optimizer!r}.")

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=_forward_step,
        extras={
            "model_cfg": model_cfg,
            "optimizer_backend": optimizer_backend,
            "pre_forward_hook": _make_aux_loss_hook(),
        },
    )


def load_hf_weights(chunk: nn.Module, hf_path: str, model_cfg: MiniMaxM3Config, ps) -> None:
    if not hf_path:
        return
    from megatron.lite.model.minimax_m3.lite.checkpoint import load_hf_weights as load_impl

    load_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(chunks, model_cfg: MiniMaxM3Config, ps, **kwargs):
    from megatron.lite.model.minimax_m3.lite.checkpoint import export_hf_weights as export_impl

    yield from export_impl(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(chunks, path: str, model_cfg: MiniMaxM3Config, ps, **kwargs) -> None:
    from megatron.lite.model.minimax_m3.lite.checkpoint import save_hf_weights as save_impl

    save_impl(chunks, path, model_cfg, ps, **kwargs)


def vocab_size(model_cfg) -> int | None:
    return getattr(model_cfg, "vocab_size", None)


__all__ = [
    "EXPERT_CLASSIFIER",
    "ImplConfig",
    "PLACEMENT_FN",
    "build_chunk",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "get_msa_runtime_key",
    "is_expert_param",
    "load_hf_weights",
    "prepare_forward_kwargs",
    "save_hf_weights",
    "vocab_size",
]
