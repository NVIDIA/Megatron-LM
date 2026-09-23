# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime protocol for the MiniMax-M3 lite implementation (mirrors qwen3_5/lite/protocol.py)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import warnings

import torch
import torch.nn as nn

from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
from megatron.lite.model.minimax_m3.lite.checkpoint import EXPERT_CLASSIFIER, PLACEMENT_FN
from megatron.lite.model.minimax_m3.lite.checkpoint import (
    export_hf_weights as _export_hf_weights_impl,
)
from megatron.lite.model.minimax_m3.lite.checkpoint import load_hf_weights as _load_hf_weights_impl
from megatron.lite.model.minimax_m3.lite.checkpoint import save_hf_weights as _save_hf_weights_impl
from megatron.lite.model.protocol_utils import (
    add_cross_entropy_fusion,
    nested_from_packed,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.kernels import magi_msa
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch

__all__ = [
    "EXPERT_CLASSIFIER",
    "ImplConfig",
    "PLACEMENT_FN",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
    "vocab_size",
]


def is_expert_param(name: str) -> bool:
    return EXPERT_CLASSIFIER(name)


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"
    recompute: list[str] = field(default_factory=list)
    offload: list[str] = field(default_factory=list)
    moe_dispatcher: str = "alltoall"  # alltoall | deepep | hybridep
    use_thd: bool = False
    cross_entropy_fusion: bool = False
    hf_path: str = ""
    router_aux_loss_coef: float | None = None
    deterministic: bool = False  # also selects the ordered (bitwise-reproducible) msa_v1 backward
    optimizer_config: OptimizerConfig | None = None
    # Magi CP dispatch chunk (tokens; batch padded to chunk*cp), FP32 dK group-reduce, and the kernel
    # backend of the dense layers' native calc_attn. ``fa4`` is the production path; ``sdpa_ol`` is
    # retained for tests only and materialises [H, S, S] logits in backward.
    magi_chunk_size: int = 2048
    magi_high_precision_reduce: bool = False
    magi_dense_kernel_backend: str = "fa4"


def _maybe(module_name: str) -> Callable[[nn.Module], nn.Module | None]:
    def getter(layer: nn.Module) -> nn.Module | None:
        return getattr(layer, module_name, None)

    return getter


def _moe_module(name: str) -> Callable[[nn.Module], nn.Module | None]:
    def getter(layer: nn.Module) -> nn.Module | None:
        moe = getattr(layer, "moe", None)
        return getattr(moe, name, None) if moe is not None else None

    return getter


def _core_attn(layer: nn.Module) -> nn.Module | None:
    attn = getattr(layer, "attn", None)
    # magi backend: the TE core_attn is never called; recompute/offload must target the Magi core instead
    return getattr(attn, "magi_core", None) or getattr(attn, "core_attn", None)


MODULE_MAP = {
    "core_attn": _core_attn,
    "attn": _maybe("attn"),
    "experts": _moe_module("experts"),
    "moe": _maybe("moe"),
    "router": _moe_module("router"),
    "mlp": _maybe("mlp"),
    "mlp_norm": _maybe("mlp_norm"),
    "attn_proj": lambda layer: getattr(getattr(layer, "attn", None), "proj", None),
}


def build_model_config(source: str | Path | dict, **overrides) -> MiniMaxM3Config:
    if isinstance(source, dict):
        cfg = MiniMaxM3Config._from_hf_dict(source)
    else:
        cfg = MiniMaxM3Config.from_hf(str(source))
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _unwrap(model: nn.Module) -> nn.Module:
    """Peel DDP / FSDP-style wrappers until the ``MiniMaxM3Model`` chunk (has ``layers``) is reached."""
    current = model
    while current is not None and not hasattr(current, "layers"):
        current = getattr(current, "module", None) or getattr(current, "_model", None)
    return current if current is not None else model


def _magi_plan(model: nn.Module, batch: PackedBatch) -> magi_msa.MagiMsaContext:
    """(Cached) Magi runtimes for this packed batch; every CP rank must hold the identical batch."""
    from megatron.lite.primitive.parallel.thd import parallel_state_from_model

    chunk = _unwrap(model)
    settings: magi_msa.MagiMsaSettings = chunk.magi_settings
    return magi_msa.plan_magi_batch(
        batch.seq_lens.tolist(),
        ps=parallel_state_from_model(model),
        settings=settings,
        msa_config=chunk.magi_msa_config,
        need_dense_key=chunk.has_dense_attention,
    )


def _validate_packed_batch(batch: PackedBatch) -> None:
    """Validate the global packed-batch contract consumed by MiniMax-M3 Magi."""
    seq_lens = [int(length) for length in batch.seq_lens.tolist()]
    if not seq_lens or min(seq_lens) <= 0:
        raise ValueError("MiniMax-M3 requires at least one non-empty packed document")
    total = sum(seq_lens)
    token_tensors = {
        "input_ids": batch.input_ids,
        "labels": batch.labels,
        "loss_mask": batch.loss_mask,
    }
    for name, tensor in token_tensors.items():
        if tensor is not None and tensor.numel() != total:
            raise ValueError(
                f"MiniMax-M3 packed {name} has {tensor.numel()} tokens, but seq_lens sums to {total}"
            )
    if batch.position_ids is not None:
        expected = torch.cat(
            [torch.arange(length, device=batch.position_ids.device) for length in seq_lens]
        )
        if batch.position_ids.numel() != total or not torch.equal(
            batch.position_ids.reshape(-1), expected
        ):
            raise ValueError(
                "MiniMax-M3 Magi derives document-local position_ids from seq_lens; explicit "
                "position_ids must match those document boundaries"
            )


def _forward_step(model: nn.Module, batch: PackedBatch) -> dict:
    """Pad to ``chunk*cp`` (extra trailing doc), then dispatch tokens onto Magi's CP layout.

    The forward step runs on every PP stage (the runtime key is cached), so ``magi_ctx`` needs no PP plumbing.
    ``labels``/``loss_mask`` are dispatched with the same permutation as ``input_ids``; pad tokens get
    ``loss_mask=0`` so ``_reduce_loss`` (CP-global token count) stays correct.
    """
    _validate_packed_batch(batch)
    ctx = _magi_plan(model, batch)
    input_ids = batch.input_ids.reshape(-1)
    labels = batch.labels.reshape(-1) if batch.labels is not None else None
    loss_mask = batch.loss_mask.reshape(-1) if batch.loss_mask is not None else None
    if loss_mask is None and labels is not None:
        loss_mask = torch.ones_like(labels, dtype=torch.float32)
    if ctx.pad:
        input_ids = torch.nn.functional.pad(input_ids, (0, ctx.pad), value=0)
        labels = torch.nn.functional.pad(labels, (0, ctx.pad), value=0) if labels is not None else None
        loss_mask = torch.nn.functional.pad(loss_mask, (0, ctx.pad), value=0) if loss_mask is not None else None
    kwargs: dict[str, Any] = {
        "input_ids": magi_msa.dispatch_tokens(input_ids, ctx).reshape(1, -1),
        "labels": magi_msa.dispatch_tokens(labels, ctx).reshape(1, -1) if labels is not None else None,
        "loss_mask": magi_msa.dispatch_tokens(loss_mask, ctx).reshape(1, -1) if loss_mask is not None else None,
        "packed_seq_params": None,
        "magi_ctx": ctx,
    }
    add_cross_entropy_fusion(kwargs, model)
    return model(**kwargs)


def unpack_forward_output(model: nn.Module, batch: PackedBatch, output) -> Any:
    """Restore Magi output to global packed order, drop padding, and split it by document."""
    ctx = _magi_plan(model, batch)
    flat = output[0] if output.dim() >= 2 and output.shape[0] == 1 else output
    full = magi_msa.undispatch_tokens(flat.contiguous(), ctx)[: ctx.cu_seqlens_host[-1] - ctx.pad]
    if full.dim() == 1:
        return nested_from_packed(full, batch.seq_lens)
    pieces, offset = [], 0
    for length in batch.seq_lens.tolist():
        pieces.append(full.narrow(0, offset, int(length)))
        offset += int(length)
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def _make_aux_loss_hook():
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

    def hook(scale: torch.Tensor) -> None:
        MoEAuxLossAutoScaler.set_loss_scale(scale)

    return hook


def _build_dist_opt_optimizer(chunks, model_cfg: MiniMaxM3Config, impl_cfg: ImplConfig, ps: ParallelState):
    from megatron.lite.primitive.optimizers.megatron_wrap import build_dist_opt_training_optimizer

    return build_dist_opt_training_optimizer(
        chunks,
        model_cfg=model_cfg,
        impl_cfg=impl_cfg,
        ps=ps,
        is_expert=is_expert_param,
        model_name="minimax_m3",
        deterministic=impl_cfg.deterministic,
    )


def build_model(model_cfg: MiniMaxM3Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model

    p = impl_cfg.parallel
    if impl_cfg.use_thd:
        raise NotImplementedError(
            "MiniMax-M3 lite: use_thd is not supported; packed multi-document batches are handled natively by "
            "Magi (cu_seqlens from PackedBatch.seq_lens)"
        )
    if p.tp > 1:
        raise NotImplementedError(
            "MiniMax-M3 Magi requires tp=1 in this drop (msa_v1 kernels are fixed to 64/4/4 heads); use CP/EP/PP"
        )
    if p.vpp > 1:
        raise NotImplementedError("MiniMax-M3 Magi requires vpp=1: interleaved VPP uses a fixed inter-stage shape")
    magi_msa.validate_kernel_shapes(
        num_attention_heads=model_cfg.num_attention_heads,
        num_key_value_heads=model_cfg.num_key_value_heads,
        head_dim=model_cfg.head_dim,
        index_n_heads=model_cfg.index_n_heads,
        index_head_dim=model_cfg.index_head_dim,
        block_size=model_cfg.index_block_size,
        topk_blocks=model_cfg.index_topk_blocks,
        local_blocks=model_cfg.index_local_blocks,
    )
    magi_msa.validate_device()
    if impl_cfg.deterministic and impl_cfg.magi_dense_kernel_backend == "fa4":
        warnings.warn("deterministic=True: fa4 dense backward is not deterministic; use sdpa_ol for bitwise runs", stacklevel=2)
    magi_settings = magi_msa.MagiMsaSettings(
        chunk_size=impl_cfg.magi_chunk_size,
        high_precision_reduce=impl_cfg.magi_high_precision_reduce,
        dense_kernel_backend=impl_cfg.magi_dense_kernel_backend,
        deterministic=impl_cfg.deterministic,
    )
    magi_msa_config = magi_msa.build_msa_config(
        magi_settings,
        head_dim=model_cfg.head_dim,
        index_head_dim=model_cfg.index_head_dim,
    )
    import os

    os.environ.setdefault("MAGI_ATTENTION_KERNEL_BACKEND", magi_settings.dense_kernel_backend)
    if magi_settings.dense_kernel_backend == "fa4":
        # FA4 rejects a non-zero SM margin (Magi defaults to 4 when CUDA_DEVICE_MAX_CONNECTIONS > 1)
        os.environ.setdefault("MAGI_ATTENTION_FFA_FORWARD_SM_MARGIN", "0")
        os.environ.setdefault("MAGI_ATTENTION_FFA_BACKWARD_SM_MARGIN", "0")
    if impl_cfg.moe_dispatcher not in ("alltoall", "deepep", "hybridep"):
        raise ValueError(f"moe_dispatcher must be alltoall, deepep or hybridep, got {impl_cfg.moe_dispatcher!r}")
    if impl_cfg.moe_dispatcher != "alltoall" and (p.etp is not None and p.etp > 1):
        raise ValueError(f"moe_dispatcher={impl_cfg.moe_dispatcher!r} and etp>1 are mutually exclusive")
    if impl_cfg.router_aux_loss_coef is not None:
        model_cfg.router_aux_loss_coef = impl_cfg.router_aux_loss_coef

    ps = init_parallel(p)
    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    vpp = None if p.vpp == 1 else p.vpp
    train_cfg = SimpleNamespace(
        tp=ps.tp_size,
        ep=ps.ep_size,
        etp=ps.etp_size,
        pp=ps.pp_size,
        cp=ps.cp_size,
        vpp=vpp,
        moe_dispatcher=impl_cfg.moe_dispatcher,
        fp8=False,
        recompute_modules=recompute_spec,
        deterministic=impl_cfg.deterministic,
    )
    kwargs = dict(msa_backend="magi")
    if vpp is None:
        chunks = [MiniMaxM3Model(model_cfg, train_cfg, ps, **kwargs).to(torch.bfloat16).cuda()]
    else:
        chunks = [
            MiniMaxM3Model(model_cfg, train_cfg, ps, vpp_chunk_id=i, **kwargs).to(torch.bfloat16).cuda()
            for i in range(vpp)
        ]
    for chunk in chunks:
        chunk.magi_settings = magi_settings
        chunk.magi_msa_config = magi_msa_config
        chunk.has_dense_attention = any(not layer.is_sparse_attention for layer in chunk.layers)
    if recompute_spec:
        for chunk in chunks:
            apply_recompute(chunk.layers, recompute_spec, MODULE_MAP)
    if impl_cfg.offload:
        from megatron.lite.primitive.recompute import apply_offload

        for chunk in chunks:
            apply_offload(chunk.layers, impl_cfg.offload, MODULE_MAP)

    optimizer = None
    finalize_grads = None
    post_model_load_hook = None
    optimizer_backend = "none"
    if impl_cfg.optimizer == "dist_opt":
        optimizer, finalize_grads = _build_dist_opt_optimizer(chunks, model_cfg, impl_cfg, ps)
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        attach_model_sharded_state_dict(chunks, ps, get_placements=PLACEMENT_FN, is_expert=is_expert_param)
        register_training_hooks(chunks, optimizer)
        optimizer_backend = "dist_opt"
    elif impl_cfg.optimizer == "fsdp2":
        optimizer_backend = "fsdp2"

        def _post_model_load_hook():
            from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Layer
            from megatron.lite.primitive.optimizers.fsdp2 import build_fsdp2_training_optimizer

            return {
                "optimizer": build_fsdp2_training_optimizer(
                    chunks,
                    impl_cfg.optimizer_config,
                    ps,
                    unit_modules=(MiniMaxM3Layer,),
                    expert_classifier=is_expert_param,
                    deterministic=impl_cfg.deterministic,
                    vpp=impl_cfg.parallel.vpp,
                    leaf_module_names=(),
                )
            }

        post_model_load_hook = _post_model_load_hook
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
            "post_model_load_hook": post_model_load_hook,
            "pre_forward_hook": _make_aux_loss_hook(),
        },
    )


def load_hf_weights(chunk: nn.Module, hf_path: str, model_cfg: MiniMaxM3Config, ps: ParallelState) -> None:
    if hf_path:
        _load_hf_weights_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(chunks: list[nn.Module], model_cfg: MiniMaxM3Config, ps: ParallelState, **kwargs):
    yield from _export_hf_weights_impl(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(chunks: list[nn.Module], path: str, model_cfg: MiniMaxM3Config, ps: ParallelState, **kwargs) -> None:
    _save_hf_weights_impl(chunks, path, model_cfg, ps, **kwargs)


def vocab_size(model_cfg) -> int | None:
    cfg = getattr(model_cfg, "text_config", model_cfg)
    return getattr(cfg, "vocab_size", None)
