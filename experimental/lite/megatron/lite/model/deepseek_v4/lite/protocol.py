# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.checkpoint import (
    EXPERT_CLASSIFIER,
    PLACEMENT_FN,
    export_hf_weights as _export_hf_weights_impl,
    load_hf_weights as _load_hf_weights_impl,
    save_hf_weights as _save_hf_weights_impl,
)
from megatron.lite.model.protocol_utils import add_loss_context_kwargs
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.parallel.cp import (
    contiguous_position_ids_for_cp,
    contiguous_slice_for_cp,
    local_position_ids_for_cp,
    local_sequence_tensor_for_cp,
)
from megatron.lite.primitive.parallel.thd import (
    pack_nested_thd,
    parallel_state_from_model,
    thd_pack_meta,
    unpack_thd_to_nested,
)
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, PackedBatch, ParallelConfig


def is_expert_param(name: str) -> bool:
    return EXPERT_CLASSIFIER(name)


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"
    optimizer_config: OptimizerConfig | None = None
    hf_path: str = ""
    recompute: list[str] = field(default_factory=list)
    offload: list[str] = field(default_factory=list)
    use_thd: bool = False
    use_deepep: bool = False
    attention_backend_override: str | None = None
    deterministic: bool = True
    mtp_enable: bool = True
    mtp_enable_train: bool = False
    mtp_detach_encoder: bool = False
    mtp_num_layers: int | None = None
    num_nextn_predict_layers: int | None = None
    mtp_loss_scaling_factor: float = 0.1


MODULE_MAP = {
    "attn": lambda layer: layer.self_attn,
    "core_attn": lambda layer: layer.self_attn,
    "moe": lambda layer: layer.mlp,
    "experts": lambda layer: layer.mlp.experts,
    "router": lambda layer: layer.mlp.gate,
    "attn_norm": lambda layer: layer.input_layernorm,
    "ffn_norm": lambda layer: layer.post_attention_layernorm,
}

# The Kimi-derived model has no ``attention_mask`` arg (CSA derives its causal /
# sliding-window masking from ``position_ids``, as the previous DS4 did with
# attention_mask=None); keep it out of the forward whitelist.
_MODEL_FORWARD_KEYS = (
    "input_ids",
    "position_ids",
    "labels",
    "loss_mask",
    "temperature",
    "calculate_entropy",
    "enable_mtp",
)


def build_model_config(source: str | Path | dict, **overrides) -> DeepseekV4Config:
    if isinstance(source, dict):
        cfg = DeepseekV4Config._from_hf_dict(source)
    else:
        cfg = DeepseekV4Config.from_hf(str(source))
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _normalize_ds4_position_ids(position_ids):
    if position_ids is None:
        return None
    if position_ids.dim() == 3:
        if position_ids.size(0) == 3:
            position_ids = position_ids[0]
        elif position_ids.size(1) == 1:
            position_ids = position_ids.squeeze(1)
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    return position_ids


def _as_batch_row(tensor):
    if tensor is not None and tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def _infer_cp_local_seq_len(
    *,
    input_ids,
    position_ids,
    cp_size,
):
    seq_len = input_ids.size(1)
    if cp_size <= 1:
        return seq_len
    if position_ids is not None and position_ids.size(-1) in (seq_len, seq_len * cp_size):
        return seq_len
    return seq_len // cp_size if seq_len % cp_size == 0 else seq_len


def _nested_from_packed_tensor(tensor, seq_lens):
    if tensor is None:
        return None
    if tensor.dim() == 2 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 1:
        raise ValueError(f"PackedBatch tensor must be 1-D, got {tuple(tensor.shape)}.")

    pieces = []
    offset = 0
    for length_t in seq_lens:
        length = int(length_t.item())
        pieces.append(tensor.narrow(0, offset, length))
        offset += length
    if offset != tensor.numel():
        raise ValueError(f"PackedBatch sizes sum to {offset}, tensor has {tensor.numel()} tokens.")
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def _prepare_packed_batch_kwargs(model, batch: PackedBatch) -> dict[str, Any]:
    ps = parallel_state_from_model(model) or ParallelState()
    seq_lens = batch.sizes().to(device=batch.input_ids.device)
    packed = pack_nested_thd(
        _nested_from_packed_tensor(batch.input_ids, seq_lens),
        cp_size=ps.cp_size,
        cp_rank=ps.cp_rank,
        cp_group=ps.cp_group,
        split_cp=False,
        labels=_nested_from_packed_tensor(batch.labels, seq_lens),
        roll_labels=True,
        roll_loss_mask=True,
        loss_mask=_nested_from_packed_tensor(batch.loss_mask, seq_lens),
    )
    kwargs: dict[str, Any] = {
        "input_ids": packed.input_ids,
        "labels": packed.labels,
        "loss_mask": packed.loss_mask,
        "position_ids": packed.position_ids,
        "packed_seq_params": packed.packed_seq_params,
        "enable_mtp": False,
    }
    add_loss_context_kwargs(kwargs)
    _prepare_packed_contiguous_cp_kwargs(model, kwargs)
    kwargs.pop("packed_seq_params", None)
    return {key: value for key, value in kwargs.items() if key in _MODEL_FORWARD_KEYS}


def _base_model_forward_kwargs(batch: PackedBatch):
    kwargs: dict[str, Any] = {"input_ids": _as_batch_row(batch.input_ids)}
    if batch.labels is not None:
        kwargs["labels"] = _as_batch_row(batch.labels)
    if batch.loss_mask is not None:
        kwargs["loss_mask"] = _as_batch_row(batch.loss_mask)
    add_loss_context_kwargs(kwargs)
    position_ids = _normalize_ds4_position_ids(getattr(batch, "position_ids", None))
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return kwargs


def _prepare_packed_contiguous_cp_kwargs(model, kwargs):
    ps = parallel_state_from_model(model) or ParallelState()
    if ps.cp_size <= 1:
        return kwargs
    for key in ("input_ids", "labels", "loss_mask", "position_ids"):
        tensor = kwargs.get(key)
        if tensor is not None:
            kwargs[key] = contiguous_slice_for_cp(tensor, ps.cp_rank, ps.cp_size, seq_dim=1)
    return kwargs


def _prepare_contiguous_cp_kwargs(model, kwargs):
    ps = parallel_state_from_model(model) or ParallelState()
    local_seq_len = _infer_cp_local_seq_len(
        input_ids=kwargs["input_ids"],
        position_ids=kwargs.get("position_ids"),
        cp_size=ps.cp_size,
    )
    kwargs["input_ids"] = local_sequence_tensor_for_cp(
        kwargs["input_ids"],
        local_seq_len=local_seq_len,
        cp_rank=ps.cp_rank,
        cp_size=ps.cp_size,
        name="input_ids",
    )
    if kwargs.get("position_ids") is None:
        full_seq_len = local_seq_len * ps.cp_size
        position_ids = contiguous_position_ids_for_cp(
            full_seq_len,
            cp_rank=ps.cp_rank,
            cp_size=ps.cp_size,
            device=kwargs["input_ids"].device,
        ).expand(kwargs["input_ids"].size(0), -1)
    else:
        position_ids = local_position_ids_for_cp(
            kwargs["position_ids"],
            batch=kwargs["input_ids"].size(0),
            local_seq_len=kwargs["input_ids"].size(1),
            cp_rank=ps.cp_rank,
            cp_size=ps.cp_size,
        )
    kwargs["position_ids"] = position_ids
    for key in ("labels", "loss_mask"):
        if kwargs.get(key) is not None:
            kwargs[key] = local_sequence_tensor_for_cp(
                kwargs[key],
                local_seq_len=kwargs["input_ids"].size(1),
                cp_rank=ps.cp_rank,
                cp_size=ps.cp_size,
                name=key,
            )
    return kwargs


def _prepare_model_forward_kwargs(model, batch: PackedBatch):
    # THD-packed inputs (1-D values, or a single padded [1, S] row) carry their own
    # cu_seqlens and go through the packed builder. A dense multi-row [B, S] batch is
    # split per row under contiguous CP, where contiguous_position_ids_for_cp rebuilds
    # the per-rank global position ids.
    input_ids = batch.input_ids
    is_thd_packed = input_ids.dim() == 1 or (input_ids.dim() == 2 and input_ids.size(0) == 1)
    if is_thd_packed:
        return _prepare_packed_batch_kwargs(model, batch)
    kwargs = _base_model_forward_kwargs(batch)
    return _prepare_contiguous_cp_kwargs(model, kwargs)


def _forward_step(model: nn.Module, batch: PackedBatch) -> dict:
    return model(**_prepare_model_forward_kwargs(model, batch))


def unpack_forward_output(model: nn.Module, batch: PackedBatch, output) -> Any:
    # DeepSeek-V4 packs each sequence to the (zigzag) TE alignment but slices CP
    # contiguously for the fused DSA indexer, so reconstruct contiguously.
    ps = parallel_state_from_model(model) or ParallelState()
    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
    )
    return unpack_thd_to_nested(output, meta, contiguous=True)


def _apply_mtp_config(model_cfg: DeepseekV4Config, impl_cfg: ImplConfig) -> None:
    override = impl_cfg.num_nextn_predict_layers
    if override is None:
        override = impl_cfg.mtp_num_layers
    if override is not None:
        if override < 0:
            raise ValueError(f"DeepSeek V4 MTP layer count must be >=0, got {override}.")
        model_cfg.num_nextn_predict_layers = int(override)
    if impl_cfg.mtp_enable:
        if model_cfg.num_nextn_predict_layers <= 0:
            raise ValueError("mtp_enable=True but DeepSeek V4 config has no MTP layers.")
        model_cfg.mtp_loss_scaling_factor = impl_cfg.mtp_loss_scaling_factor
    else:
        model_cfg.num_nextn_predict_layers = 0


def _make_aux_loss_hook():
    """Per-step hook that syncs the MTP auxiliary-loss backward scale to the main
    loss scale (DP size / gradient accumulation), mirroring the sibling protocols
    (kimi_k2 / glm5 / qwen3_5 / qwen3_moe).

    DS4 only injects an MTP auxiliary loss: its MoE router is aux-loss-free
    (``SigmoidTopKRouter(..., compute_aux_loss=False)``) and its CSA indexer runs
    with ``sparse_loss=False``, so -- unlike GLM-5, which also scales the MoE-aux
    and DSA-indexer losses -- only ``MTPLossAutoScaler`` needs scaling here.
    Without this hook the injected MTP gradient keeps ``MTPLossAutoScaler``'s
    class-default scale of 1.0 and is mis-weighted relative to the main loss.
    """
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler

    def hook(scale: torch.Tensor) -> None:
        MTPLossAutoScaler.set_loss_scale(scale)

    return hook


def _optimizer_backend_name(optimizer: Any) -> str | None:
    if isinstance(optimizer, dict) or isinstance(optimizer, OptimizerConfig):
        return "dist_opt"
    return optimizer


def _configure_attention_backend(chunks: list[nn.Module], *, backend: str | None) -> None:
    backend_name = backend or "torch"
    for chunk in chunks:
        for module in chunk.modules():
            if hasattr(module, "attention_backend"):
                module.attention_backend = backend_name


def _iter_transformer_units(chunk: nn.Module) -> list[nn.Module]:
    model = getattr(chunk, "model", chunk)
    if model is None:
        return []
    layers = list(getattr(model, "layers", {}).values())
    mtp_layers = list(getattr(model, "mtp", []))
    return [*layers, *mtp_layers]


def _validate_parallel_scope(p: ParallelConfig) -> None:
    """DS4 CSA attention is not tensor-parallel-capable (documented TP=1 case).

    PP / VPP / EP / CP are inherited from the Kimi skeleton and work; only
    TP>1 / ETP>1 are unsupported.  Mirrors GLM-5's gate.
    """
    etp = 1 if p.etp is None else p.etp
    if p.tp > 1:
        raise NotImplementedError(
            "DeepSeek V4 native CSA attention does not support tensor parallelism; "
            f"got tp={p.tp}. Use tp=1 (PP/VPP/EP/CP are supported)."
        )
    if etp > 1:
        raise NotImplementedError(
            "DeepSeek V4 native CSA attention does not support expert tensor parallelism; "
            f"got etp={etp}. Use etp=1 (EP is supported)."
        )


def build_model(model_cfg: DeepseekV4Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Model

    p = impl_cfg.parallel
    _validate_parallel_scope(p)
    _apply_mtp_config(model_cfg, impl_cfg)
    mtp_enable = bool(impl_cfg.mtp_enable) and model_cfg.num_nextn_predict_layers > 0
    mtp_enable_train = mtp_enable and bool(impl_cfg.mtp_enable_train)
    ps = init_parallel(impl_cfg.parallel)
    vpp = None if p.vpp == 1 else p.vpp
    train_cfg = SimpleNamespace(
        tp=ps.tp_size,
        ep=ps.ep_size,
        etp=ps.etp_size,
        pp=ps.pp_size,
        cp=ps.cp_size,
        vpp=vpp,
        fp8=False,
        use_deepep=impl_cfg.use_deepep,
    )

    def _chunk(i: int | None = None):
        return (
            DeepseekV4Model(
                model_cfg,
                train_cfg,
                ps,
                vpp_chunk_id=i,
                use_deepep=impl_cfg.use_deepep,
                use_thd=impl_cfg.use_thd,
                hf_path=impl_cfg.hf_path,
                attention_backend_override=impl_cfg.attention_backend_override,
                mtp_enable=mtp_enable,
                mtp_enable_train=mtp_enable_train,
                mtp_detach_encoder=impl_cfg.mtp_detach_encoder,
            )
            .to(torch.bfloat16)
            .cuda()
        )

    chunks = [_chunk(i) for i in range(vpp)] if vpp is not None else [_chunk()]
    _configure_attention_backend(chunks, backend=impl_cfg.attention_backend_override)

    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    if recompute_spec:
        for chunk in chunks:
            apply_recompute(_iter_transformer_units(chunk), recompute_spec, MODULE_MAP)

    if impl_cfg.offload:
        from megatron.lite.primitive.recompute import apply_offload

        for chunk in chunks:
            apply_offload(_iter_transformer_units(chunk), impl_cfg.offload, MODULE_MAP)

    optimizer = None
    finalize_grads = None
    post_model_load_hook = None
    optimizer_backend = "none"
    optimizer_name = _optimizer_backend_name(impl_cfg.optimizer)
    if optimizer_name == "dist_opt":
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_dist_opt_training_optimizer,
        )
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        optimizer, finalize_grads = build_dist_opt_training_optimizer(
            chunks,
            model_cfg=model_cfg,
            impl_cfg=impl_cfg,
            ps=ps,
            model_name="deepseek_v4",
            is_expert=is_expert_param,
            deterministic=impl_cfg.deterministic,
        )
        attach_model_sharded_state_dict(
            chunks, ps, get_placements=PLACEMENT_FN, is_expert=is_expert_param
        )
        register_training_hooks(chunks, optimizer)
        optimizer_backend = "dist_opt"
    elif optimizer_name == "fsdp2":
        optimizer_backend = "fsdp2"

        def _post_model_load_hook():
            from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Layer
            from megatron.lite.primitive.optimizers.fsdp2 import build_fsdp2_training_optimizer

            return {
                "optimizer": build_fsdp2_training_optimizer(
                    chunks,
                    impl_cfg.optimizer_config,
                    ps,
                    unit_modules=(DeepseekV4Layer,),
                    expert_classifier=is_expert_param,
                    deterministic=impl_cfg.deterministic,
                    vpp=impl_cfg.parallel.vpp,
                    leaf_module_names=(),
                    use_fp32_shards=False,
                )
            }

        post_model_load_hook = _post_model_load_hook
    elif optimizer_name is None:
        optimizer_backend = "none"
    else:
        raise ValueError(f"Unknown DeepSeek V4 lite optimizer: {impl_cfg.optimizer!r}.")

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


def load_hf_weights(
    chunk: nn.Module, hf_path: str, model_cfg: DeepseekV4Config, ps: ParallelState
) -> None:
    if not hf_path:
        return
    _load_hf_weights_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(
    chunks: list[nn.Module], model_cfg: DeepseekV4Config, ps: ParallelState, **kwargs
):
    yield from _export_hf_weights_impl(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(
    chunks: list[nn.Module], path: str, model_cfg: DeepseekV4Config, ps: ParallelState, **kwargs
) -> None:
    _save_hf_weights_impl(chunks, path, model_cfg, ps, **kwargs)


def vocab_size(model_cfg: DeepseekV4Config) -> int | None:
    return model_cfg.vocab_size
