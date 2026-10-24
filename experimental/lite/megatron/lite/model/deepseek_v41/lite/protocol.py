# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""BF16 DS4.1 training protocol over native Core components."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import SimpleNamespace

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite.components.config import core_config
from megatron.lite.model.protocol_utils import nested_from_packed
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.parallel.thd import (
    pack_nested_thd,
    parallel_state_from_model,
    thd_pack_meta,
    unpack_thd_to_nested,
)
from megatron.lite.runtime.contracts import ParallelConfig
from megatron.lite.runtime.contracts.loss import get_loss_context


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    hf_path: str = ""
    initialize_weights: bool = True
    prefetch_weights: bool = True
    token_map: object = None
    trainable_engram: bool = False
    shard_engram: bool = True
    recompute: object = None
    pipeline_split_layer: object = None
    optimizer: str | None = "adam"
    optimizer_config: object = None
    attention_backend_override: str | None = None
    deterministic_mode: bool = False
    router_aux_loss_coef: float = 0.0
    quantized: bool = False
    w4a8_experts: bool = False
    # VERL connector knobs (verl_mlite always injects them).
    use_thd: bool = True
    cross_entropy_fusion: bool = False
    mtp_enable: bool = False
    mtp_enable_train: bool = False
    mtp_detach_encoder: bool = False
    mtp_loss_scaling_factor: float = 0.1
    deterministic: bool = False
    # Generic MLite QAT schema; DS4.1 fake-quant is part of the deployment forward.
    qat: object = None

    def __post_init__(self):
        if self.deterministic:
            object.__setattr__(self, "deterministic_mode", True)


_CONNECTOR_REQUIRED = {
    'use_thd': True,
    'cross_entropy_fusion': False,
    'mtp_enable': False,
    'mtp_enable_train': False,
}


def validate_impl_cfg_keys(values):
    extra = set(values) - {f.name for f in fields(ImplConfig)}
    if extra:
        raise ValueError(f"Unsupported DS4.1 implementation options: {sorted(extra)}")
    qat = values.get("qat") or {}
    if qat.get("enabled") or qat.get("enable"):
        raise ValueError("DS4.1 applies deployment fake-quant itself; MLite QAT must stay off")
    for key, required in _CONNECTOR_REQUIRED.items():
        if key in values and bool(values[key]) != required:
            raise ValueError(f"DS4.1 requires {key}={required}, got {values[key]!r}")


def build_model_config(source, **overrides):
    cfg = (
        DeepseekV41Config._from_hf_dict(source)
        if isinstance(source, dict)
        else DeepseekV41Config.from_hf(str(source))
    )
    for name, value in overrides.items():
        if not hasattr(cfg, name):
            raise ValueError(f"Unknown DS4.1 release field: {name}")
        setattr(cfg, name, value)
    return cfg


def _validate(model_cfg, impl):
    if (
        impl.quantized
        or impl.w4a8_experts
        or impl.dtype not in (torch.bfloat16, "bfloat16", "bf16")
    ):
        raise ValueError("DS4.1 standard training supports BF16 execution only")
    if impl.trainable_engram:
        raise ValueError("Trainable Engram is not supported; tables must be frozen")
    if impl.optimizer not in (None, "adam", "dist_opt"):
        raise ValueError(f"Unsupported DS4.1 optimizer: {impl.optimizer}")
    options = impl.optimizer_config
    options = dict(options) if isinstance(options, Mapping) else vars(options) if options else {}
    if options.get("optimizer", "adam") not in ("adam", "dist_opt"):
        raise ValueError("DS4.1 supports only the native Adam distributed optimizer")
    if impl.parallel.vpp != 1 or impl.parallel.tp != 1 or impl.parallel.etp not in (None, 1):
        raise ValueError("DS4.1 supports TP=ETP=VPP=1")
    if impl.router_aux_loss_coef not in (None, 0):
        raise ValueError("DS4.1 standard post-training has no auxiliary indexer/router loss")
    modes = impl.recompute.split(",") if isinstance(impl.recompute, str) else impl.recompute or ()
    if not set(modes) <= {"full", "attn", "moe"}:
        raise ValueError("DS4.1 recomputation supports only full, attn and moe")
    if impl.pipeline_split_layer is not None:
        cuts = impl.pipeline_split_layer
        cuts = [cuts] if isinstance(cuts, int) else list(cuts)
        bounds = [0, *cuts, model_cfg.num_hidden_layers]
        if len(cuts) != impl.parallel.pp - 1 or any(a >= b for a, b in zip(bounds, bounds[1:])):
            raise ValueError("DS4.1 pipeline cuts must define nonempty ordered stages")
        rows = [["decoder"] * (b - a) for a, b in zip(bounds, bounds[1:])]
        rows[0].insert(0, "embedding")
        rows[-1].append("loss")
        impl.parallel.pp_layout = rows
    core_config(
        model_cfg, impl.parallel, initialize=False, deterministic_mode=impl.deterministic_mode
    )


def _optimizer(chunks, cfg, impl, ps):
    from megatron.lite.primitive.optimizers.megatron_wrap import (
        build_dist_opt_stack,
        finalize_dist_opt_grads,
    )

    values = impl.optimizer_config
    values = dict(values) if isinstance(values, Mapping) else dict(vars(values)) if values else {}
    if values.pop("segmented_host", False):
        raise ValueError("DS4.1 native Core optimizer does not support segmented_host")
    if values.get("optimizer", "adam") not in ("adam", "dist_opt"):
        raise ValueError("DS4.1 supports only the native Adam distributed optimizer")
    values["optimizer"] = "adam"
    values.setdefault("lr", 1e-5)
    values.setdefault("weight_decay", 0.0)
    values.setdefault("clip_grad", 1.0)
    engine = SimpleNamespace(
        parallel=impl.parallel,
        optimizer=SimpleNamespace(**values),
        deterministic=False,
        ddp_config_overrides={"overlap_grad_reduce": False},
    )
    chunks[:], optimizer = build_dist_opt_stack(chunks, model_cfg=cfg, engine_cfg=engine, ps=ps)
    # Keep the existing checkpoint/consumer convention without adding another optimizer.
    optimizer.optimizers = [
        getattr(o, "optimizer", o) for o in getattr(optimizer, "chained_optimizers", [optimizer])
    ]
    return optimizer, lambda: finalize_dist_opt_grads(chunks, optimizer)


def build_model(model_cfg, *, impl_cfg, _model_class=None):
    _validate(model_cfg, impl_cfg)
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    ps = init_parallel(impl_cfg.parallel)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(
        torch.initial_seed(),
        tp_rank=ps.tp_rank,
        ep_rank=ps.ep_rank,
        etp_rank=ps.etp_rank,
        gtp_remat_rank=0,
        egtp_remat_rank=0,
        gtp_remat_world_size=1,
        egtp_remat_world_size=1,
    )
    token_map = impl_cfg.token_map
    if token_map is None and impl_cfg.hf_path:
        from pathlib import Path

        path = Path(impl_cfg.hf_path) / "token_map.pt"
        if path.is_file():
            token_map = torch.load(path, map_location="cpu", weights_only=True)
    constructor = DeepseekV41Model if _model_class is None else _model_class
    reader = None
    if impl_cfg.prefetch_weights and not impl_cfg.initialize_weights and impl_cfg.hf_path:
        from megatron.lite.primitive.ckpt.row_reader import RowSliceReader

        reader = RowSliceReader(impl_cfg.hf_path, device=impl_cfg.device)
    try:
        chunks = [constructor(model_cfg, impl_cfg, ps, token_map=token_map, load_reader=reader)]
    except BaseException:
        if reader is not None:
            reader.close()
        raise
    optimizer, finalize = (None, None)
    if impl_cfg.optimizer is not None:
        optimizer, finalize = _optimizer(chunks, model_cfg, impl_cfg, ps)
        from megatron.lite.primitive.ckpt.row_state import attach_checkpoint_metadata

        attach_checkpoint_metadata(chunks, ps)
    return ModelBundle(
        chunks,
        ps,
        optimizer,
        finalize,
        forward_step,
        extras={"model_cfg": model_cfg, "optimizer_backend": "dist_opt" if optimizer else "none"},
    )


def _original_rows(physical, lengths, padded):
    """Map physical packed rows to the unpadded PackedBatch row; -1 for padding."""
    starts = torch.cumsum(lengths, 0) - lengths
    document = torch.searchsorted(padded[1:].to(physical), physical, right=True)
    offset = physical - padded[document].to(physical)
    valid = offset < lengths[document]
    return torch.where(valid, starts[document] + offset, -1), valid


def forward_step(model, batch):
    ps = parallel_state_from_model(model)
    lens = batch.sizes().to(device=batch.input_ids.device)
    nested = lambda x: None if x is None else nested_from_packed(x, lens)
    packed = pack_nested_thd(
        nested(batch.input_ids),
        cp_size=ps.cp_size,
        cp_rank=ps.cp_rank,
        cp_group=ps.cp_group,
        split_cp=False,
        labels=nested(batch.labels),
        roll_labels=batch.labels is not None,
        loss_mask=nested(
            batch.loss_mask
            if batch.loss_mask is not None
            else torch.ones_like(batch.input_ids, dtype=torch.float32)
        ),
        roll_loss_mask=True,
    )
    original = packed.packed_seq_params
    names = (
        "qkv_format cu_seqlens_q cu_seqlens_kv cu_seqlens_q_padded "
        "cu_seqlens_kv_padded max_seqlen_q max_seqlen_kv"
    ).split()
    params = PackedSeqParams(
        **{name: getattr(original, name) for name in names},
        cp_partition_mode="contiguous",
        cp_group=ps.cp_group,
        local_cp_size=ps.cp_size,
    )
    total = packed.input_ids.numel()
    local = total // ps.cp_size
    span = slice(ps.cp_rank * local, (ps.cp_rank + 1) * local)
    select = lambda x: None if x is None else x.reshape(-1)[span]
    context = get_loss_context()
    output = model(
        select(packed.input_ids),
        packed_seq_params=params,
        full_input_ids=packed.input_ids.reshape(-1),
        labels=select(packed.labels),
        loss_mask=select(packed.loss_mask),
        temperature=context.temperature if context else 1.0,
        calculate_entropy=context.calculate_entropy if context else False,
    )
    if context is not None and context.normalization_denominator is not None and "loss" in output:
        denominator = context.normalization_denominator
        if denominator <= 0:
            raise ValueError("DS4.1 loss normalization denominator must be positive")
        mask = select(packed.loss_mask)
        if mask is None:
            mask = torch.ones_like(output["log_probs"])
        output["loss"] = -(output["log_probs"] * mask).sum() / denominator
    if "rows" in output:
        # Report rows in the caller's unpadded coordinates; padding rows are invalid.
        output["rows"], valid = _original_rows(output["rows"], lens, original.cu_seqlens_q_padded)
        output["valid_mask"] = valid & output.get("valid_mask", valid)
    return output


def load_hf_weights(chunk, hf_path, model_cfg, ps):
    from megatron.lite.model.deepseek_v41.lite.checkpoint import load_hf_weights as load

    return load(chunk, hf_path, model_cfg, ps)


def export_hf_weights(chunks, model_cfg, ps, **kwargs):
    from megatron.lite.model.deepseek_v41.lite.export import export_hf_weights as export

    yield from export(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(chunks, path, model_cfg, ps, **kwargs):
    from megatron.lite.model.deepseek_v41.lite.export import save_hf_weights as save

    save(chunks, path, model_cfg, ps, **kwargs)


def unpack_forward_output(model, batch, output):
    """Reverse ``forward_step`` token outputs to jagged per-sequence form.

    ``forward_step`` pads each sequence to ``pack_nested_thd``'s alignment and
    gives each CP rank one contiguous slice of the packed row.
    """
    ps = parallel_state_from_model(model)
    meta = thd_pack_meta(
        batch.sizes(),
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
    )
    return unpack_thd_to_nested(output, meta, contiguous=True)
