"""Training-capable protocol for the DeepSeek-V4 vLLM implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.checkpoint import (
    export_hf_weights,
    invalidate_bound_source_scales,
    load_hf_weights,
    save_hf_weights,
)
from megatron.lite.model.deepseek_v4.lite.protocol import (
    ImplConfig as LiteImplConfig,
    MODULE_MAP,
    build_model_config,
    build_training_backend,
    is_expert_param,
    pack_packed_batch,
    pack_r3_replay_mask,
    pack_routed_experts,
    unpack_forward_output,
)
from megatron.lite.model.deepseek_v4.vllm.runtime_metadata import (
    DS4SparseIndexerCompressorMetadataAdapter,
    build_moe_metadata,
    ds4_vllm_forward_context,
    initialize_ds4_vllm_batch_invariance,
)
from megatron.lite.model.protocol_utils import (
    add_loss_context_kwargs,
    router_replay_roots as router_replay_roots,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.parallel.cp import contiguous_slice_for_cp
from megatron.lite.primitive.parallel.thd import parallel_state_from_model
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec


@dataclass(frozen=True)
class ImplConfig(LiteImplConfig):
    optimizer: str | None = None
    mtp_enable: bool = False
    dsa_indexer_loss_coeff: float = 0.0
    max_tokens_per_rank: int = 8192


def _validate_contract(model_cfg: DeepseekV4Config, impl_cfg: ImplConfig) -> None:
    if impl_cfg.dsa_indexer_loss_coeff < 0.0:
        raise ValueError("dsa_indexer_loss_coeff must be >= 0")
    if impl_cfg.max_tokens_per_rank <= 0:
        raise ValueError("max_tokens_per_rank must be positive")
    if not 0 <= model_cfg.num_hash_layers <= model_cfg.num_hidden_layers:
        raise ValueError(
            "num_hash_layers is a zero-based prefix length and must be between "
            f"0 and num_hidden_layers; got {model_cfg.num_hash_layers} for "
            f"{model_cfg.num_hidden_layers} layers."
        )
    parallel = impl_cfg.parallel
    unsupported = {
        name: size
        for name, size in (
            ("tp", parallel.tp),
            ("etp", parallel.etp or 1),
            ("vpp", parallel.vpp),
        )
        if size > 1
    }
    if unsupported:
        values = ", ".join(f"{name}={size}" for name, size in unsupported.items())
        raise NotImplementedError(
            f"DeepSeek V4 vLLM keeps TP/ETP/VPP at one; got {values}."
        )
    if parallel.ep <= 0 or model_cfg.n_routed_experts % parallel.ep:
        raise ValueError(
            f"EP={parallel.ep} must divide {model_cfg.n_routed_experts} routed experts."
        )
    if not impl_cfg.use_deepep:
        raise NotImplementedError(
            "DeepSeek V4 vLLM requires normal DeepEP training transport."
        )
    if impl_cfg.mtp_enable:
        raise NotImplementedError("DeepSeek V4 vLLM skeleton does not support MTP yet.")
    enabled = [
        name
        for name, value in (
            ("offload", impl_cfg.offload),
            ("use_thd", impl_cfg.use_thd),
            (
                "attention_backend_override",
                impl_cfg.attention_backend_override not in (None, "flash"),
            ),
            ("qat", impl_cfg.qat),
        )
        if value
    ]
    if enabled:
        raise NotImplementedError(
            "DeepSeek V4 vLLM skeleton does not install training/runtime features: "
            + ", ".join(enabled)
        )
    if len(model_cfg.compress_ratios) < model_cfg.num_hidden_layers:
        raise ValueError(
            "compress_ratios must cover every decoder layer; got "
            f"{len(model_cfg.compress_ratios)} entries for "
            f"{model_cfg.num_hidden_layers} layers."
        )
    unsupported_ratios = {
        layer: ratio
        for layer, ratio in enumerate(
            model_cfg.compress_ratios[: model_cfg.num_hidden_layers]
        )
        if max(1, ratio) not in (1, 4, 128)
    }
    if unsupported_ratios:
        raise ValueError(
            "DeepSeek V4 vLLM metadata compress_ratios supports only 1, 4, and "
            f"128; got {unsupported_ratios}."
        )
    if max(1, model_cfg.compress_ratios[0]) != 1:
        raise ValueError(
            "decoder layer 0 must use the official SWA-only ratio-1 metadata "
            f"contract; got compress_ratio={model_cfg.compress_ratios[0]}."
        )


def _forward_step(
    model: nn.Module,
    batch,
    *,
    attention_metadata=None,
    moe_metadata=None,
    forward_inputs: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    kwargs = {
        "input_ids": batch.input_ids,
        "position_ids": getattr(batch, "position_ids", None),
        "attention_metadata": (
            getattr(batch, "attention_metadata", None)
            if attention_metadata is None
            else attention_metadata
        ),
        "moe_metadata": (
            getattr(batch, "moe_metadata", None)
            if moe_metadata is None
            else moe_metadata
        ),
        "labels": getattr(batch, "labels", None),
        "loss_mask": getattr(batch, "loss_mask", None),
        "temperature": getattr(batch, "temperature", 1.0),
    }
    if forward_inputs is not None:
        kwargs.update(forward_inputs)
    add_loss_context_kwargs(kwargs)
    return model(**kwargs)


def _prepare_cp_forward_inputs(
    model: nn.Module, batch
) -> tuple[dict[str, Any], list[int], Any | None]:
    """Pad then slice packed rows using DS4 lite's contiguous CP layout."""

    ps = parallel_state_from_model(model)
    packed = pack_packed_batch(model, batch, batch.seq_lens)

    def local(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if ps is not None and ps.cp_size > 1:
            tensor = contiguous_slice_for_cp(
                tensor, ps.cp_rank, ps.cp_size, seq_dim=1
            )
        return tensor.reshape(-1).contiguous()

    inputs = {
        "input_ids": local(packed.input_ids),
        "position_ids": local(packed.position_ids),
        "labels": local(packed.labels),
        "loss_mask": local(packed.loss_mask),
    }
    return (
        inputs,
        [int(value) for value in packed.padded_lengths.detach().cpu().tolist()],
        packed.packed_seq_params,
    )


def build_model(model_cfg: DeepseekV4Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    _validate_contract(model_cfg, impl_cfg)
    initialize_ds4_vllm_batch_invariance()
    from megatron.lite.model.deepseek_v4.vllm.model import (
        DeepseekV4Layer,
        DeepseekV4Model,
    )

    parallel_state = init_parallel(impl_cfg.parallel)
    model = DeepseekV4Model(
        model_cfg,
        ps=parallel_state,
        use_deepep=impl_cfg.use_deepep,
        indexer_loss_coeff=impl_cfg.dsa_indexer_loss_coeff,
    )
    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    if recompute_spec:
        apply_recompute(
            list(model.layers.values()),
            recompute_spec,
            MODULE_MAP,
        )
    # The runtime loads replicated HF tensors through NCCL before the deferred
    # FSDP2 wrap.  Keep that lifecycle identical to the Lite protocol: masters
    # must already live on the rank-local CUDA device during checkpoint load.
    if torch.cuda.is_available():
        model = model.cuda()
        # The production vLLM GPUWorker creates this process-global manager
        # before constructing its model runner.  The mLite runtime directly
        # reuses vLLM sparse-indexer and attention kernels, so it must honor
        # the same lifecycle even though it does not instantiate GPUWorker.
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            is_workspace_manager_initialized,
        )

        if not is_workspace_manager_initialized():
            init_workspace_manager(next(model.parameters()).device, num_ubatches=1)
    selected_layers = tuple(model.layer_indices)
    attention_builders = None
    moe_metadata = None

    def ensure_runtime_assets():
        nonlocal attention_builders, moe_metadata
        if attention_builders is not None and moe_metadata is not None:
            return attention_builders, moe_metadata
        device = next(model.parameters()).device
        attention_builders = {
            layer_idx: DS4SparseIndexerCompressorMetadataAdapter.from_hf(
                impl_cfg.hf_path,
                model_cfg,
                layer_idx=layer_idx,
                device=device,
            )
            for layer_idx in selected_layers
        }
        moe_metadata = {
            layer_idx: build_moe_metadata(model_cfg, device)
            for layer_idx in selected_layers
        }
        return attention_builders, moe_metadata
    from vllm.config import VllmConfig

    vllm_config = VllmConfig()

    def forward_step(model: nn.Module, batch) -> dict[str, torch.Tensor]:
        attention_metadata = getattr(batch, "attention_metadata", None)
        moe_metadata = getattr(batch, "moe_metadata", None)
        if (attention_metadata is None) != (moe_metadata is None):
            raise ValueError(
                "caller-owned attention_metadata and moe_metadata must be "
                "provided together"
            )
        current_attention_builders = None
        current_moe_metadata = None
        if attention_metadata is None or moe_metadata is None:
            current_attention_builders, current_moe_metadata = ensure_runtime_assets()
        seq_lens = getattr(batch, "seq_lens", None)
        if seq_lens is None:
            if parallel_state.cp_size > 1:
                raise ValueError("DeepSeek V4 vLLM CP requires batch.seq_lens")
            forward_inputs = {}
            token_counts = [int(batch.input_ids.numel())]
            cp_packed_seq_params = None
        else:
            forward_inputs, token_counts, cp_packed_seq_params = _prepare_cp_forward_inputs(
                model, batch
            )
        local_tokens = sum(token_counts) // parallel_state.cp_size
        if local_tokens > impl_cfg.max_tokens_per_rank:
            raise ValueError(
                f"CP-local packed batch has {local_tokens} tokens, exceeding "
                f"max_tokens_per_rank={impl_cfg.max_tokens_per_rank}"
            )
        if attention_metadata is None:
            assert current_attention_builders is not None
            if parallel_state.cp_size > 1:
                from megatron.lite.model.deepseek_v4.vllm.runtime_metadata import (
                    build_native_cp_attention_metadata,
                )

                attention_metadata = {
                    layer_idx: build_native_cp_attention_metadata(
                        model_cfg,
                        layer_idx=layer_idx,
                        cos_sin_cache=current_attention_builders[layer_idx].cos_sin_cache,
                        local_positions=forward_inputs["position_ids"],
                        packed_seq_params=cp_packed_seq_params,
                    )
                    for layer_idx in selected_layers
                }
            else:
                attention_metadata = {
                    layer_idx: current_attention_builders[
                        layer_idx
                    ].build_prefill_batch(token_counts)
                    for layer_idx in selected_layers
                }
        if moe_metadata is None:
            assert current_moe_metadata is not None
            moe_metadata = current_moe_metadata
        if parallel_state.cp_size > 1:
            if cp_packed_seq_params is None:
                raise RuntimeError("DeepSeek V4 vLLM CP requires packed sequence metadata")
            local_positions = forward_inputs.get("position_ids")
            if local_positions is None:
                raise RuntimeError("DeepSeek V4 vLLM CP requires local position ids")
            values = (
                attention_metadata.values()
                if isinstance(attention_metadata, dict)
                else (attention_metadata,)
            )
            for layer_metadata in values:
                layer_metadata.cp_packed_seq_params = cp_packed_seq_params
                layer_metadata.cp_positions = local_positions
        with ds4_vllm_forward_context(
            batch,
            parallel_state,
            vllm_config=vllm_config,
        ):
            return _forward_step(
                model,
                batch,
                attention_metadata=attention_metadata,
                moe_metadata=moe_metadata,
                forward_inputs=forward_inputs,
            )

    # ``build_training_backend`` replaces this list in-place with Megatron DDP
    # wrappers for dist-opt (Lite follows the same ownership contract).  Keep
    # and return that mutated list; returning the raw model would bypass DDP's
    # grad-accumulation hooks, leaving ``main_grad`` zero while ``.grad`` fills.
    chunks = [model]
    optimizer, finalize_grads, post_model_load_hook, optimizer_backend = (
        build_training_backend(
            chunks,
            model_cfg,
            impl_cfg,
            parallel_state,
            unit_modules=(DeepseekV4Layer,),
            use_fp32_shards=True,
            cast_forward_inputs=False,
        )
    )

    extras = {
        "model_cfg": model_cfg,
        "optimizer_backend": optimizer_backend,
    }
    if torch.cuda.is_available():
        from vllm.v1.worker.workspace import reset_workspace_manager

        extras["close_hook"] = reset_workspace_manager
    extras["post_optimizer_step_hook"] = lambda: invalidate_bound_source_scales(
        model
    )
    if post_model_load_hook is not None:
        extras["post_model_load_hook"] = post_model_load_hook

    return ModelBundle(
        chunks=chunks,
        parallel_state=parallel_state,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=forward_step,
        extras=extras,
    )


def vocab_size(model_cfg: DeepseekV4Config) -> int | None:
    return model_cfg.vocab_size


__all__ = [
    "ImplConfig",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "is_expert_param",
    "load_hf_weights",
    "save_hf_weights",
    "unpack_forward_output",
    "vocab_size",
]
