from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.model_executor.layers.batch_invariant import init_batch_invariance

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.checkpoint import (
    export_hf_weights as export_hf_weights,
    invalidate_bound_source_scales,
    load_hf_weights as load_hf_weights,
    save_hf_weights as save_hf_weights,
)
from megatron.lite.model.deepseek_v4.lite.protocol import (
    MODULE_MAP,
    _optimizer_backend_name,
    build_model_config as build_model_config,
    build_training_backend,
    is_expert_param as is_expert_param,
    pack_packed_batch,
    pack_r3_replay_mask as pack_r3_replay_mask,
    pack_routed_experts as pack_routed_experts,
    unpack_forward_output as unpack_forward_output,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.attention.metadata import (
    build_attention_metadata_builders,
)
from megatron.lite.model.protocol_utils import add_loss_context_kwargs
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.parallel.cp import contiguous_slice_for_cp
from megatron.lite.primitive.parallel.thd import parallel_state_from_model
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = None
    optimizer_config: OptimizerConfig | None = None
    hf_path: str = ""
    recompute: list[str] = field(default_factory=list)
    deterministic: bool = True
    dsa_indexer_loss_coeff: float = 0.0
    logprob_chunk_size: int = 8192
    cache_deployment_weights: bool | None = None


def _deployment_weight_cache_enabled(impl_cfg: ImplConfig) -> bool:
    if impl_cfg.cache_deployment_weights is not None:
        return impl_cfg.cache_deployment_weights
    return _optimizer_backend_name(impl_cfg.optimizer) == "dist_opt"


def _local_num_tokens(batch: Any) -> int:
    total_tokens = getattr(batch, "total_tokens", None)
    if total_tokens is not None:
        value = int(total_tokens)
    else:
        input_ids = getattr(batch, "input_ids", None)
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("DeepSeek V4 vLLM batches require tensor input_ids")
        value = int(input_ids.numel())
    if value <= 0:
        raise ValueError("DeepSeek V4 vLLM batches must contain at least one token")
    return value


@contextmanager
def _vllm_forward_context(batch: Any, parallel_state, vllm_config) -> Iterator[None]:
    from vllm.forward_context import (
        DPMetadata,
        create_forward_context,
        override_forward_context,
    )

    input_ids = getattr(batch, "input_ids", None)
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("DeepSeek V4 vLLM batches require tensor input_ids")
    if parallel_state.ep_group is None:
        raise RuntimeError("DeepSeek V4 vLLM requires an initialized EP group")
    local_tokens = torch.tensor(
        [_local_num_tokens(batch)], dtype=torch.int32, device=input_ids.device
    )
    gathered_tokens = [
        torch.empty_like(local_tokens) for _ in range(parallel_state.ep_size)
    ]
    dist.all_gather(gathered_tokens, local_tokens, group=parallel_state.ep_group)
    dp_metadata = DPMetadata(torch.cat(gathered_tokens).cpu())
    forward_context = create_forward_context(
        None, vllm_config, dp_metadata=dp_metadata
    )
    with override_forward_context(forward_context):
        yield


def _post_optimizer_step(model: nn.Module) -> None:
    invalidate_bound_source_scales(model)
    for module in model.modules():
        clear_cache = getattr(module, "clear_deployment_weight_cache", None)
        if clear_cache is not None:
            clear_cache()


def _validate_contract(model_cfg: DeepseekV4Config, impl_cfg: ImplConfig) -> None:
    if impl_cfg.dsa_indexer_loss_coeff < 0.0:
        raise ValueError("dsa_indexer_loss_coeff must be >= 0")
    if impl_cfg.logprob_chunk_size <= 0:
        raise ValueError("logprob_chunk_size must be positive")
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
) -> tuple[dict[str, Any], Any]:

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
    return inputs, packed.packed_seq_params


def build_model(model_cfg: DeepseekV4Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    _validate_contract(model_cfg, impl_cfg)
    init_batch_invariance()
    from megatron.lite.model.deepseek_v4.vllm.model import (
        DeepseekV4Layer,
        DeepseekV4Model,
    )

    parallel_state = init_parallel(impl_cfg.parallel)
    model = DeepseekV4Model(
        model_cfg,
        ps=parallel_state,
        indexer_loss_coeff=impl_cfg.dsa_indexer_loss_coeff,
        logprob_chunk_size=impl_cfg.logprob_chunk_size,
        cache_deployment_weights=_deployment_weight_cache_enabled(impl_cfg),
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
    from vllm.config import VllmConfig

    vllm_config = VllmConfig()

    def ensure_runtime_assets():
        nonlocal attention_builders
        if attention_builders is not None:
            return attention_builders
        device = next(model.parameters()).device
        attention_builders = build_attention_metadata_builders(
            impl_cfg.hf_path, model_cfg, selected_layers, device
        )
        return attention_builders

    def forward_step(model: nn.Module, batch) -> dict[str, torch.Tensor]:
        seq_lens = getattr(batch, "seq_lens", None)
        if seq_lens is None:
            raise ValueError("DeepSeek V4 vLLM requires packed batch.seq_lens")
        forward_inputs, packed_seq_params = _prepare_cp_forward_inputs(model, batch)
        current_attention_builders = ensure_runtime_assets()
        attention_metadata = {
            layer_idx: current_attention_builders[layer_idx].build(
                forward_inputs["position_ids"], packed_seq_params
            )
            for layer_idx in selected_layers
        }
        with _vllm_forward_context(batch, parallel_state, vllm_config):
            return _forward_step(
                model,
                batch,
                attention_metadata=attention_metadata,
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
    extras["post_optimizer_step_hook"] = lambda: _post_optimizer_step(model)
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
