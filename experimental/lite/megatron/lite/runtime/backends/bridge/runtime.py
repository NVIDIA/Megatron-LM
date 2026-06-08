"""Runtime backend backed by Megatron-Bridge."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from megatron.lite.primitive.optimizers.megatron_wrap import build_mc_optimizer_config
from megatron.lite.runtime.backends import Runtime as RuntimeBase
from megatron.lite.runtime.backends.bridge.config import BridgeConfig
from megatron.lite.runtime.contracts.data import Batch, ForwardResult, ModelOutputs
from megatron.lite.runtime.contracts.handle import ModelHandle
from megatron.lite.runtime.megatron_utils import (
    build_sharded_state_dict,
    is_mp_src_rank_with_outputs,
    load_model_to_gpu,
    load_optimizer,
    offload_model_to_cpu,
    offload_optimizer,
    register_training_hooks,
)

logger = logging.getLogger(__name__)


class _MpuParallelState:
    """Adapter exposing Megatron-Core mpu through ``ModelHandle`` properties."""

    def __init__(self, mpu):
        self._mpu = mpu

    @property
    def dp_rank(self) -> int:
        return self._mpu.get_data_parallel_rank()

    @property
    def dp_size(self) -> int:
        return self._mpu.get_data_parallel_world_size()

    @property
    def dp_group(self):
        return self._mpu.get_data_parallel_group()

    @property
    def tp_rank(self) -> int:
        return self._mpu.get_tensor_model_parallel_rank()

    @property
    def tp_size(self) -> int:
        return self._mpu.get_tensor_model_parallel_world_size()

    @property
    def pp_rank(self) -> int:
        return self._mpu.get_pipeline_model_parallel_rank()

    @property
    def pp_size(self) -> int:
        return self._mpu.get_pipeline_model_parallel_world_size()

    @property
    def cp_rank(self) -> int:
        return self._mpu.get_context_parallel_rank()

    @property
    def cp_size(self) -> int:
        return self._mpu.get_context_parallel_world_size()

    @property
    def cp_group(self):
        return self._mpu.get_context_parallel_group()


def _lower_transformer_overrides(cfg: BridgeConfig) -> dict[str, Any]:
    overrides = {"attention_backend": "flash"}
    overrides.update(cfg.override_transformer_config)
    return overrides


def _bridge_hf_config(bridge):
    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    if hf_pretrained is None:
        return None
    return getattr(hf_pretrained, "config", hf_pretrained)


def _lower_provider_value(key: str, value: Any) -> Any:
    if key == "attention_backend" and isinstance(value, str):
        from megatron.core.transformer.enums import AttnBackend

        return AttnBackend[value]
    return value


def _configure_provider(provider, cfg: BridgeConfig) -> None:
    from megatron.lite.primitive.deterministic import deterministic_requested

    p = cfg.parallel
    provider.tensor_model_parallel_size = p.tp
    provider.pipeline_model_parallel_size = p.pp
    provider.context_parallel_size = p.cp
    provider.expert_model_parallel_size = p.ep
    if p.etp is not None:
        provider.expert_tensor_parallel_size = p.etp
    if p.vpp > 1:
        provider.virtual_pipeline_model_parallel_size = p.vpp
    provider.sequence_parallel = p.tp > 1
    provider.bf16 = True
    provider.fp16 = False
    provider.deterministic_mode = deterministic_requested()
    if provider.deterministic_mode:
        os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")

    for key, value in _lower_transformer_overrides(cfg).items():
        setattr(provider, key, _lower_provider_value(key, value))


def _register_bridge_compat_aliases() -> None:
    """Register local Megatron-Bridge aliases for supported checkpoint variants."""
    from megatron.bridge.models.conversion import model_bridge
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GDNConv1dMapping,
        GatedMLPMapping,
        QKVMapping,
        ReplicatedMapping,
        RMSNorm2ZeroCenteredRMSNormMapping,
        merge_gdn_linear_weights,
        split_gdn_linear_weights,
    )
    from megatron.bridge.models.qwen.qwen3_next_bridge import Qwen3NextBridge
    from megatron.bridge.utils.common_utils import extract_expert_number_from_param
    from megatron.core.models.gpt.gpt_model import GPTModel

    class Qwen35SplitGDNMapping(AutoMapping):
        """Bridge mapping for Qwen3.5 split GDN in-proj weights."""

        def __init__(self, megatron_param: str, qkv: str, z: str, b: str, a: str):
            super().__init__(megatron_param=megatron_param, hf_param={"qkv": qkv, "z": z, "b": b, "a": a})
            self._tp_mapping = AutoMapping(megatron_param, megatron_param)

        def hf_to_megatron(self, hf_weights: dict[str, torch.Tensor], megatron_module) -> torch.Tensor:
            if self.tp_rank == 0:
                config = self._get_config(megatron_module)
                qkvz = torch.cat([hf_weights["qkv"], hf_weights["z"]], dim=0)
                ba = torch.cat([hf_weights["b"], hf_weights["a"]], dim=0)
                merged = merge_gdn_linear_weights(config, qkvz, ba, tp_size=self.tp_size)
            else:
                merged = None
            return self._tp_mapping.hf_to_megatron(merged, megatron_module)

        def megatron_to_hf(self, megatron_weights, megatron_module) -> dict[str, torch.Tensor]:
            if megatron_weights is not None:
                megatron_weights = self.maybe_dequantize(megatron_weights)

            if megatron_module is None:
                config = self.broadcast_obj_from_pp_rank(None)
            else:
                config = self._get_config(megatron_module)
                config = self.broadcast_obj_from_pp_rank(config)

            packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
            if not packed_dict:
                return {}

            packed = next(iter(packed_dict.values()))
            qkvz, ba = split_gdn_linear_weights(config, packed, tp_size=self.tp_size)
            qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
            v_dim = config.linear_value_head_dim * config.linear_num_value_heads
            qkv, z = qkvz.split([2 * qk_dim + v_dim, v_dim], dim=0)
            b, a = ba.chunk(2, dim=0)
            return {
                self.hf_param["qkv"]: qkv,
                self.hf_param["z"]: z,
                self.hf_param["b"]: b,
                self.hf_param["a"]: a,
            }

        def resolve(self, captures):
            megatron_param, hf_param = self._resolve_names(captures)
            return type(self)(
                megatron_param,
                hf_param["qkv"],
                hf_param["z"],
                hf_param["b"],
                hf_param["a"],
            )

    class Qwen35PackedExpertDownMapping(AutoMapping):
        """Bridge mapping for Qwen3.5 packed expert down-projection weights."""

        def __init__(self, megatron_param: str, hf_param: str, permute_dims=None):
            super().__init__(megatron_param=megatron_param, hf_param=hf_param, permute_dims=permute_dims)
            self.allow_hf_name_mismatch = True

        def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module) -> torch.Tensor:
            expert_number = extract_expert_number_from_param(self.megatron_param)
            expert_weight = hf_weights[expert_number].contiguous()
            return super().hf_to_megatron(expert_weight, megatron_module)

        def megatron_to_hf(self, megatron_weights, megatron_module) -> dict[str, torch.Tensor]:
            converted = super().megatron_to_hf(megatron_weights, megatron_module)
            return converted

        def _validate_patterns(self, *args, **kwargs):
            pass

    class Qwen35RouterMapping(AutoMapping):
        """Bridge mapping for Qwen3.5 router weights with bench expert truncation."""

        def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module) -> torch.Tensor:
            config = self._get_config(megatron_module)
            num_experts = getattr(config, "num_moe_experts", None)
            if num_experts is not None and hf_weights.shape[0] != num_experts:
                hf_weights = hf_weights[:num_experts].contiguous()
            return super().hf_to_megatron(hf_weights, megatron_module)

    class Qwen35PackedExpertGateUpMapping(AutoMapping):
        """Bridge mapping for Qwen3.5 packed expert gate/up projection weights."""

        def __init__(self, megatron_param: str, hf_param: str, permute_dims=None):
            super().__init__(megatron_param=megatron_param, hf_param=hf_param, permute_dims=permute_dims)
            self.allow_hf_name_mismatch = True
            GatedMLPMapping._validate_patterns = lambda *args, **kwargs: None
            self._gated_mapping = GatedMLPMapping(
                megatron_param=self.megatron_param,
                gate=f"{self.hf_param}.gate",
                up=f"{self.hf_param}.up",
            )

        def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module) -> torch.Tensor:
            expert_number = extract_expert_number_from_param(self.megatron_param)
            expert_weight = hf_weights[expert_number].contiguous()
            gate, up = torch.chunk(expert_weight, 2, dim=0)
            return self._gated_mapping.hf_to_megatron({"gate": gate, "up": up}, megatron_module)

        def megatron_to_hf(self, megatron_weights, megatron_module) -> dict[str, torch.Tensor]:
            converted = self._gated_mapping.megatron_to_hf(megatron_weights, megatron_module)
            if not converted:
                return {}

            fused = {}
            for name, tensor in converted.items():
                if not name.endswith(".gate"):
                    continue
                base_name = name[: -len(".gate")]
                up_tensor = converted.get(f"{base_name}.up")
                if up_tensor is None:
                    continue
                gate_tensor = tensor.contiguous()
                up_tensor = up_tensor.contiguous()
                fused[base_name] = torch.stack([gate_tensor, up_tensor], dim=0 if up_tensor.ndim == 2 else 1)
            return fused

        def _validate_patterns(self, *args, **kwargs):
            pass

    class Qwen35MoEBridge(Qwen3NextBridge):
        """Megatron-Bridge Qwen3-Next bridge adjusted for Qwen3.5 HF naming."""

        def _text_config(self, hf_pretrained):
            config = getattr(hf_pretrained, "config", hf_pretrained)
            text_config = getattr(config, "text_config", config)

            if getattr(text_config, "intermediate_size", None) is None:
                text_config.intermediate_size = 5120

            rope_parameters = getattr(text_config, "rope_parameters", None)
            if isinstance(rope_parameters, dict):
                rope_theta = rope_parameters.get("rope_theta")
                if rope_theta is not None:
                    text_config.rope_theta = rope_theta
                partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
                if partial_rotary_factor is not None:
                    text_config.partial_rotary_factor = partial_rotary_factor

            if not hasattr(text_config, "tie_word_embeddings") and hasattr(config, "tie_word_embeddings"):
                text_config.tie_word_embeddings = config.tie_word_embeddings

            return text_config

        def provider_bridge(self, hf_pretrained):
            text_config = self._text_config(hf_pretrained)
            shim = type("_Qwen35TextConfigShim", (), {"config": text_config})()
            provider = super().provider_bridge(shim)

            aux_loss = getattr(text_config, "router_aux_loss_coef", None)
            if aux_loss is not None:
                provider.moe_aux_loss_coeff = aux_loss

            return provider

        def mapping_registry(self):
            prefix = "model.language_model"
            param_mappings = {
                "embedding.word_embeddings.weight": f"{prefix}.embed_tokens.weight",
                "output_layer.weight": "lm_head.weight",
                "decoder.final_layernorm.weight": f"{prefix}.norm.weight",
                "decoder.layers.*.pre_mlp_layernorm.weight": f"{prefix}.layers.*.post_attention_layernorm.weight",
                "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                    f"{prefix}.layers.*.input_layernorm.weight"
                ),
                "decoder.layers.*.self_attention.q_layernorm.weight": f"{prefix}.layers.*.self_attn.q_norm.weight",
                "decoder.layers.*.self_attention.k_layernorm.weight": f"{prefix}.layers.*.self_attn.k_norm.weight",
                "decoder.layers.*.self_attention.linear_proj.weight": f"{prefix}.layers.*.self_attn.o_proj.weight",
                "decoder.layers.*.self_attention.in_proj.layer_norm_weight": (
                    f"{prefix}.layers.*.input_layernorm.weight"
                ),
                "decoder.layers.*.self_attention.out_proj.weight": f"{prefix}.layers.*.linear_attn.out_proj.weight",
                "decoder.layers.*.self_attention.A_log": f"{prefix}.layers.*.linear_attn.A_log",
                "decoder.layers.*.self_attention.dt_bias": f"{prefix}.layers.*.linear_attn.dt_bias",
            }

            mapping_list = [
                AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
                for megatron_param, hf_param in param_mappings.items()
            ]
            AutoMapping.register_module_type("SharedExpertMLP", "column")
            AutoMapping.register_module_type("GatedDeltaNet", "column")

            mapping_list.extend(
                [
                    QKVMapping(
                        megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                        q=f"{prefix}.layers.*.self_attn.q_proj.weight",
                        k=f"{prefix}.layers.*.self_attn.k_proj.weight",
                        v=f"{prefix}.layers.*.self_attn.v_proj.weight",
                    ),
                    GDNConv1dMapping(
                        megatron_param="decoder.layers.*.self_attention.conv1d.weight",
                        hf_param=f"{prefix}.layers.*.linear_attn.conv1d.weight",
                    ),
                    Qwen35SplitGDNMapping(
                        megatron_param="decoder.layers.*.self_attention.in_proj.weight",
                        qkv=f"{prefix}.layers.*.linear_attn.in_proj_qkv.weight",
                        z=f"{prefix}.layers.*.linear_attn.in_proj_z.weight",
                        b=f"{prefix}.layers.*.linear_attn.in_proj_b.weight",
                        a=f"{prefix}.layers.*.linear_attn.in_proj_a.weight",
                    ),
                    Qwen35RouterMapping(
                        megatron_param="decoder.layers.*.mlp.router.weight",
                        hf_param=f"{prefix}.layers.*.mlp.gate.weight",
                    ),
                    Qwen35PackedExpertGateUpMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        hf_param=f"{prefix}.layers.*.mlp.experts.gate_up_proj",
                    ),
                    Qwen35PackedExpertDownMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param=f"{prefix}.layers.*.mlp.experts.down_proj",
                    ),
                    GatedMLPMapping(
                        megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                        gate=f"{prefix}.layers.*.mlp.shared_expert.gate_proj.weight",
                        up=f"{prefix}.layers.*.mlp.shared_expert.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                        hf_param=f"{prefix}.layers.*.mlp.shared_expert.down_proj.weight",
                    ),
                    ReplicatedMapping(
                        megatron_param="decoder.layers.*.mlp.shared_experts.gate_weight",
                        hf_param=f"{prefix}.layers.*.mlp.shared_expert_gate.weight",
                    ),
                    RMSNorm2ZeroCenteredRMSNormMapping(
                        "decoder.layers.*.self_attention.out_norm.weight",
                        f"{prefix}.layers.*.linear_attn.norm.weight",
                    ),
                ]
            )

            return MegatronMappingRegistry(*mapping_list)

    registry = getattr(model_bridge.get_model_bridge, "_exact_types", {})
    for source in ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"):
        if source not in registry:
            model_bridge.register_bridge_implementation(
                source=source,
                target=GPTModel,
                bridge_class=Qwen35MoEBridge,
            )


def _build_bridge(hf_path: str, cfg: BridgeConfig):
    """Build Megatron-Bridge AutoBridge lazily from an HF model path."""
    from megatron.bridge import AutoBridge

    _register_bridge_compat_aliases()
    bridge = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True)
    hf_config = _bridge_hf_config(bridge)
    if hf_config is not None and not hasattr(hf_config, "rope_theta"):
        hf_config.rope_theta = hf_config.to_dict().get("rope_theta", 1000000.0)

    if callable(cfg.bridge_post_init):
        cfg.bridge_post_init(bridge)

    return bridge


def _build_optimizer(model_list: list, cfg: BridgeConfig):
    from megatron.core.optimizer import get_megatron_optimizer

    return get_megatron_optimizer(
        config=build_mc_optimizer_config(
            cfg.optimizer,
            override_optimizer_config=cfg.override_optimizer_config,
        ),
        model_chunks=model_list,
    )


def _build_lr_scheduler(optimizer, cfg: BridgeConfig):
    opt = cfg.optimizer
    total_steps = opt.total_training_steps
    if total_steps <= 0:
        return None

    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    warmup_steps = opt.lr_warmup_steps
    if warmup_steps <= 0 and opt.lr_warmup_steps_ratio > 0:
        warmup_steps = int(opt.lr_warmup_steps_ratio * total_steps)
    warmup_steps = max(warmup_steps, 0)
    decay_steps = opt.lr_decay_steps if opt.lr_decay_steps is not None else total_steps

    return OptimizerParamScheduler(
        optimizer,
        init_lr=opt.lr_warmup_init,
        max_lr=opt.lr,
        min_lr=opt.min_lr,
        lr_warmup_steps=warmup_steps,
        lr_decay_steps=decay_steps,
        lr_decay_style=opt.lr_decay_style,
        start_wd=opt.weight_decay,
        end_wd=opt.weight_decay,
        wd_incr_steps=total_steps,
        wd_incr_style=opt.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=opt.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=not opt.use_checkpoint_opt_param_scheduler,
        wsd_decay_steps=opt.lr_wsd_decay_steps,
        lr_wsd_decay_style=opt.lr_wsd_decay_style,
    )


def _build_ddp_config(cfg: BridgeConfig):
    from megatron.core.distributed import DistributedDataParallelConfig

    ddp_kwargs = {
        "use_distributed_optimizer": True,
        "overlap_grad_reduce": False,
        "grad_reduce_in_fp32": True,
    }
    ddp_kwargs.update(cfg.override_ddp_config)
    return DistributedDataParallelConfig(**ddp_kwargs)


def _resolve_benchmark_protocol(cfg: BridgeConfig, bridge) -> Any | None:
    """Best-effort protocol lookup for model stats used by bench examples."""
    from megatron.lite.model.registry import get_train_runtime_module, resolve_model_type_from_hf

    model_name = cfg.model_name
    if model_name == "auto":
        try:
            model_name = resolve_model_type_from_hf(_bridge_hf_config(bridge))
        except ValueError:
            return None

    try:
        return get_train_runtime_module(model_name)
    except ValueError:
        return None


def _as_data_iter(data: Any):
    if hasattr(data, "__next__"):
        return data
    if isinstance(data, list):
        return iter(data)
    return iter([data])


class BridgeRuntime(RuntimeBase):
    """Megatron-Bridge training backend using Megatron-Core optimizer state."""

    def __init__(self, hf_path: str, cfg: BridgeConfig | dict[str, Any]):
        self._hf_path = hf_path
        self._cfg = cfg if isinstance(cfg, BridgeConfig) else BridgeConfig.from_dict(cfg)
        self._offload_param = self._cfg.param_offload
        self._offload_optimizer = self._cfg.optimizer_offload

    def build_model(
        self,
        hf_path: str | None = None,
        cfg: BridgeConfig | dict[str, Any] | None = None,
        **kwargs,
    ) -> ModelHandle:
        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.enums import ModelType

        if cfg is None:
            rt_cfg = self._cfg
        elif isinstance(cfg, BridgeConfig):
            rt_cfg = cfg
        else:
            rt_cfg = BridgeConfig.from_dict(cfg)
        hf_path = hf_path or self._hf_path

        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(minutes=10))
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

        p = rt_cfg.parallel
        init_kwargs = dict(
            tensor_model_parallel_size=p.tp,
            pipeline_model_parallel_size=p.pp,
            expert_model_parallel_size=p.ep,
            context_parallel_size=p.cp,
        )
        if p.etp is not None:
            init_kwargs["expert_tensor_parallel_size"] = p.etp
        if p.vpp > 1:
            init_kwargs["virtual_pipeline_model_parallel_size"] = p.vpp

        if not mpu.model_parallel_is_initialized():
            mpu.initialize_model_parallel(**init_kwargs)
        model_parallel_cuda_manual_seed(rt_cfg.seed)

        bridge = _build_bridge(hf_path, rt_cfg)
        provider = bridge.to_megatron_provider(
            load_weights=rt_cfg.load_hf_weights,
            hf_path=hf_path if rt_cfg.load_hf_weights else None,
        )
        _configure_provider(provider, rt_cfg)
        if hasattr(provider, "finalize"):
            provider.finalize()

        model_list = provider.provide_distributed_model(
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
            ddp_config=_build_ddp_config(rt_cfg),
            bf16=True,
        )

        optimizer = _build_optimizer(model_list, rt_cfg) if rt_cfg.build_optimizer else None
        lr_scheduler = _build_lr_scheduler(optimizer, rt_cfg) if optimizer is not None else None
        register_training_hooks(model_list, optimizer)

        if self._offload_param:
            offload_model_to_cpu(model_list)
        if self._offload_optimizer and optimizer is not None:
            offload_optimizer(optimizer)

        logger.info(
            "BridgeRuntime: model built, tp=%d ep=%d pp=%d cp=%d",
            p.tp,
            p.ep,
            p.pp,
            p.cp,
        )

        return ModelHandle(
            model=model_list[0],
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            parallel_state=_MpuParallelState(mpu),
            config=rt_cfg,
            _extras={
                "bridge": bridge,
                "provider": provider,
                "model_list": model_list,
                "mpu": mpu,
                "model_cfg": _bridge_hf_config(bridge),
                "protocol": _resolve_benchmark_protocol(rt_cfg, bridge),
                "optimizer_backend": "distopt" if optimizer is not None else "none",
                "world_size": dist.get_world_size(),
            },
        )

    def forward_backward(
        self,
        handle: ModelHandle,
        data: Any,
        loss_fn,
        *,
        num_microbatches: int = 1,
        forward_only: bool = False,
    ) -> ForwardResult:
        from megatron.core import parallel_state as mpu
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        if num_microbatches < 1:
            raise ValueError("num_microbatches must be >= 1")

        model_list = handle._extras["model_list"]
        data_iter = _as_data_iter(data)
        last_loss: list[float | None] = [None]

        def _fwd_step(data_iterator, model):
            sample = next(data_iterator)
            if isinstance(sample, Batch):
                sample = {
                    "input_ids": sample["input_ids"],
                    "labels": sample["labels"],
                    "position_ids": getattr(sample, "position_ids", None),
                }
            if not isinstance(sample, dict):
                raise TypeError(
                    f"BridgeRuntime expected dict or Batch data, got {type(sample).__name__}."
                )

            output_tensor = model(
                input_ids=sample["input_ids"],
                position_ids=sample.get("position_ids"),
                attention_mask=sample.get("attention_mask"),
                labels=sample["labels"],
                packed_seq_params=sample.get("packed_seq_params"),
            )
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]

            def _mc_loss_fn(output_tensor, non_loss_data=False):
                if loss_fn is not None:
                    loss, _metrics = loss_fn({"output_tensor": output_tensor}, sample)
                else:
                    loss = output_tensor.mean()
                last_loss[0] = float(loss.detach().item())
                return loss, {}

            return output_tensor, _mc_loss_fn

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            batches = [next(data_iter) for _ in range(num_microbatches)]
            batch_generator = [iter(batches) for _ in range(vpp_size)]
        else:
            batch_generator = data_iter

        get_forward_backward_func()(
            forward_step_func=_fwd_step,
            data_iterator=batch_generator,
            model=model_list,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            seq_length=1,
            micro_batch_size=1,
        )

        if not forward_only:
            from megatron.core.distributed.finalize_model_grads import finalize_model_grads

            finalize_model_grads(model_list)

        loss_val = last_loss[0]
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            loss_t = torch.tensor([loss_val or 0.0], device="cuda")
            dist.broadcast(loss_t, src=mpu.get_pipeline_model_parallel_last_rank())
            loss_val = float(loss_t.item())

        result_loss = torch.tensor(loss_val or 0.0)
        return ForwardResult(
            model_output=ModelOutputs(loss=result_loss),
            metrics={"loss": loss_val if loss_val is not None else 0.0},
        )

    def zero_grad(self, handle: ModelHandle) -> None:
        if handle._optimizer is not None:
            handle._optimizer.zero_grad()
        for model in handle._extras["model_list"]:
            if handle._optimizer is None:
                model.zero_grad(set_to_none=True)
            if hasattr(model, "zero_grad_buffer"):
                model.zero_grad_buffer()

    def optimizer_step(self, handle: ModelHandle) -> tuple[bool, float, int | None]:
        if handle._optimizer is None:
            return True, 0.0, 0
        update_successful, grad_norm, num_zeros = handle._optimizer.step()
        return update_successful, float(grad_norm), num_zeros

    def lr_scheduler_step(self, handle: ModelHandle) -> float | list[float]:
        if handle._lr_scheduler is not None:
            handle._lr_scheduler.step(1)
            return handle._optimizer.param_groups[0]["lr"]
        return 0.0

    def is_mp_src_rank_with_outputs(self, handle: ModelHandle) -> bool:
        return is_mp_src_rank_with_outputs()

    def to(
        self,
        handle: ModelHandle,
        device: str,
        *,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
    ) -> None:
        model_list = handle._extras["model_list"]
        opt = handle._optimizer
        if device == "cuda":
            if model:
                load_model_to_gpu(model_list, load_grad=grad)
            if optimizer and opt is not None:
                load_optimizer(opt)
        elif device == "cpu":
            if model:
                offload_model_to_cpu(model_list)
            if optimizer and opt is not None:
                offload_optimizer(opt)
        else:
            raise ValueError(f"BridgeRuntime.to supports only 'cpu' or 'cuda', got {device!r}.")

    def train_mode(self, handle: ModelHandle):
        return _BridgeTrainCtx(self, handle)

    def eval_mode(self, handle: ModelHandle):
        return _BridgeEvalCtx(self, handle)

    def export_weights(self, handle: ModelHandle, **kwargs) -> Iterator[tuple[str, torch.Tensor]]:
        bridge = handle._extras["bridge"]
        model_list = handle._extras["model_list"]
        load_model_to_gpu(model_list, load_grad=False)
        return bridge.export_hf_weights(model_list, cpu=bool(kwargs.get("cpu", False)))

    def save_checkpoint(self, handle: ModelHandle, path: str, **kwargs) -> None:
        model_list = handle._extras["model_list"]
        opt = handle._optimizer
        load_model_to_gpu(model_list, load_grad=True)

        from megatron.core import dist_checkpointing

        state = build_sharded_state_dict(model_list, opt, handle._lr_scheduler)
        os.makedirs(path, exist_ok=True)
        dist_checkpointing.save(state, path)
        dist.barrier()

        if self._offload_param:
            offload_model_to_cpu(model_list)

    def load_checkpoint(self, handle: ModelHandle, path: str, **kwargs) -> None:
        model_list = handle._extras["model_list"]
        opt = handle._optimizer
        load_model_to_gpu(model_list, load_grad=True)

        from megatron.core import dist_checkpointing

        state = build_sharded_state_dict(model_list, opt, handle._lr_scheduler)
        dist_checkpointing.load(state, path)

        if self._offload_param:
            offload_model_to_cpu(model_list)
        if self._offload_optimizer and opt is not None:
            offload_optimizer(opt)


class _BridgeTrainCtx:
    def __init__(self, runtime: BridgeRuntime, handle: ModelHandle):
        self._runtime = runtime
        self._handle = handle

    def __enter__(self):
        if self._runtime._offload_param or self._runtime._offload_optimizer:
            self._runtime.to(
                self._handle,
                "cuda",
                model=self._runtime._offload_param,
                optimizer=self._runtime._offload_optimizer,
                grad=self._runtime._offload_param,
            )
        for model in self._handle._extras["model_list"]:
            model.train()
        return self

    def __exit__(self, *exc):
        self._runtime.zero_grad(self._handle)
        if self._runtime._offload_param or self._runtime._offload_optimizer:
            self._runtime.to(
                self._handle,
                "cpu",
                model=self._runtime._offload_param,
                optimizer=self._runtime._offload_optimizer,
                grad=self._runtime._offload_param,
            )
        return False


class _BridgeEvalCtx:
    def __init__(self, runtime: BridgeRuntime, handle: ModelHandle):
        self._runtime = runtime
        self._handle = handle
        self._prev_grad = torch.is_grad_enabled()

    def __enter__(self):
        if self._runtime._offload_param:
            self._runtime.to(self._handle, "cuda", model=True, optimizer=False, grad=False)
        for model in self._handle._extras["model_list"]:
            model.eval()
        torch.set_grad_enabled(False)
        return self

    def __exit__(self, *exc):
        torch.set_grad_enabled(self._prev_grad)
        if self._runtime._offload_param:
            self._runtime.to(self._handle, "cpu", model=True, optimizer=False, grad=False)
        return False


__all__ = ["BridgeRuntime"]
