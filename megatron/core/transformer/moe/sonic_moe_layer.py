# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SonicMoE module adapter for Megatron-Core model specs.

For EP=1 this module wraps ``sonicmoe.MoE`` directly, so SonicMoE owns routing
and expert compute. For EP>1 it uses Megatron's router and token dispatcher for
EP communication, while SonicMoE computes the rank-local experts. Checkpoint
keys stay compatible with Megatron MoE.
"""

from __future__ import annotations

import functools
from typing import Optional, Union

import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import BaseMoELayer, MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    MoECudaGraphTensorStore,
    compute_routing_scores_for_aux_loss,
    get_default_pg_collection,
    router_gating_linear,
    save_to_aux_losses_tracker,
    switch_load_balancing_loss_func,
    z_loss_func,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_object_for_checkpoint,
)
from megatron.core.utils import get_pg_rank, get_pg_size, make_sharded_tensor_for_checkpoint

try:
    from megatron.core.extensions.transformer_engine import te_checkpoint
except ImportError:
    te_checkpoint = None


try:
    from sonicmoe import KernelBackendMoE
    from sonicmoe import MoE as _SonicMoE
    from sonicmoe.enums import ActivationType
except ImportError as exc:
    KernelBackendMoE = None
    ActivationType = None
    _SonicMoE = torch.nn.Module
    _SONICMOE_IMPORT_ERROR = exc
else:
    _SONICMOE_IMPORT_ERROR = None


_SONIC_ROUTER_SCORE_FUNCTIONS = ("softmax", "sigmoid")


def _requires_megatron_router(config: TransformerConfig) -> bool:
    return (
        config.moe_router_score_function not in _SONIC_ROUTER_SCORE_FUNCTIONS
        or config.moe_router_enable_expert_bias
        or config.moe_router_force_load_balancing
        or getattr(config, "moe_router_force_biased", None) is not None
    )


def _require_sonicmoe() -> None:
    if _SONICMOE_IMPORT_ERROR is not None:
        raise ImportError(
            "SonicMoELayer requires the optional sonic-moe package. "
            "Install it on the training machine"
        ) from _SONICMOE_IMPORT_ERROR


class _MegatronMoE(_SonicMoE):
    """Sonic MoE variant that follows Megatron gradient-accumulation fusion."""

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function,
        add_bias: bool,
        std: float,
        router_score_function: str = "softmax",
        router_score_over_topk: bool = True,
        router_dtype: Optional[str] = None,
        accumulate_wgrad_into_main_grad: bool = False,
    ) -> None:
        _require_sonicmoe()
        super().__init__(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            add_bias=add_bias,
            std=std,
            router_score_function=router_score_function,
            router_score_over_topk=router_score_over_topk,
        )
        self.megatron_router_dtype = router_dtype
        self.accumulate_wgrad_into_main_grad = accumulate_wgrad_into_main_grad
        self._maintain_router_param_dtype()

    def _target_router_param_dtype(self) -> torch.dtype:
        return self.c_fc.weight.dtype

    def _maintain_router_param_dtype(self) -> None:
        target_dtype = self._target_router_param_dtype()
        if self.router.weight.dtype != target_dtype:
            self.router.to(dtype=target_dtype)

    def _apply(self, fn):
        module = super()._apply(fn)
        self._maintain_router_param_dtype()
        return module


def _import_sonicmoe_functional():
    try:
        from sonicmoe.enums import ActivationType, is_glu
        from sonicmoe.functional import (
            TC_Softmax_Topk_Router_Function,
            TC_topk_router_metadata_triton,
            _DownProjection,
            _UpProjection,
            moe_general_routing_inputs,
        )
    except ImportError as exc:
        raise ImportError(
            "SonicMoELayer requires sonic-moe functional kernels from the optional "
            "sonic-moe package."
        ) from exc
    return (
        ActivationType,
        is_glu,
        TC_Softmax_Topk_Router_Function,
        TC_topk_router_metadata_triton,
        _DownProjection,
        _UpProjection,
        moe_general_routing_inputs,
    )


def _sonic_activation_type(config: TransformerConfig):
    _require_sonicmoe()
    if not config.gated_linear_unit:
        raise ValueError("SonicMoELayer only supports gated MoE MLPs.")
    activation_name = getattr(config.activation_func, "__name__", None)
    if config.activation_func is F.silu or activation_name == "silu":
        return ActivationType.SWIGLU
    if config.activation_func is F.gelu or activation_name == "gelu":
        return ActivationType.GEGLU
    raise ValueError(
        "SonicMoELayer only supports SonicMoE GLU activations SwiGLU and GEGLU. "
        f"Got activation_func={config.activation_func}."
    )


def _as_list(value: Union[str, list]) -> list:
    return value if isinstance(value, list) else [value]


def _loss_coeff(config: TransformerConfig, loss_type: str) -> float:
    routing_types = _as_list(config.moe_router_load_balancing_type)
    aux_coeffs = _as_list(config.moe_aux_loss_coeff)
    if len(aux_coeffs) == 1 and len(routing_types) > 1:
        aux_coeffs = aux_coeffs * len(routing_types)
    for routing_type, coeff in zip(routing_types, aux_coeffs):
        if routing_type == loss_type:
            return float(coeff)
    return 0.0


def _has_positive_unsupported_aux_loss(config: TransformerConfig) -> bool:
    routing_types = _as_list(config.moe_router_load_balancing_type)
    aux_coeffs = _as_list(config.moe_aux_loss_coeff)
    if len(aux_coeffs) == 1 and len(routing_types) > 1:
        aux_coeffs = aux_coeffs * len(routing_types)
    for routing_type, coeff in zip(routing_types, aux_coeffs):
        if (
            routing_type not in ("aux_loss", "seq_aux_loss", "global_aux_loss", "none")
            and float(coeff) > 0.0
        ):
            return True
    return False


def _get_tokens_per_expert_and_token_count(
    routing_map: torch.Tensor,
    reduce_group: torch.distributed.ProcessGroup,
    topk: int = None,
    with_padding_mask: bool = False,
):
    """Target-local copy of the newer MoE aux-loss token-count helper."""
    local_tokens_per_expert = routing_map.sum(dim=0)
    global_tokens_per_expert = local_tokens_per_expert
    group_size = reduce_group.size() if reduce_group is not None else 1
    if group_size > 1:
        global_tokens_per_expert = local_tokens_per_expert.clone()
        torch.distributed.all_reduce(global_tokens_per_expert, group=reduce_group)

    if with_padding_mask:
        local_num_tokens = local_tokens_per_expert.sum() / topk
        total_num_tokens = global_tokens_per_expert.sum() / topk
    else:
        local_num_tokens = routing_map.shape[0]
        total_num_tokens = local_num_tokens * group_size
    return global_tokens_per_expert, local_num_tokens, total_num_tokens


def _check_supported_config(config: TransformerConfig) -> None:
    if config.num_moe_experts is None:
        raise ValueError("SonicMoELayer requires config.num_moe_experts.")
    if config.expert_tensor_parallel_size != 1:
        raise ValueError("SonicMoELayer does not integrate expert tensor parallelism yet.")
    if config.tensor_model_parallel_size > 1 and not config.sequence_parallel:
        raise ValueError(
            "SonicMoELayer supports tensor parallelism only with sequence parallelism "
            "and replicated experts (expert_tensor_parallel_size=1)."
        )
    if getattr(config, "moe_latent_size", None) is not None:
        raise ValueError("SonicMoELayer does not support MoE latent projections.")
    if config.moe_shared_expert_overlap:
        raise ValueError(
            "SonicMoELayer supports Megatron shared experts only without "
            "moe_shared_expert_overlap."
        )
    if config.fp8 or config.fp4:
        raise ValueError("SonicMoELayer does not support fp8/fp4 expert compute.")
    if config.moe_expert_capacity_factor is not None:
        raise ValueError("SonicMoELayer does not support token dropping or expert capacity.")
    if config.moe_router_padding_for_quantization:
        raise ValueError("SonicMoELayer does not support router padding for quantization.")
    if config.moe_router_num_groups is not None or config.moe_router_group_topk is not None:
        raise ValueError("SonicMoELayer does not support group-limited routing.")
    if config.moe_input_jitter_eps is not None:
        raise ValueError("SonicMoELayer does not support Megatron router input jitter.")
    if _has_positive_unsupported_aux_loss(config):
        raise ValueError(
            "SonicMoELayer only supports aux_loss, seq_aux_loss, global_aux_loss, or no aux loss."
        )
    if getattr(config, "glu_linear_offset", 0.0) != 0.0:
        raise ValueError("SonicMoELayer does not support nonzero glu_linear_offset.")
    activation_type = _sonic_activation_type(config)
    if (
        getattr(config, "activation_func_clamp_value", None) is not None
        and activation_type != ActivationType.SWIGLU
    ):
        raise ValueError("SonicMoELayer fused activation clamp currently supports SwiGLU only.")


def _set_sonic_param_dtypes(module: torch.nn.Module, config: TransformerConfig) -> None:
    module.c_fc.to(dtype=config.params_dtype)
    module.c_proj.to(dtype=config.params_dtype)
    module.router.to(dtype=config.params_dtype)
    # With EP, ordinary TP peers can own different expert IDs. Only EP1 experts are
    # replicated across the TP group and need sequence-parallel gradient reduction.
    sequence_parallel = config.sequence_parallel and config.expert_model_parallel_size == 1
    for param in module.parameters():
        setattr(param, "sequence_parallel", sequence_parallel)


def _sync_sonic_params_across_tensor_parallel(
    module: torch.nn.Module, tp_group: Optional[torch.distributed.ProcessGroup]
) -> bool:
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or tp_group is None
        or tp_group.size() <= 1
    ):
        return True
    if any(param.device.type != "cuda" for param in module.parameters()):
        return False
    src_rank = torch.distributed.get_global_rank(tp_group, 0)
    for param in module.parameters():
        torch.distributed.broadcast(param.data, src=src_rank, group=tp_group)
    return True


def _maybe_move_to_runtime_device(module: torch.nn.Module, config: TransformerConfig) -> None:
    if not config.use_cpu_initialization and torch.cuda.is_available():
        module.to(device=torch.cuda.current_device())
    _set_sonic_param_dtypes(module, config)


class _SonicParamSync(torch.nn.Module):
    """Expose SonicMoE's directly-read params to Megatron DDP pre-hooks.

    SonicMoE's fast path does not call the ``router``, ``c_fc``, or ``c_proj``
    modules; it reads their parameters directly. Megatron's overlapped
    distributed optimizer waits for param all-gathers in module forward
    pre-hooks, so this no-op module is called immediately before SonicMoE.
    """

    def __init__(self, sonic_moe: torch.nn.Module, include_router: bool = True) -> None:
        super().__init__()
        if include_router:
            self.register_parameter("router_weight", sonic_moe.router.weight)
        self.register_parameter("c_fc_weight", sonic_moe.c_fc.weight)
        self.register_parameter("c_proj_weight", sonic_moe.c_proj.weight)
        if sonic_moe.c_fc.bias is not None:
            self.register_parameter("c_fc_bias", sonic_moe.c_fc.bias)
        if sonic_moe.c_proj.bias is not None:
            self.register_parameter("c_proj_bias", sonic_moe.c_proj.bias)

    def forward(self) -> None:
        return None

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        del destination, prefix, keep_vars

    # pylint: disable=arguments-differ
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        del state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        del prefix, sharded_offsets, metadata
        return {}


class SonicMoELayer(MoELayer):
    """Megatron-compatible wrapper around ``sonicmoe.MoE``.

    This class subclasses ``MoELayer`` so ``TransformerLayer`` treats it as a
    MoE block and forwards MoE-only kwargs. With EP=1 it bypasses Megatron token
    dispatchers. With EP>1 it reuses Megatron dispatchers and runs Sonic kernels
    over local expert shards.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        name: Optional[str] = None,
    ) -> None:
        del name
        _check_supported_config(config)
        if pg_collection is None:
            pg_collection = get_default_pg_collection()

        BaseMoELayer.__init__(
            self,
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        self.tp_group = pg_collection.tp
        self.expt_tp_group = pg_collection.expt_tp
        self.expt_dp_group = pg_collection.expt_dp
        self.tp_cp_group = pg_collection.tp_cp
        self.tp_dp_cp_group = pg_collection.tp_dp_cp
        self.layer_number = layer_number
        self.is_mtp = is_mtp_layer
        self.submodules = submodules
        self._use_megatron_dispatch = (
            config.expert_model_parallel_size > 1 or _requires_megatron_router(config)
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )
        # SonicMoELayer calls BaseMoELayer.__init__ (not MoELayer.__init__), so it skips the
        # cudagraph tensor store that MoELayer sets up. The inherited route/preprocess/
        # shared_experts_compute are @maybe_skip_or_early_return_by_cudagraph-decorated and read
        # self.cudagraph_tensor_store even on the non-cudagraph path (moe_utils: is_empty()), which
        # the fine-grained / EP-comm-overlap scheduler exercises. Provide it so Sonic runs
        # under overlap.
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()
        self._sonic_params_synced_across_tp = False
        self._aux_loss_coeff = _loss_coeff(config, "aux_loss")
        self._seq_aux_loss_coeff = _loss_coeff(config, "seq_aux_loss")
        self._global_aux_loss_coeff = _loss_coeff(config, "global_aux_loss")
        self._z_loss_coeff = config.moe_z_loss_coeff

        if self._global_aux_loss_coeff > 0.0:
            device = torch.cuda.current_device() if torch.cuda.is_available() else None
            self.register_buffer(
                "global_tokens_per_expert",
                torch.zeros(config.num_moe_experts, dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "ga_steps", torch.tensor(0, dtype=torch.float32, device=device), persistent=False
            )
        else:
            self.global_tokens_per_expert = None
            self.ga_steps = None

        if self._use_megatron_dispatch:
            router = TopKRouter
            if self.submodules is not None and self.submodules.router is not None:
                router = self.submodules.router
            self.router = build_module(
                router,
                config=self.config,
                pg_collection=pg_collection,
                is_mtp_layer=is_mtp_layer,
                layer_number=layer_number,
            )

            if config.moe_token_dispatcher_type == "allgather":
                self.token_dispatcher = MoEAllGatherTokenDispatcher(
                    self.num_local_experts,
                    self.local_expert_indices,
                    config=self.config,
                    pg_collection=pg_collection,
                )
            elif config.moe_token_dispatcher_type == "alltoall":
                self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                    self.num_local_experts,
                    self.local_expert_indices,
                    config=self.config,
                    pg_collection=pg_collection,
                )
            elif config.moe_token_dispatcher_type == "flex":
                self.token_dispatcher = MoEFlexTokenDispatcher(
                    self.num_local_experts,
                    self.local_expert_indices,
                    config=self.config,
                    pg_collection=pg_collection,
                )
            else:
                raise ValueError(
                    f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
                )

        if self.use_shared_expert:
            if self.submodules is None or self.submodules.shared_experts is None:
                raise ValueError(
                    "SonicMoELayer shared experts require a MoESubmodules.shared_experts spec. "
                    "Use replace_moe_layer_specs_with_sonic_moe() on the original MoE spec."
                )
            self.shared_experts = build_module(
                self.submodules.shared_experts,
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
            )

        _require_sonicmoe()
        self.kernel_backend_moe = KernelBackendMoE.sonicmoe
        self.sonic_moe = _MegatronMoE(
            num_experts=(
                self.num_local_experts if self._use_megatron_dispatch else config.num_moe_experts
            ),
            num_experts_per_tok=config.moe_router_topk,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_ffn_hidden_size,
            activation_function=_sonic_activation_type(config),
            add_bias=config.add_bias_linear,
            std=config.init_method_std,
            router_score_function=(
                "softmax" if self._use_megatron_dispatch else config.moe_router_score_function
            ),
            router_score_over_topk=self._score_over_topk(),
            router_dtype=config.moe_router_dtype,
            accumulate_wgrad_into_main_grad=config.gradient_accumulation_fusion,
        )
        _maybe_move_to_runtime_device(self.sonic_moe, config)
        self._sync_sonic_params_across_tensor_parallel_once()
        self.sonic_param_sync = _SonicParamSync(
            self.sonic_moe, include_router=not self._use_megatron_dispatch
        )

        if self._use_megatron_dispatch:
            self.sonic_moe.router.weight.requires_grad_(False)
            if self.sonic_moe.router.bias is not None:
                self.sonic_moe.router.bias.requires_grad_(False)

        # The optimizer groups params by is_expert_parallel = (not param.allreduce), and MoE
        # builds a [dense, expert] ChainedOptimizer. To keep the saved optimizer structure
        # identical to standard MoE (so sonic<->standard ckpts interop, incl. optimizer
        # state), the fused experts (c_fc/c_proj) must be expert-parallel even on the EP1
        # fast-path, while the router stays dense. At EP1 the expert-DP group == the full DP
        # group, so gradients still all-reduce correctly.
        if self._use_megatron_dispatch:
            for param in self.sonic_moe.parameters():
                setattr(param, "allreduce", False)
        else:
            router_param_ids = {id(p) for p in self.sonic_moe.router.parameters()}
            for param in self.sonic_moe.parameters():
                setattr(param, "allreduce", id(param) in router_param_ids)

    def _apply(self, fn):
        module = super()._apply(fn)
        self.sonic_moe._maintain_router_param_dtype()
        self._sync_sonic_params_across_tensor_parallel_once()
        return module

    def _sync_sonic_params_across_tensor_parallel_once(self) -> None:
        if self._sonic_params_synced_across_tp:
            return
        # With EP, ordinary TP peers own different expert IDs and must not synchronize
        # their expert parameters. Expert replicas are synchronized over expt_dp instead.
        if self.config.expert_model_parallel_size > 1:
            self._sonic_params_synced_across_tp = True
            return
        self._sonic_params_synced_across_tp = _sync_sonic_params_across_tensor_parallel(
            self.sonic_moe, self.tp_group
        )

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number
        if self.router is not None:
            self.router.set_layer_number(layer_number)

    def set_is_mtp(self):
        self.is_mtp = True
        self.is_mtp_layer = True

    def reset_global_aux_loss_tracker(self):
        if self._use_megatron_dispatch and hasattr(self.router, "reset_global_aux_loss_tracker"):
            self.router.reset_global_aux_loss_tracker()
        if self.global_tokens_per_expert is not None:
            self.global_tokens_per_expert.zero_()
            self.ga_steps.zero_()

    def _num_layers_for_loss_tracker(self) -> int:
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        return num_layers

    def _attach_scaled_loss(
        self,
        output: torch.Tensor,
        loss: torch.Tensor,
        coeff: float,
        name: str,
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
        reduce_group_has_dp: bool = False,
        valid_token_count: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if coeff == 0.0:
            return output

        save_to_aux_losses_tracker(
            name,
            loss / coeff,
            self.layer_number,
            self._num_layers_for_loss_tracker(),
            reduce_group=reduce_group,
        )
        if self.config.calculate_per_token_loss:
            num_tokens = valid_token_count if valid_token_count is not None else output.shape[0]
            loss = loss * num_tokens
        return MoEAuxLossAutoScaler.apply(output, loss)

    def _needs_router_losses(self) -> bool:
        return (
            self.training
            and torch.is_grad_enabled()
            and (
                self._aux_loss_coeff > 0.0
                or self._seq_aux_loss_coeff > 0.0
                or self._global_aux_loss_coeff > 0.0
                or (self._z_loss_coeff is not None and self._z_loss_coeff != 0.0)
            )
        )

    def _tokens_per_expert_and_count(
        self, local_tokens_per_expert: torch.Tensor, reduce_group: torch.distributed.ProcessGroup
    ):
        global_tokens_per_expert = local_tokens_per_expert.float()
        if reduce_group.size() > 1:
            global_tokens_per_expert = global_tokens_per_expert.clone()
            torch.distributed.all_reduce(global_tokens_per_expert, group=reduce_group)
        local_num_tokens = local_tokens_per_expert.sum() / self.config.moe_router_topk
        total_num_tokens = global_tokens_per_expert.sum() / self.config.moe_router_topk
        return global_tokens_per_expert, local_num_tokens, total_num_tokens

    def _score_over_topk(self) -> bool:
        return not self.config.moe_router_pre_softmax

    def _apply_sonic_router_topk(self, router_func, router_logits: torch.Tensor, num_experts: int):
        args = (
            router_logits,
            num_experts,
            self.config.moe_router_topk,
            self._score_over_topk(),
            False,
        )
        if self.config.moe_router_score_function == "sigmoid":
            return router_func.apply(*args, "sigmoid")
        try:
            return router_func.apply(*args, "softmax")
        except TypeError:
            return router_func.apply(*args)

    def _normalize_topk_scores(self, topk_scores: torch.Tensor) -> torch.Tensor:
        if self.config.moe_router_score_function == "sigmoid" and self.config.moe_router_topk > 1:
            topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-20)
        if self.config.moe_router_topk_scaling_factor is not None:
            topk_scores = topk_scores * self.config.moe_router_topk_scaling_factor
        return topk_scores

    def _scores_for_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        if self.config.moe_router_score_function == "softmax":
            return F.softmax(router_logits, dim=-1, dtype=torch.float32)
        if self.config.moe_router_score_function == "sigmoid":
            scores = torch.sigmoid(router_logits.float())
            return scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        raise ValueError(f"Invalid score_function: {self.config.moe_router_score_function}")

    def _router_dtype(self, input: torch.Tensor) -> torch.dtype:
        if self.config.moe_router_dtype == "fp32":
            return torch.float32
        if self.config.moe_router_dtype == "fp64":
            return torch.float64
        return input.dtype

    def _activation_func_clamp_value(self) -> Optional[float]:
        value = getattr(self.config, "activation_func_clamp_value", None)
        return None if value is None else float(value)

    def _sonic_tc_forward(
        self, hidden_states: torch.Tensor, is_inference_mode_enabled: bool = False
    ):
        (
            ActivationType,
            is_glu,
            TC_Softmax_Topk_Router_Function,
            TC_topk_router_metadata_triton,
            _DownProjection,
            _UpProjection,
            _,
        ) = _import_sonicmoe_functional()

        original_shape = hidden_states.shape
        x = hidden_states.view(-1, self.config.hidden_size)
        router_logits = router_gating_linear(
            x, self.sonic_moe.router.weight, self.sonic_moe.router.bias, self._router_dtype(x)
        )
        num_experts = self.sonic_moe.router.weight.size(0)
        topk_scores, topk_indices = self._apply_sonic_router_topk(
            TC_Softmax_Topk_Router_Function, router_logits, num_experts
        )
        topk_scores = self._normalize_topk_scores(topk_scores)

        num_tokens, topk = topk_indices.size()
        num_routed_tokens = num_tokens * topk
        device = topk_indices.device

        s_scatter_idx = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
        expert_frequency = torch.empty(num_experts, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)

        TC_topk_router_metadata_triton(
            topk_indices,
            num_experts,
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        )

        activation_type = self.sonic_moe.activation_function
        if type(activation_type) == str:
            activation_type = ActivationType(activation_type)

        assert not torch.compiler.is_compiling()
        assert is_glu(activation_type), "SonicMoELayer only supports GLU Sonic kernels."

        w1 = self.sonic_moe.c_fc.weight.permute(1, 2, 0)
        w2 = self.sonic_moe.c_proj.weight.permute(1, 2, 0)
        accumulate_wgrad_into_main_grad = getattr(
            self.sonic_moe, "accumulate_wgrad_into_main_grad", False
        )
        a, h = _UpProjection.apply(
            x,
            w1,
            self.sonic_moe.c_fc.bias,
            expert_frequency_offset,
            num_routed_tokens,
            topk,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            None,
            False,
            activation_type,
            is_inference_mode_enabled,
            True,
            accumulate_wgrad_into_main_grad,
            self._activation_func_clamp_value(),
        )

        output = _DownProjection.apply(
            a,
            h,
            w2,
            self.sonic_moe.c_proj.bias,
            topk_scores,
            expert_frequency_offset,
            num_tokens,
            topk,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            None,
            False,
            activation_type,
            accumulate_wgrad_into_main_grad,
            self._activation_func_clamp_value(),
        )

        return output.view(original_shape), router_logits, expert_frequency

    def _local_expert_indices_for_tokens(
        self, tokens_per_expert: torch.Tensor, num_tokens: int, device: torch.device
    ) -> torch.Tensor:
        counts = tokens_per_expert.to(device=device, dtype=torch.long)
        # output_size avoids a CUDA sync to infer sum(counts) inside repeat_interleave.
        expert_indices = torch.repeat_interleave(
            torch.arange(self.num_local_experts, device=device, dtype=torch.int32),
            counts,
            output_size=num_tokens,
        )
        if expert_indices.numel() != num_tokens:
            raise RuntimeError(
                "SonicMoELayer local dispatch metadata mismatch: "
                f"got {expert_indices.numel()} expert assignments for {num_tokens} tokens."
            )
        return expert_indices

    def _local_expert_indices_from_dispatcher(
        self, num_tokens: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        comm_manager = getattr(self.token_dispatcher, "_comm_manager", None)
        routing_map = getattr(comm_manager, "dispatched_routing_map", None)
        if routing_map is None or routing_map.device != device:
            return None

        expert_offsets = routing_map.sum(dim=0, dtype=torch.int32).cumsum(dim=0)
        positions = torch.arange(num_tokens, device=device, dtype=torch.int32)
        expert_indices = torch.searchsorted(expert_offsets, positions, right=True, out_int32=True)
        if expert_indices.numel() != num_tokens:
            raise RuntimeError(
                "SonicMoELayer local dispatch routing-map mismatch: "
                f"got {expert_indices.numel()} expert assignments for {num_tokens} tokens."
            )
        return expert_indices

    def _sonic_dispatched_forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> torch.Tensor:
        (
            ActivationType,
            _,
            _TC_Softmax_Topk_Router_Function,
            _TC_topk_router_metadata_triton,
            _DownProjection,
            _UpProjection,
            moe_general_routing_inputs,
        ) = _import_sonicmoe_functional()
        del _TC_Softmax_Topk_Router_Function, _TC_topk_router_metadata_triton
        del _DownProjection, _UpProjection

        if hidden_states.numel() == 0:
            return hidden_states

        num_tokens = hidden_states.size(0)
        device = hidden_states.device
        token_indices = torch.arange(num_tokens, device=device, dtype=torch.int32)
        expert_indices = self._local_expert_indices_from_dispatcher(num_tokens, device)
        if expert_indices is None:
            expert_indices = self._local_expert_indices_for_tokens(
                tokens_per_expert, num_tokens, device
            )
        router_scores = permuted_probs.reshape(-1).contiguous()
        if router_scores.numel() != num_tokens:
            raise RuntimeError(
                "SonicMoELayer local dispatch probability mismatch: "
                f"got {router_scores.numel()} scores for {num_tokens} tokens."
            )

        activation_type = self.sonic_moe.activation_function
        if type(activation_type) == str:
            activation_type = ActivationType(activation_type)

        w1 = self.sonic_moe.c_fc.weight.permute(1, 2, 0)
        w2 = self.sonic_moe.c_proj.weight.permute(1, 2, 0)
        accumulate_wgrad_into_main_grad = getattr(
            self.sonic_moe, "accumulate_wgrad_into_main_grad", False
        )
        output, _ = moe_general_routing_inputs(
            hidden_states,
            router_scores,
            token_indices,
            expert_indices,
            w1,
            self.sonic_moe.c_fc.bias,
            w2,
            self.sonic_moe.c_proj.bias,
            self.num_local_experts,
            None,
            activation_type,
            is_inference_mode_enabled=(not self.training),
            concat_layout=True,
            accumulate_wgrad_into_main_grad=accumulate_wgrad_into_main_grad,
            activation_func_clamp_value=self._activation_func_clamp_value(),
        )
        return output

    def _compute_seq_aux_inputs(self, router_logits: torch.Tensor):
        routing_map, scores = compute_routing_scores_for_aux_loss(
            router_logits,
            self.config.moe_router_topk,
            self.config.moe_router_score_function,
            fused=False,
        )
        return routing_map, scores

    def _apply_router_losses(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or not torch.is_grad_enabled():
            return output
        if (
            self._aux_loss_coeff == 0.0
            and self._seq_aux_loss_coeff == 0.0
            and self._global_aux_loss_coeff == 0.0
            and self._z_loss_coeff is None
        ):
            return output

        if self._z_loss_coeff is not None and self._z_loss_coeff != 0.0:
            moe_z_loss_coeff = self._z_loss_coeff / self.tp_cp_group.size()
            z_loss = z_loss_func(router_logits, moe_z_loss_coeff)
            output = self._attach_scaled_loss(
                output, z_loss, moe_z_loss_coeff, "z_loss", valid_token_count=router_logits.shape[0]
            )

        scores = None
        if self._aux_loss_coeff > 0.0:
            scores = self._scores_for_aux_loss(router_logits)
            global_tokens_per_expert, local_num_tokens, total_num_tokens = (
                self._tokens_per_expert_and_count(tokens_per_expert, self.tp_cp_group)
            )
            aux_loss = switch_load_balancing_loss_func(
                probs=scores,
                tokens_per_expert=global_tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.config.moe_router_topk,
                num_experts=self.config.num_moe_experts,
                moe_aux_loss_coeff=self._aux_loss_coeff,
                fused=False,
            )
            output = self._attach_scaled_loss(
                output,
                aux_loss,
                self._aux_loss_coeff,
                "load_balancing_loss",
                reduce_group=self.tp_cp_group,
                valid_token_count=local_num_tokens,
            )

        if self._seq_aux_loss_coeff > 0.0:
            if scores is None:
                scores = self._scores_for_aux_loss(router_logits)
            routing_map, _ = self._compute_seq_aux_inputs(router_logits)
            if hidden_states.dim() >= 3:
                seq_length, bsz = hidden_states.shape[0], hidden_states.shape[1]
            else:
                seq_length, bsz = hidden_states.shape[0], 1
            seq_scores = scores.reshape(seq_length, -1)
            seq_routing_map = routing_map.reshape(seq_length, -1)
            global_tokens_per_expert, local_num_tokens, total_num_tokens = (
                _get_tokens_per_expert_and_token_count(
                    routing_map=seq_routing_map,
                    reduce_group=self.tp_cp_group,
                    topk=self.config.moe_router_topk * bsz,
                )
            )
            seq_aux_loss = (
                switch_load_balancing_loss_func(
                    probs=seq_scores,
                    tokens_per_expert=global_tokens_per_expert,
                    total_num_tokens=total_num_tokens,
                    topk=self.config.moe_router_topk,
                    num_experts=self.config.num_moe_experts,
                    moe_aux_loss_coeff=self._seq_aux_loss_coeff,
                    fused=False,
                )
                / bsz
            )
            output = self._attach_scaled_loss(
                output,
                seq_aux_loss,
                self._seq_aux_loss_coeff,
                "seq_load_balancing_loss",
                reduce_group=self.tp_cp_group,
                valid_token_count=local_num_tokens,
            )

        if self._global_aux_loss_coeff > 0.0:
            if scores is None:
                scores = self._scores_for_aux_loss(router_logits)
            global_tokens_per_expert, local_num_tokens, total_num_tokens = (
                self._tokens_per_expert_and_count(tokens_per_expert, self.tp_dp_cp_group)
            )
            self.global_tokens_per_expert += global_tokens_per_expert
            self.ga_steps += 1
            averaged_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps
            global_aux_loss = switch_load_balancing_loss_func(
                probs=scores,
                tokens_per_expert=averaged_tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.config.moe_router_topk,
                num_experts=self.config.num_moe_experts,
                moe_aux_loss_coeff=self._global_aux_loss_coeff,
                fused=False,
            )
            output = self._attach_scaled_loss(
                output,
                global_aux_loss,
                self._global_aux_loss_coeff,
                "global_load_balancing_loss",
                reduce_group=self.tp_dp_cp_group,
                reduce_group_has_dp=True,
                valid_token_count=local_num_tokens,
            )

        return output

    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Compute the Megatron shared expert path when configured."""
        if not self.use_shared_expert:
            return None
        if self.shared_expert_overlap:
            raise ValueError("SonicMoELayer does not support shared expert overlap.")

        if self.shared_experts_recompute:
            if self.config.fp8 or self.config.fp4:
                if te_checkpoint is None:
                    raise ImportError("Transformer Engine checkpointing is required for fp8/fp4.")
                return te_checkpoint(
                    self.shared_experts,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            return tensor_parallel.checkpoint(self.shared_experts, False, hidden_states)

        return self.shared_experts(hidden_states)

    def postprocess(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Apply Megatron shared expert output after routed expert combine."""
        if self._use_megatron_dispatch:
            output = self.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            if output.shape != shared_expert_output.shape:
                if output.numel() != shared_expert_output.numel():
                    raise RuntimeError(
                        "SonicMoELayer shared expert shape mismatch: "
                        f"routed output shape={tuple(output.shape)}, "
                        f"shared expert output shape={tuple(shared_expert_output.shape)}."
                    )
                output = output.view_as(shared_expert_output)
            output = output + shared_expert_output
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        if intermediate_tensors is not None:
            raise ValueError("SonicMoELayer does not support partial MoE CUDA graph replay.")
        if padding_mask is not None and torch.any(padding_mask):
            raise ValueError("Pure SonicMoE routing does not support padding masks.")

        if self._use_megatron_dispatch:
            return self._forward_megatron_dispatch(hidden_states, input_ids=input_ids)

        self.sonic_param_sync()
        shared_expert_output = self.shared_experts_compute(hidden_states)
        need_router_losses = self._needs_router_losses()
        output, router_logits, tokens_per_expert = self._sonic_tc_forward(
            hidden_states, is_inference_mode_enabled=(not self.training)
        )
        if need_router_losses:
            output = self._apply_router_losses(
                output, hidden_states, router_logits, tokens_per_expert
            )
        output = self.postprocess(output, shared_expert_output)
        return output, None

    def _forward_megatron_dispatch(self, hidden_states: torch.Tensor, input_ids=None):
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism "
                "are enabled without also enabling sequence parallelism."
            )

        def custom_forward(hidden_states):
            shared_expert_output = self.shared_experts_compute(hidden_states)
            probs, routing_map = self.router(hidden_states, input_ids=input_ids)
            residual = hidden_states
            hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
                hidden_states, routing_map, probs
            )
            dispatched_input, probs = self.token_dispatcher.token_dispatch(hidden_states, probs)
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs, residual)
            output = self.token_dispatcher.token_combine(output)
            output = self.postprocess(output, shared_expert_output)
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8 or self.config.fp4:
                if te_checkpoint is None:
                    raise ImportError("Transformer Engine checkpointing is required for fp8/fp4.")
                return te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            return tensor_parallel.checkpoint(custom_forward, False, hidden_states)

        return custom_forward(hidden_states)

    def routed_experts_compute(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ):
        if not self._use_megatron_dispatch:
            return MoELayer.routed_experts_compute(self, hidden_states, probs)
        del residual

        self.sonic_param_sync()
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        # Under EP-comm overlap the fine-grained scheduler drives this method directly and bypasses
        # the forward-level moe_layer_recompute checkpoint (which lives only in the monolithic
        # forward's custom_forward). To still recompute the expert activations in backward WITHOUT
        # wrapping the dispatch/combine all-to-all (they are separate overlap nodes),
        # checkpoint ONLY
        # the pure-compute expert GEMM here. Gated on overlap so the non-overlap path -- already
        # wrapped by the outer custom_forward checkpoint -- is not double-checkpointed. Routed wgrad
        # accumulates inline (accumulate_wgrad_into_main_grad); backward_dw is a no-op for routed
        # experts, so the single recompute+backward computes dgrad+wgrad exactly once.
        if self.moe_layer_recompute and self.config.overlap_moe_expert_parallel_comm:
            expert_output = tensor_parallel.checkpoint(
                self._sonic_dispatched_forward,
                False,
                dispatched_input,
                tokens_per_expert,
                permuted_probs,
            )
        else:
            expert_output = self._sonic_dispatched_forward(
                dispatched_input, tokens_per_expert, permuted_probs
            )
        expert_output = expert_output.to(dtype=dispatched_input.dtype).contiguous()
        output = self.token_dispatcher.combine_preprocess(expert_output)
        return output, None

    def backward_dw(self, routed_experts: bool = True, shared_experts: bool = True):
        # Routed experts accumulate wgrad inline (accumulate_wgrad_into_main_grad), so there is
        # nothing to flush. Shared experts defer their wgrad under delay_wgrad_compute and must
        # be flushed here. The reference wires this via _BackwardDWWrapper calling
        # backward_dw(shared_experts=True); the Scitix fine-grained backport dropped that
        # wrapper and calls layer.mlp.backward_dw() with no args, so the default must be True
        # or the shared-expert weight-grad is silently dropped (grad norm ~2.5% low under
        # --delay-wgrad-compute). Called once per layer per step, so there is no double flush.
        del routed_experts
        if shared_experts and self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        raise ValueError("SonicMoELayer does not support fp8/fp4 pre-MLP layernorm recompute.")

    def _state_tensor(self, tensor: torch.Tensor, keep_vars: bool) -> torch.Tensor:
        return tensor if keep_vars else tensor.detach()

    def _local_expert_index(self, expert_idx: int) -> int:
        first_local_expert = self.local_expert_indices[0]
        local_idx = expert_idx - first_local_expert
        if local_idx < 0 or local_idx >= self.num_local_experts:
            raise IndexError(
                f"Expert {expert_idx} is not local to rank-local experts "
                f"{self.local_expert_indices}."
            )
        return local_idx

    def _router_weight(self) -> torch.Tensor:
        if self._use_megatron_dispatch:
            return self.router.weight
        return self.sonic_moe.router.weight

    def _router_bias(self) -> Optional[torch.Tensor]:
        if self._use_megatron_dispatch:
            return self.router.bias
        return self.sonic_moe.router.bias

    def _origin_expert_fc1_weight(self, expert_idx: int, keep_vars: bool) -> torch.Tensor:
        return self._state_tensor(self.sonic_moe.c_fc.weight, keep_vars)[
            self._local_expert_index(expert_idx)
        ]

    def _origin_expert_fc2_weight(self, expert_idx: int, keep_vars: bool) -> torch.Tensor:
        return self._state_tensor(self.sonic_moe.c_proj.weight, keep_vars)[
            self._local_expert_index(expert_idx)
        ]

    def _origin_state_dict_entries(self, prefix: str, keep_vars: bool) -> dict:
        entries = {f"{prefix}router.weight": self._state_tensor(self._router_weight(), keep_vars)}
        router_bias = self._router_bias()
        if router_bias is not None:
            entries[f"{prefix}router.bias"] = self._state_tensor(router_bias, keep_vars)
        for expert_idx in self.local_expert_indices:
            local_expert_idx = self._local_expert_index(expert_idx)
            expert_prefix = f"{prefix}experts.local_experts.{local_expert_idx}"
            entries[f"{expert_prefix}.linear_fc1.weight"] = self._origin_expert_fc1_weight(
                expert_idx, keep_vars
            )
            entries[f"{expert_prefix}.linear_fc1._extra_state"] = None
            entries[f"{expert_prefix}.linear_fc2.weight"] = self._origin_expert_fc2_weight(
                expert_idx, keep_vars
            )
            entries[f"{expert_prefix}.linear_fc2._extra_state"] = None
            if self.sonic_moe.c_fc.bias is not None:
                entries[f"{expert_prefix}.linear_fc1.bias"] = self._state_tensor(
                    self.sonic_moe.c_fc.bias, keep_vars
                )[local_expert_idx]
            if self.sonic_moe.c_proj.bias is not None:
                entries[f"{expert_prefix}.linear_fc2.bias"] = self._state_tensor(
                    self.sonic_moe.c_proj.bias, keep_vars
                )[local_expert_idx]
        return entries

    def _remove_sonic_state_dict_entries(self, state_dict, prefix: str) -> None:
        for key in (
            "sonic_moe.router.weight",
            "sonic_moe.router.bias",
            "sonic_moe.c_fc.weight",
            "sonic_moe.c_fc.bias",
            "sonic_moe.c_proj.weight",
            "sonic_moe.c_proj.bias",
            "sonic_param_sync.router_weight",
            "sonic_param_sync.c_fc_weight",
            "sonic_param_sync.c_fc_bias",
            "sonic_param_sync.c_proj_weight",
            "sonic_param_sync.c_proj_bias",
        ):
            state_dict.pop(f"{prefix}{key}", None)

    # pylint: disable=arguments-differ
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        self._remove_sonic_state_dict_entries(state_dict, prefix)
        state_dict.update(self._origin_state_dict_entries(prefix, keep_vars))
        return state_dict

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        dp_cp_group = metadata["dp_cp_group"]
        prepend_axis_num = len(sharded_offsets)
        num_global_experts = self.config.num_moe_experts
        expt_tp_rank = get_pg_rank(self.expt_tp_group)
        expt_tp_size = get_pg_size(self.expt_tp_group)
        expt_dp_rank = get_pg_rank(self.expt_dp_group)
        replica_id = (0, expt_tp_rank, expt_dp_rank)

        def fc1_build_fn(key, tensor, replica_id, flattened_range, expert_idx):
            if flattened_range is not None:
                raise ValueError(
                    "SonicMoELayer does not support flattened-range MoE fc1 checkpointing."
                )
            expert_offsets = (*sharded_offsets, (prepend_axis_num, expert_idx, num_global_experts))
            expert_prepend_axis_num = prepend_axis_num + 1
            gate_weight, up_weight = torch.chunk(tensor, 2, dim=0)
            return [
                ShardedTensor.from_rank_offsets(
                    key,
                    gate_weight.contiguous(),
                    *expert_offsets,
                    (expert_prepend_axis_num, expt_tp_rank, expt_tp_size * 2),
                    replica_id=replica_id,
                    prepend_axis_num=expert_prepend_axis_num,
                ),
                ShardedTensor.from_rank_offsets(
                    key,
                    up_weight.contiguous(),
                    *expert_offsets,
                    (expert_prepend_axis_num, expt_tp_size + expt_tp_rank, expt_tp_size * 2),
                    replica_id=replica_id,
                    prepend_axis_num=expert_prepend_axis_num,
                ),
            ]

        def fc1_whole_build_fn(key, tensor, replica_id, flattened_range):
            # tensor is the WHOLE fused fc1 param (or an optimizer state reshaped to it),
            # shape [num_local_experts, 2 * moe_ffn_hidden_size, hidden]. Emit per-expert
            # gate/up shards by reusing the (verified) per-expert layout, so the on-disk
            # ckpt bytes are byte-identical to the previous per-expert-factory format.
            if flattened_range is not None:
                raise ValueError(
                    "SonicMoELayer optimizer checkpointing requires the fully-reshardable "
                    "format (--dist-ckpt-optim-fully-reshardable); the flattened-range "
                    "(dp_reshardable) path is not supported for sonic fused experts."
                )
            shards = []
            for local_idx, expert_idx in enumerate(self.local_expert_indices):
                shards.extend(fc1_build_fn(key, tensor[local_idx], replica_id, None, expert_idx))
            return shards

        def fc1_merge_fn(sub_state_dict):
            # Inverse of fc1_whole_build_fn: sub_state_dict is the flat list
            # [gate_0, up_0, gate_1, up_1, ...] (2 entries per local expert). Reassemble
            # into sonic's native fused layout [num_local_experts, 2 * ffn, hidden].
            experts = []
            for local_idx in range(len(self.local_expert_indices)):
                gate = sub_state_dict[2 * local_idx]
                up = sub_state_dict[2 * local_idx + 1]
                experts.append(torch.cat((gate, up), dim=-2))
            return torch.stack(experts, dim=0).contiguous()

        def fc2_build_fn(key, tensor, replica_id, flattened_range, expert_idx):
            if flattened_range is not None:
                raise ValueError(
                    "SonicMoELayer does not support flattened-range MoE fc2 checkpointing."
                )
            return [
                ShardedTensor.from_rank_offsets(
                    key,
                    tensor.contiguous(),
                    *sharded_offsets,
                    (prepend_axis_num, expert_idx, num_global_experts),
                    (prepend_axis_num + 2, expt_tp_rank, expt_tp_size),
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num + 1,
                )
            ]

        def fc2_whole_build_fn(key, tensor, replica_id, flattened_range):
            # tensor is the WHOLE fused fc2 param (or an optimizer state reshaped to it),
            # shape [num_local_experts, hidden, moe_ffn_hidden_size].
            if flattened_range is not None:
                raise ValueError(
                    "SonicMoELayer optimizer checkpointing requires the fully-reshardable "
                    "format (--dist-ckpt-optim-fully-reshardable); the flattened-range "
                    "(dp_reshardable) path is not supported for sonic fused experts."
                )
            shards = []
            for local_idx, expert_idx in enumerate(self.local_expert_indices):
                shards.extend(fc2_build_fn(key, tensor[local_idx], replica_id, None, expert_idx))
            return shards

        def fc2_merge_fn(sub_state_dict):
            # Inverse of fc2_whole_build_fn: one entry per local expert (no gate/up split).
            return torch.stack(list(sub_state_dict), dim=0).contiguous()

        sharded_state_dict = {
            f"{prefix}router.weight": make_sharded_tensor_for_checkpoint(
                self._router_weight(),
                f"{prefix}router.weight",
                prepend_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=dp_cp_group,
            )
        }
        router_bias = self._router_bias()
        if router_bias is not None:
            sharded_state_dict[f"{prefix}router.bias"] = make_sharded_tensor_for_checkpoint(
                router_bias,
                f"{prefix}router.bias",
                prepend_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=dp_cp_group,
            )

        # Register the fused fc1/fc2 experts as WHOLE-parameter factories whose `data` is
        # the registered c_fc_weight / c_proj_weight param. This is what lets
        # DistributedOptimizer's param_to_sharded_metadata resolve these params (it keys by
        # the exact param tensor), so optimizer-state resharding works in the
        # fully_reshardable format. The build_fn emits the same per-expert gate/up shards as
        # the previous per-expert factories, so the on-disk model-weight layout is unchanged
        # (and the single `experts.linear_fc1.weight` key is handled on load by
        # `_move_or_convert_packed_if_present`, which direct-moves the [E,2ffn,h] tensor).
        sharded_state_dict[f"{prefix}experts.linear_fc1.weight"] = ShardedTensorFactory(
            f"{prefix}experts.experts.linear_fc1.weight",
            self.sonic_moe.c_fc.weight,
            fc1_whole_build_fn,
            fc1_merge_fn,
            replica_id,
        )
        sharded_state_dict[f"{prefix}experts.linear_fc2.weight"] = ShardedTensorFactory(
            f"{prefix}experts.experts.linear_fc2.weight",
            self.sonic_moe.c_proj.weight,
            fc2_whole_build_fn,
            fc2_merge_fn,
            replica_id,
        )

        for expert_idx in self.local_expert_indices:
            suffix = "" if expert_idx == 0 else str(expert_idx)
            # Keep the sharded checkpoint schema invariant across EP sizes. These empty
            # placeholders are required when an EP=1 checkpoint is loaded with EP>1.
            sharded_state_dict[f"{prefix}experts.linear_fc1._extra_state{suffix}"] = (
                make_sharded_object_for_checkpoint(
                    torch.empty((0,), dtype=torch.uint8),
                    f"{prefix}experts.experts.linear_fc1._extra_state",
                    (*sharded_offsets, (prepend_axis_num, expert_idx, num_global_experts)),
                    replica_id=replica_id,
                )
            )
            sharded_state_dict[f"{prefix}experts.linear_fc2._extra_state{suffix}"] = (
                make_sharded_object_for_checkpoint(
                    torch.empty((0,), dtype=torch.uint8),
                    f"{prefix}experts.experts.linear_fc2._extra_state",
                    (*sharded_offsets, (prepend_axis_num, expert_idx, num_global_experts)),
                    replica_id=replica_id,
                )
            )

            if self.sonic_moe.c_fc.bias is not None:
                name = f"experts.linear_fc1.bias{expert_idx}"
                local_expert_idx = self._local_expert_index(expert_idx)
                sharded_state_dict[f"{prefix}{name}"] = make_sharded_tensor_for_checkpoint(
                    self.sonic_moe.c_fc.bias[local_expert_idx],
                    f"{prefix}{name}",
                    prepend_offsets=sharded_offsets,
                    tp_group=self.expt_tp_group,
                    dp_cp_group=self.expt_dp_group,
                )
            if self.sonic_moe.c_proj.bias is not None:
                name = f"experts.linear_fc2.bias{expert_idx}"
                local_expert_idx = self._local_expert_index(expert_idx)
                sharded_state_dict[f"{prefix}{name}"] = make_sharded_tensor_for_checkpoint(
                    self.sonic_moe.c_proj.bias[local_expert_idx],
                    f"{prefix}{name}",
                    prepend_offsets=sharded_offsets,
                    tp_group=self.expt_tp_group,
                    dp_cp_group=self.expt_dp_group,
                )

        if self.shared_experts is not None:
            sharded_state_dict.update(
                self.shared_experts.sharded_state_dict(
                    f"{prefix}shared_experts.", sharded_offsets, metadata
                )
            )

        # The router owns hash-routing `tid2eid` and the aux-free `expert_bias`. The manual
        # router.weight/bias entries above omit them, so they were never checkpoint-loaded
        # (tid2eid stayed at its init -1 -> the sqrtsoftplus hash-routing assert fired). Pull
        # them from the router's own sharded_state_dict so the keys/format match the standard
        # MoELayer (which uses the default recursive sharded_state_dict) and the checkpoint.
        # tid2eid/expert_bias live only on the Megatron TopKRouter (megatron-dispatch path, e.g.
        # DSV4). In the EP1 sonic fast-path self.router is None (routing is fused inside
        # sonic_moe.router) and there is nothing to pull — guard it, else sharded_state_dict
        # AttributeErrors at EP1 (Qwen EP1 + global_aux_loss).
        if self.router is not None:
            router_sd = self.router.sharded_state_dict(
                prefix=f"{prefix}router.", sharded_offsets=sharded_offsets, metadata=metadata
            )
            for _rk in (f"{prefix}router.tid2eid", f"{prefix}router.expert_bias"):
                if _rk in router_sd:
                    sharded_state_dict[_rk] = router_sd[_rk]

        return sharded_state_dict

    def _move_if_present(self, state_dict, src_key: str, dst_key: str) -> None:
        if dst_key not in state_dict and src_key in state_dict:
            state_dict[dst_key] = state_dict.pop(src_key)

    def _stack_per_expert_if_present(
        self, state_dict, prefix: str, src_prefix: str, dst_key: str
    ) -> None:
        if dst_key in state_dict:
            return
        keys = [f"{prefix}{src_prefix}{expert_idx}" for expert_idx in self.local_expert_indices]
        if all(key in state_dict for key in keys):
            state_dict[dst_key] = torch.stack([state_dict.pop(key) for key in keys], dim=0)

    def _stack_local_experts_if_present(
        self, state_dict, prefix: str, component: str, dst_key: str
    ) -> None:
        if dst_key in state_dict:
            return
        keys = [
            f"{prefix}experts.local_experts.{local_idx}.{component}"
            for local_idx in range(self.num_local_experts)
        ]
        if all(key in state_dict for key in keys):
            state_dict[dst_key] = torch.stack([state_dict.pop(key) for key in keys], dim=0)

    def _move_or_convert_packed_if_present(
        self, state_dict, src_key: str, dst_key: str, weight_type: str
    ) -> None:
        if dst_key in state_dict or src_key not in state_dict:
            return
        weight = state_dict.pop(src_key)
        if weight_type == "fc1":
            if weight.shape[0] == self.config.hidden_size and (
                weight.shape[1] % (2 * self.config.moe_ffn_hidden_size) == 0
            ):
                num_experts = weight.shape[1] // (2 * self.config.moe_ffn_hidden_size)
                weight = (
                    weight.view(
                        num_experts, self.config.hidden_size, 2 * self.config.moe_ffn_hidden_size
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
                if num_experts == self.config.num_moe_experts:
                    weight = weight[
                        self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
                    ]
        elif weight_type == "fc2":
            if weight.shape[1] == self.config.hidden_size and (
                weight.shape[0] % self.config.moe_ffn_hidden_size == 0
            ):
                num_experts = weight.shape[0] // self.config.moe_ffn_hidden_size
                weight = (
                    weight.view(
                        num_experts, self.config.moe_ffn_hidden_size, self.config.hidden_size
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
                if num_experts == self.config.num_moe_experts:
                    weight = weight[
                        self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
                    ]
        else:
            raise ValueError(f"Unexpected packed weight type: {weight_type}")
        state_dict[dst_key] = weight

    def _merge_legacy_grouped_mlp_state_dict(self, state_dict, prefix: str) -> None:
        weight1_key = f"{prefix}experts.weight1"
        weight2_key = f"{prefix}experts.weight2"
        if f"{prefix}sonic_moe.c_fc.weight" not in state_dict and weight1_key in state_dict:
            weight1 = state_dict.pop(weight1_key)
            state_dict[f"{prefix}sonic_moe.c_fc.weight"] = (
                weight1.view(
                    self.num_local_experts,
                    self.config.hidden_size,
                    2 * self.config.moe_ffn_hidden_size,
                )
                .transpose(1, 2)
                .contiguous()
            )
        if f"{prefix}sonic_moe.c_proj.weight" not in state_dict and weight2_key in state_dict:
            weight2 = state_dict.pop(weight2_key)
            state_dict[f"{prefix}sonic_moe.c_proj.weight"] = (
                weight2.view(
                    self.num_local_experts, self.config.moe_ffn_hidden_size, self.config.hidden_size
                )
                .transpose(1, 2)
                .contiguous()
            )

    # pylint: disable=arguments-differ
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        router_weight_key = (
            f"{prefix}router.weight"
            if self._use_megatron_dispatch
            else f"{prefix}sonic_moe.router.weight"
        )
        self._move_if_present(state_dict, f"{prefix}router.weight", router_weight_key)
        router_bias = self._router_bias()
        if router_bias is None:
            state_dict.pop(f"{prefix}router.bias", None)
        else:
            router_bias_key = (
                f"{prefix}router.bias"
                if self._use_megatron_dispatch
                else f"{prefix}sonic_moe.router.bias"
            )
            self._move_if_present(state_dict, f"{prefix}router.bias", router_bias_key)
        for expert_idx in range(self.config.num_moe_experts):
            suffix = "" if expert_idx == 0 else str(expert_idx)
            state_dict.pop(f"{prefix}experts.linear_fc1._extra_state{suffix}", None)
            state_dict.pop(f"{prefix}experts.linear_fc2._extra_state{suffix}", None)
            state_dict.pop(f"{prefix}experts.experts.linear_fc1._extra_state{suffix}", None)
            state_dict.pop(f"{prefix}experts.experts.linear_fc2._extra_state{suffix}", None)
        for local_idx in range(self.num_local_experts):
            state_dict.pop(
                f"{prefix}experts.local_experts.{local_idx}.linear_fc1._extra_state", None
            )
            state_dict.pop(
                f"{prefix}experts.local_experts.{local_idx}.linear_fc2._extra_state", None
            )

        self._stack_local_experts_if_present(
            state_dict, prefix, "linear_fc1.weight", f"{prefix}sonic_moe.c_fc.weight"
        )
        self._stack_local_experts_if_present(
            state_dict, prefix, "linear_fc1.bias", f"{prefix}sonic_moe.c_fc.bias"
        )
        self._stack_local_experts_if_present(
            state_dict, prefix, "linear_fc2.weight", f"{prefix}sonic_moe.c_proj.weight"
        )
        self._stack_local_experts_if_present(
            state_dict, prefix, "linear_fc2.bias", f"{prefix}sonic_moe.c_proj.bias"
        )

        self._move_or_convert_packed_if_present(
            state_dict,
            f"{prefix}experts.linear_fc1.weight",
            f"{prefix}sonic_moe.c_fc.weight",
            "fc1",
        )
        self._move_or_convert_packed_if_present(
            state_dict,
            f"{prefix}experts.experts.linear_fc1.weight",
            f"{prefix}sonic_moe.c_fc.weight",
            "fc1",
        )
        self._move_if_present(
            state_dict, f"{prefix}experts.linear_fc1.bias", f"{prefix}sonic_moe.c_fc.bias"
        )
        self._move_if_present(
            state_dict, f"{prefix}experts.experts.linear_fc1.bias", f"{prefix}sonic_moe.c_fc.bias"
        )
        self._move_or_convert_packed_if_present(
            state_dict,
            f"{prefix}experts.linear_fc2.weight",
            f"{prefix}sonic_moe.c_proj.weight",
            "fc2",
        )
        self._move_or_convert_packed_if_present(
            state_dict,
            f"{prefix}experts.experts.linear_fc2.weight",
            f"{prefix}sonic_moe.c_proj.weight",
            "fc2",
        )
        self._move_if_present(
            state_dict, f"{prefix}experts.linear_fc2.bias", f"{prefix}sonic_moe.c_proj.bias"
        )
        self._move_if_present(
            state_dict, f"{prefix}experts.experts.linear_fc2.bias", f"{prefix}sonic_moe.c_proj.bias"
        )
        self._stack_per_expert_if_present(
            state_dict, prefix, "experts.linear_fc1.weight", f"{prefix}sonic_moe.c_fc.weight"
        )
        self._stack_per_expert_if_present(
            state_dict,
            prefix,
            "experts.experts.linear_fc1.weight",
            f"{prefix}sonic_moe.c_fc.weight",
        )
        self._stack_per_expert_if_present(
            state_dict, prefix, "experts.linear_fc1.bias", f"{prefix}sonic_moe.c_fc.bias"
        )
        self._stack_per_expert_if_present(
            state_dict, prefix, "experts.experts.linear_fc1.bias", f"{prefix}sonic_moe.c_fc.bias"
        )
        self._stack_per_expert_if_present(
            state_dict, prefix, "experts.linear_fc2.weight", f"{prefix}sonic_moe.c_proj.weight"
        )
        self._stack_per_expert_if_present(
            state_dict,
            prefix,
            "experts.experts.linear_fc2.weight",
            f"{prefix}sonic_moe.c_proj.weight",
        )
        self._stack_per_expert_if_present(
            state_dict, prefix, "experts.linear_fc2.bias", f"{prefix}sonic_moe.c_proj.bias"
        )
        self._stack_per_expert_if_present(
            state_dict, prefix, "experts.experts.linear_fc2.bias", f"{prefix}sonic_moe.c_proj.bias"
        )
        self._merge_legacy_grouped_mlp_state_dict(state_dict, prefix)

        if self._use_megatron_dispatch:
            # The internal Sonic router is unused in EP mode because Megatron's
            # TopKRouter owns routing and checkpoint state. Keep strict module
            # loading happy without exposing Sonic router keys in checkpoints.
            state_dict.setdefault(
                f"{prefix}sonic_moe.router.weight", self.sonic_moe.router.weight.detach()
            )
            if self.sonic_moe.router.bias is not None:
                state_dict.setdefault(
                    f"{prefix}sonic_moe.router.bias", self.sonic_moe.router.bias.detach()
                )

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


def get_sonic_moe_module_spec(submodules: Optional[MoESubmodules] = None) -> ModuleSpec:
    """Return an MLP/MoE ModuleSpec for use in a TransformerLayer spec."""
    return ModuleSpec(
        module=SonicMoELayer, submodules=submodules, metainfo={"fuse_pre_mlp_layernorm": False}
    )


def replace_moe_layer_specs_with_sonic_moe(transformer_layer_spec) -> int:
    """Replace Megatron MoELayer MLP specs with SonicMoELayer specs in-place.

    Accepts a TransformerBlockSubmodules object, a TransformerLayer ModuleSpec,
    or a list/tuple of layer specs. Returns the number of MoE MLP specs that
    use SonicMoELayer after the replacement.
    """
    import os

    if os.environ.get("DISABLE_SONIC_MOE", "0") == "1":
        # Fall back to standard Megatron MoELayer (no sonic-moe swap).
        return 0
    if transformer_layer_spec is None:
        return 0
    if isinstance(transformer_layer_spec, (list, tuple)):
        return sum(replace_moe_layer_specs_with_sonic_moe(spec) for spec in transformer_layer_spec)

    layer_specs = getattr(transformer_layer_spec, "layer_specs", None)
    if layer_specs is not None:
        return replace_moe_layer_specs_with_sonic_moe(layer_specs)

    submodules = getattr(transformer_layer_spec, "submodules", None)
    if submodules is None:
        return 0

    mlp_spec = getattr(submodules, "mlp", None)
    if isinstance(mlp_spec, functools.partial):
        module = mlp_spec.func
        old_moe_submodules = mlp_spec.keywords.get("submodules")
    else:
        module = getattr(mlp_spec, "module", None)
        old_moe_submodules = getattr(mlp_spec, "submodules", None)
    if module is SonicMoELayer:
        return 1
    if module is not MoELayer:
        return 0

    sonic_submodules = None
    if old_moe_submodules is not None:
        sonic_submodules = MoESubmodules(
            experts=None,
            shared_experts=old_moe_submodules.shared_experts,
            router=old_moe_submodules.router,
        )
    if isinstance(mlp_spec, functools.partial):
        submodules.mlp = functools.partial(SonicMoELayer, submodules=sonic_submodules)
    else:
        submodules.mlp = get_sonic_moe_module_spec(submodules=sonic_submodules)
    return 1
