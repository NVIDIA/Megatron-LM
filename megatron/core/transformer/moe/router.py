# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_observation import is_observing_tensor, observe_tensor
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    is_batch_invariant_mode_enabled,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP_DENSE_ROUTING
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    ProcessGroupCollection,
    apply_biased_logits,
    apply_random_logits,
    apply_router_token_dropping,
    compute_normalized_router_scores,
    compute_routing_scores_for_aux_loss,
    fused_topk_with_score_function_supports_topk_indices,
    get_tokens_per_expert_and_token_count,
    router_gating_linear,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
)
from megatron.core.transformer.moe.router_diagnostics import build_router_diagnostics
from megatron.core.transformer.moe.router_replay import RouterReplay
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_HYBRIDEP_INT16_EXPERT_LIMIT = 1 << 15


@dataclass(frozen=True)
class _AuxLossGroupConfig:
    """Process groups for local aux/seq-aux loss and its metric logging."""

    loss_reduce_groups: Sequence[torch.distributed.ProcessGroup]
    metric_reduce_group: Optional[torch.distributed.ProcessGroup]
    metric_avg_group: Optional[torch.distributed.ProcessGroup]
    metric_needs_dp_avg: bool

    @property
    def metric_pre_reduce_groups(self) -> Optional[Sequence[torch.distributed.ProcessGroup]]:
        """Groups to reduce eagerly before recording metrics, if tracker reduction is unsafe."""
        return self.loss_reduce_groups if self.metric_avg_group is not None else None


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        layer_number: Optional[int] = None,
        hash_moe_layer_threshold: Optional[int] = None,
    ) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
            is_mtp_layer (bool): Flag indicating if this router is part of an MTP layer.
            hash_moe_layer_threshold (int, optional): Explicit layer-number threshold for
                hash routing. When omitted, use config.moe_n_hash_layers.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        # An explicit threshold is a *global* layer number (hybrid models translate their
        # MoE-position count into one). When omitted, fall back to the config's MoE-position
        # count. Remember which of the two applies so set_layer_number() can reject the
        # ambiguous hybrid case, where the config count is not a global layer number.
        self._explicit_hash_moe_layer_threshold = hash_moe_layer_threshold
        self.hash_moe_layer_threshold = (
            self.config.moe_n_hash_layers
            if hash_moe_layer_threshold is None
            else hash_moe_layer_threshold
        )
        self.tp_group = pg_collection.tp
        self.expt_tp_group = pg_collection.expt_tp
        self.cp_group = pg_collection.cp
        self.tp_cp_group = pg_collection.tp_cp
        self.tp_dp_cp_group = pg_collection.tp_dp_cp

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if self.config.add_bias_linear:
            self.bias = torch.nn.Parameter(
                torch.empty((self.config.num_moe_experts), dtype=torch.float32)
            )
        else:
            self.bias = None
        if self.config.moe_router_skip_muon:
            setattr(self.weight, 'use_muon', False)
            if self.bias is not None:
                setattr(self.bias, 'use_muon', False)
        # If calculate per token loss, we need to scale up moe aux loss by the number of tokens.
        # So we need to know if the model is configured to calculate per token loss.
        self.calculate_per_token_loss = self.config.calculate_per_token_loss
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the router parameters."""
        if self.config.perform_initialization:
            self.config.init_method(self.weight)
            if self.bias is not None:
                self.config.init_method(self.bias)
        self.weight.data = self.weight.data.to(dtype=self.config.params_dtype)
        setattr(self.weight, 'sequence_parallel', self.config.sequence_parallel)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype=self.config.params_dtype)
            setattr(self.bias, 'sequence_parallel', self.config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        if self.bias is not None and self.bias.device.type == 'cpu':
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number
        if getattr(self, "router_replay", None) is not None:
            self.router_replay.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts.

    The workflow of TopKRouter is as follows:
    (1) Calculate the logits by the router gating network.
    (2) Calculate the routing probabilities and map for top-k selection with score function.
    (3) [Optional] Apply token dropping to top-k expert selection.
    (4) [Optional] Apply the auxiliary load balancing loss for the given scores and routing map.

    Naming convention:
        logits: The output logits by the router gating network.
        scores: The scores after score function used to select the experts and calculate aux loss.
        probs: The topk weights used to combined the experts' outputs.
        routing_map: The masked routing map between tokens and experts.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        layer_number: Optional[int] = None,
        hash_moe_layer_threshold: Optional[int] = None,
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
            is_mtp_layer (bool): Flag indicating if this router is part of an MTP layer.
            hash_moe_layer_threshold (int, optional): Explicit layer-number threshold for
                hash routing. Hybrid models use this to translate a MoE-position count.
        """
        super().__init__(
            config=config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            layer_number=layer_number,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
        )
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None
        self.frozen_expert_bias = False
        self.mtp_layer_number: Optional[int] = None
        self.is_hash_layer = False
        self.register_buffer('tid2eid', None)

        # Hash-layer selection needs a layer number. Callers that build the router without
        # one (or that only learn it later) resolve it in set_layer_number() instead.
        if layer_number is not None:
            self._resolve_hash_layer(layer_number)

        self.use_quantile_balancing = (
            self.routing_type == "quantile_balancing" and not self.is_hash_layer
        )
        self.enable_expert_bias = (
            self.config.moe_router_enable_expert_bias and not self.is_hash_layer
        )
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    # Keep token counts exact through accumulation and cross-rank reduction.
                    # Float16Module leaves integer buffers in their original dtype.
                    dtype=torch.int64,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
        else:
            self.local_tokens_per_expert = None

        if self.enable_expert_bias or self.use_quantile_balancing:
            self.register_buffer(
                'expert_bias',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
            )
        else:
            self.expert_bias = None

        if self.use_quantile_balancing:
            self.register_buffer(
                'qb_histogram',
                torch.zeros(
                    self.config.num_moe_experts,
                    self.config.moe_router_qb_num_bins,
                    dtype=torch.int32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'qb_bin_bounds',
                torch.tensor([-1.0, 1.0], dtype=torch.float32, device=torch.cuda.current_device()),
            )
        else:
            self.qb_histogram = None
            self.qb_bin_bounds = None

        # Initialize global tokens per expert for global aux loss
        if self.get_aux_loss_coeff("global_aux_loss") > 0:
            self.register_buffer(
                'global_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'ga_steps',
                torch.tensor(0, dtype=torch.float32, device=torch.cuda.current_device()),
                persistent=False,
            )
        else:
            self.global_tokens_per_expert = None
            self.ga_steps = None

        self.router_replay = None
        if self.config.moe_enable_routing_replay:
            self.router_replay = RouterReplay()

    def set_layer_number(self, layer_number: int):
        """Set the layer number and initialize hash routing for eligible layers."""
        super().set_layer_number(layer_number)
        self._resolve_hash_layer(layer_number)

    def _resolve_hash_layer(self, layer_number: int) -> None:
        """Decide whether this layer is hash-routed and initialize its lookup table.

        Idempotent: safe to call from ``__init__`` (when the layer number is already known)
        and again from ``set_layer_number``.
        """
        hash_moe_layer_threshold = self.hash_moe_layer_threshold
        if self._explicit_hash_moe_layer_threshold is None:
            # Without an explicit threshold, the value came from the config and counts MoE
            # positions rather than global layers. Hybrid models interleave non-MoE layers,
            # so they must translate the count into a global layer number themselves.
            if (
                self.config.is_hybrid_model
                and hash_moe_layer_threshold > 0
                and not self.is_mtp_layer
            ):
                raise ValueError("Hybrid hash routing requires an explicit global layer threshold.")
        self.is_hash_layer = (
            not self.is_mtp_layer
            and hash_moe_layer_threshold > 0
            and layer_number <= hash_moe_layer_threshold
        )
        if self.is_hash_layer:
            self._initialize_hash_routing()

    def _initialize_hash_routing(self) -> None:
        """Initialize the hash lookup table and disable learned-routing expert bias."""
        if self.tid2eid is None:
            log_single_rank(
                logger,
                logging.WARNING,
                "Hash MoE initialized with a placeholder round-robin token-to-expert table. "
                "Load a trained table from a checkpoint or provide a workload-aware "
                "initialization before training.",
            )
            token_ids = torch.arange(self.config.hash_moe_vocab_size, device=self.weight.device)
            expert_offsets = torch.arange(self.topk, device=token_ids.device)
            self.tid2eid = ((token_ids[:, None] + expert_offsets) % self.num_experts).to(
                torch.int32
            )

        # Dynamic expert bias applies to learned top-k selection, not a fixed lookup table.
        self.enable_expert_bias = False
        self.local_tokens_per_expert = None
        self.expert_bias = None

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)
        if hasattr(self, 'qb_bin_bounds') and self.qb_bin_bounds is not None:
            if self.qb_bin_bounds.dtype != torch.float32:
                self.qb_bin_bounds.data = self.qb_bin_bounds.data.to(torch.float32)

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def get_aux_loss_coeff(self, aux_loss_type: str) -> float:
        """Return the aux loss coeff for the given auxiliary loss type.
        If the auxiliary loss type is not found, return 0.0.
        """
        if isinstance(self.routing_type, str):
            if self.routing_type == aux_loss_type:
                return self.config.moe_aux_loss_coeff
        if isinstance(self.routing_type, list):
            try:
                idx = self.routing_type.index(aux_loss_type)
                return self.config.moe_aux_loss_coeff[idx]
            except ValueError:
                return 0.0
        return 0.0

    def is_aux_loss_enabled(self) -> bool:
        """Check if the auxiliary loss is enabled."""
        for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
            if self.get_aux_loss_coeff(aux_loss_type) > 0:
                return True
        return False

    def _dense_route_indices_dtype(self) -> Optional[torch.dtype]:
        """Return the route-index dtype for Flex backends that consume dense top-k indices."""
        if not self.config.moe_router_fusion:
            return None
        if self.config.moe_token_dispatcher_type != "flex":
            return None
        if self.config.moe_expert_capacity_factor is not None:
            return None
        if not fused_topk_with_score_function_supports_topk_indices:
            return None

        backend = self.config.moe_flex_dispatcher_backend
        if backend in ("deepep", "deepepv2", "ncclep"):
            return torch.int64
        if backend != "hybridep":
            return None
        if self.config.moe_hybridep_routing_map_mode != "indices":
            return None
        if not HAVE_HYBRIDEP_DENSE_ROUTING:
            return None

        num_experts = self.expt_tp_group.size() * self.config.num_moe_experts
        if num_experts <= _HYBRIDEP_INT16_EXPERT_LIMIT:
            return torch.int16
        return None

    def _apply_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        with_padding_mask: bool = False,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Apply the auxiliary loss for the given scores and routing map."""
        aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
        if aux_loss_coeff == 0:
            return probs

        aux_loss_groups = self._get_aux_loss_groups(packed_seq_params)
        global_tokens_per_expert, local_num_tokens, total_num_tokens = (
            get_tokens_per_expert_and_token_count(
                routing_map=routing_map,
                reduce_group=aux_loss_groups.loss_reduce_groups[0],
                reduce_groups=aux_loss_groups.loss_reduce_groups,
                topk=self.topk,
                with_padding_mask=with_padding_mask,
            )
        )

        aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=global_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=aux_loss_coeff,
            fused=self.config.moe_router_aux_loss_fusion,
        )

        probs = self.attach_and_log_load_balancing_loss(
            probs,
            aux_loss_coeff,
            aux_loss,
            "load_balancing_loss",
            aux_loss_groups.metric_reduce_group,
            avg_group=aux_loss_groups.metric_avg_group,
            needs_dp_avg=aux_loss_groups.metric_needs_dp_avg,
            valid_token_count=local_num_tokens,
            aux_loss_logging_reduce_groups=aux_loss_groups.metric_pre_reduce_groups,
            aux_loss_scale_reduce_groups=aux_loss_groups.loss_reduce_groups,
            aux_loss_scale_num_tokens=total_num_tokens,
        )
        return probs

    def _apply_seq_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        seq_length: int,
        bsz: int,
        with_padding_mask: bool = False,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Apply the sequence-level auxiliary loss for the given scores and routing map.

        To calculate the sequence-level aux loss, we reshape the batch_size dimension to
        experts dimension. The resulted loss by switch_load_balancing_loss_func is equal
        to the sum of aux loss for each sequence in the batch. And then we divide the aux
        loss by the batch size to get averaged aux loss.
        """
        seq_aux_loss_coeff = self.get_aux_loss_coeff("seq_aux_loss")
        if seq_aux_loss_coeff == 0:
            return probs

        scores_for_aux_loss = scores_for_aux_loss.reshape(seq_length, -1)
        routing_map = routing_map.reshape(seq_length, -1)

        aux_loss_groups = self._get_aux_loss_groups(packed_seq_params)
        global_tokens_per_expert, local_num_tokens, total_num_tokens = (
            get_tokens_per_expert_and_token_count(
                routing_map=routing_map,
                reduce_group=aux_loss_groups.loss_reduce_groups[0],
                reduce_groups=aux_loss_groups.loss_reduce_groups,
                with_padding_mask=with_padding_mask,
                topk=self.topk * bsz,
            )
        )

        aux_loss = (
            switch_load_balancing_loss_func(
                probs=scores_for_aux_loss,
                tokens_per_expert=global_tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.topk,
                num_experts=self.config.num_moe_experts,
                moe_aux_loss_coeff=seq_aux_loss_coeff,
                fused=self.config.moe_router_aux_loss_fusion,
            )
            / bsz
        )

        probs = self.attach_and_log_load_balancing_loss(
            probs,
            seq_aux_loss_coeff,
            aux_loss,
            "seq_load_balancing_loss",
            aux_loss_groups.metric_reduce_group,
            avg_group=aux_loss_groups.metric_avg_group,
            needs_dp_avg=aux_loss_groups.metric_needs_dp_avg,
            # local_num_tokens is per-sequence (bsz is folded into the expert dimension);
            # restore the micro-batch total for per-token-loss gradient scaling.
            valid_token_count=local_num_tokens * bsz,
            aux_loss_logging_reduce_groups=aux_loss_groups.metric_pre_reduce_groups,
            aux_loss_scale_reduce_groups=aux_loss_groups.loss_reduce_groups,
        )
        return probs

    def _apply_global_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        with_padding_mask: bool = False,
    ):
        """Apply the global auxiliary loss for the given scores and routing map."""
        global_aux_loss_coeff = self.get_aux_loss_coeff("global_aux_loss")
        if global_aux_loss_coeff == 0:
            return probs

        # Global aux loss intentionally uses the full static TP x DP x CP domain.
        # Dynamic CP subgroups only affect local aux/seq-aux domains.
        global_tokens_per_expert, local_num_tokens, total_num_tokens = (
            get_tokens_per_expert_and_token_count(
                routing_map=routing_map,
                reduce_group=self.tp_dp_cp_group,
                with_padding_mask=with_padding_mask,
                topk=self.topk,
            )
        )
        self.global_tokens_per_expert += global_tokens_per_expert
        self.ga_steps += 1
        averated_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps

        global_aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=averated_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=global_aux_loss_coeff,
            fused=self.config.moe_router_aux_loss_fusion,
        )

        probs = self.attach_and_log_load_balancing_loss(
            probs,
            global_aux_loss_coeff,
            global_aux_loss,
            "global_load_balancing_loss",
            self.tp_dp_cp_group,
            needs_dp_avg=False,
            valid_token_count=local_num_tokens,
            # The global aux-loss statistics/logging domain is TP x DP x CP, but
            # per-token-loss gradient normalization already reduces the denominator
            # across DP x CP in finalize_model_grads.  Scale the aux-loss numerator
            # over TP x CP only, matching the original static behavior while still
            # using an exact valid-token count when padding is present.
            aux_loss_scale_reduce_groups=(self.tp_cp_group,),
        )
        return probs

    def _get_aux_loss_groups(
        self, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> _AuxLossGroupConfig:
        """Return process groups for MoE aux-loss statistics and logging."""
        if (
            packed_seq_params is not None
            and packed_seq_params.local_cp_size is not None
            and packed_seq_params.cp_group is not None
        ):
            return _AuxLossGroupConfig(
                loss_reduce_groups=(packed_seq_params.cp_group, self.tp_group),
                metric_reduce_group=None,
                metric_avg_group=self.tp_dp_cp_group,
                metric_needs_dp_avg=False,
            )

        return _AuxLossGroupConfig(
            loss_reduce_groups=(self.tp_cp_group,),
            metric_reduce_group=self.tp_cp_group,
            metric_avg_group=None,
            metric_needs_dp_avg=True,
        )

    def _get_metric_layer_number(self) -> tuple[int, int]:
        """Return the 1-based metric index and the total number of metric slots."""
        num_layers = self.config.num_layers + (self.config.mtp_num_layers or 0)
        if not self.is_mtp_layer:
            return self.layer_number, num_layers

        # Hybrid MTP depths can contain multiple internal sublayers (for example `/WE`).
        # Metrics are allocated per MTP depth, not per internal hybrid sublayer.
        mtp_layer_number = self.mtp_layer_number or self.layer_number
        if self.config.mtp_num_layers is not None:
            mtp_layer_number = min(mtp_layer_number, self.config.mtp_num_layers)
        return self.config.num_layers + mtp_layer_number, num_layers

    def attach_and_log_load_balancing_loss(
        self,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss: torch.Tensor,
        aux_loss_name: str,
        reduce_group: Optional[torch.distributed.ProcessGroup],
        avg_group: Optional[torch.distributed.ProcessGroup] = None,
        needs_dp_avg: bool = True,
        valid_token_count: Optional[Union[int, torch.Tensor]] = None,
        aux_loss_logging_reduce_groups: Optional[Sequence[torch.distributed.ProcessGroup]] = None,
        aux_loss_scale_reduce_groups: Optional[Sequence[torch.distributed.ProcessGroup]] = None,
        aux_loss_scale_num_tokens: Optional[Union[int, torch.Tensor]] = None,
    ):
        """Attach aux loss function to activation and add to logging.

        Args:
            activation (torch.Tensor): Activation tensor to attach the aux loss to.
            aux_loss_coeff (float): Coefficient for the aux loss.
            aux_loss (torch.Tensor): Computed aux loss.
            aux_loss_name (str): Name of the aux loss for logging.
            reduce_group (torch.distributed.ProcessGroup, optional): Process group for deferred
                logging reduction.
            avg_group (torch.distributed.ProcessGroup, optional): Process group for deferred
                logging average.
            needs_dp_avg (bool): Whether to average this metric across DP ranks after reduce_group.
            valid_token_count (int or torch.Tensor, optional): Number of valid tokens excluding
                padding tokens. Can be a Python int or a torch.Tensor (typically 0-d tensor).
                If None, uses activation.shape[0]. Defaults to None.
        """
        # When using repeated MTP layers, the loss is counted "mtp_num_layers" times.
        # To avoid accumulating the load balancing loss multiple times, we scale it by
        # 1/mtp_num_layers so the total loss is correct.
        if (
            self.is_mtp_layer
            and self.config.mtp_use_repeated_layer
            and self.config.mtp_num_layers is not None
        ):
            aux_loss = aux_loss / self.config.mtp_num_layers

        # TODO (zijiey): fix the per_layer_logging for MTP, currently it will incorrectly
        # add the aux loss logging value to other layer's since it is difficult to get the
        # correct layer_number for MTP. It does not affect the correctness of the calculation
        # results and the reduced load_balancing_loss logging value.
        layer_number, num_layers = self._get_metric_layer_number()

        metric_value = aux_loss / aux_loss_coeff
        if aux_loss_logging_reduce_groups is not None:
            metric_value = metric_value.detach().clone()
            for group in aux_loss_logging_reduce_groups:
                torch.distributed.all_reduce(metric_value, group=group)

        get_moe_metrics_tracker().record(
            aux_loss_name,
            metric_value,
            layer_number,
            num_layers,
            reduce_group=reduce_group,
            avg_group=avg_group,
            needs_dp_avg=needs_dp_avg,
        )
        if self.calculate_per_token_loss:
            # --calculate-per-token-loss divides all parameter gradients by the global
            # non-padded token count in finalize_model_grads.  Pre-multiplying by the
            # valid-token count from this aux-loss domain makes the final objective a
            # token-weighted average of per-domain aux losses.  Use the reduced count
            # directly: with THD padding or dynamic CP, valid token counts can differ
            # by rank/group, so local_num_tokens * group_size is not generally correct.
            if aux_loss_scale_num_tokens is None:
                num_local_tokens = (
                    valid_token_count if valid_token_count is not None else activation.shape[0]
                )
                if torch.is_tensor(num_local_tokens):
                    aux_loss_scale_num_tokens = num_local_tokens.clone().to(
                        device=activation.device
                    )
                else:
                    aux_loss_scale_num_tokens = torch.tensor(
                        num_local_tokens, device=activation.device
                    )
                if aux_loss_scale_reduce_groups is None:
                    assert reduce_group is not None, "reduce_group is required for aux-loss scaling"
                    aux_loss_scale_reduce_groups = (reduce_group,)
                for group in aux_loss_scale_reduce_groups:
                    torch.distributed.all_reduce(aux_loss_scale_num_tokens, group=group)
            activation = MoEAuxLossAutoScaler.apply(
                activation, aux_loss * aux_loss_scale_num_tokens
            )
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits, padding_mask: Optional[torch.Tensor] = None):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.
            padding_mask (torch.Tensor, optional): Boolean mask indicating padding positions.
                                                   Shape [num_tokens]. True = padding,
                                                   False = valid. Defaults to None.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training and torch.is_grad_enabled():
            # Skip Z loss calculations when using torch.no_grad() or checkpointing.
            logsum = torch.logsumexp(logits, dim=-1)
            z_loss_values = torch.square(logsum)
            if padding_mask is not None:
                valid_mask = ~padding_mask
                z_loss_values = z_loss_values * valid_mask
                num_local_tokens = valid_mask.sum()
                z_loss_sum = z_loss_values.sum()
                z_loss_mean = z_loss_sum / torch.clamp(num_local_tokens, min=1)
            else:
                z_loss_sum = z_loss_values.sum()
                # Keep the token count as a Python scalar so CUDA graph capture does not
                # record a CPU-to-CUDA tensor creation.
                z_loss_mean = z_loss_sum / max(logits.shape[0], 1)

            # When using repeated MTP layers, the same MTP layer is called mtp_num_layers times.
            # To avoid accumulating the z_loss multiple times, we scale it by 1/mtp_num_layers
            # so the total loss is correct. The scale is resolved before the attach point below,
            # so it affects both the gradient and the logged value.
            mtp_loss_scale = 1
            if (
                self.is_mtp_layer
                and self.config.mtp_use_repeated_layer
                and self.config.mtp_num_layers is not None
            ):
                mtp_loss_scale = self.config.mtp_num_layers

            if self.calculate_per_token_loss:
                # --calculate-per-token-loss divides gradients by the global non-padded
                # token count. Attach the local z-loss numerator directly so the final
                # objective is a token-weighted z-loss over valid tokens.
                z_loss = z_loss_sum * self.config.moe_z_loss_coeff / mtp_loss_scale
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            else:
                moe_z_loss_coeff = self.config.moe_z_loss_coeff / self.tp_cp_group.size()
                z_loss = z_loss_mean * moe_z_loss_coeff / mtp_loss_scale
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss)

            layer_number, num_layers = self._get_metric_layer_number()

            get_moe_metrics_tracker().record(
                "z_loss", z_loss_mean / mtp_loss_scale, layer_number, num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, dtype=input.dtype, device=input.device),
                    torch.tensor(1.0 + eps, dtype=input.dtype, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    @jit_fuser
    def _apply_expert_bias(
        self, routing_map: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Update expert bias and tokens_per_expert
        Prevent extra local tokens accumulation on evaluation or activation recomputation
        """
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                use_dense_indices = routing_map.dtype != torch.bool
                if padding_mask is not None:
                    flat_mask = padding_mask.reshape(-1)
                    assert (
                        flat_mask.shape[0] == routing_map.shape[0]
                    ), f"padding_mask flat {flat_mask.shape} vs routing_map {routing_map.shape}"
                    if use_dense_indices:
                        routing_map = routing_map[~flat_mask]
                    else:
                        routing_map = routing_map & (~flat_mask).unsqueeze(-1)
                if use_dense_indices:
                    expert_indices = routing_map.reshape(-1).to(torch.long)
                    token_counts = torch.ones_like(
                        expert_indices, dtype=self.local_tokens_per_expert.dtype
                    )
                    if torch.are_deterministic_algorithms_enabled():
                        self.local_tokens_per_expert.index_add_(0, expert_indices, token_counts)
                    else:
                        self.local_tokens_per_expert.scatter_add_(0, expert_indices, token_counts)
                else:
                    self.local_tokens_per_expert += routing_map.sum(dim=0)

    def _hash_routing(
        self, logits: torch.Tensor, input_ids: torch.Tensor, dense_output: bool = False
    ):
        """Route tokens through the token-to-expert lookup table.

        Gating logits still provide the combination weights, while expert selection
        comes from ``tid2eid``.

        Args:
            logits (torch.Tensor): Gating logits, shape [num_tokens, num_experts].
            input_ids (torch.Tensor): Token IDs, shape [batch, sequence].
            dense_output (bool): If True, return dense [num_tokens, topk] probs/indices
                instead of sparse [num_tokens, num_experts] tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: routing_probs and routing_map, or dense
            probs and top-k indices when ``dense_output`` is set.
        """
        if self.score_function == "softmax":
            scores = (
                torch.softmax(logits, dim=-1, dtype=torch.float32)
                if self.config.moe_router_pre_softmax
                else logits
            )
        elif self.score_function == "sigmoid":
            scores = torch.sigmoid(logits.float())
        elif self.score_function == "sqrtsoftplus":
            scores = torch.nn.functional.softplus(logits.float()).sqrt()
        else:
            raise ValueError(f"Invalid score_function: {self.score_function}")

        # Hidden states are flattened from [sequence, batch, hidden], whereas
        # model token IDs arrive as [batch, sequence].
        flat_ids = input_ids.T.reshape(-1)
        assert flat_ids.numel() == logits.shape[0], (
            f"input_ids contains {flat_ids.numel()} tokens, but router logits contain "
            f"{logits.shape[0]}."
        )
        # Token IDs address the real tokenizer vocabulary; reject out-of-range IDs,
        # including negative IDs, without clamping them to another token's experts.
        default_top_indices = self.tid2eid.index_select(0, flat_ids).long()
        if (
            self.config.moe_router_force_load_balancing
            or self.config.moe_router_force_biased is not None
        ):
            # Benchmark forcing must override the fixed table, just as it overrides
            # learned top-k routing.
            default_top_indices = torch.topk(logits, k=self.topk, dim=1).indices

        def _compute_hash_topk(scores, topk, num_groups=None, group_topk=None):
            del topk, num_groups, group_topk
            return scores.gather(1, default_top_indices), default_top_indices

        if self.router_replay is not None:
            probs, top_indices = self.router_replay.get_replay_topk(
                scores, self.topk, default_compute_topk=_compute_hash_topk
            )
        else:
            probs, top_indices = _compute_hash_topk(scores, self.topk)
        if self.score_function == "softmax" and not self.config.moe_router_pre_softmax:
            probs = torch.softmax(probs, dim=-1, dtype=torch.float32)
        elif self.score_function != "softmax" and self.topk > 1:
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
        if self.config.moe_router_topk_scaling_factor:
            probs = probs * self.config.moe_router_topk_scaling_factor

        probs = probs.type_as(logits)

        if dense_output:
            return probs, top_indices

        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, top_indices, True)
        return routing_probs, routing_map

    def routing(
        self,
        logits: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.
            padding_mask (torch.Tensor, optional): Boolean mask indicating padding positions.
                                                   Shape [seq_length, bsz]. True = padding,
                                                   False = valid. Defaults to None.
            input_ids (torch.Tensor, optional): Token IDs with shape [batch, sequence].
                Required when this is a hash-routing layer. Defaults to None.
            packed_seq_params (PackedSeqParams, optional): Packed-sequence metadata used to
                pick the aux-loss reduction domain. Defaults to None.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts], or dense top-k indices with shape
                [num_tokens, topk] for supported Flex backends.
        """
        seq_length, bsz = logits.shape[:2]
        observe_router_diagnostics = is_observing_tensor("router_diagnostics")
        if self.training and observe_router_diagnostics:
            # The compact summaries normalize within each sequence and cannot reconstruct a
            # complete sequence after it has been partitioned across ranks.
            partitioned_sequence_axes = []
            if self.config.sequence_parallel and self.tp_group.size() > 1:
                partitioned_sequence_axes.append("tensor")
            if self.cp_group.size() > 1:
                partitioned_sequence_axes.append("context")
            if partitioned_sequence_axes:
                raise NotImplementedError(
                    "Router diagnostic observations require complete local sequences and do not "
                    "yet support sequence partitioning across "
                    + " and ".join(partitioned_sequence_axes)
                    + " parallel ranks."
                )
        logits = logits.view(-1, self.config.num_moe_experts)

        # Flatten padding_mask to [num_tokens] if provided
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits, padding_mask=padding_mask)

        # Calculate probs and routing_map for token dispatching
        if self.config.moe_n_hash_layers > 0:
            assert self.layer_number is not None, (
                "Hash routing requires a layer number. Construct the router through MoELayer "
                "or call set_layer_number() before routing."
            )
        if self.is_hash_layer:
            assert input_ids is not None, (
                "input_ids is required for hash-based routing. Pass token IDs through "
                "the model, transformer block, and transformer layer."
            )
            probs, routing_map = self._hash_routing(logits, input_ids)
        elif self.routing_type == "sinkhorn":
            probs, routing_map = self.sinkhorn_load_balancing(logits)
        else:
            # Activation checkpointing runs the original forward under no-grad and its
            # recompute under enable-grad, so this gate records each token exactly once.
            accumulate_qb_histogram = (
                self.use_quantile_balancing
                and self.training
                and torch.is_grad_enabled()
                and not self.frozen_expert_bias
            )
            if accumulate_qb_histogram and padding_mask is not None:
                raise RuntimeError(
                    "Quantile Balancing does not yet support padding masks because the "
                    "histogram APIs do not accept a valid-token mask."
                )
            topk_indices_dtype = self._dense_route_indices_dtype()
            topk_indices = (
                torch.empty(
                    (logits.shape[0], self.topk), dtype=topk_indices_dtype, device=logits.device
                )
                if topk_indices_dtype is not None
                else None
            )
            probs, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
                fused=self.config.moe_router_fusion,
                router_replay=self.router_replay,
                qb_histogram=self.qb_histogram if accumulate_qb_histogram else None,
                qb_bin_bounds=self.qb_bin_bounds if accumulate_qb_histogram else None,
                topk_indices=topk_indices,
            )

        # Dropless HybridEP consumes the sparse routing map directly, so exclude padding
        # rows before dispatch. Other dispatchers retain their existing fixed-route
        # assumptions until they support sparse routing maps end to end.
        use_dropless_hybridep = (
            self.config.moe_token_dispatcher_type == "flex"
            and self.config.moe_flex_dispatcher_backend == "hybridep"
            and self.config.moe_expert_capacity_factor is None
            and self.config.moe_expert_rank_capacity_factor is None
        )
        if padding_mask is not None and use_dropless_hybridep:
            valid_tokens = (~padding_mask).unsqueeze(-1)
            probs = probs * valid_tokens
            if routing_map.dtype == torch.bool:
                routing_map = routing_map & valid_tokens
            else:
                routing_map = routing_map.masked_fill(padding_mask.unsqueeze(-1), -1)

        # Apply token dropping to probs and routing_map.
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        # Reuse the unbiased aux-loss routing calculation for due router diagnostics.
        # Hash routing assignments come from tid2eid rather than the learned logits, so a
        # learned top-k aux loss would optimize routing decisions that were never dispatched.
        # Keep aux loss enabled for the non-hash MoE layers that share this configuration.
        aux_loss_enabled = not self.is_hash_layer and self.is_aux_loss_enabled()
        apply_aux_loss = torch.is_grad_enabled() and aux_loss_enabled
        compute_aux_inputs = self.training and (observe_router_diagnostics or apply_aux_loss)
        if compute_aux_inputs:
            score_context = nullcontext() if apply_aux_loss else torch.no_grad()
            with score_context:
                routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
                    logits,
                    self.topk,
                    self.score_function,
                    fused=self.config.moe_router_aux_loss_fusion,
                    padding_mask=padding_mask,
                )
            if observe_router_diagnostics:
                # Report the additive selection correction: Quantile Balancing tracks it in
                # expert_bias, the same slot ordinary bias routing adds to routing scores.
                diagnostics = build_router_diagnostics(
                    scores_for_aux_loss,
                    routing_map_for_aux_loss,
                    routing_map,
                    self.expert_bias,
                    seq_length,
                    bsz,
                    padding_mask=padding_mask,
                )
                observe_tensor(
                    self,
                    "router_diagnostics",
                    "router_diagnostics",
                    diagnostics,
                    tp_shard_dim=None,
                    sequence_dim=None,
                    batch_dim=0,
                )
            if apply_aux_loss:
                probs = self._apply_aux_loss(
                    probs,
                    scores_for_aux_loss,
                    routing_map_for_aux_loss,
                    with_padding_mask=padding_mask is not None,
                    packed_seq_params=packed_seq_params,
                )
                probs = self._apply_seq_aux_loss(
                    probs,
                    scores_for_aux_loss,
                    routing_map_for_aux_loss,
                    seq_length,
                    bsz,
                    with_padding_mask=padding_mask is not None,
                    packed_seq_params=packed_seq_params,
                )
                probs = self._apply_global_aux_loss(
                    probs,
                    scores_for_aux_loss,
                    routing_map_for_aux_loss,
                    with_padding_mask=padding_mask is not None,
                )

        # Optionally apply expert bias
        self._apply_expert_bias(routing_map, padding_mask=padding_mask)

        return probs, routing_map

    def reset_global_aux_loss_tracker(self):
        """Reset the global aux loss tracker."""
        if self.global_tokens_per_expert is not None:
            self.global_tokens_per_expert.zero_()
            self.ga_steps.zero_()

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass of the router.

        Raw router observations exclude padding when a mask is supplied. Their shape is then
        `[valid_tokens, num_experts]`, with sequence and batch populations both on dimension zero.
        Without a mask, observations retain the original sequence and batch dimensions. Routing
        itself always receives the full logits and the original mask.

        Args:
            input (torch.Tensor): Input tensor.
            padding_mask (torch.Tensor, optional): Boolean mask indicating padding positions.
                                                   Shape [seq_length, bsz]. True = padding,
                                                   False = valid. Defaults to None.
            input_ids (torch.Tensor, optional): Token IDs with shape [batch, sequence].
                Required when this is a hash-routing layer. Defaults to None.
            packed_seq_params (PackedSeqParams, optional): Packed-sequence metadata used to
                pick the aux-loss reduction domain. Defaults to None.
        """
        self._maintain_float32_expert_bias()

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        observe_logits = is_observing_tensor("router_logits")
        observe_scores = is_observing_tensor("router_scores")
        if observe_logits or observe_scores:
            with torch.no_grad():
                observed_logits = logits.detach()
                if padding_mask is not None:
                    observed_logits = observed_logits[~padding_mask]
                tp_shard_dim = 0 if self.config.sequence_parallel else None
                batch_dim = 0 if padding_mask is not None else 1
                if observe_logits:
                    observe_tensor(
                        self,
                        "router_logits",
                        "router_logits",
                        observed_logits,
                        tp_shard_dim=tp_shard_dim,
                        sequence_dim=0,
                        batch_dim=batch_dim,
                    )
                # Materialize the full decision distribution only for a due metric.
                if observe_scores:
                    scores = compute_normalized_router_scores(observed_logits, self.score_function)
                    observe_tensor(
                        self,
                        "router_scores",
                        "router_scores",
                        scores,
                        tp_shard_dim=tp_shard_dim,
                        sequence_dim=0,
                        batch_dim=batch_dim,
                    )

        if self.config.moe_router_force_load_balancing:
            # Apply force load balancing with random logits for benchmark
            logits = apply_random_logits(logits)

        if self.config.moe_router_force_biased is not None:
            # Apply biased logits with shared random bias across all ranks
            logits = apply_biased_logits(
                logits, self.config.moe_router_force_biased, self.layer_number
            )

        probs, routing_map = self.routing(
            logits,
            padding_mask=padding_mask,
            input_ids=input_ids,
            packed_seq_params=packed_seq_params,
        )

        return probs, routing_map

    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def _save_to_state_dict(self, *args, **kwargs):
        """Save the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before saving
        return super()._save_to_state_dict(*args, **kwargs)


class InferenceTopKRouter(TopKRouter):
    """Inference-only top-k router that strips out training-specific overhead.

    A stripped-down version of TopKRouter that skips z-loss, auxiliary load
    balancing losses, token dropping, and expert bias updates. The _forward()
    method is @torch.compile()'d and returns dense [num_tokens, topk] tensors
    instead of sparse [num_tokens, num_experts] for compatibility with FlashInfer.

    Falls back to the parent TopKRouter.forward() for training or
    non-CUDA-graphed inference iterations.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        layer_number: Optional[int] = None,
        hash_moe_layer_threshold: Optional[int] = None,
    ) -> None:
        """Initialize the specialized inference top-k router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        # Enforce constraints before calling super().__init__
        assert config.moe_router_num_groups is None, (
            f"InferenceTopKRouter requires moe_router_num_groups=None, "
            f"got {config.moe_router_num_groups}"
        )
        supported_compiled_scores = ["sigmoid", "softmax"]
        assert config.moe_router_score_function in supported_compiled_scores or (
            config.moe_num_hash_layers > 0 and config.moe_router_score_function == "sqrtsoftplus"
        ), (
            "InferenceTopKRouter requires moe_router_score_function to be sigmoid/softmax, "
            "or sqrtsoftplus for hash routing; got "
            f"{config.moe_router_score_function!r}"
        )

        super().__init__(
            config=config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            layer_number=layer_number,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
        )

    @staticmethod
    @torch.compile
    def _compiled_topk_routing(
        logits,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        expert_bias,
        fused,
        router_replay,
        dense_output,
    ):
        return topk_routing_with_score_function(
            logits,
            topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
            fused=fused,
            router_replay=router_replay,
            dense_output=dense_output,
        )

    def _forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        logits = self.gating(input)

        if self.is_hash_layer:
            logits = logits.view(-1, self.config.num_moe_experts)
            assert input_ids is not None, (
                "input_ids is required for hash-based routing. Pass token IDs through "
                "the model, transformer block, and transformer layer."
            )
            return self._hash_routing(logits, input_ids, dense_output=True)

        # Preserve the compiled inference router's established single-batch layout.
        logits = logits.squeeze(1)  # [num_tokens, num_experts]

        # Quantile Balancing keeps its selection correction in `expert_bias`, which is passed
        # to the routing function below, so no separate pre-computed top-k is needed here.
        routing = (
            topk_routing_with_score_function
            if is_batch_invariant_mode_enabled()
            else self._compiled_topk_routing
        )
        probs, top_indices = routing(
            logits,
            self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            fused=self.config.moe_router_fusion,
            router_replay=self.router_replay,
            dense_output=True,
        )
        return probs.squeeze(1), top_indices.squeeze(1)

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Simplified forward pass for inference - returns dense tensors only.

        Args:
            input (torch.Tensor): Input tensor of shape [seq_length, bsz, hidden_size].
            padding_mask (torch.Tensor, optional): Not used in inference.
            input_ids (torch.Tensor, optional): Token IDs used by hash routing.
            packed_seq_params (PackedSeqParams, optional): Not used in inference; forwarded
                to the training path when inference mode is inactive.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - probs: Normalized routing probabilities [num_tokens, topk]
                - top_indices: Selected expert indices [num_tokens, topk]
        """

        if not InferenceMode.is_active():
            return super().forward(input, padding_mask, input_ids, packed_seq_params)

        return self._forward(input, padding_mask, input_ids)
