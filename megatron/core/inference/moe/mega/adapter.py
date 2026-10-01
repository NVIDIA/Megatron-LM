# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lazy wrapper around FlashInfer ``MoEEpMegaLayer`` for one MoE block."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from megatron.core.inference.moe.mega._deps import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpMegaLayer,
    MoEEpTensors,
    require_flashinfer_moe_ep,
)
from megatron.core.inference.moe.mega.registry import build_megakernel_config
from megatron.core.inference.moe.mega.weights import megatron_grouped_weights_to_moe_pack
from megatron.core.utils import get_pg_rank, get_pg_size

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


class MegatronMegaMoEAdapter:
    """One FlashInfer mega layer per Megatron ``InferenceGroupedMLP``.

    In inference, ``MoEEpMegaLayer`` transforms the canonical weight pack at
    construction and releases the source tensors, so the expert weights are
    snapshotted on the first forward. Refitting Megatron's expert parameters
    afterwards does not reach the kernel; the adapter must be rebuilt instead.

    ``owns_transformed_weights`` inverts that for training, where the optimizer
    rewrites the parameters every step. The caller then supplies weights that
    are already in the kernel's layout and keeps rewriting them in place, and
    FlashInfer's own preprocessing is bypassed so it cannot snapshot anything.
    See :mod:`megatron.core.inference.moe.mega.training_weights`.

    The training forward shares one adapter across every MoE layer; see
    :meth:`shared_for_training`. Generation cannot, because each of its layers
    binds its own persistent weight buffer.
    """

    _shared_training: Optional["MegatronMegaMoEAdapter"] = None

    def __init__(
        self,
        config: "TransformerConfig",
        ep_group: torch.distributed.ProcessGroup,
        owns_transformed_weights: bool = False,
    ) -> None:
        require_flashinfer_moe_ep()
        self._config = config
        self._ep_group = ep_group
        self._layer: Optional[MoEEpMegaLayer] = None
        self._warmed_up = False
        self._owns_transformed_weights = owns_transformed_weights
        self._key: Optional[tuple] = None
        # data_ptrs the layer was constructed against, for the shared training
        # adapter's check that every layer really does hand it the same buffer.
        self._bound_weights: Optional[tuple[int, int]] = None

    @staticmethod
    def _training_key(config: "TransformerConfig", ep_group) -> tuple:
        """The geometry a shared training layer is valid for.

        The EP group enters by its member ranks rather than by ``id()``. Two
        distinct group objects over the same ranks are interchangeable here, and
        keying on identity would reject the second one for a geometry difference
        that does not exist.
        """
        try:
            ep = tuple(torch.distributed.get_process_group_ranks(ep_group))
        except Exception:  # pylint: disable=broad-except
            # Any group this cannot describe, including None for a single rank.
            ep = (get_pg_size(ep_group), get_pg_rank(ep_group))
        return (
            config.num_moe_experts,
            config.hidden_size,
            config.moe_ffn_hidden_size,
            config.moe_router_topk,
            config.inference_mega_max_tokens_per_rank,
            config.inference_mega_precision,
            ep,
        )

    @classmethod
    def shared_for_training(
        cls, config: "TransformerConfig", ep_group: torch.distributed.ProcessGroup
    ) -> "MegatronMegaMoEAdapter":
        """The one training adapter for this process, allocating it on first use.

        ``MoEEpMegaLayer`` holds a workspace sized by ``max_tokens_per_rank``.
        One per MoE layer would make that workspace the dominant memory term of
        the model, so the training forward shares a single layer, as it already
        shares a single weight buffer (see
        :class:`~megatron.core.inference.moe.mega.training_weights.MegaTrainingWeightScratch`).

        Sharing is sound because every MoE layer builds its kernel layer from
        the same inputs: the weights are views of one scratch buffer, and the
        geometry and EP group are model-wide. Each layer repacks the scratch
        immediately before its own forward, so the shared kernel sees that
        layer's weights. :meth:`_ensure_layer` checks the buffer identity, and a
        second geometry is rejected instead of allocating another workspace.
        Warmup and CuTeDSL compilation also happen once per process.
        """
        key = cls._training_key(config, ep_group)
        if cls._shared_training is None:
            cls._shared_training = cls(config, ep_group, owns_transformed_weights=True)
            cls._shared_training._key = key
        elif cls._shared_training._key != key:
            raise RuntimeError(
                "the mega training adapter is shared across MoE layers and was built "
                f"for {cls._shared_training._key}, but a layer requested {key}. Mixed "
                "expert geometries in one process are not supported."
            )
        return cls._shared_training

    @classmethod
    def reset_shared_training(cls) -> None:
        """Drop the shared training adapter. For teardown between tests."""
        cls._shared_training = None

    def _fleet_params(self) -> FleetParams:
        # dtype_bytes/algorithm/layout are split-transport fields the mega path
        # ignores; they are left at their defaults.
        return FleetParams(
            num_experts=self._config.num_moe_experts,
            max_tokens_per_rank=self._config.inference_mega_max_tokens_per_rank,
            token_hidden_size=self._config.hidden_size,
        )

    def _bootstrap(self) -> BootstrapConfig:
        return BootstrapConfig(
            world_size=get_pg_size(self._ep_group),
            rank=get_pg_rank(self._ep_group),
            process_group=self._ep_group,
            device=torch.cuda.current_device(),
        )

    def _ensure_layer(
        self,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc1_scale: Optional[torch.Tensor] = None,
        fc2_scale: Optional[torch.Tensor] = None,
    ) -> MoEEpMegaLayer:
        if self._layer is not None:
            if self._owns_transformed_weights:
                self._assert_bound_to(fc1_weight, fc2_weight, fc1_scale, fc2_scale)
            return self._layer
        # Construction runs collective symmetric-heap bootstrap and weight
        # preprocessing, neither of which can be captured.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "flashinfer_mega layer construction cannot run during CUDA graph "
                "capture. Run one eager forward on all EP ranks before capturing."
            )
        megakernel = build_megakernel_config(self._config)
        # quantize_input must stay True: the megakernels have no pre-quantized
        # activation path and reject quantize_input=False. The quantized
        # precisions derive activation scales in-kernel, so no calibration data
        # is needed here.
        if self._owns_transformed_weights:
            # fc1/fc2 are already K-major kernel layout and the caller mutates
            # them in place every step, so there is nothing to preprocess and
            # nothing may be released. FlashInfer validates the layout here,
            # which is what catches a repack that does not match its contract.
            weights = None
            mega_config = MegaConfig(
                megakernel=megakernel,
                preprocess_weights=False,
                # The scale slots are None for bf16 and carry the swizzled block
                # scales for mxfp8, which is the shape the quantized kernels
                # validate transformed_weights against.
                transformed_weights=((fc1_weight, fc1_scale), (fc2_weight, fc2_scale)),
            )
        else:
            weights = megatron_grouped_weights_to_moe_pack(fc1_weight, fc2_weight)
            mega_config = MegaConfig(megakernel=megakernel, preprocess_weights=True)
        self._layer = MoEEpMegaLayer(self._bootstrap(), self._fleet_params(), weights, mega_config)
        if self._owns_transformed_weights:
            self._bound_weights = self._addresses(fc1_weight, fc2_weight, fc1_scale, fc2_scale)
        return self._layer

    @staticmethod
    def _addresses(*tensors: Optional[torch.Tensor]) -> tuple:
        return tuple(None if tensor is None else tensor.data_ptr() for tensor in tensors)

    def _assert_bound_to(
        self,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc1_scale: Optional[torch.Tensor] = None,
        fc2_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Fail if the caller's weight buffer is not the one the kernel reads.

        FlashInfer binds ``transformed_weights`` at construction, so a caller
        that later hands over a different buffer would be silently ignored and
        the kernel would keep reading the first one. Compared by address
        because ``views()`` returns a fresh transpose each call.
        """
        got = self._addresses(fc1_weight, fc2_weight, fc1_scale, fc2_scale)
        if self._bound_weights != got:
            raise RuntimeError(
                "the mega layer was constructed against a different weight buffer "
                f"(bound {self._bound_weights}, got {got}). Caller-owned weights must "
                "stay in the same storage for the kernel to see writes to them."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc1_scale: Optional[torch.Tensor] = None,
        fc2_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run local-token mega MoE; EP communication is inside the kernel."""
        num_tokens = hidden_states.shape[0]
        max_cap = self._config.inference_mega_max_tokens_per_rank
        if num_tokens > max_cap:
            raise ValueError(
                f"Mega MoE received {num_tokens} local tokens, exceeding "
                f"inference_mega_max_tokens_per_rank={max_cap}. "
                "Increase the cap or reduce batch tokens per EP rank."
            )

        layer = self._ensure_layer(fc1_weight, fc2_weight, fc1_scale, fc2_scale)
        if not self._warmed_up:
            # warmup() is an EP collective that forces workspace allocation and
            # CuTeDSL compilation; both are illegal under capture.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "flashinfer_mega requires warmup() on all EP ranks before CUDA "
                    "graph capture. Run one eager forward first."
                )
            layer.warmup()
            self._warmed_up = True

        # Canonical expert order, so the result depends on the token and not on
        # which pass produced the list. Generation hands over
        # InferenceTopKRouter's top-k in descending score order; the training
        # forward recovers the same set from TopKRouter's dense mask in mask
        # order. The kernel combines prob * expert_out in the order given and
        # float addition is not associative, so the two orders can differ in the
        # last bit. Expert ids are unique within a token, so the sort is total.
        order = routing_map.argsort(dim=-1)
        tensors = MoEEpTensors(
            hidden_states=hidden_states.contiguous(),
            topk_ids=routing_map.gather(-1, order),
            topk_weights=probs.to(torch.float32).gather(-1, order),
        )
        return layer.forward(tensors)
