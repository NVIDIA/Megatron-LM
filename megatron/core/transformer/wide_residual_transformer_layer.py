# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Transformer adapter for wide-residual read/write connections."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.wide_residual import (
    StreamwiseSigmoidWideResidualConnection,
    load_residual_state,
    map_residual_sharded_keys,
)


@dataclass
class WideResidualTransformerLayerSubmodules(TransformerLayerSubmodules):
    """TransformerLayerSubmodules plus the branches' wide-residual connections (a fused spec
    selects WriteReadNorm reads and DeferredWrite writes)."""

    residual_read_self_attn: type = StreamwiseSigmoidWideResidualConnection
    residual_write_self_attn: type = StreamwiseSigmoidWideResidualConnection
    residual_read_mlp: type = StreamwiseSigmoidWideResidualConnection
    residual_write_mlp: type = StreamwiseSigmoidWideResidualConnection

    @classmethod
    def widen(cls, submodules: TransformerLayerSubmodules, **connections):
        """``submodules`` with these connections (the defaults for the others)."""
        values = {f.name: getattr(submodules, f.name) for f in fields(submodules)}
        return cls(**{**values, **connections})


class WideResidualTransformerLayer(TransformerLayer):
    """Transformer layer carrying a wide stream around ordinary-width branches."""

    supports_wide_residual_connections: bool = True

    @staticmethod
    def _is_attention_layer(submodules: TransformerLayerSubmodules) -> bool:
        """Whether a single-branch layer's branch is attention (else its MLP)."""
        attention = submodules.self_attention is not IdentityOp
        if attention == (submodules.mlp is not IdentityOp):
            raise ValueError("a fused layer has exactly one of self_attention and mlp")
        return attention

    @classmethod
    def get_input_layernorm_module(cls, submodules: TransformerLayerSubmodules) -> tuple[str, ...]:
        """Where a spec puts this single-branch layer's input RMSNorm, as attribute names from
        the layer: fused into attention's linear_qkv (or a GDN mixer's in_proj); fused into the
        MLP's linear_fc1, or held in pre_mlp_layernorm (MoE). Raises elsewhere."""
        if not cls._is_attention_layer(submodules):
            if submodules.pre_mlp_layernorm is not IdentityOp:
                return ("pre_mlp_layernorm",)
            return ("mlp", "linear_fc1")
        attention = getattr(submodules.self_attention, "submodules", None)
        for projection in ("linear_qkv", "in_proj"):
            if submodules.input_layernorm is IdentityOp and hasattr(attention, projection):
                return ("self_attention", projection)
        raise ValueError(
            "its input RMSNorm must be fused into self_attention's linear_qkv or in_proj"
        )

    @classmethod
    def get_wide_residual_modules(cls, submodules: TransformerLayerSubmodules) -> tuple[str, str]:
        """This single-branch layer's wide-residual (read, write) connection fields."""
        if cls._is_attention_layer(submodules):
            return "residual_read_self_attn", "residual_write_self_attn"
        return "residual_read_mlp", "residual_write_mlp"

    def __init__(
        self,
        config: TransformerConfig,
        submodules: WideResidualTransformerLayerSubmodules | TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
        hash_moe_layer_threshold: Optional[int] = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(submodules, WideResidualTransformerLayerSubmodules):
            submodules = WideResidualTransformerLayerSubmodules.widen(submodules)
        if is_mtp_layer:
            raise ValueError(
                "MTP auxiliary stacks must use ordinary-width TransformerLayer, not "
                "WideResidualTransformerLayer."
            )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
            name=name,
        )

        if config.wide_residual is None:
            raise ValueError("WideResidualTransformerLayer requires wide_residual config.")
        if not isinstance(self.cross_attention, IdentityOp):
            raise NotImplementedError(
                "WideResidualTransformerLayer does not support cross-attention branches."
            )

        self.residual_read_self_attn = None
        self.residual_write_self_attn = None
        if not isinstance(self.self_attention, IdentityOp):
            self.residual_read_self_attn = build_module(
                submodules.residual_read_self_attn,
                config,
                self.layer_number,
                "self_attention",
                operation="read",
            )
            self.residual_write_self_attn = build_module(
                submodules.residual_write_self_attn,
                config,
                self.layer_number,
                "self_attention",
                operation="write",
            )

        self.residual_read_mlp = None
        self.residual_write_mlp = None
        if not isinstance(self.mlp, IdentityOp):
            self.residual_read_mlp = build_module(
                submodules.residual_read_mlp, config, self.layer_number, "mlp", operation="read"
            )
            self.residual_write_mlp = build_module(
                submodules.residual_write_mlp, config, self.layer_number, "mlp", operation="write"
            )

        self.register_load_state_dict_pre_hook(load_residual_state)
        residual_connections = [
            reader
            for reader in (self.residual_read_self_attn, self.residual_read_mlp)
            if reader is not None
        ]

        if self.residual_read_self_attn is not None and self._input_layernorm_returns_residual:
            raise ValueError(
                "A self-attention residual connection cannot be combined with a layer norm "
                "that returns its own residual."
            )
        if self.residual_read_mlp is not None and self._pre_mlp_layernorm_returns_residual:
            raise ValueError(
                "An MLP residual connection cannot be combined with a layer norm that returns "
                "its own residual."
            )
        if residual_connections and self.config.inference_fuse_tp_communication:
            raise NotImplementedError(
                "Wide-residual connections do not support fused tensor-parallel inference."
            )

        self.residual_stream_hidden_size = (
            config.wide_residual.num_streams * self.config.hidden_size
        )
        for connection in residual_connections:
            if connection.residual_stream_hidden_size != self.residual_stream_hidden_size:
                raise ValueError(
                    "All wide-residual connections must carry num_streams * hidden_size "
                    f"features, expected {self.residual_stream_hidden_size}, got "
                    f"{connection.residual_stream_hidden_size}."
                )
            if connection.branch_hidden_size != self.config.hidden_size:
                raise ValueError(
                    "Wide-residual connections around TransformerLayer branches must produce "
                    f"hidden_size={self.config.hidden_size}, got "
                    f"{connection.branch_hidden_size}."
                )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Key the residual read/write parameters as existing checkpoints store them."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        map_residual_sharded_keys(sharded_state_dict, prefix)
        return sharded_state_dict
