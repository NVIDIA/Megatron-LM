# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Loop-conditioned branch hooks for Transformer and Mamba/GDP hybrid layers."""

from contextvars import ContextVar

from torch import Tensor

from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject
from megatron.core.models.hybrid.loop_conditioning import (
    LayerLoopConditioning,
    LoopConditioningConfig,
)
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import MoETransformerLayer, TransformerLayer


class _LoopConditionedLayer:
    supports_loop_conditioning = True

    def __init__(
        self,
        *args,
        loop_conditioning_config: LoopConditioningConfig,
        loop_conditioning_norm_key_map=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.config.inference_fuse_tp_communication:
            raise ValueError("Loop conditioning requires unfused residual branch updates")
        pg_collection = kwargs.get("pg_collection", getattr(self, "pg_collection", None))
        if pg_collection is None:
            raise ValueError("Loop-conditioned layers require explicit pg_collection")
        self.loop_conditioning_norm_key_map = dict(loop_conditioning_norm_key_map or {})
        is_mamba = isinstance(self, MambaLayer)
        mixer = is_mamba or not isinstance(self.self_attention, IdentityOp)
        mlp = not is_mamba and not isinstance(self.mlp, IdentityOp)
        if not is_mamba and not isinstance(self.cross_attention, IdentityOp):
            raise ValueError("Loop conditioning does not support cross-attention")
        if mixer and isinstance(self.norm if is_mamba else self.input_layernorm, IdentityOp):
            raise ValueError("Loop conditioning requires a separate mixer input normalization")
        if mlp and isinstance(self.pre_mlp_layernorm, IdentityOp):
            raise ValueError("Loop conditioning requires a separate MLP input normalization")
        self.loop_conditioning = LayerLoopConditioning(
            self.config, loop_conditioning_config, pg_collection, mixer=mixer, mlp=mlp
        )
        # A call-local scope lets the base layer retain ownership of its normal
        # attention/MLP orchestration. Tensor operations capture the condition in
        # their autograd graph; backward never reads this scope. ContextVar also
        # isolates this value across nested/concurrent calls and resets on exceptions.
        self._loop_condition: ContextVar[Tensor | None] = ContextVar("loop_condition", default=None)

    def forward(self, *args, loop_condition=None, **kwargs):
        """Run the base layer with the supplied condition scoped to this call."""
        if loop_condition is None:
            raise ValueError("Loop-conditioned layers require an explicit loop_condition")
        token = self._loop_condition.set(loop_condition)
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._loop_condition.reset(token)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Keep new normalization auxiliary state local on unconditioned imports."""
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if (metadata or {}).get("initialize_loop_conditioning", False):
            for norm_prefix in self.loop_conditioning_norm_key_map.values():
                key = prefix + norm_prefix + "_extra_state"
                if key in state:
                    state[key] = LocalNonpersistentObject(state[key].data)
        return state

    def _condition(self) -> Tensor:
        condition = self._loop_condition.get()
        if condition is None:
            raise ValueError("Loop-conditioned branch executed outside its forward scope")
        return condition


class LoopConditionedTransformerLayer(_LoopConditionedLayer, TransformerLayer):
    """Apply ELR Eq. 4 after branch normalization and before residual addition."""

    def _run_input_layernorm(self, *args, **kwargs):
        normalized, residual, state = super()._run_input_layernorm(*args, **kwargs)
        if self.loop_conditioning.mixer is not None:
            normalized = self.loop_conditioning.mixer.input(normalized, self._condition())
        return normalized, residual, state

    def _apply_self_attn_bda_step(self, attention_output_with_bias, *args, **kwargs):
        if self.loop_conditioning.mixer is not None:
            attention_output_with_bias = self.loop_conditioning.mixer.output(
                attention_output_with_bias, self._condition()
            )
        return super()._apply_self_attn_bda_step(attention_output_with_bias, *args, **kwargs)

    def _forward_pre_mlp_layernorm(self, *args, **kwargs):
        normalized = super()._forward_pre_mlp_layernorm(*args, **kwargs)
        if self.loop_conditioning.mlp is not None:
            normalized = self.loop_conditioning.mlp.input(normalized, self._condition())
        return normalized

    def _apply_mlp_bda_step(self, mlp_output_with_bias, *args, **kwargs):
        if self.loop_conditioning.mlp is not None:
            mlp_output_with_bias = self.loop_conditioning.mlp.output(
                mlp_output_with_bias, self._condition()
            )
        return super()._apply_mlp_bda_step(mlp_output_with_bias, *args, **kwargs)


class LoopConditionedMoETransformerLayer(LoopConditionedTransformerLayer, MoETransformerLayer):
    """Loop-conditioned branches retaining MoETransformerLayer's routing behavior."""


class LoopConditionedMambaLayer(_LoopConditionedLayer, MambaLayer):
    """Apply the same modulation to the normalized Mamba/GDP mixer branch."""

    def _prepare_mixer_input(self, *args, **kwargs):
        normalized = super()._prepare_mixer_input(*args, **kwargs)
        return self.loop_conditioning.mixer.input(normalized, self._condition())

    def _apply_mixer_update(self, mixer_out_with_bias, *args, **kwargs):
        scaled = self.loop_conditioning.mixer.output(mixer_out_with_bias, self._condition())
        return super()._apply_mixer_update(scaled, *args, **kwargs)
