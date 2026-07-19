# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Self-attention integration for Multi-Decay Attention (MDA)."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.random import _fork_rng, get_cuda_rng_tracker
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.typed_torch import apply_module
from megatron.core.utils import divide


def _load_mda_operators():
    """Load the optional MDA operators only when an MDA layer is constructed."""
    try:
        from fla.ops.mda import (
            fa4_mda_attn,
            fused_parallel_mda_attn_integrated,
            get_fa4_mda_availability,
            parallel_mda_attn,
            reference_mda_attn,
        )
    except ImportError:
        try:
            from fla.ops import multi_decay_fox as legacy_mda

            fa4_mda_attn = legacy_mda.fa4_multi_decay_fox_attn
            fused_parallel_mda_attn_integrated = (
                legacy_mda.fused_parallel_multi_decay_fox_attn_integrated
            )
            get_fa4_mda_availability = legacy_mda.get_fa4_multi_decay_availability
            parallel_mda_attn = legacy_mda.parallel_multi_decay_fox_attn
            reference_mda_attn = legacy_mda.reference_multi_decay_fox_attn
        except ImportError as import_error:
            raise ImportError(
                "MDA is unavailable. Put the multi-decay-att checkout first on PYTHONPATH "
                "(for example, export PYTHONPATH=$HOME/multi-decay-att:$PYTHONPATH)."
            ) from import_error
    return {
        'fa4': fa4_mda_attn,
        'fa4_availability': get_fa4_mda_availability,
        'fused': fused_parallel_mda_attn_integrated,
        'parallel': parallel_mda_attn,
        'reference': reference_mda_attn,
    }


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


class MultiDecayAttention(MegatronModule):
    """QKV-level MDA core used inside :class:`MultiDecaySelfAttention`."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(config=config)
        del layer_number, attention_type, cp_comm_type

        if pg_collection is None:
            raise ValueError("pg_collection must be provided for MultiDecayAttention")
        if pg_collection.cp.size() != 1:
            raise ValueError("MultiDecayAttention does not support context parallelism")
        if config.multi_decay_num_channels < 1:
            raise ValueError("multi_decay_num_channels must be at least 1")
        if config.multi_decay_decay_generation not in {'full', 'scaled_basis'}:
            raise ValueError("multi_decay_decay_generation must be 'full' or 'scaled_basis'")
        if config.multi_decay_aggregate_mode not in {'query_mix', 'mean', 'concat'}:
            raise ValueError("multi_decay_aggregate_mode must be query_mix, mean, or concat")
        if config.multi_decay_training_kernel not in {
            'reference',
            'fused',
            'bridge',
            'fa4',
            'auto',
        }:
            raise ValueError(
                "multi_decay_training_kernel must be reference, fused, bridge, fa4, or auto"
            )
        if config.multi_decay_window_size is not None and config.multi_decay_window_size < 1:
            raise ValueError("multi_decay_window_size must be positive")
        if config.apply_query_key_layer_scaling:
            raise ValueError("MultiDecayAttention does not support query-key layer scaling")
        if config.softmax_type != 'vanilla':
            raise ValueError("MultiDecayAttention requires vanilla softmax")
        if attn_mask_type != AttnMaskType.causal:
            raise ValueError("MultiDecayAttention requires a causal attention mask")

        dropout = config.attention_dropout if attention_dropout is None else attention_dropout
        if dropout != 0.0:
            raise ValueError("MultiDecayAttention requires zero attention dropout")

        self.attn_mask_type = attn_mask_type
        self.softmax_scale = softmax_scale
        self.num_decay_channels = config.multi_decay_num_channels
        self.decay_generation = config.multi_decay_decay_generation
        self.aggregate_mode = config.multi_decay_aggregate_mode
        self.training_kernel = config.multi_decay_training_kernel
        self.window_size = config.multi_decay_window_size
        self.use_nope = config.multi_decay_use_nope
        self._operators = _load_mda_operators()

    def _should_auto_use_fused(self, query: Tensor, value: Tensor) -> bool:
        decay_channel_fused = self.num_decay_channels in {4, 8, 16} or (
            self.num_decay_channels == 2 and query.shape[1] >= 2048
        )
        return (
            self.training_kernel == 'auto'
            and decay_channel_fused
            and self.aggregate_mode in {'query_mix', 'mean'}
            and self.window_size is None
            and query.is_cuda
            and query.dtype in {torch.float16, torch.bfloat16}
            and value.shape[-1] <= 128
            and (
                query.shape[-1] <= 64
                or (query.shape[-1] <= 128 and query.shape[1] <= 8192)
                or (query.shape[-1] <= 128 and query.shape[0] <= 8 and query.shape[1] <= 16384)
            )
        )

    def _should_auto_use_fa4_forward(self, query: Tensor, key: Tensor, value: Tensor) -> bool:
        batch, context = query.shape[:2]
        measured_shape = (context == 2048 and batch in {4, 8}) or (
            context in {4096, 8192, 16384} and batch in {1, 2, 4, 8}
        )
        return (
            self.training_kernel == 'auto'
            and self.num_decay_channels == 4
            and self.decay_generation == 'scaled_basis'
            and self.aggregate_mode == 'query_mix'
            and self.window_size is None
            and query.is_cuda
            and query.dtype in {torch.float16, torch.bfloat16}
            and measured_shape
            and query.shape == (batch, context, 32, 128)
            and key.shape == value.shape == (batch, context, 8, 128)
            and self._operators['fa4_availability'](query.device).forward
        )

    def _should_auto_use_fa4_training(self, query: Tensor, key: Tensor, value: Tensor) -> bool:
        if not query.is_cuda:
            return False
        batch, context = query.shape[:2]
        measured_shape = (
            query.dtype is torch.bfloat16
            and (
                (context == 2048 and batch in {2, 4, 8})
                or (context in {4096, 8192, 16384} and batch in {1, 2, 4, 8})
            )
        ) or (
            query.dtype is torch.float16
            and (
                (context == 2048 and batch in {4, 8})
                or (context in {4096, 8192, 16384} and batch in {1, 2, 4, 8})
            )
        )
        availability = self._operators['fa4_availability'](query.device)
        return (
            self.training_kernel == 'auto'
            and self.num_decay_channels == 8
            and self.decay_generation == 'scaled_basis'
            and self.aggregate_mode == 'query_mix'
            and self.window_size is None
            and measured_shape
            and query.shape == (batch, context, 32, 128)
            and key.shape == value.shape == (batch, context, 8, 128)
            and getattr(availability, 'training', False)
        )

    def _fused_mixer_logits(self, query: Tensor, mixer_logits: Tensor | None) -> Tensor:
        if self.aggregate_mode == 'query_mix':
            if mixer_logits is None:
                raise RuntimeError("query_mix requires mixer logits when R > 1")
            return mixer_logits
        if self.aggregate_mode == 'mean':
            return torch.zeros(
                (*query.shape[:3], self.num_decay_channels),
                device=query.device,
                dtype=torch.float32,
            )
        raise NotImplementedError("The fused MDA kernel supports query_mix and mean")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        log_f: Tensor | None = None,
        base_log_f: Tensor | None = None,
        decay_scales: Tensor | None = None,
        mixer_logits: Tensor | None = None,
        output_gate: Tensor | None = None,
    ) -> Tensor:
        """Apply MDA to pre-projected Megatron Q/K/V tensors."""
        if attention_mask is not None:
            raise NotImplementedError("MultiDecayAttention requires an implicit causal mask")
        if attention_bias is not None:
            raise NotImplementedError("MultiDecayAttention does not support attention bias")
        if packed_seq_params is not None:
            raise NotImplementedError("MultiDecayAttention does not support packed sequences")
        if attn_mask_type not in {None, AttnMaskType.causal}:
            raise NotImplementedError("MultiDecayAttention supports only causal attention")
        if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
            raise NotImplementedError("MultiDecayAttention does not yet support KV-cache decode")

        query = query.permute(1, 0, 2, 3).contiguous()
        key = key.permute(1, 0, 2, 3).contiguous()
        value = value.permute(1, 0, 2, 3).contiguous()
        if log_f is None:
            if self.num_decay_channels != 1 or not self.use_nope:
                raise RuntimeError("The configured MDA layer did not provide decay controls")
            log_f = torch.zeros((*query.shape[:3], 1), device=query.device, dtype=torch.float32)

        if log_f.shape != (*query.shape[:3], self.num_decay_channels):
            raise ValueError(
                f"log_f must have shape {(*query.shape[:3], self.num_decay_channels)}, "
                f"got {tuple(log_f.shape)}"
            )

        if self.training_kernel == 'fused':
            if self.aggregate_mode not in {'query_mix', 'mean'}:
                raise NotImplementedError("The fused MDA kernel supports query_mix and mean")
            if self.window_size is not None:
                raise NotImplementedError("The fused MDA kernel does not support window_size")
            if not query.is_cuda:
                raise NotImplementedError("The fused MDA kernel requires CUDA tensors")

        use_fa4 = self.training_kernel == 'fa4' or self._should_auto_use_fa4_training(
            query, key, value
        )
        if self.training_kernel == 'fa4':
            if self.aggregate_mode != 'query_mix':
                raise NotImplementedError("The FA4 MDA kernel supports query_mix only")
            if self.decay_generation != 'scaled_basis' or self.num_decay_channels != 8:
                raise NotImplementedError(
                    "The FA4 MDA kernel currently supports scaled_basis R=8 only"
                )
            if self.window_size is not None:
                raise NotImplementedError("The FA4 MDA kernel does not support window_size")
            if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
                raise NotImplementedError("The FA4 MDA kernel requires CUDA fp16/bf16 tensors")
            if query.shape[-1] > 128 or value.shape[-1] > 128:
                raise NotImplementedError("The FA4 MDA kernel supports dimensions up to 128")
            availability = self._operators['fa4_availability'](query.device)
            if not getattr(availability, 'training', False):
                raise NotImplementedError("The FA4 MDA training kernel is unavailable")

        use_fused = self.training_kernel == 'fused' or self._should_auto_use_fused(query, value)
        if self.num_decay_channels == 1:
            use_fused = False

        kernel_base_log_f = base_log_f
        kernel_decay_scales = decay_scales
        if self.use_nope:
            # Scaled-basis fused kernels divide by channel scales in backward.
            kernel_base_log_f = kernel_decay_scales = None

        if use_fa4:
            output = self._operators['fa4'](
                q=query,
                k=key,
                v=value,
                log_f=log_f,
                mixer_logits=self._fused_mixer_logits(query, mixer_logits),
                scale=self.softmax_scale,
                output_dtype=query.dtype,
                base_log_f=kernel_base_log_f,
                decay_scales=kernel_decay_scales,
            )
        elif use_fused:
            if self.window_size is not None:
                raise NotImplementedError("The fused MDA kernel does not support window_size")
            output = self._operators['fused'](
                q=query,
                k=key,
                v=value,
                log_f=log_f,
                mixer_logits=self._fused_mixer_logits(query, mixer_logits),
                scale=self.softmax_scale,
                output_dtype=query.dtype,
                base_log_f=kernel_base_log_f,
                decay_scales=kernel_decay_scales,
                use_fa4_forward=self._should_auto_use_fa4_forward(query, key, value),
            )
        elif self.training_kernel == 'reference' or (
            self.training_kernel == 'auto' and not query.is_cuda
        ):
            output = self._operators['reference'](
                q=query,
                k=key,
                v=value,
                log_f=log_f,
                mixer_logits=mixer_logits,
                scale=self.softmax_scale,
                window_size=self.window_size,
                aggregate_mode=self.aggregate_mode,
            ).to(query.dtype)
        else:
            if self.training_kernel == 'bridge' and not query.is_cuda:
                raise NotImplementedError("The bridge MDA kernel requires CUDA tensors")
            output = self._operators['parallel'](
                q=query,
                k=key,
                v=value,
                log_f=log_f,
                mixer_logits=mixer_logits,
                scale=self.softmax_scale,
                window_size=self.window_size,
                aggregate_mode=self.aggregate_mode,
                output_dtype=query.dtype,
            )

        if output_gate is not None:
            output = output * torch.sigmoid(output_gate.float()).to(output.dtype)
        output = output.permute(1, 0, 2, 3).contiguous()
        return output.view(output.shape[0], output.shape[1], -1)


@dataclass
class MultiDecaySelfAttentionSubmodules(SelfAttentionSubmodules):
    """Standard self-attention modules plus MDA control projections."""

    f_proj: ModuleSpec | type = IdentityOp
    mix_proj: ModuleSpec | type = IdentityOp
    g_proj: ModuleSpec | type = IdentityOp


class MultiDecaySelfAttention(SelfAttention):
    """General SelfAttention-integrated MDA for hybrid ``#`` layers.

    Q/K/V and output use the regular fused Megatron projections. Learned decay,
    query-mixing, and output-gate controls are projected from the exact normalized
    activation returned by the fused input-norm/QKV module.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MultiDecaySelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: int | None = None,
        name: str | None = None,
    ) -> None:
        if config.multi_decay_qk_norm and (config.qk_layernorm or config.qk_l2_norm):
            raise ValueError(
                "multi_decay_qk_norm cannot be combined with qk_layernorm or qk_l2_norm"
            )

        q_norm_spec = submodules.q_layernorm
        k_norm_spec = submodules.k_layernorm
        base_submodules = SelfAttentionSubmodules(
            linear_qkv=submodules.linear_qkv,
            core_attention=submodules.core_attention,
            linear_proj=submodules.linear_proj,
            q_layernorm=(q_norm_spec if config.qk_layernorm or config.qk_l2_norm else None),
            k_layernorm=(k_norm_spec if config.qk_layernorm or config.qk_l2_norm else None),
        )
        super().__init__(
            config=config,
            submodules=base_submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )

        if not hasattr(self.linear_qkv, 'return_layernorm_output'):
            raise TypeError(
                "MultiDecaySelfAttention requires a fused input-norm/QKV module that can "
                "return its normalized input"
            )
        self.linear_qkv.return_layernorm_output = True

        if config.multi_decay_qk_norm:
            if q_norm_spec in {None, IdentityOp} or k_norm_spec in {None, IdentityOp}:
                raise ValueError("multi_decay_qk_norm requires q_layernorm and k_layernorm specs")
            self.q_layernorm = build_module(
                q_norm_spec,
                hidden_size=self.hidden_size_per_attention_head,
                config=config,
                eps=config.layernorm_epsilon,
            )
            self.k_layernorm = build_module(
                k_norm_spec,
                hidden_size=self.hidden_size_per_attention_head,
                config=config,
                eps=config.layernorm_epsilon,
            )

        self.num_decay_channels = config.multi_decay_num_channels
        self.num_heads_per_partition = divide(config.num_attention_heads, self.world_size)
        self._core_attention_extra_kwargs: dict[str, Tensor] = {}

        def build_control_projection(spec, output_size, bias, suffix):
            return build_module(
                spec,
                config.hidden_size,
                output_size,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=suffix,
                tp_group=self.pg_collection.tp,
                name=(f'{name}.{suffix}' if name is not None else None),
            )

        control_free_nope = self.num_decay_channels == 1 and config.multi_decay_use_nope
        rng_context = (
            _fork_rng()
            if get_cuda_rng_tracker().is_initialized()
            else torch.random.fork_rng(devices=[])
        )
        with rng_context:
            # Auxiliary MDA parameters must not advance the regular model-init stream.
            # This keeps all shared and downstream parameters aligned with a baseline
            # SelfAttention model initialized from the same seed.
            self.f_proj = None
            if not control_free_nope:
                f_dim = (
                    config.num_attention_heads * self.num_decay_channels
                    if config.multi_decay_decay_generation == 'full'
                    else config.num_attention_heads
                )
                self.f_proj = build_control_projection(
                    submodules.f_proj, f_dim, config.multi_decay_decay_bias, 'f_proj'
                )

            self.mix_proj = None
            if config.multi_decay_aggregate_mode == 'query_mix' and self.num_decay_channels > 1:
                self.mix_proj = build_control_projection(
                    submodules.mix_proj,
                    config.num_attention_heads * self.num_decay_channels,
                    True,
                    'mix_proj',
                )

            self.g_proj = None
            if config.multi_decay_use_output_gate:
                self.g_proj = build_control_projection(
                    submodules.g_proj, self.query_projection_size, False, 'g_proj'
                )

        needs_decay_scales = not control_free_nope and (
            config.multi_decay_decay_generation == 'scaled_basis'
            or config.multi_decay_decay_type == 'mamba2'
        )
        if needs_decay_scales:
            parameter = nn.Parameter(
                self.linear_qkv.weight.new_empty(
                    self.num_heads_per_partition, self.num_decay_channels, dtype=torch.float32
                )
            )
            set_tensor_model_parallel_attributes(parameter, True, 0, 1)
            self.decay_log_scales = parameter
            self.reset_decay_scales()
        else:
            self.register_parameter('decay_log_scales', None)

    def _get_qkv_bias(self) -> bool:
        return self.config.multi_decay_qkv_bias

    def _get_linear_proj_input_size(self) -> int:
        if self.config.multi_decay_aggregate_mode == 'concat':
            return self.query_projection_size * self.config.multi_decay_num_channels
        return self.query_projection_size

    def reset_decay_scales(self) -> None:
        """Initialize decay channels to logarithmically spaced timescales."""
        if self.decay_log_scales is None:
            return
        if self.num_decay_channels == 1:
            scales = self.decay_log_scales.new_ones(1)
        else:
            scales = torch.logspace(
                -3,
                0,
                self.num_decay_channels,
                device=self.decay_log_scales.device,
                dtype=self.decay_log_scales.dtype,
            )
        if self.config.multi_decay_decay_type == 'mamba2':
            initial_value = torch.log(scales)
        else:
            initial_value = _inverse_softplus(scales)
        with torch.no_grad():
            self.decay_log_scales.copy_(initial_value.expand(self.num_heads_per_partition, -1))

    def _decay_scales(self) -> Tensor:
        if self.decay_log_scales is None:
            scales = self.linear_qkv.weight.new_ones(
                self.num_heads_per_partition, self.num_decay_channels, dtype=torch.float32
            )
        elif self.config.multi_decay_decay_type == 'mamba2':
            scales = torch.exp(self.decay_log_scales.float())
        else:
            scales = F.softplus(self.decay_log_scales.float())
        if self.config.multi_decay_use_nope:
            scales = torch.cat((torch.zeros_like(scales[:, :1]), scales[:, 1:]), dim=-1)
        return scales

    def _parameterize_decay(self, projected_decay: Tensor) -> Tensor:
        if self.config.multi_decay_decay_type == 'mamba2':
            return -F.softplus(projected_decay.float())
        return F.logsigmoid(projected_decay.float())

    @staticmethod
    def _apply_projection(projection: nn.Module, hidden_states: Tensor) -> Tensor:
        output, bias = apply_module(projection)(hidden_states)
        return output if bias is None else output + bias

    def _compute_controls(self, normalized_hidden_states: Tensor) -> dict[str, Tensor]:
        batch_first_shape = (normalized_hidden_states.shape[1], normalized_hidden_states.shape[0])
        controls: dict[str, Tensor] = {}

        if self.f_proj is None:
            log_f = torch.zeros(
                (*batch_first_shape, self.num_heads_per_partition, self.num_decay_channels),
                device=normalized_hidden_states.device,
                dtype=torch.float32,
            )
        else:
            projected_decay = self._apply_projection(self.f_proj, normalized_hidden_states)
            projected_decay = projected_decay.transpose(0, 1).contiguous()
            if self.config.multi_decay_decay_generation == 'full':
                log_f = self._parameterize_decay(projected_decay).view(
                    *batch_first_shape, self.num_heads_per_partition, self.num_decay_channels
                )
                if self.config.multi_decay_decay_type == 'mamba2':
                    log_f = log_f * self._decay_scales()[None, None]
            else:
                base_log_f = self._parameterize_decay(projected_decay)
                scales = self._decay_scales()
                if self.config.multi_decay_use_nope:
                    learned_log_f = base_log_f[..., None] * scales[None, None, :, 1:]
                    log_f = torch.cat(
                        (torch.zeros_like(base_log_f[..., None]), learned_log_f), dim=-1
                    )
                else:
                    log_f = base_log_f[..., None] * scales[None, None]
                    controls['base_log_f'] = base_log_f
                    controls['decay_scales'] = scales

        if self.config.multi_decay_use_nope:
            log_f = torch.cat((torch.zeros_like(log_f[..., :1]), log_f[..., 1:]), dim=-1)
        controls['log_f'] = log_f

        if self.mix_proj is not None:
            mixer_logits = self._apply_projection(self.mix_proj, normalized_hidden_states)
            controls['mixer_logits'] = (
                mixer_logits.transpose(0, 1)
                .contiguous()
                .view(*batch_first_shape, self.num_heads_per_partition, self.num_decay_channels)
                .float()
            )

        if self.g_proj is not None:
            output_gate = self._apply_projection(self.g_proj, normalized_hidden_states)
            controls['output_gate'] = (
                output_gate.transpose(0, 1)
                .contiguous()
                .view(
                    *batch_first_shape,
                    self.num_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
        return controls

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> (
        tuple[Tensor, Tensor, Tensor, Tensor]
        | tuple[Tensor, Tensor, Tensor]
        | tuple[Tensor, list[int]]
    ):
        if key_value_states is not None:
            raise NotImplementedError("MultiDecaySelfAttention supports self-attention only")
        if output_gate:
            raise ValueError("Use multi_decay_use_output_gate for an MDA output gate")

        projection_output, bias = apply_module(self.linear_qkv)(hidden_states)
        if bias is not None:
            raise RuntimeError("The fused MDA QKV projection returned an unexpected deferred bias")
        if not isinstance(projection_output, tuple) or len(projection_output) != 2:
            raise RuntimeError(
                "The fused MDA QKV projection did not return (QKV, normalized_hidden_states)"
            )
        mixed_qkv, normalized_hidden_states = projection_output
        self._core_attention_extra_kwargs = self._compute_controls(normalized_hidden_states)
        return self._split_mixed_qkv(mixed_qkv, output_gate=False, split_qkv=split_qkv)

    def get_core_attention_extra_kwargs(self) -> dict[str, Tensor]:
        return self._core_attention_extra_kwargs

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.decay_log_scales is not None:
            metadata = ensure_metadata_has_dp_cp_group(metadata)
            sharded_state_dict.update(
                make_sharded_tensors_for_checkpoint(
                    {'decay_log_scales': self.decay_log_scales},
                    prefix,
                    {'decay_log_scales': 0},
                    sharded_offsets,
                    tp_group=self.pg_collection.tp,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            )
        return sharded_state_dict

    def backward_dw(self) -> None:
        super().backward_dw()
        for projection in (self.f_proj, self.mix_proj, self.g_proj):
            if projection is not None and hasattr(projection, 'backward_dw'):
                projection.backward_dw()
