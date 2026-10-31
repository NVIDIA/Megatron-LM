# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
""""
NOTE: NVLM uses InternViT with tensor parallel (TP) size = 8.
Since InternViT has 25 attention heads and Megatron currently requires the number of attention heads
to be divisible by the TP size, we add 7 dummy zero attention heads to have 32 attention heads.

This workaround requires some changes to how we compute RMSNorm, Attention etc.

Additionally, InternViT introduces some unique features like Layer Scaling.

Those code changes are gathered here.
"""
from functools import partial
from typing import Optional

import torch

from examples.multimodal.layer_scaling import (
    LayerScalingTransformerLayer,
    get_bias_dropout_add_layer_scaling,
)
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.process_groups_config import global_process_group_fallback
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.typed_torch import copy_signature, not_none
from megatron.core.utils import divide, get_pg_rank, get_pg_size

try:
    import apex

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


class InternViTRMSNorm(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-6,
        sequence_parallel: bool = False,
        compute_var: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """Custom RMSNorm for InternViT.

        Args:
            config (TransformerConfig): Config.
            hidden_size (int): Input hidden size.
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
            compute_var (bool): Indicator to compute statistic manually.
            tp_group (torch.distributed.ProcessGroup, optional): Tensor-parallel group of the
              owning model. The statistic is gathered over it, and the checkpoint shards or
              replicates the weight over its ranks. None falls back to the global
              tensor-parallel group, with a ProcessGroupFallbackWarning, once the group is
              needed; that fallback is deprecated.
        """
        super().__init__(config=config)
        self.config = config
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self._compute_var = compute_var
        self.tp_group = tp_group

        assert not sequence_parallel, "Sequence parallelism is not supported with InternViT."

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _resolve_tp_group(self):
        """Return the tensor-parallel group, falling back to the global one if none was given."""
        if self.tp_group is None:
            with global_process_group_fallback(
                "examples.multimodal.nvlm.internvit.InternViTRMSNorm",
                "tp_group",
                deprecated_in="0.21",
                removed_in="0.23",
            ):
                self.tp_group = get_tensor_model_parallel_group()
        return self.tp_group

    def _norm(self, x, var):
        if var is None:
            var = x.pow(2).mean(-1, keepdim=True)

        return x * torch.rsqrt(var + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run RMSNorm with an option to compute custom statistic."""
        var = None
        if self._compute_var:
            unpadded_hidden_size = self.config.hidden_size  # 3200
            max_dim = x.shape[-1]  # 128

            x = x.reshape(x.size(0), x.size(1), -1)
            var = self._gather_var(x.float().pow(2), max_dim) / unpadded_hidden_size

        output = self._norm(x.float(), var).type_as(x)
        output = output * self.weight

        if self._compute_var:
            output = output.reshape(output.size(0), output.size(1), -1, max_dim)

        return output

    def _gather_var(self, input_, max_dim):
        """Compute statistic across the non-dummy heads."""
        tp_group = self._resolve_tp_group()
        world_size = get_pg_size(tp_group)

        # Size and dimension.
        last_dim = input_.dim() - 1
        rank = get_pg_rank(tp_group)

        num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        valid_ranks = 24 // num_attention_heads_per_partition

        residual_heads = 25 % num_attention_heads_per_partition
        if residual_heads == 0:
            residual_heads = num_attention_heads_per_partition
        max_dim = max_dim * residual_heads

        if rank < valid_ranks:  # Ranks without any dummy attention heads.
            var = input_.sum(-1, keepdim=True)
        elif (
            rank == valid_ranks
        ):  # The only rank which may contain 'residual_heads' dummy attention heads.
            var = input_[..., :max_dim].sum(-1, keepdim=True)
        else:
            var = (
                input_.sum(-1, keepdim=True) * 0.0
            )  # All heads in these ranks are dummy heads: Zero-out.

        tensor_list = [torch.empty_like(var) for _ in range(world_size)]
        tensor_list[rank] = var
        torch.distributed.all_gather(tensor_list, var, group=tp_group)

        output = torch.cat(tensor_list, dim=last_dim).contiguous()

        return output.sum(-1, keepdim=True)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata={}):
        # Both branches shard or replicate the weight over the tensor-parallel group.
        tp_group = self._resolve_tp_group()

        # in InternVitSelfAttention the q_layernorm and k_layernorm weights
        # are tensor-parallel so must be converted to sharded tensors
        if 'q_layernorm' in prefix or 'k_layernorm' in prefix:
            state_dict = self.state_dict(prefix='', keep_vars=True)
            # Fill in a missing group on a copy, so that the default argument stays empty.
            metadata = ensure_metadata_has_dp_cp_group(dict(metadata or {}))
            return make_sharded_tensors_for_checkpoint(
                state_dict,
                prefix,
                {'weight': 0},
                sharded_offsets,
                tp_group=tp_group,
                dp_cp_group=metadata['dp_cp_group'],
            )
        else:
            return super().sharded_state_dict(prefix, sharded_offsets, metadata)


def get_mlp_module_spec(use_te: bool = True):
    # Dense MLP w/ or w/o TE modules.
    return partial(
        MLP.as_mlp_submodule,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
    )


# Override a few things that are special in InternViT and not supported by the SelfAttention class.
class InternViTSelfAttention(SelfAttention):
    @copy_signature(SelfAttention.__init__)
    def __init__(
        self, config: TransformerConfig, submodules: SelfAttentionSubmodules, *args, **kwargs
    ):
        super().__init__(config=config, submodules=submodules, *args, **kwargs)

        # Need to override linear_qkv, q_layernorm and k_layernorm.
        qkv_bias = False

        self.linear_qkv = submodules.linear_qkv(
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=not_none(self.config.init_method),
            gather_output=False,
            bias=qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            tp_group=self.pg_collection.tp,
            pg_collection=self.pg_collection,
        )

        qk_layernorm_hidden_size = (
            self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        )  # 512 for internvit

        self.q_layernorm = not_none(submodules.q_layernorm)(
            hidden_size=qk_layernorm_hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
            tp_group=self.pg_collection.tp,
        )

        self.k_layernorm = not_none(submodules.k_layernorm)(
            hidden_size=qk_layernorm_hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
            tp_group=self.pg_collection.tp,
        )


class InternViTTEDotProductAttention(TEDotProductAttention):
    """Adjusted Attention for InternViT"""

    @copy_signature(TEDotProductAttention.forward)
    def forward(self, *args, **kwargs):
        """Regular TEDotProductAttention + zero-out dummy attention heads."""
        out = super().forward(*args, **kwargs)

        # This makes sure the dummy attention heads are zeroed out.
        mask = torch.ones_like(out, dtype=out.dtype, device=out.device)
        # Rank in the tensor-parallel group this attention was built with.
        rank = get_pg_rank(self._tp_group)
        max_dim = out.shape[-1]  # 128
        valid_ranks = 6

        if rank == valid_ranks:
            mask[..., max_dim:] *= 0.0
        elif rank > valid_ranks:
            mask *= 0.0
        out *= mask

        return out


class InternViTTransformerLayer(LayerScalingTransformerLayer):
    """LayerScalingTransformerLayer whose input and pre-MLP RMSNorms use the layer's TP group."""

    @copy_signature(LayerScalingTransformerLayer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TransformerLayer builds its norms without a process group. InternViTRMSNorm shards or
        # replicates its weight in the checkpoint over a tensor-parallel group, so rebuild both
        # norms on the group of this layer.
        self.input_layernorm = self.submodules_config.input_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            tp_group=self.pg_collection.tp,
        )
        self.pre_mlp_layernorm = self.submodules_config.pre_mlp_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            tp_group=self.pg_collection.tp,
        )


def get_internvit_layer_spec(use_te) -> ModuleSpec:
    mlp = get_mlp_module_spec(use_te)  # no norm

    return ModuleSpec(
        module=InternViTTransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=InternViTRMSNorm,
            self_attention=ModuleSpec(
                module=InternViTSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                    core_attention=TEDotProductAttention if use_te else DotProductAttention,
                    linear_proj=TERowParallelLinear if use_te else RowParallelLinear,
                    q_layernorm=partial(InternViTRMSNorm, compute_var=True),
                    k_layernorm=partial(InternViTRMSNorm, compute_var=True),
                ),
            ),
            self_attn_bda=get_bias_dropout_add_layer_scaling,
            pre_mlp_layernorm=InternViTRMSNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add_layer_scaling,
        ),
    )


def get_internvit300M_layer_spec(use_te) -> ModuleSpec:
    mlp = get_mlp_module_spec(use_te)  # no norm

    return ModuleSpec(
        module=LayerScalingTransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                    core_attention=TEDotProductAttention if use_te else DotProductAttention,
                    linear_proj=TERowParallelLinear if use_te else RowParallelLinear,
                    q_layernorm=None,
                    k_layernorm=None,
                ),
            ),
            self_attn_bda=get_bias_dropout_add_layer_scaling,
            pre_mlp_layernorm=LNImpl,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add_layer_scaling,
        ),
    )
