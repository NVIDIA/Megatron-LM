# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import warnings
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import (
    deprecate_inference_params,
    log_single_rank,
    nvtx_range_push,
    nvtx_range_pop,
)

# TODO: Implement GatedDeltaNetContextParallel
# from .gated_delta_net_context_parallel import GatedDeltaNetContextParallel

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from fla.modules.l2norm import l2norm

    HAVE_FLA = True
except ImportError:
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None

    HAVE_FLA = False

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None


logger = logging.getLogger(__name__)


@dataclass
class GatedDeltaNetMixerSubmodules:
    """
    Contains the module specs for the input linear, output norm, and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class GatedDeltaNetMixer(MegatronModule):
    """
    Args:
        config: The config of the model.
        submodules: Contains the module specs for the input and output linear layers.
        layer_number: The layer number of this GDN layer.
        bias: Whether to use bias in the linear layers.
        conv_bias: Whether to use bias in the causal convolution.
        conv_init: The initialization range for the causal convolution weights.
        A_init_range: The initialization range for the attention weights.
        use_qk_l2norm: Whether to use L2 normalization in the kernel of the gated delta rule.
        *headdim: The hidden size of each attention head.
        *ngroups: The number of attention heads.
        *use_mem_eff_path: Whether to use the memory-efficient path for the GDN model.
        pg_collection: The required process groups to use for tensor model parallel and context
            parallel.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaNetMixerSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install it with `pip install fla`."
            )

        super().__init__(config)

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNetMixer"
        self.pg_collection = pg_collection
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        self.conv_kernel_dim = config.gdn_conv_kernel_dim
        self.qk_head_dim = config.gdn_qk_head_dim
        self.v_head_dim = config.gdn_v_head_dim
        self.num_qk_heads = config.gdn_num_qk_heads
        self.num_v_heads = config.gdn_num_v_heads
        self.qk_dim = self.qk_head_dim * self.num_qk_heads
        self.v_dim = self.v_head_dim * self.num_v_heads

        # TODO: support TP w/o SP
        if config.tensor_model_parallel_size > 1:
            assert config.sequence_parallel, (
                "GDN forces sequence parallelism if TP > 1."
            )

        # Input projection (hidden_states -> q, k, v, gate, beta, alpha)
        # TODO: for now, output gate is forced for GDN.
        # We may remove this restriction in the future.
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_v_heads * 2
        if self.config.fp8:
            assert self.in_proj_dim % 16 == 0, (
                "For FP8, the innermost dimension of the GDN layer "
                "input projection output tensor must be a multiple of 16."
            )
        # Assume sequence parallelism: input is already partitioned along the sequence dimension
        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )

        # Conv1d for QKV
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size
        
        # weight shape: [conv_dim, 1, d_conv]
        # bias shape: [conv_dim]
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.conv1d.weight, "tensor_model_parallel", True)
        if conv_bias:
            setattr(self.conv1d.bias, "tensor_model_parallel", True)
        
        # Time step projection (discretization)
        self.num_v_heads_local_tp = self.num_v_heads // self.tp_size
        # dt_bias parameter
        self.dt_bias = nn.Parameter(torch.empty(
            self.num_v_heads_local_tp,
            dtype=config.params_dtype,
            device=torch.cuda.current_device()))
        # Just to be explicit. Without this we already don't
        # put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True
        setattr(self.dt_bias, "tensor_model_parallel", True)
        # A_log parameter
        self.A_log = nn.Parameter(torch.empty(
            self.num_v_heads_local_tp,
            dtype=config.params_dtype,
            device=torch.cuda.current_device()
        ))
        self.A_log._no_weight_decay = True
        setattr(self.A_log, "tensor_model_parallel", True)
        
        # Output layernorm before projection
        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.v_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        # Assume sequence parallelism: input is partitioned along d_inner and
        # output is partitioned along the sequence dimension
        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
        )

        # TODO: support CP

        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                # conv1d.weight
                if self.conv_init is not None:
                    nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
                # dt_bias
                torch.ones(
                    self.num_v_heads_local_tp,
                    out=self.dt_bias.data,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                )
                # A_log
                A = torch.empty(
                    self.num_v_heads_local_tp,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(A)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        hidden_states: (nL, B, D) / (L B D)
        Returns: same shape as hidden_states
        """
        # TODO: Deal with attention_mask

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN does not support inference for now.")
        
        # Transpose: s b x --> b s x
        # Transform from sbhd to bshd format
        hidden_states = hidden_states.transpose(0, 1)

        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        # Split, reorder, and reshape the tensor into q, k, v, gate, beta, alpha
        qkv, gate, beta, alpha = torch.split(qkvzba, [
            (self.qk_dim * 2 + self.v_dim) // self.sp_size,
            self.v_dim // self.sp_size,
            self.num_v_heads // self.sp_size,
            self.num_v_heads // self.sp_size,
        ], dim=-1)
        gate = gate.reshape(batch, seq_len, -1, self.v_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)

        # Convolution on qkv
        qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
        nvtx_range_push(suffix="conv1d")
        if causal_conv1d_fn is None:
            qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
        else:
            assert self.activation in ["silu", "swish"]
            qkv = causal_conv1d_fn(
                x=qkv,
                weight=self.conv1d.weight.squeeze(1),  # d, 1, w -> d, w
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        nvtx_range_pop(suffix="conv1d")
        # Split qkv into query, key, and value
        qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
        query, key, value = torch.split(qkv, [
            self.qk_dim // self.sp_size,
            self.qk_dim // self.sp_size,
            self.v_dim // self.sp_size,
        ], dim=-1)
        query = query.reshape(batch, seq_len, -1, self.qk_head_dim)
        key = key.reshape(batch, seq_len, -1, self.qk_head_dim)
        value = value.reshape(batch, seq_len, -1, self.v_head_dim)
        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            query = l2norm(query.contiguous())
            key = l2norm(key.contiguous())
        if self.num_v_heads // self.num_qk_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_qk_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_qk_heads, dim=2)
        
        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        # Calculate g and beta
        nvtx_range_push(suffix="g_and_beta")
        g = -self.A_log.exp() * F.softplus(alpha.float() + self.dt_bias) # In fp32
        beta = beta.sigmoid()
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        # RMSNorm
        nvtx_range_push(suffix="gated_norm")
        norm_out = self._torch_compiled_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        # Output projection
        nvtx_range_push(suffix="out_proj")
        norm_out = norm_out.reshape(batch, seq_len, -1)
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        # Transpose: b s x --> s b x
        # Transform from bshd back to sbhd format
        out = out.transpose(0, 1).contiguous()
        
        return out, out_bias

    @torch.compile
    def _torch_compiled_gated_norm(self, x, gate):
        # Output Norm
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)
        # Output gate
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * self.act_fn(gate.float())
        y = y.to(x_dtype)
        return y

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                "A_log": 0,
                "dt_bias": 0,
            },  # parameters sharded across TP
            sharded_offsets=sharded_offsets,
        )
        # Submodules
        for name, module in self.named_children():
            if name == "conv1d":
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {f"weight": 0}
                if self.conv_bias:
                    tp_sharding_map[f"bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd, f"{prefix}{name}.", tp_sharding_map, sharded_offsets
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata
                )

            sharded_state_dict.update(module_sharded_sd)

        # At this point the TP sharding is correctly defined for each tensor, but some of the
        # tensors must be additionally split into separate parts
        in_proj_dim_local_tp = self.in_proj_dim // self.tp_size
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local_tp, (
            in_proj_dim_local_tp,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )

        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim // self.tp_size,
                self.qk_dim // self.tp_size,
                self.v_dim // self.tp_size,
                self.v_dim // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
            ],
            ["query", "key", "value", "z", "beta", "alpha"],
            0,
        )

        conv_layer_name_list = ["conv1d.weight"]
        assert sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0) == self.conv_dim_local_tp, (
            self.conv_dim_local_tp,
            sharded_state_dict[f"{prefix}conv1d.weight"],
        )
        if self.conv_bias:
            conv_layer_name_list.append("conv1d.bias")
            assert sharded_state_dict[f"{prefix}conv1d.bias"].data.size(0) == self.conv_dim_local_tp, (
                self.conv_dim_local_tp,
                sharded_state_dict[f"{prefix}conv1d.bias"],
            )
        for conv_layer_name in conv_layer_name_list:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [
                    self.qk_dim // self.tp_size,
                    self.qk_dim // self.tp_size,
                    self.v_dim // self.tp_size,
                ],
                ["query", "key", "value"],
                0,
            )

        return sharded_state_dict


def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: List[int], split_names: List[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )

    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim], (
            split_start,
            orig_sh_ten_no_data.local_shape[split_dim],
        )
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel(), (
            chunk_sh_tens,
            t.shape,
        )
        return chunk_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        orig_sh_ten.key, orig_sh_ten.data, sh_ten_build_fn, sh_ten_merge_fn, orig_sh_ten.replica_id
    )
