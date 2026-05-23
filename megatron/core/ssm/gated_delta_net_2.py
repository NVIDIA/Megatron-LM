# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_context_parallel import (
    _all_to_all_cp2hp,
    _all_to_all_hp2cp,
    _redo_attention_load_balancing,
    _undo_attention_load_balancing,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    cat_with_oom_fallback,
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

try:
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm

    HAVE_FLA = True
except ImportError:
    causal_conv1d = None
    l2norm = None

    HAVE_FLA = False

try:
    from fla.ops.gdn2.chunk_gdn2 import chunk_gdn2

    HAVE_GDN2_KERNEL = True
except ImportError:
    try:
        from fla.ops.gdn2 import chunk_gdn2

        HAVE_GDN2_KERNEL = True
    except ImportError:
        try:
            from lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2

            HAVE_GDN2_KERNEL = True
        except ImportError:
            chunk_gdn2 = None
            HAVE_GDN2_KERNEL = False

logger = logging.getLogger(__name__)


@dataclass
class GatedDeltaNet2Submodules:
    """
    Contains the module specs for the input linear, output norm, and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class GatedDeltaNet2(MegatronModule):
    """Gated Delta Net 2 (GDN2) layer class

    GDN2 layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaNet2Submodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
        allow_neg_eigval: bool = False,
        mode: str = "chunk",
        name: str | None = None,
    ):
        """
        Args:
            config: The config of the model.
            submodules: Contains the module specs for the input and output linear layers.
            layer_number: The layer number of this GDN2 layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization in the kernel of the gated delta rule.
            A_init_range: The initialization range for the attention weights.
            pg_collection: The required process groups to use for tensor model parallel and context
                parallel.
            allow_neg_eigval: Whether to scale the erase gate from [0, 1] to [0, 2].
            mode: Kernel dispatch mode. Only "chunk" is supported in this first implementation.
            name (str | None): module instance name passed top-down from its paranet module
        """

        if not HAVE_FLA or not HAVE_GDN2_KERNEL:
            raise ImportError(
                "GatedDeltaNet2 requires FLA causal_conv1d/l2norm and a GDN2 chunk_gdn2 "
                "kernel, such as NVlabs GatedDeltaNet-2 `lit_gpt.gdn2_ops.chunk_gdn2` "
                "or an equivalent FLA implementation. It cannot fall back to the old "
                "chunk_gated_delta_rule kernel."
            )
        if mode != "chunk":
            raise NotImplementedError("GatedDeltaNet2 only supports training chunk mode for now.")
        if config.deterministic_mode:
            raise NotImplementedError(
                "GatedDeltaNet2 does not support deterministic_mode; no torch-native GDN2 "
                "fallback is implemented."
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
        self.allow_neg_eigval = allow_neg_eigval
        self.mode = mode
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet2"
        self.pg_collection = pg_collection
        self.cp_size = self.pg_collection.cp.size()
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        # Attributes from config
        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size

        # Input projection (hidden_states -> q, k, v, output gate, erase, write, alpha)
        # TODO: for now, output gate is forced for GDN2.
        # We may remove this restriction in the future.
        self.in_proj_dim = self.qk_dim * 4 + self.v_dim * 3
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN2 layer "
                "input projection output tensor must be a multiple of 16."
            )
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
            name=(name + ".in_proj") if name is not None else None,
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
        setattr(self.conv1d.weight, "partition_dim", 0)
        if conv_bias:
            setattr(self.conv1d.bias, "tensor_model_parallel", True)
            setattr(self.conv1d.bias, "partition_dim", 0)

        # Time step projection (discretization)
        self.num_k_heads_local_tp = self.num_key_heads // self.tp_size
        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size
        # dt_bias parameter
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.qk_dim_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)
        # A_log parameter
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_k_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        setattr(self.A_log, "partition_dim", 0)

        self.gated_delta_rule_2 = chunk_gdn2

        # Output layernorm before projection
        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.value_head_dim,
            eps=self.config.layernorm_epsilon,
        )

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
            name=(name + ".out_proj") if name is not None else None,
        )

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
                    self.qk_dim_local_tp,
                    out=self.dt_bias.data,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                )
                # A_log
                A = torch.empty(
                    self.num_k_heads_local_tp,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(torch.log(A))

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the GDN2 module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) GDN2 output and bias.

        """
        # TODO: Deal with attention_mask

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN2 does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN2 does not support inference for now.")

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            assert batch == 1, "Packed sequence expects batch dimension to be 1"
            assert (
                not self.config.deterministic_mode
            ), "Packed sequence does not support deterministic mode."

            # Resolve cu_seqlens with alignment padding handling.
            cu_seqlens_q = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_q_padded,
                packed_seq_params.cu_seqlens_q,
                seq_len,
                "cu_seqlens_q",
                cp_size=self.cp_size,
            )
            cu_seqlens_kv = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_kv_padded,
                packed_seq_params.cu_seqlens_kv,
                seq_len,
                "cu_seqlens_kv",
                cp_size=self.cp_size,
            )
            assert torch.equal(cu_seqlens_q, cu_seqlens_kv), (
                "Currently only support cu_seqlens_q equals to cu_seqlens_kv, "
                f"but got {cu_seqlens_q=} and {cu_seqlens_kv=}"
            )
            num_packed_seqs = cu_seqlens_q.shape[0] - 1
            assert num_packed_seqs > 0, (
                "Number of packed sequences must be greater than 0, "
                f"but got {cu_seqlens_q=} and {cu_seqlens_kv=}"
            )
        else:
            cu_seqlens_q = None
            cu_seqlens_kv = None

        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkv_g_b_w_a, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        # CP All to All: CP to HP
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            unpacked_qkv_g_b_w_a = _unpack_sequence(
                qkv_g_b_w_a, cu_seqlens_q // self.cp_size, dim=0
            )
            outputs = []
            for qkv_g_b_w_a_i in unpacked_qkv_g_b_w_a:
                qkv_g_b_w_a_i = tensor_a2a_cp2hp(
                    qkv_g_b_w_a_i,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=self.pg_collection.cp,
                    split_sections=[
                        self.qk_dim_local_tp,
                        self.qk_dim_local_tp,
                        self.v_dim_local_tp,
                        self.v_dim_local_tp,
                        self.qk_dim_local_tp,
                        self.v_dim_local_tp,
                        self.qk_dim_local_tp,
                    ],
                )
                outputs.append(qkv_g_b_w_a_i)
            qkv_g_b_w_a = torch.cat(outputs, dim=0)
        else:
            qkv_g_b_w_a = tensor_a2a_cp2hp(
                qkv_g_b_w_a,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.pg_collection.cp,
                split_sections=[
                    self.qk_dim_local_tp,
                    self.qk_dim_local_tp,
                    self.v_dim_local_tp,
                    self.v_dim_local_tp,
                    self.qk_dim_local_tp,
                    self.v_dim_local_tp,
                    self.qk_dim_local_tp,
                ],
            )

        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkv_g_b_w_a = qkv_g_b_w_a.transpose(0, 1)

        # Split, reorder, and reshape the tensor into q, k, v, output gate, erase, write, alpha.
        qkv, out_gate, b, w, alpha = torch.split(
            qkv_g_b_w_a,
            [
                (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
                self.qk_dim_local_tp // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
                self.qk_dim_local_tp // self.cp_size,
            ],
            dim=-1,
        )

        # Convolution on qkv
        nvtx_range_push(suffix="conv1d")
        seq_len = qkv.shape[1]
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        if self.config.deterministic_mode:
            qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
            conv_out = F.conv1d(
                input=qkv,  # Torch-native only accept [b, d, s] format input
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // self.cp_size,
            )
            qkv = self.act_fn(conv_out[..., :seq_len])
            qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
        else:
            assert self.activation in ["silu", "swish"]
            qkv, _ = causal_conv1d(
                x=qkv,  # FLA conv1d accepts [b, s, d] format input
                weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w
                bias=conv1d_bias,
                activation=self.activation,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens_q,
            )
        nvtx_range_pop(suffix="conv1d")

        # Prepare QKV tensors (split, reshape, L2 norm, repeat_interleave, contiguous)
        nvtx_range_push(suffix="prepare_qkv_for_gated_delta_rule_2")
        query, key, value, out_gate, b, w, alpha = self._prepare_qkv_for_gated_delta_rule_2(
            qkv, out_gate, b, w, alpha, batch, seq_len
        )
        nvtx_range_pop(suffix="prepare_qkv_for_gated_delta_rule_2")

        # Calculate g, erase gate, and write gate.
        nvtx_range_push(suffix="g_b_w")
        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias_local_cp = get_parameter_local_cp(
            self.dt_bias, dim=0, cp_group=self.pg_collection.cp
        )
        g, b, w = self._compute_g_b_w(A_log_local_cp, dt_bias_local_cp, alpha, b, w)
        nvtx_range_pop(suffix="g_b_w")

        nvtx_range_push(suffix="gated_delta_rule_2")
        core_attn_out, last_recurrent_state = self.gated_delta_rule_2(
            q=query,
            k=key,
            v=value,
            g=g,
            b=b,
            w=w,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens_q,
        )
        nvtx_range_pop(suffix="gated_delta_rule_2")

        # RMSNorm
        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, out_gate)
        nvtx_range_pop(suffix="gated_norm")

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        # CP all to all: HP to CP
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            unpacked_norm_out = _unpack_sequence(norm_out, cu_seqlens_q, dim=0)
            outputs = []
            for norm_out_i in unpacked_norm_out:
                norm_out_i = tensor_a2a_hp2cp(
                    norm_out_i, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp
                )
                outputs.append(norm_out_i)
            norm_out = torch.cat(outputs, dim=0)
        else:
            norm_out = tensor_a2a_hp2cp(
                norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp
            )

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

    @jit_fuser
    def _apply_gated_norm(self, x, gate):
        # Output Norm
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)
        # Output gate
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * self.act_fn(gate.float())
        y = y.to(x_dtype)
        return y

    @jit_fuser
    def _prepare_qkv_for_gated_delta_rule_2(self, qkv, out_gate, b, w, alpha, batch, seq_len):
        """
        Prepare query, key, value, output gate, erase, write, and alpha tensors for GDN2.
        Fuses split, reshape, L2 norm, repeat_interleave, and contiguous operations.
        """
        # Split qkv into query_key and value
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )

        # Reshape query_key and value
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            query_key = l2norm(query_key.contiguous())

        # Split query and key
        split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
        query, key = torch.split(query_key, [split_size, split_size], dim=2)
        out_gate = out_gate.reshape(batch, seq_len, -1, self.value_head_dim)
        b = b.reshape(batch, seq_len, -1, self.key_head_dim)
        w = w.reshape(batch, seq_len, -1, self.value_head_dim)
        alpha = alpha.reshape(batch, seq_len, -1, self.key_head_dim)

        # Expand key-side tensors if needed (grouped value attention).
        if self.num_value_heads // self.num_key_heads > 1:
            repeat_factor = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)
            b = b.repeat_interleave(repeat_factor, dim=2)
            alpha = alpha.repeat_interleave(repeat_factor, dim=2)

        # Make all tensors contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        out_gate = out_gate.contiguous()
        b = b.contiguous()
        w = w.contiguous()
        alpha = alpha.contiguous()

        return query, key, value, out_gate, b, w, alpha

    @jit_fuser
    def _compute_g_b_w(self, A_log_local_cp, dt_bias_local_cp, alpha, b, w):
        """
        Compute g (decay), b (erase gate), and w (write gate) for GDN2.
        """
        local_key_heads = A_log_local_cp.shape[0]
        repeat_factor = self.num_value_heads // self.num_key_heads
        A_log = A_log_local_cp.float().reshape(local_key_heads, 1)
        A_log = A_log.repeat_interleave(repeat_factor, dim=0).reshape(
            1, 1, -1, 1
        )
        dt_bias = dt_bias_local_cp.float().reshape(local_key_heads, self.key_head_dim)
        dt_bias = dt_bias.repeat_interleave(repeat_factor, dim=0).reshape(
            1, 1, -1, self.key_head_dim
        )
        g = -A_log.exp() * F.softplus(alpha.float() + dt_bias)  # In fp32
        b = b.sigmoid()
        if self.allow_neg_eigval:
            b = 2.0 * b
        w = w.sigmoid()
        return g, b, w

    def _resolve_cu_seqlens(
        self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size: int = 1
    ):
        """Resolve cu_seqlens for packed sequence all-to-all, handling alignment padding."""
        if cu_seqlens_padded is not None:
            cu_seqlens = cu_seqlens_padded
        else:
            cu_seqlens = cu_seqlens_actual

        total_cu = cu_seqlens[-1].cpu().item()
        if total_cu != total_seq_len:
            raise ValueError(
                f"GDN2: {name}[-1]={total_cu} does not match "
                f"total_sequence_length={total_seq_len}. "
                f"({cu_seqlens_padded=}, {cu_seqlens_actual=})."
            )

        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if (seq_lengths % cp_size != 0).any():
            raise ValueError(
                f"All per-sequence lengths in cu_seqlens must be divisible by cp_size={cp_size}, "
                f"but got lengths: {seq_lengths.tolist()}"
            )

        return cu_seqlens

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

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
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata['dp_cp_group'],
        )
        # Submodules
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            if name == "conv1d":
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {f"weight": 0}
                if self.conv_bias:
                    tp_sharding_map[f"bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    tp_sharding_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
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
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.v_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.qk_dim_local_tp,
            ],
            ["query", "key", "value", "out_gate", "erase", "write", "alpha"],
            0,
        )

        conv_layer_name_list = ["conv1d.weight"]
        assert (
            sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0) == self.conv_dim_local_tp
        ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.weight"])
        if self.conv_bias:
            conv_layer_name_list.append("conv1d.bias")
            assert (
                sharded_state_dict[f"{prefix}conv1d.bias"].data.size(0) == self.conv_dim_local_tp
            ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.bias"])
        for conv_layer_name in conv_layer_name_list:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp],
                ["query", "key", "value"],
                0,
            )

        return sharded_state_dict

    def backward_dw(self):
        """Execute weight gradient computation for all linear layers."""
        self._backward_in_proj()
        self._backward_out_proj()

    def _backward_in_proj(self):
        """Computes weight gradients of input projection layer."""
        self.in_proj.backward_dw()

    def _backward_out_proj(self):
        """Computes weight gradients of output projection layer."""
        self.out_proj.backward_dw()


def _unpack_sequence(x, cu_seqlens, dim=1):
    unpacked_x = []
    cu_seqlens_list = cu_seqlens.tolist()
    num_seqs = len(cu_seqlens_list) - 1
    for i in range(num_seqs):
        idx_start = cu_seqlens_list[i]
        idx_end = cu_seqlens_list[i + 1]
        chunked_index = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


####################
# Sharded state dict utilities
####################
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

    return ShardedTensorFactory(
        orig_sh_ten.key,
        orig_sh_ten.data,
        sh_ten_build_fn,
        cat_with_oom_fallback,
        orig_sh_ten.replica_id,
    )


####################
# Context parallel utilities
####################
def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """Get the local parameter for the current context parallel rank.

    Args:
        param (torch.Tensor): The entire parameter to get the local parameter for.
        dim (int): The dimension to split the parameter along. Usually the dimension of head.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None,
            first split the parameter along the dimension dim into sections,
            then get the local hidden parallel weights separately,
            finally concatenate the local hidden parallel weights along the dimension dim.

    Returns:
        torch.Tensor: The local parameter for the current context parallel rank.
    """

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    # No need to split if CP size is 1.
    if cp_size == 1:
        return param

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    # Slice the parameter.
    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    param = param[slices]
    return param


def tensor_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    """All-to-all context parallel to hidden parallel.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None, split the tensor along the dimension
            head_dim into sections first, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        undo_attention_load_balancing (bool): Whether to undo the attention load balancing of CP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    cp_size = cp_group.size()

    # No need to all-to-all if CP size is 1.
    if cp_size == 1:
        return tensor

    # Limitations of mamba_context_parallel._all_to_all_cp2hp.
    assert seq_dim == 0, f"tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert (
        tensor.dim() == 3
    ), f"tensor_a2a_cp2hp only supports 3-d input tensor for now, but got {tensor.dim()=}"

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_cp2hp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                undo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_cp2hp(tensor, cp_group)

    # Undo attention load balancing last if needed.
    if undo_attention_load_balancing:
        tensor = _undo_attention_load_balancing(tensor, cp_size)
    return tensor


def tensor_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    """All-to-all hidden parallel to context parallel.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[List[int]]): If not None, first split the tensor along the
            dimension head_dim into sections, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        redo_attention_load_balancing (bool): Whether to redo the attention load balancing of HP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    cp_size = cp_group.size()

    # No need to all-to-all if CP size is 1.
    if cp_size == 1:
        return tensor

    # Limitations of mamba_context_parallel._all_to_all_hp2cp.
    assert seq_dim == 0, f"tensor_a2a_hp2cp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_hp2cp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert (
        tensor.dim() == 3
    ), f"tensor_a2a_hp2cp only supports 3-d input tensor for now, but got {tensor.dim()=}"

    # Redo attention load balancing first if needed.
    if redo_attention_load_balancing:
        tensor = _redo_attention_load_balancing(tensor, cp_size)

    # Split first if needed.
    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_hp2cp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                redo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_hp2cp(tensor, cp_group)

    return tensor
