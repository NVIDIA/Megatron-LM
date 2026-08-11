# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.inference.contexts import BaseInferenceContext, DynamicInferenceContext
from megatron.core.inference.contexts.attention_context.triton.tensor_ops import (
    tensor_get_slice_after,
    tensor_masked_update,
    tensor_merge,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gdp_context_parallel import GDPContextParallel
from megatron.core.ssm.packed_seq_helpers import (
    build_packed_seq_idx,
    check_fla_sequence_packing_support,
    get_cu_seqlens,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import (
    deprecate_inference_params,
    is_using_quantization_scales,
)

if HAVE_GTP:
    from megatron.core.tensor_parallel.gtp_api import is_gtp_param
else:
    is_gtp_param = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

    HAVE_MAMBA_SSM = True
except ImportError:
    from unittest.mock import MagicMock

    RMSNormGated = MagicMock()
    HAVE_MAMBA_SSM = False

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

try:
    from fla.ops.gated_delta_product import chunk_gated_delta_product
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


logger = logging.getLogger(__name__)


class ExtendedRMSNorm(RMSNormGated):
    """
    RMSNormGated with sharded state dict.
    """

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias not sharded"""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


@dataclass
class GatedDeltaProductMixerSubmodules:
    """
    Contains the module specs for the input and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = None
    out_proj: Union[ModuleSpec, type] = None


class GatedDeltaProductMixer(MegatronModule):
    """Gated Delta Product (GDP) sequence mixer for hybrid models.

    The mixer accepts hidden states with shape ``[sequence, batch, hidden]`` and returns
    a projected tensor with the same shape plus the optional output-projection bias. It
    serves as the mixer inside ``MambaLayer``, allowing a hybrid stack to select GDP layers
    without changing the surrounding layer interface.

    GDP projects each token into an output gate and the ``V``, ``K``, ``Q``, beta, and decay
    terms used by a sequence of Householder updates. A depthwise causal convolution mixes
    local context in ``V/K/Q``; the FLA GDP recurrence then updates a matrix-valued state,
    and gated RMS normalization plus the output projection map the result back to the
    model hidden size.

    The module shards projections and recurrent parameters across tensor-parallel ranks,
    redistributes sequence and head dimensions for context parallelism, handles packed
    THD training sequences, manages static and dynamic inference state, and exposes
    semantic sharded-state-dict partitions for checkpoint resharding across TP sizes.

    Args:
        config: The config of the model.
        submodules: Contains the module specs for the input and output linear layers.
        d_model: The hidden size of the model.
        d_conv: The number of channels in the causal convolution.
        conv_init: The initialization range for the causal convolution weights.
        A_init_range: The initialization range for the attention weights.
        D_has_hdim: Whether the D parameter has the same number of dimensions as the hidden
            state.
        rmsnorm: Whether to use root mean square normalization.
        norm_before_gate: Whether to apply normalization before the gating mechanism.
        dt_min: The minimum value of the dt parameter.
        dt_max: The maximum value of the dt parameter.
        dt_init_floor: The minimum value of the dt parameter after initialization.
        bias: Whether to use bias in the linear layers.
        conv_bias: Whether to use bias in the causal convolution.
        chunk_size: The chunk size for the fused kernel.
        layer_number: The layer number of this Mamba layer.
        pg_collection: The required process groups to use for tensor model parallel and context
            parallel.
        name: Module instance name passed top-down from its parent module.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaProductMixerSubmodules,
        d_model,
        d_conv=4,
        conv_init=None,
        A_init_range=(0, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=False,
        # Fused kernel and sharding options
        chunk_size=128,
        layer_number=None,
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: int = 0,
        name: str | None = None,
    ):
        if not HAVE_MAMBA_SSM:
            raise ImportError(
                "MambaSSM is not installed. Please install it with `pip install mamba-ssm`."
            )

        if not HAVE_FLA:
            raise ImportError("FLA is not installed")

        super().__init__(config)

        # Inference-time contract: ``MambaInferenceStateConfig.from_model`` in
        # megatron/core/inference/config.py reads ``layer.mixer.chunk_size`` to
        # size the SSM scan blocks.
        self.chunk_size = chunk_size

        # Check that the causal_conv1d version is new enough or fail
        ok, reason = check_fla_sequence_packing_support()
        assert ok, reason

        self.num_householder = config.gdp_num_householder

        self.config = config
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        assert pg_collection is not None, "pg_collection must be provided for MambaMixer"
        self.pg_collection = pg_collection

        self.d_state = self.config.mamba_state_dim
        self.headdim = self.config.mamba_head_dim
        self.ngroups = self.config.mamba_num_groups
        self.nheads = self.config.mamba_num_heads
        assert self.nheads is not None, "mamba_num_heads must be set for GatedDeltaProductMixer"
        self.d_inner = self.nheads * self.headdim

        self.layer_number = layer_number
        self.pp_layer_offset = pp_layer_offset
        self.cached_batch_size = None

        tp_size = self.pg_collection.tp.size()

        self.nheads_local_tp = self.nheads // tp_size
        self.d_inner_local_tp = self.d_inner // tp_size
        self.ngroups_local_tp = self.ngroups // tp_size

        # Assume sequence parallelism: input is already partitioned along the sequence dimension
        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            self.d_inner * (1 + self.num_householder)
            + self.ngroups * self.d_state * (self.num_householder + 1)
            + self.nheads * (self.num_householder + 1),  # zVKQba
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
            pg_collection=self.pg_collection,
            name=(name + f".in_proj") if name is not None else None,
        )
        setattr(self.in_proj.weight, "use_muon", False)
        if self.in_proj.bias is not None:
            setattr(self.in_proj.bias, "use_muon", False)

        # The fused projection packs independently TP-sharded components. Refit
        # uses their local sizes to preserve semantic order when TP size changes.
        in_proj_partition_sizes, _ = _get_in_proj_checkpoint_split_layout(
            self.d_inner_local_tp,
            self.ngroups_local_tp * self.d_state,
            self.nheads_local_tp,
            self.num_householder,
        )
        setattr(self.in_proj.weight, "partition_sizes", in_proj_partition_sizes)
        if self.in_proj.bias is not None:
            setattr(self.in_proj.bias, "partition_sizes", in_proj_partition_sizes)

        conv_dim = (
            self.d_inner_local_tp * self.num_householder
            + (self.num_householder + 1) * self.ngroups_local_tp * self.d_state
        )  # V K Q
        with get_cuda_rng_tracker().fork():
            # weight shape: [conv_dim, 1, d_conv]
            # bias shape: [conv_dim]
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
            setattr(self.conv1d.weight, "tensor_model_parallel", True)
            setattr(self.conv1d.weight, "partition_dim", 0)
            if conv_bias:
                setattr(self.conv1d.bias, "tensor_model_parallel", True)
                setattr(self.conv1d.bias, "partition_dim", 0)

            conv_partition_sizes, _ = _get_conv_checkpoint_split_layout(
                self.d_inner_local_tp, self.ngroups_local_tp * self.d_state, self.num_householder
            )
            setattr(self.conv1d.weight, "partition_sizes", conv_partition_sizes)
            if conv_bias:
                setattr(self.conv1d.bias, "partition_sizes", conv_partition_sizes)

            if self.conv_init is not None:
                nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.activation = "silu"
        self.act = nn.SiLU()

        with get_cuda_rng_tracker().fork():
            # MCore Mamba2 initialization
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(
                    self.nheads_local_tp,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)

            # Our initialization would set all Linear.bias to zero,
            # need to mark this one as _no_reinit
            self.dt_bias._no_reinit = True
            # Just to be explicit. Without this we already don't
            # put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True
            setattr(self.dt_bias, "tensor_model_parallel", True)

            # A parameter
            assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
            A = torch.empty(
                self.nheads_local_tp, dtype=torch.float32, device=torch.cuda.current_device()
            ).uniform_(*A_init_range)
            A_log = torch.log(A)  # Keep A_log in fp32
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True
            setattr(self.A_log, "tensor_model_parallel", True)

        # D "skip", in Mamba2 but not in GDN or GDP
        self.D = None

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = ExtendedRMSNorm(
                self.d_inner_local_tp,
                eps=1e-5,
                group_size=self.d_inner_local_tp // self.ngroups_local_tp,
                norm_before_gate=self.norm_before_gate,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
            self.norm.tp_group = self.pg_collection.tp
            setattr(self.norm.weight, 'tensor_model_parallel', True)

        # Assume sequence parallelism: input is partitioned along d_inner and
        # output is partitioned along the sequence dimension
        self.out_proj = build_module(
            submodules.out_proj,
            self.d_inner,
            self.d_model,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
            pg_collection=self.pg_collection,
            name=(name + f".out_proj") if name is not None else None,
        )

        # Regarding `conv1d`.{`weight`, `bias`}, `dt_bias`, `A_log`, and `D`: these are the
        # trainable variables for the current tensor parallel rank, with each tensor parallel rank
        # having indepdendent trainable variables. All context parallel ranks in a tensor parallel
        # rank store the same trainable variables, but only use and update their unique/independent
        # slice of them.
        self.cp = GDPContextParallel(
            cp_group=self.pg_collection.cp,
            d_inner_local_tp=self.d_inner_local_tp,
            nheads_local_tp=self.nheads_local_tp,
            ngroups_local_tp=self.ngroups_local_tp,
            d_state=self.d_state,
            num_householder=self.num_householder,
            headdim=self.headdim,
            conv1d_cp1=self.conv1d,
            dt_bias_cp1=self.dt_bias,
            A_log_cp1=self.A_log,
            D_cp1=self.D,
            D_has_hdim=self.D_has_hdim,
        )

    def forward(
        self,
        hidden_states,
        inference_context=None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params=None,
    ):
        """Run the gated delta product mixer on hidden states."""
        seq_len, batch_size, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_context is not None:
            if inference_context.is_dynamic_batching():
                return self._dynamic_inference(hidden_states, inference_context)
            assert (
                inference_context.is_static_batching()
            ), "GDP inference must be either static or dynamic batching."
            assert not self.config.sequence_parallel
            assert packed_seq_params is None, (
                "GDP does not currently support packed sequences during inference. "
                "Packing is only wired through the training/prefill (chunk) path."
            )
            conv_state, ssm_state = self._get_states_from_cache(inference_context, batch_size)

        # Build cu_seqlens for the chunked recurrence (FLA) when running with
        # packed (THD) sequences on the training/prefill path.
        cu_seqlens_packed = None
        if packed_seq_params is not None:
            # ``hidden_states`` is [seq_len, batch, dim]; THD requires batch=1.
            assert batch_size == 1, "Packed sequences require batch=1 (THD/varlen format)."
            cu_seqlens_packed = get_cu_seqlens(packed_seq_params)

        zVKQba, _ = self.in_proj(hidden_states)

        zVKQba = self.cp.pre_conv_ssm(zVKQba, packed_seq_params=packed_seq_params)

        # Build seq_idx *after* in_proj's SP all-gather and pre_conv_ssm's CP
        # all-to-all. ``zVKQba.shape[0]`` is now the true pack_length, so the
        # helper produces a seq_idx matching the conv1d's input length
        # regardless of SP/CP/TP upstream-slicing. Mirrors mamba_mixer.py
        # which calls _create_packed_seq_idx after the same gather points.
        seq_idx_packed = None
        if packed_seq_params is not None:
            seq_idx_packed = build_packed_seq_idx(packed_seq_params, zVKQba.shape[0])

        zVKQba = rearrange(zVKQba, "l b d -> b l d").contiguous()

        z, VKQ, ba = torch.split(
            zVKQba,
            [
                self.cp.d_inner_local_tpcp,
                self.cp.d_inner_local_tpcp * self.num_householder
                + (self.num_householder + 1) * self.cp.ngroups_local_tpcp * self.d_state,
                self.cp.nheads_local_tpcp * (self.num_householder + 1),
            ],
            dim=-1,
        )

        # ``causal_conv1d_fn`` expects a ``[B, D, L]`` tensor.
        # But the expected memory layout varies depending on whether seq_idx is set.
        if seq_idx_packed is None:
            # Default path: channels-first contiguous, stride(2) == 1.
            VKQ = rearrange(VKQ, "b l d -> b d l").contiguous()
        else:
            # ``causal_conv1d_fn(seq_idx=...)`` requires channels-last memory but [B, D, L]
            # logical shape. This keeps the channels contiguous in memory.
            VKQ = VKQ.contiguous()
            VKQ = rearrange(VKQ, "b l d -> b d l")

        # Decode
        if inference_context is not None and inference_context.seqlen_offset > 0:
            VKQ = causal_conv1d_update(
                VKQ,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )
        else:
            # Prefill
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(
                    F.pad(VKQ, (self.d_conv - VKQ.shape[-1], 0))
                )  # Update state (B D W)
            # Train
            # causal_conv1d uses seq_idx_packed to reset the convolution boundaries
            VKQ = causal_conv1d_fn(
                x=VKQ,
                weight=rearrange(self.cp.get_conv1d_weight(), "d 1 w -> d w"),
                bias=self.cp.get_conv1d_bias(),
                activation=self.activation,
                seq_idx=seq_idx_packed,
            )

        VKQ = rearrange(VKQ, "b d l ->  b l d").contiguous()

        value, key, query = torch.split(
            VKQ,
            [
                self.cp.d_inner_local_tpcp * self.num_householder,
                self.cp.ngroups_local_tpcp * self.d_state * self.num_householder,
                self.cp.ngroups_local_tpcp * self.d_state,
            ],
            dim=-1,
        )

        b, a = torch.split(
            ba,
            [self.cp.nheads_local_tpcp * self.num_householder, self.cp.nheads_local_tpcp],
            dim=-1,
        )

        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim).contiguous()
        value = rearrange(
            value, "b l (m h p) -> b (l m) h p", m=self.num_householder, p=self.headdim
        ).contiguous()
        key = rearrange(
            key, "b l (m g n) -> b (l m) g n", m=self.num_householder, n=self.d_state
        ).contiguous()
        query = rearrange(query, "b l (g n) -> b l g n", n=self.d_state).contiguous()

        b, a = b.contiguous(), a.contiguous()
        beta = b.sigmoid()
        beta = rearrange(beta, "b l (m h) -> b (l m) h", m=self.num_householder).contiguous()

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.cp.get_A_log().float().exp() * F.softplus(a.float() + self.cp.get_dt_bias())

        if self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp > 1:
            query = query.repeat_interleave(
                self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp, dim=2
            )
            key = key.repeat_interleave(
                self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp, dim=2
            )

        # Decode
        if inference_context is not None and inference_context.seqlen_offset > 0:

            g_new = g.new_zeros(g.shape[0], g.shape[1], self.num_householder, g.shape[2])
            g_new[:, :, 0] = g
            g = rearrange(g_new, '... t n h -> ... (t n) h')

            query_new = query.new_zeros(
                query.shape[0], query.shape[1], self.num_householder, query.shape[2], query.shape[3]
            )
            query_new[:, :, -1] = query
            query = rearrange(query_new, '... t n h d-> ... (t n) h d')

            core_attn_out, last_recurrent_state = fused_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=ssm_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            core_attn_out = rearrange(
                core_attn_out, '... (t n) h d -> ... t n h d', n=self.num_householder
            )[..., -1, :, :].contiguous()
        # Train or Prefill
        else:
            core_attn_out, last_recurrent_state = chunk_gated_delta_product(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=(ssm_state is not None),
                num_householder=self.num_householder,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens_packed,
            )

        if ssm_state is not None:
            ssm_state.copy_(last_recurrent_state)

        y = rearrange(core_attn_out, "b l h p -> l b (h p)").contiguous()
        y = self.cp.post_conv_ssm(y, packed_seq_params=packed_seq_params)
        if self.rmsnorm:
            z = rearrange(z, "b l h p -> l b (h p)").contiguous()
            z = self.cp.post_conv_ssm(z, packed_seq_params=packed_seq_params)
            y = self.norm(y, z)

        out, out_bias = self.out_proj(y)

        return out, out_bias

    # ------------------------------------------------------------------
    # Dynamic-batching inference.
    #
    # Mirrors ``MambaMixer._dynamic_inference`` / ``_ssm_decode`` / ``_ssm_prefill``
    # (same ``_ssm_`` naming and the same request-level control flow), but runs
    # the Gated Delta Product kernels instead of the Mamba2 scan. The per-request
    # recurrent state (short-conv state + matrix-valued SSM state) is read/written
    # through the slot-indexed caches owned by ``DynamicInferenceContext``.
    #
    # MVP scope: this path does not yet support context parallelism (cp_size > 1),
    # speculative decoding, chunked prefill, Mamba prefix caching, or CUDA-graph
    # capture. The reshapes mirror the static ``forward`` math with batch/seq
    # repurposed for the packed dynamic layout.
    # ------------------------------------------------------------------
    def _dynamic_inference(
        self, hidden_states: torch.Tensor, context: DynamicInferenceContext
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute one dynamic inference step by separating decode and prefill
        requests, running each through the GDP kernels independently, and merging
        the results back into packed token order."""
        ok, reason = check_fla_sequence_packing_support()
        assert ok, reason
        assert self.cp.cp_size == 1, "Context parallel is not supported for GDP dynamic inference"
        assert (
            not context.is_chunked_prefill_enabled()
        ), "GDP dynamic inference does not support chunked prefill yet."

        # GDP-style layers register as Mamba layers, so the same (conv_state,
        # ssm_state) accessor and per-layer slab layout apply.
        conv_state, ssm_state = context.mamba_states_cache(self.layer_number - self.pp_layer_offset)

        padded_dims = context.padded_batch_dimensions
        token_count = padded_dims.token_count
        decode_req_count = padded_dims.decode_req_count
        prefill_req_count = padded_dims.prefill_req_count
        metadata = context.mamba_metadata

        # Input projection over the full packed batch.
        zVKQba, _ = self.in_proj(hidden_states)

        y_decode = None
        y_prefill = None

        # --- Decode partition (placed first in the packed batch) ---------
        if decode_req_count > 0:
            # MVP: exactly one token per decode request (no speculative tokens).
            zVKQba_decode = zVKQba[:decode_req_count] if prefill_req_count > 0 else zVKQba
            y_decode = self._ssm_decode(
                zVKQba_decode.transpose(0, 1), conv_state, ssm_state, metadata.batch_indices_decode
            ).transpose(0, 1)

        # --- Prefill partition -------------------------------------------
        if prefill_req_count > 0:
            if decode_req_count > 0:
                # Mixed batch: gather the prefill tokens out of the packed tensor.
                zVKQba_prefill = torch.empty_like(zVKQba)
                tensor_get_slice_after(
                    zVKQba, zVKQba_prefill, metadata.device_decode_prefill, check_bounds=False
                )
            else:
                zVKQba_prefill = zVKQba
            y_prefill = self._ssm_prefill(
                zVKQba_prefill,
                conv_state=conv_state,
                ssm_state=ssm_state,
                seq_idx=metadata.seq_idx,
                cu_seqlens=metadata.cu_seqlens,
                batch_indices=metadata.batch_indices_prefill,
            )

        # --- Merge back into packed token order --------------------------
        if y_decode is not None and y_prefill is not None:
            y = torch.empty(
                [token_count, 1, y_prefill.shape[-1]],
                dtype=y_prefill.dtype,
                device=y_prefill.device,
            )
            tensor_merge(y_decode, y_prefill, metadata.device_decode_prefill, output_tensor=y)
        elif y_decode is not None:
            y = y_decode
        elif y_prefill is not None:
            y = y_prefill
        else:
            raise RuntimeError("Dynamic inference called with 0 decode and 0 prefill requests")

        # Zero padding positions to avoid corrupting quantization amax calculations.
        if is_using_quantization_scales(self.config):
            y[context.padding_slice] = 0.0

        out, out_bias = self.out_proj(y)
        return out, out_bias

    def _ssm_decode(
        self,
        zVKQba: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single-token-per-request decode. ``zVKQba`` is ``[1, decode_req_count,
        proj_dim]``; returns ``[1, decode_req_count, d_inner]``. The conv and SSM
        states are read/written in place at the slots named by ``batch_indices``
        (``-1`` marks padding slots)."""
        seq_len, _, _ = zVKQba.shape
        assert seq_len == 1, "GDP decode supports one token per request"
        zVKQba = zVKQba.squeeze(0)  # [n, proj_dim]
        M = self.num_householder

        z, VKQ, ba = torch.split(
            zVKQba,
            [
                self.cp.d_inner_local_tpcp,
                self.cp.d_inner_local_tpcp * M
                + (M + 1) * self.cp.ngroups_local_tpcp * self.d_state,
                self.cp.nheads_local_tpcp * (M + 1),
            ],
            dim=-1,
        )

        # Indexed conv update: reads/writes the per-request conv state rows
        # selected by ``batch_indices``, in place. ``self.activation`` must be the
        # activation *string* so the kernel enables SiLU (a bool would disable it).
        VKQ = causal_conv1d_update(
            VKQ,
            conv_state,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.activation,
            conv_state_indices=batch_indices,
        )

        value, key, query = torch.split(
            VKQ,
            [
                self.cp.d_inner_local_tpcp * M,
                self.cp.ngroups_local_tpcp * self.d_state * M,
                self.cp.ngroups_local_tpcp * self.d_state,
            ],
            dim=-1,
        )
        b, a = torch.split(ba, [self.cp.nheads_local_tpcp * M, self.cp.nheads_local_tpcp], dim=-1)

        # Reshape to the fla layout with batch=n requests, seq length 1, and the
        # householder copies folded into the sequence dimension (static path, l=1).
        value = rearrange(value, "n (m h p) -> n m h p", m=M, p=self.headdim).contiguous()
        key = rearrange(key, "n (m g s) -> n m g s", m=M, s=self.d_state).contiguous()
        query = rearrange(query, "n (g s) -> n 1 g s", s=self.d_state).contiguous()
        z = rearrange(z, "n (h p) -> n 1 h p", p=self.headdim).contiguous()
        beta = rearrange(b.sigmoid(), "n (m h) -> n m h", m=M).contiguous()
        g = -self.cp.get_A_log().float().exp() * F.softplus(a.float() + self.cp.get_dt_bias())
        g = rearrange(g, "n h -> n 1 h").contiguous()

        if self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp > 1:
            rep = self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        # Interleave the (length-1) query / decay with householder zeros so the
        # recurrent kernel sees an (1 * M)-length sequence (matches static decode).
        g_new = g.new_zeros(g.shape[0], g.shape[1], M, g.shape[2])
        g_new[:, :, 0] = g
        g = rearrange(g_new, "n t m h -> n (t m) h")
        query_new = query.new_zeros(
            query.shape[0], query.shape[1], M, query.shape[2], query.shape[3]
        )
        query_new[:, :, -1] = query
        query = rearrange(query_new, "n t m h d -> n (t m) h d")

        # Gather this step's per-request initial states. ``.clamp`` (NOT in-place)
        # returns a new tensor, so ``batch_indices`` keeps its -1 padding sentinels
        # for the scatter below; the padding rows' outputs are never scattered back.
        gather_idx = batch_indices.clamp(min=0)
        initial_state = ssm_state[gather_idx]

        core_attn_out, last_recurrent_state = fused_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out = rearrange(core_attn_out, "n (t m) h d -> n t m h d", m=M)[
            ..., -1, :, :
        ].contiguous()  # [n, 1, h, d]

        # Scatter updated states back into the cache (skips -1 padding slots).
        tensor_masked_update(ssm_state, batch_indices, last_recurrent_state)

        y = rearrange(core_attn_out, "n t h p -> t n (h p)").contiguous()  # [1, n, d_inner]
        if self.rmsnorm:
            z = rearrange(z, "n t h p -> t n (h p)").contiguous()
            y = self.norm(y, z)
        return y

    def _ssm_prefill(
        self,
        zVKQba: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Variable-length prefill over all prefill requests in one varlen call.
        ``zVKQba`` is ``[l, 1, proj_dim]``; returns ``[l, 1, d_inner]``. Fresh
        requests start from a zero recurrent state (no prefix caching in the MVP);
        the resulting final conv/SSM states are written back into the caches."""
        is_dynamic_batching = seq_idx is not None
        M = self.num_householder

        # l b d -> b l d
        zVKQba = rearrange(zVKQba, "l b d -> b l d").contiguous()

        z, VKQ, ba = torch.split(
            zVKQba,
            [
                self.cp.d_inner_local_tpcp,
                self.cp.d_inner_local_tpcp * M
                + (M + 1) * self.cp.ngroups_local_tpcp * self.d_state,
                self.cp.nheads_local_tpcp * (M + 1),
            ],
            dim=-1,
        )

        if conv_state is not None and is_dynamic_batching:
            assert batch_indices is not None
            # Capture per-request final conv states (before the conv consumes the
            # inputs) and write them into the prefill requests' cache rows.
            conv_varlen_states = causal_conv1d_varlen_states(
                VKQ.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
            )
            tensor_masked_update(conv_state, batch_indices, conv_varlen_states)
            # Maintain channels-last memory layout so causal_conv1d_fn can use seq_idx.
            VKQ = VKQ.transpose(1, 2)
        else:
            VKQ = rearrange(VKQ, "b l d -> b d l").contiguous()

        seqlen = VKQ.size(2)
        if causal_conv1d_fn is None:
            VKQ = self.act(self.cp.conv1d(VKQ)[..., :seqlen])
        else:
            VKQ = causal_conv1d_fn(
                x=VKQ,
                weight=rearrange(self.cp.get_conv1d_weight(), "d 1 w -> d w"),
                bias=self.cp.get_conv1d_bias(),
                activation=self.activation,
                seq_idx=seq_idx,
            )
        VKQ = rearrange(VKQ, "b d l -> b l d").contiguous()

        value, key, query = torch.split(
            VKQ,
            [
                self.cp.d_inner_local_tpcp * M,
                self.cp.ngroups_local_tpcp * self.d_state * M,
                self.cp.ngroups_local_tpcp * self.d_state,
            ],
            dim=-1,
        )
        b, a = torch.split(ba, [self.cp.nheads_local_tpcp * M, self.cp.nheads_local_tpcp], dim=-1)

        # batch = 1 packed sequence of length T; householder folded into seq.
        value = rearrange(value, "b l (m h p) -> b (l m) h p", m=M, p=self.headdim).contiguous()
        key = rearrange(key, "b l (m g s) -> b (l m) g s", m=M, s=self.d_state).contiguous()
        query = rearrange(query, "b l (g s) -> b l g s", s=self.d_state).contiguous()
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim).contiguous()
        beta = rearrange(b.sigmoid(), "b l (m h) -> b (l m) h", m=M).contiguous()
        g = -self.cp.get_A_log().float().exp() * F.softplus(a.float() + self.cp.get_dt_bias())
        g = g.contiguous()

        if self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp > 1:
            rep = self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        core_attn_out, last_recurrent_state = chunk_gated_delta_product(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=ssm_state is not None,
            num_householder=M,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        # Write per-request final SSM states into the cache for subsequent decode.
        if ssm_state is not None and is_dynamic_batching:
            tensor_masked_update(ssm_state, batch_indices, last_recurrent_state)

        y = rearrange(core_attn_out, "b l h p -> l b (h p)").contiguous()
        if self.rmsnorm:
            z = rearrange(z, "b l h p -> l b (h p)").contiguous()
            y = self.norm(y, z)
        return y

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        allocate inference cache
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.conv1d.weight.shape[0], self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.nheads_local_tp,
            self.d_state,
            self.headdim,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def mamba_state_shapes_per_request(self) -> Tuple[Tuple[int], Tuple[int]]:
        """Returns the Mamba conv and SSM state shapes per request."""
        conv_states_shape = (self.conv1d.weight.shape[0], self.d_conv)
        ssm_states_shape = (self.nheads_local_tp, self.d_state, self.headdim)
        return (conv_states_shape, ssm_states_shape)

    def _get_states_from_cache(self, inference_context, batch_size, *, inference_params=None):
        """Initializes or retrieves the SSM state tensors from the cache.

        At the start of any inference (at the prefill step), if there is no cache or if the
        cached batch size has changed, then new tensors are initialized and stored in the cache.
        Otherwise the existing tensors are retrieved from the cache and zeroed out.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        assert inference_context is not None
        assert self.layer_number is not None
        if (
            self.layer_number not in inference_context.key_value_memory_dict
            or batch_size != self.cached_batch_size
        ):
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads_local_tp,
                self.d_state,
                self.headdim,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_context.key_value_memory_dict[self.layer_number] = (conv_state, ssm_state)
            self.cached_batch_size = batch_size
        else:
            conv_state, ssm_state = inference_context.key_value_memory_dict[self.layer_number]
            # TODO: Remove reference to `inference_context.sequence_len_offset` for dynamic batching
            if inference_context.sequence_len_offset == 0:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
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
                "D": 0,
            },  # parameters sharded across TP
            sharded_offsets=sharded_offsets,
            tp_group=self.pg_collection.tp,
            dp_cp_group=metadata["dp_cp_group"],
        )
        # Submodules
        for name, module in self.named_children():
            if name == "conv1d":
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix="", keep_vars=True)
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    {"weight": 0, "bias": 0},
                    sharded_offsets,
                    tp_group=self.pg_collection.tp,
                    dp_cp_group=metadata["dp_cp_group"],
                )

            else:
                module_sharded_sd = sharded_state_dict_default(
                    module,
                    f"{prefix}{name}.",
                    sharded_offsets,
                    metadata,
                    tp_group=self.pg_collection.tp,
                )

            sharded_state_dict.update(module_sharded_sd)

        # At this point the TP sharding is correctly defined for each tensor, but some of the
        # tensors must be additionally split into separate parts
        in_proj_dim = (
            self.d_inner_local_tp * (1 + self.num_householder)
            + (1 + self.num_householder) * self.ngroups_local_tp * self.d_state
            + self.nheads_local_tp * (1 + self.num_householder)
        )
        # Under GTP, in_proj.weight is GTP-sliced along axis 0. The [z|V*|K*|Q|b*|a] split
        # boundaries don't line up with GTP slice boundaries, so gather the shards back to
        # TP-local size (strip the trailing pad rows from the gathered tail) and fall through
        # to the same split path the non-GTP run uses — saved ckpt matches a non-GTP run.
        in_proj_gtp_remat_size = getattr(self.in_proj.weight, "gtp_remat_size", 1)
        in_proj_is_gtp = (
            in_proj_gtp_remat_size > 1 and HAVE_GTP and is_gtp_param(self.in_proj.weight)
        )
        if in_proj_is_gtp:
            # Same gather/split contract MambaMixer and GatedDeltaNet go through; the helper
            # also folds gtp_rank into replica_id, which a per-module copy of this block kept
            # getting wrong (two GTP peers electing themselves DCP writer for one chunk).
            from megatron.core.tensor_parallel.gtp_ckpt import _gtp_gather_rows_for_save

            sharded_state_dict[f"{prefix}in_proj.weight"] = _gtp_gather_rows_for_save(
                sharded_state_dict[f"{prefix}in_proj.weight"],
                f"{prefix}in_proj.weight",
                self.in_proj.weight,
                in_proj_dim,
                self.pg_collection.tp,
                metadata["dp_cp_group"],
                sharded_offsets,
            )

        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim, (
            in_proj_dim,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )

        # V, K, and b are laid out householder-major on every TP rank:
        #
        #   rank r: [M0-local-r, M1-local-r, ..., M(M-1)-local-r]
        #
        # Treating the entire M-expanded block as one TP shard would make a
        # resharded TP=1 tensor rank-major instead:
        #
        #   [rank0-all-M, rank1-all-M, ...]
        #
        # That ordering is incompatible with the forward rearranges, which
        # expect [M0-all-ranks, M1-all-ranks, ...].  Give every householder
        # copy its own checkpoint key so DCP concatenates TP shards within a
        # copy before the copies are concatenated by the factory merge.
        in_proj_split_sections, in_proj_split_names = _get_in_proj_checkpoint_split_layout(
            self.d_inner_local_tp,
            self.ngroups_local_tp * self.d_state,
            self.nheads_local_tp,
            self.num_householder,
        )
        for in_proj_param in ["in_proj.weight", "in_proj.bias"]:
            key = f"{prefix}{in_proj_param}"
            if key in sharded_state_dict:
                sharded_state_dict[key] = _split_tensor_factory(
                    sharded_state_dict[key], in_proj_split_sections, in_proj_split_names, 0
                )

        # GTP load-side inverse of the save-time all-gather (see
        # docs/api-guide/core/generalized_tensor_parallel.md §3.3, in_proj note): the checkpoint
        # stores the FULL TP-local in_proj.weight (pad stripped) under the per-householder split
        # keys, so the default merge_fn cats them back to ``in_proj_dim`` rows with no
        # padding. To reload into the live GTP param we must mirror init
        # (``_gtp_slice_one_param``): F.pad the merged tensor with zeros up to
        # ``gtp_remat_local_size * gtp_remat_size``, then slice by ``gtp_remat_local_rank``.
        # gtp_remat_size=1 has no pad/slice.
        if in_proj_is_gtp:
            from megatron.core.tensor_parallel.gtp_ckpt import _gtp_slice_rows_on_load

            sharded_state_dict[f"{prefix}in_proj.weight"] = _gtp_slice_rows_on_load(
                sharded_state_dict[f"{prefix}in_proj.weight"], self.in_proj.weight
            )

        conv_dim = (
            self.d_inner_local_tp * self.num_householder
            + (1 + self.num_householder) * self.ngroups_local_tp * self.d_state
        )
        assert sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0) == conv_dim, (
            conv_dim,
            sharded_state_dict[f"{prefix}conv1d.weight"],
        )

        conv_split_sections, conv_split_names = _get_conv_checkpoint_split_layout(
            self.d_inner_local_tp, self.ngroups_local_tp * self.d_state, self.num_householder
        )
        for conv_param in ["conv1d.weight", "conv1d.bias"]:
            key = f"{prefix}{conv_param}"
            if key in sharded_state_dict:
                sharded_state_dict[key] = _split_tensor_factory(
                    sharded_state_dict[key], conv_split_sections, conv_split_names, 0
                )

        return sharded_state_dict


def _get_in_proj_checkpoint_split_layout(
    d_inner_local_tp: int, group_state_local_tp: int, nheads_local_tp: int, num_householder: int
) -> Tuple[List[int], List[str]]:
    """Return TP-reshardable splits for the packed ``[z,V,K,Q,b,a]`` projection."""
    sections = (
        [d_inner_local_tp]
        + [d_inner_local_tp] * num_householder
        + [group_state_local_tp] * num_householder
        + [group_state_local_tp]
        + [nheads_local_tp] * num_householder
        + [nheads_local_tp]
    )
    names = (
        ["z"]
        + [f"V{i}" for i in range(num_householder)]
        + [f"K{i}" for i in range(num_householder)]
        + ["Q"]
        + [f"b{i}" for i in range(num_householder)]
        + ["a"]
    )
    return sections, names


def _get_conv_checkpoint_split_layout(
    d_inner_local_tp: int, group_state_local_tp: int, num_householder: int
) -> Tuple[List[int], List[str]]:
    """Return TP-reshardable splits for the packed ``[V,K,Q]`` convolution."""
    sections = (
        [d_inner_local_tp] * num_householder
        + [group_state_local_tp] * num_householder
        + [group_state_local_tp]
    )
    names = (
        [f"V{i}" for i in range(num_householder)]
        + [f"K{i}" for i in range(num_householder)]
        + ["Q"]
    )
    return sections, names


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
