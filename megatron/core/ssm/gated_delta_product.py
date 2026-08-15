# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import math
from dataclasses import dataclass, replace
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.inference.contexts import BaseInferenceContext, DynamicInferenceContext
from megatron.core.inference.contexts.attention_context.triton.tensor_ops import (
    tensor_masked_update,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gdp_context_parallel import GDPContextParallel
from megatron.core.ssm.packed_seq_helpers import (
    build_packed_seq_idx,
    check_fla_sequence_packing_support,
    get_cu_seqlens,
)
from megatron.core.ssm.ssm_inference import SSMDynamicInferenceMixin
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params

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
    from fla.modules.l2norm import l2_norm
    from fla.ops.gated_delta_product import chunk_gated_delta_product
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

try:
    from gdp_attn import chunk_gated_delta_product as cutedsl_chunk_gated_delta_product

    HAVE_CUTEDSL_GDP = True
except ImportError:
    HAVE_CUTEDSL_GDP = False


logger = logging.getLogger(__name__)


def _kernel_accepts_kwarg(kernel, name: str) -> bool:
    """Return True if `kernel` explicitly declares a keyword argument called `name`."""
    try:
        parameters = inspect.signature(kernel).parameters
    except (TypeError, ValueError):
        # Builtins / C extensions without introspectable signatures.
        return False
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class ExtendedRMSNorm(RMSNormGated):
    """
    RMSNormGated with sharded state dict.
    """

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias not sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0}, sharded_offsets
        )


@dataclass
class GatedDeltaProductMixerSubmodules:
    """
    Contains the module specs for the input and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = None
    out_proj: Union[ModuleSpec, type] = None


class GatedDeltaProductMixer(SSMDynamicInferenceMixin, MegatronModule):
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

        if config.gdp_cutedsl_kernel:
            if not HAVE_CUTEDSL_GDP:
                raise ImportError("gdp_attn (CuTeDSL GatedDeltaProduct) is not installed")
        elif not HAVE_FLA:
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

        # Select the chunked gated delta product kernel once. The CuTeDSL and FLA
        # implementations share the same keyword API but are imported under distinct
        # names so neither shadows the other when both packages are installed.
        self.gdp_kernel = (
            cutedsl_chunk_gated_delta_product
            if config.gdp_cutedsl_kernel
            else chunk_gated_delta_product
        )

        # Newer CuTeDSL kernel builds accept num_chunk_states_to_recompute; probe the
        # signature once so older builds that predate the argument still work.
        self.gdp_kernel_extra_kwargs = {}
        if _kernel_accepts_kwarg(self.gdp_kernel, "num_chunk_states_to_recompute"):
            self.gdp_kernel_extra_kwargs["num_chunk_states_to_recompute"] = (
                config.gdp_num_chunk_states_to_recompute
            )

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

        self.recompute_in_proj = (
            self.config.recompute_granularity == "selective"
            and "gdp_in_proj" in self.config.recompute_modules
        )

        self.recompute_qkv = (
            self.config.recompute_granularity == "selective"
            and "gdp_qkv" in self.config.recompute_modules
        )

        self.offload_gdp_qkv = (
            self.config.fine_grained_activation_offloading
            and "gdp_qkv" in self.config.offload_modules
        )

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
        inference_context = deprecate_inference_params(inference_context, inference_params)

        _, batch_size, _ = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_context is not None:
            if inference_context.is_dynamic_batching():
                ok, reason = check_fla_sequence_packing_support()
                assert ok, reason
                assert (
                    self.cp.cp_size == 1
                ), "Context parallel is not supported for GDP dynamic inference"
                assert not self.config.gdp_cutedsl_kernel, (
                    "Dynamic inference uses the FLA kernels directly in ssm_prefill/"
                    "ssm_decode, which expect the batched layout; gdp_cutedsl_kernel is "
                    "only supported for training and static-batching prefill."
                )
                return self.ssm_dynamic_inference(hidden_states, inference_context)
            assert (
                inference_context.is_static_batching()
            ), "GDP inference must be either static or dynamic batching."
            assert not self.config.sequence_parallel
            assert packed_seq_params is None, (
                "GDP does not currently support packed sequences during inference. "
                "Packing is only wired through the training/prefill (chunk) path."
            )
            conv_state, ssm_state = self._get_states_from_cache(inference_context, batch_size)
            if inference_context.seqlen_offset > 0:
                # The states are updated in place.
                return self._static_decode(hidden_states, conv_state, ssm_state)

        if packed_seq_params is not None:
            # ``hidden_states`` is [seq_len, batch, dim]; THD requires batch=1.
            assert batch_size == 1, "Packed sequences require batch=1 (THD/varlen format)."

        y = self._gdp_training(
            hidden_states,
            conv_state=conv_state,
            ssm_state=ssm_state,
            packed_seq_params=packed_seq_params,
        )

        out, out_bias = self.out_proj(y)

        return out, out_bias

    def _packed_metadata(self, packed_seq_params, pack_length):
        """Return the ``(seq_idx, cu_seqlens)`` pair describing packed (THD) sequences.

        ``seq_idx`` is built from ``pack_length`` rather than the pre-projection sequence
        length so that it matches the conv1d's input after in_proj's SP all-gather and
        ``pre_conv_ssm``'s CP all-to-all, regardless of SP/CP/TP upstream-slicing.
        Mirrors mamba_mixer.py, which calls _create_packed_seq_idx after the same gather
        points. Both are ``None`` when sequences are not packed.
        """
        if packed_seq_params is None:
            return None, None
        return build_packed_seq_idx(packed_seq_params, pack_length), get_cu_seqlens(
            packed_seq_params
        )

    def _gdp_training(self, hidden_states, conv_state=None, ssm_state=None, packed_seq_params=None):
        """Training forward: input projection, causal conv, QKV preparation, and the
        chunked gated delta product kernel, with optional in_proj/QKV recompute and
        fine-grained activation offloading of the causal conv input.

        Static-batching prefill runs the same computation and passes conv_state and
        ssm_state so the trailing conv window and the final recurrent state are cached
        for the decode steps.
        """
        if self.recompute_in_proj:
            # Checkpoint the input projection and its preprocessing, discard the z, VKQ,
            # and ba outputs after the forward pass, and recompute them in the backward
            # pass to save memory. Only the (much smaller) hidden_states input is kept.
            #
            # Quantized replay is safe for the same reason as in MLASelfAttention:
            # CheckpointWithoutOutput records the forward recipe and amax state and
            # replays under the recorded fp8_autocast, and the only quantized op inside
            # _in_proj_preprocess is in_proj itself. Everything downstream of it (causal
            # conv, gated delta product kernel) runs unquantized.
            quantization = self.config.fp8 or self.config.fp4
            in_proj_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            # ``packed_seq_params`` is bound rather than passed through ``checkpoint`` so
            # that only tensors reach ``ctx.save_for_backward``.
            z, VKQ, ba = in_proj_checkpoint.checkpoint(
                partial(self._in_proj_preprocess, packed_seq_params=packed_seq_params),
                hidden_states,
            )
        else:
            z, VKQ, ba = self._in_proj_preprocess(
                hidden_states, packed_seq_params=packed_seq_params
            )

        seq_idx_packed, cu_seqlens_packed = self._packed_metadata(packed_seq_params, VKQ.shape[1])

        # The offload group captures the tensors saved for backward inside the causal
        # conv and QKV preparation: the checkpoint's saved input VKQ when QKV recompute
        # is enabled, or the conv/QKV-prep internal saves otherwise.
        qkv_manager = off_interface(self.offload_gdp_qkv, VKQ, "gdp_qkv")
        with qkv_manager as VKQ:
            if self.recompute_qkv:
                # Checkpoint the causal conv and QKV preparation, discard the
                # query/key/value outputs after the forward pass, and recompute them
                # in the backward pass to save memory.
                qkv_checkpoint = tensor_parallel.CheckpointWithoutOutput()
                query, key, value = qkv_checkpoint.checkpoint(
                    partial(self._prepare_qkv, conv_state=conv_state, seq_idx=seq_idx_packed), VKQ
                )
            else:
                # _prepare_qkv also seeds conv_state during prefill
                query, key, value = self._prepare_qkv(
                    VKQ, conv_state=conv_state, seq_idx=seq_idx_packed
                )

        beta, g = self._compute_gating(ba)

        core_attn_out, last_recurrent_state = self._run_gdp_kernel(
            query,
            key,
            value,
            g,
            beta,
            VKQ,
            output_final_state=(ssm_state is not None),
            cu_seqlens=cu_seqlens_packed,
        )

        if ssm_state is not None:
            ssm_state.copy_(last_recurrent_state)

        if self.recompute_qkv:
            qkv_checkpoint.discard_output_and_register_recompute(core_attn_out)

        y = self._postprocess(core_attn_out, z, packed_seq_params=packed_seq_params)

        # Commit the offload group downstream of core_attn_out: the commit node's
        # backward, which waits on the reload event, then runs before the recompute
        # hook on core_attn_out unpacks the saved conv input.
        y = qkv_manager.group_offload(y, forced_released_tensors=[])

        if self.recompute_in_proj:
            # Hook on the mixer output so the recompute runs at the very start of this
            # mixer's backward: z is needed by the output norm, and VKQ is needed by the
            # causal conv backward or by the QKV recompute hook on core_attn_out, all of
            # which come later in the backward pass.
            in_proj_checkpoint.discard_output_and_register_recompute(y)

        return y

    def _run_gdp_kernel(self, query, key, value, g, beta, VKQ, output_final_state, cu_seqlens=None):
        """Run the selected chunked gated delta product kernel and return
        (core_attn_out, final_state).

        For the CuTeDSL path the kernel expects packed varlen sequences described by
        cu_seqlens and applies the query/key L2 norm itself, and its output is collapsed
        to (b l) h p and restored here; the FLA path uses the batched layout and has
        already applied the L2 norm in _prepare_qkv. ``cu_seqlens`` is the packed-sequence
        layout when THD packing is active; otherwise the CuTeDSL path synthesizes the
        uniform-length equivalent and the FLA path leaves it unset."""
        if self.config.gdp_cutedsl_kernel and cu_seqlens is None:
            # query/key/value are already collapsed to (b l) ..., so read the batch and
            # sequence dimensions from VKQ, which is still in (b, l, d) layout.
            b, l = VKQ.shape[0], VKQ.shape[1]
            cu_seqlens = torch.arange(0, (b + 1) * l, l, device=VKQ.device, dtype=torch.int32)

        core_attn_out, final_state = self.gdp_kernel(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            num_householder=self.num_householder,
            initial_state=None,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=self.config.gdp_cutedsl_kernel,
            cu_seqlens=cu_seqlens,
            **self.gdp_kernel_extra_kwargs,
        )
        if self.config.gdp_cutedsl_kernel:
            core_attn_out = rearrange(
                core_attn_out, "(b l) (h p) -> b l h p", b=VKQ.shape[0], p=self.headdim
            )
        return core_attn_out, final_state

    def _in_proj_preprocess(self, hidden_states, packed_seq_params=None):
        """Run the input projection, gather its output across CP ranks, switch to
        (b, l, d) layout, and split it into the z, VKQ, and ba groups.

        Kept as a single function so that it can be checkpointed as one unit: the in_proj
        output and the intermediate CP-gathered/transposed copies of it are then freed
        inside the checkpoint, leaving only hidden_states saved for the backward pass.
        """
        zVKQba, _ = self.in_proj(hidden_states)

        zVKQba = self.cp.pre_conv_ssm(zVKQba, packed_seq_params=packed_seq_params)

        return self._preprocess(zVKQba)

    def _preprocess(self, zVKQba):
        """Switch the (l, b, proj_dim) input projection to the batch-first layout the
        causal conv and the kernels expect, and split it into the z, VKQ, and ba groups.
        """
        zVKQba = rearrange(zVKQba, "l b d -> b l d").contiguous()

        return torch.split(
            zVKQba,
            [
                self.cp.d_inner_local_tpcp,
                self.cp.d_inner_local_tpcp * self.num_householder
                + (self.num_householder + 1) * self.cp.ngroups_local_tpcp * self.d_state,
                self.cp.nheads_local_tpcp * (self.num_householder + 1),
            ],
            dim=-1,
        )

    def _prepare_qkv(
        self, x, conv_state=None, seq_idx=None, is_decode=False, conv_state_indices=None
    ):
        """Run the causal conv on the VKQ slice and split/reshape it into query, key,
        and value.

        x: (b, l, d). Keep the transposes to the conv layout inside this function so that
        the checkpointed subgraph takes the (b, l, d) slice as input and hands back a
        gradient that is already contiguous in (b, l, d) for the split backward's concat.

        Decode passes ``is_decode`` to step the cached conv state one token at a time
        with ``causal_conv1d_update`` (l == 1), selecting per-request cache rows with
        ``conv_state_indices``.
        """
        # ``causal_conv1d_fn`` expects a ``[B, D, L]`` tensor in channels-last memory,
        # which is also what it requires when ``seq_idx`` is set. ``x`` is a view into
        # the channels-last ``zVKQba``, so the transpose alone already gives
        # stride(1) == 1 and no copy is needed on either path.
        x = rearrange(x, "b l d -> b d l")

        if is_decode:
            # Indexed conv update: reads/writes the per-request conv state rows selected
            # by ``conv_state_indices``, in place. ``self.activation`` must be the
            # activation *string* so the kernel enables SiLU (a bool would disable it).
            # ``causal_conv1d_update`` accepts (b, d) or (b, d, l); here l == 1.
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
                conv_state_indices=conv_state_indices,
            )
        else:
            if conv_state is not None:
                # Static-batching prefill: seed the conv state from the prompt's tail.
                # If we just take x[:, :, -self.d_conv :], it errors if seqlen < d_conv.
                # Instead F.pad pads with zeros if seqlen < d_conv, and truncates otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # state (B D W)
            if causal_conv1d_fn is None:
                seqlen = x.size(2)
                x = self.act(self.cp.conv1d(x)[..., :seqlen])
            else:
                # causal_conv1d uses seq_idx to reset the convolution boundaries
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.cp.get_conv1d_weight(), "d 1 w -> d w"),
                    bias=self.cp.get_conv1d_bias(),
                    activation=self.activation,
                    seq_idx=seq_idx,
                )

        x = rearrange(x, "b d l ->  b l d")

        value, key, query = torch.split(
            x,
            [
                self.cp.d_inner_local_tpcp * self.num_householder,
                self.cp.ngroups_local_tpcp * self.d_state * self.num_householder,
                self.cp.ngroups_local_tpcp * self.d_state,
            ],
            dim=-1,
        )

        if self.config.gdp_cutedsl_kernel:
            # value is not GQA-expanded, so flatten it straight to the kernel layout.
            # Keep query/key's group axis (g) separate so it can be GQA-expanded below;
            # they are collapsed to the flat (b l) (...) layout afterwards.
            value = rearrange(
                value, "b l (m h p) -> (b l) (m h p)", m=self.num_householder, p=self.headdim
            )
            key = rearrange(key, "b l (m g n) -> b l m g n", m=self.num_householder, n=self.d_state)
            query = rearrange(query, "b l (g n) -> b l g n", n=self.d_state)
        else:
            value = rearrange(
                value, "b l (m h p) -> b (l m) h p", m=self.num_householder, p=self.headdim
            ).contiguous()
            key = rearrange(
                key, "b l (m g n) -> b (l m) g n", m=self.num_householder, n=self.d_state
            ).contiguous()
            query = rearrange(query, "b l (g n) -> b l g n", n=self.d_state).contiguous()

        if not is_decode and not self.config.gdp_cutedsl_kernel:
            # The CuTeDSL path applies the L2 norm inside chunk_gated_delta_product
            # (use_qk_l2norm_in_kernel); apply it here for the FLA path so that it falls
            # inside the QKV recompute boundary. Decode defers it to the fused recurrent
            # kernel, which normalizes after the householder zeros are interleaved in.
            query = l2_norm(query)
            key = l2_norm(key)

        if self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp > 1:
            query = query.repeat_interleave(
                self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp, dim=-2
            )
            key = key.repeat_interleave(
                self.cp.nheads_local_tpcp // self.cp.ngroups_local_tpcp, dim=-2
            )

        if self.config.gdp_cutedsl_kernel:
            # Collapse batch/seq and the feature axes into the flat layout the kernel wants.
            key = rearrange(key, "b l m g n -> (b l) (m g n)")
            query = rearrange(query, "b l g n -> (b l) (g n)")

        return query, key, value

    def _compute_gating(self, ba):
        """Compute the beta and g gating tensors from the ba slice."""
        b, a = torch.split(
            ba,
            [self.cp.nheads_local_tpcp * self.num_householder, self.cp.nheads_local_tpcp],
            dim=-1,
        )

        b, a = b.contiguous(), a.contiguous()
        beta = b.sigmoid()

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.cp.get_A_log().float().exp() * F.softplus(a.float() + self.cp.get_dt_bias())

        if self.config.gdp_cutedsl_kernel:
            beta = rearrange(beta, "b l d -> (b l) d")
            g = rearrange(g, "b l h -> (b l) h")
        else:
            beta = rearrange(beta, "b l (m h) -> b (l m) h", m=self.num_householder).contiguous()

        return beta, g

    def _postprocess(self, core_attn_out, z, packed_seq_params=None):
        """Switch back to (l, b, d) layout, scatter across CP ranks, and apply the
        gated output norm."""
        y = rearrange(core_attn_out, "b l h p -> l b (h p)").contiguous()
        y = self.cp.post_conv_ssm(y, packed_seq_params=packed_seq_params)
        if self.rmsnorm:
            z = rearrange(z, "b l d -> l b d").contiguous()
            z = self.cp.post_conv_ssm(z, packed_seq_params=packed_seq_params)
            y = self.norm(y, z)
        return y

    # ==================================================================
    # Static / eager inference
    #
    # ``_static_decode`` implements legacy static-batching decode. It is
    # deliberately kept separate from the dynamic inference hooks below so that
    # static-batching bookkeeping does not pollute the interface defined by
    # ``SSMDynamicInferenceMixin``. Static-batching prefill shares the training
    # body in ``forward``, which seeds the conv/SSM state when the caches are
    # present. Mirrors ``MambaMixer._static_decode``.
    # ==================================================================
    def _static_decode(
        self, hidden_states: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-token static-batching decode step (updates state in place)."""
        assert hidden_states.shape[0] == 1, "Only support decoding with 1 token at a time for now"
        assert self.cp.cp_size == 1, "Context parallel not supported for GDP inference decode"
        assert not self.config.gdp_cutedsl_kernel, (
            "Decode uses the FLA fused_recurrent_gated_delta_rule and the batched tensor "
            "layout; gdp_cutedsl_kernel is only supported for training and prefill."
        )

        # (1, b, d_model) -> (1, b, proj_dim)
        zVKQba, _ = self.in_proj(hidden_states)

        # The decode kernels are batch-first: (1, b, proj_dim) -> (b, 1, proj_dim).
        # Static batching has no slot remapping, so batch_indices is None.
        y = self.ssm_decode(
            zVKQba.transpose(0, 1), conv_state=conv_state, ssm_state=ssm_state, batch_indices=None
        )

        # (b, 1, d_inner) -> (1, b, d_inner), which is what out_proj expects.
        return self.out_proj(y.transpose(0, 1))

    # ------------------------------------------------------------------
    # Dynamic-batching inference.
    #
    # These are the two hooks required by ``SSMDynamicInferenceMixin``; the
    # mixin owns the surrounding decode/prefill partitioning and merge. They run
    # the Gated Delta Product kernels instead of the Mamba2 scan. The
    # per-request recurrent state (short-conv state + matrix-valued SSM state)
    # is read/written through the slot-indexed caches owned by
    # ``DynamicInferenceContext``.
    #
    # MVP scope: this path does not yet support context parallelism (cp_size > 1),
    # speculative decoding, chunked prefill, Mamba prefix caching, or CUDA-graph
    # capture. The reshapes mirror the static ``forward`` math with batch/seq
    # repurposed for the packed dynamic layout.
    # ------------------------------------------------------------------
    def ssm_decode(
        self,
        zVKQba: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
        intermediate_conv_state: Optional[torch.Tensor] = None,
        intermediate_ssm_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single-token-per-request decode. ``zVKQba`` is ``[n, seq_len,
        proj_dim]``; returns ``[n, seq_len, d_inner]``. The conv and SSM states
        are read/written in place at the slots named by ``batch_indices``
        (``-1`` marks padding slots); ``batch_indices=None`` means static
        batching, where the caches are already in request order."""
        _, seq_len, _ = zVKQba.shape
        assert seq_len == 1, "GDP decode supports one token per request"
        assert (
            intermediate_conv_state is None and intermediate_ssm_state is None
        ), "GDP decode does not support speculative decoding yet"

        # Keep the length-1 sequence dimension so the shared helpers apply: with l == 1
        # their (b, l, ...) reshapes collapse to exactly the layouts the fla recurrent
        # kernel wants here, i.e. "b (l m) h p" is "n m h p" and "b l g n" is "n 1 g n".
        # ``_preprocess`` takes the sequence-first layout that in_proj produces, so hand
        # it back the (1, n, proj_dim) view; both transposes are free at l == 1.
        z, VKQ, ba = self._preprocess(zVKQba.transpose(0, 1))

        query, key, value = self._prepare_qkv(
            VKQ, conv_state=conv_state, is_decode=True, conv_state_indices=batch_indices
        )

        beta, g = self._compute_gating(ba)

        # Interleave the (length-1) query / decay with householder zeros so the recurrent
        # kernel sees an (1 * num_householder)-length sequence (matches static decode).
        g_new = g.new_zeros(g.shape[0], g.shape[1], self.num_householder, g.shape[2])
        g_new[:, :, 0] = g
        g = rearrange(g_new, "n t m h -> n (t m) h")
        query_new = query.new_zeros(
            query.shape[0], query.shape[1], self.num_householder, query.shape[2], query.shape[3]
        )
        query_new[:, :, -1] = query
        query = rearrange(query_new, "n t m h d -> n (t m) h d")

        if batch_indices is None:
            # Static batching: the cache rows are already in request order.
            initial_state = ssm_state
        else:
            # Gather this step's per-request initial states. ``.clamp`` (NOT in-place)
            # returns a new tensor, so ``batch_indices`` keeps its -1 padding sentinels
            # for the scatter below; the padding rows' outputs are never scattered back.
            initial_state = ssm_state[batch_indices.clamp(min=0)]

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
        core_attn_out = rearrange(
            core_attn_out, "n (t m) h d -> n t m h d", m=self.num_householder
        )[
            ..., -1, :, :
        ].contiguous()  # [n, 1, h, d]

        if batch_indices is None:
            ssm_state.copy_(last_recurrent_state)
        else:
            # Scatter updated states back into the cache (skips -1 padding slots).
            tensor_masked_update(ssm_state, batch_indices, last_recurrent_state)

        # ``_postprocess`` returns the sequence-first layout, so transpose back to the
        # batch-first [n, seq_len, d_inner] this method contracts to return; the
        # transpose is free at l == 1. post_conv_ssm inside it is a no-op here: decode
        # only runs at cp_size == 1.
        return self._postprocess(core_attn_out, z).transpose(0, 1)

    def ssm_prefill(
        self,
        zVKQba: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        context: DynamicInferenceContext,
    ) -> torch.Tensor:
        """Variable-length prefill over all prefill requests in one varlen call.
        ``zVKQba`` is ``[l, 1, proj_dim]``; returns ``[l, 1, d_inner]``. Fresh
        requests start from a zero recurrent state (no prefix caching in the MVP);
        the resulting final conv/SSM states are written back into the caches."""
        assert (
            not context.is_chunked_prefill_enabled()
        ), "GDP dynamic inference does not support chunked prefill yet."

        metadata = context.mamba_metadata
        seq_idx = metadata.seq_idx
        cu_seqlens = metadata.cu_seqlens
        batch_indices = metadata.batch_indices_prefill

        z, VKQ, ba = self._preprocess(zVKQba)

        # Capture per-request final conv states (before the conv consumes the
        # inputs) and write them into the prefill requests' cache rows. Done here
        # rather than via _prepare_qkv's conv_state argument, which seeds a single
        # static-batching window with F.pad instead of one row per request.
        conv_varlen_states = causal_conv1d_varlen_states(
            VKQ.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
        )
        tensor_masked_update(conv_state, batch_indices, conv_varlen_states)

        query, key, value = self._prepare_qkv(VKQ, seq_idx=seq_idx)

        beta, g = self._compute_gating(ba)

        core_attn_out, last_recurrent_state = self._run_gdp_kernel(
            query, key, value, g, beta, VKQ, output_final_state=True, cu_seqlens=cu_seqlens
        )

        # Write per-request final SSM states into the cache for subsequent decode.
        tensor_masked_update(ssm_state, batch_indices, last_recurrent_state)

        # post_conv_ssm inside _postprocess is a no-op here: dynamic inference asserts
        # cp_size == 1.
        return self._postprocess(core_attn_out, z)

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
        )
        # Submodules
        for name, module in self.named_children():
            if name == "conv1d":
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix="", keep_vars=True)
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd, f"{prefix}{name}.", {f"weight": 0, f"bias": 0}, sharded_offsets
                )

            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata
                )

            sharded_state_dict.update(module_sharded_sd)

        # At this point the TP sharding is correctly defined for each tensor, but some of the
        # tensors must be additionally split into separate parts
        in_proj_dim = (
            self.d_inner_local_tp * (1 + self.num_householder)
            + (1 + self.num_householder) * self.ngroups_local_tp * self.d_state
            + self.nheads_local_tp * (1 + self.num_householder)
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
