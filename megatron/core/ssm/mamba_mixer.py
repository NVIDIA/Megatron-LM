# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
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

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.inference.contexts import BaseInferenceContext, DynamicInferenceContext
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import (
    check_mamba_sequence_packing_support,
    deprecate_inference_params,
    log_single_rank,
    maybe_cat,
)

from .mamba_context_parallel import MambaContextParallel

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )

    HAVE_MAMBA_SSM = True
except ImportError:
    from unittest.mock import MagicMock

    RMSNormGated = MagicMock()
    HAVE_MAMBA_SSM = False

try:
    from einops import rearrange, repeat

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

logger = logging.getLogger(__name__)


class ExtendedRMSNorm(RMSNormGated):
    """
    RMSNormGated with sharded state dict.
    """

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias not sharded"""
        if not hasattr(self, 'tp_group'):
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
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
class MambaMixerSubmodules:
    """
    Contains the module specs for the input and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = None
    out_proj: Union[ModuleSpec, type] = None


class MambaMixer(MegatronModule):
    """
    Args:
        config: The config of the model.
        submodules: Contains the module specs for the input and output linear layers.
        d_model: The hidden size of the model.
        d_state: The state size of the SSM.
        d_conv: The number of channels in the causal convolution.
        conv_init: The initialization range for the causal convolution weights.
        expand: The expansion factor for the SSM.
        headdim: The hidden size of each attention head.
        ngroups: The number of attention heads.
        A_init_range: The initialization range for the attention weights.
        D_has_hdim: Whether the D parameter has the same number of dimensions as the hidden
            state.
        rmsnorm: Whether to use root mean square normalization.
        norm_before_gate: Whether to apply normalization before the gating mechanism.
        dt_min: The minimum value of the dt parameter.
        dt_max: The maximum value of the dt parameter.
        dt_init: The initialization value of the dt parameter.
        dt_scale: The scaling factor for the dt parameter.
        dt_init_floor: The minimum value of the dt parameter after initialization.
        bias: Whether to use bias in the linear layers.
        conv_bias: Whether to use bias in the causal convolution.
        chunk_size: The chunk size for the fused kernel.
        use_mem_eff_path: Whether to use the memory-efficient path for the Mamba model.
        layer_number: The layer number of this Mamba layer.
        pg_collection: The required process groups to use for tensor model parallel and context
            parallel.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaMixerSubmodules,
        d_model,
        d_conv=4,
        conv_init=None,
        expand=2,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=128,
        layer_number=None,
        use_mem_eff_path=None,
        d_state=None,
        headdim=None,
        ngroups=None,
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_MAMBA_SSM:
            raise ImportError(
                "MambaSSM is not installed. Please install it with `pip install mamba-ssm`."
            )

        if not HAVE_EINOPS:
            raise ImportError("einops is required by the Mamba model but cannot be imported")

        super().__init__(config)
        self.config = config
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.chunk_size = chunk_size
        self.layer_number = layer_number
        self.cached_batch_size = None
        assert pg_collection is not None, "pg_collection must be provided for MambaMixer"
        self.pg_collection = pg_collection

        # Check for deprecated arguments and raise warnings
        if use_mem_eff_path is not None:
            warnings.warn(
                "The 'use_mem_eff_path' argument is deprecated and will be removed in the future. "
                "Please use the value from the TransformerConfig object instead.",
                DeprecationWarning,
            )
        if d_state is not None:
            warnings.warn(
                "The 'd_state' argument is deprecated and will be removed in the future. "
                "Please use the value from the TransformerConfig object instead.",
                DeprecationWarning,
            )
        if headdim is not None:
            warnings.warn(
                "The 'headdim' argument is deprecated and will be removed in the future. "
                "Please use the value from the TransformerConfig object instead.",
                DeprecationWarning,
            )
        if ngroups is not None:
            warnings.warn(
                "The 'ngroups' argument is deprecated and will be removed in the future. "
                "Please use the value from the TransformerConfig object instead.",
                DeprecationWarning,
            )

        self.use_mem_eff_path = self.config.use_mamba_mem_eff_path
        self.d_state = self.config.mamba_state_dim
        self.headdim = self.config.mamba_head_dim
        self.ngroups = self.config.mamba_num_groups

        assert self.d_state is not None and self.d_state > 0
        assert self.headdim is not None and self.headdim > 0
        assert self.ngroups is not None and self.ngroups > 0

        if self.config.mamba_num_heads is not None:
            self.nheads = self.config.mamba_num_heads
            assert self.nheads > 0
            self.d_inner = self.nheads * self.headdim
        else:
            assert self.d_inner % self.headdim == 0, "d_inner must be evenly divisible by headdim"
            self.nheads = self.d_inner // self.headdim

        if self.config.fp8:
            assert (2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads) % 16 == 0, (
                "For FP8, the innermost dimension of the Mamba layer "
                "input projection output tensor must be a multiple of 16."
            )

        tp_size = self.pg_collection.tp.size()

        # Ensure that each TP rank gets at least one head:
        assert self.nheads % tp_size == 0, "nheads must be evenly divisble by tp_size"
        self.nheads_local_tp = self.nheads // tp_size

        # Note that we do not need to confirm that `d_inner % tp_size == 0` because
        # `d_inner % headdim == 0`, `nheads = d_inner // headdim`, and `nheads % tp_size == 0`
        self.d_inner_local_tp = self.d_inner // tp_size

        # Ensure that each TP rank gets at least one group:
        assert self.ngroups % tp_size == 0, "ngroups must be evenly divisible by tp_size"
        self.ngroups_local_tp = self.ngroups // tp_size

        # Ensure that each group has a positive integer number of heads:
        assert self.nheads % self.ngroups == 0, "nheads must be evenly divisible by ngroups"

        assert not bias
        assert not self.norm_before_gate

        # Assume sequence parallelism: input is already partitioned along the sequence dimension
        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            self.d_inner * 2 + 2 * self.ngroups * self.d_state + self.nheads,  # z x B C dt
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )

        if not self.use_mem_eff_path:
            log_single_rank(
                logger,
                logging.WARNING,
                (
                    "We are not currently using or functionally testing use_mem_eff_path==False "
                    "for training. It may not work as expected."
                ),
            )

        conv_dim = self.d_inner_local_tp + 2 * self.ngroups_local_tp * self.d_state  # x B C
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
        setattr(self.conv1d.bias, "tensor_model_parallel", True)

        if self.config.perform_initialization and self.conv_init is not None:
            with get_cuda_rng_tracker().fork():
                nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.activation = "silu"
        self.act = nn.SiLU()

        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
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
        else:
            inv_dt = torch.empty(
                self.nheads_local_tp, device=torch.cuda.current_device(), dtype=config.params_dtype
            )

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
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(
            self.nheads_local_tp, dtype=torch.float32, device=torch.cuda.current_device()
        )
        if self.config.perform_initialization:
            A = A.uniform_(*A_init_range)
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        setattr(self.A_log, "tensor_model_parallel", True)

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(
                self.d_inner_local_tp if self.D_has_hdim else self.nheads_local_tp,
                device=torch.cuda.current_device(),
            )
        )  # Keep in fp32
        self.D._no_weight_decay = True
        setattr(self.D, "tensor_model_parallel", True)

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
        )

        # Regarding `conv1d`.{`weight`, `bias`}, `dt_bias`, `A_log`, and `D`: these are the
        # trainable variables for the current tensor parallel rank, with each tensor parallel rank
        # having indepdendent trainable variables. All context parallel ranks in a tensor parallel
        # rank store the same trainable variables, but only use and update their unique/independent
        # slice of them.
        self.cp = MambaContextParallel(
            cp_group=self.pg_collection.cp,
            d_inner_local_tp=self.d_inner_local_tp,
            nheads_local_tp=self.nheads_local_tp,
            ngroups_local_tp=self.ngroups_local_tp,
            d_state=self.d_state,
            conv1d_cp1=self.conv1d,
            dt_bias_cp1=self.dt_bias,
            A_log_cp1=self.A_log,
            D_cp1=self.D,
            D_has_hdim=self.D_has_hdim,
        )
        self.tp_group = pg_collection.tp

    def forward(
        self,
        hidden_states,
        inference_context=None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        hidden_states: (nL, B, D) / (L B D)
        Returns: same shape as hidden_states
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        in_inference_mode = inference_context is not None and not self.training

        _, batch, dim = hidden_states.shape
        conv_state, ssm_state = None, None

        if in_inference_mode:
            if inference_context.is_dynamic_batching():
                return self.dynamic_inference(hidden_states, inference_context)
            else:
                assert inference_context.is_static_batching()
                assert not self.config.sequence_parallel
                conv_state, ssm_state = self._get_states_from_cache(inference_context, batch)
                if inference_context.seqlen_offset > 0:
                    # The states are updated inplace
                    out, out_bias = self.decode(hidden_states, conv_state, ssm_state)
                    return out, out_bias

        zxBCdt, _ = self.in_proj(hidden_states)

        zxBCdt = self.cp.pre_conv_ssm(zxBCdt)

        if in_inference_mode or not self.use_mem_eff_path:
            # TODO(ksanthanam): Consider deprecating this path for training
            y = self.ssm_prefill(zxBCdt, conv_state=conv_state, ssm_state=ssm_state)
        else:
            assert ssm_state is None
            y = self.ssm_training(zxBCdt)

        out, out_bias = self.out_proj(y)

        return out, out_bias

    def dynamic_inference(self, hidden_states: torch.Tensor, context: DynamicInferenceContext):
        """
        Executes dynamic inference by separating decode and prefill requests and
        running them independently. Also runs the chunked prefill request independently
        if it exists.
        """
        sequence_packing_available, reason_for_no_sequence_packing = (
            check_mamba_sequence_packing_support()
        )
        assert sequence_packing_available, reason_for_no_sequence_packing

        conv_state, ssm_state = context.mamba_states_cache(self.layer_number)

        # Fast path: decode-only
        if context.is_decode_only():
            batch_indices = context.mamba_metadata.request_to_mamba_state_idx_cudagraph_only[
                : context.padded_active_token_count
            ]
            out, out_bias = self.decode(
                hidden_states, conv_state, ssm_state, batch_indices=batch_indices
            )
            return out, out_bias

        # Compute input projection before splitting into prefill and decode
        # to ensure sequence parallel all-gather.
        zxBCdt, _ = self.in_proj(hidden_states)

        # Compute split between decode and prefill.
        seq_idx, cu_seqlens, return_varlen_states = self._get_varlen_generation_state(context)
        active_query_lengths = context.request_query_lengths[
            context.paused_request_count : context.total_request_count
        ]
        batch_indices = context.mamba_metadata.request_to_mamba_state_idx

        # First request with query len > 1 is prefill-start.
        first_prefill_token_idx = torch.nonzero(active_query_lengths > 1)[0].int()

        # Process decode requests if there are any.
        if first_prefill_token_idx > 0:
            zxBCdt_decode = zxBCdt[:first_prefill_token_idx]
            batch_indices_decode = batch_indices[:first_prefill_token_idx]
            y_decode = self.ssm_decode(
                zxBCdt_decode.transpose(0, 1), conv_state, ssm_state, batch_indices_decode
            ).transpose(0, 1)
        else:
            y_decode = None

        active_token_count = context.active_token_count
        active_request_count = context.get_active_request_count()
        padded_active_token_count = context.padded_active_token_count

        # Process the chunked prefill request if it exists.
        if context.chunked_prefill_request_id != -1:
            chunked_prefill_request_token_count = active_query_lengths[-1]
            zxBCdt_chunked_prefill = zxBCdt[
                active_token_count - chunked_prefill_request_token_count : active_token_count
            ]
            batch_index_chunked_prefill = batch_indices[context.chunked_prefill_request_id]

            y_prefill_chunked = self.ssm_prefill(
                zxBCdt_chunked_prefill,
                conv_state=conv_state[batch_index_chunked_prefill].unsqueeze(0),
                ssm_state=ssm_state[batch_index_chunked_prefill].unsqueeze(0),
                is_chunked_prefill=True,
            )

            # Remove the chunked prefill request from the request / token counts so
            # the subsequent prefill computation ignores the chunked prefill request.
            active_token_count -= chunked_prefill_request_token_count
            active_request_count -= 1
        else:
            y_prefill_chunked = None

        # Process non-chunked prefill requests if there are any.
        if (remaining_prefill_tokens := active_token_count - first_prefill_token_idx) > 0:
            zxBCdt_prefill = zxBCdt[first_prefill_token_idx:active_token_count]
            cu_seqlens_prefill = F.pad(
                cu_seqlens[first_prefill_token_idx + 1 : active_request_count + 1]
                - first_prefill_token_idx,
                (1, 0),
            )
            seq_idx_prefill = (
                seq_idx[:, first_prefill_token_idx:active_token_count] - first_prefill_token_idx
            )
            batch_indices_prefill = batch_indices[first_prefill_token_idx:active_request_count]

            y_prefill = self.ssm_prefill(
                zxBCdt_prefill,
                conv_state=conv_state,
                ssm_state=ssm_state,
                seq_idx=seq_idx_prefill,
                cu_seqlens=cu_seqlens_prefill,
                return_varlen_states=return_varlen_states,
                batch_indices=batch_indices_prefill,
            )
        else:
            y_prefill = None

        # Assemble the final output by concatenating the decode output,
        # non-chunked prefill output, and chunked prefill output together.
        y_prefill = maybe_cat(y_prefill, y_prefill_chunked, required=True)
        y = maybe_cat(y_decode, y_prefill, required=True)

        # Add padding tokens back if necessary. Note that we use the context active token count
        # in case we modified the local count for chunked prefill above.
        if (num_padding_tokens := padded_active_token_count - context.active_token_count) > 0:
            y = torch.cat((y, y.new_zeros(num_padding_tokens, *y.shape[1:])), dim=0)

        # The output projection will perform the sequence parallel reduce-scatter if necessary.
        out, out_bias = self.out_proj(y)

        return out, out_bias

    def decode(
        self, hidden_states, conv_state, ssm_state, batch_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs inference step for decoding."""
        # assert self.ngroups_local_tp == 1, "Only support ngroups=1 for inference for now"
        is_dynamic_batching = batch_indices is not None

        if not is_dynamic_batching:
            assert (
                hidden_states.shape[0] == 1
            ), "Only support decoding with 1 token at a time for now"

        # (1, b, d_model) -> (1, b, proj_dim)
        zxBCdt, _ = self.in_proj(hidden_states)

        # Make batch size leading dimension since that is 1
        if is_dynamic_batching:
            zxBCdt = zxBCdt.transpose(0, 1)

        assert self.cp.cp_size == 1, "Context parallel not supported for Mamba inferenece decode"

        y = self.ssm_decode(
            zxBCdt, conv_state=conv_state, ssm_state=ssm_state, batch_indices=batch_indices
        )

        # Restore sequence length as first dimension
        if is_dynamic_batching:
            y = y.transpose(0, 1)

        # y has shape (1, b, d_inner), which is what out_proj expects
        out, out_bias = self.out_proj(y)

        return out, out_bias

    def ssm_training(self, zxBCdt: torch.Tensor) -> torch.Tensor:
        """
        Performs SSM computation for training step.

        Uses the memory-efficient kernel `mamba_split_conv1d_scan_combined` which reduces the size
        of forward activations stored for backprop and therefore reduces memory pressure during
        training.
        """

        # transpose: l b pd --> b l pd
        zxBCdt = rearrange(zxBCdt, "l b d -> b l d").contiguous()

        # (nheads_local_tpcp)
        A = -torch.exp(self.cp.get_A_log().float())

        # TODO(duncan): Can this code be removed?
        if self.conv1d.bias is not None:
            self.conv1d.bias.data_ptr()

        y = mamba_split_conv1d_scan_combined(
            zxBCdt,
            rearrange(self.cp.get_conv1d_weight(), "d 1 w -> d w"),
            self.cp.get_conv1d_bias(),
            self.cp.get_dt_bias().float(),
            A,
            D=(
                rearrange(self.cp.get_D().float(), "(h p) -> h p", p=self.headdim)
                if self.D_has_hdim
                else self.cp.get_D()
            ),
            chunk_size=self.chunk_size,
            activation=self.activation,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.cp.ngroups_local_tpcp,
            norm_before_gate=self.norm_before_gate,
        )

        y = rearrange(y, "b l d -> l b d").contiguous()
        y = self.cp.post_conv_ssm(y)

        if self.rmsnorm:
            y = self.norm(y)

        return y

    def ssm_prefill(
        self,
        zxBCdt: torch.Tensor,
        conv_state: Optional[torch.Tensor],
        ssm_state: Optional[torch.Tensor],
        seq_idx: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        return_varlen_states: bool = False,
        batch_indices: Optional[torch.Tensor] = None,
        is_chunked_prefill: bool = False,
    ) -> torch.Tensor:
        """
        Performs SSM computation for inference prefill step.

        Args:
            zxBCdt: The input tensor of shape (l, b, d), which is a concatenation of
                z, x, B, C, and dt projections.
            conv_state: The convolution state tensor for inference.
            ssm_state: The selective scan state tensor for inference.
            seq_idx: A map from token index to request index for variable-length sequences.
            cu_seqlens: Cumulative sequence lengths for variable-length sequences.
            return_varlen_states: Whether to return variable-length states from the SSM kernel.
            batch_indices: A map from batch id to position in the Mamba state tensors for
                dynamic inference.
            is_chunked_prefill: Whether the request is a chunked prefill request.

        Returns:
            The output tensor of shape (l, b, d).
        """
        is_dynamic_batching = seq_idx is not None
        assert not (
            is_dynamic_batching and is_chunked_prefill
        ), "Cannot use chunked prefill with dynamic batching"

        # transpose: l b pd --> b l pd
        zxBCdt = rearrange(zxBCdt, "l b d -> b l d").contiguous()

        # (nheads_local_tpcp)
        A = -torch.exp(self.cp.get_A_log().float())

        z, xBC, dt = torch.split(
            zxBCdt,
            [
                self.cp.d_inner_local_tpcp,
                self.cp.d_inner_local_tpcp + 2 * self.cp.ngroups_local_tpcp * self.d_state,
                self.cp.nheads_local_tpcp,
            ],
            dim=-1,
        )

        # Compute short convolution
        if conv_state is not None and is_dynamic_batching:
            # xBC should have shape (b l d) for causal_conv1d_varlen_states
            assert batch_indices is not None
            conv_state[batch_indices] = causal_conv1d_varlen_states(
                xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
            )

            # Maintain channels-last memory layout to use seq_idx for causal_conv1d_fn
            # See https://github.com/Dao-AILab/causal-conv1d/blob/69e6dadc28b169a4c49cb86b586f64ee90242c70/csrc/causal_conv1d.cpp#L174 # pylint: disable=line-too-long
            xBC = xBC.transpose(1, 2)
        elif is_chunked_prefill:
            # Maintain channels-last memory layout to use initial_states for causal_conv1d_fn
            # See https://github.com/Dao-AILab/causal-conv1d/blob/69e6dadc28b169a4c49cb86b586f64ee90242c70/csrc/causal_conv1d.cpp#L200 # pylint: disable=line-too-long
            xBC = xBC.transpose(1, 2)
        else:
            # transpose: b l pd --> b pd l
            xBC = rearrange(xBC, "b l d -> b d l").contiguous()
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(
                    F.pad(xBC, (self.d_conv - xBC.shape[-1], 0))
                )  # Update state (B D W)

        seqlen = xBC.size(2)
        if causal_conv1d_fn is None:
            xBC = self.act(self.cp.conv1d(xBC)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            if is_chunked_prefill:
                initial_conv_state = (
                    conv_state[:, :, 1:].permute(0, 2, 1).contiguous().transpose(1, 2)
                )
            else:
                initial_conv_state = None
            xBC = causal_conv1d_fn(
                x=xBC,
                weight=rearrange(self.cp.get_conv1d_weight(), "d 1 w -> d w"),
                bias=self.cp.get_conv1d_bias(),
                activation=self.activation,
                seq_idx=seq_idx,
                initial_states=initial_conv_state,
            )

        # transpose b pd l --> b l pd
        xBC = rearrange(xBC, "b d l -> b l d").contiguous()

        x, B, C = torch.split(
            xBC,
            [
                self.cp.d_inner_local_tpcp,
                self.cp.ngroups_local_tpcp * self.d_state,
                self.cp.ngroups_local_tpcp * self.d_state,
            ],
            dim=-1,
        )

        # TODO Vijay: fuse most of the transposes with the GEMMS
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim).contiguous()
        dt = dt.contiguous()
        B = rearrange(B, "b l (g n) -> b l g n", n=self.d_state).contiguous()
        C = rearrange(C, "b l (g n) -> b l g n", n=self.d_state).contiguous()
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim).contiguous()

        # If `rmsnorm == False`, then the norm inside `mamba_chunk_scan_combined` will be used.
        # In this case, if `cp_size > 1` then that norm could be performed on less heads than if
        # `cp_size == 1` (groups of heads can be sharded across CP ranks), which would be
        # mathematically incorrect, and potentially arithmetically unstable.
        assert (
            self.cp.cp_size == 1 or self.rmsnorm
        ), "Context parallel not supported for use_mem_eff_path==False and rmsnorm==False"

        if is_chunked_prefill:
            initial_ssm_state = ssm_state
        else:
            initial_ssm_state = None

        # Note that both `seq_idx` and `cu_seqlens` must be passed in
        # for variable length generation.
        # See https://github.com/state-spaces/mamba/blob/e0761ece1db07e0949dd88b4f4cd440420a19fd9/tests/test_generation.py#L97 # pylint: disable=line-too-long
        y = mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            self.chunk_size,
            D=(
                rearrange(self.cp.get_D().float(), "(h p) -> h p", p=self.headdim)
                if self.D_has_hdim
                else self.cp.get_D()
            ),
            z=z if not self.rmsnorm else None,
            dt_bias=self.cp.get_dt_bias().float(),
            dt_softplus=True,
            return_final_states=ssm_state is not None,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            return_varlen_states=return_varlen_states,
            initial_states=initial_ssm_state,
        )

        if ssm_state is not None:
            if return_varlen_states:
                assert batch_indices is not None

                y, _, varlen_states = y

                # This has to be varlen_states, NOT last_state
                # See reference implementation:
                # https://github.com/state-spaces/mamba/blob/e0761ece1db07e0949dd88b4f4cd440420a19fd9/mamba_ssm/modules/mamba2.py#L267 # pylint: disable=line-too-long
                ssm_state[batch_indices] = varlen_states
            else:
                y, last_state = y
                ssm_state.copy_(last_state)

        y = rearrange(y, "b l h p -> l b (h p)").contiguous()
        y = self.cp.post_conv_ssm(y)

        if self.rmsnorm:
            z = rearrange(z, "b l h p -> l b (h p)").contiguous()
            z = self.cp.post_conv_ssm(z)
            y = self.norm(y, z)

        return y

    def ssm_decode(
        self,
        zxBCdt: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs SSM computation for inference decode step.

        Args:
            zxBCdt: The input tensor of shape (l, b, d), which is a concatenation of
                z, x, B, C, and dt projections. For decoding, l must be 1.
            conv_state: The convolution state tensor for inference.
            ssm_state: The selective scan state tensor for inference.
            batch_indices: A map from batch id to position in the Mamba state tensors for
                dynamic inference.

        Returns:
            The output tensor of shape (l, b, d).
        """
        seq_len, batch_size, _ = zxBCdt.shape
        dtype = zxBCdt.dtype
        assert seq_len == 1, "Only support decoding with 1 token at a time for now"

        # Remove sequence dimension
        zxBCdt = zxBCdt.squeeze(0)

        z, xBC, dt = torch.split(
            zxBCdt,
            [
                self.d_inner_local_tp,
                self.d_inner_local_tp + 2 * self.ngroups_local_tp * self.d_state,
                self.nheads_local_tp,
            ],
            dim=-1,
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=xBC.dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
                conv_state_indices=batch_indices,
            )

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner_local_tp,
                self.ngroups_local_tp * self.d_state,
                self.ngroups_local_tp * self.d_state,
            ],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())

        # SSM step
        if selective_state_update is None:
            if self.ngroups_local_tp > 1:
                B = rearrange(B, "b (g n) -> b g n", n=self.d_state)
                C = rearrange(C, "b (g n) -> b g n", n=self.d_state)
                B = repeat(
                    B, "b g n -> b (g h) n", h=self.d_inner_local_tp // self.ngroups_local_tp
                )
                C = repeat(
                    C, "b g n -> b (g h) n", h=self.d_inner_local_tp // self.ngroups_local_tp
                )

                dt = repeat(dt, "b h -> b (h p)", p=self.headdim)
                dt_bias = repeat(self.dt_bias, "h -> (h p)", p=self.headdim)
                A = repeat(A, "h -> (h p) n", p=self.headdim, n=self.d_state)
                D = repeat(self.D, "h -> (h p)", p=self.headdim)

                dt = F.softplus(dt + dt_bias.to(dtype=dt.dtype))
                dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))

                dB_x = torch.einsum("bd,bdn,bd->bdn", dt, B, x)
                ssm_state.copy_(
                    ssm_state * rearrange(dA, "b (h p) n -> b h p n", p=self.headdim)
                    + rearrange(dB_x, "b (h p) n -> b h p n", p=self.headdim)
                )

                y = torch.einsum(
                    "bdn,bdn->bd",
                    rearrange(ssm_state.to(dtype), "b h p n -> b (h p) n", p=self.headdim),
                    C,
                )
                y = y + D.to(dtype) * x
                if not self.rmsnorm:
                    y = y * self.act(z)  # (B D)
            else:
                # Discretize A and B (b (g n))
                dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
                dA = torch.exp(dt * A)
                x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
                dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
                ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
                y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
                y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
                y = rearrange(y, "b h p -> b (h p)")
                if not self.rmsnorm:
                    y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups_local_tp)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups_local_tp)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state,
                x_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=z if not self.rmsnorm else None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=batch_indices,
            )
            y = rearrange(y, "b h p -> b (h p)")

        if self.rmsnorm:
            y = self.norm(y, z)

        # Restore sequence dimension
        return y.unsqueeze(0)

    def _get_varlen_generation_state(
        self, inference_context: Optional[BaseInferenceContext] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Constructs the variable length generation state for non-decode dynamic inference.

        The returned state includes the following:
            `seq_idx` (Tensor): A map from token idx to request idx.
            `cu_seqlens` (Tensor): The cumulative sequence lengths.
            `return_varlen_states` (bool): Whether to return a varlen states tensor for
                `mamba_chunk_scan_combined`.

        Returns empty state for training, static inference, or decode-only dynamic inference.

        Args:
            inference_context (InferenceContext): The inference context.

        Returns:
            A tuple of (`seq_idx`, `cu_seqlens`, `return_varlen_states`)
        """

        if (
            inference_context is None
            or not inference_context.is_dynamic_batching()
            or inference_context.is_decode_only()
        ):
            return None, None, False

        active_token_count = inference_context.active_token_count
        seq_idx = (
            inference_context.token_to_request_idx[:active_token_count]
            .clone()
            .to(torch.int32)
            .unsqueeze(0)
        )

        # Get the list of cumulative sequence lengths for active requests.
        cu_seqlens, _ = inference_context.cu_query_lengths()

        return seq_idx, cu_seqlens, True

    def mamba_state_shapes_per_request(self) -> Tuple[Tuple[int], Tuple[int]]:
        """Returns the Mamba conv and ssm states shapes per request."""
        conv_states_shape = (self.conv1d.weight.shape[0], self.d_conv)
        ssm_states_shape = (self.nheads_local_tp, self.headdim, self.d_state)
        return (conv_states_shape, ssm_states_shape)

    def _get_states_from_cache(self, inference_context, batch_size, *, inference_params=None):
        """Initializes or retrieves the SSM state tensors from the cache.

        At the start of any inference (at the prefill step), if there is no cache or if the
        cached batch size has changed, then new tensors are initialized and stored in the cache.
        Otherwise the existing tensors are retrieved from the cache and zeroed out.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        assert inference_context is not None
        assert inference_context.is_static_batching()
        assert self.layer_number is not None

        if (
            self.layer_number not in inference_context.key_value_memory_dict
            or batch_size != self.cached_batch_size
        ):
            conv_state_shape, ssm_state_shape = self.mamba_state_shapes_per_request()
            conv_state = torch.zeros(
                batch_size,
                *conv_state_shape,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                *ssm_state_shape,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_context.key_value_memory_dict[self.layer_number] = (conv_state, ssm_state)
            self.cached_batch_size = batch_size
        else:
            conv_state, ssm_state = inference_context.key_value_memory_dict[self.layer_number]
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
        )
        # Submodules
        for name, module in self.named_children():
            if name == "conv1d":
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix="", keep_vars=True)
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    {f"weight": 0, f"bias": 0},
                    sharded_offsets,
                    tp_group=self.tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )

            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=self.tp_group
                )

            sharded_state_dict.update(module_sharded_sd)

        # At this point the TP sharding is correctly defined for each tensor, but some of the
        # tensors must be additionally split into separate parts
        in_proj_dim = (
            self.d_inner_local_tp * 2
            + 2 * self.ngroups_local_tp * self.d_state
            + self.nheads_local_tp
        )
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim, (
            in_proj_dim,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )

        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.d_inner_local_tp,
                self.d_inner_local_tp,
                self.ngroups_local_tp * self.d_state,
                self.ngroups_local_tp * self.d_state,
                self.nheads_local_tp,
            ],
            ["z", "x", "B", "C", "dt"],
            0,
        )

        conv_dim = self.d_inner_local_tp + 2 * self.ngroups_local_tp * self.d_state
        assert sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0) == conv_dim, (
            conv_dim,
            sharded_state_dict[f"{prefix}conv1d.weight"],
        )
        assert sharded_state_dict[f"{prefix}conv1d.bias"].data.size(0) == conv_dim, (
            conv_dim,
            sharded_state_dict[f"{prefix}conv1d.bias"],
        )

        for conv_layer_name in ["conv1d.weight", "conv1d.bias"]:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [
                    self.d_inner_local_tp,
                    self.ngroups_local_tp * self.d_state,
                    self.ngroups_local_tp * self.d_state,
                ],
                ["x", "B", "C"],
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
