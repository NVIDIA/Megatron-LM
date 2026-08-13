# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import

import logging
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Callable, Optional, Protocol, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.context_parallel_layout import (
    contiguous_to_zigzag_chunks,
    zigzag_to_contiguous_chunks,
)
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams, resolve_cp_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_context_parallel import (
    _all_to_all_cp2hp,
    _all_to_all_hp2cp,
    _redo_attention_load_balancing,
    _undo_attention_load_balancing,
)
from megatron.core.ssm.utils import _split_tensor_factory
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

try:
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.cp import build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    causal_conv1d = None
    l2norm = None
    build_cp_context = None
    chunk_gated_delta_rule = None

    HAVE_FLA = False

logger = logging.getLogger(__name__)


@dataclass
class GatedDeltaNetSubmodules:
    """
    Contains the module specs for the input linear, output norm, and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class GatedDeltaRuleInterface(Protocol):
    """
    Unified typing protocol for linear attention interfaces, compliant to upstream FLA interfaces.

    Only ``q``/``k``/``v``/``g`` are common to every kernel, and only as keywords: each
    variant inserts its own gates after ``g`` (e.g., ``beta`` for GDN, ``b``/``w`` for GDN2).
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        *,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cp_context: object | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


class _GDNBase(MegatronModule):
    """Common base class for the Gated Delta Net (GDN) family of layers.

    Hosts everything the GDN variants share: the fused input projection, causal
    convolution on q/k/v, the CP all-to-all plumbing, the kernel-input preparation
    skeleton, the gated output norm + projection, and sharded checkpointing.
    """

    # Human-readable variant name, used in error messages. Overridden by each subclass.
    variant_name: str = "GDN"

    dt_bias_dim: int
    a_log_dim: int
    in_proj_qkvg_dim: int
    in_proj_extra_dim: int
    in_proj_dim: int

    dt_bias: nn.Parameter
    A_log: nn.Parameter

    gated_delta_rule: GatedDeltaRuleInterface

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaNetSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: float | None = None,
        use_qk_l2norm: bool = True,
        A_init_range: tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
        *,
        name: str | None = None,
        cp_comm_type: str | None = None,
    ):
        """
        Args:
            config: The config of the model.
            submodules: Contains the module specs for the input and output linear layers.
            layer_number: The layer number of this GDN layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization in the kernel of the gated delta rule.
            A_init_range: The initialization range for the attention weights.
            pg_collection: The required process groups to use for tensor model parallel and context
                parallel.
            name (str | None): module instance name passed top-down from its paranet module
            cp_comm_type (Optional[str]): Accepted for TransformerLayer compatibility and
                ignored; GDN implements context parallelism with its own all-to-alls rather
                than the attention CP communication schemes.
        """
        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install it with "
                "`pip install flash-linear-attention[cuda]`."
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
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet"
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.cp_size = self.pg_collection.cp.size()
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1
        self.gdn_pre_gated_delta_rule_fusion = config.gdn_pre_gated_delta_rule_fusion

        # Attributes from config
        self.config = config
        if self.config.deterministic_mode and self.gdn_pre_gated_delta_rule_fusion:
            raise ValueError(
                "Pre-GDR fusion is non-deterministic, but deterministic_mode=True. "
                "Disable gdn_pre_gated_delta_rule_fusion or deterministic_mode."
            )
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

        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size
        self.num_k_heads_local_tp = self.num_key_heads // self.tp_size

        # Headwise CP uses head-parallel layout: each CP rank handles a slice of
        # heads. The static cp_size (== max dynamic cp_size) must evenly divide
        # the per-TP head counts so that every possible runtime cp_size also
        # divides. Chunkwise CP keeps heads local and does not need this split.
        if self.config.linear_cp_mode == "headwise":
            assert self.num_k_heads_local_tp % self.cp_size == 0, (
                f"GDN head-parallel CP requires the static (max) cp_size ({self.cp_size}) "
                f"to evenly divide num_key_heads per TP rank ({self.num_k_heads_local_tp}); "
                f"all runtime dynamic cp_size values divide the static one and so will also divide."
            )
            assert self.num_v_heads_local_tp % self.cp_size == 0, (
                f"GDN head-parallel CP requires the static (max) cp_size ({self.cp_size}) "
                f"to evenly divide num_value_heads per TP rank ({self.num_v_heads_local_tp}); "
                f"all runtime dynamic cp_size values divide the static one and so will also divide."
            )

        attrs_to_check = (
            "dt_bias_dim",
            "a_log_dim",
            "in_proj_extra_dim",
            "in_proj_split_names",
            "in_proj_split_sections",
            "feat_dim_split",
            "gated_delta_rule",
        )
        self._setup_variant_attrs()
        for attr in attrs_to_check:
            assert getattr(self, attr, None) is not None, f"Attribute {attr} for GDN is not set"
        # QK, V, gate, shared across all variants
        self.in_proj_qkvg_dim = self.qk_dim * 2 + self.v_dim * 2
        self.in_proj_dim = self.in_proj_qkvg_dim + self.in_proj_extra_dim

        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN layer "
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

        self.dt_bias = nn.Parameter(
            torch.empty(
                self.dt_bias_dim, dtype=self.config.params_dtype, device=torch.cuda.current_device()
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)

        self.A_log = nn.Parameter(
            torch.empty(
                self.a_log_dim, dtype=self.config.params_dtype, device=torch.cuda.current_device()
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        setattr(self.A_log, "partition_dim", 0)

        # Output layernorm before projection
        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.value_head_dim,
            eps=self.config.layernorm_epsilon,
        )
        self.recompute_norm_out = False
        self.norm_out_checkpoint = None
        if self.config.recompute_granularity == "selective":
            self.recompute_norm_out = "gdn_norm_out" in self.config.recompute_modules

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

        # Whole-module recompute: when "gdn" is in recompute_modules (selective granularity),
        # the entire GDN compute is wrapped in a normal checkpoint and recomputed in the
        # backward pass. This is coarser than ``recompute_norm_out`` above; enabling both
        # would nest a CheckpointWithoutOutput inside a recomputed region, so "gdn" wins.
        self.recompute_gdn = False
        if self.config.recompute_granularity == "selective" and self.config.recompute_modules:
            self.recompute_gdn = "gdn" in self.config.recompute_modules
        if self.recompute_gdn:
            self.recompute_norm_out = False

        # Cache for CP context objects consumed by FLA kernels. Rebuilding these per-forward
        # is unsafe under CUDA graph capture because build_cp_context allocates
        # fresh tensors whose memory pointers are baked into the captured graph;
        # on the next call those tensors are reallocated, leaving the replayed
        # graph pointing at stale memory. For non-packed (SBHD) input the
        # cu_seqlens is fully determined by the (static) global sequence length
        # and batch size, so we cache the (cu_seqlens, cp_context) pair keyed on
        # both values.
        self._chunkwise_cp_context_cache = {}

        self.reset_parameters()

    def _resolve_cp_groups(
        self,
        packed_seq_params: PackedSeqParams | None,
        pg_collection: ProcessGroupCollection = None,
    ) -> tuple[
        torch.distributed.ProcessGroup | None, torch.distributed.ProcessGroup | None, int, int
    ]:
        """Split the CP group into the headwise and chunkwise roles for this forward.

        The two linear-attention CP paths are mutually exclusive — whichever one is
        active owns the full CP group, and the other is given a size-1 group (None).
        The unused-path helpers already treat a None group as size 1, avoiding a
        costly and CUDA-graph-unsafe ``torch.distributed.new_group`` on every forward.

        Returns:
            (headwise group, chunkwise group, headwise size, chunkwise size).
        """
        base_cp_group = pg_collection.cp if pg_collection is not None else self.pg_collection.cp
        cp_group = resolve_cp_group(base_cp_group, packed_seq_params)
        if self.config.linear_cp_mode == "chunkwise":
            cp_group_chunkwise = cp_group
            cp_group_headwise = None
        elif self.config.linear_cp_mode == "headwise":
            cp_group_chunkwise = None
            cp_group_headwise = cp_group
        elif cp_group.size() == 1:
            cp_group_chunkwise = None
            cp_group_headwise = None
        else:
            raise ValueError(
                f"Unsupported linear_cp_mode {self.config.linear_cp_mode!r}; "
                "expected 'headwise' or 'chunkwise'."
            )
        cp_size_chunkwise = cp_group_chunkwise.size() if cp_group_chunkwise is not None else 1
        cp_size_headwise = cp_group_headwise.size() if cp_group_headwise is not None else 1
        return cp_group_headwise, cp_group_chunkwise, cp_size_headwise, cp_size_chunkwise

    def _build_chunkwise_cp_context(
        self,
        cu_seqlens_q: torch.Tensor | None,
        cp_group_chunkwise: torch.distributed.ProcessGroup | None,
        seq_len_global: int,
        batch: int,
    ) -> tuple[torch.Tensor | None, object | None]:
        """Build (or reuse) the chunkwise-CP context the FLA kernels consume.

        For non-packed (SBHD) input the cu_seqlens is fully determined by the static
        global sequence length and batch size, so both it and the resulting context are
        cached — reallocating them per forward breaks CUDA graph capture.

        Returns:
            The (possibly synthesized) cu_seqlens and the chunkwise CP context, or
            ``(cu_seqlens_q, None)`` when chunkwise CP is inactive.
        """
        if cp_group_chunkwise is None or cp_group_chunkwise.size() == 1:
            return cu_seqlens_q, None

        if cu_seqlens_q is None:
            cache_key = (seq_len_global, batch)
            cached = self._chunkwise_cp_context_cache.get(cache_key)
            if cached is None:
                cached_cu_seqlens = (
                    torch.arange(batch + 1, device=torch.cuda.current_device(), dtype=torch.long)
                    * seq_len_global
                )
                cached_ctx = build_cp_context(
                    cu_seqlens=cached_cu_seqlens,
                    group=cp_group_chunkwise,
                    conv1d_kernel_size=self.conv_kernel_dim,
                )
                cached = (cached_cu_seqlens, cached_ctx)
                self._chunkwise_cp_context_cache[cache_key] = cached
            return cached

        return cu_seqlens_q, build_cp_context(
            cu_seqlens=cu_seqlens_q,
            group=cp_group_chunkwise,
            conv1d_kernel_size=self.conv_kernel_dim,
        )

    def _setup_variant_attrs(self):
        """Set variant specifics on the module. Called once from ``__init__``.

        Must set:
        - ``in_proj_dim``
        - ``in_proj_split_names``
        - ``in_proj_split_sections``
        - ``feat_dim_split``
        - ``dt_bias_dim`` / ``a_log_dim`` (sizes of the gate parameters, which the
          base class creates after the conv1d module to preserve the original
          parameter registration order)
        - ``gated_delta_rule`` (the kernel callable).
        """
        raise NotImplementedError

    def _reset_dt_bias(self):
        """Initialize ``dt_bias``. Called from ``reset_parameters`` under the RNG tracker.

        Defaults to ones; variants whose kernel expects a different step-size
        parametrization override this.
        """
        torch.ones(
            self.dt_bias_dim,
            dtype=self.config.params_dtype,
            device=torch.cuda.current_device(),
            out=self.dt_bias.data,
        )

    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                if self.conv_init is not None:
                    nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
                self._reset_dt_bias()
                A = torch.empty(
                    self.A_log.shape[0],
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(torch.log(A))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        pg_collection: Optional[ProcessGroupCollection] = None,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Perform a forward pass through the GDN module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            packed_seq_params (Optional[PackedSeqParams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.
            pg_collection (Optional[ProcessGroupCollection]): Per-forward process groups,
                overriding the ones captured at construction time (dynamic CP).

        Return:
            (tuple[Tensor, Tensor]) GDN output and bias.
        """
        # TODO: Deal with attention_mask

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Route the CP group to either the headwise (Ulysses-style) path or the
        # chunkwise CP path according to config.linear_cp_mode.
        (cp_group_headwise, cp_group_chunkwise, cp_size_headwise, cp_size_chunkwise) = (
            self._resolve_cp_groups(packed_seq_params, pg_collection)
        )

        seq_len_local, batch, _ = hidden_states.shape
        seq_len_post_headwise = seq_len_local * self.sp_size * cp_size_headwise
        seq_len_global = seq_len_post_headwise * cp_size_chunkwise

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), f"{self.variant_name} does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError(f"{self.variant_name} does not support inference for now.")

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            assert batch == 1, "Packed sequence expects batch dimension to be 1"
            assert (
                not self.config.deterministic_mode
            ), "Packed sequence does not support deterministic mode."

            # Resolve cu_seqlens with alignment padding handling.
            # cu_seqlens in packed_seq_params is the global (pre-CP-split) cu_seqlens, so we
            # validate against the global sequence length.
            cu_seqlens_q = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_q_padded,
                packed_seq_params.cu_seqlens_q,
                seq_len_global,
                "cu_seqlens_q",
                cp_size=self.cp_size,
            )
            cu_seqlens_kv = self._resolve_cu_seqlens(
                packed_seq_params.cu_seqlens_kv_padded,
                packed_seq_params.cu_seqlens_kv,
                seq_len_global,
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

        cu_seqlens_q, chunkwise_cp_context = self._build_chunkwise_cp_context(
            cu_seqlens_q, cp_group_chunkwise, seq_len_global, batch
        )

        forward_compute = partial(
            self._forward_compute,
            batch=batch,
            seq_len=seq_len_post_headwise,
            cp_group_headwise=cp_group_headwise,
            cp_group_chunkwise=cp_group_chunkwise,
            cp_size_headwise=cp_size_headwise,
            cp_size_chunkwise=cp_size_chunkwise,
            cu_seqlens_q=cu_seqlens_q,
            packed_seq_params=packed_seq_params,
            chunkwise_cp_context=chunkwise_cp_context,
        )

        if self.recompute_gdn and self.training:
            return tensor_parallel.checkpoint(forward_compute, False, hidden_states)
        return forward_compute(hidden_states)

    def pre_gated_delta_rule(
        self,
        qkvz_extra,
        batch,
        seq_len,
        cp_size_headwise,
        cp_group_headwise,
        cu_seqlens_q=None,
        chunkwise_cp_context=None,
        packed_seq_params=None,
    ):
        """Build every gated-delta-rule kernel input from the projection output.

        This is the unfused counterpart of ``_fused_streamed_pre_gated_delta_rule``: it
        splits the projection, runs the causal convolution, and computes the variant's
        gates. Kept as a separate method so the fusion tests can compare the two paths
        directly against each other.

        Returns:
            (tuple[dict, Tensor]): the gated delta rule kernel keyword arguments, and the
            output gate (z), which the kernel does not consume.
        """
        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvz_extra = qkvz_extra.transpose(0, 1)

        # Split the tensor into q/k/v, gate (z), and the variant-specific gate features
        # (beta, alpha for GDN; f, b, w for GDN2)
        feat_dim_split = tuple(d // cp_size_headwise for d in self.feat_dim_split)
        qkv, gate, *gate_feats = torch.split(qkvz_extra, feat_dim_split, dim=-1)
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)

        # Convolution on qkv
        nvtx_range_push(suffix="conv1d")
        assert qkv.shape[1] == seq_len, f"Shape mismatch: {qkv.shape[1]=} != {seq_len=}"
        conv1d_weight, conv1d_bias = self._local_conv1d_params(cp_group_headwise)
        qkv = self._apply_causal_conv1d(
            qkv,
            conv1d_weight,
            conv1d_bias,
            cp_size_headwise,
            cu_seqlens_q,
            chunkwise_cp_context,
            packed_seq_params,
        )
        nvtx_range_pop(suffix="conv1d")

        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=cp_group_headwise)
        dt_bias_local_cp = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=cp_group_headwise)

        # Prepare all kernel inputs (split, reshape, L2 norm, gates, contiguous)
        nvtx_range_push(suffix="prepare_input_for_gated_delta_rule")
        kernel_inputs = self._prepare_input_for_gated_delta_rule(
            qkv,
            gate,
            A_log_local_cp,
            dt_bias_local_cp,
            batch,
            seq_len,
            *gate_feats,
            cp_size_headwise=cp_size_headwise,
        )
        gate = kernel_inputs.pop("gate")
        nvtx_range_pop(suffix="prepare_input_for_gated_delta_rule")

        return kernel_inputs, gate

    def _forward_compute(
        self,
        hidden_states: torch.Tensor,
        *,
        batch: int,
        seq_len: int,
        cp_group_headwise: torch.distributed.ProcessGroup | None,
        cp_group_chunkwise: torch.distributed.ProcessGroup | None,
        cp_size_headwise: int,
        cp_size_chunkwise: int,
        cu_seqlens_q: torch.Tensor | None,
        packed_seq_params: PackedSeqParams | None,
        chunkwise_cp_context,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Core GDN computation (in_proj -> conv1d -> gated_delta_rule -> gated norm -> out_proj).

        Split out of ``forward`` so the entire module can be wrapped in a recompute
        checkpoint when ``recompute_modules`` contains ``"gdn"`` (selective full-module
        recompute, normal checkpointing).

        Returns:
            (tuple[Tensor, Tensor | None]): output and output bias.
        """
        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvz_extra, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        # Chunkwise CP expects the contiguous-time chunk layout (rank r holds chunks
        # [2r, 2r+1]) inside conv1d / the gated delta rule kernel. Megatron attention CP
        # feeds us the zigzag attention-load-balanced layout (rank r holds
        # [r, 2*cp-r-1]), so reshuffle chunks over the CP group with a single
        # all-to-all — no full-sequence gather required.
        # TODO: Move CP layout ownership to a model/region-level scheduler so hybrid models can
        # enter contiguous layout before GDN regions instead of paying module-local conversions.
        if cp_size_chunkwise > 1:
            nvtx_range_push(suffix="zigzag_to_contiguous")
            if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
                qkvz_extra = zigzag_to_contiguous_chunks(
                    qkvz_extra, cp_group_chunkwise, seq_dim=0, cu_seqlens=cu_seqlens_q
                )
            else:
                qkvz_extra = zigzag_to_contiguous_chunks(qkvz_extra, cp_group_chunkwise, seq_dim=0)
            nvtx_range_pop(suffix="zigzag_to_contiguous")

        qkvz_extra, thd_cp_a2a_inv = a2a_cp_to_hp(
            qkvz_extra,
            self.in_proj_split_sections,
            cp_size_headwise,
            cp_group_headwise,
            cu_seqlens_q,
            seq_len,
            packed_seq_params,
        )

        if self.gdn_pre_gated_delta_rule_fusion:
            self._reject_unsupported_chunkwise_cp(cp_size_chunkwise, batch)
            nvtx_range_push(suffix="fused_streamed_pre_gated_delta_rule")
            is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
            seq_idx = packed_seq_params.seq_idx if is_thd and cp_size_chunkwise == 1 else None
            query, key, value, gate, beta, g = self._fused_streamed_pre_gated_delta_rule(
                qkvz_extra,
                cu_seqlens_q=cu_seqlens_q if is_thd else None,
                seq_idx=seq_idx,
                cp_group=cp_group_chunkwise if cp_size_chunkwise > 1 else None,
                cp_size_headwise=cp_size_headwise,
                cp_group_headwise=cp_group_headwise,
            )
            nvtx_range_pop(suffix="fused_streamed_pre_gated_delta_rule")
            kernel_inputs = {"q": query, "k": key, "v": value, "g": g, "beta": beta}
        else:
            self._reject_unsupported_chunkwise_cp(cp_size_chunkwise, batch)
            kernel_inputs, gate = self.pre_gated_delta_rule(
                qkvz_extra,
                batch,
                seq_len,
                cp_size_headwise,
                cp_group_headwise,
                cu_seqlens_q=cu_seqlens_q,
                chunkwise_cp_context=chunkwise_cp_context,
                packed_seq_params=packed_seq_params,
            )

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, _ = self.gated_delta_rule(
            **kernel_inputs,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens_q,
            cp_context=chunkwise_cp_context,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        norm_and_a2a = partial(
            self._gated_norm_and_a2a,
            thd_cp_a2a_inv=thd_cp_a2a_inv,
            batch=batch,
            seq_len=seq_len,
            packed_seq_params=packed_seq_params,
            cp_group_headwise=cp_group_headwise,
            cp_group_chunkwise=cp_group_chunkwise,
            cu_seqlens_q=cu_seqlens_q,
        )
        if self.recompute_norm_out:
            self.norm_out_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            norm_out = self.norm_out_checkpoint.checkpoint(norm_and_a2a, core_attn_out, gate)
        else:
            norm_out = norm_and_a2a(core_attn_out, gate)

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        if self.recompute_norm_out:
            self.norm_out_checkpoint.discard_output_and_register_recompute(out)

        return out, out_bias

    def _gated_norm_and_a2a(
        self,
        core_attn_out: torch.Tensor,
        gate: torch.Tensor,
        thd_cp_a2a_inv: torch.Tensor | None,
        batch: int,
        seq_len: int,
        packed_seq_params: PackedSeqParams | None = None,
        cp_group_headwise: torch.distributed.ProcessGroup | None = None,
        cp_group_chunkwise: torch.distributed.ProcessGroup | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # RMSNorm
        nvtx_range_push(suffix="gated_norm")
        norm_out_hp = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out_hp = norm_out_hp.reshape(batch, seq_len, -1)
        norm_out_hp = norm_out_hp.transpose(0, 1).contiguous()

        # Inverse of the zigzag -> contiguous reshuffle performed before conv1d.
        # Restores the Megatron attention-load-balanced layout that downstream
        # layers and loss computation expect.
        # TODO: The planned CP layout refactor should keep consecutive GDN layers contiguous and
        # restore zigzag only at SDPA/canonical-layout boundaries.
        if cp_group_chunkwise is not None and cp_group_chunkwise.size() > 1:
            nvtx_range_push(suffix="contiguous_to_zigzag")
            if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
                norm_out_hp = contiguous_to_zigzag_chunks(
                    norm_out_hp, cp_group=cp_group_chunkwise, seq_dim=0, cu_seqlens=cu_seqlens_q
                )
            else:
                norm_out_hp = contiguous_to_zigzag_chunks(
                    norm_out_hp, cp_group=cp_group_chunkwise, seq_dim=0
                )
            nvtx_range_pop(suffix="contiguous_to_zigzag")

        cp_size_headwise = cp_group_headwise.size() if cp_group_headwise is not None else 1
        return a2a_hp_to_cp(
            norm_out_hp, cp_size_headwise, cp_group_headwise, packed_seq_params, thd_cp_a2a_inv
        )

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
    def _prepare_input_for_gated_delta_rule(
        self,
        qkv: torch.Tensor,
        gate: torch.Tensor,
        A_log_local_cp: torch.Tensor,
        dt_bias_local_cp: torch.Tensor,
        batch: int,
        seq_len: int,
        *gate_feats: tuple[torch.Tensor],
        cp_size_headwise: int = 1,
    ) -> dict[str, torch.Tensor]:
        """
        Prepare all gated delta rule kernel inputs.

        Fuses split, reshape, L2 norm, decay/gate activations, repeat_interleave, and
        contiguous operations. ``gate_feats`` holds the variant-specific in_proj
        sections, which ``_compute_gates`` turns into the decay and gating tensors.

        ``cp_size_headwise`` is the headwise (Ulysses-style) CP world size, which is the
        factor the head dim was sharded by; chunkwise CP keeps every head local and so
        passes 1.

        Returns:
            (dict[str, Tensor]): Kernel inputs keyed by kernel argument name (``q``,
            ``k``, ``v``, ``g``, plus the variant-specific gates), and the output
            gate (z) tensor under the ``gate`` key, which is not a kernel input.
        """
        # Split qkv into query_key and value
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // cp_size_headwise, self.v_dim_local_tp // cp_size_headwise],
            dim=-1,
        )

        # Reshape query_key and value
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            query_key = l2norm(query_key.contiguous())

        # Split query and key
        split_size = self.qk_dim_local_tp // self.key_head_dim // cp_size_headwise
        query, key = torch.split(query_key, [split_size, split_size], dim=2)

        # Expand query and key if needed (grouped query attention)
        repeat_factor = self.num_value_heads // self.num_key_heads
        if repeat_factor > 1:
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        g, variant_kernel_inputs = self._compute_gates(
            A_log_local_cp, dt_bias_local_cp, batch, seq_len, *gate_feats
        )

        kernel_inputs = {
            "q": query.contiguous(),
            "k": key.contiguous(),
            "v": value.contiguous(),
            "g": g.contiguous(),
            "gate": gate.contiguous(),
            **variant_kernel_inputs,
        }
        return kernel_inputs

    def _compute_gates(
        self,
        A_log_local_cp: torch.Tensor,
        dt_bias_local_cp: torch.Tensor,
        batch: int,
        seq_len: int,
        *gate_feats: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the log-decay ``g`` and the variant-specific kernel inputs.

        Args:
            A_log_local_cp: CP-local slice of ``A_log``.
            dt_bias_local_cp: CP-local slice of ``dt_bias``.
            batch: Batch size.
            seq_len: Sequence length.
            gate_feats: The variant-specific in_proj output sections (everything after
                the qkv and output-gate sections, in ``feat_dim_split`` order).

        Returns:
            (tuple[Tensor, dict[str, Tensor]]): The log-decay ``g`` and a dict of the
            remaining variant-specific kernel inputs keyed by kernel argument name.
        """
        raise NotImplementedError

    def _local_conv1d_params(
        self, cp_group_headwise: torch.distributed.ProcessGroup | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the headwise-CP-local causal-conv1d weight and bias."""
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=cp_group_headwise,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=cp_group_headwise,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        return conv1d_weight, conv1d_bias

    def _apply_causal_conv1d(
        self,
        qkv: torch.Tensor,
        conv1d_weight: torch.Tensor,
        conv1d_bias: torch.Tensor | None,
        cp_size_headwise: int,
        cu_seqlens_q: torch.Tensor | None,
        chunkwise_cp_context,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        """Run the causal convolution on the qkv projection in [b, s, d] layout.

        Handles ``gdn_conv_pad_alignment``, which pads the packed-THD time dim up to a
        multiple of the alignment so the conv kernel sees a whole number of blocks, then
        trims the padding tail back off.
        """
        seq_len = qkv.shape[1]
        if self.config.deterministic_mode:
            qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
            conv_out = F.conv1d(
                input=qkv,  # Torch-native only accept [b, d, s] format input
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // cp_size_headwise,
            )
            qkv = self.act_fn(conv_out[..., :seq_len])
            return qkv.transpose(1, 2)  # b, d, s -> b, s, d

        assert self.activation in ["silu", "swish"]
        pad_n = 0
        conv_input = qkv.contiguous()
        conv_cu_seqlens = cu_seqlens_q
        if self.config.gdn_conv_pad_alignment is not None:
            if packed_seq_params is None or cu_seqlens_q is None:
                raise ValueError(
                    "gdn_conv_pad_alignment is only supported with packed sequence "
                    "parameters in THD format. SBHD inputs do not need causal-conv padding."
                )
            if chunkwise_cp_context is not None:
                raise ValueError(
                    "gdn_conv_pad_alignment is incompatible with GDN chunkwise CP. Padding "
                    "chunk-local causal-conv inputs can change later chunk numerics."
                )
            pad_n = -seq_len % self.config.gdn_conv_pad_alignment
        if pad_n > 0:
            conv_input = torch.nn.functional.pad(conv_input, (0, 0, 0, pad_n))
            # Only the last-segment offset needs to grow to cover the padding tail.
            conv_cu_seqlens = cu_seqlens_q.clone()
            conv_cu_seqlens[-1] += pad_n
        qkv, _ = causal_conv1d(
            x=conv_input,  # FLA conv1d accepts [b, s, d] format input
            weight=conv1d_weight.squeeze(1),  # d, 1, w -> d, w
            bias=conv1d_bias,
            activation=self.activation,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=conv_cu_seqlens,
            cp_context=chunkwise_cp_context,
        )
        if pad_n > 0:
            qkv = qkv[:, :seq_len, :]
        return qkv

    def _reject_unsupported_chunkwise_cp(self, cp_size_chunkwise: int, batch: int) -> None:
        """Reject the chunkwise-CP input shapes the FLA kernels cannot serve."""
        if cp_size_chunkwise <= 1:
            return
        if batch > 1:
            # TODO: If additional gated delta rule backends are added, handle this
            # SBHD + chunkwise CP + batch>1 case per backend instead of
            # unconditionally rejecting it.
            raise ValueError(
                "GDN chunkwise CP with SBHD inputs currently requires micro_batch_size == 1 "
                "because the FLA gated delta rule backend requires a single batch dimension "
                "when cp_context is used. Use packed THD input or micro_batch_size=1."
            )
        if self.config.gdn_conv_pad_alignment is not None:
            raise ValueError(
                "gdn_conv_pad_alignment is incompatible with GDN chunkwise CP. Padding "
                "chunk-local causal-conv inputs can change later chunk numerics."
            )

    def _fused_streamed_pre_gated_delta_rule(
        self,
        qkvzba: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
        cp_group: torch.distributed.ProcessGroup | None = None,
        cp_size_headwise: int = 1,
        cp_group_headwise: torch.distributed.ProcessGroup | None = None,
    ):
        """Call the streamed fused pre-GDR wrapper.

        Fuses the split / conv1d / L2-norm / gate computation that
        :meth:`_apply_causal_conv1d` and :meth:`_prepare_input_for_gated_delta_rule`
        otherwise perform as separate kernels.
        """
        try:
            from megatron.core.fusions.fused_pre_gated_delta_rule import (
                fused_streamed_pre_gated_delta_rule,
            )
        except ImportError as exc:
            raise ImportError(
                "gdn_pre_gated_delta_rule_fusion requires the streamed pre-GDR fusion "
                "dependencies, including causal-conv1d."
            ) from exc

        conv1d_weight, conv1d_bias = self._local_conv1d_params(cp_group_headwise)
        A_log = get_parameter_local_cp(self.A_log, dim=0, cp_group=cp_group_headwise)
        dt_bias = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=cp_group_headwise)
        num_key_heads = self.qk_dim_local_tp // self.key_head_dim // cp_size_headwise
        num_value_heads = self.v_dim_local_tp // self.value_head_dim // cp_size_headwise

        return fused_streamed_pre_gated_delta_rule(
            qkvzba,
            conv1d_weight,
            conv1d_bias,
            A_log,
            dt_bias,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            use_qk_l2norm=self.use_qk_l2norm,
            cu_seqlens=cu_seqlens_q,
            seq_idx=seq_idx,
            cp_group=cp_group,
        )

    def _resolve_cu_seqlens(
        self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size: int = 1
    ) -> torch.Tensor:
        """Resolve cu_seqlens for packed sequence all-to-all, handling alignment padding."""
        if cu_seqlens_padded is not None:
            cu_seqlens = cu_seqlens_padded
        else:
            cu_seqlens = cu_seqlens_actual

        total_cu = cu_seqlens[-1].cpu().item()
        if total_cu != total_seq_len:
            raise ValueError(
                f"GDN: {name}[-1]={total_cu} does not match "
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
            list(self.in_proj_split_sections),
            self.in_proj_split_names,
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


def _build_thd_cp_a2a_perm(
    cu_seqlens: torch.Tensor, cp_size: int, t_global: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cu = cu_seqlens.to(dtype=torch.long)
    t_local = t_global // cp_size

    positions = torch.arange(t_global, device=cu.device)
    seq_idx = torch.bucketize(positions, cu[1:], right=True)
    seq_lens = torch.diff(cu)
    halves = seq_lens // (2 * cp_size)  # per-sequence half-chunk size
    local_starts = cu[:-1] // cp_size
    global_starts = cu[:-1]

    half_i = halves[seq_idx]
    pos_in_seq = positions - global_starts[seq_idx]

    natural_chunk = pos_in_seq // half_i  # in [0, 2*cp)
    offset = pos_in_seq - natural_chunk * half_i

    # Invert the ordering produced by `_undo_attention_load_balancing`:
    #   natural_chunk < cp:   load_balanced = 2 * natural_chunk
    #   natural_chunk >= cp:  load_balanced = 4*cp - 2*natural_chunk - 1
    lb_chunk = torch.where(
        natural_chunk < cp_size, 2 * natural_chunk, 4 * cp_size - 2 * natural_chunk - 1
    )

    # In the per-sequence load-balanced layout each rank owns load-balanced
    # chunks (2r) and (2r+1), in that order, of every sequence.
    rank = lb_chunk // 2
    half_within_rank = lb_chunk - 2 * rank
    k = half_within_rank * half_i + offset

    idx = rank * t_local + local_starts[seq_idx] + k

    inv = torch.empty_like(idx)
    inv[idx] = positions

    return idx, inv


@lru_cache(maxsize=8)
def _build_head_perm_for_split_sections(
    split_sections: tuple[int, ...], cp_size: int, device: torch.device
) -> torch.Tensor:
    assert all(
        s % cp_size == 0 for s in split_sections
    ), f"split_sections {split_sections} must be divisible by cp_size {cp_size} for GDN"
    offset = 0
    parts = []
    for s in split_sections:
        parts.append(
            torch.arange(offset, offset + s, device=device, dtype=torch.long).view(cp_size, -1)
        )
        offset += s

    return torch.cat(parts, dim=-1).view(-1)


####################
# Context parallel utilities
####################
def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[list[int]] = None,
) -> torch.Tensor:
    """Get the local parameter for the current context parallel rank.

    Args:
        param (torch.Tensor): The entire parameter to get the local parameter for.
        dim (int): The dimension to split the parameter along. Usually the dimension of head.
        cp_group (torch.distributed.ProcessGroup): The context parallel group. ``None`` is
            treated as a size-1 group, which is how the inactive linear-attention CP path
            is passed in (see ``_GDNBase._resolve_cp_groups``).
        split_sections (Optional[list[int]]): If not None,
            first split the parameter along the dimension dim into sections,
            then get the local hidden parallel weights separately,
            finally concatenate the local hidden parallel weights along the dimension dim.

    Returns:
        torch.Tensor: The local parameter for the current context parallel rank.
    """

    cp_size = cp_group.size() if cp_group is not None else 1

    # No need to split if CP size is 1.
    if cp_size == 1:
        return param

    cp_rank = cp_group.rank()

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
    split_sections: Optional[list[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    """All-to-all context parallel to hidden parallel.

    This communication primitive is used by GDN headwise CP mode.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group. ``None`` is
            treated as a size-1 group (the inactive linear-attention CP path).
        split_sections (Optional[list[int]]): If not None, split the tensor along the dimension
            head_dim into sections first, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        undo_attention_load_balancing (bool): Whether to undo the attention load balancing of CP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    cp_size = cp_group.size() if cp_group is not None else 1

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
    split_sections: Optional[list[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    """All-to-all hidden parallel to context parallel.

    This communication primitive is used by GDN headwise CP mode.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
        split_sections (Optional[list[int]]): If not None, first split the tensor along the
            dimension head_dim into sections, then do all-to-all for each section separately,
            finally concatenate the separated tensors along the dimension head_dim.
        redo_attention_load_balancing (bool): Whether to redo the attention load balancing of HP.

    Returns:
        torch.Tensor: The all-to-all tensor.
    """

    cp_size = cp_group.size() if cp_group is not None else 1

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


def a2a_cp_to_hp(
    qkvzba: torch.Tensor,
    in_proj_split_sections: tuple[int, ...],
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    cu_seqlens_q: torch.Tensor | None,
    seq_len: int,
    packed_seq_params: PackedSeqParams | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run GDN context-parallel to hidden-parallel A2A and return its inverse context.

    Args:
        qkvzba: in_proj output in sbhd format, sharded along the sequence dim over CP.
        in_proj_split_sections: per-section sizes of the in_proj output, local to this
            TP rank, used to build the pre-a2a head permutation.
        cp_size: context-parallel world size.
        cp_group: context-parallel process group.
        cu_seqlens_q: cumulative sequence lengths, required for the ``thd`` path.
        seq_len: global (unsharded) sequence length.
        packed_seq_params: packed-sequence params; the ``thd`` path is taken when its
            ``qkv_format`` is ``'thd'``.

    Returns:
        The hidden-parallel tensor and the sequence-dim inverse permutation to hand to
        :func:`a2a_hp_to_cp` (``None`` outside the ``thd`` + CP>1 case).
    """
    if cp_size > 1:
        # Pre-permute head dim so a single unsectioned a2a is equivalent to per-section a2a.
        head_perm = _build_head_perm_for_split_sections(
            in_proj_split_sections, cp_size, qkvzba.device
        )
        qkvzba = qkvzba.index_select(-1, head_perm)

    thd_cp_a2a_inv = None
    if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
        qkvzba = tensor_a2a_cp2hp(
            qkvzba, seq_dim=0, head_dim=-1, cp_group=cp_group, undo_attention_load_balancing=False
        )
        if cp_size > 1:
            # Permute at the seq dim so that a single unsectioned a2a
            # is equivalent to per-sequence a2a.
            # This also folds the ``_undo_attention_load_balancing`` step.
            thd_cp_a2a_idx, thd_cp_a2a_inv = _build_thd_cp_a2a_perm(cu_seqlens_q, cp_size, seq_len)
            qkvzba = qkvzba.index_select(0, thd_cp_a2a_idx)
    else:
        qkvzba = tensor_a2a_cp2hp(qkvzba, seq_dim=0, head_dim=-1, cp_group=cp_group)

    return qkvzba, thd_cp_a2a_inv


def a2a_hp_to_cp(
    norm_out: torch.Tensor,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    packed_seq_params: PackedSeqParams | None,
    thd_cp_a2a_inv: torch.Tensor | None,
) -> torch.Tensor:
    """Run GDN hidden-parallel to context-parallel A2A using CP-to-HP context.

    Args:
        norm_out: gated-norm output in sbhd format, sharded along the head dim over CP.
        cp_size: context-parallel world size.
        cp_group: context-parallel process group.
        packed_seq_params: packed-sequence params; the ``thd`` path is taken when its
            ``qkv_format`` is ``'thd'``.
        thd_cp_a2a_inv: sequence-dim inverse permutation returned by
            :func:`a2a_cp_to_hp`, required on the ``thd`` path when ``cp_size > 1``.

    Returns:
        The context-parallel tensor, matching the layout of the GDN module input.
    """
    if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
        if cp_size > 1:
            assert thd_cp_a2a_inv is not None
            norm_out = norm_out.index_select(0, thd_cp_a2a_inv)
        norm_out = tensor_a2a_hp2cp(
            norm_out, seq_dim=0, head_dim=-1, cp_group=cp_group, redo_attention_load_balancing=False
        )
    else:
        norm_out = tensor_a2a_hp2cp(norm_out, seq_dim=0, head_dim=-1, cp_group=cp_group)

    return norm_out
