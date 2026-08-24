# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Protocol, Union

import torch
import torch.nn as nn

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
from megatron.core.ssm.utils import _split_tensor_factory
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)

try:
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.cp import build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    build_cp_context = None
    causal_conv1d = None
    l2norm = None
    chunk_gated_delta_rule = None

    HAVE_FLA = False

try:
    from fla.utils import FLA_DISABLE_TENSOR_CACHE
except ImportError:
    # Guarded separately from the block above: an fla build without this symbol is
    # still a usable fla, and folding the import into that try would downgrade it to
    # "FLA is not installed". Assume the cache is disabled so we never block a capture
    # we cannot actually verify - but say so, because that assumption is exactly what
    # _GDNBase._check_fla_tensor_cache_disabled exists to avoid making silently.
    FLA_DISABLE_TENSOR_CACHE = True
    warnings.warn(
        "fla.utils.FLA_DISABLE_TENSOR_CACHE not found; GDN cannot verify that fla's "
        "tensor cache is disabled before capturing THD CUDA graphs. Set "
        "FLA_DISABLE_TENSOR_CACHE=1 manually if you intend to capture.",
        stacklevel=2,
    )

# Chunk size fla's varlen kernels decompose sequences with. Must track both
# ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``'s ``chunk_size`` default and
# ``fla.modules.conv``'s ``BT`` default; the chunk_indices tables we build below are
# consumed by both.
_FLA_CHUNK_SIZE = 64

__all__ = [
    "HAVE_FLA",
    "GatedDeltaNetSubmodules",
    "_GDNBase",
    "_build_head_perm_for_split_sections",
    "_build_thd_cp_a2a_perm",
    "_split_tensor_factory",
    "a2a_cp_to_hp",
    "a2a_hp_to_cp",
    "build_cp_context",
    "causal_conv1d",
    "chunk_gated_delta_rule",
    "get_parameter_local_cp",
    "l2norm",
    "tensor_a2a_cp2hp",
    "tensor_a2a_hp2cp",
]


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
    Unified typing protocol for GDN core computation interfaces.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


class _GDNBase(MegatronModule):
    """Shared implementation for the Gated Delta Net (GDN) layer.

    Hosts the fused input projection, causal convolution on q/k/v, CP all-to-all
    plumbing, kernel-input preparation, gated output norm + projection, and
    sharded checkpointing.
    """

    dt_bias_dim: int
    a_log_dim: int
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
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
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
            name (str | None): Optional module path prefix used for child module names.
            cp_comm_type (Optional[str]): Accepted for TransformerLayer compatibility and
                ignored; GDN implements context parallelism with its own all-to-alls rather
                than the attention CP communication schemes.
            pp_layer_offset (Optional[int]): Pipeline layer offset forwarded by
                TransformerLayer. Stored for MTP/TransformerLayer API compatibility.
            is_mtp_layer (bool): Whether this module is inside an MTP prediction depth.
        """
        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install it with "
                "`pip install flash-linear-attention[cuda]`."
            )

        super().__init__(config)

        self._check_fla_tensor_cache_disabled(config)

        # Single-entry ``(cu_seqlens, cpu_mirror)`` cache for static
        # (CUDA-graph-capturable) cu_seqlens buffers - see _cu_seqlens_cpu_mirror.
        # The source tensor is kept alive by this reference so ``is`` can never
        # alias a freed tensor that happens to be reallocated at the same address.
        self._cu_seqlens_cpu_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        # Config-derived constants for _fixed_shape_chunk_indices, materialized lazily
        # once the device is known: {(nt_max, device): slot_arange}.
        self._chunk_slot_cache: dict[tuple, torch.Tensor] = {}

        # Attributes from arguments
        self.layer_number = layer_number
        self._pp_layer_offset = pp_layer_offset
        self.is_mtp_layer = is_mtp_layer
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet"
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        # Static/max CP size from model construction. Runtime dynamic CP paths must resolve
        # the effective group from packed_seq_params instead of using this value.
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

        # Headwise CP shards heads over the CP group; chunkwise CP keeps heads local.
        if self.config.linear_cp_mode == "headwise":
            num_key_heads_per_tp = self.num_key_heads // self.tp_size
            num_value_heads_per_tp = self.num_value_heads // self.tp_size
            assert num_key_heads_per_tp % self.cp_size == 0, (
                f"GDN head-parallel CP requires the static (max) cp_size ({self.cp_size}) "
                f"to evenly divide num_key_heads per TP rank ({num_key_heads_per_tp}); "
                f"all runtime dynamic cp_size values divide the static one and so will also divide."
            )
            assert num_value_heads_per_tp % self.cp_size == 0, (
                f"GDN head-parallel CP requires the static (max) cp_size ({self.cp_size}) "
                f"to evenly divide num_value_heads per TP rank ({num_value_heads_per_tp}); "
                f"all runtime dynamic cp_size values divide the static one and so will also divide."
            )

        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size

        attrs_to_check = (
            "dt_bias_dim",
            "a_log_dim",
            "in_proj_extra_dim",
            "in_proj_split_names",
            "in_proj_split_sections",
            "gated_delta_rule",
        )
        self._setup_variant_attrs()
        for attr in attrs_to_check:
            assert hasattr(self, attr), f"Attribute {attr} for GDN is not set"
            assert getattr(self, attr) is not None, f"Attribute {attr} for GDN is not set"
        # Full input projection width: q, k, v, output gate, and variant-specific gate features.
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.in_proj_extra_dim

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
        self.recompute_gdn = False
        if self.config.recompute_granularity == "selective" and self.config.recompute_modules:
            self.recompute_norm_out = "gdn_norm_out" in self.config.recompute_modules
            self.recompute_gdn = "gdn" in self.config.recompute_modules

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
        # TODO: Packed sequence cu_seqlens can vary per batch; cache only static SBHD
        # cp_context entries here and revisit routing metadata lifetime in the CP layout refactor.
        self._chunkwise_cp_context_cache: dict[tuple[int, int], tuple[torch.Tensor, object]] = {}

        self.reset_parameters()

    def _setup_variant_attrs(self):
        """Set GDN projection sections, gate parameter sizes, and kernel callable.

        Must set:
        - ``in_proj_extra_dim`` (the in_proj sections beyond q/k/v/z; the base
          class derives ``in_proj_dim`` from it)
        - ``in_proj_split_names``
        - ``in_proj_split_sections``
        - ``dt_bias_dim`` / ``a_log_dim`` (sizes of the gate parameters, which the
          base class creates after the conv1d module to preserve the original
          parameter registration order)
        - ``gated_delta_rule`` (the kernel callable).
        """
        raise NotImplementedError

    def _reset_dt_bias(self):
        """Initialize ``dt_bias``. Called from ``reset_parameters`` under the RNG tracker.

        Defaults to ones; subclasses can override this if their kernel expects a
        different step-size parametrization.
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
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        raise NotImplementedError

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
        cp_size_headwise: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Prepare all gated delta rule kernel inputs.

        Fuses split, reshape, L2 norm, decay/gate activations, repeat_interleave, and
        contiguous operations. ``gate_feats`` holds the in_proj sections after qkv
        and gate, which ``_compute_gates`` turns into the decay and gating tensors.

        Returns:
            (dict[str, Tensor]): Kernel inputs keyed by kernel argument name (``q``,
            ``k``, ``v``, ``g``, and ``beta``), and the output
            gate (z) tensor under the ``gate`` key, which is not a kernel input.
        """
        cp_size = 1 if cp_size_headwise is None else cp_size_headwise

        # Split qkv into query_key and value
        query_key, value = torch.split(
            qkv, [2 * self.qk_dim_local_tp // cp_size, self.v_dim_local_tp // cp_size], dim=-1
        )

        # Reshape query_key and value
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            query_key = l2norm(query_key.contiguous())

        # Split query and key
        split_size = self.qk_dim_local_tp // self.key_head_dim // cp_size
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
        Compute the log-decay ``g`` and remaining kernel inputs.

        Args:
            A_log_local_cp: CP-local slice of ``A_log``.
            dt_bias_local_cp: CP-local slice of ``dt_bias``.
            batch: Batch size.
            seq_len: Sequence length.
            gate_feats: The in_proj output sections after the qkv and output-gate sections.

        Returns:
            (tuple[Tensor, dict[str, Tensor]]): The log-decay ``g`` and a dict of the
            remaining kernel inputs keyed by kernel argument name.
        """
        raise NotImplementedError

    def _resolve_cu_seqlens(
        self,
        cu_seqlens_padded,
        cu_seqlens_actual,
        total_seq_len,
        name,
        cp_size: int = 1,
        validate: bool = True,
    ) -> torch.Tensor:
        """Resolve cu_seqlens for packed sequence all-to-all, handling alignment padding.

        Args:
            validate: Whether to run the consistency checks. They read cu_seqlens
                *values* on the host, so the caller must pass ``False`` while a CUDA
                graph is being captured. Kept as an explicit argument rather than
                probing the stream here so the policy lives with the caller and this
                stays a pure metadata resolver.
        """
        if cu_seqlens_padded is not None:
            cu_seqlens = cu_seqlens_padded
        else:
            cu_seqlens = cu_seqlens_actual

        if validate:
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
                    f"All per-sequence lengths in cu_seqlens must be divisible by "
                    f"cp_size={cp_size}, but got lengths: {seq_lengths.tolist()}"
                )

        return cu_seqlens

    def _fla_varlen_kwargs(
        self,
        cu_seqlens: Optional[torch.Tensor],
        cp_context=None,
        extra_tokens: int = 0,
        cu_seqlens_unpadded: Optional[torch.Tensor] = None,
    ) -> dict:
        """Capture-safe varlen metadata to forward to fla's chunked kernels.

        fla's varlen kernels need a ``[NT, 2]`` ``chunk_indices`` table mapping each
        chunk slot to ``(sequence index, chunk index within that sequence)``. Left to
        itself fla builds that table on the host from the *contents* of ``cu_seqlens``
        (``prepare_chunk_indices``), which CUDA graph capture forbids. This returns
        whichever of fla's two escape hatches applies:

        * ``chunk_indices``: a device-built, fixed-shape table (see
          :meth:`_fixed_shape_chunk_indices`). This is the capture-safe path.
        * ``cu_seqlens_cpu``: a cached CPU mirror, which only removes the D2H half of
          fla's host build. Eager-only fallback, used when the static THD bounds
          needed to fix the table's shape are not configured.

        The two are mutually exclusive by fla's own contract: ``cu_seqlens_cpu`` is
        read only when ``chunk_indices is None`` (``fla/ops/gated_delta_rule/chunk.py``
        and ``fla/modules/conv/triton/ops.py``), so computing the mirror when a static
        table exists would be dead work that can additionally fail capture.

        Args:
            cu_seqlens: The cu_seqlens actually passed to the fla call, i.e. after any
                caller-side padding. ``None`` (SBHD) returns an empty dict.
            cp_context: The ``FLACPContext`` passed to the same fla call, if any.
            extra_tokens: Tokens the caller appended to ``cu_seqlens``'s last sequence
                beyond the configured static buffer size (conv alignment padding).
            cu_seqlens_unpadded: The tensor ``cu_seqlens`` was derived from, before those
                ``extra_tokens`` were added. Required whenever ``extra_tokens`` is
                non-zero. Passing it lets the CPU mirror be taken from the buffer both
                call sites share, so the padded variant is host-side arithmetic on a cache
                hit instead of a second D2H sync that evicts the unpadded entry.
        """
        assert extra_tokens == 0 or cu_seqlens_unpadded is not None, (
            "cu_seqlens_unpadded is required when extra_tokens is non-zero, otherwise the "
            "CPU mirror would double-count the padding."
        )
        if cu_seqlens is None:
            return {}

        # --- CP > 1 is deliberately not handled here. ---------------------------
        # When a cp_context is supplied, fla overrides cu_seqlens with the *rank-local*
        # partition (fla/ops/gated_delta_rule/chunk.py: `cu_seqlens = cp_context.cu_seqlens`;
        # fla/modules/conv/cp/ops.py builds its table from cp_context.cu_seqlens), but it
        # does *not* override a caller-supplied chunk_indices - it is consumed verbatim.
        # Any table we build here is derived from the global cu_seqlens and would
        # therefore describe the wrong tensor. GDN chunkwise CP is out of scope for CUDA
        # graph capture today, so we simply hand the job back to fla, which derives the
        # table from the CP-local cu_seqlens correctly (at the cost of a host round trip
        # that only matters under capture).
        #
        # To lift this later: build the table from ``cp_context.cu_seqlens`` and derive
        # nt_max from the CP-local token count (``max_seqlen_per_dp_cp_rank`` without the
        # ``context_parallel_size`` factor in :meth:`_fixed_shape_chunk_indices`). The
        # ``cp_context`` argument exists so that change is local to this method.
        if cp_context is not None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "GDN: CUDA graph capture is not supported with GDN chunkwise context "
                    "parallelism. fla ignores a caller-supplied chunk_indices override "
                    "for the CP-local cu_seqlens, so no capture-safe table can be built."
                )
            return {}

        chunk_indices = self._fixed_shape_chunk_indices(cu_seqlens, extra_tokens=extra_tokens)
        if chunk_indices is not None:
            return {"chunk_indices": chunk_indices}
        if cu_seqlens_unpadded is None:
            return {"cu_seqlens_cpu": self._cu_seqlens_cpu_mirror(cu_seqlens)}
        return {
            "cu_seqlens_cpu": self._cu_seqlens_cpu_mirror(
                cu_seqlens_unpadded, extra_tokens=extra_tokens
            )
        }

    @staticmethod
    def _check_fla_tensor_cache_disabled(config) -> None:
        """Reject CUDA graph capture of GDN while fla's tensor cache is live.

        ``fla.ops.utils.prepare_chunk_offsets`` is ``@tensor_cache``'d on argument
        identity, and it produces the ``chunk_offsets`` that fla's ``h``/``dh`` writer
        kernels index with. Static THD packing reuses one ``cu_seqlens`` tensor object
        across warmup/capture/replay, which is exactly the cache-hit condition: the hit
        elides those device kernels from the graph and freezes ``chunk_offsets`` at the
        warmup packing, while the ``chunk_indices`` table supplied by
        :meth:`_fla_varlen_kwargs` is recomputed from the real packing on every replay.
        The two then disagree and fla's ``row == chunk_offsets[seq] + intra`` invariant -
        which the reader kernels rely on to index ``h`` - silently breaks.

        This has to be an error rather than a documentation note: the failure mode is a
        slowly diverging loss, with no crash and no warning. It is checked at construction
        rather than at capture time because everything it reads is known before the model
        is built, and because raising inside a live ``cudaStreamBeginCapture`` region
        aborts the capture and surfaces as an unrelated downstream error.

        Both predicates mirror ``TransformerConfig.__post_init__``: the THD one guards the
        sibling ``--thd-max-packed-sequences`` requirement, and the capture one is its
        ``cuda_graph_captures_attention``. Reusing them keeps the two checks from drifting.
        """
        if FLA_DISABLE_TENSOR_CACHE:
            return

        # SBHD is unaffected. fla only calls the @tensor_cache'd prepare_chunk_offsets on
        # the varlen branch (``cu_seqlens is None`` -> ``chunk_offsets = None`` in
        # fla/ops/common/chunk_delta_h.py), and with cu_seqlens None it never builds a
        # chunk_indices table for those offsets to desynchronize from. So there is nothing
        # to freeze, and SBHD capture must keep working without the environment variable.
        is_thd = config.sequence_packing_scheduler is not None or config.dynamic_context_parallel
        if not is_thd:
            return

        captures_attention = config.cuda_graph_impl == "full_iteration" or (
            config.cuda_graph_impl in ("local", "transformer_engine")
            and (not config.cuda_graph_modules or CudaGraphModule.attn in config.cuda_graph_modules)
        )
        if not captures_attention:
            return

        raise RuntimeError(
            "GDN: capturing attention into a THD CUDA graph requires "
            "FLA_DISABLE_TENSOR_CACHE=1. fla's @tensor_cache on prepare_chunk_offsets would "
            "freeze chunk_offsets at the warmup packing and silently desynchronize it from "
            "the chunk_indices recomputed on each replay, diverging the loss with no crash. "
            "Set the environment variable before starting training, or drop attn from "
            "--cuda-graph-modules."
        )

    def _cu_seqlens_cpu_mirror(
        self, cu_seqlens: torch.Tensor, extra_tokens: int = 0
    ) -> torch.Tensor:
        """Return a CPU mirror of ``cu_seqlens``, without syncing during capture.

        Eager-only fallback used by :meth:`_fla_varlen_kwargs` when the static THD
        bounds are not configured. It removes the D2H half of fla's host-side
        ``prepare_chunk_indices`` build, but not the H2D copy of the resulting table, so
        it is *not* sufficient for CUDA graph capture.

        Precondition: never called while capturing. :meth:`_fla_varlen_kwargs` only
        reaches this fallback when :meth:`_fixed_shape_chunk_indices` returned ``None``,
        and that raises under capture rather than returning ``None``. The ``.cpu()``
        below would otherwise be an illegal D2H sync.

        The cache holds a strong reference to the source tensor and compares with ``is``,
        mirroring the contract of fla's own ``@tensor_cache``. Keying on bare ``id()``
        would be unsafe: nothing keeps the source alive, so a freed tensor's address can
        be reused by a new one and return a stale mirror.

        Args:
            cu_seqlens: Tensor to mirror. Callers pass the *unpadded* buffer so repeated
                calls within a forward hit the same cache entry.
            extra_tokens: Tokens the caller appended to the last sequence. Applied to a
                copy of the cached mirror with host-side arithmetic, so a padded variant
                costs no extra D2H sync and does not evict the unpadded entry.
        """
        cached = self._cu_seqlens_cpu_cache
        if cached is not None and cached[0] is cu_seqlens:
            cpu_mirror = cached[1]
        else:
            cpu_mirror = cu_seqlens.cpu()
            self._cu_seqlens_cpu_cache = (cu_seqlens, cpu_mirror)
        if extra_tokens:
            cpu_mirror = cpu_mirror.clone()
            cpu_mirror[-1] += extra_tokens
        return cpu_mirror

    def _fixed_shape_chunk_indices(
        self, cu_seqlens: torch.Tensor, extra_tokens: int = 0, chunk_size: int = _FLA_CHUNK_SIZE
    ) -> Optional[torch.Tensor]:
        """Device-side, fixed-shape ``chunk_indices`` for fla's varlen kernels.

        fla's chunked varlen kernels need a ``[NT, 2]`` table mapping each chunk slot to
        ``(sequence index, chunk index within that sequence)``. fla builds it on the host
        from the *contents* of ``cu_seqlens`` (``prepare_chunk_indices``), which is fatal
        for CUDA graphs twice over:

        1. the build itself is a host<->device round trip that capture forbids
           (supplying ``cu_seqlens_cpu`` only removes the D2H half; the result is still
           H2D-copied back);
        2. even if it were capture-safe, ``NT`` and the table contents would be frozen at
           capture time, while the static ``cu_seqlens`` buffer is refilled with a
           *different* packing on every replay - so the graph would silently compute the
           wrong chunk decomposition.

        This builds the same table entirely with device ops and at a fixed ``NT_max``
        upper bound, so the shape (and therefore the Triton grid) is a capture-time
        constant while the *values* are recomputed by the replayed kernels from whatever
        ``cu_seqlens`` holds. Note the name: only the *shape* is fixed, not the contents.

        Slots past the real chunk count are pointed at sequence 0 with an intra-chunk
        index beyond any possible sequence length, so every masked load/store in fla's
        kernels (all gated on ``o_t < T``) turns them into no-ops. Real rows stay packed
        contiguously from row 0 in sequence order, preserving fla's
        ``row == chunk_offsets[seq] + intra`` invariant and keeping the padding rows
        trailing (which ``chunk_o``'s unmasked ``h``/``dh`` loads rely on to stay in
        bounds).

        This is only needed for THD/varlen. In SBHD ``cu_seqlens`` is ``None``, fla never
        builds the table at all, and ``NT`` is ``cdiv(T, BT)`` - a pure function of the
        static tensor shape - so SBHD capture was always safe. The one exception is SBHD
        with GDN chunkwise CP, which synthesizes a non-``None`` ``cu_seqlens``; that case
        is filtered out by the CP guard in :meth:`_fla_varlen_kwargs` before reaching here.

        Args:
            cu_seqlens: The cu_seqlens the fla call will run on.
            extra_tokens: Tokens the caller appended beyond the configured static buffer
                size (conv alignment padding). Folded into the ``NT_max`` bound, since
                under-sizing it silently drops chunks: ``NT`` is the Triton grid, so
                surplus chunks are simply never launched.
            chunk_size: Must match the chunk size the consuming fla kernel uses.

        Returns:
            The ``[NT_max, 2]`` table, or ``None`` when the static THD bounds are unknown,
            in which case the caller lets fla build the table itself (eager-only path).
        """
        max_num_seqs = getattr(self.config, 'thd_max_packed_sequences', None)
        max_seqlen = getattr(self.config, 'max_seqlen_per_dp_cp_rank', None)
        if max_num_seqs is None or max_seqlen is None:
            if torch.cuda.is_current_stream_capturing():
                # TransformerConfig.__post_init__ already refuses this at startup for the
                # supported launch paths ("THD CUDA Graph requires
                # --thd-max-packed-sequences to be set"). This backstop only fires for
                # callers that capture with cuda_graph_impl="none", e.g. an external or
                # hand-rolled graph around the module.
                raise RuntimeError(
                    "GDN: cannot build a capture-safe chunk_indices table without "
                    "thd_max_packed_sequences and max_seqlen_per_dp_cp_rank; see the "
                    "THD CUDA Graph requirements validated in "
                    "TransformerConfig.__post_init__."
                )
            return None

        # Chunkwise CP never reaches here (see the CP guard in _fla_varlen_kwargs), so
        # cu_seqlens still spans the full global sequence. Under dynamic CP the runtime
        # group can be smaller than context_parallel_size, which makes this an
        # overestimate: NT_max is only an upper bound, so the table stays correct and
        # merely carries unused trailing slots. Threading the runtime cp size down from
        # forward() would tighten it; dynamic CP cannot be captured anyway, so this only
        # costs eager runs. Once chunkwise CP is supported, this becomes the CP-local
        # token count.
        total_t = int(max_seqlen) * int(self.config.context_parallel_size) + int(extra_tokens)
        # Each sequence wastes at most one partial chunk, so the total chunk
        # count never exceeds ceil(total_t / chunk_size) + num_sequences.
        nt_max = (total_t + chunk_size - 1) // chunk_size + int(max_num_seqs)

        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        n_seq = lens.shape[0]
        n_chunks = (lens + (chunk_size - 1)).div(chunk_size, rounding_mode='floor').to(torch.int64)
        # offsets[i] = first chunk slot owned by sequence i; offsets[-1] = total.
        # Bit-identical to fla's prepare_chunk_offsets, which the h/dh writer kernels
        # index with - this is what keeps `row == chunk_offsets[seq] + intra` true.
        offsets = torch.nn.functional.pad(n_chunks.cumsum(0), (1, 0))
        slot = self._chunk_slot_arange(nt_max, cu_seqlens.device)
        # searchsorted over offsets[1:] maps a slot to its owning sequence; slots
        # past the last boundary come back as n_seq, which flags them unused.
        seg_id = torch.searchsorted(offsets[1:].contiguous(), slot, right=True)
        valid = seg_id < n_seq
        seg_id = seg_id.clamp(max=n_seq - 1)
        intra = slot - offsets[seg_id]
        zero = torch.zeros((), device=cu_seqlens.device, dtype=torch.int64)
        beyond = torch.full((), nt_max, device=cu_seqlens.device, dtype=torch.int64)
        seg_id = torch.where(valid, seg_id, zero)
        intra = torch.where(valid, intra, beyond)
        return torch.stack([seg_id, intra], 1).to(cu_seqlens)

    def _chunk_slot_arange(self, nt_max: int, device: torch.device) -> torch.Tensor:
        """Cached ``arange(nt_max)`` used as the chunk-slot axis.

        ``nt_max`` is a config-derived constant, so this tensor is identical on every
        forward of every microbatch. Only *shape*-invariant state is cached here - the
        table itself is deliberately never cached, because its values depend on the
        contents of a ``cu_seqlens`` buffer that is refilled every microbatch.
        """
        key = (nt_max, device)
        slot = self._chunk_slot_cache.get(key)
        if slot is not None:
            return slot
        slot = torch.arange(nt_max, device=device, dtype=torch.int64)
        if not torch.cuda.is_current_stream_capturing():
            # Do not cache an allocation made inside a graph's private memory pool:
            # that storage is not valid for use outside the graph. A cold cache during
            # capture just means one extra (capturable) arange per replay.
            self._chunk_slot_cache[key] = slot
        return slot

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
    cp_group: torch.distributed.ProcessGroup | None,
    split_sections: Optional[list[int]] = None,
) -> torch.Tensor:
    """Get the local parameter for the current context parallel rank.

    Args:
        param (torch.Tensor): The entire parameter to get the local parameter for.
        dim (int): The dimension to split the parameter along. Usually the dimension of head.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
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

    assert cp_group is not None
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
    cp_group: torch.distributed.ProcessGroup | None,
    split_sections: Optional[list[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    """All-to-all context parallel to hidden parallel.

    Args:
        tensor (torch.Tensor): The tensor to all-to-all.
            Currently only support (seq_len, batch, head_dim) shaped tensor.
        seq_dim (int): The dimension of sequence length. Currently only supports seq_dim == 0.
        head_dim (int): The dimension of head. Currently only supports head_dim == -1 or 2.
        cp_group (torch.distributed.ProcessGroup): The context parallel group.
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

    assert cp_group is not None

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
    cp_group: torch.distributed.ProcessGroup | None,
    split_sections: Optional[list[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    """All-to-all hidden parallel to context parallel.

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

    assert cp_group is not None

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
