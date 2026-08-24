# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

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

    uses_attention_mask: bool = False
    """GDN occupies the ``self_attention`` slot but is a linear-attention variant: it
    accepts ``attention_mask`` only for signature compatibility with
    ``TransformerLayer`` and never reads it, and it has no ``attn_mask_type``. See
    ``Attention.uses_attention_mask`` for the contract."""

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
