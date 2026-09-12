# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Context parallel support for Gated Delta Product (GDP) with num_householder > 1.

The key difference from GDNContextParallel (which assumes a single copy of V/K/b)
is that GDP has `num_householder` copies of V, K, and b (beta). The in_proj output
layout is:

    [z(d_inner), V(d_inner*M), K(ngroups*d_state*M), Q(ngroups*d_state), b(nheads*M), a(nheads)]

where M = num_householder. Similarly, the conv1d operates on:

    [V(d_inner*M), K(ngroups*d_state*M), Q(ngroups*d_state)]

The all-to-all communication and parameter slicing must account for this.

Strategy for householder-multiplied tensors (V, K, b):
    We fold the M (householder) dimension into the batch dimension before calling
    the standard all-to-all, then unfold afterward. This ensures each householder
    copy is independently partitioned by heads across CP ranks.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams

try:
    from einops import repeat

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

# Re-use the load balancing and all-to-all helpers from the existing module.
# The load-balancing helpers already handle packed (THD) input via their
# ``packed_seq_params`` argument, so GDP just threads it through below.
from megatron.core.ssm.mamba_context_parallel import (
    _all_to_all_cp2hp,
    _all_to_all_hp2cp,
    _redo_attention_load_balancing,
    _undo_attention_load_balancing,
)


class GDPContextParallel:
    """
    Context parallel support for Gated Delta Product (GDP) models with num_householder >= 1.

    Handles the "all-to-all" CP strategy where heads are partitioned across CP ranks
    and each rank processes the full sequence for its head partition. Correctly handles
    the num_householder multiplier on V, K, and beta projections.

    Args:
        cp_group: The process group for context parallel.
        d_inner_local_tp: d_inner on the current TP rank.
        nheads_local_tp: nheads on the current TP rank.
        ngroups_local_tp: ngroups on the current TP rank.
        d_state: SSM state dimension.
        num_householder: Number of householder reflections (M).
        headdim: Dimension per head.
        conv1d_cp1: The conv1d module for cp_size=1.
        dt_bias_cp1: The dt_bias parameter for cp_size=1.
        A_log_cp1: The A_log parameter for cp_size=1.
        D_cp1: The D parameter for cp_size=1 (can be None).
        D_has_hdim: Whether D is sized to the hidden dimension.
        sequence_is_contiguous: Whether CP ranks already hold contiguous causal intervals.
    """

    def __init__(
        self,
        cp_group: torch.distributed.ProcessGroup,
        d_inner_local_tp: int,
        nheads_local_tp: int,
        ngroups_local_tp: int,
        d_state: int,
        num_householder: int,
        headdim: int,
        conv1d_cp1: nn.Conv1d,
        dt_bias_cp1: torch.Tensor,
        A_log_cp1: torch.Tensor,
        D_cp1: torch.Tensor,
        D_has_hdim: bool,
        sequence_is_contiguous: bool = False,
    ) -> None:
        if not HAVE_EINOPS:
            raise ImportError("einops is required but cannot be imported")

        self.cp_group = cp_group
        self.d_inner_local_tp = d_inner_local_tp
        self.nheads_local_tp = nheads_local_tp
        self.ngroups_local_tp = ngroups_local_tp
        self.d_state = d_state
        self.num_householder = num_householder
        self.headdim = headdim
        self.conv1d_cp1 = conv1d_cp1
        self.dt_bias_cp1 = dt_bias_cp1
        self.A_log_cp1 = A_log_cp1
        self.D_cp1 = D_cp1
        self.D_has_hdim = D_has_hdim
        self.sequence_is_contiguous = sequence_is_contiguous

        self.cp_size = self.cp_group.size()

        M = self.num_householder

        if self.cp_size == 1:
            self.d_inner_local_tpcp = self.d_inner_local_tp
            self.nheads_local_tpcp = self.nheads_local_tp
            self.ngroups_local_tpcp = self.ngroups_local_tp
            return

        self.cp_rank = self.cp_group.rank()

        assert (
            self.nheads_local_tp % self.cp_size == 0
        ), "nheads must be evenly divisible by tp_size * cp_size"
        self.nheads_local_tpcp = self.nheads_local_tp // self.cp_size

        self.d_inner_local_tpcp = self.d_inner_local_tp // self.cp_size

        # Group repeat logic (same as GDNContextParallel)
        if self.ngroups_local_tp < self.cp_size:
            assert (
                self.cp_size % self.ngroups_local_tp == 0
            ), "cp_size must be evenly divisible by ngroups/tp_size"
            self.group_repeat_count = self.cp_size // self.ngroups_local_tp
            self.ngroups_local_tpcp = 1
        else:
            assert (
                self.ngroups_local_tp % self.cp_size == 0
            ), "ngroups must be evenly divisible by tp_size * cp_size"
            self.group_repeat_count = 1
            self.ngroups_local_tpcp = self.ngroups_local_tp // self.cp_size

    def pre_conv_ssm(
        self, input_: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> torch.Tensor:
        """
        All-to-all from sequence-partitioned to head-partitioned layout, before conv + SSM.

        Input layout (last dim):
            [z, V, K, Q, b, a] with sizes
            [d_inner, d_inner*M, ngroups*d_state*M, ngroups*d_state, nheads*M, nheads]

        Output layout (last dim, after head partitioning):
            [z, V, K, Q, b, a] with sizes
            [d_inner/cp, d_inner/cp*M, ngroups_cp*d_state*M,
             ngroups_cp*d_state, nheads/cp*M, nheads/cp]

        ``packed_seq_params`` must be passed for THD/SFT input — without it
        the post-all-to-all undo uses the non-packed zigzag pattern, which
        scrambles token order across pack boundaries.
        """
        if self.cp_size == 1:
            return input_

        M = self.num_householder
        l, b, _ = input_.shape

        z, V, K, Q, b_proj, a = torch.split(
            input_,
            [
                self.d_inner_local_tp,  # z
                self.d_inner_local_tp * M,  # V (M copies)
                self.ngroups_local_tp * self.d_state * M,  # K (M copies)
                self.ngroups_local_tp * self.d_state,  # Q (single)
                self.nheads_local_tp * M,  # beta (M copies)
                self.nheads_local_tp,  # a (single)
            ],
            dim=-1,
        )

        # z: [l, b, d_inner] -> [l*cp, b, d_inner/cp]
        z = _all_to_all_cp2hp(z, self.cp_group)

        # V: [l, b, M * d_inner] -> fold M into batch -> all-to-all -> unfold
        # Layout within last dim is (m, h, p). Folding M into batch ensures each
        # householder copy is independently split by heads.
        V = V.view(l, b, M, self.d_inner_local_tp)
        V = V.reshape(l, b * M, self.d_inner_local_tp)
        V = _all_to_all_cp2hp(V, self.cp_group)  # (l*cp, b*M, d_inner/cp)
        V = V.reshape(l * self.cp_size, b, M * self.d_inner_local_tpcp)

        # K: [l, b, M * ngroups * d_state] -> group repeat each copy -> fold M -> all-to-all
        K = K.view(l, b, M, self.ngroups_local_tp * self.d_state)
        K_parts = []
        for i in range(M):
            Ki = K[:, :, i, :]  # (l, b, ngroups_tp * d_state)
            Ki = repeat(
                Ki,
                "l b (g n) -> l b (g r n)",
                g=self.ngroups_local_tp,
                n=self.d_state,
                r=self.group_repeat_count,
            )
            K_parts.append(Ki)
        K = torch.stack(K_parts, dim=2)  # (l, b, M, ngroups_tp * r * d_state)
        K = K.reshape(l, b * M, -1)
        K = _all_to_all_cp2hp(K, self.cp_group)  # (l*cp, b*M, ngroups_tpcp * d_state)
        K = K.reshape(l * self.cp_size, b, M * self.ngroups_local_tpcp * self.d_state)

        # Q: [l, b, ngroups * d_state] -> group repeat -> all-to-all (single copy, no M)
        Q = repeat(
            Q,
            "l b (g n) -> l b (g r n)",
            g=self.ngroups_local_tp,
            n=self.d_state,
            r=self.group_repeat_count,
        )
        Q = _all_to_all_cp2hp(Q, self.cp_group)  # (l*cp, b, ngroups_tpcp * d_state)

        # b_proj (beta): [l, b, M * nheads] -> fold M -> all-to-all -> unfold
        b_proj = b_proj.view(l, b, M, self.nheads_local_tp)
        b_proj = b_proj.reshape(l, b * M, self.nheads_local_tp)
        b_proj = _all_to_all_cp2hp(b_proj, self.cp_group)  # (l*cp, b*M, nheads/cp)
        b_proj = b_proj.reshape(l * self.cp_size, b, M * self.nheads_local_tpcp)

        # a: [l, b, nheads] -> [l*cp, b, nheads/cp]
        a = _all_to_all_cp2hp(a, self.cp_group)

        output = torch.cat([z, V, K, Q, b_proj, a], dim=-1)
        if not self.sequence_is_contiguous:
            output = _undo_attention_load_balancing(output, self.cp_size, packed_seq_params)

        return output

    def post_conv_ssm(
        self, input_: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> torch.Tensor:
        """Method to be applied after the conv + SSM (on y and z, which have no M dim)."""
        if self.cp_size == 1:
            return input_
        if not self.sequence_is_contiguous:
            input_ = _redo_attention_load_balancing(input_, self.cp_size, packed_seq_params)
        return _all_to_all_hp2cp(input_, self.cp_group)

    def conv1d(self, input_: torch.Tensor) -> torch.Tensor:
        """Performs conv1d using sliced weights for the current CP rank."""
        if self.cp_size == 1:
            return self.conv1d_cp1(input_)
        else:
            return F.conv1d(
                input=input_,
                weight=self.get_conv1d_weight(),
                bias=self.get_conv1d_bias(),
                stride=self.conv1d_cp1.stride,
                padding=self.conv1d_cp1.padding,
                dilation=self.conv1d_cp1.dilation,
                groups=self.conv1d_channels(),
            )

    def conv1d_channels(self):
        """Number of conv channels on the current CP rank."""
        M = self.num_householder
        return (
            self.d_inner_local_tpcp * M
            + self.ngroups_local_tpcp * self.d_state * M
            + self.ngroups_local_tpcp * self.d_state
        )

    def get_conv1d_weight(self) -> torch.Tensor:
        """Returns sliced conv1d weight for the current CP rank."""
        return self._slice_conv_param(self.conv1d_cp1.weight)

    def get_conv1d_bias(self) -> torch.Tensor:
        """Returns sliced conv1d bias for the current CP rank."""
        return self._slice_conv_param(self.conv1d_cp1.bias)

    def get_dt_bias(self) -> torch.Tensor:
        """Returns sliced dt_bias for the current CP rank."""
        return self._slice_vector_param(self.dt_bias_cp1)

    def get_A_log(self) -> torch.Tensor:
        """Returns sliced A_log for the current CP rank."""
        return self._slice_vector_param(self.A_log_cp1)

    def get_D(self) -> torch.Tensor:
        """Returns sliced D for the current CP rank."""
        return self._slice_vector_param(self.D_cp1, has_hdim=self.D_has_hdim)

    def _slice_conv_param(self, param: torch.Tensor) -> torch.Tensor:
        """
        Slices a cp_size=1 conv1d parameter along the channel dimension,
        returning the channels needed on the current CP rank.

        Conv param layout (dim 0):
            [V(d_inner * M), K(ngroups * d_state * M), Q(ngroups * d_state)]

        For V and K (which have M copies), we reshape to (M, per_copy_channels, ...),
        slice the per-copy channels for this CP rank, then flatten back.
        """
        if self.cp_size == 1 or param is None:
            return param

        M = self.num_householder
        extra_dims = param.shape[1:]  # (1, d_conv) for weight, () for bias

        V, K, Q = torch.split(
            param,
            [
                self.d_inner_local_tp * M,
                self.ngroups_local_tp * self.d_state * M,
                self.ngroups_local_tp * self.d_state,
            ],
            dim=0,
        )

        # V: (M * d_inner_tp, ...) -> slice heads for this CP rank
        V = V.view(M, self.d_inner_local_tp, *extra_dims)
        v_size = self.d_inner_local_tpcp
        v_start = self.cp_rank * v_size
        V_sliced = V[:, v_start : v_start + v_size, ...].reshape(M * v_size, *extra_dims)

        # K: (M * ngroups_tp * d_state, ...) -> slice groups for this CP rank
        K = K.view(M, self.ngroups_local_tp * self.d_state, *extra_dims)
        k_size = self.ngroups_local_tpcp * self.d_state
        k_start = (self.cp_rank // self.group_repeat_count) * k_size
        K_sliced = K[:, k_start : k_start + k_size, ...].reshape(M * k_size, *extra_dims)

        # Q: (ngroups_tp * d_state, ...) -> slice groups (single copy, no M)
        q_size = self.ngroups_local_tpcp * self.d_state
        q_start = (self.cp_rank // self.group_repeat_count) * q_size
        Q_sliced = Q[q_start : q_start + q_size, ...]

        return torch.cat([V_sliced, K_sliced, Q_sliced], dim=0).contiguous()

    def _slice_vector_param(self, param: torch.Tensor, has_hdim: bool = False) -> torch.Tensor:
        """
        Slices a per-head vector parameter (dt_bias, A_log, D) for the current CP rank.
        These are single-copy (no householder dimension).
        """
        if self.cp_size == 1:
            return param

        size = self.d_inner_local_tpcp if has_hdim else self.nheads_local_tpcp
        start = self.cp_rank * size
        return param[start : start + size]
