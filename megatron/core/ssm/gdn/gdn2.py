# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.ssm.gdn.gdn_common import _GDNBase

try:
    # The GDN2 kernel is only available in flash-linear-attention >= 0.5.1.
    from fla.ops.gdn2.chunk import chunk_gdn2
except ImportError:
    chunk_gdn2 = None

logger = logging.getLogger(__name__)


class GatedDeltaNet2(_GDNBase):
    """GDN2 (Gated DeltaNet-2) layer class.

    GDN2 replaces GDN's per-head scalar decay and write strength with channel-wise
    gates, decoupling erase and write:

        S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T

    where ``g_t`` is a per-key-channel log-decay, ``b_t`` (in R^{d_k}) is the
    channel-wise erase gate, and ``w_t`` (in R^{d_v}) is the channel-wise write gate.
    Reference: "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
    (https://github.com/NVlabs/GatedDeltaNet-2).

    Note: unlike the GDN2 reference implementation, which uses low-rank decay and
    output-gate projections, all GDN2 projections are fused full-rank into the single
    column-parallel in_proj for TP/CP/SP simplicity.

    The layer takes input with size [s, b, h] and returns output of the same size.
    """

    def _setup_variant(self):
        """Set the GDN2 in_proj sizing, split tables, gate parameters, and kernel."""
        assert (
            chunk_gdn2 is not None
        ), "GDN2 requires flash-linear-attention >= 0.5.1 with the fla.ops.gdn2 kernel."
        assert (
            not self.config.deterministic_mode
        ), "GDN2 has no torch-native implementation for deterministic mode."

        # Input projection (hidden_states -> q, k, v, gate (z), f (decay), b (erase),
        # w (write)).
        # TODO: for now, output gate is forced for GDN2.
        # We may remove this restriction in the future.
        self.in_proj_dim = self.qk_dim * 4 + self.v_dim * 3

        # Per-section sizes (and names) of the in_proj output, local to this TP rank.
        # Used for the CP head permutation (pre-a2a), for splitting the projection
        # output (post-a2a), and for the sharded checkpoint split of in_proj.weight.
        self.in_proj_split_names = ["query", "key", "value", "z", "f", "b", "w"]
        self.in_proj_split_sections = (
            self.qk_dim_local_tp,  # q
            self.qk_dim_local_tp,  # k
            self.v_dim_local_tp,  # v
            self.v_dim_local_tp,  # gate (z)
            self.qk_dim_local_tp,  # f (decay pre-activation)
            self.qk_dim_local_tp,  # b (erase gate)
            self.v_dim_local_tp,  # w (write gate)
        )
        self.feat_dim_split = (
            (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,  # qkv
            self.v_dim_local_tp // self.cp_size,  # gate (z)
            self.qk_dim_local_tp // self.cp_size,  # f
            self.qk_dim_local_tp // self.cp_size,  # b
            self.v_dim_local_tp // self.cp_size,  # w
        )

        # Time step projection (discretization): per-key-channel dt_bias and
        # per-key-head A_log, following the GDN2 reference implementation.
        self._create_gate_params(
            dt_bias_dim=self.qk_dim_local_tp, a_log_dim=self.num_k_heads_local_tp
        )

        self.gated_delta_rule = chunk_gdn2

    def _reset_dt_bias(self):
        """Softplus-inverse init of dt_bias.

        Initializes so the initial per-channel step size lands in [1e-3, 0.1],
        following the GDN2 reference implementation.
        """
        dt = torch.exp(
            torch.rand(
                self.dt_bias.shape[0], dtype=torch.float32, device=torch.cuda.current_device()
            )
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.data.copy_(inv_dt)

    @jit_fuser
    def _compute_gates(self, gate_feats, A_log_local_cp, dt_bias_local_cp, batch, seq_len):
        """Compute the per-channel log-decay g and the erase/write gates b/w."""
        f, b, w = gate_feats
        # Channel-wise log-decay, computed in fp32 for numerical stability. A_log is a
        # per-key-head rate broadcast over the head's key channels; dt_bias is per-channel.
        g = -A_log_local_cp.float().exp().repeat_interleave(self.key_head_dim) * F.softplus(
            f.float() + dt_bias_local_cp
        )
        g = g.reshape(batch, seq_len, -1, self.key_head_dim)

        # Channel-wise erase (key axis) and write (value axis) gates, squashed to [0, 1]
        b = b.sigmoid().reshape(batch, seq_len, -1, self.key_head_dim)
        w = w.sigmoid().reshape(batch, seq_len, -1, self.value_head_dim)

        # Expand key-side gates across value-head groups (grouped value attention)
        repeat_factor = self.num_value_heads // self.num_key_heads
        if repeat_factor > 1:
            g = g.repeat_interleave(repeat_factor, dim=2)
            b = b.repeat_interleave(repeat_factor, dim=2)

        return g, {"b": b.contiguous(), "w": w.contiguous()}
