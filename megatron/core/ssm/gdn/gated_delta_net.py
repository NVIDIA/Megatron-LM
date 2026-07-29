# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.ssm.gdn.gdn_common import (
    _GDNBase,
    chunk_gated_delta_rule,
    torch_chunk_gated_delta_rule,
)


class GatedDeltaNet(_GDNBase):
    """Gated Delta Net (GDN) layer class

    GDN layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def _setup_variant(self):
        """Set the GDN in_proj sizing, split tables, gate parameters, and kernel."""
        # Input projection (hidden_states -> q, k, v, gate (z), beta, alpha)
        # TODO: for now, output gate is forced for GDN.
        # We may remove this restriction in the future.
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads * 2

        # Per-section sizes (and names) of the in_proj output, local to this TP rank.
        # Used for the CP head permutation (pre-a2a), for splitting the projection
        # output (post-a2a), and for the sharded checkpoint split of in_proj.weight.
        self.in_proj_split_names = ["query", "key", "value", "z", "beta", "alpha"]
        self.in_proj_split_sections = (
            self.qk_dim_local_tp,  # q
            self.qk_dim_local_tp,  # k
            self.v_dim_local_tp,  # v
            self.v_dim_local_tp,  # gate (z)
            self.num_value_heads // self.tp_size,  # beta
            self.num_value_heads // self.tp_size,  # alpha
        )
        self.feat_dim_split = (
            (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,  # qkv
            self.v_dim_local_tp // self.cp_size,  # gate (z)
            self.num_value_heads // self.tp_size // self.cp_size,  # beta
            self.num_value_heads // self.tp_size // self.cp_size,  # alpha
        )

        # Time step projection (discretization): per-value-head dt_bias and A_log.
        self._create_gate_params(
            dt_bias_dim=self.num_v_heads_local_tp, a_log_dim=self.num_v_heads_local_tp
        )

        if self.config.deterministic_mode:
            self.gated_delta_rule = torch_chunk_gated_delta_rule
        else:
            self.gated_delta_rule = chunk_gated_delta_rule

    def _reset_dt_bias(self):
        """Initialize dt_bias to ones."""
        torch.ones(
            self.num_v_heads_local_tp,
            out=self.dt_bias.data,
            dtype=self.config.params_dtype,
            device=torch.cuda.current_device(),
        )

    @jit_fuser
    def _compute_gates(self, gate_feats, A_log_local_cp, dt_bias_local_cp, batch, seq_len):
        """Compute the per-head log-decay g and the write strength beta."""
        beta, alpha = gate_feats
        # Per-head decay g and write strength beta
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)  # In fp32
        beta = beta.sigmoid()
        return g, {"beta": beta.contiguous()}
