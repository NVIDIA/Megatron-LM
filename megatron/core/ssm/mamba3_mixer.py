# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.
#
# Mamba 3 mixer wrapper for Megatron-LM.
#
# Wraps the official mamba_ssm.modules.Mamba3 (state-spaces/mamba, ICLR 2026
# release: Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu) so it plugs into
# Megatron's MambaLayer + hybrid_stack_spec the same way MambaMixer (Mamba 2)
# does. The upstream class is a plain nn.Module with forward(u) where u is
# [B, S, D]; this wrapper handles the sbhd<->bshd transpose and exposes the
# (mixer_out_with_bias) tuple Megatron's MambaLayer expects.
#
# TP=1 / CP=1 only. The upstream Mamba3 has no TP plumbing and no CP
# all-to-all. Sharded checkpointing is also not wired here. Sufficient for
# the 1-node 350M ablation runs in this repo.

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

try:
    from mamba_ssm.modules.mamba3 import Mamba3 as _UpstreamMamba3
    HAVE_MAMBA3 = True
except ImportError:
    _UpstreamMamba3 = None
    HAVE_MAMBA3 = False


@dataclass
class Mamba3MixerSubmodules:
    """Empty submodule spec — Mamba 3 has its own internal projections.

    Kept for parity with MambaMixerSubmodules so the spec system can swap
    Mamba 3 in for Mamba 2 without other changes.
    """

    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class Mamba3Mixer(nn.Module):
    """Mamba 3 mixer (wraps state-spaces' Mamba3 for Megatron's MambaLayer).

    Constructor signature mirrors MambaMixer so MambaLayer's build_module call
    works unchanged.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[Mamba3MixerSubmodules] = None,
        d_model: Optional[int] = None,
        layer_number: int = 1,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: int = 0,
    ):
        super().__init__()
        if not HAVE_MAMBA3:
            raise ImportError(
                "Mamba 3 requires mamba_ssm>=2.3.1 with the modules.mamba3 module. "
                "Install via _research/launch/install_python_deps.sh (which sets "
                "MAMBA_SKIP_CUDA_BUILD=TRUE so the build does not fail on ARM64)."
            )
        self.config = config
        self.layer_number = layer_number
        self.d_model = d_model if d_model is not None else config.hidden_size

        # TP=1 / CP=1 hard requirement — the upstream Mamba3 has no parallelism.
        if pg_collection is not None:
            tp_size = getattr(pg_collection.tp, "size", lambda: 1)() if pg_collection.tp is not None else 1
            cp_size = getattr(pg_collection.cp, "size", lambda: 1)() if pg_collection.cp is not None else 1
            assert tp_size == 1, f"Mamba3Mixer requires TP=1, got {tp_size}"
            assert cp_size == 1, f"Mamba3Mixer requires CP=1, got {cp_size}"

        # Read Mamba 2 config flags (they double as Mamba 3 base config). Specific
        # Mamba 3 knobs (rope_fraction, MIMO, etc.) use the upstream defaults; expose
        # them via additional config fields if we need to ablate later.
        ngroups = config.mamba_num_groups if config.mamba_num_groups is not None else 1
        self.mamba3 = _UpstreamMamba3(
            d_model=self.d_model,
            d_state=config.mamba_state_dim,
            expand=2,
            headdim=config.mamba_head_dim,
            ngroups=ngroups,
            rope_fraction=0.5,           # Mamba 3 default
            chunk_size=64,
            is_outproj_norm=False,
            is_mimo=False,                # SISO baseline; MIMO needs TileLang kernels
            layer_idx=layer_number,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )

        # GPT-2-style residual rescaling on out_proj — Mamba 3 upstream relies
        # on its MixerModel calling _init_weights to do this; bypassed when we
        # instantiate the layer directly.
        n_layer = getattr(config, "num_layers", 24)
        with torch.no_grad():
            import math
            # n_residuals_per_layer=2 for hybrid blocks (mamba sublayer + MLP sublayer).
            scale = 1.0 / math.sqrt(2 * n_layer)
            self.mamba3.out_proj.weight.mul_(scale)

        # torch.compile disabled by default — interacts poorly with FLA-style
        # autograd Functions and can hide gradient-NaN issues.
        if getattr(config, "mamba3_torch_compile", False):
            self.mamba3 = torch.compile(self.mamba3, dynamic=False, mode="default")

    def forward(self, hidden_states: Tensor, inference_context=None, packed_seq_params=None):
        """sbhd in -> sbhd out, returns (out, bias=None) for Megatron's bda."""
        if inference_context is not None:
            raise NotImplementedError("Mamba3Mixer does not support inference yet.")
        if packed_seq_params is not None:
            raise NotImplementedError("Mamba3Mixer does not support packed sequences yet.")

        # [s, b, d] -> [b, s, d]
        u = hidden_states.transpose(0, 1).contiguous()
        out = self.mamba3(u)
        # [b, s, d] -> [s, b, d]
        out = out.transpose(0, 1).contiguous()
        return out, None

    def mamba_state_shapes_per_request(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Inference-only. Stub returning empty shapes; we don't use it for training."""
        return ((), ())
