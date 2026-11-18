# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Output-gating replay coverage adapted from Layali Rashid's PR #7368.

``fused_gated_norm`` specialises one Triton forward/backward pair on ``GATE_ACTIVATION``:
SiLU behind ``GatedDeltaNet._apply_gated_norm`` and, since PR #7609, sigmoid behind
``KimiDeltaAttention._apply_gated_norm``. Both module dispatchers and both kernel
specialisations are replayed here.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.fusions import fused_gated_norm
from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet
from megatron.core.ssm.gated_delta_net.kda import KimiDeltaAttention
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    seeded,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")

# Module dispatcher and the gate activation its fused call must select.
_VARIANTS = {"gdn": (GatedDeltaNet, "silu"), "kda": (KimiDeltaAttention, "sigmoid")}


@pytest.mark.parametrize("variant", ["gdn", "kda"])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("strided_gate", [False, True])
@pytest.mark.parametrize(
    "batch,length,heads,dim,cp", [(1, 1027, 16, 128, 1), (2, 37, 8, 64, 2), (3, 13, 4, 256, 4)]
)
def test_output_norm_replays(variant, zero_centered, strided_gate, batch, length, heads, dim, cp):
    """Replay each module's fused dispatch with ``gdn_gated_output_norm_fusion`` enabled."""
    te = pytest.importorskip("transformer_engine.pytorch")
    module_cls, gate_activation = _VARIANTS[variant]
    seeded()
    norm = te.RMSNorm(
        dim, eps=1e-6, params_dtype=torch.bfloat16, zero_centered_gamma=zero_centered, device="cuda"
    )

    class GatedNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                deterministic_mode=False, gdn_gated_output_norm_fusion=True
            )
            self.cp_size = cp
            self.activation = "silu"
            self.out_norm = norm
            self.value_head_dim = dim

        def forward(self, x, gate):
            # The module entry checks the supported layout on every replay.
            result = module_cls._apply_gated_norm(self, x, gate)
            # GDN keeps the 4-D view; KDA flattens to [tokens * heads, dim] for out_proj.
            expected_shape = x.shape if variant == "gdn" else (x.numel() // dim, dim)
            assert result.shape == expected_shape
            return result

    x = torch.randn(
        (batch, length, heads, dim), device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    projection = torch.randn((length, batch, heads * dim * 2 + 32), device="cuda", dtype=x.dtype)
    gate = (
        projection[..., 17 : 17 + heads * dim].view(length, batch, heads, dim).permute(1, 0, 2, 3)
    )
    if not strided_gate:
        gate = gate.contiguous()
    gate = gate.detach().requires_grad_()

    with patch.object(
        fused_gated_norm, "fused_gated_norm", wraps=fused_gated_norm.fused_gated_norm
    ) as fused:
        outputs, grads = assert_module_replays_bit_exact(
            GatedNorm(),
            (x, gate),
            replays=3,
            contention=True,
            what=f"{variant.upper()} output RMSNorm/{gate_activation}",
        )
        assert fused.call_count >= 3
        # GDN relies on the SiLU default; KDA must ask for the sigmoid specialisation.
        assert all(
            call.kwargs.get("gate_activation", "silu") == gate_activation
            for call in fused.call_args_list
        )
    assert len(outputs) == 1 and len(grads) == 3


@pytest.mark.parametrize("gate_activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
def test_fused_gated_norm_kernel_replays(gate_activation, gate_dtype):
    """Replay the Triton forward/backward of each ``GATE_ACTIVATION`` specialisation directly.

    ``dx`` and ``dgate`` use unique stores; the RMSNorm weight gradient is reduced over fixed
    row tiles. The sigmoid branch differs from SiLU in the forward product and the gate
    gradient. Rows do not divide the tile size, and the FP32 gate ``validate_gated_norm``
    admits is covered alongside the activation-dtype gate.
    """
    pytest.importorskip("triton")
    seeded()
    batch, length, heads, dim = 2, 1027, 8, 128
    x = torch.randn(
        (batch, length, heads, dim), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    gate = torch.randn(
        (batch, length, heads, dim), device="cuda", dtype=gate_dtype, requires_grad=True
    )
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def run(x, gate, weight):
        return fused_gated_norm.fused_gated_norm(
            x, gate, weight, 1e-6, gate_activation=gate_activation
        )

    outputs, grads = assert_replays_bit_exact(
        run,
        (x, gate, weight),
        replays=3,
        contention=True,
        what=f"fused gated norm[{gate_activation}, {gate_dtype} gate]",
    )
    assert outputs["out"].dtype == torch.bfloat16
    assert len(grads) == 3
