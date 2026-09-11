# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GDP training and inference declarations; importing them loads no kernels."""

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)

_CONTRACT = "megatron.core.ops.ssm.gdp"
_TRAINING = DeterminismResult(
    Determinism.UNKNOWN,
    "GDP training forward/backward has not been audited for bit-exact repeatability.",
)
_INFERENCE = DeterminismResult(
    Determinism.UNKNOWN,
    "Forward-only dynamic prefill/decode. Caller-owned states and autotuning settings affect "
    "execution; no cross-batch or cross-device repeatability guarantee is declared.",
)
GDP_FLA = KernelMetadata(
    name="gdp.fla.chunk_gated_delta_product",
    requires=(
        Dependency(
            "flash-linear-attention", "fla.ops.gated_delta_product", ("chunk_gated_delta_product",)
        ),
    ),
    determinism=_TRAINING,
    contract=_CONTRACT,
)
GDP_CUTEDSL = KernelMetadata(
    name="gdp.cutedsl.chunk_gated_delta_product",
    requires=(Dependency("gdp-attn", "gdp_attn", ("chunk_gated_delta_product",)),),
    determinism=_TRAINING,
    contract=_CONTRACT,
)
GDP_PREFILL = KernelMetadata(
    name="gdp.triton.chunk_gated_delta_product_varlen",
    requires=(Dependency("triton", "triton"),),
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
GDP_DECODE = KernelMetadata(
    name="gdp.triton.fused_recurrent_gated_delta_rule_update",
    requires=GDP_PREFILL.requires,
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
GDP_PREPARE = KernelMetadata(
    name="gdp.triton.gdp_decode_prepare",
    requires=GDP_PREFILL.requires,
    determinism=_INFERENCE,
    contract=_CONTRACT,
)
GDP_L2NORM = KernelMetadata(
    name="gdp.fla.l2_norm",
    requires=(Dependency("flash-linear-attention", "fla.modules.l2norm", ("l2_norm",)),),
    determinism=_TRAINING,
    contract=_CONTRACT,
)

KERNELS = (GDP_FLA, GDP_CUTEDSL, GDP_PREFILL, GDP_DECODE, GDP_PREPARE, GDP_L2NORM)
