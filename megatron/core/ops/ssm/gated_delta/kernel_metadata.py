# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dependency and determinism declarations for GDN/GDN2 entry points."""

from megatron.core.ops.kernel_metadata import (
    Dependency,
    Determinism,
    DeterminismResult,
    KernelMetadata,
)

_CONTRACT = "megatron.core.ops.ssm.gated_delta"
_REFERENCE = DeterminismResult(
    Determinism.UNKNOWN,
    "Existing deterministic-mode recurrence, for unpacked inputs. Bit-exact forward/backward "
    "has not been audited across dtypes and devices; optional Q/K normalization uses FLA.",
)
_FLA_TRAINING = DeterminismResult(
    Determinism.NONDETERMINISTIC,
    "The GDN/GDN2 training paths use the Torch reference when deterministic_mode is requested; "
    "the FLA chunk recurrence is not the deterministic-mode implementation.",
)
_FLA_OTHER = DeterminismResult(
    Determinism.UNKNOWN, "FLA auxiliary and recurrent entry points have not been audited."
)
_L2NORM = Dependency("flash-linear-attention", "fla.modules.l2norm", ("l2norm",))

GDN_TORCH = KernelMetadata(
    name="gdn.torch_chunk_gated_delta_rule",
    requires=(
        Dependency(
            "flash-linear-attention", "fla.modules.l2norm", ("l2norm",), feature="qk_l2norm"
        ),
    ),
    determinism=_REFERENCE,
    contract=_CONTRACT,
)
GDN2_TORCH = KernelMetadata(
    name="gdn2.torch_chunk_gdn2",
    requires=GDN_TORCH.requires,
    determinism=_REFERENCE,
    contract=_CONTRACT,
)
GDN_FLA = KernelMetadata(
    name="gdn.fla.chunk_gated_delta_rule",
    requires=(
        Dependency(
            "flash-linear-attention", "fla.ops.gated_delta_rule", ("chunk_gated_delta_rule",)
        ),
    ),
    determinism=_FLA_TRAINING,
    contract=_CONTRACT,
)
GDN2_FLA = KernelMetadata(
    name="gdn2.fla.chunk_gdn2",
    requires=(Dependency("fla-core>=0.5.1", "fla.ops.gdn2.chunk", ("chunk_gdn2",)),),
    determinism=_FLA_TRAINING,
    contract=_CONTRACT,
)
GDN_RECURRENT = KernelMetadata(
    name="gdn.fla.fused_recurrent_gated_delta_rule",
    requires=(
        Dependency(
            "flash-linear-attention",
            "fla.ops.gated_delta_rule",
            ("fused_recurrent_gated_delta_rule",),
        ),
    ),
    determinism=_FLA_OTHER,
    contract=_CONTRACT,
)
FLA_CONV = KernelMetadata(
    name="gdn.fla.causal_conv1d",
    requires=(Dependency("flash-linear-attention", "fla.modules.convolution", ("causal_conv1d",)),),
    determinism=_FLA_OTHER,
    contract=_CONTRACT,
)
FLA_CONV_UPDATE = KernelMetadata(
    name="gdn.fla.causal_conv1d_update",
    requires=(
        Dependency("flash-linear-attention", "fla.modules.convolution", ("causal_conv1d_update",)),
    ),
    determinism=_FLA_OTHER,
    contract=_CONTRACT,
)
FLA_L2NORM = KernelMetadata(
    name="gdn.fla.l2norm", requires=(_L2NORM,), determinism=_FLA_OTHER, contract=_CONTRACT
)

KERNELS = (
    GDN_TORCH,
    GDN2_TORCH,
    GDN_FLA,
    GDN2_FLA,
    GDN_RECURRENT,
    FLA_CONV,
    FLA_CONV_UPDATE,
    FLA_L2NORM,
)
