"""BackendSpecProvider that uses Liger-Kernel Triton operations.

This is a thin extension over ``LocalSpecProvider`` that swaps Liger
implementations into select slots. Currently overrides only ``layer_norm``
(returns Liger's RMSNorm); other slots inherit the local defaults. The
provider is enabled via ``use_liger=True`` in ``get_gpt_layer_local_submodules``.

Cross-entropy is wired through the existing ``cross_entropy_fusion_impl``
config field (``'native' | 'te' | 'liger'``) and lives outside the spec
system; this provider does not configure it. See
``megatron/core/fusions/liger_cross_entropy.py``.
"""
try:
    from liger_kernel.megatron import LigerMegatronRMSNorm

    HAVE_LIGER = True
except ImportError:
    LigerMegatronRMSNorm = None
    HAVE_LIGER = False

from megatron.core.models.backends import LocalSpecProvider
from megatron.core.transformer.torch_norm import LayerNormBuilder


class LigerSpecProvider(LocalSpecProvider):
    """A spec provider that uses Liger Triton kernels where available.

    Inherits all defaults from ``LocalSpecProvider``; currently overrides only
    ``layer_norm`` to return ``LigerMegatronRMSNorm`` when ``rms_norm=True``.
    Cross-entropy is selected separately via
    ``TransformerConfig.cross_entropy_fusion_impl='liger'`` (with
    ``cross_entropy_loss_fusion=True``). As more Liger kernels are integrated
    (RoPE, SwiGLU MLP, fused linear cross-entropy), this provider will gain
    the corresponding overrides.
    """

    def __init__(self) -> None:
        if not HAVE_LIGER:
            raise ImportError(
                "Liger-Kernel is required for LigerSpecProvider. "
                "Install with `pip install liger-kernel`."
            )

    def layer_norm(
        self,
        rms_norm: bool = False,
        for_qk: bool = False,
        has_residual: bool = False,
    ) -> LayerNormBuilder:
        """Which module to use for layer norm."""
        if rms_norm:
            return LigerMegatronRMSNorm
        return super().layer_norm(
            rms_norm=rms_norm, for_qk=for_qk, has_residual=has_residual
        )
