"""Liger-Kernel Triton vocab-parallel cross-entropy.

Wraps ``liger_kernel.megatron.LigerMegatronCrossEntropy`` so it can be
dispatched from ``LanguageModule.compute_language_model_loss`` via the
``cross_entropy_fusion_impl='liger'`` config knob. Mirrors the shape of
``fused_cross_entropy.fused_vocab_parallel_cross_entropy``.

Liger-Kernel is an optional runtime dependency; the import is deferred to
call time and raises a clear error if the package is missing.
"""
import torch

try:
    from liger_kernel.megatron import LigerMegatronCrossEntropy

    HAVE_LIGER = True
except ImportError:
    LigerMegatronCrossEntropy = None
    HAVE_LIGER = False


def liger_vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Cross-entropy over vocab-parallel logits using Liger's Triton kernel.

    Args:
        vocab_parallel_logits: Logits split across the tensor-parallel group,
            shape ``[sequence_length, batch_size, vocab_size // tp_size]``.
        target: Global vocabulary indices, shape ``[sequence_length, batch_size]``.
        tp_group: The tensor-parallel process group used for the in-kernel
            all-reduces.

    Returns:
        Per-token loss of shape ``[sequence_length, batch_size]``.
    """
    if not HAVE_LIGER:
        raise ImportError(
            "Liger-Kernel is required for cross_entropy_fusion_impl='liger'. "
            "Install with `pip install liger-kernel`."
        )
    # ``LigerMegatronCrossEntropy`` is a stateless ``nn.Module``; instantiating
    # per-call matches how the TE wrapper is used and avoids a global handle.
    return LigerMegatronCrossEntropy()(vocab_parallel_logits, target, tp_group=tp_group)
