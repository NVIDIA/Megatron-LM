# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Caller-owned kernel-layout expert weights for the generation megakernel.

By default ``MoEEpMegaLayer`` transforms the expert weights into the kernel's
layout at construction and releases the source, so the kernel serves a snapshot
taken on the first forward. A weight refit between generations would then never
reach the kernel, and the layer would have to be torn down and rebuilt (an EP
collective plus a recompile).

Owning the transformed buffer instead lets a refit rewrite it in place: no
teardown, no EP collective, no recompile. Memory is unchanged, because FlashInfer
holds exactly one transformed copy either way, and the buffer addresses stay
fixed, so CUDA graphs that captured them stay valid. That path covers ``bf16``,
``mxfp8``, and ``bf16_nvfp4`` -- see :class:`MegaKernelWeightBuffer`,
:class:`MegaMxfp8KernelWeightBuffer`, and :class:`MegaBf16Nvfp4KernelWeightBuffer`.
The quantized buffers call FlashInfer's own preprocessing, so the bytes match
what its snapshot path would have produced.
"""

from __future__ import annotations

from typing import Optional

import torch

# Gate/up interleave granularity required by the sm100 BF16 CuTeDSL mega
# kernel. Must match flashinfer.moe_ep's own preprocessing, which interleaves
# gate and up rows in blocks of this size before transposing to K-major.
_GATE_UP_INTERLEAVE = 32


class MegaKernelWeightBuffer:
    """A pair of kernel-layout expert weight buffers, and the repack into them.

    The buffers are in canonical Megatron layout and the kernel consumes
    transposed views of them, so rewriting them in place is visible to a
    FlashInfer layer that was handed those views once.

    Each MoE layer has its own persistent buffer, rewritten only when the weights
    change, i.e. after a refit. The packing has to be byte-identical to
    FlashInfer's own ``preprocess_mega_weights``, which
    ``tests/unit_tests/inference/test_mega_kernel_weights.py`` checks.

    The buffers themselves are BF16: this reproduces only FlashInfer's interleave
    and transpose, not the weight quantization the quantized precisions also do
    during preprocessing. That quantization is a further step on top of these
    views; see :func:`mxfp8_kernel_weights_from_views`.
    """

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._key = (num_local_experts, hidden_size, intermediate_size, dtype, device)
        # Canonical Megatron layout. The kernel views are transposes of these,
        # so writing here in place updates what the kernel sees.
        self._w13 = torch.empty(
            num_local_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device=device
        )
        self._w2 = torch.empty(
            num_local_experts, hidden_size, intermediate_size, dtype=dtype, device=device
        )

    def views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``(fc1, fc2)`` K-major views the kernel consumes.

        Stable across repacks: the transposes alias the buffers, so a layer
        handed these once keeps seeing whatever was last packed into them.
        """
        return self._w13.transpose(1, 2), self._w2.transpose(1, 2)

    def repack(
        self, fc1_weights: list[torch.Tensor], fc2_weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuild the kernel layout from one layer's live expert weights.

        Args:
            fc1_weights: per-expert ``[2 * intermediate, hidden]`` gate-then-up
                weights, in local expert order.
            fc2_weights: per-expert ``[hidden, intermediate]`` weights.

        Returns:
            The ``(fc1, fc2)`` K-major views, as :meth:`views`.
        """
        num_local_experts, hidden_size, intermediate_size, _, _ = self._key
        if len(fc1_weights) != num_local_experts or len(fc2_weights) != num_local_experts:
            raise ValueError(
                f"expected {num_local_experts} expert weights, got "
                f"{len(fc1_weights)} fc1 and {len(fc2_weights)} fc2"
            )

        blocks = intermediate_size // _GATE_UP_INTERLEAVE
        # [E, I/32, 2, 32, H]: index 0 of the size-2 axis holds gate blocks and
        # index 1 holds up blocks, which is the interleave the kernel expects.
        interleaved = self._w13.view(num_local_experts, blocks, 2, _GATE_UP_INTERLEAVE, hidden_size)
        for expert, (fc1, fc2) in enumerate(zip(fc1_weights, fc2_weights)):
            gate, up = fc1[:intermediate_size], fc1[intermediate_size:]
            interleaved[expert, :, 0].copy_(gate.view(blocks, _GATE_UP_INTERLEAVE, hidden_size))
            interleaved[expert, :, 1].copy_(up.view(blocks, _GATE_UP_INTERLEAVE, hidden_size))
            self._w2[expert].copy_(fc2)

        return self.views()


class MegaMxfp8KernelWeightBuffer:
    """Persistent kernel-layout MXFP8 expert weights, and the requantize into them.

    The MXFP8 analogue of :class:`MegaKernelWeightBuffer`, and it exists for the
    same reason: FlashInfer binds the kernel's weights once, so a refit only
    reaches the kernel if the buffers it reads are ours to rewrite. The
    difference is that rewriting them means requantizing rather than copying,
    so this holds the fp8 data and the swizzled scale planes instead of weights
    in the parameters' own dtype.
    """

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._key = (num_local_experts, hidden_size, intermediate_size, dtype, device)
        self._quantized: Optional[tuple[torch.Tensor, ...]] = None

    def views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(fc1, fc2, fc1_scale, fc2_scale)``, stable across repacks."""
        if self._quantized is None:
            raise RuntimeError(
                "mega MXFP8 kernel weights were read before the first repack. The "
                "quantized buffers are allocated by the first repack, so the "
                "parameters have to be packed once before the kernel can be handed them."
            )
        return self._quantized

    def repack(
        self, fc1_weights: list[torch.Tensor], fc2_weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Requantize one layer's live expert weights into the kernel's buffers.

        Args:
            fc1_weights: per-expert ``[2 * intermediate, hidden]`` gate-then-up
                weights, in local expert order.
            fc2_weights: per-expert ``[hidden, intermediate]`` weights.

        Returns:
            The views as :meth:`views`.
        """
        # The bf16 staging buffer is transient. Repacks happen only on a refit, and
        # holding a bf16 copy per layer for a whole run would cost more than the
        # fp8 weights themselves.
        staging = MegaKernelWeightBuffer(*self._key)
        (fc1, fc1_scale), (fc2, fc2_scale) = mxfp8_kernel_weights_from_views(
            *staging.repack(fc1_weights, fc2_weights)
        )
        fresh = (fc1, fc2, fc1_scale, fc2_scale)
        if self._quantized is None:
            # The first repack fixes the addresses; every later one writes into
            # them, so a CUDA graph that captured these pointers stays valid.
            self._quantized = fresh
        else:
            for destination, source in zip(self._quantized, fresh):
                # Through uint8 because copy_ between fp8 tensors is a byte copy
                # anyway and not every fp8 dtype has a copy kernel on every build.
                destination.view(torch.uint8).copy_(source.view(torch.uint8))
        return self._quantized


def mxfp8_kernel_weights_from_views(
    fc1: torch.Tensor, fc2: torch.Tensor
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Quantize bf16 kernel-layout expert weights into kernel-ready MXFP8.

    Composes with :class:`MegaKernelWeightBuffer` rather than repeating it: the
    buffer already produces the gate/up interleave and the K-major orientation,
    which is exactly the input FlashInfer quantizes from, so this takes its
    views and only adds the quantization and the scale swizzle.

    The quantizer and the swizzle are *called*, not reimplemented, so they stay
    bit-identical to FlashInfer's ``preprocess_mega_weights`` across upgrades.
    Both are exported from the ``cutedsl_megamoe`` shim, the same boundary
    FlashInfer's own backends import them through.

    Args:
        fc1: ``[E, hidden, 2 * intermediate]`` gate/up-interleaved K-major view,
            as returned by :meth:`MegaKernelWeightBuffer.views`.
        fc2: ``[E, intermediate, hidden]`` K-major view.

    Returns:
        ``((fc1, fc1_sf), (fc2, fc2_sf))``, the layout ``MoEEpMegaLayer`` accepts
        as ``transformed_weights`` under ``preprocess_weights=False``: fp8 data in
        the shapes of the inputs, and flat swizzled uint8 scales of shape
        ``[E, swizzled_sf_size]``. Byte-identical to what
        ``preprocess_mega_weights`` produces from the same weights, which
        ``tests/unit_tests/inference/test_mega_kernel_weights.py`` pins.
    """
    from megatron.core.inference.moe.mega._deps import require_flashinfer_moe_ep

    require_flashinfer_moe_ep()
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        _stack_byte_reinterpretable_tensors,
        mxfp8_quantize_per_block_32,
        to_blocked,
    )

    quantized = []
    for view in (fc1, fc2):
        num_local_experts, k_size, n_size = view.shape
        # The quantizer wants the reduction dimension trailing, which is the
        # transpose of the kernel view. That is the orientation FlashInfer
        # quantizes in, so the 32-element block boundaries land on the same
        # elements; quantizing the kernel view directly would block along the
        # output dimension instead and silently disagree.
        #
        # Every expert's rows go in one call rather than one call per expert.
        # The quantizer reduces within a single row's 32-element block and is
        # elementwise everywhere else, so rows from different experts cannot
        # influence each other and stacking them first is bit-identical, which
        # the byte-equality test against preprocess_mega_weights checks. It
        # avoids a launch-bound per-expert loop of small eager ops.
        rows = view.transpose(1, 2).reshape(num_local_experts * n_size, k_size).float()
        data, scale = mxfp8_quantize_per_block_32(rows, torch.float8_e4m3fn)
        # Back to the per-expert kernel layout, through a uint8 view because not
        # every build has a copy kernel for every fp8 dtype (FlashInfer stacks
        # its own weights the same way).
        stacked_data = (
            data.view(torch.uint8)
            .reshape(num_local_experts, n_size, k_size)
            .transpose(1, 2)
            .contiguous()
            .view(data.dtype)
        )
        # to_blocked stays per expert. It pads its row count up to a swizzle
        # block and rearranges within that block, so batching the experts would
        # let one expert's rows land in another's block unless the row count
        # happened to be block-aligned. It also reads 1/32 of the bytes the
        # quantizer does, so the loop that matters is the one above.
        scale_parts = [
            to_blocked(expert_scale)
            for expert_scale in scale.reshape(num_local_experts, n_size, -1)
        ]
        stacked_scale = _stack_byte_reinterpretable_tensors(scale_parts, dim=0)
        quantized.append(
            (stacked_data, stacked_scale.view(num_local_experts, scale_parts[0].numel()))
        )
        return quantized[0], quantized[1]


class MegaBf16Nvfp4KernelWeightBuffer:
    """Persistent kernel-layout BF16-activation, NVFP4-weight expert weights.

    The W4A16 analogue of :class:`MegaMxfp8KernelWeightBuffer`. FlashInfer's
    ``bf16_nvfp4`` megakernel (flashinfer-python >= 0.7.1) keeps activations in
    BF16 and stores expert weights as packed NVFP4 with a 16-row gate/up
    interleave and swizzled E4M3 block scales. Those bytes come from
    ``preprocess_mega_weights`` itself; this buffer only pins them at a stable
    address so a refit can rewrite them in place.
    """

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._key = (num_local_experts, hidden_size, intermediate_size, dtype, device)
        # (fc1 base, fc2 base, fc1 scale, fc2 scale). The bases are the
        # contiguous N-major tensors; the kernel is handed their transposes.
        self._storage: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = (
            None
        )

    def views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(fc1, fc2, fc1_scale, fc2_scale)``, stable across repacks.

        ``fc1`` and ``fc2`` are K-major transpose views. Their
        ``transpose(1, 2)`` is the contiguous packed storage, which is the
        layout ``validate_transformed_mega_weights`` requires.
        """
        if self._storage is None:
            raise RuntimeError(
                "mega bf16_nvfp4 kernel weights were read before the first repack. "
                "The packed buffers are allocated by the first repack, so the "
                "parameters have to be packed once before the kernel can be handed them."
            )
        fc1_base, fc2_base, fc1_scale, fc2_scale = self._storage
        return fc1_base.transpose(1, 2), fc2_base.transpose(1, 2), fc1_scale, fc2_scale

    def repack(
        self, fc1_weights: list[torch.Tensor], fc2_weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Requantize one layer's live expert weights into the kernel's buffers.

        Args:
            fc1_weights: per-expert ``[2 * intermediate, hidden]`` gate-then-up
                weights, in local expert order.
            fc2_weights: per-expert ``[hidden, intermediate]`` weights.

        Returns:
            The views as :meth:`views`.
        """
        num_local_experts, hidden_size, intermediate_size, _, _ = self._key
        if len(fc1_weights) != num_local_experts or len(fc2_weights) != num_local_experts:
            raise ValueError(
                f"expected {num_local_experts} expert weights, got "
                f"{len(fc1_weights)} fc1 and {len(fc2_weights)} fc2"
            )
        (fc1, fc1_scale), (fc2, fc2_scale) = _bf16_nvfp4_from_canonical(
            torch.stack(fc1_weights),
            torch.stack(fc2_weights),
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )
        # The kernel view is a transpose. Keep the contiguous base so the view
        # handed out below satisfies the K-major-transpose contract, and so a
        # later refit can copy into that same storage.
        fresh = (
            _owned_contiguous_base(fc1),
            _owned_contiguous_base(fc2),
            fc1_scale.detach().contiguous().clone(),
            fc2_scale.detach().contiguous().clone(),
        )
        if self._storage is None:
            self._storage = fresh
        else:
            for destination, source in zip(self._storage, fresh):
                destination.view(torch.uint8).copy_(source.view(torch.uint8))
        return self.views()


def _owned_contiguous_base(kernel_view: torch.Tensor) -> torch.Tensor:
    """Clone the contiguous tensor behind a K-major transpose view."""
    base = kernel_view.transpose(1, 2)
    if not base.is_contiguous():
        base = base.contiguous()
    return base.detach().clone()


def _bf16_nvfp4_from_canonical(
    w13: torch.Tensor, w2: torch.Tensor, *, intermediate_size: int, hidden_size: int
):
    """Pack canonical BF16 expert weights with FlashInfer's W4A16 preprocessor.

    Returns ``((fc1, fc1_scale), (fc2, fc2_scale))`` exactly as
    ``preprocess_mega_weights`` does. The quantizer, the 16-row gate/up
    interleave, and the scale swizzle are called rather than reimplemented.
    """
    from megatron.core.inference.moe.mega._deps import require_flashinfer_moe_ep

    require_flashinfer_moe_ep()
    try:
        from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_nvfp4_bf16_cutedsl.weights import (
            preprocess_mega_weights,
        )
    except ImportError as exc:
        raise RuntimeError(
            "inference_mega_precision='bf16_nvfp4' requires flashinfer-python>=0.7.1, "
            "which adds the BF16-activation NVFP4-weight megakernel."
        ) from exc
    from flashinfer.moe_ep.weights import MoEWeightPack

    pack = MoEWeightPack(w13=w13.contiguous(), w2=w2.contiguous())
    return preprocess_mega_weights(
        pack, intermediate_size=intermediate_size, hidden_size=hidden_size
    )
