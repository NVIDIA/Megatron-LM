# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kernel-layout expert weights for the mega MoE training forward.

The inference path can hand FlashInfer the expert weights once and forget
them: they never change, so ``MoEEpMegaLayer`` transforms them at construction
and releases the source. Training cannot do that. The optimizer rewrites the
expert parameters every step, and the kernel layout is not the parameter
layout, so the kernel's copy has to be rebuilt from the live parameters before
every forward.

Two constraints shape this module.

Megatron's DDP owns parameter storage. It repoints ``param.data`` at views
into its own contiguous buffer and asserts the incoming data is not already a
view, so the inference trick of aliasing parameters onto one big tensor
(``InferenceGroupedMLP._build_concatenated_weights``) cannot be used while DDP
is active. The kernel's weights therefore have to be a genuine second copy,
gathered from the parameters rather than aliased onto them.

A full second copy per layer would roughly double expert-weight memory, which
is the dominant term for an MoE model. Instead one scratch buffer is shared by
every MoE layer and repacked as execution walks them, so the cost is one
layer's worth of expert weights for the whole model. That makes the buffer
single-use: its contents are only valid for the layer that most recently
repacked it, and only until the next repack. The mega kernel reads it
synchronously on the same stream inside that layer's forward, which is what
makes the sharing sound.

Generation reuses the packing for a different reason. A layer that lets
FlashInfer preprocess the weights keeps serving the copy it snapshotted at
construction, so a weight refit between generations would not reach the kernel.
Owning the buffer instead lets a refit rewrite it in place, with no teardown, no
EP collective and no recompile. Memory is unchanged, because FlashInfer holds
exactly one transformed copy either way. That path covers bf16 and mxfp8 -- see
:class:`MegaKernelWeightBuffer` and :class:`MegaMxfp8KernelWeightBuffer` -- and
for mxfp8 the rewrite requantizes with FlashInfer's own quantizer, so the bytes
match what its preprocessing would have produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

# Gate/up interleave granularity required by the sm100 BF16 CuTeDSL mega
# kernel. Must match flashinfer.moe_ep's own preprocessing, which interleaves
# gate and up rows in blocks of this size before transposing to K-major.
_GATE_UP_INTERLEAVE = 32


class MegaKernelWeightBuffer:
    """A pair of kernel-layout expert weight buffers, and the repack into them.

    The buffers are in canonical Megatron layout and the kernel consumes
    transposed views of them, so rewriting them in place is visible to a
    FlashInfer layer that was handed those views once.

    Two callers with opposite lifetimes share this. The training forward wants
    one buffer for the whole model, repacked as execution walks the layers
    (:class:`MegaTrainingWeightScratch`). Generation wants one buffer per layer,
    persistent, rewritten only when the weights change, i.e. after a refit. Both
    need the packing to be byte-identical to FlashInfer's own
    ``preprocess_mega_weights``, which
    ``tests/unit_tests/inference/test_mega_training_weights.py`` checks.

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


class _SharedAcrossLayers:
    """Singleton and ownership bookkeeping for a scratch every MoE layer shares.

    Mixed in ahead of the buffer being shared, so :meth:`repack` records the
    owner and then defers to the buffer's own packing. Each subclass keeps its
    own ``_instance``: assignment through ``cls`` lands on the subclass, so the
    bf16 and MXFP8 scratches cannot be handed each other's buffers.
    """

    _instance: Optional["_SharedAcrossLayers"] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._owner: Optional[int] = None

    @classmethod
    def get(
        cls,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "_SharedAcrossLayers":
        """Return the shared scratch, allocating it on first use.

        Every MoE layer in a model has the same expert geometry, so one buffer
        serves all of them. A second geometry in the same process is rejected
        rather than silently allocating another multi-GiB buffer.
        """
        key = (num_local_experts, hidden_size, intermediate_size, dtype, device)
        if cls._instance is None:
            cls._instance = cls(num_local_experts, hidden_size, intermediate_size, dtype, device)
        elif cls._instance._key != key:
            raise RuntimeError(
                "mega training weight scratch is shared across MoE layers and was "
                f"allocated for {cls._instance._key}, but a layer requested {key}. "
                "Mixed expert geometries in one process are not supported."
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the shared buffers. For teardown between tests."""
        cls._instance = None

    def repack(
        self, fc1_weights: list[torch.Tensor], fc2_weights: list[torch.Tensor], owner: int
    ) -> tuple[torch.Tensor, ...]:
        """Repack for ``owner``, recording it so a stale read can be reported.

        Returns views valid only until the next call, unlike the base class,
        where each layer has its own buffer.
        """
        views = super().repack(fc1_weights, fc2_weights)
        self._owner = owner
        return views

    def assert_owned_by(self, owner: int) -> None:
        """Fail loudly if another layer repacked the scratch since we did.

        Training against a stale buffer looks like a broken learning rate
        rather than an error, so the invariant is checked instead of assumed.
        """
        if self._owner != owner:
            raise RuntimeError(
                "mega training weight scratch was repacked by another MoE layer "
                f"(owner={self._owner}, expected {owner}). The scratch is only "
                "valid for the layer that most recently repacked it; concurrent "
                "or reordered MoE layer execution is not supported."
            )


class MegaMxfp8KernelWeightBuffer:
    """Persistent kernel-layout MXFP8 expert weights, and the requantize into them.

    The MXFP8 analogue of :class:`MegaKernelWeightBuffer`, and it exists for the
    same reason: FlashInfer binds the kernel's weights once, so a refit only
    reaches the kernel if the buffers it reads are ours to rewrite. The
    difference is that rewriting them means requantizing rather than copying,
    so this holds the fp8 data and the swizzled scale planes instead of weights
    in the parameters' own dtype.

    ``cache_staging`` decides whether the bf16 buffer the quantizer reads from is
    kept. Generation leaves it transient: it repacks only on a refit, and holding
    a bf16 copy per layer for a whole run would cost more than the fp8 weights
    themselves. Training keeps it, because it repacks every layer on every step
    and there is only one shared buffer to keep -- the same bf16 buffer the bf16
    scratch holds anyway.
    """

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_staging: bool = False,
    ) -> None:
        self._key = (num_local_experts, hidden_size, intermediate_size, dtype, device)
        self._quantized: Optional[tuple[torch.Tensor, ...]] = None
        self._cache_staging = cache_staging
        self._staging: Optional[MegaKernelWeightBuffer] = None

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
        staging = self._staging or MegaKernelWeightBuffer(*self._key)
        if self._cache_staging:
            self._staging = staging
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


class MegaTrainingWeightScratch(_SharedAcrossLayers, MegaKernelWeightBuffer):
    """One process-wide bf16 buffer pair, shared by every MoE layer in training.

    :meth:`repack` overwrites the buffers from one layer's parameters and returns
    the views the kernel consumes; they stay valid only until the next
    :meth:`repack`, hence the ownership check.
    """

    _instance: Optional["MegaTrainingWeightScratch"] = None


class MegaMxfp8TrainingWeightScratch(_SharedAcrossLayers, MegaMxfp8KernelWeightBuffer):
    """The same, quantized: one shared MXFP8 buffer set for every MoE layer.

    Requantizes from the live parameters on every repack, so unlike generation
    -- where a refit is occasional -- this pays FlashInfer's weight quantizer
    once per MoE layer per step.
    """

    _instance: Optional["MegaMxfp8TrainingWeightScratch"] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, cache_staging=True, **kwargs)


def reset_training_scratches() -> None:
    """Drop every shared training scratch. For teardown between tests.

    Both of them, so a test that ran one precision cannot leave a buffer behind
    for a test running the other -- and so adding a precision does not quietly
    require every fixture in the tree to be found and updated.
    """
    MegaTrainingWeightScratch.reset()
    MegaMxfp8TrainingWeightScratch.reset()


def training_scratch_class(config: "TransformerConfig") -> type:
    """The shared scratch class the mega training forward packs into.

    Kept in one place because two callers have to agree on it: the repack, and
    the ownership check that runs after the kernel has read the result.
    """
    precision = config.inference_mega_precision
    if precision == 'bf16':
        return MegaTrainingWeightScratch
    if precision == 'mxfp8':
        return MegaMxfp8TrainingWeightScratch
    raise NotImplementedError(
        f"moe_inference_training_forward with flashinfer_mega does not support "
        f"inference_mega_precision="
        f"{precision!r}: the training forward packs the kernel's weights itself, "
        "and only bf16 and mxfp8 have a packer. TransformerConfig rejects this "
        "combination, so reaching here means that validation was bypassed."
    )


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
        ``tests/unit_tests/inference/test_mega_training_weights.py`` pins.
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


def kernel_layout_from_parameters(
    config: "TransformerConfig",
    fc1_weights: list[torch.Tensor],
    fc2_weights: list[torch.Tensor],
    owner: int,
) -> tuple[torch.Tensor, ...]:
    """Repack one layer's expert parameters into the shared kernel-layout scratch.

    Returns ``(fc1, fc2)`` for bf16 and ``(fc1, fc2, fc1_scale, fc2_scale)`` for
    mxfp8, which is what the adapter takes.
    """
    num_local_experts = len(fc1_weights)
    scratch = training_scratch_class(config).get(
        num_local_experts=num_local_experts,
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_ffn_hidden_size,
        dtype=fc1_weights[0].dtype,
        device=fc1_weights[0].device,
    )
    return scratch.repack(fc1_weights, fc2_weights, owner=owner)
