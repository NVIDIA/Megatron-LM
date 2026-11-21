# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Balanced HybridEP tensor allocations for single-rank memory diagnostics.

This is deliberately a partial allocation model, not a collective emulator.
Expert computation and its saved tensors remain real. Transport buffers,
native routing handles and workspace are excluded and declared in the receipt.
No native HybridEP constructor, IPC or network operation is used here.
"""

from dataclasses import dataclass

import torch


def fake_hybridep_memory_model(options):
    """Describe the supported HybridEP allocation model and its excluded memory."""
    args = {key.replace("_", "-"): value for key, value in options.items()}
    unsupported = []
    if (
        args.get("moe-token-dispatcher-type") != "flex"
        or args.get("moe-flex-dispatcher-backend") != "hybridep"
    ):
        raise ValueError("fake HybridEP requires flex/hybridep")
    if not args.get("bf16") or args.get("fp16") or args.get("fp8") or args.get("fp4"):
        unsupported.append("non-BF16 or quantized training")
    etp = args.get("expert-tensor-parallel-size")
    if etp is None:
        etp = args.get("tensor-model-parallel-size", 1)
    if etp != 1:
        unsupported.append("expert tensor parallelism other than ETP1")
    if (
        args.get("moe-expert-capacity-factor") is not None
        or args.get("moe-expert-rank-capacity-factor") is not None
        or args.get("moe-pad-expert-input-to-capacity")
        or args.get("moe-router-padding-for-quantization")
        or args.get("moe-hybridep-pad-uneven-dispatch-inputs")
    ):
        unsupported.append("capacity/drop/padding or uneven input slots")
    if args.get("moe-virtual-expert-load-balance"):
        unsupported.append("virtual expert load balancing")
    if args.get("cuda-graph-impl", "none") not in (None, "none"):
        unsupported.append("CUDA graphs")
    if unsupported:
        raise ValueError("fake Flex HybridEP does not support " + ", ".join(unsupported))
    return {
        "policy": "hybridep-balanced-expert-activations-v1",
        "backend": "hybridep",
        "routing": "synthetic_balanced",
        "receive_tokens": (
            "local_tokens * router_topk plus per-expert alignment (ETP1, equal source sizes)"
        ),
        "receive_layout": "expert_contiguous",
        "expert_padding": "BF16 GroupedTensor/op-fuser per-expert alignment",
        "receive_token_deduplication": False,
        "effective_topk": "router_topk; ETP1 only",
        "communication_streams": "current_stream_only",
        "communication_metadata": "shape-only Python handle; native metadata excluded",
        "measurement_scope": "model_state_and_activations; not total device capacity",
        "unmodeled_memory": [
            "HybridEP persistent transport buffers and scaling-factor storage",
            "HybridEP native routing metadata and workspace",
            "real asynchronous communication lifetimes",
            "natural routing imbalance and remote-rank inputs",
        ],
        "peak_is_upper_bound": False,
        "capacity_guarantee": False,
    }


@dataclass(eq=False)
class FakeHybridEPHandle:
    """Keep shape/count metadata alive for precisely the autograd handle lifetime.

    The handle does NOT retain the source hidden/probs, synthesize native handle
    arrays, or own a reference from the buffer back to itself.
    """

    owner: object
    source_tokens: int
    receive_tokens: int
    hidden_dim: int
    topk: int
    source_probs_shape: tuple
    tokens_per_expert: torch.Tensor
    alignment: int


class FakeHybridEPBuffer:
    """Subset of HybridEPBuffer consumed by Megatron's native autograd wrappers."""

    def __init__(
        self, group, hidden_dim, max_num_of_tokens_per_rank, num_local_experts, *, device=None
    ) -> None:
        if torch.distributed.get_backend(group) != "fake":
            raise RuntimeError("FakeHybridEPBuffer requires an explicitly fake process group")
        if min(group.size(), hidden_dim, max_num_of_tokens_per_rank, num_local_experts) <= 0:
            raise ValueError("fake HybridEP dimensions must be positive")
        self.group = group
        self.hidden_dim = hidden_dim
        self.num_local_experts = num_local_experts
        self.max_num_of_tokens_per_rank = max_num_of_tokens_per_rank
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )

    def _check_hidden(self, hidden):
        if hidden.ndim != 2 or hidden.dtype != torch.bfloat16:
            raise ValueError("fake HybridEP requires a BF16 hidden matrix")
        if hidden.device != self.device or hidden.shape[1] != self.hidden_dim:
            raise ValueError("fake HybridEP hidden device/dimension mismatch")

    def _check_handle(self, handle):
        if not isinstance(handle, FakeHybridEPHandle) or handle.owner is not self:
            raise ValueError("fake HybridEP handle belongs to another buffer")

    @staticmethod
    def _check_scope(pad_multiple, num_permuted_tokens, expected_rows):
        # BF16 GroupedTensor also needs alignment padding. This is distinct
        # from router capacity padding, which can drop or duplicate assignments.
        if pad_multiple is not None and (not isinstance(pad_multiple, int) or pad_multiple < 0):
            raise ValueError("fake HybridEP expert alignment must be a nonnegative integer")
        if num_permuted_tokens is not None and int(num_permuted_tokens) != expected_rows:
            raise ValueError("fake HybridEP static receive size differs from balanced assignments")

    def dispatch_with_permute(
        self,
        *,
        hidden,
        probs=None,
        scaling_factor=None,
        num_of_experts_per_rank=None,
        routing_map=None,
        topk_idx=None,
        num_of_experts=None,
        handle=None,
        pad_multiple=None,
        num_permuted_tokens=None,
        non_blocking=False,
        fuse_permute_dispatch=False,
    ):
        """Allocate balanced expert rows or replay an existing dispatch handle."""
        self._check_hidden(hidden)
        if scaling_factor is not None:
            raise ValueError("fake HybridEP does not model quantization scaling factors")
        if handle is not None:
            self._check_handle(handle)
            if hidden.shape[0] != handle.source_tokens:
                raise ValueError("fake HybridEP cached dispatch source shape mismatch")
            self._check_scope(pad_multiple, num_permuted_tokens, handle.receive_tokens)
            if (pad_multiple or 1) != handle.alignment:
                raise ValueError("fake HybridEP cached dispatch expert alignment changed")
            # HybridEPCombine.backward: replay the saved receive shape without
            # constructing or retaining another copy of routing metadata.
            return (
                hidden.new_zeros((handle.receive_tokens, self.hidden_dim)),
                None,
                None,
                handle.tokens_per_expert,
                handle,
            )

        tokens = hidden.shape[0]
        experts = self.num_local_experts * self.group.size()
        if tokens <= 0 or num_of_experts_per_rank != self.num_local_experts:
            raise ValueError("fake HybridEP requires positive tokens and equal local expert counts")
        if (
            probs is None
            or probs.shape != (tokens, experts)
            or probs.dtype != torch.float32
            or probs.device != hidden.device
        ):
            raise ValueError("fake HybridEP requires full-width FP32 routing probabilities")
        if topk_idx is not None:
            if num_of_experts != experts or topk_idx.ndim != 2 or topk_idx.shape[0] != tokens:
                raise ValueError("fake HybridEP dense routing shape mismatch")
            if (
                topk_idx.dtype not in (torch.int16, torch.int32, torch.int64)
                or topk_idx.device != hidden.device
            ):
                raise ValueError("fake HybridEP dense routing dtype/device mismatch")
            topk = topk_idx.shape[1]
            if torch.any(topk_idx < 0) or torch.any(topk_idx >= experts):
                raise ValueError("fake HybridEP does not model padding/dropped dense routes")
            ordered = topk_idx.sort(dim=-1).values
            if topk > 1 and torch.any(ordered[:, 1:] == ordered[:, :-1]):
                raise ValueError("fake HybridEP requires distinct expert IDs per token")
        else:
            if (
                routing_map is None
                or routing_map.dtype != torch.bool
                or routing_map.shape != probs.shape
                or routing_map.device != hidden.device
            ):
                raise ValueError("fake HybridEP bool routing shape/dtype/device mismatch")
            counts = routing_map.sum(dim=1)
            topk = int(counts[0])
            if not torch.all(counts == topk):
                raise ValueError(
                    "fake HybridEP requires uniform top-k; masked/dropped rows unsupported"
                )
        if not 0 < topk <= experts:
            raise ValueError("fake HybridEP top-k out of range")
        self._check_scope(pad_multiple, None, 0)
        quotient, remainder = divmod(tokens * topk, self.num_local_experts)
        raw_counts = [quotient + (i < remainder) for i in range(self.num_local_experts)]
        alignment = pad_multiple or 1
        padded_counts = [(count + alignment - 1) // alignment * alignment for count in raw_counts]
        received = sum(padded_counts)
        self._check_scope(pad_multiple, num_permuted_tokens, received)
        # Native blocking path exposes host expert counts; Megatron's expert
        # code consumes them. The native GPU handle metadata is explicitly excluded.
        counts = torch.tensor(padded_counts, dtype=torch.int64, device="cpu")
        handle = FakeHybridEPHandle(
            self, tokens, received, self.hidden_dim, topk, tuple(probs.shape), counts, alignment
        )
        dispatched_probs = probs.new_zeros((received,))
        offset = 0
        for raw_count, padded_count in zip(raw_counts, padded_counts):
            dispatched_probs[offset : offset + raw_count] = 1.0 / topk
            offset += padded_count
        return (
            hidden.new_zeros((received, self.hidden_dim)),
            dispatched_probs,
            None,
            counts,
            handle,
        )

    def combine_with_unpermute(
        self, *, hidden, probs=None, handle=None, pad_multiple=None, fuse_unpermute_combine=False
    ):
        """Allocate source-shaped output and optional routing gradients."""
        self._check_hidden(hidden)
        self._check_handle(handle)
        self._check_scope(pad_multiple, None, handle.receive_tokens)
        if (pad_multiple or 1) != handle.alignment:
            raise ValueError("fake HybridEP combine expert alignment changed")
        if hidden.shape[0] != handle.receive_tokens:
            raise ValueError("fake HybridEP combine receive shape mismatch")
        if probs is not None and (
            probs.shape != (handle.receive_tokens,)
            or probs.dtype != torch.float32
            or probs.device != hidden.device
        ):
            raise ValueError("fake HybridEP combine probability shape/dtype/device mismatch")
        return (
            hidden.new_zeros((handle.source_tokens, self.hidden_dim)),
            probs.new_zeros(handle.source_probs_shape) if probs is not None else None,
        )
