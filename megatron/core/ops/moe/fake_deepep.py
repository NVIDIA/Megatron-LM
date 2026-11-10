# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Synthetic balanced DeepEP allocations for single-rank FakePG memory probes.

This independently implemented model uses DeepEP's public size hints. It does
not exchange data or initialize IPC/NVSHMEM. Allocation-size reference:
https://github.com/deepseek-ai/DeepEP/blob/eb9cee7de5a24193bf09500668d3a619d3d3f3fb/csrc/deep_ep.cpp
"""

import logging

import torch

logger = logging.getLogger(__name__)
_fake_buffer = None

FAKE_FLEX_MEMORY_POLICY = "balanced-expanded-assignments-v1"


def fake_flex_memory_model(options):
    """Describe the supported synthetic balanced allocation model."""
    args = {key.replace("_", "-"): value for key, value in options.items()}
    if args.get("moe-token-dispatcher-type") != "flex":
        return None
    unsupported = []
    if args.get("moe-flex-dispatcher-backend", "deepep") != "deepep":
        unsupported.append("HybridEP")
    if not args.get("bf16") or args.get("fp16") or args.get("fp8") or args.get("fp4"):
        unsupported.append("non-BF16 or quantized training")
    if (
        args.get("moe-expert-capacity-factor") is not None
        or args.get("moe-pad-expert-input-to-capacity")
        or args.get("moe-router-padding-for-quantization")
    ):
        unsupported.append("capacity/drop/padding")
    if args.get("cuda-graph-impl", "none") not in (None, "none"):
        unsupported.append("CUDA graphs")
    if unsupported:
        raise ValueError("fake Flex does not support " + ", ".join(unsupported))
    return {
        "policy": FAKE_FLEX_MEMORY_POLICY,
        "routing": "synthetic_balanced",
        "receive_tokens": "local_tokens * effective_topk",
        "receive_token_deduplication": False,
        "effective_topk": "expert_tensor_parallel_size * router_topk",
        "communication_streams": "current_stream_only",
        "communication_metadata": "synthetic_handle",
        "buffer_model_reference": "DeepEP@eb9cee7de5a24193bf09500668d3a619d3d3f3fb",
        "unmodeled_memory": [
            "NVSHMEM runtime overhead",
            "real asynchronous lifetimes",
            "exact RDMA handle layout",
        ],
        "peak_is_upper_bound": False,
    }


class FakeDeepEPEvent:
    """Operations are synchronous on the current stream; no remote completion."""

    def __init__(self, *args, **kwargs):
        pass

    def current_stream_wait(self):
        """Return immediately because fake allocations use the current stream."""
        pass


class FakeDeepEPBuffer:
    """DeepEP API for BF16 memory probes, not a distributed communication emulator.

    Each received row represents exactly one expert assignment. No source-rank
    routing reconstruction or token deduplication is performed. Per-dispatch
    handles are owned by the original Megatron autograd contexts, not this cache.
    """

    def __init__(self, group, hidden_bytes, buffer_type, device=None):
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
        self.torch, self.group, self.device = torch, group, device
        self.group_size = group.size()
        # Config's C++ fields are not exposed by all DeepEP Python bindings.
        # Buffer.set_num_sms() stores this public Python class attribute, which
        # its get_dispatch/combine_config() methods use to build native configs.
        self.num_sms = buffer_type.num_sms
        self.dispatch_config = buffer_type.get_dispatch_config(self.group_size)
        configs = (self.dispatch_config, buffer_type.get_combine_config(self.group_size))
        self.num_nvl_bytes = max(
            c.get_nvl_buffer_size_hint(hidden_bytes, self.group_size) for c in configs
        )
        self.num_rdma_bytes = max(
            c.get_rdma_buffer_size_hint(hidden_bytes, self.group_size) for c in configs
        )
        self.hidden_bytes = hidden_bytes
        # Native DeepEP allocates these outside the torch allocator. Represent
        # them explicitly in the fake snapshot; do not claim allocator parity.
        self.workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.nvl_buffer = torch.empty(
            self.num_nvl_bytes + (160 if self.num_nvl_bytes else 0),
            dtype=torch.uint8,
            device=device,
        )
        self.rdma_buffer = torch.empty(self.num_rdma_bytes, dtype=torch.uint8, device=device)
        self.allocation_summary = {
            "group_size": self.group_size,
            "hidden_bytes": hidden_bytes,
            "nvl_bytes": self.nvl_buffer.numel(),
            "rdma_bytes": self.rdma_buffer.numel(),
            "workspace_bytes": self.workspace.numel(),
            "policy": FAKE_FLEX_MEMORY_POLICY,
            "native_allocator_external_buffers_modeled_in_torch": True,
        }
        logger.info("Fake Flex memory buffer: %s", self.allocation_summary)

    def get_dispatch_layout(self, topk_idx, num_experts, **kwargs):
        """Allocate synthetic routing metadata with the native layout shapes."""
        torch, group_size = self.torch, self.group_size
        tokens, topk = topk_idx.shape
        if num_experts < group_size or num_experts % group_size:
            raise ValueError("fake Flex requires equal local expert counts")
        # These arrays model layout allocation sizes. Their contents do not
        # drive the balanced receive shape, which is computed in dispatch().
        per_rank = torch.zeros(group_size, dtype=torch.int32, device=self.device)
        rdma = (
            torch.zeros(group_size // 8, dtype=torch.int32, device=self.device)
            if group_size > 8
            else None
        )
        per_expert = torch.zeros(num_experts, dtype=torch.int32, device=self.device)
        mask = torch.zeros((tokens, group_size), dtype=torch.bool, device=self.device)
        return per_rank, rdma, per_expert, mask, FakeDeepEPEvent()

    def dispatch(
        self,
        x,
        handle=None,
        topk_idx=None,
        topk_weights=None,
        num_tokens_per_expert=None,
        is_token_in_rank=None,
        **kwargs,
    ):
        """Allocate balanced received assignments or replay a cached handle."""
        torch = self.torch
        if x.dtype != torch.bfloat16 or x.ndim != 2:
            raise ValueError("fake Flex dispatch requires a BF16 matrix")
        if handle is not None:
            # Cached dispatch is FusedCombine.backward. Reuse the forward
            # handle, without allocating another copy of its metadata.
            if x.shape[0] != handle["source_tokens"]:
                raise ValueError("fake Flex cached dispatch source shape mismatch")
            return (
                x.new_zeros((handle["receive_tokens"], x.shape[1])),
                None,
                None,
                None,
                None,
                FakeDeepEPEvent(),
            )
        tokens, topk = topk_idx.shape
        if tokens != x.shape[0] or topk_weights.shape != topk_idx.shape or topk < 1:
            raise ValueError("fake Flex routing metadata shape mismatch")
        local_experts = num_tokens_per_expert.numel() // self.group_size
        receive_tokens = tokens * topk
        quotient, remainder = divmod(receive_tokens, local_experts)
        counts = [quotient + (i < remainder) for i in range(local_experts)]
        received = x.new_zeros((receive_tokens, x.shape[1]))
        indices = torch.full((receive_tokens, topk), -1, dtype=torch.int64, device=x.device)
        probabilities = torch.zeros((receive_tokens, topk), dtype=torch.float32, device=x.device)
        indices[:, 0] = torch.arange(receive_tokens, device=x.device) % local_experts
        probabilities[:, 0] = 1.0 / topk
        channels = self.num_sms // 2
        group_size = self.group_size
        # Retain source masks and prefix/source/head arrays until backward.
        # This is synthetic communication metadata, not an exact RDMA layout.
        shapes = [
            (group_size, group_size),
            (group_size, channels),
            (group_size, channels),
            (receive_tokens,),
            (tokens, group_size),
        ]
        metadata = [torch.empty(shape, dtype=torch.int32, device=x.device) for shape in shapes]
        if group_size > 8:
            rdma = group_size // 8
            metadata.extend(
                torch.empty(shape, dtype=torch.int32, device=x.device)
                for shape in [
                    (rdma, channels),
                    (rdma, channels),
                    (rdma,),
                    (group_size,),
                    (receive_tokens,),
                    (receive_tokens, 8),
                ]
            )
        handle = {
            "source_tokens": tokens,
            "receive_tokens": receive_tokens,
            "topk": topk,
            "metadata": metadata,
            "is_token_in_rank": is_token_in_rank,
        }
        return received, indices, probabilities, counts, handle, FakeDeepEPEvent()

    def combine(self, x, handle, topk_weights=None, **kwargs):
        """Allocate source-shaped outputs and optional routing gradients."""
        if x.shape[0] != handle["receive_tokens"]:
            raise ValueError("fake Flex combine receive shape mismatch")
        output = x.new_zeros((handle["source_tokens"], x.shape[1]))
        probabilities = (
            self.torch.zeros(
                (handle["source_tokens"], handle["topk"]), dtype=self.torch.float32, device=x.device
            )
            if topk_weights is not None
            else None
        )
        return output, probabilities, FakeDeepEPEvent()


def is_fake_process_group(group):
    """Check the explicitly supplied group, not global model-parallel state."""
    return torch.distributed.get_backend(group) == "fake"


def validate_fake_flex_config(config):
    """Reject unsupported allocation paths before constructing a Flex manager."""
    options = config if isinstance(config, dict) else vars(config)
    model = fake_flex_memory_model(options)
    if model is None:
        raise ValueError("fake Flex validation requires the flex dispatcher")
    force = options.get(
        "moe_router_force_load_balancing", options.get("moe-router-force-load-balancing")
    )
    if not force:
        raise ValueError("fake Flex requires moe-router-force-load-balancing")
    return model


def get_fake_buffer(group, hidden_bytes, buffer_type):
    """Reuse one buffer as the native DeepEP integration does."""
    global _fake_buffer
    if not is_fake_process_group(group):
        raise RuntimeError("fake Flex requires a fake process group")
    if buffer_type is None:
        raise RuntimeError("fake Flex requires DeepEP v1 for buffer size hints")
    device = torch.device("cuda", torch.cuda.current_device())
    if (
        _fake_buffer is None
        or _fake_buffer.group is not group
        or _fake_buffer.device != device
        or _fake_buffer.hidden_bytes < hidden_bytes
    ):
        _fake_buffer = FakeDeepEPBuffer(group, hidden_bytes, buffer_type, device)
    return _fake_buffer


def capture_event(group, async_finish, event_type, overlap_type):
    """Real groups retain their stream event; fake allocations use the current stream."""
    if not async_finish or is_fake_process_group(group):
        return None
    return overlap_type(event_type())
