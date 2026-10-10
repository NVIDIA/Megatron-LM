# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""cuDNN BF16 local experts with discrete parameters and MCore main gradients.

Each outstanding autograd call owns its intermediates. Execution uses one CUDA
stream; returned tensors never alias reusable workspace. Activation storage is
released after forward and recreated for fused recomputation in backward.
The caller provides expert-major rows and nonnegative CUDA expert counts whose
sum equals the input row count. Source SwiGLU arithmetic is FP32 between BF16
GEMM boundaries. This path does not implement quantization or delayed wgrad.
"""

from __future__ import annotations

import math
import weakref
from typing import Any

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _cudnn_offsets(COUNTS, OFFSETS, PADDED, E: tl.constexpr, BE: tl.constexpr):
    expert = tl.arange(0, BE)
    count = tl.load(COUNTS + expert, expert < E, other=0).to(tl.int32)
    tl.store(OFFSETS + expert, tl.cumsum(count), expert < E)
    tl.store(PADDED + expert, tl.cumsum(tl.cdiv(count, 256) * 256), expert < E)


@triton.jit
def _cudnn_maps(OFF, PAD, FORWARD, INVERSE, CAP: tl.constexpr, E: tl.constexpr, BE: tl.constexpr):
    rows = tl.program_id(0) * 128 + tl.arange(0, 128)
    experts = tl.arange(0, BE)
    ends = tl.load(PAD + experts, experts < E, other=2147483647)
    expert = tl.sum((rows[:, None] >= ends[None, :]).to(tl.int32), 1)
    safe = tl.minimum(expert, E - 1)
    pstart = tl.load(PAD + safe - 1, safe > 0, other=0)
    start = tl.load(OFF + safe - 1, safe > 0, other=0)
    end = tl.load(OFF + safe)
    source = start + rows - pstart
    valid = (rows < CAP) & (expert < E) & (source < end)
    tl.store(FORWARD + rows, tl.where(valid, source, -1), rows < CAP)
    tl.store(INVERSE + source, rows, valid)


@triton.jit
def _cudnn_pack_forward(X, P, MAP, PX, PP, CAP: tl.constexpr, H: tl.constexpr):
    index = tl.program_id(0) * 2048 + tl.arange(0, 2048)
    row, col = index // H, index % H
    source = tl.load(MAP + row, row < CAP, other=-1)
    valid = (row < CAP) & (source >= 0)
    tl.store(PX + index, tl.load(X + source * H + col, valid, other=0), row < CAP)
    p = tl.load(P + source, valid & (col == 0), other=0)
    tl.store(PP + row, p, (row < CAP) & (col == 0))


@triton.jit
def _cudnn_pack_backward(DY, MAP, PDY, CAP: tl.constexpr, H: tl.constexpr):
    index = tl.program_id(0) * 2048 + tl.arange(0, 2048)
    row, col = index // H, index % H
    source = tl.load(MAP + row, row < CAP, other=-1)
    valid = (row < CAP) & (source >= 0)
    tl.store(PDY + index, tl.load(DY + source * H + col, valid, other=0), row < CAP)


@triton.jit(do_not_specialize=["R"])
def _cudnn_unpack_forward(PY, MAP, Y, R, H: tl.constexpr):
    index = tl.program_id(0) * 2048 + tl.arange(0, 2048)
    row, col = index // H, index % H
    source = tl.load(MAP + row, row < R, other=0)
    tl.store(Y + index, tl.load(PY + source * H + col, row < R, other=0), row < R)


@triton.jit(do_not_specialize=["R"])
def _cudnn_unpack_backward(PDX, PDP, MAP, DX, DP, R, H: tl.constexpr):
    index = tl.program_id(0) * 2048 + tl.arange(0, 2048)
    row, col = index // H, index % H
    source = tl.load(MAP + row, row < R, other=0)
    tl.store(DX + index, tl.load(PDX + source * H + col, row < R, other=0), row < R)
    dp = tl.load(PDP + source, (row < R) & (col == 0), other=0)
    tl.store(DP + row, dp, (row < R) & (col == 0))


def _weighted_swiglu(
    intermediate: torch.Tensor, probability: torch.Tensor, output: torch.Tensor, clamp: float
) -> None:
    """Match Megatron's FP32 clamped SwiGLU between BF16 GEMM boundaries."""
    rows, width = intermediate.shape
    pair = intermediate.float().view(rows, width // 64, 2, 32)
    gate = pair[:, :, 0].reshape(rows, width // 2).clamp(max=clamp)
    up = pair[:, :, 1].reshape(rows, width // 2).clamp(-clamp, clamp)
    output.copy_((torch.nn.functional.silu(gate) * up) * probability.view(rows, 1))


def _mkl(value: torch.Tensor) -> torch.Tensor:
    rows, width = value.shape
    return value.as_strided((rows, width, 1), (width, 1, rows * width))


def _release_context(plan):
    plan._free("x", "c", "a", "y", "dy", "dx", "dc")
    plan.in_use = False


class _ExpertContext:
    """One reusable invocation's storage and compiled descriptor plans."""

    def __init__(self, owner, capacity, device):
        import cudnn

        self.in_use = False
        self.capacity = capacity
        e, h, i = owner.experts, owner.hidden, owner.intermediate
        self.offsets = torch.zeros(e, device=device, dtype=torch.int32)
        # Compile with a valid positive sample; runtime offsets are written on-device.
        self.padded = torch.arange(1, e + 1, device=device, dtype=torch.int32) * 256
        self.forward_map = torch.empty(capacity, device=device, dtype=torch.int32)
        self.inverse_map = torch.empty_like(self.forward_map)
        self.x = torch.empty(capacity, h, device=device, dtype=torch.bfloat16)
        self.c = torch.empty(capacity, 2 * i, device=device, dtype=torch.bfloat16)
        self.a = torch.empty(capacity, i, device=device, dtype=torch.bfloat16)
        self.activation_bytes = self.a.numel() * self.a.element_size()
        self.y, self.dy, self.dx = (torch.empty_like(self.x) for _ in range(3))
        self.dc = torch.empty_like(self.c)
        self.p = torch.empty(capacity, 1, 1, device=device, dtype=torch.float32)
        self.dp = torch.empty_like(self.p)
        self.alpha = torch.ones(e, device=device, dtype=torch.float32)
        self.unity = torch.ones_like(self.p)
        v = self.views = {
            key: _mkl(getattr(self, key)) for key in ("x", "c", "a", "y", "dy", "dx", "dc")
        }
        if owner.compiled_plans is not None:
            # These plans accept a dynamic, 256-aligned token dimension. Their
            # descriptor scratch is reused in order on the owner's one execution
            # stream; every invocation retains its own activation/gradient data.
            self.fc1, self.fc2, self.dglu, self.dinput, self.wgrad1, self.wgrad2 = (
                owner.compiled_plans
            )
            self._free("x", "c", "a", "y", "dy", "dx", "dc")
            return
        self.fc1 = cudnn.GroupedGemmSm100(
            v["x"],
            v["c"],
            v["c"],
            self.padded,
            self.alpha,
            num_experts=e,
            b_shape=(2 * i, h),
            b_dtype=torch.bfloat16,
            sample_prob=self.unity,
        )
        self.fc2 = cudnn.GroupedGemmSm100(
            v["a"],
            v["y"],
            v["y"],
            self.padded,
            self.alpha,
            num_experts=e,
            b_shape=(h, i),
            b_dtype=torch.bfloat16,
            sample_prob=self.unity,
        )
        self.dglu = cudnn.GroupedGemmDgluSm100(
            v["dy"],
            v["c"],
            v["dc"],
            None,
            None,
            self.padded,
            self.alpha,
            self.alpha,
            self.p,
            self.dp,
            num_experts=e,
            b_shape=(i, h),
            b_dtype=torch.bfloat16,
            b_major="n",
            act_func="dgeglu",
            vector_f32=True,
            geglu_alpha=1.0,
            glu_clamp_min=-owner.clamp,
            glu_clamp_max=owner.clamp,
            linear_offset=0.0,
            round_dgrad_to_input_dtype=True,
            sample_activation=v["a"],
        )
        self.dinput = cudnn.GroupedGemmSm100(
            v["dc"],
            v["dx"],
            v["dx"],
            self.padded,
            self.alpha,
            num_experts=e,
            b_shape=(h, 2 * i),
            b_dtype=torch.bfloat16,
            b_major="n",
            sample_prob=self.unity,
        )
        self.wgrad1 = cudnn.GroupedGemmWgradSm100(
            self.dc.T,
            self.x,
            None,
            None,
            self.padded,
            num_experts=e,
            wgrad_shape=(2 * i, h),
            wgrad_dtype=torch.float32,
            accumulate_on_output=True,
        )
        self.wgrad2 = cudnn.GroupedGemmWgradSm100(
            self.dy.T,
            self.a,
            None,
            None,
            self.padded,
            num_experts=e,
            wgrad_shape=(h, i),
            wgrad_dtype=torch.float32,
            accumulate_on_output=True,
        )
        for plan in (self.fc1, self.fc2, self.dglu, self.dinput, self.wgrad1, self.wgrad2):
            plan.compile()
        owner.workspaces = tuple(
            torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=device)
            for plan in (self.fc1, self.fc2, self.dglu, self.dinput, self.wgrad1, self.wgrad2)
        )
        owner.compiled_plans = (
            self.fc1,
            self.fc2,
            self.dglu,
            self.dinput,
            self.wgrad1,
            self.wgrad2,
        )
        self._free("x", "c", "a", "y", "dy", "dx", "dc")

    def _allocate(self, *names):
        for name in names:
            tensor = getattr(self, name)
            tensor.untyped_storage().resize_(tensor.numel() * tensor.element_size())

    def _free(self, *names):
        # All accesses use the owner's single CUDA stream. The allocator preserves
        # stream ordering; compiled plans receive the current storage pointers on
        # every execution. Tensor shapes/strides and their views remain reusable.
        for name in names:
            getattr(self, name).untyped_storage().resize_(0)

    def forward(self, owner, x, counts, p, bindings):
        """Pack routed rows and execute the two projections with FP32 SwiGLU."""
        e, h, cap, rows = owner.experts, owner.hidden, self.capacity, x.shape[0]
        if rows == 0:
            return torch.empty_like(x)
        self._allocate("x", "c", "a", "y")
        _cudnn_offsets[(1,)](counts, self.offsets, self.padded, e, triton.next_power_of_2(e))
        _cudnn_maps[(triton.cdiv(cap, 128),)](
            self.offsets,
            self.padded,
            self.forward_map,
            self.inverse_map,
            cap,
            e,
            triton.next_power_of_2(e),
        )
        _cudnn_pack_forward[(triton.cdiv(cap * h, 2048),)](
            x, p, self.forward_map, self.x, self.p, cap, h
        )
        v = self.views
        self.fc1.execute(
            v["x"],
            v["c"],
            v["c"],
            self.padded,
            self.alpha,
            b_ptrs=bindings[0],
            prob_tensor=self.unity,
            workspace=owner.workspaces[0],
        )
        owner.activation(self.c, self.p, self.a, owner.clamp)
        self.fc2.execute(
            v["a"],
            v["y"],
            v["y"],
            self.padded,
            self.alpha,
            b_ptrs=bindings[1],
            workspace=owner.workspaces[1],
            prob_tensor=self.unity,
        )
        y = torch.empty_like(x)
        if rows:
            _cudnn_unpack_forward[(triton.cdiv(rows * h, 2048),)](
                self.y, self.inverse_map, y, rows, h
            )
        self._free("a", "y")
        return y

    def backward(self, owner, dy, bindings, p_shape):
        """Recompute activation and accumulate expert main gradients before unpacking."""
        rows, h = dy.shape
        if rows == 0:
            return torch.empty_like(dy), torch.empty(p_shape, device=dy.device, dtype=torch.float32)
        self._allocate("a", "dy", "dx", "dc")
        cap = self.capacity
        _cudnn_pack_backward[(triton.cdiv(cap * h, 2048),)](dy, self.forward_map, self.dy, cap, h)
        self.dp.zero_()
        v = self.views
        self.dglu.execute(
            v["dy"],
            v["c"],
            v["dc"],
            None,
            None,
            self.padded,
            self.alpha,
            self.alpha,
            self.p,
            self.dp,
            b_ptrs=bindings[1],
            activation_tensor=v["a"],
            workspace=owner.workspaces[2],
        )
        self.wgrad2.execute(
            self.dy.T,
            self.a,
            None,
            None,
            self.padded,
            wgrad_ptrs=bindings[3],
            workspace=owner.workspaces[5],
        )
        self.wgrad1.execute(
            self.dc.T,
            self.x,
            None,
            None,
            self.padded,
            wgrad_ptrs=bindings[2],
            workspace=owner.workspaces[4],
        )
        self.dinput.execute(
            v["dc"],
            v["dx"],
            v["dx"],
            self.padded,
            self.alpha,
            b_ptrs=bindings[0],
            workspace=owner.workspaces[3],
            prob_tensor=self.unity,
        )
        dx = torch.empty_like(dy)
        dp = torch.empty(p_shape, device=dy.device, dtype=torch.float32)
        if rows:
            _cudnn_unpack_backward[(triton.cdiv(rows * h, 2048),)](
                self.dx, self.dp, self.inverse_map, dx, dp, rows, h
            )
        return dx, dp


class _CudnnExpertsFunction(torch.autograd.Function):
    """Bind one invocation to its parameter storage and DDP main-gradient buffers."""

    @staticmethod
    def forward(ctx, x, counts, p, owner, *weights):
        """Retain an exclusive invocation context until backward or graph release."""
        plan, bindings, signature = owner.acquire(x, weights, backward=True)
        ctx.owner, ctx.plan, ctx.bindings = owner, plan, bindings
        ctx.release = weakref.finalize(ctx, _release_context, plan)
        ctx.signature, ctx.weights, ctx.used = signature, weights, False
        # Keep the original allocations alive even if a caller replaces parameter storage.
        ctx.bound_storage = tuple(
            t.detach() for weight in weights for t in (weight, weight.main_grad)
        )
        ctx.save_for_backward(x, p, *weights)
        try:
            y = plan.forward(owner, x, counts, p, bindings)
            owner.forward_calls += 1
            return y
        except Exception:
            ctx.release()
            raise

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        """Accumulate weights directly and return TE-compatible DDP hook gradients."""
        from transformer_engine.pytorch.ops._common import get_dummy_wgrads_for_params

        if ctx.used:
            raise RuntimeError(
                "cuDNN experts require one backward per forward; "
                "retained-graph replay is unsupported"
            )
        ctx.used = True
        try:
            # Materializing saved_tensors checks input/parameter version counters before writes.
            x, p, *_ = ctx.saved_tensors
            if ctx.owner.signature(ctx.weights, backward=True) != ctx.signature:
                raise RuntimeError(
                    "cuDNN expert parameter or main_grad storage changed before backward"
                )
            if (
                torch.cuda.current_stream(x.device).cuda_stream
                != ctx.owner.context_streams[id(ctx.plan)]
            ):
                raise RuntimeError("cuDNN expert backward must use its forward CUDA stream")
            dx, dp = ctx.plan.backward(ctx.owner, dy.contiguous(), ctx.bindings, p.shape)
            dummy = get_dummy_wgrads_for_params(list(ctx.weights))
            ctx.owner.backward_calls += 1
            return dx, None, dp, None, *dummy
        finally:
            ctx.release()
            ctx.bound_storage = ()
            ctx.plan = None


class CudnnBf16Experts(torch.nn.Module):
    """cuDNN local expert execution over MCore's existing discrete parameters.

    Args:
        experts: Number of local experts.
        hidden: Input and output width.
        intermediate: Per-expert hidden width before the down projection.
        clamp: Positive gate/up clamp limit.
    """

    def __init__(self, experts: int, hidden: int, intermediate: int, clamp: float) -> None:
        super().__init__()
        if (
            experts <= 0
            or hidden <= 0
            or intermediate <= 0
            or not math.isfinite(clamp)
            or clamp <= 0
        ):
            raise ValueError("cuDNN expert dimensions and clamp must be positive")
        self.experts, self.hidden, self.intermediate, self.clamp = (
            experts,
            hidden,
            intermediate,
            clamp,
        )
        self.pools: dict[tuple[torch.device, int, int], list[_ExpertContext]] = {}
        self.context_streams: dict[int, int] = {}
        self.pointer_cache: tuple[tuple, tuple[torch.Tensor | None, ...]] | None = None
        self.compiled_plans: tuple[Any, ...] | None = None
        self.workspaces: tuple[torch.Tensor, ...] | None = None
        self.activation = torch.compile(
            _weighted_swiglu, fullgraph=True, options={"emulate_precision_casts": True}
        )
        self.execution_stream: tuple[torch.device, int] | None = None
        self.forward_calls = self.backward_calls = self.created_contexts = self.max_inflight = 0

    def signature(self, weights: tuple[torch.Tensor, ...], *, backward: bool) -> tuple:
        """Validate discrete weights and identify their current gradient storage."""
        signature = []
        for index, weight in enumerate(weights):
            shape = (
                (2 * self.intermediate, self.hidden)
                if index < self.experts
                else (self.hidden, self.intermediate)
            )
            if (
                type(weight) is not torch.nn.Parameter
                or weight.dtype != torch.bfloat16
                or tuple(weight.shape) != shape
                or not weight.is_cuda
                or not weight.is_contiguous()
            ):
                raise ValueError("cuDNN requires contiguous discrete BF16 expert weights")
            grad = getattr(weight, "main_grad", None) if backward else None
            if backward and (
                not weight.requires_grad
                or grad is None
                or grad.dtype != torch.float32
                or tuple(grad.shape) != shape
                or not grad.is_contiguous()
                or grad.device != weight.device
                or not hasattr(weight, "grad_added_to_main_grad")
            ):
                raise ValueError(
                    "cuDNN training requires trainable parameters with MCore FP32 main_grad "
                    "and DDP hooks"
                )
            signature.append(
                (weight.data_ptr(), None if grad is None else grad.data_ptr(), weight.device)
            )
        return tuple(signature)

    def acquire(
        self, x: torch.Tensor, weights: tuple[torch.Tensor, ...], *, backward: bool
    ) -> tuple:
        """Bind current pointers and reserve a context for one autograd invocation."""
        signature = self.signature(weights, backward=backward)
        if any(entry[2] != x.device for entry in signature):
            raise ValueError("cuDNN inputs and expert parameters must be on the same CUDA device")
        if self.pointer_cache is None or self.pointer_cache[0] != signature:
            e = self.experts
            ptrs = [
                torch.tensor(
                    [v[0] for v in signature[j : j + e]], device=x.device, dtype=torch.int64
                )
                for j in (0, e)
            ]
            ptrs += [
                (
                    torch.tensor(
                        [v[1] for v in signature[j : j + e]], device=x.device, dtype=torch.int64
                    )
                    if backward
                    else None
                )
                for j in (0, e)
            ]
            self.pointer_cache = (signature, tuple(ptrs))
        # Counts are device metadata; this upper bound needs no host count copy.
        capacity = max(self.experts * 256, triton.cdiv(x.shape[0] + self.experts * 255, 256) * 256)
        # Small routing changes must not repeatedly rebuild six compiled GEMM
        # plans for every in-flight microbatch. Bound extra staging to 4095 rows.
        capacity = triton.cdiv(capacity, 4096) * 4096
        stream = torch.cuda.current_stream(x.device).cuda_stream
        key = (x.device, stream, capacity)
        reusable = [
            plan
            for pool in self.pools.values()
            for plan in pool
            if not plan.in_use and plan.capacity >= capacity
        ]
        plan = min(reusable, key=lambda plan: plan.capacity) if reusable else None
        if plan is None:
            # Dynamic routing changes row counts. Keep live contexts, not an unbounded
            # cache of idle activation buffers for every historical capacity.
            for old_key, old_pool in list(self.pools.items()):
                for old in old_pool:
                    if not old.in_use:
                        self.context_streams.pop(id(old), None)
                live = [old for old in old_pool if old.in_use]
                if live:
                    self.pools[old_key] = live
                else:
                    del self.pools[old_key]
            plan = _ExpertContext(self, capacity, x.device)
            pool = self.pools.setdefault(key, [])
            pool.append(plan)
            self.context_streams[id(plan)] = stream
            self.created_contexts += 1
        plan.in_use = True
        self.max_inflight = max(
            self.max_inflight, sum(p.in_use for pool in self.pools.values() for p in pool)
        )
        return plan, self.pointer_cache[1], signature

    def forward(
        self, x: torch.Tensor, counts: torch.Tensor, p: torch.Tensor, *weights: torch.Tensor
    ) -> torch.Tensor:
        """Run compact expert-major BF16 inputs through the existing expert weights."""
        if (
            not x.is_cuda
            or x.dtype != torch.bfloat16
            or x.ndim != 2
            or x.shape[1] != self.hidden
            or not x.is_contiguous()
        ):
            raise ValueError("cuDNN requires contiguous CUDA BF16 [rows, hidden] inputs")
        if torch.cuda.get_device_capability(x.device) != (10, 0):
            raise ValueError("cuDNN BF16 experts currently require SM100")
        if x.device.index != torch.cuda.current_device():
            raise ValueError("cuDNN inputs must be on the current CUDA device")
        stream = (x.device, torch.cuda.current_stream(x.device).cuda_stream)
        if self.execution_stream is not None and stream != self.execution_stream:
            raise RuntimeError("cuDNN BF16 experts currently require one execution CUDA stream")
        self.execution_stream = stream
        if torch.are_deterministic_algorithms_enabled():
            raise ValueError(
                "cuDNN expert wgrad uses atomic accumulation; " "deterministic mode is unsupported"
            )
        if torch.cuda.is_current_stream_capturing() or torch.is_autocast_enabled():
            raise ValueError(
                "cuDNN BF16 experts currently require eager execution without autocast"
            )
        if (
            counts.device != x.device
            or counts.dtype not in (torch.int32, torch.int64)
            or counts.shape != (self.experts,)
            or not counts.is_contiguous()
        ):
            raise ValueError("cuDNN requires contiguous CUDA integer counts, one per local expert")
        if (
            p.device != x.device
            or p.dtype != torch.float32
            or p.numel() != x.shape[0]
            or not p.is_contiguous()
        ):
            raise ValueError("cuDNN requires one contiguous FP32 probability per input row")
        if len(weights) != 2 * self.experts:
            raise ValueError("cuDNN requires FC1 weights followed by FC2 weights for every expert")
        torch._assert_async((counts >= 0).all(), "cuDNN expert counts must be nonnegative")
        torch._assert_async(
            counts.sum() == x.shape[0], "cuDNN expert counts must sum to the input row count"
        )
        need_backward = torch.is_grad_enabled() and any(t.requires_grad for t in (x, p, *weights))
        if need_backward:
            return _CudnnExpertsFunction.apply(x, counts, p, self, *weights)
        plan, bindings, _ = self.acquire(x, weights, backward=False)
        try:
            y = plan.forward(self, x, counts, p, bindings)
            self.forward_calls += 1
            return y
        finally:
            _release_context(plan)
