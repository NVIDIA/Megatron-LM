"""TP-parallel linear layers and vocab-parallel embedding/output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
import transformer_engine.pytorch as te  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.utils import ensure_divisible

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel.state import ParallelState


# ---------------------------------------------------------------------------
# Vanilla column-parallel linear (torch.matmul kernel, NOT TE).
#
# Matches Megatron-Core `tensor_parallel.ColumnParallelLinear` bit-for-bit
# in bf16: same `torch.matmul(input, weight.t())` forward, same
# all-reduce-on-backward-grad_input pattern. Use this for heads like the
# vocab LM projection where MC's GPT model uses vanilla torch matmul
# (hardcoded in `LinearCrossEntropyModule(tensor_parallel.ColumnParallelLinear)`)
# — TE's `te.Linear` uses a different cuBLAS algo selection that introduces
# ~3e-4 loss-level drift under bf16 vs torch.matmul.
#
# For QKV / MoE experts we still prefer the TE path (fused LN+linear, FP8
# readiness) — this vanilla path is a drop-in substitute only when kernel
# parity with bridge is required.
# ---------------------------------------------------------------------------


class _VanillaColParallelMatmul(torch.autograd.Function):
    """forward: output = matmul(input, weight.t()) — replicated input, sharded output.
    backward: grad_input = grad_output @ weight (+ all-reduce across TP);
              grad_weight = grad_output.T @ input.
    Mirrors MC `LinearWithGradAccumulationAndAsyncCommunication`.
    """

    @staticmethod
    def forward(ctx, input_, weight, tp_group):
        ctx.save_for_backward(input_, weight)
        ctx.tp_group = tp_group
        return torch.matmul(input_, weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)
        if ctx.tp_group is not None and dist.get_world_size(ctx.tp_group) > 1:
            dist.all_reduce(grad_input, group=ctx.tp_group)
        # grad_weight = grad_output^T @ input (sum over leading dims).
        gi = grad_output.reshape(-1, grad_output.shape[-1])
        xi = input_.reshape(-1, input_.shape[-1])
        grad_weight = gi.t().matmul(xi)
        return grad_input, grad_weight, None


class _VanillaColParallelMatmulSP(torch.autograd.Function):
    """SP-aware column-parallel matmul matching MC's
    `ColumnParallelLinear(sequence_parallel=True)` kernel bit-for-bit.

    forward:
      - all-gather input along dim-0 from [S/tp, B, H] → [S, B, H]
      - matmul(gathered, weight.t()) → [S, B, V/tp]
    backward:
      - grad_input_full = grad_output @ weight  → [S, B, H]
      - reduce-scatter dim-0 → [S/tp, B, H]
      - grad_weight = grad_output^T @ gathered_input
    """

    @staticmethod
    def forward(ctx, input_, weight, tp_group):
        ws = dist.get_world_size(tp_group) if tp_group is not None else 1
        if ws > 1:
            s_local = input_.shape[0]
            total_shape = (s_local * ws, *input_.shape[1:])
            total_input = torch.empty(
                total_shape, dtype=input_.dtype, device=input_.device,
            )
            dist.all_gather_into_tensor(total_input, input_.contiguous(), group=tp_group)
        else:
            total_input = input_
        ctx.save_for_backward(total_input, weight)
        ctx.tp_group = tp_group
        ctx.tp_size = ws
        ctx.local_s = input_.shape[0]
        return torch.matmul(total_input, weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        total_input, weight = ctx.saved_tensors
        grad_input_full = grad_output.matmul(weight)
        if ctx.tp_size > 1:
            out_shape = (ctx.local_s, *grad_input_full.shape[1:])
            grad_input = torch.empty(
                out_shape,
                dtype=grad_input_full.dtype,
                device=grad_input_full.device,
            )
            dist.reduce_scatter_tensor(
                grad_input, grad_input_full.contiguous(), group=ctx.tp_group,
            )
        else:
            grad_input = grad_input_full
        gi = grad_output.reshape(-1, grad_output.shape[-1])
        xi = total_input.reshape(-1, total_input.shape[-1])
        grad_weight = gi.t().matmul(xi)
        return grad_input, grad_weight, None


class _VanillaColLinear(nn.Module):
    """Drop-in for `te.Linear(parallel_mode='column')` using torch.matmul.

    Shape: `self.weight` is (out_features_per_tp, in_features). Exposes
    `.weight` at the same attribute path as TE Linear for checkpoint-loader
    compatibility.

    When `sp=True`, input is assumed SP-sharded on dim-0; forward gathers
    before matmul and backward reduce-scatters grad_input — matching MC's
    `ColumnParallelLinear(sequence_parallel=True)` bit-for-bit. When
    `sp=False`, input is assumed replicated and grad_input is all-reduced.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ps: ParallelState,
        *,
        sp: bool = False,
    ):
        super().__init__()
        self.tp_group = ps.tp_group
        self.tp_size = ps.tp_size
        self.sp = sp
        local_out = ensure_divisible(out_features, ps.tp_size)
        self.weight = nn.Parameter(
            torch.empty(local_out, in_features, dtype=torch.bfloat16)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sp:
            return _VanillaColParallelMatmulSP.apply(x, self.weight, self.tp_group)
        return _VanillaColParallelMatmul.apply(x, self.weight, self.tp_group)


class VanillaColumnParallelLinear(nn.Module):
    """Public vanilla column-parallel linear matching MCore torch matmul."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ps: ParallelState,
        *,
        sp: bool = False,
        gather_output: bool = False,
    ):
        super().__init__()
        self.linear = _VanillaColLinear(in_features, out_features, ps, sp=sp)
        self.tp_size = ps.tp_size
        self.tp_group = ps.tp_group
        self.gather_output = gather_output

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.gather_output and self.tp_size > 1:
            out = _AllGatherLastDim.apply(out, self.tp_size, self.tp_group)
        return out


class ColumnParallelLinear(nn.Module):
    """TE-based column-parallel linear. Splits output dim across TP."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ps: ParallelState,
        bias: bool = False,
        gather_output: bool = False,
        normalization: str | None = None,
        eps: float = 1e-6,
        zero_centered_gamma: bool = False,
        sequence_parallel: bool | None = None,
    ):
        super().__init__()
        self.tp_size = ps.tp_size
        self.tp_rank = ps.tp_rank
        self.tp_group = ps.tp_group
        self.local_out = ensure_divisible(out_features, ps.tp_size)
        self.use_sp = ps.tp_size > 1 and not gather_output if sequence_parallel is None else sequence_parallel
        if normalization is not None:
            self.linear = te.LayerNormLinear(
                in_features,
                out_features,
                bias=bias,
                normalization=normalization,
                eps=eps,
                zero_centered_gamma=zero_centered_gamma,
                params_dtype=torch.bfloat16,
                parallel_mode="column",
                sequence_parallel=self.use_sp,
                tp_group=ps.tp_group,
                tp_size=ps.tp_size,
            )
        else:
            self.linear = te.Linear(
                in_features,
                out_features,
                bias=bias,
                params_dtype=torch.bfloat16,
                parallel_mode="column",
                sequence_parallel=self.use_sp,
                tp_group=ps.tp_group,
                tp_size=ps.tp_size,
            )
        self.gather_output = gather_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.gather_output and self.tp_size > 1:
            out = _AllGatherLastDim.apply(out, self.tp_size, self.tp_group)
        return out


class RowParallelLinear(nn.Module):
    """TE-based row-parallel linear. Splits input dim across TP."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ps: ParallelState,
        bias: bool = False,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        self.tp_size = ps.tp_size
        self.tp_rank = ps.tp_rank
        self.tp_group = ps.tp_group
        self.use_sp = ps.tp_size > 1
        self.local_in = ensure_divisible(in_features, ps.tp_size)
        self.linear = te.Linear(
            in_features,
            out_features,
            bias=bias,
            params_dtype=torch.bfloat16,
            parallel_mode="row",
            sequence_parallel=self.use_sp,
            tp_group=ps.tp_group,
            tp_size=ps.tp_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def pad_vocab_for_tp(vocab_size: int, tp_size: int) -> int:
    """Round vocab up to be divisible by `lcm(128, tp_size)`.

    Matches MC's `_vocab_size_with_padding(..., make_vocab_size_divisible_by=128)`:
    pad to 128-multiple for GEMM alignment, and also require tp-divisibility.
    For typical `tp_size ∈ {1,2,4,...,128}`, `lcm = 128` so a vocab already
    divisible by 128 (e.g. Qwen3-MoE's 151936) stays unchanged — which is
    what MC's `output_layer` sees. Using `128 * tp_size` instead would
    over-pad (e.g. 151936 -> 152064
    at tp=2), introducing 128 extra logits into the vocab-parallel cross-
    entropy log-sum-exp and driving a ~3e-4 loss drift.
    """
    import math

    divisor = math.lcm(128, tp_size)
    return ((vocab_size + divisor - 1) // divisor) * divisor


class VocabParallelEmbedding(nn.Module):
    """Embedding table split across TP on the vocab dimension."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        ps: ParallelState,
        *,
        deterministic: bool = False,
    ):
        super().__init__()
        self.tp_size = ps.tp_size
        self.tp_rank = ps.tp_rank
        self.deterministic = bool(deterministic)
        padded_vocab = pad_vocab_for_tp(vocab_size, ps.tp_size)
        self.local_vocab = ensure_divisible(padded_vocab, ps.tp_size)
        self.vocab_start = self.tp_rank * self.local_vocab
        self.vocab_end = self.vocab_start + self.local_vocab
        self.embedding = nn.Embedding(self.local_vocab, hidden_size)
        self.tp_group = ps.tp_group

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, S] → out: [S, B, H]
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_start).clamp(min=0, max=self.local_vocab - 1)
        if self.deterministic:
            out = self.embedding.weight[local_ids]
        else:
            out = self.embedding(local_ids)
        out = out * mask.unsqueeze(-1)
        if self.tp_size > 1:
            out = _ReduceFromTP.apply(out, self.tp_group)
        return out.transpose(0, 1).contiguous()


class _ColForLMHead(nn.Module):
    """Thin wrapper exposing `.linear` (with `.weight`) for checkpoint-loader
    compat, switchable between TE and vanilla torch.matmul kernel.

    `sp=True` threads to the underlying linear: vanilla uses the SP-aware
    matmul (gather-in / reduce-scatter-on-bwd); TE uses `sequence_parallel=True`
    on `te.Linear`. Both match MC's `output_layer(sequence_parallel=True)`
    semantics so the upstream final_layernorm can run on SP-sharded input.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ps: ParallelState,
        *,
        backend: str = "vanilla",
        sp: bool = False,
    ):
        super().__init__()
        if backend == "vanilla":
            # Matches MC's LinearCrossEntropyModule → tensor_parallel.ColumnParallelLinear
            # kernel bit-for-bit in bf16. Preferred for LM head parity with bridge.
            self.linear = _VanillaColLinear(in_features, out_features, ps, sp=sp)
        elif backend == "te":
            # TE-backed path — cuBLAS algo may differ from MC's torch.matmul
            # under bf16, producing ~3e-4 loss-level drift. Keep for future
            # FP8 / fused-norm paths.
            _wrapper = ColumnParallelLinear(
                in_features, out_features, ps,
                bias=False, gather_output=not sp,
            )
            self.linear = _wrapper.linear
        else:
            raise ValueError(f"Unknown LM-head backend: {backend!r}")


class VocabParallelOutput(nn.Module):
    """Output projection split across TP on the vocab dimension (column parallel).

    Default backend is `"vanilla"` (torch.matmul) to match bridge backend's
    `tensor_parallel.ColumnParallelLinear` bit-for-bit. Pass `backend="te"`
    to use TE's `te.Linear` kernel (e.g. for FP8 inference paths).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        ps: ParallelState,
        *,
        backend: str = "vanilla",
    ):
        super().__init__()
        padded_vocab = pad_vocab_for_tp(vocab_size, ps.tp_size)
        # SP-aware head: when tp>1 we run on SP-sharded input (matches MC
        # GPTModel where final_layernorm runs on SP-sharded hiddens and
        # output_layer gathers internally + reduce-scatters on backward).
        self.col = _ColForLMHead(
            hidden_size, padded_vocab, ps, backend=backend, sp=ps.tp_size > 1,
        )
        self.padded_vocab = padded_vocab
        self.local_vocab = padded_vocab // ps.tp_size
        self.vocab_size = vocab_size
        self.ps = ps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.col.linear(x)

    def gather(self, logits: torch.Tensor) -> torch.Tensor:
        """All-gather TP-sharded logits and trim to actual vocab_size."""
        if self.ps.tp_size > 1:
            chunks = [torch.empty_like(logits) for _ in range(self.ps.tp_size)]
            dist.all_gather(chunks, logits, group=self.ps.tp_group)
            logits = torch.cat(chunks, dim=-1)
        return logits[..., :self.vocab_size]


class _AllGatherLastDim(torch.autograd.Function):
    """all-gather along last dim; backward = take local shard."""

    @staticmethod
    def forward(ctx, x, tp_size, group):
        ctx.tp_size = tp_size
        ctx.group = group
        ctx.local_dim = x.shape[-1]
        ctx.rank = dist.get_rank(group)
        chunks = [torch.empty_like(x) for _ in range(tp_size)]
        dist.all_gather(chunks, x.contiguous(), group=group)
        return torch.cat(chunks, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.rank * ctx.local_dim
        return grad_output[..., start:start + ctx.local_dim].contiguous(), None, None


class _ReduceFromTP(torch.autograd.Function):
    """all-reduce forward; identity backward."""

    @staticmethod
    def forward(ctx, x, group):
        out = x.clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VanillaColumnParallelLinear",
    "VocabParallelEmbedding",
    "VocabParallelOutput",
    "pad_vocab_for_tp",
]
