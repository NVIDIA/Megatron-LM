# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared pieces of the expert-tensor-parallel (ETP) inference MoE oracles (R03 plan 1).

Contract, written from the model math, the checkpoint layout and the dispatcher's documented
interface before reading the dispatcher or the experts:

* **Output.** For each real token t on any rank,
  ``out_t = sum_k p_tk * FC2_e(act(FC1_e x_t))`` with ``e = routing[t, k]`` and the *unsharded*
  expert weights: ``FC1_e`` is ``[F, H]`` (``[2F, H] = [gate; up]`` for SwiGLU), ``FC2_e`` is
  ``[H, F]``. ``act`` is squared ReLU, ``(s tanh(relu(z) / s))**2`` with the tanh clamp
  (``activation_func_tanh_clamp_scale = s``), or ``silu(gate) * up``.
* **Layout independence.** The output does not depend on tensor parallelism TP (the tokens of a
  TP group are the sequence-parallel shards of one padded sequence), expert TP (rank ``r`` of an
  expert-TP group holds FFN columns ``[r F/ETP, (r+1) F/ETP)`` of FC1 and FC2; for SwiGLU of gate
  and of up separately, as the checkpoint's sharded factory splits them,
  ``megatron/core/transformer/mlp.py`` ``apply_swiglu_sharded_factory``), expert parallelism EP
  (rank ``q`` holds experts ``[q E/EP, (q+1) E/EP)``), or which rank holds which token. The
  tokens of ``ETP * EP / TP`` data-parallel replicas meet in one expert tensor-and-model-parallel
  (tp_ep) group.
* **Padding.** Rows of a TP group's padded sequence at or past the real token count the context
  publishes route to no expert (``NVLSAllGatherVDispatcher.token_dispatch``): their output is 0.
* **Rounding points** of the inference design, used by the trusted implementation: bf16 FC1
  output, bf16 activation, bf16 FC2 output per expert-TP shard, fp32 sums over the top-k slots
  and over the shards (the reduce-scatter buffer is fp32), one bf16 rounding of the layer output.
  Batch-invariant mode applies the probability at the activation (one more bf16 rounding)
  instead of after FC2.
* **Memory.** The NVLS all-gather and reduce-scatter buffers hold ``per_rank_max * tp_ep`` rows.
  Rows past the step's gathered tokens and the bytes past each buffer in its symmetric
  allocation are not written by the gather; the torch and vLLM experts leave the reduce-scatter
  rows past the gathered tokens untouched (``unpermute_tokens`` and ``_moe_sum`` docstrings).
  Inputs are read-only.

Tolerance classes: floating point bounded by the trusted implementation (harness policy, per
row, factor 2); exact against the fp64 reference rounded once, on integer inputs whose every
intermediate is representable; bitwise metamorphic in batch-invariant mode.
"""

import contextlib
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterator, List, Optional, Sequence, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.test_utilities import Utils

NVLS_KEYS = ("ep_agv_h", "ep_agv_r", "ep_agv_p", "ep_rsv", "ep_meta")
# NaN payloads no kernel produces (the harness's guard bits): a write changes the bits, a read
# turns an output into NaN.
POISON_BITS = {torch.bfloat16: 0x7FA5, torch.float32: 0x7FA5A5A5}
TAIL_BYTE = 0xA5
CLAMP_SCALE = 0.5  # small against N(0, 1) FC1 outputs, so the tanh clamp saturates


# ---------------------------------------------------------------------------------------------
# Parallel layouts
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Attention TP, expert TP and EP sizes (PP = CP = 1)."""

    tp: int
    etp: int
    ep: int

    @property
    def tp_ep(self) -> int:
        """Size of the expert tensor-and-model-parallel group the NVLS dispatcher spans."""
        return self.etp * self.ep

    @property
    def world(self) -> int:
        """Smallest world the layout fits."""
        return math.lcm(self.tp, self.tp_ep)

    @property
    def replicas(self) -> int:
        """Data-parallel replicas whose tokens one tp_ep group gathers."""
        return max(1, self.tp_ep // self.tp)

    @property
    def name(self) -> str:
        return f"w{self.world}-tp{self.tp}-etp{self.etp}-ep{self.ep}"


H100_LAYOUTS = [
    Layout(1, 1, 2),  # ETP 1 control: tp_ep = EP
    Layout(1, 2, 1),  # ETP > TP: tp_ep spans two replicas
    Layout(2, 2, 1),  # ETP = TP: tp_ep is the TP group (sequence-parallel shards)
    Layout(2, 1, 2),  # ETP < TP
    Layout(1, 2, 2),  # four replicas
    Layout(2, 2, 2),  # two replicas of two SP shards
    Layout(1, 4, 1),  # four expert-TP shards
]
LAYOUTS = {layout.name: layout for layout in H100_LAYOUTS}


def sync_all() -> None:
    """Every rank's GPU work done and every rank here: NVLS buffers can be touched safely."""
    torch.cuda.synchronize()
    dist.barrier()


def release_buffers() -> None:
    """Drop the NVLS symmetric buffers and the vLLM pool (class-level state across tests)."""
    from megatron.core.inference.moe.vllm_fused_moe import VllmFusedMoeBuffers
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    if dist.is_initialized():
        sync_all()
    NVLSAllGatherVDispatcher._delete_buffers()
    for key in NVLS_KEYS:
        SymmetricMemoryManager.destroy(key)
    VllmFusedMoeBuffers._delete_buffers()


@contextlib.contextmanager
def parallel_layout(layout: Layout) -> Iterator[SimpleNamespace]:
    """Process groups for ``layout`` on every rank of the world (it must divide the world)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA GPUs")
    if Utils.world_size % layout.world:
        pytest.skip(f"{layout.name} needs a multiple of {layout.world} ranks")
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if dist.is_initialized():
        release_buffers()
    Utils.initialize_model_parallel(
        layout.tp, 1, expert_model_parallel_size=layout.ep, expert_tensor_parallel_size=layout.etp
    )
    try:
        model_parallel_cuda_manual_seed(123)
        tp_ep_group = ps.get_expert_tensor_and_model_parallel_group()
        assert dist.get_world_size(tp_ep_group) == layout.tp_ep
        dp_rank = ps.get_data_parallel_rank()
        env = SimpleNamespace(
            layout=layout,
            rank=dist.get_rank(),
            tp_rank=ps.get_tensor_model_parallel_rank(),
            dp_rank=dp_rank,
            ep_rank=ps.get_expert_model_parallel_rank(),
            etp_rank=ps.get_expert_tensor_parallel_rank(),
            tp_ep_group=tp_ep_group,
            tp_ep_rank=dist.get_rank(tp_ep_group),
            # TP is innermost in both rank layouts, so a tp_ep group holds consecutive TP groups.
            replica=dp_rank % layout.replicas,
            nvls_dims=None,
        )
        yield env
    finally:
        release_buffers()
        Utils.destroy_model_parallel()


def allocate_nvls(env: SimpleNamespace, per_rank_max: int, topk: int, hidden: int) -> None:
    """NVLS buffers over the layout's tp_ep group, reused while the dimensions stay the same."""
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    dims = (per_rank_max, topk, hidden)
    if env.nvls_dims == dims and NVLSAllGatherVDispatcher._symm_rsv is not None:
        return
    release_buffers()
    NVLSAllGatherVDispatcher.allocate_buffers(
        per_rank_worst_case_token_count=per_rank_max,
        topk=topk,
        hidden_size=hidden,
        tp_ep_group=env.tp_ep_group,
    )
    env.nvls_dims = dims


# ---------------------------------------------------------------------------------------------
# Problems: global weights, tokens, placements
# ---------------------------------------------------------------------------------------------


@dataclass
class Problem:
    """Global (unsharded) expert weights, identical on every rank."""

    experts: int
    hidden: int
    ffn: int
    topk: int
    activation: str  # "squared_relu", "clamp" or "swiglu"
    fc1: torch.Tensor  # [E, F or 2F, H] bf16; [gate; up] for SwiGLU
    fc2: torch.Tensor  # [E, H, F] bf16
    integer: bool = False

    @property
    def glu(self) -> bool:
        return self.activation == "swiglu"

    @property
    def clamp(self) -> Optional[float]:
        return CLAMP_SCALE if self.activation == "clamp" else None


def _generator(seed: int) -> torch.Generator:
    return ou.make_generator(seed, device="cuda")


def random_problem(
    experts: int, hidden: int, ffn: int, topk: int, activation: str, seed: int
) -> Problem:
    """Gaussian weights scaled by 1/sqrt(fan-in), so every activation is O(1)."""
    g = _generator(seed)
    rows = 2 * ffn if activation == "swiglu" else ffn
    fc1 = torch.randn(experts, rows, hidden, device="cuda", generator=g) / math.sqrt(hidden)
    fc2 = torch.randn(experts, hidden, ffn, device="cuda", generator=g) / math.sqrt(ffn)
    return Problem(experts, hidden, ffn, topk, activation, fc1.bfloat16(), fc2.bfloat16())


FC1_NONZEROS = 4  # per FC1 row: |FC1 x| <= 4 for x in {-1, 0, 1}, squared ReLU <= 16
FC2_NONZEROS = 8  # per FC2 row: every FC2 output (and every shard's) <= 8 * 16 = 128


def _sparse_signs(rows: int, cols: int, nonzeros: int, g: torch.Generator) -> torch.Tensor:
    """``[rows, cols]`` with exactly ``nonzeros`` entries of +-1 per row."""
    where = torch.rand(rows, cols, device="cuda", generator=g).argsort(dim=-1)[:, :nonzeros]
    signs = torch.randint(0, 2, (rows, nonzeros), device="cuda", generator=g) * 2 - 1
    out = torch.zeros(rows, cols, device="cuda")
    out.scatter_(1, where, signs.float())
    return out


def integer_problem(experts: int, hidden: int, ffn: int, topk: int, seed: int) -> Problem:
    """Squared-ReLU experts on which every rounding point of every backend is exact.

    With tokens in {-1, 0, 1} and powers of two as probabilities: FC1 outputs are integers of
    magnitude <= 4 and squared ReLU <= 16 (exact in bf16); every FC2 output, and every expert-TP
    shard's partial, is an integer of magnitude <= 128 (exact in bf16); ``p * y`` and
    ``p * act`` are exact; every fp32 sum of them is exact in any order. So the layer output must
    equal the fp64 reference rounded once to bf16, bit for bit, on every layout and backend.
    """
    g = _generator(seed)
    fc1 = _sparse_signs(experts * ffn, hidden, FC1_NONZEROS, g).view(experts, ffn, hidden)
    fc2 = _sparse_signs(experts * hidden, ffn, FC2_NONZEROS, g).view(experts, hidden, ffn)
    return Problem(
        experts, hidden, ffn, topk, "squared_relu", fc1.bfloat16(), fc2.bfloat16(), integer=True
    )


def fc1_shard(problem: Problem, expert: int, cols: slice) -> torch.Tensor:
    """FC1 rows of one expert-TP shard: ``[gate_r; up_r]`` for SwiGLU (checkpoint convention)."""
    w = problem.fc1[expert]
    if problem.glu:
        return torch.cat([w[: problem.ffn][cols], w[problem.ffn :][cols]], dim=0)
    return w[cols]


def shard_columns(problem: Problem, etp: int, etp_rank: int) -> slice:
    width = problem.ffn // etp
    return slice(etp_rank * width, (etp_rank + 1) * width)


@dataclass
class Tokens:
    """A pool of tokens: hidden states, routing and probabilities."""

    hidden: torch.Tensor  # [N, H] bf16
    routing: torch.Tensor  # [N, k] int64
    probs: torch.Tensor  # [N, k] fp32
    experts: int


ROUTINGS = ("random", "one-rank", "empty-expert", "hot-expert")


def make_routing(n: int, experts: int, topk: int, pattern: str, ep: int, g) -> torch.Tensor:
    """``[n, topk]`` distinct experts per token.

    one-rank: every slot of a token on one EP rank's experts (needs topk <= E/EP); empty-expert:
    expert 0 never chosen (an EP rank with an empty local expert); hot-expert: slot 0 is always
    expert E-1.
    """
    scores = torch.rand(n, experts, device="cuda", generator=g)
    if pattern == "one-rank":
        local = experts // ep
        assert topk <= local, f"one-rank routing needs topk {topk} <= E/EP {local}"
        owner = torch.arange(experts, device="cuda") // local
        rank = torch.randint(0, ep, (n, 1), device="cuda", generator=g)
        scores = scores + 2.0 * (owner[None, :] != rank)
    elif pattern == "empty-expert":
        assert topk < experts
        scores[:, 0] = 3.0
    elif pattern == "hot-expert":
        scores[:, experts - 1] = -1.0
    else:
        assert pattern == "random", pattern
    return scores.argsort(dim=-1)[:, :topk].contiguous()


def make_tokens(
    problem: Problem, n: int, seed: int, routing: str = "random", ep: int = 1
) -> Tokens:
    """``n`` tokens; integer problems get tokens in {-1, 0, 1} and power-of-two probabilities."""
    g = _generator(seed)
    shape = (n, problem.hidden)
    if problem.integer:
        values = torch.randint(-1, 2, shape, device="cuda", generator=g)
        hidden = (values * (torch.rand(shape, device="cuda", generator=g) < 0.6)).bfloat16()
        exponents = torch.randint(0, 4, (n, problem.topk), device="cuda", generator=g)
        probs = torch.pow(2.0, -exponents.float())
    else:
        hidden = torch.randn(shape, device="cuda", generator=g).bfloat16()
        probs = 0.05 + 0.95 * torch.rand(n, problem.topk, device="cuda", generator=g)
    route = make_routing(n, problem.experts, problem.topk, routing, ep, g)
    return Tokens(hidden, route, probs.float().contiguous(), problem.experts)


COUNTS = ("one", "uneven", "full", "edge", "padded")


def replica_counts(pattern: str, layout: Layout, per_rank_max: int) -> List[Tuple[int, int]]:
    """``(real, padded)`` sequence lengths of each replica in a tp_ep group.

    ``padded`` is a multiple of TP (the context rounds the step to TP), so every rank of a TP
    group holds ``padded / TP`` rows; ``real`` is what the context publishes. Patterns, from the
    gather's structure (per-rank capacity ``per_rank_max`` rows, variable per-rank counts):
    one token; uneven odd counts; every rank at capacity (the gather buffer full); capacity - 1
    beside capacity; graph-bucket padding to capacity with 0 (an idle replica's dummy step), 1
    and odd real tokens.
    """
    tp, full = layout.tp, per_rank_max * layout.tp

    def pad(n: int) -> int:
        return max(tp, -(-n // tp) * tp)

    reps = range(layout.replicas)
    if pattern == "padded":
        return [([full - 7, 0, 9, 1][j % 4], full) for j in reps]
    real = {
        "one": [1 for _ in reps],
        "uneven": [[17, 5, 33, 2][j % 4] for j in reps],
        "full": [full for _ in reps],
        "edge": [full - 1 if j % 2 == 0 else full for j in reps],
    }[pattern]
    return [(n, pad(n)) for n in real]


@dataclass
class Batch:
    """One rank's MoE input rows (its shard of its replica's padded sequence)."""

    x: torch.Tensor  # [L, H] bf16
    routing: torch.Tensor  # [L, k] int64
    probs: torch.Tensor  # [L, k] fp32
    ids: torch.Tensor  # [L] token id in the pool, -1 for padding rows
    real: int  # real tokens of the replica's sequence (what the context publishes)

    @property
    def is_real(self) -> torch.Tensor:
        return self.ids >= 0


def local_batch(
    tokens: Tokens,
    ids: torch.Tensor,
    padded: int,
    env: SimpleNamespace,
    padding_hidden: float = float("nan"),
) -> Batch:
    """This rank's rows of a replica sequence of ``ids`` padded to ``padded`` rows.

    Padding rows hold NaN hidden states, NaN probabilities and valid expert ids, as stale
    CUDA-graph inputs would: only the dispatcher's padding mask keeps them out of the experts.
    """
    n, tp = ids.numel(), env.layout.tp
    assert padded % tp == 0 and n <= padded
    k, hidden = tokens.routing.shape[1], tokens.hidden.shape[1]
    seq_x = torch.full((padded, hidden), padding_hidden, dtype=torch.bfloat16, device="cuda")
    seq_p = torch.full((padded, k), float("nan"), dtype=torch.float32, device="cuda")
    seq_r = torch.arange(padded * k, device="cuda").view(padded, k) % tokens.experts
    seq_ids = torch.full((padded,), -1, dtype=torch.int64, device="cuda")
    seq_x[:n], seq_p[:n], seq_r[:n], seq_ids[:n] = (
        tokens.hidden[ids],
        tokens.probs[ids],
        tokens.routing[ids],
        ids,
    )
    rows = slice(env.tp_rank * padded // tp, (env.tp_rank + 1) * padded // tp)
    return Batch(
        seq_x[rows].contiguous(),
        seq_r[rows].contiguous(),
        seq_p[rows].contiguous(),
        seq_ids[rows].contiguous(),
        n,
    )


def placement(
    counts: Sequence[Tuple[int, int]], order: Optional[torch.Tensor] = None
) -> List[Tuple[torch.Tensor, int]]:
    """Token ids and padded length per replica: consecutive ids of a pool, or of ``order``."""
    total = sum(n for n, _ in counts)
    order = torch.arange(total, device="cuda") if order is None else order
    out, start = [], 0
    for n, padded in counts:
        out.append((order[start : start + n], padded))
        start += n
    return out


def rank_batch(
    tokens: Tokens, counts: Sequence[Tuple[int, int]], env: SimpleNamespace, order=None, **kw
) -> Batch:
    ids, padded = placement(counts, order)[env.replica]
    return local_batch(tokens, ids, padded, env, **kw)


# ---------------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------------


def _activate(problem: Problem, h: torch.Tensor) -> torch.Tensor:
    if problem.glu:
        gate, up = h.chunk(2, dim=-1)
        return F.silu(gate) * up
    r = torch.relu(h)
    if problem.clamp is not None:
        r = problem.clamp * torch.tanh(r / problem.clamp)
    return r * r


def _routed(routing: torch.Tensor) -> Iterator[Tuple[int, torch.Tensor, torch.Tensor]]:
    """``(expert, rows routed to it, the slot of each)``; each expert's weights are touched once."""
    for expert in routing.unique().tolist():
        rows, slots = (routing == expert).nonzero(as_tuple=True)
        yield expert, rows, slots


def reference(
    problem: Problem, hidden: torch.Tensor, routing: torch.Tensor, probs: torch.Tensor
) -> torch.Tensor:
    """fp64 per-token MoE with the unsharded weights (each expert upcast exactly when used)."""

    def moe(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], problem.hidden, dtype=torch.float64, device=x.device)
        for expert, rows, slots in _routed(routing):
            h = x[rows] @ problem.fc1[expert].double().t()
            y = _activate(problem, h) @ problem.fc2[expert].double().t()
            out.index_add_(0, rows, p[rows, slots, None] * y)
        return out

    return ou.fp64_reference(moe, hidden, probs)


@contextlib.contextmanager
def fp32_matmul() -> Iterator[None]:
    """Full fp32 GEMMs (no TF32) for the trusted implementation."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def trusted(
    problem: Problem,
    hidden: torch.Tensor,
    routing: torch.Tensor,
    probs: torch.Tensor,
    etp: int,
    probs_at_activation: bool = False,
) -> torch.Tensor:
    """torch MoE at the inference design's rounding points, per expert-TP shard (bf16 out)."""
    total = torch.zeros(hidden.shape[0], problem.hidden, dtype=torch.float32, device="cuda")
    with fp32_matmul():
        for shard in range(etp):
            cols = shard_columns(problem, etp, shard)
            partial = torch.zeros_like(total)
            for expert, rows, slots in _routed(routing):
                w1 = fc1_shard(problem, expert, cols).float()
                w2 = problem.fc2[expert][:, cols].float()
                h = (hidden[rows].float() @ w1.t()).bfloat16()
                a = _activate(problem, h.float()).bfloat16()
                p = probs[rows, slots, None]
                if probs_at_activation:
                    a = (a.float() * p).bfloat16()
                y = (a.float() @ w2.t()).bfloat16().float()
                partial.index_add_(0, rows, y if probs_at_activation else y * p)
            total += partial
    return total.bfloat16()


# ---------------------------------------------------------------------------------------------
# The layer under test
# ---------------------------------------------------------------------------------------------


def make_config(
    problem: Problem, layout: Layout, backend: str, *, batch_invariant: bool = False, **overrides
):
    """An inference-optimized MoE config for ``layout`` with the NVLS dispatcher."""
    from megatron.core.activations import squared_relu
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    kwargs = dict(
        num_layers=1,
        hidden_size=problem.hidden,
        num_attention_heads=4,
        num_moe_experts=problem.experts,
        moe_ffn_hidden_size=problem.ffn,
        moe_router_topk=problem.topk,
        # Top-1 routing needs the pre-softmax router (TransformerConfig check); the router is
        # bypassed except in test_router_through_the_layer, which uses top-2.
        moe_router_pre_softmax=problem.topk == 1,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="inference_optimized",
        use_cpu_initialization=True,
        tensor_model_parallel_size=layout.tp,
        sequence_parallel=layout.tp > 1,
        expert_tensor_parallel_size=layout.etp,
        expert_model_parallel_size=layout.ep,
        inference_grouped_gemm_backend=backend,
        inference_moe_token_dispatcher_type="nvls",
        activation_func=F.silu if problem.glu else squared_relu,
        gated_linear_unit=problem.glu,
        activation_func_tanh_clamp_scale=problem.clamp,
    )
    if batch_invariant:
        kwargs.update(
            batch_invariant_mode=True,
            batch_invariant_backend="te_native",
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            attention_dropout=0.0,
        )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def local_expert_range(problem: Problem, env: SimpleNamespace) -> range:
    local = problem.experts // env.layout.ep
    return range(env.ep_rank * local, (env.ep_rank + 1) * local)


def build_layer(config, problem: Problem, env: SimpleNamespace):
    """The inference MoE layer from the production spec, with this rank's shards loaded."""
    from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec

    layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
    load_shards(layer, problem, env)
    return layer


def load_shards(layer, problem: Problem, env: SimpleNamespace) -> None:
    """Write this rank's expert shards into the layer's per-expert weights.

    The experts concatenate (or quantize) these at their first inference forward, as after a
    checkpoint load or refit; MCore MXFP8 weights requantize in place (``MXFP8Tensor.copy_``).
    """
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    assert isinstance(layer.experts, InferenceGroupedMLP), type(layer.experts)
    assert isinstance(layer._inference_token_dispatcher, NVLSAllGatherVDispatcher)
    experts = local_expert_range(problem, env)
    assert list(layer.local_expert_indices) == list(experts), layer.local_expert_indices
    cols = shard_columns(problem, env.layout.etp, env.etp_rank)
    with torch.no_grad():
        for i, expert in enumerate(experts):
            for name, value in (
                ("linear_fc1", fc1_shard(problem, expert, cols)),
                ("linear_fc2", problem.fc2[expert][:, cols]),
            ):
                weight = getattr(getattr(layer.experts, name), f"weight{i}")
                assert tuple(weight.shape) == tuple(value.shape), (
                    f"{name}.weight{i} is {tuple(weight.shape)}; the {env.layout.name} shard of"
                    f" expert {expert} is {tuple(value.shape)}"
                )
                weight.copy_(value.contiguous())


@contextlib.contextmanager
def batch_invariant_mode(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    with set_batch_invariant_mode(True, backend="te_native"):
        yield


def require_batch_invariant_backend(backend: str) -> None:
    """The torch backend's batch-invariant grouped GEMM is DeepGEMM; the CI image ships it."""
    if backend == "torch":
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            HAVE_DEEPGEMM_BF16,
        )

        assert HAVE_DEEPGEMM_BF16, "batch-invariant torch MoE needs DeepGEMM bf16 grouped GEMMs"


def bind_real_count(real: int) -> torch.Tensor:
    """Publish this TP group's real token count to the dispatcher, as the context does."""
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    count = torch.tensor([real], dtype=torch.int32, device="cuda")
    NVLSAllGatherVDispatcher.set_real_token_count_tensor(count)
    return count


def bypass_router(layer, probs: torch.Tensor, routing: torch.Tensor) -> None:
    """Make ``layer.route`` return the given routing (the router has its own oracle, O8a)."""

    def route(*args, **kwargs):
        return probs, routing

    layer.route = route


def forward(layer, x: torch.Tensor) -> torch.Tensor:
    """The layer's inference forward on ``[L, H]`` rows of this rank."""
    from megatron.core.inference.utils import InferenceMode

    with torch.no_grad(), InferenceMode.active():
        out, bias = layer(x.view(x.shape[0], 1, x.shape[1]))
    assert bias is None
    return out.view(x.shape[0], -1)


def gathered_rows(env: SimpleNamespace, local_rows: int) -> int:
    """Rows the step gathers over this rank's tp_ep group."""
    count = torch.tensor([local_rows], dtype=torch.int64, device="cuda")
    dist.all_reduce(count, group=env.tp_ep_group)
    return int(count.item())


def _fill_bits(t: torch.Tensor, bits: int) -> None:
    int_dtype = {2: torch.int16, 4: torch.int32}[t.element_size()]
    t.view(int_dtype).fill_(bits)


def _tail(key: str, tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    raw = SymmetricMemoryManager.get_buffer(key).symm_buffer
    return raw[tensor.numel() * tensor.element_size() :]


def _nvls_tensors() -> dict:
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    d = NVLSAllGatherVDispatcher
    return {
        "ep_agv_h": d._symm_agv_hidden["tensor"],
        "ep_agv_r": d._symm_agv_routing["tensor"],
        "ep_agv_p": d._symm_agv_probs["tensor"],
        "ep_rsv": d._symm_rsv["tensor"],
    }


def poison_nvls(stale_expert: int) -> None:
    """Poison every NVLS buffer and the bytes past it, on every rank, between steps.

    Hidden states, probabilities and reduce-scatter rows get NaN payloads no kernel writes;
    routing rows get a valid expert id of this rank (stale rows of an earlier step look like
    that), so an expert kernel that ignored the valid-token count would compute them.
    """
    sync_all()
    tensors = _nvls_tensors()
    _fill_bits(tensors["ep_agv_h"], POISON_BITS[torch.bfloat16])
    _fill_bits(tensors["ep_agv_p"], POISON_BITS[torch.float32])
    tensors["ep_agv_r"].fill_(stale_expert)
    _fill_bits(tensors["ep_rsv"], POISON_BITS[torch.float32])
    for key, tensor in tensors.items():
        _tail(key, tensor).fill_(TAIL_BYTE)
    sync_all()


def check_nvls(gathered: int, stale_expert: int, *, flashinfer: bool = False, what: str) -> None:
    """After a step: rows past the gathered tokens and the bytes past each buffer untouched.

    The FlashInfer backends are the exception their contract states: the dispatcher sets every
    routing row to -1 before the gather (``update_metadata``), and the kernels write the whole
    reduce-scatter buffer (zero rows for -1 routing).
    """
    sync_all()
    tensors = _nvls_tensors()
    problems = []
    for key, tensor in tensors.items():
        changed = (_tail(key, tensor) != TAIL_BYTE).nonzero()
        if changed.numel():
            problems.append(f"{key}: {changed.numel()} bytes past the buffer written")
    expected = {
        "ep_agv_h": POISON_BITS[torch.bfloat16],
        "ep_agv_p": POISON_BITS[torch.float32],
        "ep_agv_r": -1 if flashinfer else stale_expert,
    }
    if not flashinfer:
        expected["ep_rsv"] = POISON_BITS[torch.float32]
    for key, value in expected.items():
        rows = tensors[key][gathered:]
        bits = (
            rows
            if key == "ep_agv_r"
            else rows.view({2: torch.int16, 4: torch.int32}[rows.element_size()])
        )
        bad = (bits != value).any(dim=-1).nonzero()
        if bad.numel():
            problems.append(
                f"{key}: {bad.numel()} rows past the {gathered} gathered rows written (first at"
                f" row {gathered + int(bad[0])})"
            )
    assert not problems, f"{what}: " + "; ".join(problems)


def check_output(
    out: torch.Tensor,
    batch: Batch,
    tokens: Tokens,
    problem: Problem,
    etp: int,
    *,
    probs_at_activation: bool = False,
    exact: bool = False,
    trusted_fn=None,
    what: str,
) -> None:
    """Real rows against the fp64 reference, padding rows exactly 0.

    ``trusted_fn(problem, x, routing, probs)`` replaces the default trusted implementation.
    """
    assert out.shape == batch.x.shape and out.dtype == torch.bfloat16, (out.shape, out.dtype)
    padding = out[~batch.is_real]
    ou.assert_exact(padding, torch.zeros_like(padding), what=f"{what}: padding rows")
    ids = batch.ids[batch.is_real]
    if ids.numel() == 0:
        return
    x, routing, probs = tokens.hidden[ids], tokens.routing[ids], tokens.probs[ids]
    ref = reference(problem, x, routing, probs)
    if exact:
        ou.assert_exact(out[batch.is_real], ref.to(torch.bfloat16), what=what)
    else:
        ou.assert_close_to_reference(
            out[batch.is_real],
            ref,
            (
                trusted_fn(problem, x, routing, probs)
                if trusted_fn is not None
                else trusted(problem, x, routing, probs, etp, probs_at_activation)
            ),
            dim=-1,
            what=what,
        )


def run_step(
    env: SimpleNamespace,
    layer,
    problem: Problem,
    batch: Batch,
    *,
    flashinfer: bool = False,
    bind: bool = True,
    route: bool = False,
):
    """One poisoned, checked inference step of ``layer`` on this rank's ``batch``.

    ``bind=False`` keeps the real-token count a context bound; ``route=True`` runs the layer's
    own router instead of ``batch``'s routing.
    """
    stale_expert = local_expert_range(problem, env)[0]
    poison_nvls(stale_expert)
    if bind:
        bind_real_count(batch.real)
    x, probs, routing = batch.x.clone(), batch.probs.clone(), batch.routing.clone()
    if not route:
        bypass_router(layer, probs, routing)
    out = forward(layer, x)
    gathered = gathered_rows(env, batch.x.shape[0])
    check_nvls(
        gathered, stale_expert, flashinfer=flashinfer, what=f"{env.layout.name} rank {env.rank}"
    )
    ou.assert_exact(x, batch.x, what="hidden states are read-only")
    ou.assert_exact(probs, batch.probs, what="probabilities are read-only")
    return out


def gather_by_token(
    env: SimpleNamespace, out: torch.Tensor, batch: Batch, num_tokens: int, per_rank_max: int
) -> torch.Tensor:
    """Every real token's output in the tp_ep group, indexed by its pool id, on every rank."""
    rows, hidden = out.shape
    size = env.layout.tp_ep
    send = torch.zeros(per_rank_max, hidden, dtype=out.dtype, device="cuda")
    send_ids = torch.full((per_rank_max,), -1, dtype=torch.int64, device="cuda")
    send[:rows], send_ids[:rows] = out, batch.ids
    recv = torch.empty(size * per_rank_max, hidden, dtype=out.dtype, device="cuda")
    recv_ids = torch.empty(size * per_rank_max, dtype=torch.int64, device="cuda")
    dist.all_gather_into_tensor(recv, send, group=env.tp_ep_group)
    dist.all_gather_into_tensor(recv_ids, send_ids, group=env.tp_ep_group)
    held = recv_ids >= 0
    assert (
        recv_ids[held].sort().values.equal(torch.arange(num_tokens, device="cuda"))
    ), "every pool token is held by exactly one rank"
    result = torch.empty(num_tokens, hidden, dtype=out.dtype, device="cuda")
    result[recv_ids[held]] = recv[held]
    return result


# ---------------------------------------------------------------------------------------------
# Lossless inputs for the FlashInfer MXFP8 / NVFP4 backends (Blackwell)
# ---------------------------------------------------------------------------------------------

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Exponents of the lossless construction (powers of two in the values below).
LOSSLESS_EXP_V = -3  # FC1 weight magnitude 6 * 2**3
LOSSLESS_EXP_X = -2  # token magnitude 6 * 2**2
LOSSLESS_EXP_W2 = 8  # FC2 weight magnitude 6 * 2**-8


def grid_values(shape: Sequence[int], exponent: int, g: torch.Generator) -> torch.Tensor:
    """E2M1-grid values times ``2**-exponent`` with ``+-6 * 2**-exponent`` first in every
    16-block of the last dim: exact under MXFP8 (32-blocks), NVFP4 (16-blocks) and bf16."""
    grid = torch.tensor(E2M1_GRID, device="cuda")
    numel = math.prod(shape)
    assert shape[-1] % 16 == 0
    values = grid[torch.randint(0, 8, (numel,), device="cuda", generator=g)]
    signs = torch.randint(0, 2, (numel,), device="cuda", generator=g).float() * 2 - 1
    blocks = (values * signs).view(-1, 16)
    pins = torch.randint(0, 2, (blocks.shape[0],), device="cuda", generator=g).float() * 2 - 1
    blocks[:, 0] = 6.0 * pins
    return (blocks.view(*shape) * 2.0**-exponent).bfloat16()


def _block_levels(rows: int, g: torch.Generator) -> torch.Tensor:
    """``+-2**-n`` with ``n`` in {0, 1} constant over every 32-row block (an FC2-input
    quantization block), random signs per row."""
    assert rows % 32 == 0
    level = torch.pow(
        2.0, -torch.randint(0, 2, (rows // 32, 1), device="cuda", generator=g).float()
    )
    signs = torch.randint(0, 2, (rows // 32, 32), device="cuda", generator=g).float() * 2 - 1
    return (level * signs).view(rows)


def lossless_problem(
    experts: int, hidden: int, ffn: int, topk: int, activation: str, seed: int
) -> Problem:
    """Experts whose FC2 input every MXFP8/NVFP4 recipe represents exactly (after O4b).

    FC1 of expert e is rank one, ``u_e (c e_j)^T`` with ``c = 6 * 2**3`` at one coordinate
    ``j = coordinate(e)`` of the hidden state and ``u_e = +-2**-n`` constant in magnitude over
    every 32-block of FFN rows (``n`` in {0, 1}); every FC1 row then has a single nonzero, a
    power of two times 6 (exact under every recipe, zero blocks included). Tokens
    (``lossless_tokens``) carry ``+-{1, 2, 4} * 2**2`` at those coordinates, so every FC1
    output is ``+-6 * 2**m`` with ``|.| >= 96`` and the FC2 input is a block-constant
    ``0`` or ``36 * 4**m`` (squared ReLU), exactly ``s**2 = 0.25`` (tanh clamp, ``tanh`` of at
    least 192 is 1 in fp32 and fp64), or ``gate * up`` with ``gate >= 96`` (SwiGLU, ``silu``
    exact) -- each block a single magnitude, which E4M3 and E2M1 codes with a power-of-two
    scale hold exactly. FC2 is ``grid_values`` (exact). So the fp64 MoE of these values is the
    exact contract, and a backend's only roundings are its bf16 outputs.
    """
    g = _generator(seed)
    c = 6.0 * 2.0**-LOSSLESS_EXP_V
    rows = 2 * ffn if activation == "swiglu" else ffn
    fc1 = torch.zeros(experts, rows, hidden, device="cuda")
    for e in range(experts):
        if activation == "swiglu":
            u = torch.cat([_block_levels(ffn, g), _block_levels(ffn, g)])  # [gate; up]
        else:
            u = _block_levels(rows, g)
        fc1[e, :, lossless_coordinate(e, hidden)] = u * c
    fc2 = grid_values((experts, hidden, ffn), LOSSLESS_EXP_W2, g)
    return Problem(experts, hidden, ffn, topk, activation, fc1.bfloat16(), fc2)


def lossless_coordinate(expert: int, hidden: int) -> int:
    """The hidden coordinate expert ``e``'s FC1 reads: never a block's pinned first element."""
    return (16 * (expert % (hidden // 16)) + 1 + expert % 15) % hidden


def lossless_tokens(
    problem: Problem, n: int, seed: int, routing: str = "random", ep: int = 1
) -> Tokens:
    """Tokens for ``lossless_problem``: ``grid_values`` with ``+-{1, 2, 4} * 2**2`` at every
    expert's coordinate; probabilities ``k / 16`` (exact in bf16, which routed MXFP8 packs)."""
    g = _generator(seed)
    hidden = grid_values((n, problem.hidden), LOSSLESS_EXP_X, g).float()
    for e in range(problem.experts):
        j = lossless_coordinate(e, problem.hidden)
        magnitude = torch.pow(2.0, torch.randint(0, 3, (n,), device="cuda", generator=g).float())
        sign = torch.randint(0, 2, (n,), device="cuda", generator=g).float() * 2 - 1
        hidden[:, j] = sign * magnitude * 2.0**-LOSSLESS_EXP_X
    probs = torch.randint(4, 16, (n, problem.topk), device="cuda", generator=g).float() / 16
    route = make_routing(n, problem.experts, problem.topk, routing, ep, g)
    return Tokens(hidden.bfloat16(), route, probs.contiguous(), problem.experts)


def trusted_flashinfer(
    problem: Problem,
    hidden: torch.Tensor,
    routing: torch.Tensor,
    probs: torch.Tensor,
    layout: Layout,
) -> torch.Tensor:
    """torch MoE at the FlashInfer design's least precise rounding points (bf16 out).

    Each rank (EP rank q, expert-TP shard r) finalizes its local routes into a bf16 partial,
    accumulating the bf16 route outputs ``p * y`` in bf16 (the documented atomic finalize), the
    reduce-scatter sums the partials in fp32, and the output is rounded once to bf16. The GEMMs
    run in fp32 (exact on these inputs).
    """
    local = problem.experts // layout.ep
    total = torch.zeros(hidden.shape[0], problem.hidden, dtype=torch.float32, device="cuda")
    with fp32_matmul():
        for q in range(layout.ep):
            for shard in range(layout.etp):
                cols = shard_columns(problem, layout.etp, shard)
                partial = torch.zeros_like(total).bfloat16()
                for slot in range(routing.shape[1]):
                    for expert in routing[:, slot].unique().tolist():
                        if expert // local != q:
                            continue
                        rows = routing[:, slot] == expert
                        w1 = fc1_shard(problem, expert, cols).float()
                        w2 = problem.fc2[expert][:, cols].float()
                        h = (hidden[rows].float() @ w1.t()).bfloat16()
                        a = _activate(problem, h.float()).bfloat16()
                        y = a.float() @ w2.t()
                        routed = (y * probs[rows, slot, None]).bfloat16()
                        partial[rows] = (partial[rows].float() + routed.float()).bfloat16()
                total += partial.float()
    return total.bfloat16()
