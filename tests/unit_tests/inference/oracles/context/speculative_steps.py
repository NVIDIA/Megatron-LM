# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Speculative-decoding steps as the dynamic context lays them out, and their bookkeeping oracle.

Shared by the oracle tests of the MTP kernels in this package (verify, prepare, Mamba state
copy). Nothing here reads the kernels: the layout comes from the dynamic context and the
controller (``DynamicInferenceContext.pad_active_slices`` and
``speculative_required_logit_indices``; ``TextGenerationController.
_dynamic_step_sample_logits_and_verify_tokens``), and the expected values from what
speculative decoding means.

**One step.** The active requests are the context's rows past the paused ones, decode
requests first, then prefill requests (the last of them may be a chunked-prefill request
whose prompt continues in a later step). The forward's packed tokens are

    [ D * (S + 1) decode tokens | the prefill chunks, back to back | graph padding ]

where decode request ``d`` owns the tokens ``d (S+1) .. d (S+1) + S``: its last accepted
token (the base) and its ``S`` draft tokens. The controller samples one token per *required
logit row*: every decode token, then the last token of each prefill chunk. Row ``i`` of the
required rows reads the logits at packed token ``rli[i]``:

* eager: ``rli = [0 .. D(S+1)) ++ [D(S+1) + cumsum(chunk_lengths) - 1]`` (int32);
* CUDA graph: the same, then zeros up to the captured ``D_pad (S+1) + P_pad`` slots;
* async scheduling: the forward compacts surviving rows, so the controller composes
  ``rli = token_row_indices[rli_ctx[: A (S+1)]]`` (int64), the original GPU rows.

``x[i] = input_ids[rli[i]]`` are the input tokens at the required rows and ``y[i]`` the
tokens sampled from their logits. Only rows ``< D(S+1) + P`` are meaningful; the padded
slots gather row 0 and hold stale samples.

**The contract (the bookkeeping the next forward relies on).** For decode request ``d``
with ``base = d (S+1)``, draft ``j`` (``x[base + 1 + j]``) is accepted iff it and every
earlier draft equal the token the model sampled at the position before it
(``y[base + j]``). With ``n_d`` the number of accepted drafts:

* ``accepted_tokens[d] = x[base+1 .. base+n_d] ++ [-1] * (S - n_d)``, ``accepted_count[d] = n_d``;
* ``sampled[d] = y[base + n_d]``: the model's token after the last accepted one;
* ``last_accepted_seq[d] = rli[base + n_d]``: the packed (or original GPU) token row whose
  hidden state and position the next MTP draft starts from;
* the verification mask holds ``base + i`` for ``i <= n_d`` and ``last_one[d] = base + n_d``.

For prefill request ``p`` (row ``a = D + p``, required row ``r = D (S+1) + p``):
``accepted_tokens[a] = [-1] * S``, ``accepted_count[a] = 0``, ``sampled[a] = y[r]``,
``last_accepted_seq[a] = rli[r]``, ``mask[r] = 1``, ``last_one[a] = r``. Buffer rows ``>= A``
belong to no active request this step.
"""

import dataclasses
from typing import Dict, List, Optional, Sequence

import torch

VOCAB = 151936
"""A production padded vocabulary (Qwen2.5); token ``VOCAB - 1`` is placed in every step."""

WIDE_TOKEN_LIMIT = 2**62
"""Token ids are int64 end to end; "wide" steps draw ids up to here to expose int32 truncation."""

INT32_ALIAS = 2**32
"""A rejected draft ``y + 2**32`` agrees with the sample in its low 32 bits."""

ACCEPTANCE = ("random", "none", "all", "after_mismatch", "int32_alias")
"""Acceptance patterns for the decode requests of a step:

* ``random``: ``n_d`` uniform in ``[0, S]``; drafts after the first mismatch are random;
* ``none`` / ``all``: ``n_d = 0`` / ``n_d = S`` for every request;
* ``after_mismatch``: ``n_d < S`` and every draft after the first mismatch *does* match its
  sample, which must not revive acceptance;
* ``int32_alias``: ``n_d < S`` and the first rejected draft equals its sample plus ``2**32``.
"""

RLI_MODES = ("eager", "graph", "async", "async_graph")


@dataclasses.dataclass
class Step:
    """One speculative step's kernel inputs, built from a request layout (see module docstring)."""

    num_spec: int
    num_decode: int
    num_prefill: int
    x: torch.Tensor  # int64 [R]: input tokens at the required rows (incl. padded slots)
    y: torch.Tensor  # int64 [R or more]: sampled tokens per required row
    rli: torch.Tensor  # int32/int64 [R]: token row of each required row
    accepted_target: List[int]  # n_d the builder constructed, per decode request

    @property
    def num_active(self) -> int:
        return self.num_decode + self.num_prefill

    @property
    def live_rows(self) -> int:
        """Required rows that belong to a request: ``D (S+1) + P``."""
        return self.num_decode * (self.num_spec + 1) + self.num_prefill

    def to(self, device) -> "Step":
        return dataclasses.replace(
            self, x=self.x.to(device), y=self.y.to(device), rli=self.rli.to(device)
        )


def _token(generator: torch.Generator, wide: bool, n: int = 1) -> torch.Tensor:
    hi = WIDE_TOKEN_LIMIT if wide else VOCAB
    return torch.randint(0, hi, (n,), generator=generator, dtype=torch.int64)


def _different(value: int, generator: torch.Generator, wide: bool) -> int:
    while True:
        t = int(_token(generator, wide))
        if t != value:
            return t


def accepted_targets(
    num_decode: int, num_spec: int, acceptance: str, generator: torch.Generator
) -> List[int]:
    """Number of accepted drafts per decode request for an ``ACCEPTANCE`` pattern."""
    if acceptance == "none":
        return [0] * num_decode
    if acceptance == "all":
        return [num_spec] * num_decode
    top = num_spec + 1 if acceptance == "random" else num_spec
    return torch.randint(0, top, (num_decode,), generator=generator).tolist()


def make_step(
    num_spec: int,
    accepted: Sequence[int],
    chunk_lengths: Sequence[int],
    *,
    acceptance: str = "random",
    rli_mode: str = "eager",
    rli_dtype: torch.dtype = torch.int32,
    wide_tokens: bool = False,
    padded_decode: Optional[int] = None,
    padded_prefill: Optional[int] = None,
    extra_samples: int = 0,
    generator: torch.Generator,
) -> Step:
    """Build one step's tensors (on CPU) from its request layout.

    Args:
        num_spec: speculative tokens per decode request (S).
        accepted: drafts to accept per decode request (``n_d``), one per decode request.
        chunk_lengths: query length of each prefill request's chunk this step; the last may be
            a chunked-prefill request.
        acceptance: how rejected drafts are built (``ACCEPTANCE``).
        rli_mode: ``RLI_MODES``; ``graph`` modes pad to ``padded_decode``/``padded_prefill``.
        rli_dtype: dtype of the required-row indices (int32 in the context, int64 in async).
        wide_tokens: draw token ids up to ``WIDE_TOKEN_LIMIT`` instead of the vocabulary.
        extra_samples: stale sampled tokens after the required rows (a longer sampler buffer).
    """
    s = num_spec
    num_decode, num_prefill = len(accepted), len(chunk_lengths)
    decode_tokens = num_decode * (s + 1)
    packed = decode_tokens + sum(chunk_lengths)

    # Required rows in the context's packed order, then graph padding (zeros).
    ends = torch.tensor(chunk_lengths, dtype=torch.int64).cumsum(0) - 1
    rli_ctx = torch.cat([torch.arange(decode_tokens, dtype=torch.int64), decode_tokens + ends])
    graphed = rli_mode in ("graph", "async_graph")
    if graphed:
        d_pad = num_decode if padded_decode is None else padded_decode
        p_pad = num_prefill if padded_prefill is None else padded_prefill
        assert d_pad >= num_decode and p_pad >= num_prefill
        slots = d_pad * (s + 1) + p_pad
        rli_ctx = torch.cat([rli_ctx, torch.zeros(slots - rli_ctx.numel(), dtype=torch.int64)])
        padded_tokens = max(packed, slots) + 3
    else:
        padded_tokens = packed

    # Async scheduling: the pending forward ran on compacted rows; map back to GPU rows.
    gpu_rows = padded_tokens
    if rli_mode in ("async", "async_graph"):
        gpu_rows = 4 * max(padded_tokens, 1)
        token_row_indices = torch.randperm(gpu_rows, generator=generator)[:padded_tokens]
        rli = token_row_indices[rli_ctx[: (num_decode + num_prefill) * (s + 1)]]
    else:
        rli = rli_ctx

    # Samples per required row; stale samples in padded slots and past them.
    live = decode_tokens + num_prefill
    y = _token(generator, wide_tokens, rli.numel() + extra_samples)
    if rli.numel() + extra_samples > live:
        y[live:] = torch.iinfo(torch.int64).min  # never a sample of a live request
    if live:
        y[int(torch.randint(0, live, (1,), generator=generator))] = VOCAB - 1

    # Input tokens at the required rows: decode bases and drafts built from the samples.
    x_live = _token(generator, wide_tokens, live)
    for d, n in enumerate(accepted):
        assert 0 <= n <= s
        base = d * (s + 1)
        for j in range(s):
            sample = int(y[base + j])
            if j < n:
                x_live[base + 1 + j] = sample
            elif j == n:
                alias = acceptance == "int32_alias"
                x_live[base + 1 + j] = (
                    sample + INT32_ALIAS if alias else _different(sample, generator, wide_tokens)
                )
            elif acceptance == "after_mismatch":
                x_live[base + 1 + j] = sample

    # Scatter them to the token rows the forward read, then gather as the controller does.
    input_ids = _token(generator, wide_tokens, gpu_rows)
    input_ids[rli[:live]] = x_live
    x = input_ids[rli]
    return Step(
        num_spec=s,
        num_decode=num_decode,
        num_prefill=num_prefill,
        x=x,
        y=y,
        rli=rli.to(rli_dtype),
        accepted_target=list(accepted),
    )


def reference(step: Step) -> Dict[str, torch.Tensor]:
    """The bookkeeping contract (module docstring), computed from ``x``, ``y`` and ``rli`` only.

    Returns int64 CPU tensors: ``accepted_tokens [A, S]``, ``accepted_counts [A]``,
    ``sampled [A]``, ``last_accepted_seq [A]``, ``last_one [A]``, and the bool verification
    ``mask`` over ``x``.
    """
    s, num_decode = step.num_spec, step.num_decode
    x, y, rli = step.x.tolist(), step.y.tolist(), step.rli.tolist()
    accepted_tokens, counts, sampled, last_seq, last_one = [], [], [], [], []
    mask = [False] * len(x)
    for d in range(num_decode):
        base = d * (s + 1)
        n = 0
        while n < s and x[base + 1 + n] == y[base + n]:
            n += 1
        accepted_tokens.append(x[base + 1 : base + 1 + n] + [-1] * (s - n))
        counts.append(n)
        sampled.append(y[base + n])
        last_seq.append(rli[base + n])
        last_one.append(base + n)
        for i in range(n + 1):
            mask[base + i] = True
    for p in range(step.num_prefill):
        row = num_decode * (s + 1) + p
        accepted_tokens.append([-1] * s)
        counts.append(0)
        sampled.append(y[row])
        last_seq.append(rli[row])
        last_one.append(row)
        mask[row] = True
    as_long = lambda v: torch.tensor(v, dtype=torch.int64)  # noqa: E731
    return {
        "accepted_tokens": as_long(accepted_tokens).reshape(len(counts), s),
        "accepted_counts": as_long(counts),
        "sampled": as_long(sampled),
        "last_accepted_seq": as_long(last_seq),
        "last_one": as_long(last_one),
        "mask": torch.tensor(mask, dtype=torch.bool),
    }
