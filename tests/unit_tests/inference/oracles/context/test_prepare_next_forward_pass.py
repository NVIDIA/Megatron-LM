# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_prepare_next_forward_pass_kernel``
(megatron/core/inference/text_generation_controllers/mtp_utils_triton.py, launcher
``prepare_next_forward_pass``).

**Contract** (from the dynamic context's bookkeeping, before reading the kernel). After the
controller samples a speculative step and verifies the drafts, this kernel writes, for each of
the ``A = D + P`` active requests, the state the next forward pass starts from (layout and
notation in ``speculative_steps``):

* ``sampled_tokens_buf[a]``: the token the model sampled after the request's last accepted
  token (decode ``d``: ``y[d(S+1) + n_d]``; prefill ``p``: ``y[D(S+1) + p]``). It becomes
  output text and the next step's base token.
* ``last_accepted_seq_buf[a]``: the token row of that last accepted token, ``rli[...]`` at the
  same required row. The next MTP draft reads its hidden state and its position there.
* ``accepted_tokens_per_request[a, :]``: the accepted drafts, ``-1``-padded to ``S``; all
  ``-1`` for prefill rows (the engine appends every non-negative entry to the request's
  output, so a stale entry on a prefill row would emit another request's drafts).
* ``accepted_token_counts[a]``: ``n_d``, and 0 for prefill rows. The KV and Mamba rewinds
  read it.

Inputs: ``output_tokens`` (``y``) int64, ``required_logit_indices`` int32 (context) or int64
(async scheduling), the verification ``last_one_indices`` int64 ``[A]`` and bool mask, the
gathered ``input_tokens`` (``x``) int64. The output buffers are the controller's
``[max_requests, S]`` / ``[max_requests]`` int64 buffers; they persist across steps and are
not cleared between them. Preconditions (not checked by the kernel): ``num_decode <= A``, the
mask is prefix-shaped per decode request and consistent with ``last_one`` (``verify``
guarantees both). Rows ``>= A`` of every buffer are not written. Degenerate: ``A = 0`` writes
nothing. Numerics: integer gathers and copies, **exact**.

**Oracle.** ``speculative_steps.reference``: a plain Python loop over requests that finds each
decode request's longest prefix of drafts matching the model's own samples and gathers from
``x``, ``y`` and ``rli`` by that count. It does not look at the mask, the per-lane layout or
``last_one``. The kernel runs both on the verify kernel's outputs (the production chain) and on
the reference's mask (so a verify bug cannot hide a prepare bug, or excuse one).

**Inputs from the kernel's structure.** One program per active request; constexpr
``STRIDE = S + 1``, ``NUM_SPEC_TOKENS = S`` and ``SPEC_BLOCK_SIZE = next_pow2(S)`` (``:275-277``,
``:338``), so S in 1..9 compiles every lane width 1/2/4/8/16 with masked lanes at S = 3, 5, 6, 7
and 9. Runtime ints ``num_decode_requests`` and ``accepted_tokens_out_stride`` get Triton's
``== 1`` and ``% 16`` specializations: decode counts 0/1/15/16/17/31/32/33 and a production 512,
contiguous (stride S) and strided (stride S + 3) output rows. Requests: decode and prefill
splits including decode-only, prefill-only and empty; chunked prefill (the last prefill
request's chunk); every acceptance pattern of ``speculative_steps.ACCEPTANCE`` (none, all,
partial, a match after a mismatch, an int32-aliased rejection); token ids in the vocabulary
(with ``VOCAB - 1``) or up to 2**62; required rows eager, CUDA-graph padded, async-composed and
both, with int32 and int64 indices; stale samples past the live rows. Factors combined
pairwise. Across steps: paused, evicted, finished and newly admitted requests change ``A`` and
turn decode rows into prefill rows while the buffers persist, with the forward padded to fixed
captured dimensions. CUDA-graph capture of verify + prepare, replayed with new data (the counts
are host ints baked into the grid, so they cannot change across replays; production launches
the kernel eagerly).

**Memory.** Every output buffer is guard-banded and starts poisoned (int64 min); rows ``>= A``
and the strided buffer's spare columns must stay poisoned, or keep the previous step's
values. Inputs are checked unchanged. One small case runs under compute-sanitizer.

**Hardware.** H100 (plain Triton, no architecture-specific path).

**Open PRs.** None changes this kernel. #8565 (s2) drops the controller's per-step reset of
these buffers, so the across-steps test is the contract it relies on; it passes on main
(#8426 added the prefill-row writes).
"""

import pytest
import torch

from megatron.core.inference.text_generation_controllers.mtp_utils_triton import (
    prepare_next_forward_pass,
    verify_speculative_tokens,
)
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.context import speculative_steps as sps

DEVICE = "cuda"

SPEC_TOKENS = tuple(range(1, 10))
"""S: next_pow2(S) in {1, 2, 4, 8, 16} (prepare's lanes), next_pow2(S + 1) (verify's)."""

DECODE_COUNTS = ou.boundary_sizes(16, production=(512,))
"""Around Triton's ``% 16`` runtime-int specialization; 0 and 1 (the ``== 1`` one)."""

PREFILL_COUNTS = ou.boundary_sizes(16, multiples=())

SPARE_ROWS = 5
"""Buffer rows past the active ones; they belong to no request and must stay untouched."""

CASES = ou.pairwise_cases(
    {
        "num_spec": SPEC_TOKENS,
        "num_decode": DECODE_COUNTS,
        "num_prefill": PREFILL_COUNTS,
        "acceptance": sps.ACCEPTANCE,
        "rli_mode": sps.RLI_MODES,
        "rli_dtype": (torch.int32, torch.int64),
        "layout": ("contiguous", "strided"),
        "wide_tokens": (False, True),
    }
)

OUTPUTS = ("accepted_tokens", "accepted_counts", "sampled", "last_accepted_seq")


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _case_id(c):
    dtype = str(c["rli_dtype"]).replace("torch.", "")
    tokens = "wide" if c["wide_tokens"] else "vocab"
    return (
        f"S{c['num_spec']}-D{c['num_decode']}-P{c['num_prefill']}-{c['acceptance']}"
        f"-{c['rli_mode']}-{dtype}-{c['layout']}-{tokens}"
    )


SANITIZER_CASE = dict(
    num_spec=3,
    num_decode=5,
    num_prefill=2,
    acceptance="random",
    rli_mode="async_graph",
    rli_dtype=torch.int64,
    layout="strided",
    wide_tokens=True,
)

CASE_PARAMS = [pytest.param(c, id=_case_id(c)) for c in CASES] + [
    pytest.param(
        SANITIZER_CASE, id="sanitizer-" + _case_id(SANITIZER_CASE), marks=pytest.mark.sanitizer
    )
]


class PrepareBuffers:
    """The controller's persistent output buffers, guard-banded and poisoned."""

    def __init__(self, max_requests: int, num_spec: int, layout: str):
        width = num_spec + 3 if layout == "strided" else num_spec
        self.num_spec = num_spec
        self.accepted_backing = ou.guarded_empty((max_requests, width), torch.int64, DEVICE)
        self.counts = ou.guarded_empty((max_requests,), torch.int64, DEVICE)
        self.sampled = ou.guarded_empty((max_requests,), torch.int64, DEVICE)
        self.last_seq = ou.guarded_empty((max_requests,), torch.int64, DEVICE)

    @property
    def tensors(self):
        return {
            "accepted_tokens": self.accepted_backing.tensor[:, : self.num_spec],
            "accepted_counts": self.counts.tensor,
            "sampled": self.sampled.tensor,
            "last_accepted_seq": self.last_seq.tensor,
        }

    def snapshot(self):
        return {k: v.clone() for k, v in self.tensors.items()}

    def launch(self, step, last_one, mask, x):
        t = self.tensors
        prepare_next_forward_pass(
            num_decode_requests=step.num_decode,
            output_tokens=step.y,
            required_logit_indices=step.rli,
            last_one_indices=last_one,
            accepted_tokens_mask=mask,
            input_tokens=x,
            sampled_tokens_buf=t["sampled"],
            last_accepted_seq_buf=t["last_accepted_seq"],
            accepted_tokens_per_request=t["accepted_tokens"],
            accepted_token_counts=t["accepted_counts"],
            num_speculative_tokens=step.num_spec,
        )

    def check(self, expected_state, what):
        """Compare whole buffers with the expected state; guards and spare columns intact."""
        ou.check_guards(
            accepted_tokens=self.accepted_backing,
            accepted_counts=self.counts,
            sampled=self.sampled,
            last_accepted_seq=self.last_seq,
        )
        for name, tensor in self.tensors.items():
            ou.assert_exact(tensor, expected_state[name], what=f"{what}: {name}")
        ou.assert_poisoned(
            self.accepted_backing.tensor[:, self.num_spec :],
            what=f"{what}: accepted_tokens columns past S (row stride {self.num_spec + 3})",
        )


def _advance(state, expected, num_active):
    """The buffers after a step: active rows from the contract, every other row unchanged."""
    out = {k: v.clone() for k, v in state.items()}
    for name in OUTPUTS:
        out[name][:num_active] = expected[name].to(out[name].device)
    return out


def _upstream(step, expected, source):
    """Verification outputs fed to prepare: the verify kernel's, or the contract's."""
    if source == "verify_kernel":
        last_one, mask, x = verify_speculative_tokens(
            input_tokens=step.x,
            output_tokens=step.y,
            num_decode_requests=step.num_decode,
            num_prefill_requests=step.num_prefill,
            num_speculative_tokens=step.num_spec,
        )
        return last_one, mask, x
    return expected["last_one"].to(DEVICE), expected["mask"].to(DEVICE), step.x


def _build(case, g, **kw):
    s, d, p = case["num_spec"], case["num_decode"], case["num_prefill"]
    accepted = sps.accepted_targets(d, s, case["acceptance"], g)
    chunks = torch.randint(1, 40, (p,), generator=g).tolist()
    step = sps.make_step(
        s,
        accepted,
        chunks,
        acceptance=case["acceptance"],
        rli_mode=case["rli_mode"],
        rli_dtype=case["rli_dtype"],
        wide_tokens=case["wide_tokens"],
        generator=g,
        **kw,
    )
    expected = sps.reference(step)
    # The step builder made the acceptance it meant to (keeps the oracle from being vacuous).
    assert expected["accepted_counts"][:d].tolist() == accepted
    return step, expected


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", CASE_PARAMS)
@pytest.mark.parametrize("source", ["verify_kernel", "reference"])
def test_prepare_matches_bookkeeping_contract(case, source):
    """One step: active rows equal the contract, all other buffer memory untouched."""
    g = ou.make_generator()
    d, p = case["num_decode"], case["num_prefill"]
    step, expected = _build(case, g, padded_decode=d + 3, padded_prefill=p + 2, extra_samples=7)
    step = step.to(DEVICE)
    last_one, mask, x = _upstream(step, expected, source)
    inputs = {"y": step.y, "rli": step.rli, "last_one": last_one, "mask": mask, "x": x}
    before = {k: v.clone() for k, v in inputs.items()}

    bufs = PrepareBuffers(step.num_active + SPARE_ROWS, step.num_spec, case["layout"])
    poisoned = bufs.snapshot()
    bufs.launch(step, last_one, mask, x)
    _sync()

    bufs.check(_advance(poisoned, expected, step.num_active), what=f"upstream {source}")
    for name, tensor in inputs.items():
        ou.assert_exact(tensor, before[name], what=f"input {name} is read-only")


# Five steps of one server, S fixed. Rows are active indices (paused requests excluded);
# the forward is padded to the captured dimensions (8 decode, 4 prefill) every step.
STEPS = (
    # 5 decode requests (one accepts every draft) and 3 prefill requests, the last a chunk.
    dict(accepted=[None, 0, "all", 1, None], chunks=[7, 1, 13]),
    # 2 decode requests finished and 1 was evicted; the 3 prefills now decode; 2 admitted.
    # Rows 3..4 held accepted drafts and are prefill rows now; rows 6..7 keep step 0's values.
    dict(accepted=[None, "all", 0], chunks=[5, 11, 2]),
    # Every decode request paused: a prefill-only step on rows that held drafts.
    dict(accepted=[], chunks=[3, 9]),
    # The paused requests resume, decode-only and wider than any step before.
    dict(accepted=["all", None, 0, "all", None, 1, None, "all"], chunks=[]),
    # Everything paused: no active request.
    dict(accepted=[], chunks=[]),
)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("num_spec", [1, 3, 4])
@pytest.mark.parametrize("rli_mode", ["graph", "async_graph"])
def test_prepare_across_steps_rewrites_active_rows_only(num_spec, rli_mode):
    """Persistent buffers over steps whose request set changes: every active row is rewritten
    (a prefill row never keeps an earlier decode request's drafts), other rows keep their
    values."""
    g = ou.make_generator()
    bufs = PrepareBuffers(8 + SPARE_ROWS, num_spec, "strided")
    state = bufs.snapshot()
    for k, spec in enumerate(STEPS):
        accepted = [
            (
                num_spec
                if a == "all"
                else int(torch.randint(0, num_spec + 1, (1,), generator=g)) if a is None else a
            )
            for a in spec["accepted"]
        ]
        step = sps.make_step(
            num_spec,
            accepted,
            spec["chunks"],
            rli_mode=rli_mode,
            rli_dtype=torch.int64 if rli_mode.startswith("async") else torch.int32,
            padded_decode=8,
            padded_prefill=4,
            generator=g,
        )
        expected = sps.reference(step)
        assert expected["accepted_counts"][: step.num_decode].tolist() == accepted
        step = step.to(DEVICE)
        last_one, mask, x = _upstream(step, expected, "verify_kernel")
        bufs.launch(step, last_one, mask, x)
        state = _advance(state, expected, step.num_active)
        bufs.check(state, what=f"step {k} (D={step.num_decode}, P={step.num_prefill})")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("num_spec", [1, 3])
def test_prepare_replays_under_cuda_graph(num_spec):
    """verify + prepare captured once, replayed with new tokens, acceptance and required rows
    of the same counts: the kernel reads everything from device memory at replay."""
    g = ou.make_generator()
    num_decode, chunks = 6, [4, 9, 1]

    def new_step():
        accepted = sps.accepted_targets(num_decode, num_spec, "random", g)
        step = sps.make_step(
            num_spec,
            accepted,
            chunks,
            rli_mode="async_graph",
            rli_dtype=torch.int64,
            padded_decode=8,
            padded_prefill=4,
            generator=g,
        )
        return step, sps.reference(step)

    step0, _ = new_step()
    static = step0.to(DEVICE)
    bufs = PrepareBuffers(16, num_spec, "contiguous")

    def run():
        last_one, mask, x = verify_speculative_tokens(
            static.x, static.y, num_decode, len(chunks), num_spec
        )
        bufs.launch(static, last_one, mask, x)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()  # compile and warm up outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    state = bufs.snapshot()
    for replay in range(3):
        step, expected = new_step()
        for name in ("x", "y", "rli"):
            getattr(static, name).copy_(getattr(step, name))
        graph.replay()
        _sync()
        state = _advance(state, expected, step.num_active)
        bufs.check(state, what=f"replay {replay}")


@ou.requires_sm(ou.HOPPER)
def test_prepare_without_active_requests_writes_nothing():
    """A = 0 (every request paused or finished): no buffer element changes."""
    g = ou.make_generator()
    step = sps.make_step(2, [], [], generator=g).to(DEVICE)
    bufs = PrepareBuffers(4, 2, "strided")
    state = bufs.snapshot()
    last_one = torch.empty(0, dtype=torch.int64, device=DEVICE)
    mask = torch.empty(0, dtype=torch.bool, device=DEVICE)
    bufs.launch(step, last_one, mask, step.x)
    bufs.check(state, what="A = 0")
