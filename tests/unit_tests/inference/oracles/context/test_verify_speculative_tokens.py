# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_verify_speculative_tokens_kernel``
(megatron/core/inference/text_generation_controllers/mtp_utils_triton.py, launcher
``verify_speculative_tokens``), the step before ``_prepare_next_forward_pass_kernel``.

**Contract** (from the speculative-decoding semantics; layout in ``speculative_steps``). Given
the input tokens ``x`` and the sampled tokens ``y`` at the required rows of a step with ``D``
decode and ``P`` prefill requests and ``S`` drafts per decode request, return

* ``last_one_indices`` int64 ``[A]``: for decode ``d``, ``d(S+1) + n_d`` where ``n_d`` is the
  length of the longest draft prefix that equals the model's samples at the positions before
  them; for prefill ``p``, ``D(S+1) + p``;
* ``accepted_tokens_mask`` bool, the shape of ``x`` (squeezed): true at each decode request's
  base token and its accepted drafts and at every prefill row, false everywhere else
  (rejected drafts, drafts that match after a mismatch, graph-padding slots);
* ``x`` itself (a ``[1, N]`` input squeezed to ``[N]``).

Inputs are int64 token tensors of at least ``D(S+1) + P`` rows; rows past that (graph padding,
stale samples) must not influence the result. Degenerate: ``A = 0`` gives an all-false mask
and an empty index. Numerics: integer comparisons, **exact**.

**Oracle.** ``speculative_steps.reference``, a per-request Python loop (longest matching
prefix); it shares nothing with the kernel's lane-parallel prefix test.

**Inputs from the kernel's structure.** Constexpr ``STRIDE = S + 1`` and
``BLOCK_SIZE = next_pow2(S + 1)`` (``mtp_utils_triton.py:176-177``, ``:240``): S in 1..9 gives
lane widths 2/4/8/16 with masked lanes at S = 2, 4, 5, 6, 8, 9. Runtime ints
``num_decode_requests`` and ``decode_len`` get Triton's ``== 1`` / ``% 16`` specializations:
decode counts 0/1/15/16/17/31/32/33/512 and prefill counts 0/1/15/16/17, pairwise with every
acceptance pattern (a match after a mismatch must stay rejected; an int32-aliased rejection
must not match), vocabulary and wide (up to 2**62) token ids, eager, graph-padded and async
required rows, and 1-D and ``[1, N]`` inputs.

**Memory.** The outputs are allocated by the launcher, so no guard bands; the inputs are
checked unchanged, and the mask past the live rows must be false (a stray write shows there).

**Hardware.** H100. No open PR changes this kernel.
"""

import pytest
import torch

from megatron.core.inference.text_generation_controllers.mtp_utils_triton import (
    verify_speculative_tokens,
)
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.context import speculative_steps as sps

DEVICE = "cuda"

CASES = ou.pairwise_cases(
    {
        "num_spec": tuple(range(1, 10)),
        "num_decode": ou.boundary_sizes(16, production=(512,)),
        "num_prefill": ou.boundary_sizes(16, multiples=()),
        "acceptance": sps.ACCEPTANCE,
        "rli_mode": sps.RLI_MODES,
        "wide_tokens": (False, True),
        "input_rank": (1, 2),
    }
)


def _case_id(c):
    tokens = "wide" if c["wide_tokens"] else "vocab"
    return (
        f"S{c['num_spec']}-D{c['num_decode']}-P{c['num_prefill']}-{c['acceptance']}"
        f"-{c['rli_mode']}-{tokens}-{c['input_rank']}d"
    )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", [pytest.param(c, id=_case_id(c)) for c in CASES])
def test_verify_matches_longest_accepted_prefix(case):
    g = ou.make_generator()
    s, d, p = case["num_spec"], case["num_decode"], case["num_prefill"]
    accepted = sps.accepted_targets(d, s, case["acceptance"], g)
    chunks = torch.randint(1, 40, (p,), generator=g).tolist()
    step = sps.make_step(
        s,
        accepted,
        chunks,
        acceptance=case["acceptance"],
        rli_mode=case["rli_mode"],
        rli_dtype=torch.int64,
        wide_tokens=case["wide_tokens"],
        padded_decode=d + 3,
        padded_prefill=p + 2,
        extra_samples=5,
        generator=g,
    )
    expected = sps.reference(step)
    assert expected["accepted_counts"][:d].tolist() == accepted
    step = step.to(DEVICE)
    x_in = step.x if case["input_rank"] == 1 else step.x.unsqueeze(0)
    x_before, y_before = x_in.clone(), step.y.clone()

    last_one, mask, x_out = verify_speculative_tokens(
        input_tokens=x_in,
        output_tokens=step.y,
        num_decode_requests=d,
        num_prefill_requests=p,
        num_speculative_tokens=s,
    )

    ou.assert_exact(last_one, expected["last_one"], what="last_one_indices")
    assert last_one.dtype == torch.int64
    ou.assert_exact(mask, expected["mask"], what="accepted_tokens_mask")
    assert mask.dtype == torch.bool
    ou.assert_exact(x_out, step.x, what="returned input tokens")
    ou.assert_exact(x_in, x_before, what="input tokens are read-only")
    ou.assert_exact(step.y, y_before, what="output tokens are read-only")
