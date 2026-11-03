# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_mamba_state_selective_copy_kernel``
(megatron/core/inference/text_generation_controllers/mtp_utils_triton.py, launcher
``mamba_state_selective_copy``), the Mamba half of the speculative rewind.

**Contract** (from the speculative rewind's semantics, ``TextGenerationController.
_rewind_kv_cache``). A hybrid model's decode step with ``S`` drafts runs each Mamba layer over
the base token and the drafts, keeping the state after each of those ``S + 1`` tokens in
``intermediate[l, slot, j]`` (``j`` = tokens consumed after the base). Once verification has
accepted ``n`` drafts, the request's live state must be the state after the base and those
``n`` drafts: for every active request ``i`` that is decoding (``prefill_status[i] == 0``) and
every layer ``l``,

    current[l, state_idx[i]] = intermediate[l, state_idx[i], accepted_counts[i]]

bit for bit (a copy: NaN payloads, signed zeros, denormals kept). Prefill requests
(``prefill_status[i] == 1``) and every slot no decode request owns are untouched, and
``intermediate`` is read-only. Inputs: ``intermediate [L, M, S+1, *state]``, ``current
[L, M, *state]``, same dtype, trailing state dims contiguous (leading strides arbitrary);
``prefill_status`` and ``state_idx`` int32 ``[N]`` (the active slice, paused requests excluded),
``accepted_counts`` int64 ``[N]`` in ``[0, S]`` (the prepare kernel's counts). Preconditions
(unchecked): unique slots, ``accepted <= S``. Degenerate: ``N = 0`` does nothing. Tolerance
class: copy, **exact**.

**Oracle.** Torch advanced indexing on a clone (``expected[:, slot] = intermediate[:, slot,
accepted]``), with no flattening or stride arithmetic. A chained test also feeds the counts
from the verify and prepare kernels for a speculative step and checks each request's state
against the accepted-prefix count from ``speculative_steps.reference``.

**Inputs from the kernel's structure.** Grid ``(N, L, cdiv(state_size, 1024))`` with
``BLOCK_SIZE = 1024`` (``mtp_utils_triton.py:445-447``): state sizes 1, 1023/1024/1025,
2047/2048/2049 and the production conv (``conv_dim x d_conv`` = 2560 x 4) and SSM
(``nheads x headdim x d_state`` = 16 x 64 x 128) states of a TP-split Nemotron-H-like layer;
fp32, bf16 and fp16 states; leading dims contiguous or strided (every other slot); S in
1/3/7; requests 0/1/many, all decode, all prefill and mixed; accepted counts 0..S. Random bit
patterns as data. int32 limits: the source offset ``slot * (S+1) * state`` crosses 2**31 for
the last of 1025 slots of a 2**20-element state (``large``).

**Memory.** ``current`` is guard-banded with a whole 1024-element tile; ``intermediate`` is
checked unchanged. One small case runs under compute-sanitizer.

**Hardware.** H100. No open PR changes this kernel.
"""

import pytest
import torch

from megatron.core.inference.text_generation_controllers.mtp_utils_triton import (
    mamba_state_selective_copy,
    prepare_next_forward_pass,
    verify_speculative_tokens,
)
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.context import speculative_steps as sps

DEVICE = "cuda"

BLOCK = 1024
"""Elements per program along the flattened state (``mtp_utils_triton.py:445``)."""

STATE_SHAPES = [(n,) for n in ou.boundary_sizes(BLOCK, degenerate=False)] + [
    (1,),
    (2560, 4),  # conv state: conv_dim x d_conv
    (16, 64, 128),  # SSM state: nheads x headdim x d_state
]

CASES = ou.pairwise_cases(
    {
        "state_shape": STATE_SHAPES,
        "dtype": (torch.float32, torch.bfloat16, torch.float16),
        "num_spec": (1, 3, 7),
        "requests": ("mixed", "decode", "prefill", "one"),
        "layout": ("contiguous", "strided"),
    }
)


def _case_id(c):
    shape = "x".join(map(str, c["state_shape"]))
    dtype = str(c["dtype"]).replace("torch.", "")
    return f"{shape}-{dtype}-S{c['num_spec']}-{c['requests']}-{c['layout']}"


def _random_bits(shape, dtype, g):
    """Arbitrary bit patterns (NaNs, infinities, denormals, -0.0 included)."""
    ints = {4: torch.int32, 2: torch.int16}[dtype.itemsize]
    info = torch.iinfo(ints)
    raw = torch.randint(info.min, info.max, tuple(shape), generator=g, dtype=torch.int64)
    return raw.to(ints).view(dtype)


def _states(num_layers, num_slots, num_spec, state_shape, dtype, layout, g):
    """``(intermediate, current_buffer, slot_step)``: the strided layout uses every other slot
    of larger buffers, so the views passed to the kernel have non-default leading strides."""
    step = 2 if layout == "strided" else 1
    inter_full = _random_bits(
        (num_layers, num_slots * step, num_spec + 1, *state_shape), dtype, g
    ).to(DEVICE)
    cur_shape = (num_layers, num_slots * step, *state_shape)
    tile = BLOCK * dtype.itemsize
    cur_buf = ou.guarded_like(_random_bits(cur_shape, dtype, g).to(DEVICE), guard_bytes=tile)
    return inter_full[:, ::step], cur_buf, step


def _reference_(expected, inter, prefill_status, state_idx, accepted):
    """In place on ``expected`` (a view of a clone of the current states)."""
    for i in range(prefill_status.numel()):
        if int(prefill_status[i]) == 0:
            slot, a = int(state_idx[i]), int(accepted[i])
            expected[:, slot] = inter[:, slot, a]


def _requests(kind, num_slots, num_spec, g):
    n = {"mixed": num_slots - 2, "decode": num_slots - 3, "prefill": 3, "one": 1}[kind]
    slots = torch.randperm(num_slots, generator=g)[:n].to(torch.int32)
    if kind == "prefill":
        status = torch.ones(n, dtype=torch.int32)
    elif kind == "mixed":
        status = torch.randint(0, 2, (n,), generator=g, dtype=torch.int32)
        status[:2] = torch.tensor([0, 1], dtype=torch.int32)
    else:
        status = torch.zeros(n, dtype=torch.int32)
    accepted = torch.randint(0, num_spec + 1, (n,), generator=g)
    accepted[: min(n, 2)] = torch.tensor([0, num_spec])[: min(n, 2)]
    # Prefill rows carry a count too; the copy must ignore it.
    return status, slots, accepted


def _run_and_check(inter, cur_buf, step, status, slots, accepted, what):
    """Run the kernel on the ``step``-strided view; check the whole buffer and its guards."""
    inter_before = inter.clone()
    expected_full = cur_buf.tensor.clone()
    _reference_(expected_full[:, ::step], inter, status, slots, accepted)
    status, slots, accepted = (t.to(DEVICE) for t in (status, slots, accepted))
    mamba_state_selective_copy(
        intermediate_states=inter,
        current_states=cur_buf.tensor[:, ::step],
        prefill_status=status,
        state_idx=slots,
        accepted_counts=accepted,
        num_layers=inter.shape[0],
    )
    ou.check_guards(current=cur_buf)
    ou.assert_exact(cur_buf.tensor, expected_full, what=what)
    ou.assert_exact(inter, inter_before, what="intermediate states are read-only")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=_case_id(c)) for c in CASES]
    + [
        pytest.param(
            dict(
                state_shape=(1025,),
                dtype=torch.bfloat16,
                num_spec=3,
                requests="mixed",
                layout="strided",
            ),
            id="sanitizer-1025-bf16-S3-mixed-strided",
            marks=pytest.mark.sanitizer,
        )
    ],
)
def test_copy_selects_the_accepted_state(case):
    g = ou.make_generator()
    num_layers, num_slots = 3, 9
    inter, cur_buf, step = _states(
        num_layers,
        num_slots,
        case["num_spec"],
        case["state_shape"],
        case["dtype"],
        case["layout"],
        g,
    )
    status, slots, accepted = _requests(case["requests"], num_slots, case["num_spec"], g)
    _run_and_check(inter, cur_buf, step, status, slots, accepted, what=_case_id(case))


@ou.requires_sm(ou.HOPPER)
def test_copy_without_requests_writes_nothing():
    g = ou.make_generator()
    inter, cur_buf, step = _states(2, 4, 2, (1025,), torch.float32, "contiguous", g)
    empty = torch.empty(0, dtype=torch.int32)
    _run_and_check(inter, cur_buf, step, empty, empty, empty.long(), what="N = 0")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("num_spec", [1, 3])
def test_rewind_keeps_the_state_after_the_last_accepted_draft(num_spec):
    """verify -> prepare -> Mamba copy for one speculative step of a hybrid model: each decode
    request's live state is its state after the base token and its accepted drafts; prefill
    requests (one a chunk) and the paused requests' slots keep theirs."""
    g = ou.make_generator()
    accepted = [num_spec, 0, 1, None, num_spec, None]
    accepted = [
        int(torch.randint(0, num_spec + 1, (1,), generator=g)) if a is None else a for a in accepted
    ]
    step = sps.make_step(
        num_spec,
        accepted,
        [6, 1, 17],
        rli_mode="async_graph",
        rli_dtype=torch.int64,
        padded_decode=8,
        padded_prefill=4,
        generator=g,
    )
    expected_counts = sps.reference(step)["accepted_counts"]
    assert expected_counts[: step.num_decode].tolist() == accepted
    step = step.to(DEVICE)

    # The context: 3 paused requests, then the active ones, each with its own Mamba slot.
    num_paused, num_slots = 3, 16
    slot_of_row = torch.randperm(num_slots, generator=g)[: num_paused + step.num_active]
    active_slots = slot_of_row[num_paused:].to(torch.int32)
    status = torch.tensor([0] * step.num_decode + [1] * step.num_prefill, dtype=torch.int32)

    last_one, mask, x = verify_speculative_tokens(
        step.x, step.y, step.num_decode, step.num_prefill, num_spec
    )
    max_requests = num_paused + step.num_active + 2
    counts = torch.full((max_requests,), -7, dtype=torch.int64, device=DEVICE)
    prepare_next_forward_pass(
        num_decode_requests=step.num_decode,
        output_tokens=step.y,
        required_logit_indices=step.rli,
        last_one_indices=last_one,
        accepted_tokens_mask=mask,
        input_tokens=x,
        sampled_tokens_buf=torch.empty(max_requests, dtype=torch.int64, device=DEVICE),
        last_accepted_seq_buf=torch.empty(max_requests, dtype=torch.int64, device=DEVICE),
        accepted_tokens_per_request=torch.empty(
            (max_requests, num_spec), dtype=torch.int64, device=DEVICE
        ),
        accepted_token_counts=counts,
        num_speculative_tokens=num_spec,
    )

    inter, cur_buf, _ = _states(
        2, num_slots, num_spec, (16, 8, 16), torch.bfloat16, "contiguous", g
    )
    cur = cur_buf.tensor
    expected = cur.clone()
    for i, slot in enumerate(active_slots.tolist()):
        if i < step.num_decode:
            expected[:, slot] = inter[:, slot, int(expected_counts[i])]
    mamba_state_selective_copy(
        intermediate_states=inter,
        current_states=cur,
        prefill_status=status.to(DEVICE),
        state_idx=active_slots.to(DEVICE),
        accepted_counts=counts[: step.num_active],
        num_layers=2,
    )
    ou.check_guards(current=cur_buf)
    ou.assert_exact(cur, expected, what="live Mamba states after the rewind")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("num_slots", ou.int32_overflow_params(2 * 2**20))
def test_copy_offsets_past_int32(num_slots):
    """Source offsets ``slot * (S+1) * state + accepted * state + i`` up to and past 2**31:
    the last slot's last element is at 2**31 - 1 with 1024 slots and past it with 1025."""
    num_spec, state = 1, 2**20
    ou.require_free_memory(3 * num_slots * state * 2 + 2**30)
    inter = torch.empty((1, num_slots, num_spec + 1, state), dtype=torch.bfloat16, device=DEVICE)
    cur = torch.zeros((1, num_slots, state), dtype=torch.bfloat16, device=DEVICE)
    last = num_slots - 1
    g = ou.make_generator(device=DEVICE)
    inter[0, last] = torch.randn((num_spec + 1, state), generator=g, device=DEVICE).bfloat16()
    inter[0, 0] = torch.randn((num_spec + 1, state), generator=g, device=DEVICE).bfloat16()
    status = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    slots = torch.tensor([last, 0], dtype=torch.int32, device=DEVICE)
    accepted = torch.tensor([num_spec, 0], dtype=torch.int64, device=DEVICE)
    mamba_state_selective_copy(inter, cur, status, slots, accepted, num_layers=1)
    ou.assert_exact(cur[0, last], inter[0, last, num_spec], what=f"slot {last}")
    ou.assert_exact(cur[0, 0], inter[0, 0, 0], what="slot 0")
    written = int(torch.count_nonzero(cur[0, 1:last]))
    assert written == 0, f"{written} elements of slots 1..{last - 1} were written"
