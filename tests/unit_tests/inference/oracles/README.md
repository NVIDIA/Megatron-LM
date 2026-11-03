# Inference kernel oracle tests

Tests that check each inference kernel against an **independent oracle**: a reference that
does not share the kernel's algorithm, written from the kernel's contract and not from its
code. The shared helpers live in [`oracle_utils.py`](oracle_utils.py); this file sets the
conventions every test here follows.

## Layout

```
tests/unit_tests/inference/oracles/
├── oracle_utils.py        shared helpers (references, tolerance policy, sizes, memory, hardware, statistics)
├── conftest.py            markers and hardware gating for everything below this directory
├── test_oracle_utils.py   CPU tests of the helpers themselves
└── <family>/              one package per kernel family, e.g. moe_routing/
    ├── __init__.py        empty; makes module names unique across families
    └── test_<kernel>.py   one module per kernel
```

A family package plugs in by existing: add the directory with an empty `__init__.py` and import
the helpers with `from tests.unit_tests.inference.oracles import oracle_utils as ou`. The
`conftest.py` here applies to it automatically. CI picks the modules up through the
`tests/unit_tests/inference/**/*.py` buckets on H100 and, for modules that carry
`launch_on_gb200`, on GB200.

## One module per kernel

Each `test_<kernel>.py` has five parts, in this order.

1. **Contract** (the module docstring), written before reading the kernel:
   what it computes (math), its inputs (shape, dtype, layout, alignment), its outputs and
   in-place effects, the preconditions on the caller (and whether the kernel checks them),
   what happens to padding and rows past the live count, degenerate inputs, the numerics
   (accumulation dtype, rounding), and the **tolerance class** (below). List the compiled
   variants and dispatch paths the module covers.
2. **Reference**: plain PyTorch in fp64 through `ou.fp64_reference(fn, *inputs)`, which hands
   `fn` exact fp64 copies of the kernel's rounded inputs and rejects a reference that rounds.
   Where no reference fits, use metamorphic properties (batch, permutation and padding
   invariance; eager against CUDA-graph replay), `torch.distributed` for collectives, or
   statistics for samplers.
3. **Inputs derived from the kernel's structure**: every variant and dispatch branch;
   `ou.size_params(BLOCK, ...)` for B−1/B/B+1 around every tile, chunk and window size plus 0/1
   and production sizes (take B from the kernel's constants); `ou.int32_overflow_params(row)`
   for 2³¹ offsets; special values (zeros, ±inf, NaN, ties, denormals) where the contract
   defines them; `ou.pairwise_cases({...})` to combine factors.
4. **Tolerance class**, one per output, never a hand-picked `atol`:

   | Output | Assertion |
   |---|---|
   | integer, index, routing, permutation, copy | `ou.assert_exact`, `ou.assert_is_permutation`, `ou.assert_rows_equal_unordered` (order unspecified) |
   | quantized (MXFP8, NVFP4) | `ou.assert_quantized_bit_exact` against a spec reference: codes and scales separately |
   | floating point | `ou.assert_close_to_reference(out, ref_fp64, trusted)`: the kernel's error against the fp64 reference is at most `factor` (default 2) times the error of a trusted implementation in the same dtype (e.g. torch's bf16 op), measured in the same test, plus a one-element ulp floor |
   | exact-by-construction floating point | `ou.dyadic_tensor` inputs, then `ou.assert_exact(out, ref.to(out.dtype))` |
   | sampler | `ou.assert_matches_distribution` (chi-square) or `ou.assert_counts_within_binomial_bounds`, false-failure rate `alpha` (default 1e-6), fixed seed `ou.DEFAULT_SEED` |

5. **Memory checks**: allocate outputs (and in-place buffers) with `ou.guarded_empty` /
   `ou.guarded_like` and call `ou.check_guards(out=buf, ...)` after the kernel; outputs start
   NaN-poisoned, so an unwritten element fails the comparison; poison padding and unused
   regions with `ou.poison_` and assert untouched ones with `ou.assert_poisoned`; check that
   read-only inputs are unchanged with `ou.assert_exact(x, x_before)`. Mark one small,
   representative case `@pytest.mark.sanitizer`.

## Markers

| Marker | Meaning |
|---|---|
| `ou.requires_sm(ou.HOPPER)` / `ou.requires_sm(ou.BLACKWELL)` | Runs only on those compute-capability majors. Elsewhere it **skips with a reason**; it **fails** when the run targets that hardware (a GB200 run selected with `-m launch_on_gb200`, or `MCORE_ORACLE_TARGET_SM` set by the runner) but the device differs, so a misplaced runner cannot pass by skipping. |
| `pytest.mark.launch_on_gb200` | Required, written literally in the file, on every test that includes `BLACKWELL` in `requires_sm`: GB200 CI launches only files containing it and selects only tests carrying it (`tests/unit_tests/find_test_cases.py`). The conftest fails a Blackwell test without it. |
| `pytest.mark.sanitizer` | A small case selected for the compute-sanitizer pass. It also runs normally. |
| `pytest.mark.large` | Needs tens of GB (int32-overflow sizes). Skipped unless the `-m` expression names `large`. `ou.int32_overflow_params` adds it. |

## Running

```bash
# Everything that fits the device (CI runs this through torch.distributed.run).
pytest tests/unit_tests/inference/oracles

# Memory-safety pass.
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -m pytest -m sanitizer tests/unit_tests/inference/oracles

# Opt-in large sizes.
pytest -m large tests/unit_tests/inference/oracles

# A runner on a known node: hardware-gated tests fail instead of skipping off-target.
MCORE_ORACLE_TARGET_SM=9 pytest tests/unit_tests/inference/oracles
```

The helper tests (`test_oracle_utils.py`) run on CPU.

## Example

```python
"""Oracle for `_rms_norm_kernel` (megatron/core/...: line).

Contract: y[t] = x[t] / sqrt(mean(x[t]**2) + eps) * w for each of the first `n` rows; rows past
`n` are not written. x: [T, H] bf16, contiguous; w: [H] bf16. fp32 accumulation, one bf16
rounding. Tolerance class: floating point, bounded by torch.nn.functional.rms_norm.
Variants: BLOCK_H in {1024, 4096} (H <= 4096 single pass, else looped).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou

BLOCK_H = 1024  # kernel constant (file: line)


def reference(x, w, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w


@pytest.mark.parametrize("hidden", ou.size_params(BLOCK_H, 4096, degenerate=False))
@pytest.mark.parametrize("rows", ou.size_params(128, maximum=257))
def test_rms_norm(rows, hidden):
    g = ou.make_generator(device="cuda")
    x = torch.randn(rows, hidden, device="cuda", generator=g).to(torch.bfloat16)
    w = torch.randn(hidden, device="cuda", generator=g).to(torch.bfloat16)
    out = ou.guarded_empty((rows, hidden), torch.bfloat16)
    rms_norm(x, w, 1e-6, out=out.tensor)  # the kernel under test
    ou.check_guards(out=out)
    ou.assert_close_to_reference(
        out.tensor,
        ou.fp64_reference(reference, x, w, 1e-6),
        torch.nn.functional.rms_norm(x, (hidden,), w, 1e-6),
        dim=-1,
        what="rms_norm",
    )
```
