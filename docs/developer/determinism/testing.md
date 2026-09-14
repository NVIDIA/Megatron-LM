---
orphan: true
---

# Kernel Determinism Testing

> For the determinism contract and the operation catalog, refer to
> [`status.md`](./status.md) and [`op-catalog.md`](./op-catalog.md).

Every GPU kernel Megatron dispatches must have a bit-exact determinism test,
and every pull request that adds or changes a kernel must add or update that
test. This page describes what counts as a kernel, where the tests live, how
the requirement is enforced, and how to add a test.

## What counts as a kernel

Any code path where Megatron itself launches or selects a GPU kernel whose
numerical result could depend on scheduling:

- Triton `@triton.jit` kernels (and TileLang / cuTile kernels)
- `@jit_fuser` / `torch.compile` fused functions
- C++/CUDA extensions (`load_inline`, `CUDAExtension`, `.cu` sources)
- Transformer Engine and external-library dispatch with algorithm choices
  (grouped GEMM, attention backends, causal_conv1d, mamba_ssm, FLA, DeepEP,
  cuTile, FlashInfer, apex multi-tensor kernels)
- torch ops with a non-deterministic accumulation: `scatter_add_`,
  `index_add_`, `index_put_(accumulate=True)`, `bincount`, embedding backward

`tests/unit_tests/determinism/kernels/manifest.py` lists the patterns
(`KERNEL_CONTENT_PATTERNS`) and the directories (`KERNEL_DIRECTORIES`) that
make a file *kernel-bearing*. External-library patterns match *call sites*
(`causal_conv1d_fn(`, `tex.rmsnorm_fwd(`, `buffer.dispatch(`, ...), never
bare imports, so a module that only imports a class or checks `isinstance`
is not kernel-bearing. Modules that select among or call external kernels
without defining one (the Mamba mixer, gated delta product, RoPE dispatch,
FP8 master-weight casts, ...) are registered with `kind="dispatch"`; they are
covered by the replay tests of the kernels they call, named in the entry's
`notes`, and by module-level tests where those exist.

## Where the tests live

| Layer | Location | What it asserts |
| --- | --- | --- |
| Kernel | `tests/unit_tests/determinism/kernels/test_*.py` | One kernel family per file. Each kernel is run several times on identical inputs and every output tensor and gradient must be byte-identical (`harness.assert_replays_bit_exact`, `assert_module_replays_bit_exact`; `harness.bytes_equal` compares bit patterns, so signed zeros and NaN payloads must match too). |
| Module / model | `tests/unit_tests/determinism/correctness/` | GPT, TransformerBlock, HybridModel and FP8/FP4 recipes across parallelism cells (`BitExactRunner`). |
| End to end | functional tests with `--deterministic-mode` | Loss and `num-zeros` compared against golden values at their recorded precision; legacy goldens use five decimals. |

The manifest (`tests/unit_tests/determinism/kernels/manifest.py`) is the
registry that ties kernel source files to the tests that cover them. Each
`KernelEntry` names the source files, the tests, the kernel kind, and either a
non-empty `tests` tuple or an explicit `exempt_reason` (for example a
multi-rank NVLS collective that needs an NVLink peer group, or an allocator
with no compute kernel). Exemptions are visible coverage debt, not silence.

## How the requirement is enforced

1. **Repository invariant** (`tests/unit_tests/determinism/kernels/test_manifest.py`,
   CPU only, runs in the unit-test bucket): every kernel-bearing file under
   `megatron/` is registered, every registered path exists, and every entry
   has tests or an exemption. Adding a Triton kernel to a new file without
   registering it fails the unit tests.
2. **Pull-request gate** (`tools/check_kernel_determinism_coverage.py`, run by
   the `linting` job in `.github/workflows/cicd-main.yml` on PR pushes):
   - a changed kernel-bearing file must be registered (not overridable);
   - a changed registered kernel source must come with a change to at least
     one of its determinism tests. Override with the `determinism-exempt`
     PR label when the change cannot affect numerics (comment-only edits,
     refactors); the check then logs the exemption instead of failing.
3. **Review**: the PR template checkbox and the `/claude review` prompt ask
   for the test explicitly.

The kernel bucket runs in the H100 unit-test recipe
(`tests/test_utils/recipes/h100/unit-tests.yaml`). Hardware-specific
scheduling is exactly what these tests are meant to catch, so running the
bucket on GB200/GB300-class runners as well (GB200 unit tests are selected by
the `launch_on_gb200` marker) is a tracked follow-up; until then, reproduce
findings on Blackwell hardware manually as described in
[`status.md`](./status.md).

Run the gate locally against `main`:

```bash
python3 tools/check_kernel_determinism_coverage.py --base-ref origin/main
```

## Adding a kernel test

1. Put the test in the `tests/unit_tests/determinism/kernels/test_*.py`
   module that matches the kernel family (or add one; the package `__init__`
   pins the determinism environment at import).
2. Use the harness:

   ```python
   from tests.unit_tests.determinism.kernels.harness import (
       assert_replays_bit_exact,
       assert_module_replays_bit_exact,
       deterministic_algorithms,
       seeded,
   )

   seeded()
   x = torch.randn(16384, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
   assert_replays_bit_exact(my_kernel, (x,), replays=3, contention=True, what="my_kernel")
   ```

   `contention=True` runs the replays under side-stream GEMM pressure so
   ordering-dependent reductions surface. Size inputs so many CTAs contend on
   the same outputs; a two-block reduction can be deterministic by accident.
   Kernels that select a branch on `torch.are_deterministic_algorithms_enabled()`
   should be exercised on both branches with `deterministic_algorithms(...)`.
   Kernels that consume the RNG (dropout) use `restore_rng=True`.
3. When the default path is known to race (atomics), add a negative control
   with `count_differing_replays(...) > 0`, generously sized, so the strict
   assertion is known to be sensitive (see `test_moe_kernels.py`).
4. Register the kernel in `manifest.py`: source files, the test file, kind,
   and a note on the non-determinism mechanism and how the deterministic
   branch is selected.
5. Run the file on a GPU node:

   ```bash
   uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
     tests/unit_tests/determinism/kernels/test_my_family.py \
     tests/unit_tests/determinism/kernels/test_manifest.py
   ```
