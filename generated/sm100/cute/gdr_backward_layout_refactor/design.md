# Fused GDR backward CuTe layout refactor

Status: implemented and validated
Baseline: `15093c8aa` (`codex/integrate-cutedsl-internal-gdr`)
Target: NVIDIA Blackwell SM100 / GB200
Primary implementation:
`megatron/core/ssm/gated_delta_net/internal_gdn_backend/kernels/fused_gdr_bwd_cute/kernel.py`

## 1. Problem statement

The fused Gated Delta Rule backward kernel is correct and performance-oriented, but
`kernel.py` has grown to roughly 6.4 KLOC. A meaningful part of that size comes from
layout construction and layout-dependent wiring rather than the reverse recurrence
itself:

- ten tcgen05 MMA orientation variants are built from a declarative table, but
  downstream wiring still unpacks them by positional index;
- six canonical physical SMEM views are recovered indirectly from selected variants;
- nineteen logical MMA operations map variant names to physical operand views;
- packed 64x128 MMA aliases are passed as tuple positions and unpacked manually;
- the kernel entry point creates dozens of aliases with repeated
  `storage.<buffer>.get_tensor(layout.outer, swizzle=layout.inner)` calls;
- TMA descriptors, MMA operands, TMEM accumulators, and epilogue copy views encode
  related layout facts in separate code paths;
- compatibility checks are valuable, but are interleaved with the main kernel class.

This refactor will make those relationships explicit and reduce repeated layout
plumbing while preserving generated layouts, descriptor semantics, instruction order,
memory allocation, synchronization, numerical behavior, and performance.

This is a structural refactor, not a new backward algorithm.

## 2. Current public contract

The refactor must preserve the current low-level contract exactly.

### Inputs

- `q`, `k`, `v`, `do`: BF16 `[1, N, 64, 128]`.
- `a`: BF16 `[1, N, 64, 64]`.
- `g`, `beta`: FP32 `[1, N, 64]`.
- `h`: BF16 `[1, N / 64, 64, 128, 128]`.
- `dht`: FP32 `[B, 64, 128, 128]`.
- `cu_seqlens`, `chunk_offsets`: contiguous CUDA int32 `[B + 1]`.
- `scale`: finite positive scalar.

`N` is packed token count and `B` is the logical sequence count. The physical token
batch dimension remains one. At this baseline, every logical sequence length is a
positive multiple of 64.

### Outputs

`dq`, `dk`, `dv`, `dg`, `dbeta`, and `dh0` keep their current shapes and dtypes.

### Fixed kernel parameters

- chunk tile `BT = 64`;
- key and value dimensions `DK = DV = 128`;
- heads and grouped heads are both 64 for the production shape;
- BF16 inputs and FP32 accumulation;
- one CTA per logical `(sequence, head)` pair;
- 384 threads per CTA;
- single-CTA tcgen05 MMA;
- 480 logical TMEM columns allocated as `256 + 128 + 32 + 64`;
- current SMEM allocation and one-stage pipelines.

## 3. Goals

1. Make one module the source of truth for MMA variants, canonical layouts, logical
   operation bindings, packed aliases, and compatibility validation.
2. Replace positional tuples with named immutable bundles.
3. Centralize SMEM tensor-view construction, including base offsets and swizzles.
4. Keep TMA descriptor construction tied to the exact physical layouts consumed by
   MMA.
5. Move pure layout and resource-contract code out of `kernel.py` without moving the
   device schedule or changing its phase ordering.
6. Make layout invariants independently traceable through the existing layout probe.
7. Reduce the risk of silently pairing an MMA variant with an incompatible SMEM alias.
8. Preserve steady-state performance within measurement noise.

## 4. Non-goals

- No mathematical change to GDR backward.
- No new dtype, head dimension, GQA, chunk size, or architecture support.
- No implementation of non-64-aligned sequence tails in this refactor.
- No new MMA tile search or pipeline-depth tuning.
- No change to warp roles, mbarrier topology, TMEM live ranges, or SMEM reuse.
- No change to autograd dispatch, fallback behavior, or public backend configuration.
- No requirement to reduce source size by weakening validation or deleting comments.
- No NCU-guided optimization unless profiling later shows a regression.

Tail support should be a separate change because padding, masks, recurrent-state
checkpoints, and output trimming alter the input contract rather than layout ownership.

## 5. Target hardware and invariants

The target is SM100 as exposed by GB200 systems.

- Kernel opt-in SMEM limit for one CTA: 232,448 bytes.
- TMEM capacity: 512 columns; current kernel consumes 480 logical columns.
- TMEM allocation units must remain legal power-of-two blocks.
- tcgen05 BF16 MMA uses K=16 instructions with FP32 accumulators.
- MMA A/B descriptors must preserve the selected major modes and swizzle.
- Matrix-tile TMA buffers retain their current 128-byte alignment; the 4x64
  `g`/`beta` vector TMA buffers retain their current 16-byte alignment.
- A composed SMEM layout must be allocated as `.outer` plus `.inner` swizzle exactly
  once.
- Packed 64x128 variants must preserve their permuted N grouping and 64-column physical
  accumulator footprint.
- Layout simplification must not rely on equal element counts alone when shape, stride,
  swizzle, or composition offset affects addressing.

The existing 384-thread role split remains unchanged:

| Warps | Role |
|---|---|
| 0-3 | recurrent state plus K/V consumers and dK/dV epilogues |
| 4-7 | A/Q consumers and dQ/dA/gate/beta work |
| 8 | tcgen05 MMA issue |
| 9 | TMA input loads |
| 10 | reserved producer slot |
| 11 | staged dQ store |

## 6. Proposed module structure

Keep the package boundary unchanged and add two focused internal modules.

```text
fused_gdr_bwd_cute/
  fused_bwd.py          # validation, packing, output restoration
  launcher.py           # descriptors, cache key, compilation, stream binding
  layouts.py            # new: MMA/layout/TMA contracts and named bundles
  storage.py            # new: SharedStorage and named SMEM/TMEM view construction
  kernel.py             # schedule, role bodies, pipeline wiring, launch entry point
  tcgen05_ws.py         # existing low-level SM100 helpers
  kernel.md             # user-facing contract and usage
```

If import-cycle or CuTe trace-time constraints make two modules impractical, merge
`storage.py` into `layouts.py`. Do not split role bodies merely to meet a line-count
target; the device schedule is easier to review when phase order remains contiguous.

### `layouts.py`

Own:

- `MmaVariantSpec`, `MmaOperationSpec`, and built-variant records;
- `MMA_VARIANT_SPECS` and `MMA_OPERATION_SPECS`;
- canonical view names and physical buffer-view bindings;
- tiled-MMA construction;
- staged and stage-free operand layout construction;
- packed 64x128 alias definitions;
- tensor-to-matrix views used by TMA;
- static TMA descriptor specifications and the tensor-bound descriptor builder;
- layout compatibility and TMEM-column checks;
- the small layout probe entry point or its callable builder.

Return one named `BackwardLayoutPlan` rather than several correlated tuples.

Conceptually the bundle contains:

```text
BackwardLayoutPlan
  variants[name]
    spec
    tiled_mma
    smem_a_staged
    smem_b_staged
  canonical
    token_direct
    token_transposed
    square_direct
    square_transposed
    state_direct
    state_transposed
  operation_bindings[name]
```

The plan is trace-time-only. It may use frozen dataclasses or `NamedTuple`, but it must
not cross the `@cute.kernel` boundary. Fields used in a CuTe trace must remain statically
resolvable; avoid runtime string lookups inside the device kernel.

Tensor-bound TMA objects live in a separate, short-lived `TmaDescriptorBundle` created
inside `FusedGdrBwdKernel.__call__`. The bundle must not be retained by compiled cache
artifacts and must be flattened to explicit kernel arguments before launch.

### `storage.py`

Own:

- `SharedStorage`;
- `LayoutBudget` and `get_layout_budget()`;
- TMEM range declarations and their Python-side overlap proof;
- named construction of physical SMEM views from a layout plan;
- named construction of TMEM accumulator and epilogue-copy views.

Return two immutable bundles:

```text
SharedViews
  canonical buffers
  transposed aliases
  packed MMA aliases
  padded scalar-copy aliases
  vector and reduction buffers

TmemViews
  p, dp, a, da, mask
  dk, dv, dq, u, vprime, dog
  dh_left, dh_right
```

These bundles are also trace-time-only and are flattened before the `@cute.kernel`
launch. The trace layer should consume named fields such as
`shared.packed.dq_state_a`, not `packed_layouts[3][0]`.

## 7. Layout construction strategy

### 7.1 Keep canonical physical families small

Six physical SMEM mappings are sufficient:

| Canonical view | Logical extent | Purpose |
|---|---:|---|
| `token_direct` | 64x128 | K-major token/state-vector operands |
| `token_transposed` | 128x64 | MN-major transposed token operands |
| `square_direct` | 64x64 | K-major chunk-square operands |
| `square_transposed` | 64x64 | MN-major chunk-square operands |
| `state_direct` | 128x128 | K-major recurrent-state operands |
| `state_transposed` | 128x128 | MN/K-oriented state operands |

Construct each canonical staged layout once with SM100 helpers. Derive the stage-free
view with one named helper that uses `cute.select` to remove only the singleton stage
mode. Preserve the `ComposedLayout` and its swizzle.

Do not replace these mappings with ad-hoc `make_layout` strides. The helper-generated
layout is part of the TMA/tcgen05 compatibility contract.

### 7.2 Build variants from declarative specifications

Keep one declarative table for `(M, N, K, A-major, B-major)`. Build `TiledMma`, A layout,
and B layout in one pass.

For 64x128 outputs, retain the existing `instruction_n = 64` plus
`permutation_mnk=(64, 128, K)` behavior. This is not syntactic noise: it controls the
interleaved TMEM representation and cannot be inferred from a generic tile helper.

### 7.3 Bind logical operations by name

Each logical MMA phase should name:

- its variant;
- its A physical view;
- its B physical view;
- whether the result is interpreted as transposed.

Validate every binding while tracing the layout plan. For ordinary variants, require
full layout equality. For packed variants, define the explicit N-grouping transform and
prove that the regrouped composed layout has the same coordinate-to-address mapping as
the canonical physical view. Matching swizzle, offset, or element count alone is not a
sufficient proof. Every scheduled MMA must obtain its variant and operands from this
binding; the table must not remain validation-only metadata.

### 7.4 Replace repetitive tensor aliases with view specifications

Represent each SMEM alias with a static specification:

```text
buffer field + layout name + optional element offset + optional dtype
```

A single helper should perform the two legal constructions:

1. zero-offset: `field.get_tensor(layout.outer, swizzle=layout.inner)`;
2. offset alias: recast `field.data_ptr() + offset` with the same swizzle, then attach
   `layout.outer`.

This removes repeated boilerplate without hiding physical offsets. Offsets for the
right 64-column halves of `sK`, `sTmp21`, and `sTmp23` must remain named constants and
must be validated against their backing buffer extents.

### 7.5 Preserve special padded views explicitly

The `(_BT, _BT)` views with row stride `_BT + 2` are deliberately different from MMA
layouts. They support scalar/stmatrix paths and bank-conflict behavior. Keep them as a
separate `PaddedSquareViews` group rather than treating them as canonical MMA aliases.

### 7.6 Keep TMA and MMA layouts coupled

Build Q/K/V/A/dO/H TMA atoms from the same built variants used by MMA. Build g/beta
vector TMA from its explicit 4x64 layout and preserve its 16-byte alignment. Build dQ
S2G TMA from the flattened store layout only after proving its physical address mapping
matches `token_direct` after the explicit flattening transform.

No descriptor, MMA operation, or accumulator should select a layout or variant by a
positional index.

### 7.7 Keep TMEM views named and range-checked

Create TMEM accumulator layouts with `tiled_mma.make_fragment_C` and keep the existing
offsets. The plan must prove:

- all physical column counts match the expected variant footprint;
- every named tensor fits its declared range;
- overlapping column ranges have disjoint live phases;
- total logical range is at most 480 columns;
- four physical allocations remain `256`, `128`, `32`, and `64` columns.

The manually specified square-copy layout remains allowed because it describes a
tcgen05 load/store data-path mapping, not an interchangeable row-major matrix.

## 8. Memory hierarchy plan

This refactor must preserve the current data movement:

1. Global tensors are viewed as token, state, or vector matrices at trace time.
2. TMA loads Q/K/V/A/dO/H into the existing matrix-tile SMEM buffers and loads g/beta
   into the existing 16-byte-aligned vector TMA buffers.
3. Canonical and packed aliases expose the same SMEM bytes to tcgen05 descriptors.
4. MMA writes FP32 accumulators into named TMEM regions.
5. Consumer warps read TMEM into registers, perform reductions and elementwise work,
   and stage outputs through SMEM or direct vector stores.
6. dQ uses its existing SMEM handoff and dedicated TMA store warp.

No buffer is added, removed, resized, or re-aligned in the structural phases. Any later
memory optimization requires a separate benchmarked change.

## 9. Math instruction and launch choices

- Use `tcgen05.MmaF16BF16Op` with BF16 A/B and FP32 accumulation.
- Use `CtaGroup.ONE` and `OperandSource.SMEM`.
- Retain the ten currently required major-mode/tile variants.
- Retain the current 21 completion events and 16 logical input-ready phases.
- Retain launch grid `(num_sequences * heads, 1, 1)`.
- Retain block `(384, 1, 1)`, cluster `(1, 1, 1)`, and
  `min_blocks_per_mp=1`.
- Retain the current dynamically marked packed-token mode and compilation cache key.

The refactor must not reorder `cute.gemm`, `tcgen05.commit`, pipeline acquire/wait,
pipeline release, TMA issue, or output-store operations.

## 10. Implementation phases

### Phase 1: Freeze the baseline

- Record baseline commit, CuTe DSL/CUDA versions, GB200 SKU, and clock mode.
- Run the layout probe and current CPU unit tests.
- Run the GB200 E2E correctness/performance test at `B=2, T=8192, H=64, D=128`.
- Save paired baseline timings from at least two warmups and 20 measured samples.
- Save deterministic trace-time layout snapshots and compile-time IR fingerprints for
  both uniform-length and packed-variable-length specializations.

Exit gate: reproducible baseline with no fallback to FLA.

### Phase 2: Extract declarative layout contracts

- Move immutable spec records and tables into `layouts.py`.
- Move tiled-MMA and canonical-view construction unchanged.
- Move layout, accumulator, and TMEM-column validations unchanged.
- Keep `kernel.py` calling the extracted builder with identical return values.

Exit gate: layout probe passes and dumped IR is instruction-equivalent.

### Phase 3: Introduce named layout and descriptor bundles

- Replace `mma_variants` and `packed_layouts` positional tuples at the host/JIT boundary.
- Keep all named bundles trace-time-only and statically unpack them before the kernel
  launch.
- Replace numeric MMA, packed-layout, TMA-descriptor, and accumulator indices with
  names at the last Python trace-time layer.
- Make `MMA_OPERATION_SPECS` drive production wiring; enforce or remove every field.

Exit gate: no positional layout index remains outside the compatibility shim; IR and
correctness remain unchanged.

### Phase 4: Centralize SMEM view creation

- Add one composed-layout allocator helper and one offset-alias helper.
- Build `SharedViews` from declarative view specifications.
- Keep special padded layouts and vector/reduction layouts explicit.
- Replace the long alias-construction block in `_fused_bwd_kernel` with named bundles.

Exit gate: exact SMEM size, descriptor layout, generated addresses, and E2E results.

### Phase 5: Centralize TMEM and resource contracts

- Move TMEM ranges, overlap checks, accumulator creation, and named views to
  `storage.py`.
- Preserve physical allocation instructions and offsets.
- Keep allocation/free synchronization in `kernel.py`.

Exit gate: 480-column budget, identical allocation instruction sequence, and no
correctness or performance regression.

### Phase 6: Cleanup and documentation

- Remove compatibility shims and dead duplicate helpers.
- Keep the role schedule and phase comments in `kernel.py`.
- Update `kernel.md` package structure and developer notes.
- Run formatting, import sorting, unit tests, GB200 E2E, and final profiling.

Exit gate: all acceptance criteria below are satisfied.

## 11. Correctness validation

### CPU/static tests

- Verify all ten MMA variants and nineteen logical operation bindings are present.
- Verify canonical layout names are unique and every operation references valid names.
- Verify all TMEM ranges and live intervals.
- Verify resource budget: 384 threads, 480 TMEM columns, SMEM at or below 232,448 bytes.
- Verify wrapper validation and arbitrary logical batch metadata remain unchanged.

### GB200 layout/compile tests

- Compile and launch `layout_probe`.
- Compare a deterministic trace-time snapshot of every variant's shape, stride, swizzle,
  composition offset, packed regrouping, and physical address mapping with baseline.
- Confirm accumulator layouts and physical TMEM column counts match baseline.
- Confirm TMA descriptors compile for Q/K/V/A/g/beta/dO/H and dQ store.
- Compile both uniform-length and packed-variable-length specializations.

### Numerical tests

Use explicit CuTe dispatch so fallback cannot mask a failure.

- Main E2E: `B=2, T=8192, H=64, D=128`, BF16.
- Small diagnostic cases for at least `B=1` and `B=3` with 64-aligned sequence lengths.
- Packed variable-length case with unequal, positive, 64-aligned lengths.
- Test both `gdn_gdr_recompute_h=False` and `True` where supported.
- Compare forward output and all backward gradients with the existing FLA reference and
  repository tolerances.
- Check outputs for NaN/Inf before tolerance comparison.

## 12. IR and generated-code equivalence

Structural Python changes can still alter CuTe specialization. After each phase:

1. dump post-lowering IR for the baseline and candidate with the same specialization;
2. normalize nondeterministic symbol names and temporary paths;
3. compare tcgen05 MMA count/order, TMA descriptors, mbarrier operations, shared-memory
   offsets, TMEM allocation/deallocation, and launch attributes;
4. inspect SASS only when IR differs or runtime changes materially.

Permitted differences are symbol names, source locations, and removed dead trace-time
assert scaffolding. Any instruction, descriptor, address, or barrier-order difference
requires explanation and a separate correctness/performance review.

## 13. Performance validation

Primary performance case:

- `B=2, T=8192, H=64, D=128`, BF16;
- explicit `fused_gdr_bwd` call with `MCORE_GDN_INTERNAL_BACKEND=cute` where E2E dispatch
  is involved;
- JIT compilation excluded;
- two or more warmups;
- at least 20 baseline/candidate CUDA-event sample pairs;
- report paired median ratio, interquartile range, and a bootstrap confidence interval;
- run baseline and candidate interleaved on the same allocated GB200 node.

Acceptance threshold:

- candidate median backward time is no worse than 2% versus baseline;
- no systematic increase in dispersion;
- full forward-plus-backward E2E continues to satisfy the existing 10% CI guard versus
  FLA;
- kernel count and fallback guards prove the intended fused kernels executed.

If regression exceeds 2%, use nsys to compare launch gaps and kernel duration. Use NCU
only when explicitly requested or when a separate profiling task is approved.

## 14. Documentation requirements

Update the user-facing `kernel.md` only for facts that affect use or maintenance:

- package structure;
- unchanged input contract;
- where layouts and storage contracts now live;
- how to run the layout probe and GB200 E2E test;
- performance reporting requirements.

Do not include migration provenance, unrelated repositories, or hard-coded performance
claims without an environment and commit.

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Python abstraction changes CuTe tracing | introduce one phase at a time and compare IR |
| Named containers are not accepted by `@cute.kernel` | keep bundles trace-time-only and flatten explicit arguments before launch |
| Composed layout swizzle applied twice | one allocator helper; validate `.outer`/`.inner` usage |
| Packed 64x128 alias loses permutation | keep explicit packed rule and physical-column checks |
| Offset alias escapes backing buffer | centralize offsets and validate cosize against storage |
| Import cycle between kernel/layout/storage | keep data records dependency-light; kernel owns schedule |
| Validation removal hides mismatch | move checks unchanged before simplifying them |
| Source shrinks but compile time grows | measure cold JIT time as a secondary metric |
| Performance noise hides regression | interleave baseline/candidate samples on one node |

## 16. Rollback strategy

Each phase should be one reviewable commit. If a phase fails IR, correctness, or
performance gates, revert only that phase and retain earlier verified extractions.

Do not combine structural refactoring with tail support or performance tuning. This
keeps rollback meaningful and prevents numerical or timing changes from being
misattributed to layout cleanup.

## 17. Acceptance criteria

The project is complete only when all items are true:

- `kernel.py` no longer owns declarative MMA/layout tables or the repeated SMEM alias
  construction block.
- Layouts, descriptors, shared views, and TMEM views have named sources of truth.
- No positional variant/layout indices remain in production wiring.
- All original layout and TMEM safety assertions are preserved or strengthened.
- Public API, dispatch, dtype, shape, sequence-alignment, and output contracts are
  unchanged.
- No `mma_variants[N]`, `packed_layouts[N]`, or `tma_inputs[N]` remains in production
  wiring, and logical operation metadata drives the actual scheduled binding.
- SMEM bytes, TMEM columns, thread count, launch geometry, and phase ordering are
  unchanged.
- CPU tests, deterministic layout snapshots, and the SM100 layout probe pass.
- GB200 E2E correctness passes for forward and backward.
- Uniform and packed-variable baseline/candidate IR comparisons have no unexplained
  operational differences.
- `B=2, T=8192, H=64` backward median is within 2% of baseline.
- Documentation reflects the final module structure and validation commands.

## 18. Implementation decisions

1. Frozen dataclasses and named tuples remain trace-time-only; all device arguments are
   flattened before the `@cute.kernel` boundary.
2. Packed operands use an explicit regrouping transform followed by a complete physical
   address-mapping equivalence check.
3. `SharedStorage` moves to `storage.py`; dependency direction remains
   `layouts -> storage -> kernel`, and `kernel.py` retains allocation/free ordering.
4. Cold JIT compile time is reported as a secondary metric, not a hard acceptance gate.
5. This change extracts only pure layout/view/fragment construction. Role-body phase
   orchestration remains in `kernel.py`.

## 19. Validation record

Validated on a single NVIDIA GB200 at baseline `15093c8aa`:

- uniform and packed-variable specializations compile and launch successfully;
- packed `(64, 128)` and `B=2, T=8192, H=64` baseline/candidate outputs are bitwise
  identical for all six backward outputs;
- 20 interleaved, post-warmup CUDA-event pairs measured 1.01250 ms baseline and
  1.01170 ms candidate median, for a paired median ratio of 0.9965; the paired-ratio
  IQR was `[0.9924, 1.0063]` and its bootstrap 95% CI was `[0.9927, 1.0056]`;
- cold first-call time was 15.32 s baseline and 15.22 s candidate;
- raw clean IR differs only in trace arguments, type-alias ordering, and SSA numbering
  introduced by flattening nineteen named operation bindings; after normalizing those
  trace-only differences, all 349 MMA, TMA, mbarrier, TMEM, and launch operations match
  in count and order;
- fused CuTe forward plus backward matches FLA for the main shape with both saved-H and
  recompute-H paths; five post-warmup samples measured 2.077 ms CuTe versus 5.656 ms FLA
  (2.72x). This run used the FLA 0.5.1 Triton path because its optional TileLang
  dispatcher raised a misaligned-address error on this GB200 environment;
- Python compilation, import sorting, Ruff, and whitespace validation pass.

The repository GB200 E2E test remains the authoritative FLA forward/backward
correctness and 10% performance guard. JIT time is excluded from its timing result.
