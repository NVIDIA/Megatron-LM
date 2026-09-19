---
name: mcore-determinism-debug
description: Root-cause method when two Megatron runs that should be bit-identical diverge: fingerprint every op, diff the two traces, first mismatch is the culprit.
license: Apache-2.0
when_to_use: Two identical training runs give different loss; a run is bit-exact at N nodes but not 2N; you need to know which op, layer or kernel first diverged; 'nondeterminism', 'not bit-exact', 'runs diverge', 'determinism trace', 'first divergence', 'which kernel is nondeterministic'.
metadata:
  author: Zhiyu Li <zhiyul@nvidia.com>
---

# Find the first op that diverges

## Answer-First Constants

- **The loss curve tells you where the difference got big enough to print, not
  where it started.** Do not debug from the divergent iteration number.
- **The method:** run the job twice with the same seeds, data and config. Each
  rank writes its own trace. Fingerprint every op. Diff the two traces offline,
  aligned op by op. **The first mismatch is the culprit.**
- **The test at each record:**
  `hash(in_A) == hash(in_B) && hash(out_A) != hash(out_B)` → that op did it.
  Everything before it matched. If the inputs differ too, keep walking back.
- **"First" is only exact within one rank.** Across ranks there is no global
  clock — order them causally by the input test, not by sequence number.
- **Two capture layers.** *Semantic* — hooks at known boundaries (collectives,
  recompute, optimizer, P2P) name the phase that diverged. *Op* — fingerprints
  every ATen op's output, so TE / Triton kernels that bypass PyTorch surface at
  the first op that reads their output.
- **Traces are rank-local JSONL.** No collectives, no cross-rank ordering added
  to the step being observed — tracing cannot perturb what it measures.
- **Digests are computed on the GPU** with an order-independent reduction, so the
  same bytes digest identically on any rank, GPU or topology. Use
  `torch.hash_tensor`; only a scalar per tensor reaches the host. Byte-viewing
  makes it dtype-agnostic, so MXFP8 / NVFP4 payloads and their scale buffers are
  covered.
- **Scope the capture.** Chosen steps (start / end / every N-th) and chosen
  ranks (all, or a list like `0,3,8-15`), plus budget caps — that is what keeps
  a multi-node campaign bounded. Off by default; zero cost unless enabled.
- **No hypothesis is required, and none should be formed before the first
  divergent record is in hand.**

This skill documents a method, not a shipped feature — expect to supply the
tracer rather than find one. `references/tracing-setup.md` §1 covers that first.

## What "first" means

The method rests on the word *first*, and it is only well-defined inside one
rank. Get this wrong and you will name the wrong site.

Within a rank the stream is totally ordered by `seq`, so the earliest divergent
record is unambiguous. Across ranks there is no global clock — ranks agree on the
*iteration* number and nothing finer, so never compare `seq` between two ranks.
Order them causally instead: take each rank's own first divergence and classify
it by the input test.

  | That rank's first divergence | Means | Verdict |
  |---|---|---|
  | inputs **matched** | it produced a bad value from good inputs | **origin** — a candidate site |
  | inputs **differed** | it consumed a bad value from elsewhere | **receiver** — downstream, ignore |

  Only origins are candidates. Receivers are consequences that arrived over a
  collective or P2P, however early their `seq` looks.

- **Read the pattern across origins**, which tells you the class of bug:
  - *Many ranks, same op identity* → the operation itself is nondeterministic;
    any one of those ranks is a valid site, and you are looking at a kernel.
  - *One rank, or a subset* → the operation is fine and the input distribution
    is not; suspect reduction order, topology, or rank placement.
  - *No origins, only receivers* → the true origin is outside the traced window
    or on an untraced rank. Widen before concluding anything.

The corollary is that **tracing a subset of ranks can only ever find an origin
that happens to be in the subset.** When you scope by rank to control cost, a
"no divergence found" result means nothing about the ranks you skipped.

## Before you trace: two cheap checks

Not the subject of this skill, but skipping them wastes a week:

1. Is the suspect operation already a **known unsupported case** in
   `docs/developer/determinism/op-catalog.md`? Then it is a documented gap, not
   a bug to trace.
2. Are the two runs genuinely comparable — same seed, data order, global batch,
   parallelism layout, image? And compare full-precision serialized metrics, not
   console logs, which print at limited precision.

Everything below assumes both pass.

## The method in code

Two independent topics. **Capture** decides what is recorded, where, and under
what identity. **The hash** decides how a tensor becomes one comparable value.
Swapping either does not touch the other — treat the hash as pluggable.

### Topic 1 — capture and diff

This is the method. Illustrative, not an API:

```python
# CAPTURE — one record per op, per rank, written to a local file.
# The occurrence counter is load-bearing: it gives every record an identity that
# does not depend on arrival order, which is the only reason two independent
# runs can be aligned offline.
seen[op_name] += 1
trace.record("op", op_name, {
    "id":  f"{op_name}:{seen[op_name]}",
    "in":  [trace.stage(x) for x in inputs],    # stage() keeps the digest on
    "out": [trace.stage(y) for y in outputs],   # device; resolved at flush
})

# DIFF — offline. Align by IDENTITY, never by arrival order: if control flow
# skews, position i in A and position i in B are different operations.
index_b = {(e["name"], e["payload"]["id"]): e for e in stream_B}
for a in stream_A:                   # A's own order is causal within one rank
    b = index_b.get((a["name"], a["payload"]["id"]))
    if b is None:
        break                        # no counterpart -> control flow diverged
    if a["payload"]["out"] != b["payload"]["out"]:
        inputs_matched = a["payload"]["in"] == b["payload"]["in"]
        break                        # True -> this op did it; False -> walk back
```

### Topic 2 — the hash function

**Use `torch.hash_tensor`, and do not spend design effort here.** It is
GPU-resident, order-independent, 1-ULP sensitive, and returns a `uint64` *tensor*
you can stage — everything the method needs, and far cheaper than a hand-written
digest.

Two rules carry the rest:

- **Never** `hash(tensor.numpy().tobytes())`. It is salted by `PYTHONHASHSEED`,
  so it differs between processes — silently wrong for a cross-launch comparison.
- **Chunk it** (`dim=`) on tensors whose failure mode is reordering: routing
  maps, MoE dispatch, sort indices. Whole-tensor xor is permutation-blind; per
  row or per chunk is not, and costs the same.

Carry `shape`/`dtype`/`numel` alongside the digest — that is what separates an
all-zero tensor from the empty-tensor sentinel. The reasoning, the benchmarks,
and the one case for a custom digest: `references/tracing-setup.md` §3.

## Workflow

### 1. Reproduce at the smallest scale that still breaks

Cheapest first — do not start on a cluster:

- **Module level.** `tests/unit_tests/determinism/` runs a model or block twice
  under restored RNG state and asserts bit-identical outputs and gradients,
  parametrized over model presets × parallelism cells × FP8/FP4 recipes. If your
  break reproduces by adding a case to `configs.PARALLELISM_CONFIGS`, you never
  need an allocation — and that case is the regression test you will need later.
- **Job level.** One node, N tasks = N **independent** runs from identical
  state. Any difference between arms is a determinism failure. Four arms lets
  you run two broken and two fixed in the same allocation, on the same hardware,
  at the same time — which removes "different node" as an explanation.

Shrink layers, iterations and sequence length as long as the break survives.

**Do not centre the trace window on the iteration where the loss split.** That is
the one mistake this skill exists to prevent, and it is easy to make twice. Bit
divergence precedes visible divergence, usually by many iterations — a loss that
splits at iteration 6 routinely has its first differing byte in iteration 1.
Start the window at the **first** iteration and widen only if nothing is found.

If iterations 1–2 are clean, the cause is state that has to accumulate before it
differs — optimizer moments, a checkpoint/resume boundary, a periodic
all-reduce. Bisect with a cheap rung (metrics or collectives only) over a wide
range to find the first iteration containing *any* divergence, then spend the op
layer on that single iteration.

### 2. Climb the capture ladder — one rung at a time

Climb only when the previous rung's blind spot is implicated. The big jump is
rung 1 → 2: collectives alone are hundreds of records per iteration, every ATen
op is tens of thousands. Rung 3 is cheap by comparison — it adds a handful of
probes to a stream you are already paying for.

| Rung | Capture | Answers |
|---|---|---|
| 0 | Full-precision metrics | First divergent **iteration** |
| 1 | **Semantic layer** — collectives, recompute pairs, optimizer, P2P, grad slices before bucket reduce | Which **phase/boundary** diverged; catches the classic cross-process reduction-order break |
| 2 | **Op layer** — every ATen op output via `TorchDispatchMode` | The exact **compute op** between boundaries |
| 3 | **Targeted probe** — a hook on the extension call rung 2 cannot see (TE GEMM, fused wgrad, MoE dispatch) | The **library call** and its operand bits |
| 4 | **Below the dispatcher** — `CUBLASLT_LOG_LEVEL=5`, `nsys` + `NVTE_NVTX_ENABLED=1`, then `ncu` | The **kernel symbol** and algo descriptor |

Rungs 1–3 write into the *same* ordered stream, so the offline diff walks them
together and the first divergence is comparable across layers.

Two-stage strategy for cost: trace **collectives only** first — cheap, and it
catches the usual multi-node reduction-order break. Add the op layer only on the
narrowed window, when you need to see *inside* the compute of the divergent
phase.

Practical rules:

- Trace a **bounded iteration window** (e.g. `1-2`), not a whole job.
- Streams go to a **shared filesystem** so one process can diff both runs.
- A traced run is slower — raise `--distributed-timeout-minutes` so the NCCL
  watchdog does not fire on the instrumented iterations.
- The op layer is **incompatible with CUDA-graph capture**. Disable graphs.

`references/tracing-setup.md` covers where the tracer lives, the call-site
lifecycle, cost controls, read-only-container injection, and writing a rung-3
probe.

### 3. Read the first divergence correctly

Three failure modes turn a correct trace into a wrong conclusion. Full versions
in `references/reading-traces.md`:

- **One hop later.** Kernels that bypass the torch dispatcher (TE GEMM and fused
  attention, Triton/FLA, DeepEP/HybridEP dispatch) are invisible to the op
  layer. A divergence born inside one surfaces at the **first ATen op that reads
  its output**. When the first divergent record is an obviously deterministic op
  — a `slice`, a `view`, an elementwise add — with matched inputs, the culprit
  is the uninstrumented producer above it. Add a rung-3 probe there.
- **Probe artifacts.** Uninitialized memory (`aten.empty*`), in-flight async
  collective buffers, and tensors caught mid-DMA (`_to_copy` of a
  `non_blocking=True` host copy) read as divergences and are not.
- **Digest semantics.** An order-independent digest is a strong *screen* for
  value divergence, not a collision-proof key; compare `shape`/`dtype` alongside
  it, and never compare digests across schema versions.

### 4. Turn a site into a mechanism

A named op is not yet a cause. Build a **discriminator table**: hold the site
fixed and vary one axis at a time across instances you already captured —
output dtype, layout, operand dtype, accumulate flag, fused vs unfused path.
The break lives in one cell. Every cell should come from records already in your
streams — no new hypothesis, no new job. Only when the table points below the
dispatcher do you climb to rung 4 and name the kernel.
`references/reading-traces.md` has a worked discriminator table.

### 5. Prove the fix, then report

A fix is proven by a **paired run**, not by one clean run:

- 2 arms with the fix, 2 without, same job, same seed, N iterations, compared on
  full-precision metrics.
- The unfixed pair **must diverge** — otherwise the repro is not live and the
  run proves nothing — and the fixed pair must be bit-identical for all N.
- Say whether the claim is within-allocation or **cross-allocation**; repeating
  work inside one allocation exercises less than the contract promises.
- State the measured cost and whether the fix is **principled** (it removes the
  nondeterministic operation) or **incidental** (it steers off the
  nondeterministic code path). Incidental fixes need a regression test, because
  nothing stops a future heuristic from routing back onto a bad path.

Write down what you **falsified**, not just what you found. Half the value of a
determinism investigation is the list of levers that do not work.

## Worked example

A break in a low-precision weight-gradient path, compressed to show the method.

```
Rung 0   Arms diverge at iteration 6.
Rung 1-3 One ordered stream, iterations 1-2: ~74K records/arm
         (~71K aten + ~2.3K gemm + ~1K grad + ~150 collective).
         The gemm and grad probes were rung-3 additions: the GEMM is a pybind
         call that never reaches the dispatcher, and the fused wgrad writes its
         buffer through a kernel that is equally invisible. Without them the
         trace reports a downstream consumer, not the site.
```

First divergence, in execution order:

```
seq=2547  gemm:NT:acc0:grad1   inputs=MATCH
  in : <low-precision operands + scale buffers>   all SAME
       float32(M,N) accumulator, zeroed           SAME
  out: float32(M,N)   A be0de2f5…  B 804b6328…    DIFFERENT
```

Before it: 2,445 ATen ops, 93 GEMMs, 9 collectives — all bit-identical. Same
result in all six arm pairs. The known `_to_copy` mid-DMA artifact appears sixth
in the divergence list, not first, so it is not confounding the verdict.

Discriminator table, built entirely from records already captured:

```
gemm:TN:acc0:grad0   forward   D=bf16   CLEAN
gemm:NN:acc0:grad1   dgrad     D=bf16   CLEAN
gemm:NT:acc0:grad1   wgrad     D=fp32   DIVERGENT
gemm:NT:acc1:grad1   wgrad     D=fp32   DIVERGENT
```

`accumulate=False` diverges too → accumulation is **not** the mechanism. Operand
dtype is identical in all four rows → not the operands alone. The discriminator
is **fp32 output on the wgrad GEMM**. Zero additional jobs were needed.

Both wrong turns here were hypothesis-first: a probe built to test "it must be
the accumulate epilogue", and a mechanism inferred from "flag X fixes it". The
first-divergence walk falsified each in one command — the first divergent GEMM
had `accumulate=False`, and flag X turned out to change two things at once.

## Anti-patterns

Each of these has cost real days.

- **Designing an experiment around a hypothesis when the stream already has the
  answer.** The first-divergence walk needs no theory and beats one.
- **Inferring a mechanism from "flag X fixes it."** Read what flag X changes.
  It usually changes more than one thing.
- **Reporting a first divergence without checking the inputs matched.**
- **Believing a divergence on a trivially deterministic op.** That is the
  one-hop-later signature of an uninstrumented producer.
- **Trusting a determinism env var without checking scope.** Verify the knob
  reaches the path you care about: grep the symbol in the shipped library and
  read the call site. `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` governs TE's
  *attention backend* selection — do not assume it covers GEMMs.
- **Claiming a kernel name from reasoning.** Name it from two independent
  instruments that agree on the launch count, or say you have not named it.
- **Stopping at "it's nondeterministic."** Push to the ceiling of what is
  observable from outside a closed binary — kernel symbol, algo descriptor,
  problem descriptor, one-flag toggle — then say explicitly that the rest needs
  `ncu` or the owning library team.

## References

- `references/tracing-setup.md` — where the tracer lives, the call-site
  lifecycle, env knobs, cost controls, read-only-container injection, writing a
  rung-3 probe.
- `references/reading-traces.md` — anatomy of a divergence record, blind spots
  and the one-hop rule, known probe artifacts, fingerprint semantics.
- `docs/developer/determinism/op-catalog.md` — the known-unsupported list, for
  the pre-trace check only.

## Related skills

- `mcore-run-on-slurm` — launching the multi-arm repro job.
- `mcore-testing` — turning the verified fix into a regression test.
