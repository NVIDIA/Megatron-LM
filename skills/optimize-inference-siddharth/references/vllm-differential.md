# Differential Analysis Against vLLM

When the goal is stated as "match vLLM," the competitor's trace is not just a
scoreboard — it is a specification. It tells you what the same model on the same
hardware costs when someone else built the decode path, which is a far more useful
target than a roofline, because it is known to be achievable.

Everything else in this skill looks inward at Megatron. This document is the one
workflow that looks at both.

Source: the Qwen3-30B-A3B EP4 campaign, where a single kernel-level differential
simultaneously justified the fusion work and killed the grouped-GEMM work.

## The one question worth asking

Given matched configs and matched trace windows:

> **Is any individual kernel slower in mcore, or does mcore just launch more
> kernels to do the same thing?**

These have opposite fixes, and guessing wrong is expensive. Slower kernels mean
tiling, occupancy, or implementation work. More kernels mean fusion, graph-node
removal, and host-dispatch reduction. A third answer — mcore has *categories the
competitor does not have at all* — means the structure differs and no amount of
kernel work will close it.

On Qwen3-30B the answer was unambiguous, and it was the second one. One forward
block: **~467 kernels in vLLM against ~1784 in mcore**, roughly 3.8×, for a step
of **4.4 ms against 9.5 ms**. Individual kernels were broadly competitive; mcore
was running three to four times as many of them, most on the serial per-layer
dependency chain.

## Getting comparable windows

The comparison is worthless unless both windows cover the same work in the same
regime. Two requirements beyond the usual matched-config discipline:

**Both must be steady-state decode.** No prefill, no graph capture, no warmup.
Verify by checking that the window's kernel sequence repeats with a stable period.

**Anchor on a once-per-block kernel, not on wall time.** Pick a kernel that fires
exactly once per transformer block with a stable name — the block's leading norm
is the natural choice — and take the window from one occurrence to the next
occurrence of the *closing* kernel. That gives exactly one forward block on each
side, independent of clock offsets between the two runs.

For Qwen the anchors were:

| Engine | Window start | Window end |
|---|---|---|
| vLLM | `triton_red_fused__to_copy_embedding_rms_norm_0` | `triton_red_fused__to_copy_add_mean_moe_forward_mul_pow_rsqrt_0` |
| mcore | `rmsnorm_fwd_tuned_kernel<...>` | `triton_poi_fused_add_copy__0` |

Extract the ordered kernel list per window from the `.sqlite` export — kernel
rows joined to the string table, filtered to one device and the window, ordered by
start time. Keep start, duration, and name; you need the order, not just the
totals. `compare_budget.py` in [scripts/](../scripts/) does the whole comparison,
taxonomy included.

**And match the sequence length, not just the config.** Decode cost grows with KV
length, so "the last N steps" of two runs can sit at different average seqlen and then
every bucket differs at once, by up to 27% on one pair here. Cross-engine windows are
especially exposed to this because the two engines reach a given step at different times.
Sanity-check with the buckets your change cannot touch, and prefer launch counts, which
are seqlen-invariant — full method in *Two traces of the same config are not the same
workload* in [measuring.md](measuring.md).

## Decompose into three buckets

Assign every kernel in both windows to a *role* — norm, QKV GEMM, attention,
router, dispatch, expert GEMM, combine, residual, sampling — and then compare
role by role. Do **not** match on kernel names: Inductor-generated Triton names
differ between engines and encode fusion structure rather than function, which is
exactly the thing you are trying to measure.

Then split the total gap three ways:

**Same role, more kernels.** The fusion signal. vLLM runs one
`triton_red_fused__to_copy_add_..._rms_norm` where mcore runs
`triton_poi_fused_add_copy` followed by `rmsnorm_fwd_tuned` — two launches, two
graph nodes, per boundary, per layer. That observation, repeated across the norm
and residual roles, is what produced the fused QK-norm and fused add-norm
optimizations, worth **+2.9%** and **+1.37%** respectively.

**Same role, slower kernel.** The implementation signal. Rank by
`(mcore_time - competitor_time)` for the role, not by mcore time alone. When the two
sides show **the same launch count and different time**, do not go tune your kernel yet
— first find out *which library the competitor is calling*, then check whether it is
already installed. See below.

**Same total work, different packing.** The scheduling signal, and the one the other
three miss entirely. Compare sum-of-durations ÷ union for both engines: mcore measured
**1.003-1.008** against vLLM's **1.050**, i.e. vLLM hides ~5% of its device work behind
other work and mcore hides essentially none. Worth ~0.3 ms/step here, and
it does not appear in any per-bucket comparison, because the buckets are identical — it
is the same work, packed differently. Note that a wide CUDA-graph capture bounds what
you can do about it: overlap becomes a property of the captured graph's structure rather
than of stream priorities or `CUDA_DEVICE_MAX_CONNECTIONS`, so the fix is a capture-time
structural change, not a runtime knob. See *Pick a clean window* in
[measuring.md](measuring.md) for the metric and
[cuda-graphs.md](cuda-graphs.md) for what the capture forecloses.

**Role present on one side only.** The structural signal, and usually the largest.
mcore's decode carried **exposed NVLS AllGather-V dispatch and ReduceScatter-V
combine on the per-layer critical path**; vLLM, using a TRT-LLM fused MoE, had
**zero exposed communication** — dispatch, grouped GEMM, activation, and finalize
were one fused unit. No fusion or tile change on mcore's side reaches that; it is
a different decomposition of the same math.

## Read the competitor's fusion boundaries as the target

The most directly actionable output is the competitor's *fusion inventory*: which
adjacent operations it has merged that you have not. For Qwen decode:

| Operation | vLLM | mcore (before) |
|---|---|---|
| Residual add + RMSNorm | one kernel | two kernels |
| Router softmax + top-k | one kernel | four kernels |
| Routing metadata / indirection table | folded in | up to five kernels |
| Dispatch + expert GEMM + activation + finalize | one fused MoE | six-plus kernels with exposed comm between |

Each row is a candidate with a known-achievable target. Price them with
[decision-gates.md](decision-gates.md) before building — the routing chain
collapsed from five kernels to one across four separate experiments, but the
full MoE mega-fusion was attempted, measured **0.68-0.80×**, and rejected. That
the competitor fuses something does not prove *your* fusion of it will be faster;
it proves the fusion is legal and bounds what it is worth.

## First ask which library the competitor is calling, and whether you already have it

Before tuning anything, resolve the competitor's kernel to a **package and an entry
point**. Kernel names in the trace are usually enough (`fmhaSm100...`, `nvjet...`,
`trtllm_gen...`), and `pip list` in the competitor's environment finishes it. Then check
your own environment: on a shared image the answer is frequently that the faster kernel
is *already installed* and you are calling a different one.

Attention was the clearest case. The differential showed the same launch count on both
sides with mcore **0.409 ms/step slower** — the largest remaining work-bucket deficit,
and unambiguously an implementation gap rather than a fusion gap. Substantial effort had
already gone into flag-flipping between FA2 and FA4 *inside flash-attn*, which is the
wrong package: vLLM was calling flashinfer's `trtllm-gen` Blackwell decode kernel, and
flashinfer was already in the same venv. A microbenchmark at mcore's exact shapes put it
**~24% under FA2**, it accepted mcore's existing paged KV layout unchanged, and wiring it
in behind `MCORE_FLASHINFER_DECODE` was worth **+2.6% end-to-end**. Enabling its
Programmatic Dependent Launch added a further +0.3%.

The generalizable order is: identify the package, check whether it is installed,
microbenchmark it at *your* shapes under graph replay, verify the memory layout it
demands, and only then consider tuning your own. Steps 1 and 2 take minutes; the effort
spent skipping them did not.

Step 1 does not have to be guesswork. The vLLM source that produced the baseline
is checked out locally, and the `vllm-codebase-expert` subagent resolves a trace
kernel name to its call site, package, and gating flag against that exact
revision — see [vllm-codebase-reference](../../vllm-codebase-reference/SKILL.md).
Ask it rather than inferring the call site from the kernel name; symbol names
identify the library but not which of vLLM's several backends selected it, and
that distinction is the whole of step 2.

An important asymmetry: this works for **leaf kernels with a narrow contract**
(attention, norms, sampling) and fails for **fused mega-kernels**, which come with layout
and weight-preparation demands — the two traps below.

## Do not conclude the competitor's kernel is better without checking why

Two traps, both hit during this campaign.

**The competitor may win by weight layout, not by kernel quality.** The TRT-LLM
BF16 fused MoE that vLLM uses was probed directly as a drop-in backend and turned
out to require **4-D pre-shuffled block-major weights**; mcore stores 3-D
`[E, 2*ffn, H]`. `weight_layout=MajorK` is rejected outright and `BlockMajorK`
indexes `size(3)`. The kernel was never the blocker — the weight preparation
pipeline was.

**An existing backend may be mis-wired rather than slow.** The already-shipped
flashinfer `cutlass_fused_moe` path was documented as "blocked for SwiGLU." It is
not: the kernel supports BF16 gated SwiGLU, and mcore's wiring was broken in two
independent ways — an `ActivationType` mis-map that hard-fails, and a gate/up
ordering mismatch that *silently corrupts numerics*. Both were root-caused and
fixed in a harness. Only then was the comparison meaningful, and the answer was
**90.81 µs against 83.63 µs** for the retuned Triton path, i.e. 8% slower. The
ledger note had been wrong for the wrong reason, and the correct measurement still
rejected the backend.

**Generalizable:** before adopting a competitor's kernel, verify what it demands
of the weight layout and whether your existing wiring to it is actually correct.
A hard failure and a silent numerics corruption look nothing alike in a profile.

## What a differential cannot tell you

It gives you counts and durations, not the critical path. A role with more kernels
in mcore may still cost nothing if those kernels sit on a side stream and overlap
with compute. Convert the differential's findings to wall-time impact with
union-busy analysis and the serial-chain reasoning in
[measuring.md](measuring.md) before ranking them.

It also will not show host-side differences. Two engines with identical kernel
sequences can differ substantially in Python overhead between steps, which on this
workload was 18.4% of wall time.
