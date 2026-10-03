# Inference Optimization Pre-PR Checklist

Copy this into the working notes and walk it before opening the PR. Most items map
to a hard rule in [SKILL.md](../SKILL.md); the rest are review feedback these
commits actually received.

## Decision gate (before writing the code, not before the PR)

```
- [ ] Floor established from MEASURED machine constants, not datasheet peaks
- [ ] Current cost measured under graph replay, at the real decode token count
- [ ] Gross ceiling stated as a ratio AND as a % of the step
- [ ] Net ceiling subtracts what the fix costs (added launches, grid syncs,
      atomics, extra passes)
- [ ] Mechanism identified (bytes / FLOPs / occupancy / CTA utilization /
      dependent load / pure launch overhead) — not just the magnitude
- [ ] Target located on the serial chain vs already-overlapped vs off-path, so the
      expected kernel-to-e2e conversion is predicted before measuring
- [ ] Verdict recorded in the ledger, including gates that came out negative
```

## Measurement

```
- [ ] Baseline captured before any change, with the config recorded
- [ ] Baseline re-run in the SAME allocation as the test arm (cross-session drift
      reached 1.6%, larger than most wins)
- [ ] OFF and ON arms run back to back, and the pair repeated at least once
- [ ] Arms do not overlap: slowest ON beats fastest OFF (report pairwise deltas,
      not one average)
- [ ] First timed iteration checked for cold-start outlier before averaging
- [ ] After-measurement uses the identical batch size, sequence length, parallelism
- [ ] Warmup ran long enough that CUDA-graph capture is excluded from both numbers
- [ ] The win is attributed to a specific kernel or host span, not just end-to-end
- [ ] Measured conversion is consistent with the predicted one; if not, the target
      was mis-located
- [ ] One mechanism per PR, so the measurement attributes cleanly
- [ ] Profiler flags known-good (no osrt, no process-tree sampling, no NVTX under
      graph capture) or the capture will not finalize
```

## Host path

```
- [ ] No .item() / .tolist() / .cpu() added to per-step code
- [ ] No dataclasses.asdict() on any object that can hold a tensor
- [ ] No torch.save / pickle on the IPC path
- [ ] New per-request fields on the wire are scalars, or opt-in behind a flag
- [ ] Work not needed for the next step is off the step loop
- [ ] New host work on the step loop has an NVTX range around it
```

## CUDA graph safety

```
- [ ] Per-step scalars go into preallocated fixed-address GPU tensors via fill_/copy_
- [ ] No new buffer allocation inside a captured region
- [ ] Grow-only buffers pre-sized to the worst case BEFORE capture
- [ ] No Python-level assignment in forward() relied on at replay time (use copy_)
- [ ] Nothing stateful or RNG-bearing newly captured without checking what gets
      frozen by value
- [ ] Per-step work bounded by the matched bucket, not global max_tokens/max_requests
- [ ] Bucket coverage verified: confirm real steps match a graph rather than
      silently falling back to eager
```

## Padding

```
- [ ] Pad rows route to expert -1 (no expert activated)
- [ ] Pad tokens point at reserved dummy storage, not real blocks
- [ ] Kernels skip rows where permutation_map == -1
- [ ] No zeroing pass over rows past valid_tokens (don't write them at all)
- [ ] Idle EP ranks use the lightweight dummy path, never add_request
```

## Triton kernels

```
- [ ] tl.constexpr only on values fixed for the process lifetime
- [ ] Per-step ints either do_not_specialize or passed as a 0-d GPU tensor
- [ ] Grid sized to worst case, body gated by `if pid >= real_count: return`
- [ ] Any autotune goes through autotune_configs(), with a short config list
- [ ] Tile/warp/stage choice driven by a typical-batch hint, not buffer capacity
- [ ] torch.empty rather than torch.zeros where the kernel overwrites every row
```

## MoE specifics

```
- [ ] moe_router_dtype is fp32 (hard requirement for inference_optimized)
- [ ] NVLS eligibility checked via are_tensors_nvls_eligible, not re-derived
- [ ] NCCL fallback path still correct for non-NVLink / non-bf16 / unaligned shapes
- [ ] TE nn.Parameter identity preserved (param.data views, built lazily after
      checkpoint load)
- [ ] Shared-expert overlap re-measured on THIS model; CTA caps not copied forward
- [ ] Unsupported config combinations rejected with a clear error, not silently wrong
```

## Buffer sizing

```
- [ ] Sized from the true per-step bound, not a loose upper bound
- [ ] The bound is a single shared value, not duplicated across modules
- [ ] Scratch reserved before durable pools
- [ ] Over-budget raises a config-time error naming the knob to reduce
- [ ] Assert on overrun rather than trusting the caller
```

## Correctness

```
- [ ] Reference implementation test (plain PyTorch or loop) with assert_close
- [ ] Gating asserted with a sentinel: padded slots still hold the sentinel after
      the kernel runs
- [ ] Sizing formulas re-derived independently in the test, covering BOTH regimes
      of any min()/max()
- [ ] Boundary cases covered (window before position 0, sub-block prefills,
      non-default chunk sizes)
- [ ] Error paths tested (too-small budget raises, not OOM later)
- [ ] Golden values regenerated if bucketization or padding changed, and the
      generated text sanity-checked for coherence
- [ ] Inference and training paths still agree where they should (e.g. router
      picks the same experts)
- [ ] Checked whether a BIT-EXACT formulation exists (hold BLOCK_SIZE_K fixed;
      masked slots adding exact 0.0) before accepting any deviation
- [ ] If not bit-exact: deviation bounded in ulps across several token counts and
      seeds, temperature-0 coherence diffed against gate-OFF, every divergence
      inspected for low-confidence branch + fluent + factually correct, and the
      divergence recorded in the ledger
```

## Shipping

```
- [ ] Kill switch exists so the next person can A/B without reverting
- [ ] New flags documented with defaults in arguments.py and the config docstring
- [ ] Imports run through `uv run isort`
- [ ] Ledger entry appended (hypothesis, changed files/flags, throughput, delta,
      correctness, run id, conclusion) — including for rejected attempts
- [ ] PR opened as a draft, commits signed with both -s and -S
```

## Skill maintenance (after the work, not before the PR)

```
- [ ] Anything that generalizes beyond this model promoted into the skill, routed
      to the right reference rather than piled into SKILL.md
- [ ] Rejections and negative gate verdicts recorded with their root cause
- [ ] Anything this work proved wrong in the skill corrected in place; contradictions
      scoped by hardware/version rather than overwritten
- [ ] Content that is now false deleted; measured negative results kept
- [ ] Links resolve, frontmatter parses, revision log appended
```

See [../references/updating-this-skill.md](../references/updating-this-skill.md).

See `add-inference-unit-tests` for where new tests belong, `mcore-testing` for
recipe structure, and `run-inference-functional-tests` for verification on the
cluster.
