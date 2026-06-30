<!---
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
-->

# Verify restart first-iteration optimizations

A shorter printed iteration 1 is not sufficient evidence of a restart win.
Initialization or model construction can absorb the same work before the
training timer starts. Validate all restart optimizations with a same-allocation
control/treatment/control run and report these four measurements separately:

1. `time to initialize megatron`;
2. `before library-setup` through `before the start of training step`;
3. printed iteration-1 elapsed time;
4. the launcher's complete `srun` wall time, including teardown.

The acceptance metric is `pre-training + iteration 1`. It must improve by a
declared practical threshold, and the complete `srun` wall must not regress.
Loss and gradient fields must match the controls exactly, with zero skipped or
NaN iterations. Performance runs must also state node/rank count and disable
profilers unless profiling is the explicit purpose of the run. Use
`--minimum-startup-through-iteration-1-improvement-s` to reject an iteration
timer reduction that merely moved the same cost into initialization. Also set
`--minimum-srun-wall-improvement-s` when work could move before the first
startup marker (for example, cache extraction or process bootstrap). Choose
both thresholds before examining the treatment.

The treatment's printed iteration 1 must also be no slower than the control
mean. Use `--minimum-iteration-1-improvement-s` when an experiment needs a
nonzero practical-significance threshold in addition to the end-to-end gates.

The controls must be close enough for causal attribution. By default the
verifier rejects an ABA when Control A versus Control B differs by more than
1 second in iteration 1, 5 seconds in startup through iteration 1, or 10 seconds
in complete `srun` wall. A large first-leg allocation-priming penalty can
otherwise make a treatment in the middle look faster even when it is not.
Override the `--max-control-*-spread-s` options only when the experiment defines
and documents a different stability budget.

When a cluster has a repeatable first-launch allocation penalty, the launcher
may run one complete, unmeasured control before the measured ABA. Mark it with
`[ITER1-ENV-PRIME]`, keep its configuration identical to the measured controls,
and do not use it as a performance baseline. The verifier deliberately reads
only `[ITER1-ENV-ABA]` legs, so the measured control/treatment/control runs must
still pass iteration-1, startup-through-iteration-1, and complete-wall gates.

`tools/first_iter/verify_restart.py` enforces these gates for launcher logs that
contain the standard `[ITER1-ENV-ABA]` and `[ULTRAHALF-ITER1CAP]` markers:

```bash
python tools/first_iter/verify_restart.py \
  --aba-log /path/to/ultrahalf_env_aba_JOB.out \
  --treatment-label nccl_topo_cache \
  --expected-nodes 16 \
  --expected-world 64 \
  --minimum-startup-through-iteration-1-improvement-s 0.5 \
  --minimum-srun-wall-improvement-s 0.5
```

For a 20-iteration numerical gate, add `--require-iteration-20`. Treat this as
mandatory before accepting changes to communicator creation, process-group
reuse, topology or graph selection, collective ordering, or compiled-kernel
selection. A short run is only a screening gate for those changes: later
iterations can expose numerical drift, deadlocks, or end-to-end regressions
that are invisible at iteration 1. The command exits nonzero if the apparent
first-iteration gain was relocated into startup, if full wall time regressed,
if the controls are too far apart, if scale differs, or if printed numerics
differ.

For settings that can change steady-state throughput, also gate an inclusive
iteration range. For example, this rejects more than a 2% median iteration-time
regression over iterations 10 through 20:

```bash
  --steady-state-start-iteration 10 \
  --steady-state-end-iteration 20 \
  --maximum-steady-state-regression-percent 2
```

The verifier compares the treatment median with the mean of the two control
medians and fails if any iteration in the requested range is missing.

When the experiment has an absolute service-level target, add, for example,
`--maximum-treatment-iteration-1-s 5`. This is an additional gate: it does not
replace the startup-through-iteration-1 or complete-wall checks, so moving work
before the iteration timer still fails.

Cluster-specific launch commands, cache implementation details, and preserved
HSG evidence belong in the companion first-iteration tooling repository rather
than Megatron Core production code.
