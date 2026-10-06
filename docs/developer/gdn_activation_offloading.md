# BF16/FLA GDN activation offloading

## Contribution scope

The GDN activation-offload item in [roadmap #6757](https://github.com/NVIDIA/Megatron-LM/issues/6757)
targets memory-constrained training. BestJuly confirmed that saved-tensor offloading
would complement the existing recomputation approaches and that either current
`main` or `dev` is an acceptable starting point in
[the follow-up reply](https://github.com/NVIDIA/Megatron-LM/issues/6757#issuecomment-6008766114).
This implementation uses `main` at `d6316eb45`.

The later [PR #7852](https://github.com/NVIDIA/Megatron-LM/pull/7852) implements
input-projection and conv/QKV selective recomputation plus convolution-input offload.
The scope here is deliberately separate: `gdn_core_attn` captures the saves inside
FLA's recurrence, without changing the shared GDN/GDN2 preparation code or adding a
second offload manager. The two PRs touch the same forward, so their integration
should preserve the recurrence boundary when merging.

## Saved-tensor lifetimes

FLA 0.5.1's `ChunkGatedDeltaRuleFunction` saves Q, K, V, cumulative log decay,
beta, and the WY matrix A for backward. Its default path saves beta twice, once as
`beta_raw` and once as `beta`; they reference the same storage. It recomputes other
recurrence intermediates in backward, so offloadable activation size is smaller than
the total forward working set.

A real BF16/FLA forward/backward on an RTX A6000, with batch 1, sequence 2048,
16 recurrence heads and key/value head dimension 128, produced:

| Save | Shape | Type | Tensor bytes (MiB) | Lifetime |
| --- | --- | --- | ---: | --- |
| Q | `[1, 2048, 16, 128]` | BF16 | 8 | Forward to recurrence backward |
| K | `[1, 2048, 16, 128]` | BF16 | 8 | Forward to recurrence backward |
| V | `[1, 2048, 16, 128]` | BF16 | 8 | Forward to recurrence backward |
| Cumulative decay | `[1, 2048, 16]` | FP32 | 0.125 | Forward to recurrence backward |
| beta_raw | `[1, 2048, 16]` | FP32 | 0.125 | Forward to recurrence backward |
| beta (same storage) | `[1, 2048, 16]` | FP32 | 0.125 | Forward to recurrence backward |
| WY A | `[1, 2048, 16, 64]` | BF16 | 4 | Forward to recurrence backward |

The seven slots total **28.375 MiB**; distinct saved storage totals **28.25 MiB**.
Every slot was first unpacked after the synchronized forward had completed. These
are recurrence-only measurements, not a claimed reduction in model peak memory.

In a complete GDN layer, some saves also have owners outside the recurrence. For
example, FLA's L2 norm saves its normalized output, and the sigmoid producing beta
saves beta. The fused pre-GDR path can also return views into a common projection.
The implementation therefore does not resize or forcibly release input storage.
Only allocations whose final GPU references disappear are freed by ordinary tensor
lifetime management. A small tensor-size threshold may copy beta twice without
freeing its upstream save; the existing manager reports duplicate-transfer bytes.

The profiling tool records all saves in the GDN stack and labels saves inside each
recurrence. It reports slot bytes, unique storage bytes, and storage held exclusively
by recurrence saves. Its timestamps are host save/unpack observations, not CUDA
execution timestamps. Profiling hooks are not enabled during timed comparisons.

The four-layer stack used below has 32 recurrence heads after Q/K head expansion.
Its eager profile finds **225 MiB** of storage held exclusively by recurrence saves
(56.25 MiB per layer). Q/K/V use 16 MiB each and WY A uses 8 MiB per layer;
cumulative decay uses 0.25 MiB. The two beta saves share an upstream sigmoid owner
and do not contribute to that 225 MiB. All recurrence slots are first unpacked
after forward. The default 1M-element threshold excludes the small FP32 saves,
leaving 56 MiB per eligible layer. With the last group resident, fraction 0.5
selects two of three eligible groups (112 MiB), and fraction 1 selects three
(168 MiB). Models without Q/K head expansion may retain normalized Q/K upstream
and therefore realize smaller memory savings.

## Offload boundary and policy

The existing `FineGrainedActivationOffloadingInterface` adds the group-start identity
to Q, captures saved tensors only during `gated_delta_rule`, and commits on the raw
recurrence output. Backward crosses the commit before reading FLA's saved tensors;
the group-start backward lets the existing manager prefetch the preceding group.
The output norm and its optional `gdn_norm_out` checkpoint remain outside this scope.

The option is disabled by default. It requires BF16 GDN1 and the FLA recurrence;
configuration rejects GDN2, FP8/FP4, deterministic reference execution, full-layer
recomputation, and CUDA graphs. Eval and no-grad forwards do not register groups.
The existing pool, transfer streams, minimum tensor size, group fraction, and
pipeline lifecycle are reused without modification.

Warmup offloads every group to learn its size. Steady state retains the final group
of each name to avoid a reload stall. Fraction is then applied to the eligible
groups, with integer rounding, rather than to individual tensors or bytes. Report
the actual number of selected groups alongside the requested fraction.

## Reproducing the evidence

Use a CUDA environment with Python 3.12+, PyTorch 2.11/CUDA 13, Transformer Engine
2.20.2 and flash-linear-attention 0.5.1. The repository's default CI container is the
preferred environment. An A6000-compatible container with these public packages can
also run the tool. No model weights or training dataset are needed.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  tools/ssm/gdn_activation_offload.py --mode profile \
  --output /tmp/gdn-profile.json

python -m torch.distributed.run --standalone --nproc-per-node=1 \
  tools/ssm/gdn_activation_offload.py --mode benchmark \
  --seq-length 2048 --fractions 0 0.5 1 \
  --warmup 3 --iterations 10 --output /tmp/gdn-benchmark.json

python -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest -q \
  --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gated_delta_net_offloading.py
```

The default stack has four GDN layers, hidden size 2048, 16 key heads, 32 value
heads, and head dimension 128. Timed steps include forward and backward, with an
explicit synchronization at the forward boundary in both arms. The tool reports
end-of-forward allocation, full-step peak allocated/reserved memory, and all step
durations. It compares outputs, input gradients, and every parameter gradient
bitwise against an independently initialized baseline. Model construction and
warmup are excluded from timing. Optimizer updates, MLPs, and distributed training
communication are not included, so these are GDN-stack results.

Use `TORCH_COMPILE_DISABLE=1` for an explicitly eager reproduction. The tool records
that setting in its JSON. It does not silently disable compilation, Triton autotuning,
or the offload manager's warmup policy. Avoid active GPU sharing for runtime claims;
record repeat runs and their raw step samples when dedicated GPUs are unavailable.

## A6000 memory/runtime measurements

Measurements on 2026-10-06 used the default compilation setting, the stack above,
batch size 1, seed 123, the 1M-element threshold, three warmup steps and ten measured
forward/backward steps per arm. Each sequence length has three independent process
runs. The GPU was not exclusive: another process held 23,304 MiB, with no concurrent
compute observed during idle checks. The host was shared and clocks were not locked.
These observations establish allocated-memory savings; runtime estimates require
confirmation on dedicated hardware. The default TE path also has native-op Dynamo
graph breaks; this is not a claim of full-graph compilation.

The time column is the median of the three run medians, followed by their range.
The change column is the median of the three within-run changes relative to that
run's disabled baseline, followed by their range. It is not the ratio of aggregated
time medians. Peak allocated memory was stable across all 30 measured steps per arm.

| Sequence | Requested fraction | Selected groups | Peak allocated (MiB) | Step time (ms), median [range] | Paired runtime change, median [range] |
| ---: | --- | ---: | ---: | --- | --- |
| 2048 | Disabled | 0 | 1274.80 | 31.67 [30.74, 32.48] | Reference |
| 2048 | 0 | 0 | 1274.80 | 31.77 [31.41, 38.50] | +2.2% [-2.2%, +21.6%] |
| 2048 | 0.5 | 2 | 1162.80 | 43.49 [36.61, 44.65] | +37.3% [+12.7%, +45.3%] |
| 2048 | 1 | 3 | 1106.80 | 41.85 [41.56, 59.44] | +36.1% [+27.9%, +87.7%] |
| 4096 | Disabled | 0 | 2272.34 | 58.19 [57.87, 58.32] | Reference |
| 4096 | 0 | 0 | 2272.34 | 58.28 [58.06, 58.99] | +0.3% [+0.2%, +1.2%] |
| 4096 | 0.5 | 2 | 2048.34 | 69.18 [68.72, 86.45] | +18.9% [+18.7%, +48.2%] |
| 4096 | 1 | 3 | 1936.34 | 83.97 [82.63, 110.27] | +44.3% [+42.8%, +89.1%] |

Fraction 1 reduces peak allocation by approximately **168 MiB (13.2%)** at sequence
2048 and **336 MiB (14.8%)** at sequence 4096. End-of-forward allocations decrease
by the same amounts. Fraction 0 adds only 512 allocated bytes, with no steady-state
transfers. Every enabled arm passed bitwise comparisons of outputs, input gradients
and all parameter gradients against its disabled baseline. Timing variance is large
even in the fraction-zero control, so these data do not establish a reliable ordering
between fractions 0.5 and 1 or a production-training overhead.

[Raw A6000 step samples](gdn_activation_offloading_a6000.csv) include all 240 measured
steps, peak allocated/reserved memory, end-of-forward allocation, selected groups,
selected transfer bytes, and each arm's correctness result. The correctness flag is
the final-step snapshot comparison for that arm; replay unit tests separately check
warmup and successive steady-state iterations.

## Acceptance and next steps

On one RTX A6000 with Python 3.12.4, PyTorch 2.11.0+cu130, Transformer Engine
2.20.2, and FLA 0.5.1, all **21 tests passed** both with `TORCH_COMPILE_DISABLE=1`
and with the default compilation setting. This includes exact output, input-gradient
and parameter-gradient comparisons over one warmup and two steady-state iterations,
fraction 0/0.5/1, threshold skipping, `gdn_norm_out` recomputation, packed sequences,
and eval/no-grad bypass. Pinned-buffer usage returned to zero after backward.

Existing single-rank GDN and output-norm recomputation tests passed **7 cases**;
one fused causal-conv1d case was skipped because its native backward extension is
not installed. Existing TransformerConfig tests passed **53 cases**. Data-free SSM
runs used `--confcutdir=tests/unit_tests/ssm` to omit root dataset-download fixtures;
configuration-only regression tests used `--noconftest`.

`tools/autoformat.sh` passes its Black, isort, Pylint and Ruff gates, and kernel
determinism coverage passes. Its non-blocking mypy step reports six existing
diagnostics; checking the original `main` versions reproduces the same six.

The measured first slice meets these acceptance criteria: outputs and all gradients
agree through warmup and multiple steady iterations, pinned buffers return to the
pool, fraction zero and threshold skipping work, packed metadata and output-norm
recomputation remain correct, and multi-layer A6000 comparisons show reduced
allocated memory with an explicit runtime cost. A speedup is not required for a
memory option. This does not replace the upstream CI or full-model qualification.

Before expanding support, review the option name and its relationship to #7852.
The proposed follow-up order is:

1. Confirm the scope/name and integration boundary with #7852. Keep recurrence saves
   independently selectable when combining projection/QKV recomputation and offload.
2. Run complete BF16 Qwen training on dedicated A6000 hardware, including optimizer
   updates and representative sequence lengths. Compare loss, gradients, full-step
   peak allocation, and throughput, with and without output-norm recomputation.
3. Qualify TP/SP/CP and PP/VPP schedules with multiple microbatches, then fused
   pre-GDR. Require matching outputs/gradients and clean manager/pool lifecycle before
   advertising those combinations.
4. Treat GDN2, quantized training and CUDA graphs as later extensions with separate
   implementation and acceptance evidence.

Each expansion needs its own correctness and memory/runtime measurements; reuse of
the manager alone does not establish support.
