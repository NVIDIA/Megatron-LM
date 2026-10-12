# Complete-model GDN offload validation

The [initial offload measurements](gdn_activation_offloading.md) exercise GDN
stacks. `tests/unit_tests/ssm/test_gdn_offload_training.py` adds a reduced-size
language-model comparison through the native training schedules. GPU results
are listed below. Configuration checks on CPU do not qualify a parallel topology.

## GPU validation status

The 2026-10-06/07/08/10/11 runs use RTX A6000 GPUs, PyTorch 2.11.0+cu130, Transformer
Engine 2.20.2, FLA 0.5.1 and cuDNN 9.19.0. These public-package container runs
do not replace upstream CI.

The GDN recurrence suite passes 65 cases in both eager mode and with default
compilation. It covers dense and packed inputs with fused pre-GDR disabled/enabled
and output-norm recomputation disabled/enabled. Fused runs use causal-conv1d 1.6.1
with its deterministic convolution-weight-gradient reduction. Gate-parameter
backward now uses fixed tiles and an ordered FP32 reduction, as described below.
The existing pre-GDR fusion suite passes all 11 cases. A new kernel test passes
16 cases with byte-exact FP32 gradient replays under side-stream contention.
Fused and unfused offload comparisons use separate disabled baselines; they do
not require the two pre-GDR implementations to produce identical bits.

The 2026-10-10 review expands the evaluation-switching coverage to both pre-GDR
variants, before and after training warmup. The expanded recurrence suite passes
71 cases with default compilation; the same run also passes all 16 gate-backward
replay cases and 14 shared-manager cases. The single-rank invocation skips one
multi-rank summary-aggregation case. Ten separate argument-boundary cases pass;
those parser tests do not qualify a GPU topology.

On 2026-10-11, all 12 expanded evaluation-switching cases pass in eager mode on
each of four ranks, together with the shared-manager summary-aggregation case:
13 passed and zero skipped per rank, launcher exit code 0. This invocation sets
`TORCH_COMPILE_DISABLE=1`; it reruns the expanded lifecycle cases, rather than
the entire 71-case recurrence suite.

The previously skipped shared-manager aggregation regression and existing GDN
fused CP-local parameter test pass on both ranks of a two-rank run. The latter
checks parameter partitioning; the reduced-model training qualification is separate
from pretrained CP, which remains unqualified.

| Workload | Result |
| --- | --- |
| Reduced model, one rank | 12 passed, 84 topology cases skipped; fused/unfused |
| Reduced model, two ranks: DP, TP, TP+SP | 36 passed on each rank; fused/unfused |
| Reduced model, two ranks: PP | 12 passed on each rank; fused/unfused |
| Reduced model, PP+VPP | 12 passed on each rank with native unbatched P2P; fused/unfused |
| Reduced model, CP | 12 passed on each rank with FlashAttention 2.8.3; fused/unfused |
| Reduced model, four ranks | 96 passed, zero skipped on each rank; all eight topologies, including TP+SP+CP and TP+SP+PP, with fused/unfused pre-GDR |
| Pretrained Qwen3.5-0.8B, one rank | All 8 comparison arms passed; 24 optimizer updates |
| Pretrained Qwen, fused pre-GDR on one rank | All 8 arms passed, also across norm recomputation settings; 24 optimizer updates; every pre-reset pool count is zero |
| Pretrained Qwen, DP on two ranks | All 8 arms passed on each rank; replicas have distinct losses and identical reduced gradients, weights and Adam states |
| Pretrained Qwen, TP on two ranks | All 8 arms passed on each rank, including across norm recomputation settings |
| Pretrained Qwen, TP+SP on two ranks | All 8 arms passed on each rank, including across norm recomputation settings |
| Pretrained Qwen, PP on two ranks | All 8 arms passed on each rank, including across norm recomputation settings |
| Pretrained Qwen, PP+VPP on two ranks | All 8 arms passed on each rank with native unbatched P2P, including across norm recomputation settings |
| Pretrained Qwen, fused pre-GDR on two ranks | DP, TP, TP+SP, PP and PP+VPP: all 8 arms per topology passed on each rank; every pre-reset pool count is zero |
| Pretrained Qwen, unfused TP+SP+PP on four ranks | All 8 arms passed, including across norm settings; 32 rank files and 96 exact complete rank-step states; every pre-reset pool count is zero |
| Pretrained Qwen, fused TP+SP+PP on four ranks | All 8 arms passed, including across norm settings; 32 rank files and 96 exact complete rank-step states; fresh disabled control also matches; every pre-reset pool count is zero |
| Pretrained Qwen, CP on two ranks | Disabled baseline cannot select an attention backend for head dimension 256 on A6000; unqualified |
| Pretrained Qwen, TP+SP+CP on four ranks | Disabled baseline executed on all four ranks; each fails native attention backend selection at head dimension 256; unqualified |
| Pretrained Qwen, unfused pre-GDR at sequence lengths 2048/4096 | Baseline and fraction 1 passed with norm recomputation off/on; all 3 step states also match across recomputation settings |
| Pretrained Qwen, fused pre-GDR at sequence lengths 2048/4096 | Disabled/fraction-1 with norm recomputation off/on: all 24 complete step states match, including across norm settings |
| Pretrained Qwen, unfused pre-GDR memory/runtime | All 48 processes completed; 480 measured samples across 2 lengths × 2 norm settings × 4 arms × 3 repeats |
| Pretrained Qwen, unfused pre-GDR DP=2 memory/runtime | Disabled/full offload, sequence 2048, three repeats; 60 measured world steps / 120 rank samples |
| Pretrained Qwen, fixed fused pre-GDR memory/runtime | All 48 fresh processes completed; 480 measured samples across 2 lengths × 2 norm settings × 4 arms × 3 repeats |
| Pretrained Qwen, fixed fused pre-GDR DP=2 memory/runtime | Disabled/full offload, sequence 2048, three repeats; 60 measured world steps / 120 rank samples |

The single-rank Qwen comparison uses all 24 text decoder layers, hidden size 1024 and
vocabulary size 248320 from the original checkpoint. Each arm starts from the
same local Hugging Face weights and consumes the same 128 real text conversations
(30,727 rendered tokens). Sequence length is 128, with four microbatches per
optimizer step, one warmup step and two steady steps. The short-sequence
checks lower the minimum tensor size to 1024 elements to exercise transfers.
All eight arms (disabled/0/0.5/1 × output-norm recomputation off/on) match exactly, including
across the recomputation settings: losses, numeric gradient norms, all 230
parameter-gradient and updated-weight fingerprints, and complete Adam state.
The manager warmup completes in enabled arms. Fraction 0 selects no transfer bytes; fractions 0.5 and 1
select 66,945,024 and 132,030,464 bytes per full iteration, respectively.
At lengths 2048 and 4096, separate disabled/fraction-1 correctness runs use the
default 1,048,576-element threshold and match every step state with norm
recomputation off/on. Fraction 1 selects 2,084,569,088 and 4,169,138,176 bytes
per full iteration, respectively.

Checkpoint revision: `eb706f593d2d43c90a10271199c10b07ced7569a`.
The single safetensors shard has SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`;
the prepared messages JSONL has SHA-256
`94f22b43be94403da5519036c05a40575bb1eed698606c0a7323fa14736ad1f5`.
Data contents and model weights are not included in the repository.

The earlier reduced-model two-rank suite passes all 72 applicable cases on each
rank, with 24 four-rank cases skipped. The 2026-10-08 four-rank run completes all
96 cases on each rank with zero skips and launcher exit code 0. It covers
fractions 0/0.5/1 and norm recomputation off/on for both pre-GDR variants in DP,
TP, TP+SP, CP, PP, PP+VPP, TP+SP+CP and TP+SP+PP. Four ranks also exercise DP=4
in the DP case and DP=2 alongside each topology with one two-way parallel factor.
The reduced model's CP attention uses head dimension 64.
The pretrained Qwen CP failure occurs in the disabled baseline with head
dimension 256. TE 2.20.2 disables FlashAttention 2 for head dimensions above 192
on SM86, and its fused CP backend rejects this head dimension. The unfused
backend does not support CP. Installing FlashAttention 2.8.3 therefore qualifies
reduced-model CP but leaves pretrained Qwen CP unqualified. These runs do not
change external backend checks or relax the comparison's determinism setting.
Native backend queries also reject all nine combinations of `p2p`, `all_gather`
and `a2a` communication with sequence lengths 128, 2048 and 4096 at the original
256-dimensional heads on this A6000 environment.
The 2026-10-11 complete pretrained TP=2/SP/CP=2 disabled baseline reproduces
this failure on all four ranks at sequence length 128. Its native attention uses
four heads and one GQA group per TP rank, with the original head dimension 256.
Every rank reports all three backends unavailable and raises
`No dot product attention backend is available for the provided inputs`.
No optimizer step completes, so this run does not qualify pretrained CP.

The reduced-model suite checks pinned-buffer ownership immediately before every
native schedule reset. A deliberate omission of one pool return fails this check;
the corresponding ordinary training case passes. Checking only after reset would
miss that omission because reset clears the usage counters. The Qwen tool's
`--check-state` path now checks the pool in native gradient finalization, before
the schedule resets it. Earlier pretrained and timed runs checked counters after
reset; their loss/state and memory/runtime results remain valid, but that check
alone does not establish buffer-return correctness.
All eight single-rank fused arms record zero ownership and match each other across
output-norm recomputation settings. The original unfused and single-rank fused
eight-arm qualification runs precede the pooled-buffer reuse fix. At code commit
`4e18562c6bcf894795d23378a8f1699c79e4b797`, fraction-1 replays use norm recomputation
for each qualified unfused topology and both norm settings for single-rank fusion.
They compare all three complete step states against the corresponding original
run and check ownership before every reset. The unfused reduced-model fraction/norm
matrix also passed at this commit. Expanded fusion coverage uses the same production
runtime files.

The earlier complete two-rank fused comparisons ran at
`b3434091b84235548945e3f8f7989c2ddb19137f`, with production runtime files identical
to `4e18562c6bcf894795d23378a8f1699c79e4b797`. DP, TP, TP+SP, PP and PP+VPP each
pass all eight arms, including across norm recomputation settings: 40 fresh
two-rank runs, 80 rank-result files and 240 complete step states. Each topology
uses its own fused, norm-recomputation-disabled baseline. Every rank matches its
baseline's losses, gradient norms, gradients, updated weights and complete Adam
state. All native resets record zero pinned-buffer ownership before clearing
counters. Disabled and fraction-zero arms select no steady transfer bytes;
fractions 0.5 and 1 select positive bytes on both ranks. DP replicas have distinct
losses and identical reduced gradients, updated weights and Adam states. These
are correctness runs; they contain no performance samples.

Before the gate-backward fix, fused sequence-2048 comparisons failed the bit-exact
full-state check with norm recomputation off or on. Losses, gradient norms and
all updated weights matched for all three steps, but a few `A_log`/`dt_bias`
gradients and their Adam moments differed. Fresh repeats of the disabled baseline
on the same GPU reproduced differences in those parameters without activation
offloading. Other parameter gradients matched exactly in these comparisons, and
all pre-reset pool counts were zero.
The old `_g_beta_backward_kernel` atomically added FP32 partial sums into shared
parameter-gradient buffers. Program scheduling changed their addition order;
backward autotuning could also change the partition between fresh processes.
`CAUSAL_CONV1D_DETERMINISTIC=1` controls convolution backward, not these reductions.

At code commit `2f1ee4f2e3dd6d909d08981338f5879632131139`, gate backward instead
uses fixed 128-token by 16-head tiles. Each program writes its FP32 parameter
partials to its own slot, followed by one Torch sum over batch and sequence blocks
on the same stream. The temporary buffer has
`2 * batch * ceil(sequence_length / 128) * value_heads` FP32 elements (4 KiB for
batch 1, sequence 4096 and 16 heads). This follows the existing gated-norm reduction
pattern and adds no new configuration option or offload component.
An isolated A6000 gate-backward measurement compares the old atomic helper with
the fixed helper using a preallocated projection-gradient buffer. It covers batch
1, sequences 128/2048/4096, 16 or 8 value heads and head dimension 128, with three
alternating trials after autotuning. The ordered helper takes approximately
2–3 microseconds longer per call. CUDA-event medians are quantized at this scale;
complete training-step measurements below remain the offload trade-off evidence.

The new replay test reproduced parameter-gradient differences with the old kernel
before passing all 16 cases with the fix. It compares all returned gradients byte
for byte, including FP32 parameter gradients before their BF16 cast, over five
replays with side-stream contention. Cases cover BF16/FP32, multiple batches,
strided upstream gradients, an explicit side stream, sequence lengths 2048/4096
and partial sequence/head tiles. A separate Torch-autograd numerical reference
checks accuracy; its tolerance does not weaken replay or training-state equality.

Eight fresh pretrained fused runs at the fixed commit compare disabled/full
offload at sequences 2048/4096 with norm recomputation off/on. All 24 complete step
states match, including across norm settings, and every pre-reset pool count is
zero. The default 1,048,576-element threshold selects 2,084,569,088 and
4,169,138,176 transfer bytes per full iteration, respectively. The earlier failed
artifacts are preserved separately from these successful runs.
Four additional fresh disabled-baseline repeats, one per sequence/norm setting
on the same device, also match every complete state. In total, 12 independent
processes produce 36 exact step states; gradient and Adam equality is required,
including for `A_log` and `dt_bias`.
Full pre-GDR fusion still rejects `deterministic_mode=True`: other native kernels
autotune, and convolution backward requires its external determinism setting.
These results qualify the recorded configuration rather than all FLA/fusion
implementations or hardware.
Changing the reduction tree can change rounded gradients relative to the old
atomic implementation. A short DP replay against the old source differed in one
first-step `A_log` gradient and Adam state while losses, gradient norms and updated
weights matched. Qualification therefore uses disabled baselines from the same
fixed implementation, rather than requiring identical bits across the old and new
reduction algorithms.

Eight fresh single-rank runs with the fixed gate kernel also compare all four
offload arms with norm recomputation off/on. All 24 complete states match the
norm-disabled baseline, and every pre-reset pool count is zero. They use sequence
128 and a 1024-element threshold to exercise transfers.

The fixed implementation also reruns all eight arms for DP, TP, TP+SP, PP and
PP+VPP against fresh disabled baselines from the same runtime source: 40 fresh
two-rank runs, 80 rank-result files and 240 complete rank-step states. All losses,
gradient norms, parameter gradients, updated weights and complete Adam states
match each rank's norm-disabled baseline, including across norm recomputation
settings. Every pre-reset pool count is zero. Disabled/fraction-zero select no
steady transfers; positive fractions select transfers on both ranks. DP replicas
use distinct data and have identical reduced gradients, weights and Adam states.
The runtime source is `2f1ee4f2e3dd6d909d08981338f5879632131139`; the later kernel
registry commit changes test metadata only. An independent artifact validator
checks all 80 two-rank files, eight single-rank files and 12 long-sequence files,
and records their hashes.

The shared-manager changes also pass all nine existing GPT/MoE CUDA Graph offload
cases on one A6000. These cover attention/MLA and expert groups, fractions 0/0.5/1
and delayed/non-delayed offload. This is regression coverage for the existing
graph path; GDN offloading still rejects CUDA graphs at configuration time.

The four-rank reduced-model run uses physical A6000 GPUs 1/3/4/5 at checkout
`5176d49bb0bdd885f6f69575fa8c39295c22b2bb`, with the same production runtime and
test source as `2f1ee4f2e3dd6d909d08981338f5879632131139`. Each rank passes all
96 cases, including the 24 TP+SP+CP and TP+SP+PP cases that required four ranks.
Independent inspection of every rank log confirms 12 cases per topology and
zero skips. The launcher returns 0 after explicit process-group cleanup.
These are correctness comparisons, with all pre-reset pool counts zero; they
provide no four-rank memory/runtime measurements. Other jobs began using the
selected GPUs during the run, and the pretrained four-rank comparisons could
not start in that session after the reduced-model run released its devices.

The 2026-10-10 pretrained TP+SP+PP comparison uses physical GPUs 0/1/2/3 at
checkout `c8ceb3b8a7067827aa6246affa8413c6bf04b9cc`, with TP=2, SP enabled and
PP=2. It uses the same complete checkpoint, dataset, short sequence and optimizer
settings described above. All eight unfused arms pass, including comparisons
across norm settings: 32 rank files contain 96 byte-identical complete rank-step
states. The fused norm-off disabled/0/0.5 arms also pass: 12 rank files and 36
complete states. Positive fractions select transfers on every rank, fraction
zero selects none, and all completed runs have zero pinned buffers in use before
each schedule reset. Each fusion variant uses its own fresh disabled baseline.

GPU work was paused at the user's request during the fused norm-off fraction-1
arm, then resumed on 2026-10-11 using idle physical GPUs 0/1/3/4. A fresh fused
norm-off disabled baseline matches the prior baseline exactly on every rank.
The five pending arms then pass against that fresh control: norm-off fraction 1
and norm-on disabled/0/0.5/1. Independent verification of the completed matrix
confirms all eight arms for each fusion variant: 64 rank files and 192 complete
rank-step states. The additional disabled control contributes four files and
12 matching states, for 17 successful four-rank runs and 204 states overall.
Every completed run has zero pinned-buffer ownership before each schedule reset.
Fraction zero selects no transfers; positive fractions select transfers on all
ranks. These correctness runs produce no timed samples and establish no
four-rank throughput result.

The checkout adds two input checks to the Qwen tool and extends evaluation-switch
tests; core runtime files remain identical to `2f1ee4f2e`. The resumed runs use
checkout `f83613e7fa4e3e34731faef1ba579b4cdea2e65f`, whose runtime and test sources
are unchanged. Completed results and source/data hashes are independently
verified in `four-gpu-completion-20261011/qwen-validation.json`; regression and
CP-baseline logs are indexed by `four-gpu-completion-20261011/acceptance.json`.
The earlier paused-session evidence remains in
`deep-review-20261010/qwen-completed-validation.json`.

Pretrained CP remains unqualified after the actual four-rank disabled-baseline
failure described above. Upstream GPU CI has not run; the PR remains a draft
pending upstream review and CI approval. Local results do not replace upstream CI.

## Complete-Qwen A6000 measurements

### Unfused pre-GDR

These measurements use unfused pre-GDR with the pretrained model and real text
dataset described above. They use
four microbatches per complete optimizer step, the default 1,048,576-element
threshold, three warmup steps and ten measured steps. Each of the four arms
runs in a fresh process; the comparison repeats three times for each sequence
length and output-norm setting. Correctness snapshots run separately.

The two norm-setting matrices ran concurrently on two otherwise idle A6000s.
Every comparison keeps its setting on the same device: physical GPU 4
(`GPU-7595108b-553d-3126-385e-30133180c720`) without norm recomputation, and GPU 1
(`GPU-5e2e1f94-e88e-48e3-247d-328b52fc68db`) with it. Per-second process samples
found no external compute process on either measured GPU. Other GPUs and the
host were shared, and clocks were not locked. Use the paired results within each
setting/device; absolute runtime comparisons between settings include device
differences.

Peak allocated memory is the median of three per-run maximums. Runtime change
is the median [range] of the three within-run percentage changes in median step
time versus disabled offloading. At sequence 2048, allocation peaks vary by up
to 1.2 MiB between processes. Runtime ranges are wider at 2048; the no-norm
full-offload comparison spans negative and positive changes under these shared-host,
unlocked-clock conditions. Fraction zero keeps the existing manager markers and
hooks without steady transfers; its measured overhead remains in the table.

| Sequence | Norm recompute | Fraction | Peak allocated (MiB) | Selected D2H/iteration (GiB) | Runtime change, median [range] |
| ---: | --- | --- | ---: | ---: | --- |
| 2048 | Off | disabled | 21886.40 | 0.000 | — |
| 2048 | Off | 0 | 21885.40 | 0.000 | +2.31% [-11.17%, +6.86%] |
| 2048 | Off | 0.5 | 21633.40 | 0.984 | +2.90% [-2.80%, +8.14%] |
| 2048 | Off | 1 | 21410.15 | 1.941 | +6.20% [-9.64%, +14.46%] |
| 4096 | Off | disabled | 31599.17 | 0.000 | — |
| 4096 | Off | 0 | 31599.17 | 0.000 | -0.03% [-0.12%, +0.15%] |
| 4096 | Off | 0.5 | 31095.17 | 1.969 | +1.04% [+0.56%, +1.09%] |
| 4096 | Off | 1 | 30647.17 | 3.883 | +1.58% [+1.56%, +2.01%] |
| 2048 | On | disabled | 21596.28 | 0.000 | — |
| 2048 | On | 0 | 21595.28 | 0.000 | +6.17% [+4.70%, +14.78%] |
| 2048 | On | 0.5 | 21344.03 | 0.984 | +9.23% [+1.99%, +21.18%] |
| 2048 | On | 1 | 21120.03 | 1.941 | +6.19% [+4.03%, +10.08%] |
| 4096 | On | disabled | 31018.67 | 0.000 | — |
| 4096 | On | 0 | 31018.67 | 0.000 | +0.05% [-0.03%, +0.14%] |
| 4096 | On | 0.5 | 30514.67 | 1.969 | +0.70% [+0.26%, +0.86%] |
| 4096 | On | 1 | 30066.67 | 3.883 | +1.06% [+0.46%, +1.48%] |

Fraction 1 saves about 476 MiB (2.18%) at 2048 and 952 MiB (3.01%) at 4096 without
norm recomputation. With norm recomputation, it saves about 476 MiB (2.21%) and
952 MiB (3.07%), respectively, relative to that setting's disabled baseline.
These complete-model percentages include attention, MLPs, vocabulary loss and
optimizer state; the earlier stack results use a different memory denominator.

[Raw single-rank and DP samples](gdn_activation_offload_training_a6000.csv) include
peak allocated/reserved CUDA bytes, synchronized step seconds, device UUID,
selected group calls and transfer bytes. The separate `offload_summary_bytes`
field reports the manager's startup overlap window. Each single-rank performance
process completed 13 optimizer updates with finite positive gradient norms. Its
13 logged loss records match exactly across arms, repeats and norm settings for
each length; full gradient/weight/Adam
fingerprints were checked in the separate three-step correctness runs above.

These measurements use code commit
`4e18562c6bcf894795d23378a8f1699c79e4b797`, including the pooled-buffer reuse fix.
The earlier 480-sample comparison is preserved at `496232fe0`; its runtime figures
are superseded here.

#### Unfused data-parallel measurement

A separate DP=2 comparison uses physical GPUs 1 and 4 together, sequence 2048,
norm recomputation disabled, four microbatches per replica, the default threshold,
three warmup steps and ten measured steps. Disabled/full offload each repeat in
three fresh two-rank runs. The global batch is eight sequences per optimizer step;
the single-rank comparison above uses four. Compare each topology with its own
baseline. Per-second samples find no external compute process on either GPU.

For each measured iteration, take the larger allocated-memory peak and the slower
rank's synchronized step time. Aggregate those world-step values using the same
repeat-level method as above.

| DP size | Sequence | Norm recompute | Disabled peak (MiB) | Full-offload peak (MiB) | Memory reduction | Runtime change, median [range] |
| ---: | ---: | --- | ---: | ---: | --- | --- |
| 2 | 2048 | Off | 21886.40 | 21410.40 | 476 MiB / 2.17% | +3.67% [+2.46%, +3.73%] |

The six two-rank runs produce 60 measured world steps and 120 raw rank samples.
All 13 logged loss records on each rank match its own baseline across arms and
repeats; replicas consume distinct data and have different losses. Fraction 1
selects 2,084,569,088 transfer bytes per replica per full iteration. The same CSV
contains these rank samples, identified by `dp_size=2` and `rank`; its 480
single-rank samples have `dp_size=1` and `rank=0`. These data qualify this DP
measurement, not TP/CP/PP throughput or scaling efficiency.

### Fixed fused pre-GDR

This separate matrix uses the fixed gate-backward implementation from
`2f1ee4f2e3dd6d909d08981338f5879632131139`, with the same model, data, default
threshold, microbatches and timing method as the unfused matrix. The later
`6f83b6c7b` commit only scopes the kernel determinism registry; production runtime
files remain identical. Fused and unfused results each use their own disabled
baselines.

Both norm-setting matrices ran sequentially on physical GPU 4
(`GPU-7595108b-553d-3126-385e-30133180c720`). Every timed process has per-second
occupancy evidence with no external compute process on its measured GPU.
The host and other GPUs were shared, and clocks were not locked. These results
measure the offload trade-off within each fusion/norm setting; they do not isolate
the cost of changing the gate kernel or compare fusion speed across environments.

| Sequence | Norm recompute | Fraction | Peak allocated (MiB) | Selected D2H/iteration (GiB) | Runtime change, median [range] |
| ---: | --- | --- | ---: | ---: | --- |
| 2048 | Off | disabled | 21737.65 | 0.000 | — |
| 2048 | Off | 0 | 21737.03 | 0.000 | +0.34% [-6.45%, +2.76%] |
| 2048 | Off | 0.5 | 21484.90 | 0.984 | +3.30% [-2.48%, +5.37%] |
| 2048 | Off | 1 | 21261.03 | 1.941 | +7.65% [-2.42%, +10.20%] |
| 4096 | Off | disabled | 31302.17 | 0.000 | — |
| 4096 | Off | 0 | 31302.17 | 0.000 | +0.38% [+0.30%, +0.59%] |
| 4096 | Off | 0.5 | 30798.17 | 1.969 | +2.85% [+2.53%, +3.60%] |
| 4096 | Off | 1 | 30350.17 | 3.883 | +5.80% [+4.95%, +8.10%] |
| 2048 | On | disabled | 21446.40 | 0.000 | — |
| 2048 | On | 0 | 21446.78 | 0.000 | +2.37% [-0.22%, +7.92%] |
| 2048 | On | 0.5 | 21194.65 | 0.984 | +9.45% [+2.97%, +9.77%] |
| 2048 | On | 1 | 20970.78 | 1.941 | +9.68% [+2.49%, +19.03%] |
| 4096 | On | disabled | 30721.67 | 0.000 | — |
| 4096 | On | 0 | 30721.92 | 0.000 | +0.48% [+0.41%, +1.45%] |
| 4096 | On | 0.5 | 30217.92 | 1.969 | +2.99% [+2.89%, +5.01%] |
| 4096 | On | 1 | 29769.92 | 3.883 | +6.21% [+4.30%, +8.50%] |

Full offload saves 476.62 MiB (2.19%) at sequence 2048 and 952.00 MiB (3.04%) at
4096 without norm recomputation. With norm recomputation, it saves 475.62 MiB
(2.22%) and 951.75 MiB (3.10%), respectively.
Complete-model peaks include attention, MLPs, vocabulary loss and optimizer state.

The 48 fresh single-rank processes produce 480 measured samples. All 13 logged
loss records match exactly across arms, repeats and norm settings at each length.
Gradient, weight and complete Adam fingerprints are checked separately in the
three-step correctness runs; those snapshots are excluded from timed runs.

#### Fused data-parallel measurement

A separate fused DP=2 comparison uses physical GPUs 1 and 4 together, sequence
2048, norm recomputation disabled and the same timing method. Four microbatches
per replica give a global batch of eight sequences per update. Both measured GPUs
have occupancy evidence without external compute processes. As in the unfused
DP comparison, each world step uses the larger allocated peak and the slower
rank's synchronized time.

| DP size | Sequence | Norm recompute | Disabled peak (MiB) | Full-offload peak (MiB) | Memory reduction | Runtime change, median [range] |
| ---: | ---: | --- | ---: | ---: | --- | --- |
| 2 | 2048 | Off | 21737.65 | 21261.78 | 475.87 MiB / 2.19% | +6.16% [+2.07%, +7.01%] |

The six fresh two-rank runs produce 60 measured world steps and 120 rank
samples. All 13 loss records on each rank match its own disabled baseline across
arms and repeats; replicas consume distinct data. Full offload selects
2,084,569,088 transfer bytes per replica per iteration.

[Raw fixed-fusion single-rank and DP samples](gdn_activation_offload_training_fused_a6000.csv)
contain all 600 rank samples. The earlier unfused CSV remains separate with its
original runtime-source provenance; historical failures are retained separately.
These DP figures qualify the measured configuration, not TP/CP/PP throughput.

## Training comparison

The test builds eight complete decoder blocks with the `GGG*GGG*` pattern: three
GDN blocks for each gated GQA block, with a SwiGLU MLP in every block. It includes
RoPE, RMSNorm, Q/K normalization, token embeddings, a tied output head, vocabulary
cross-entropy, DDP gradient buffers and Megatron's BF16 Adam optimizer with FP32
master weights. Hidden size is 256, MLP size 896, vocabulary size 512 and sequence
length 128. Weights and token batches are synthetic.

GDN uses four key heads, sixteen value heads and head dimension 64. The sixteen
value heads keep local counts divisible by four for the listed TP/CP shards, so
the packed projection stride meets native causal-conv1d backward's eight-element
alignment requirement. The initial four-value-head fixture failed in its disabled
fused TP baseline with an unaligned stride. Both comparison arms use aligned
dimensions, preserving the native convolution contract and external backend checks.

For each topology, baseline and offloaded models start from identical weights and
consume the same changing batches. Three optimizer iterations each accumulate
four microbatches from two new examples repeated in the same order, covering
manager warmup and two steady iterations. With fixed weights during accumulation
and dropout disabled, repeated examples must yield identical losses in both
arms. This checks the baseline's microbatch correspondence as well as offload
equivalence. Each rank compares losses, accumulated gradients before clipping,
gradient norm, updated weights, FP32 master weights and Adam states exactly. The test also requires a
successful optimizer update with a positive gradient norm, the expected number of
loss records on the last pipeline stage, zero outstanding pinned buffers before
each native schedule reset and a completed offload warmup. Fraction zero must select no transfer bytes after
warmup; other fractions must select some.

The comparison crosses fractions 0, 0.5 and 1 with output-norm recomputation
and pre-GDR fusion disabled/enabled for these topologies:

| Test ID | TP | SP | CP | PP | Virtual chunks per PP rank | Minimum ranks |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `dp` | 1 | No | 1 | 1 | 1 | 1 |
| `tp` | 2 | No | 1 | 1 | 1 | 2 |
| `tp_sp` | 2 | Yes | 1 | 1 | 1 | 2 |
| `cp` | 1 | No | 2 | 1 | 1 | 2 |
| `pp` | 1 | No | 1 | 2 | 1 | 2 |
| `pp_vpp` | 1 | No | 1 | 2 | 2 | 2 |
| `tp_sp_cp` | 2 | Yes | 2 | 1 | 1 | 4 |
| `tp_sp_pp` | 2 | Yes | 1 | 2 | 1 | 4 |

DP size is world size divided by TP × CP × PP. Each DP replica uses different
tokens, shared by its model-parallel ranks. A topology is skipped if world size is
not divisible by TP × CP × PP. The test does not compare numerical results between
different topologies. CP cases let Transformer Engine select an attention backend and require
a compatible CP backend; non-CP cases use its unfused attention path. GDN uses FLA
and has no deterministic reference fallback. The test sets
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` for reproducible attention backward while
leaving GDN's FLA dispatch enabled.

Both training entry points select native unbatched P2P. In the two-rank VPP
ring, previous and next stages are the same peer. Batched P2P delivered
mismatched forward activations after the initial forward-only microbatches in
the Qwen baseline. Tracing all four microbatches confirmed unchanged parameters and
mismatched activation hashes at pipeline boundaries. Native unbatched P2P
restored matching send/receive hashes and initial losses identical to the
single-GPU baseline. Pipeline-output deallocation did not cause this failure;
the tool retains Bridge's deallocation setting and records the communication
and deallocation settings. Baseline/offload equivalence alone does not qualify
an unhealthy training baseline. A regression run with batched P2P failed the
repeated-example check on both ranks; the unbatched two-rank run passed all
72 applicable cases on each rank (24 four-rank cases skipped).

## Run on available GPUs

Use the repository's development container with FLA installed. Select GPUs that
have no other compute processes and expose those devices to the container before
launching. A zero utilization sample alone does not establish availability. The
commands below assume those GPUs are already the visible devices in the container.
The `--confcutdir` option omits unrelated root dataset-download fixtures. The
reduced-model CP cases require a compatible attention backend; this A6000 run
adds FlashAttention 2.8.3 to the environment.
When omitting the root conftest, also restore its NCCL session defaults and
explicit process-group cleanup below. An isolated run reported all assertions
passed but remained in NCCL teardown at interpreter exit; the cleanup-aware
invocation completed with exit code 0.
The expanded multi-topology suite stalled near device capacity without the
single-channel limit; these flags match the repository's unit-test configuration.

```bash
# Match tests/unit_tests/conftest.py before NCCL initialization.
export NCCL_MAX_NCHANNELS=1
export NCCL_NVLS_ENABLE=0

# Keep the root conftest's session cleanup in this isolated test invocation.
cat > /tmp/gdn-training-tests.py <<'PY'
import sys

import pytest
import torch

status = pytest.main(sys.argv[1:])
if torch.distributed.is_initialized():
    torch.cuda.synchronize()
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    torch.distributed.destroy_process_group()
sys.exit(status)
PY

# One rank: 12 cases run and 84 topology cases skip.
uv run python -m torch.distributed.run --standalone --nproc-per-node=1 \
  /tmp/gdn-training-tests.py -q --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gdn_offload_training.py

# Two ranks: 72 cases run and 24 topology cases skip on each rank.
uv run python -m torch.distributed.run --standalone --nproc-per-node=2 \
  /tmp/gdn-training-tests.py -q --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gdn_offload_training.py

# Four ranks: all 96 cases run on each rank.
uv run python -m torch.distributed.run --standalone --nproc-per-node=4 \
  /tmp/gdn-training-tests.py -q --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gdn_offload_training.py
```

Start with the one-rank run, then qualify the additional topologies on two and
four ranks. An eight-rank run also exercises DP alongside each topology. Preserve
per-rank output and the actual pass/skip counts. These are correctness runs;
snapshot copies and comparisons exclude them from memory/runtime reporting.

## Local small-Qwen training entry point

`tools/ssm/qwen_gdn_offload_training.py` prepares and runs dense Qwen3.5 text
models, starting with Qwen3.5-0.8B. It uses Megatron Bridge's existing VL
checkpoint conversion and pre-wrap weight-loading hook, then extracts and wraps
the loaded language model in native DDP. The text decoder,
embeddings and output head retain their checkpoint dimensions and weights;
frozen vision modules are released after import and MTP is disabled. MoE models
are outside this pilot. Bridge's dense VL importer currently rejects
`text_only=True`; this entry point adds no custom checkpoint mapping.
The pinned Bridge Qwen forward also omits GPT's offload preprocessing call. The
tool invokes the existing native preprocessing method before each forward when
offloading is enabled, so each pipeline chunk uses the native manager lifecycle.
It uses TE's automatic attention selection to keep language and vision backend
settings compatible during checkpoint import. CP still requires a compatible
attention backend on the selected hardware.

Use a Bridge environment with Qwen3.5 VL mappings. The prepared
environment uses Bridge revision `c860f8a5bc5fddd78690f32baa0b8696774308b4`,
Transformers 5.15.0 and Tokenizers 0.22.2, with the local Megatron-LM checkout on
`PYTHONPATH`. Bridge is an optional dependency of this tool, not a new core runtime
dependency. Only the configurations reported above are qualified by these runs.

The tool rejects incompatible partitions before checkpoint loading or distributed
setup. With SP, sequence length must be divisible by TP times CP so each CP-local
sequence can be split across TP ranks. CP also requires divisibility by twice CP
for zigzag partitioning. With VPP, microbatch count must be divisible by PP because
the tool uses the native default of PP-sized microbatch groups. The tokenizer is
imported only when reading the dataset.

The input is local JSONL containing text conversations:

```json
{"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
```

The tokenizer's chat template renders each conversation. The tool concatenates
the tokenized conversations and uses cyclic windows, assigning different windows
to DP replicas. It computes causal-LM loss on all tokens, including system/user
tokens. This is a training validation workload; it does not implement
assistant-only SFT masking or establish model quality. Input preparation and
copies are outside timing; gradient-buffer zeroing, native forward/backward,
gradient communication and BF16 Adam updates are inside timing.

```bash
QWEN_WEIGHTS=/path/to/Qwen3.5-0.8B
QWEN_DATA=/path/to/text-messages.jsonl

# No CUDA model construction or training: check config and tokenize real data.
CUDA_VISIBLE_DEVICES= uv run python tools/ssm/qwen_gdn_offload_training.py \
  --weights "$QWEN_WEIGHTS" --data "$QWEN_DATA" --dry-run \
  --output /tmp/qwen-preflight.json

# Run on an available GPU. Each arm starts from the original local checkpoint.
# Short-sequence correctness checks lower the threshold to exercise transfers.
for ARM in disabled 0 0.5 1; do
  OFFLOAD_ARGS=()
  if [[ "$ARM" != disabled ]]; then
    OFFLOAD_ARGS=(--fraction "$ARM")
  fi
  uv run python -m torch.distributed.run --standalone --nproc-per-node=1 \
    tools/ssm/qwen_gdn_offload_training.py \
    --weights "$QWEN_WEIGHTS" --data "$QWEN_DATA" \
    --check-state --seq-length 128 --min-offloaded-tensor-size 1024 \
    --warmup 1 --iterations 2 "${OFFLOAD_ARGS[@]}" \
    --output "/tmp/qwen-check-$ARM.json"
done

python - <<'PY'
import json
from pathlib import Path
baseline = json.loads(Path('/tmp/qwen-check-disabled.rank0.json').read_text())['states']
for arm in ('0', '0.5', '1'):
    actual = json.loads(Path(f'/tmp/qwen-check-{arm}.rank0.json').read_text())['states']
    assert baseline == actual, arm
PY
```

`--check-state` records per-step loss and gradient norm, plus SHA-256 fingerprints
of accumulated gradients before clipping, updated weights and optimizer-state
tensors, and verifies pinned-buffer ownership before schedule reset. It streams fingerprints through CPU memory and writes no model-sized
checkpoint or second GPU model copy. These runs emit no timed samples. Every rank
writes a separate `.rankN.json`; compare every rank for multi-rank correctness.
Repeat all four arms with `--recompute-norm` before claiming that combination.
Check the separate `offload_status` in each result: fraction zero must select no
transfer bytes, and a positive fraction must select transfers on ranks that
contain GDN layers. Otherwise the comparison has not exercised offloading.

`--fused-pre-gdr` selects the existing pre-GDR fusion. Its native backward requires
causal-conv1d; these runs use version 1.6.1. The tool and fused replay tests set
`CAUSAL_CONV1D_DETERMINISTIC=1` for a fixed convolution-weight-gradient reduction
order. GDN still uses FLA with `deterministic_mode=False`; comparisons remain
bit-exact. Use causal-conv1d 1.6.1 or newer for these comparisons; the fused tests
skip older versions. The optional extension is installed separately from the base
runtime.

For memory/runtime comparisons, omit `--check-state`, use sequence lengths 2048
and 4096 and preserve the default three warmup and ten measured iterations.
The default minimum tensor size is 1,048,576 elements (2 MiB for BF16 or 4 MiB
for FP32), rather than a byte threshold. Run
disabled/0/0.5/1 in separate processes, repeating the full comparison three times.
Samples include CUDA peak allocated/reserved memory, synchronized step time and
selected group calls/transfer bytes for the full iteration. Selection counts and
bytes come from the fixed policy learned during warmup; the native manager's
separate byte summary covers its startup overlap window. Aggregate across ranks
using maximum memory and the slowest rank's step time for each iteration.

The same entry point accepts `--tp 2`, `--tp 2 --sp`, `--cp 2`, `--pp 2`,
`--pp 2 --vp 2`, `--tp 2 --sp --cp 2` and `--tp 2 --sp --pp 2` with the rank
counts in the table above. Use four or more microbatches for PP/VPP. DP is the
remaining world-size factor. The status table lists the GPU-qualified
configurations, including plain TP and plain PP. Pretrained four-rank TP+SP+PP
is qualified for both fused and unfused correctness. Pretrained CP combinations
remain unqualified. The reduced-model four-rank matrix passes all 96 cases per
rank; its results do not qualify the pretrained architecture.

## Acceptance for additional pretrained configurations

For another checkpoint or topology, use its local model weights and training
dataset for complete-model qualification. Record the exact architecture and
checkpoint format first. A Hugging Face directory requires a compatible weight
conversion/loading path; Megatron's `--load` cannot directly consume its safetensors.
Use the model's tokenizer and an explicitly selected text-training schema. Record
the treatment of MoE, vision and MTP components when present.

Each baseline/offload arm must start from the same weights, optimizer state and
data order. Require matching loss, gradients and optimizer updates across
warmup and steady iterations, plus clean offload-manager and pinned-pool state.
Run fractions 0, 0.5 and 1 with and without output-norm recomputation for each
claimed topology.

Measure complete training steps separately, including gradient communication and
the optimizer. Exclude initialization, compilation, warmup and correctness
snapshots from timing. Record representative sequence lengths, microbatch count,
selected groups/transfer bytes, per-rank peak allocated/reserved CUDA memory and
synchronized step time. Report the maximum memory and slowest rank's step time,
preserve raw samples and repeat the comparisons on dedicated A6000s. Any failed or
unrun topology remains unqualified. Use complete pretrained-model runs to
establish memory/runtime qualification.
