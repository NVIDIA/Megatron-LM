# Complete-model GDN offload validation

The [initial offload measurements](gdn_activation_offloading.md) exercise GDN
stacks. `tests/unit_tests/ssm/test_gdn_offload_training.py` adds a reduced-size
language-model comparison through the native training schedules. GPU results
are listed below. Configuration checks on CPU do not qualify a parallel topology.

## GPU validation status

The 2026-10-06/07 runs use RTX A6000 GPUs, PyTorch 2.11.0+cu130, Transformer
Engine 2.20.2, FLA 0.5.1 and cuDNN 9.19.0. These public-package container runs
do not replace upstream CI.

The GDN recurrence suite passes 65 cases in both eager mode and with default
compilation. It covers dense and packed inputs with fused pre-GDR disabled/enabled
and output-norm recomputation disabled/enabled. Fused runs use causal-conv1d 1.6.1
with its deterministic convolution-weight-gradient reduction. The existing
pre-GDR fusion suite also passes all 11 cases.
Fused and unfused offload comparisons use separate disabled baselines; they do
not require the two pre-GDR implementations to produce identical bits.
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
| Reduced model, four-rank combinations | Not run |
| Pretrained Qwen3.5-0.8B, one rank | All 8 comparison arms passed; 24 optimizer updates |
| Pretrained Qwen, fused pre-GDR on one rank | All 8 arms passed, also across norm recomputation settings; 24 optimizer updates; every pre-reset pool count is zero |
| Pretrained Qwen, DP on two ranks | All 8 arms passed on each rank; replicas have distinct losses and identical reduced gradients, weights and Adam states |
| Pretrained Qwen, TP on two ranks | All 8 arms passed on each rank, including across norm recomputation settings |
| Pretrained Qwen, TP+SP on two ranks | All 8 arms passed on each rank, including across norm recomputation settings |
| Pretrained Qwen, PP on two ranks | All 8 arms passed on each rank, including across norm recomputation settings |
| Pretrained Qwen, PP+VPP on two ranks | All 8 arms passed on each rank with native unbatched P2P, including across norm recomputation settings |
| Pretrained Qwen, CP on two ranks | Disabled baseline cannot select an attention backend for head dimension 256 on A6000; unqualified |
| Pretrained Qwen, sequence lengths 2048/4096 | Baseline and fraction 1 passed with norm recomputation off/on; all 3 step states also match across recomputation settings |
| Pretrained Qwen memory/runtime | All 48 processes completed; 480 measured samples across 2 lengths × 2 norm settings × 4 arms × 3 repeats |
| Pretrained Qwen, DP=2 memory/runtime | Disabled/full offload, sequence 2048, three repeats; 60 measured world steps / 120 rank samples |

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

The reduced-model two-rank suite passes all 72 applicable cases on each rank,
with 24 four-rank cases skipped. Its CP attention uses head dimension 64.
The pretrained Qwen CP failure occurs in the disabled baseline with head
dimension 256. TE 2.20.2 disables FlashAttention 2 for head dimensions above 192
on SM86, and its fused CP backend rejects this head dimension. The unfused
backend does not support CP. Installing FlashAttention 2.8.3 therefore qualifies
reduced-model CP but leaves pretrained Qwen CP unqualified. These runs do not
change external backend checks or relax the comparison's determinism setting.

The reduced-model suite checks pinned-buffer ownership immediately before every
native schedule reset. A deliberate omission of one pool return fails this check;
the corresponding ordinary training case passes. Checking only after reset would
miss that omission because reset clears the usage counters. The Qwen tool's
`--check-state` path now checks the pool in native gradient finalization, before
the schedule resets it. Earlier pretrained and timed runs checked counters after
reset; their loss/state and memory/runtime results remain valid, but that check
alone does not establish buffer-return correctness.
All eight single-rank fused arms record zero ownership and match each other across
output-norm recomputation settings. The eight-arm qualification runs precede the
pooled-buffer reuse fix. At code commit
`4e18562c6bcf894795d23378a8f1699c79e4b797`, fraction-1 replays use norm recomputation
for each qualified unfused topology and both norm settings for single-rank fusion.
They compare all three complete step states against the corresponding original
run and check ownership before every reset. The unfused reduced-model fraction/norm
matrix also passed at this commit. Expanded fusion coverage uses the same production
runtime files.

## Complete-Qwen A6000 measurements

These runs use the pretrained model and real text dataset described above, with
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
field reports the manager's startup overlap window. Every single-rank performance process
completed 13 optimizer updates with finite positive gradient norms. All 13 logged
single-rank loss records match exactly across
arms, repeats and norm settings for each length; full gradient/weight/Adam
fingerprints were checked in the separate three-step correctness runs above.

These measurements use code commit
`4e18562c6bcf894795d23378a8f1699c79e4b797`, including the pooled-buffer reuse fix.
The earlier 480-sample comparison is preserved at `496232fe0`; its runtime figures
are superseded here.

### Two-rank data-parallel measurement

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
fused TP baseline with an unaligned stride. Both comparison arms now use aligned
dimensions; no production fusion kernel or external backend check is changed.

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
When omitting the root conftest, also restore its NCCL session defaults below.
The expanded multi-topology suite stalled near device capacity without the
single-channel limit; these flags match the repository's unit-test configuration.

```bash
# Match tests/unit_tests/conftest.py before NCCL initialization.
export NCCL_MAX_NCHANNELS=1
export NCCL_NVLS_ENABLE=0

# One rank: 12 cases run and 84 topology cases skip.
uv run python -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m pytest -q --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gdn_offload_training.py

# Two ranks: 72 cases run and 24 topology cases skip on each rank.
uv run python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m pytest -q --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gdn_offload_training.py

# Four ranks: all 96 cases run on each rank.
uv run python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m pytest -q --confcutdir=tests/unit_tests/ssm \
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
configurations, including plain TP and plain PP. Pretrained CP and four-rank
combinations remain unqualified. Reduced-model results are listed separately.

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
