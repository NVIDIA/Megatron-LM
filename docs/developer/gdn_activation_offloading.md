# GDN recurrence activation offloading

`gdn_core_attn` offloads tensors saved by BF16/FLA GatedDeltaNet and GatedDeltaNet2 recurrences.
It reuses the existing activation-offload manager, streams and pinned-memory pool.
The option is disabled by default. Projection and convolution/QKV recomputation
and convolution-input offload are separate scopes proposed in
[PR #7852](https://github.com/NVIDIA/Megatron-LM/pull/7852).

## Usage and supported configurations

Add these options to an existing GDN training configuration:

```bash
export NVTE_CPU_OFFLOAD_V1=1
# Training arguments:
--bf16 --experimental-attention-variant gdn \
--fine-grained-activation-offloading --offload-modules gdn_core_attn
```

The training CLI requires `NVTE_CPU_OFFLOAD_V1=1` with Transformer Engine 2.10
or newer. Programmatic configurations set `fine_grained_activation_offloading=True`,
`offload_modules=["gdn_core_attn"]` and `gdn_kernel_backend="fla"`.
Selective `gdn_norm_out` recomputation and the existing opt-in pre-GDR fusion are
supported for GDN. GDN2 supports `gdn_norm_out` recomputation and requires FLA's
`fla.ops.gdn2` kernel. Configuration requires BF16 recurrence activations and
rejects other recurrence backends and full-layer recomputation. CUDA graphs require
`cuda_graph_impl="transformer_engine"`; attention or whole-layer capture uses the
existing transfer stream/event callbacks. Replay stays on the caller's autograd
stream to preserve the order of layer graphs sharing a memory pool. Local and
full-iteration graphs remain unsupported.
`delay_offload_until_cuda_graph` uses the existing replay flush.

The manager's minimum tensor size, offload fraction and last-group margin apply.
After warmup, fraction zero selects no transfers. Warmup can still copy eligible
tensors to collect size statistics. The default minimum is 1,048,576 elements;
short correctness tests use 1024 elements to exercise copies. Current validation
covers fixed-length training initialized by a training step; evaluation-first
initialization and tensor-size changes after warmup are not qualified. Use multiple GDN
layers when measuring steady-state savings: one GDN group is kept resident after
warmup. Evaluation, no-grad and fully frozen recurrence execution bypass the scope.

## TE projection and MLP precision

The recurrence and its offloaded tensors remain BF16/FP32 when Transformer Engine
projections and MLPs use FP8 or NVFP4. Select these through the existing FP8/FP4
training options. The local Qwen tool accepts `--precision fp8-delayed`,
`--precision fp8-tensorwise` or `--precision fp4`; its default remains BF16.
GDN2 supports these recipes in eager training. TE attention/whole-layer graphs
support FP8 and NVFP4. NVFP4 graph offloading requires
`TransformerConfig.disable_parameter_transpose_cache=True` through programmatic
configuration. Graph capture honors this existing option, as do the native TE
linear wrappers. Cached NVFP4 graph baselines failed even with offloading
disabled on the tested SM120 stack; that combination remains rejected. Disabling
the cache re-quantizes weights on each microbatch. No numerical kernels change.

On RTX PRO 6000 (SM120), exact NVFP4 training comparisons require
`NVTE_NVFP4_DISABLE_RHT=1` and `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1`
before Python starts. These tests cover a constrained recipe, not full NVFP4
or convergence. Hardware and recipe support remain Transformer Engine's
responsibility.

## Saved tensors and integration boundary

Profiling records saved-tensor storage identity, size, save location and first
backward access. The recurrence saves Q/K/V, gates and its internal WY
representation. Some saves alias inputs retained by neighboring operations;
moving one saved reference does not free a storage still used on GPU.

GDN2 keeps differentiable write-gate outputs on GPU because sigmoid backward
also retains their storage. The erase gate stays on GPU when its heads do not
need expansion; expanded and frozen gates remain eligible for offloading.

The offload scope wraps only the FLA recurrence. A differentiable recurrence input
anchors backward prefetch, including a trainable gate when Q is frozen. The scope
commits on the raw recurrence output, before gated output normalization, so reload
completes before FLA backward reads its saved tensors. The integration does not
force-release inputs or break their storage aliases. The output norm and its
optional recomputation checkpoint remain outside the scope.

The integration preserves the existing pre-GDR kernels, including their
autotuning and atomic parameter-gradient reductions.

## Reproduce profiling and training checks

Profile a GDN stack, then benchmark disabled and enabled offloading:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  tools/ssm/gdn_activation_offload.py --mode profile --layers 4 \
  --hidden-size 1024 --seq-length 4096 --output /tmp/gdn-profile.json
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  tools/ssm/gdn_activation_offload.py --mode benchmark --layers 4 \
  --hidden-size 1024 --seq-length 4096 --fractions 0 0.5 1 \
  --output /tmp/gdn-benchmark.json
```

Add `--variant gdn2` to profile or benchmark the GDN2 recurrence. These stack
benchmarks exclude vocabulary loss and optimizer updates.

The reduced-model test trains complete GDN/attention decoder blocks, embeddings,
MLPs and vocabulary loss with the native schedules and BF16 Adam. Four ranks
cover DP, TP, TP+SP, CP, PP, PP+VPP, TP+SP+CP and TP+SP+PP, with GDN's two fusion
settings and unfused GDN2, both norm-recomputation settings and fractions 0/0.5/1:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /tmp/gdn-training-ranks --redirects 3 --tee 3 \
  -m pytest -q tests/unit_tests/ssm/test_gdn_offload_training.py
```

The TE graph test captures attention or whole layers after three native optimizer
steps, verifies that graphs were created, then compares three replay steps with
offloading disabled and enabled. It covers GDN/GDN2, norm recomputation and
delayed offload on single-stage DP, TP2+SP and PP2 configurations:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m pytest -q tests/unit_tests/ssm/test_gdn_offload_cuda_graphs.py
```

For a one-GPU check, use `--nproc-per-node=1` and select `-k tp1`.

The optional Qwen tool imports local Hugging Face Qwen3.5-0.8B weights through
Megatron Bridge and trains all 24 text decoder blocks. Vision and MTP are excluded.
It requires a Bridge environment with Qwen3.5 VL mappings; the historical runs
used revision `c860f8a5bc5fddd78690f32baa0b8696774308b4` and the compatibility
overlay recorded in the evidence. Bridge is not a core runtime dependency.

Give each rank its own persistent Triton and TorchInductor caches before starting
the training interpreter. Inductor stores tuning choices for captured FLA
L2-normalization kernels separately from Triton. Concurrent ranks can select
different autotuned configurations for the same key; sharing a cache can overwrite those choices and make even repeated
offload-disabled baselines differ. Create the root once for a fixed
topology/fusion/norm-recomputation setting and reuse it for both comparison arms:

```bash
# Run once for this comparison; preserve this exported root between arms.
export TRITON_CACHE_DIR="$(mktemp -d /tmp/qwen-offload-triton.XXXXXX)"
export TORCHINDUCTOR_CACHE_DIR="$(mktemp -d /tmp/qwen-offload-inductor.XXXXXX)"
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /tmp/qwen-disabled-ranks --redirects 3 --tee 3 --no-python \
  bash -c 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}/rank-${RANK}"; export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/rank-${RANK}"; exec python "$@"' \
  qwen-training tools/ssm/qwen_gdn_offload_training.py \
  --weights /path/to/Qwen3.5-0.8B --data /path/to/messages.jsonl \
  --tp 2 --sp --pp 2 --seq-length 128 --min-offloaded-tensor-size 1024 \
  --warmup 1 --iterations 2 --check-state --output /tmp/qwen-disabled.json
```

Run a fresh enabled process with the same exported cache roots and arguments plus
`--fraction 1` and a distinct output/log directory. Do not recreate either cache root
or clear its rank subdirectories between matched runs. Repeat with
`--recompute-norm` off/on and `--fused-pre-gdr` off/on, using a new root and a
separate disabled baseline for each fusion/recomputation pair. Keep topology,
rank mapping and model inputs identical within each pair.
For unfused pre-GDR, require exact equality of recorded losses, gradient norms,
all accumulated gradients, updated weights and complete Adam/master-weight state
on every rank. Fused pre-GDR uses native atomic gate-gradient reductions and can
differ even between disabled baselines. Its fingerprints can detect differences
but cannot measure their numerical size; a fused correctness comparison must
inspect the actual tensors against repeated disabled baselines with explicit
numerical tolerances.
Check zero pinned-buffer ownership before native schedule resets; positive
fractions must select transfers. The supplied JSONL must contain a `messages`
field accepted by the local tokenizer's chat template.

To check restart behavior, finish a run with `--save-checkpoint /path/to/new-checkpoint`,
then start a fresh process with `--load-checkpoint /path/to/new-checkpoint`.
The tool uses Megatron's distributed checkpointing for model parameters, complete
optimizer/master-weight state and per-rank RNG state. Token windows continue from
the saved next-step index. Keep the model, data, topology and other training
arguments unchanged between runs. The offload manager is transient and learns
its transfer policy again after restart. Compare the resumed states with the
corresponding steps of an uninterrupted `--check-state` run.

Exact replay assumes the same per-rank tuning choices; independently autotuned
configurations do not have a global bitwise-reproducibility guarantee.
[The recorded cold-cache controls](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/8c5cc72379de4e9ef8d80856973dd64f23cc0dac/gpu-20261016)
include the shared-cache failure and passing isolated-cache runs.

For performance runs, omit `--check-state`, use the default tensor-size threshold,
warm up three optimizer steps and measure ten, with three fresh-process repeats.
Compare the same topology and fusion setting on otherwise idle devices. Report
peak allocated memory and median complete optimizer-step time. Snapshots and
input preparation are outside measured steps. Short correctness comparisons
provide no throughput or convergence qualification.

## Results and evidence provenance

Evidence must identify the executed source commit, launcher commands, per-rank
outputs and exit codes. [The pinned validation bundle](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/6ca000df1d21df1fe042be13f008bd900de1f9b4/restore-upstream-gate-20261016)
contains the restored-kernel checks and their source identities. CPU checks do not
qualify GPU training, and earlier fixed-reduction results do not qualify the
native fused path. [Historical logs, CSVs and reports](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/cf8df32ec2424bcc6c42a5486d2fe68213c74294/split-20261015/historical)
retain their original revisions separately.

Complete-Qwen measurements at `1f2c9ae4f7e617b12caa115e387d95e668517c62`
use one RTX A6000, sequence 4096, four microbatches, fraction 1, the default
tensor threshold and norm recomputation off. Each setting has three
fresh-process on/off pairs, three warmup steps and ten measured complete optimizer
steps. These runs include the separately retained evaluation-first and size-change
policy fixes; they are measurements at the recorded revision, not the current head.
Peak allocated memory decreases by 952 MiB: 3.43% with unfused pre-GDR
and 3.47% with native fused pre-GDR. Paired median step-time changes are
+0.50% (−0.26% to +0.55%) and +4.11% (+3.58% to +6.11%), respectively.
Peak reserved memory increases by 650–652 MiB and 738–756 MiB; offloading
retains 1008 MiB in 72 pinned buffers, all returned to the pool before cleanup.
These allocator and host-memory costs accompany the active-allocation saving.
The GPU had no competing active workload during measurement; other GPUs on the
shared host were in use. Raw commands, rank logs, timing samples and telemetry
are [publicly retained with the executed source revision](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/evidence/gdn-acceptance-20261026/gdn-acceptance-20261026).
Input preparation and snapshots
are outside timed steps. These results compare on/off within each setting and
do not establish a cross-version speedup.

Historical complete-Qwen measurements at `87c22d1abae13abf375e8ed4ddfe62301cde87f4`
use one RTX A6000, sequence 4096, four microbatches, fraction 1, the default
tensor threshold and norm recomputation off. Each fusion setting has three
fresh-process pairs, three warmup steps and ten timed optimizer steps. Peak
allocated memory falls by 952 MiB: 3.01% with unfused pre-GDR and 3.04% with
native fused pre-GDR. Paired median step-time increases are 1.19% (0.87–1.54%)
and 1.89% (1.35–2.57%), respectively. Input preparation and state snapshots are
outside the timing boundary. These compare offload on/off within each fusion
setting; they do not compare the fusion implementations.
[Raw samples, device telemetry and commands](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/f35f8532de106bcb49c1d9ae7b5dea512567d1e4/current-head-gpu-20261016/performance)
retain the executed revision and PyTorch/TE/FLA versions.

At `485cb5fa2ffedaaaa0a1a4d095e43ddd99487c1d`, checkpoint comparisons cover
six continuous steps versus two steps followed by four in a fresh resumed
process: disabled control and all four offloaded fusion/norm settings on one
GPU, plus unfused TP2+SP with norm recomputation. Recorded loss, gradients,
weights and complete optimizer states are byte-identical in these runs.
A separate 250-step unfused Qwen run at `87c22d1ab`, sequence 2048, has finite
loss/gradient norms, a stable 72-buffer pinned pool and stable steady-state GPU
memory; another job overlaps that run, so its timing is not qualified.
[Checkpoint and stability evidence](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/34371ca51fa2dffb5c362f1c52c89b1801670226/gdn-followup-20261016)
includes per-rank logs and state comparisons. The production `megatron` tree is
identical at these two source revisions; the later update adds checkpoint-tool
support and documentation.

[Historical measurements and their version table](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/f35f8532de106bcb49c1d9ae7b5dea512567d1e4/current-head-gpu-20261016#current-source-performance)
retain earlier results, including the fixed-reduction fused implementation at
`2f1ee4f2e` and unfused measurements at `4e18562c6`. Those results do not qualify
the restored native fused implementation or establish a cross-version speedup.

Whole-layer TE graph measurements at
`f19ebc5b1da1c9b0075eecfd2f93fc6ca4b3ba8d` cover the complete Qwen text model
and a randomly initialized eight-block GDN2/attention model. Both include
embeddings, MLPs, vocabulary loss and native BF16 Adam. Each model uses one A6000,
sequence 2048, four microbatches, unfused pre-GDR, norm recomputation off, fraction 1
and the default tensor threshold. Three fresh-process pairs alternate arm order;
each arm trains three native steps, captures graphs, then trains three replay
warmup and ten timed optimizer steps. Input preparation, capture and state
fingerprints are outside performance timing. The optimizer remains eager.

| Model | Allocated-peak change | Reserved-peak reduction, median [range] | Paired step-time increase, median [range] |
| --- | --- | --- | --- |
| Qwen3.5-0.8B, 24 text blocks | +512 bytes | 444 MiB [442, 446] | +21.00% [20.68%, 21.11%] |
| GDN2, 8 blocks, hidden size 1024, vocabulary 4096 | approximately +1 MiB | 258 MiB [248, 286] | +96.45% [96.05%, 96.66%] |

These graph runs reduce reserved memory but show no allocated-peak reduction and
substantial runtime cost. Evaluate the trade-off on the target workload before
enabling graph offloading. The pairs hold TE graph mode fixed. Six-step
correctness runs match losses, gradients, weights and complete optimizer-state
fingerprints; all performance pairs also match their loss and gradient-norm
sequences. Pinned-buffer ownership returns to zero before schedule resets and
pool allocation counters stay constant during replay.
[Commands, raw timings, rank logs and verification](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/6ec7df0dc5adcd9f704077636398999ebdfea027/gdn-graph-training-20261017)
identify the exact source, environment and workload. These measurements precede
the GDN2 live-gate change below; `d18040ba5a` used the same runtime as this source.

Profiling at `d18040ba5a` identifies transfer time without overlapping kernels as
the main graph-offload cost. GDN2 copies 1600 MiB in each direction per optimizer
step; its write-gate storage is also retained by sigmoid backward outside the
recurrence. At `367d014c7bf74ba3553c9db58da47fa5bcba906e`, keeping live sigmoid
gates resident reduces each direction to 1440 MiB with unchanged kernel call counts.

Three fresh-process pairs compare the previous and adjusted GDN2 implementations,
both with offloading enabled and the same whole-layer graph workload above.
Paired median step time falls by 5.93% (5.89–5.95%); the medians of process-median
times are 451.71 and 424.91 ms. Peak reserved falls by an additional paired median
22 MiB (20–38 MiB); allocated peaks are unchanged. The retained pinned-buffer count
falls from 54 to 48. These results measure the gate-selection adjustment, not a
speedup from enabling offloading. Transfer overhead remains substantial.

At the adjusted source, two ranks each pass 20 scope, 18 native training and nine
real TE graph cases without skips. Separate complete-model comparisons with
linear key/value heads 8/16 and 16/16 match all six steps of losses, gradients,
weights and complete optimizer/master-state fingerprints. Evaluation and frozen
gate regressions exercise the resident-gate boundary.
[Transfer traces, paired samples, rank logs and verification](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/a2cd3e9ddc1bfefe92c2a2b4052251c996c16544/gdn-graph-profile-20261017)
retain exact commands, source identities and the earlier configuration probes.

The earlier GDN2 stack result at `043e3173451bea74e63c02b197867bf356ad72e9`
saved 192 MiB allocated (16.88%), with +58.42% median runtime. It excludes
vocabulary loss, optimizer updates and CUDA graphs and remains a separate
historical measurement in the linked evidence.

Pretrained Qwen CP remains unqualified: the recorded disabled baseline cannot
select an attention backend for head dimension 256 on A6000. This does not qualify
new environments or other CP configurations. Local results do not replace
upstream GPU CI. The checkpoint checks assume the same data, topology and
configuration. Repeated local data and finite losses do not establish convergence.
