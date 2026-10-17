# GDN recurrence activation offloading

`gdn_core_attn` offloads tensors saved by the BF16/FLA GatedDeltaNet recurrence.
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
supported. Configuration rejects GDN2, FP8/FP4, non-BF16 execution, other recurrence
backends, full-layer recomputation and CUDA graphs.

The manager's minimum tensor size, offload fraction and last-group margin apply.
After warmup, fraction zero selects no transfers. Warmup can still copy eligible
tensors to collect size statistics. The default minimum is 1,048,576 elements;
short correctness tests use 1024 elements to exercise copies. Use multiple GDN
layers when measuring steady-state savings: one GDN group is kept resident after
warmup. Evaluation, no-grad and fully frozen recurrence execution bypass the scope.

## Saved tensors and integration boundary

Profiling records saved-tensor storage identity, size, save location and first
backward access. The recurrence saves Q/K/V, gates and its internal WY
representation. Some saves alias inputs retained by neighboring operations;
moving one saved reference does not free a storage still used on GPU.

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

The reduced-model test trains complete GDN/attention decoder blocks, embeddings,
MLPs and vocabulary loss with the native schedules and BF16 Adam. Four ranks
cover DP, TP, TP+SP, CP, PP, PP+VPP, TP+SP+CP and TP+SP+PP, with both fusion
variants, both norm-recomputation settings and fractions 0/0.5/1:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /tmp/gdn-training-ranks --redirects 3 --tee 3 \
  -m pytest -q tests/unit_tests/ssm/test_gdn_offload_training.py
```

The optional Qwen tool imports local Hugging Face Qwen3.5-0.8B weights through
Megatron Bridge and trains all 24 text decoder blocks. Vision and MTP are excluded.
It requires a Bridge environment with Qwen3.5 VL mappings; the historical runs
used revision `c860f8a5bc5fddd78690f32baa0b8696774308b4` and the compatibility
overlay recorded in the evidence. Bridge is not a core runtime dependency.

Give each rank its own persistent Triton cache before starting the training
interpreter. Concurrent ranks can select different autotuned configurations for
the same key; sharing a cache can overwrite those choices and make even repeated
offload-disabled baselines differ. Create the root once for a fixed
topology/fusion/norm-recomputation setting and reuse it for both comparison arms:

```bash
# Run once for this comparison; preserve this exported root between arms.
export TRITON_CACHE_DIR="$(mktemp -d /tmp/qwen-offload-triton.XXXXXX)"
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --log-dir /tmp/qwen-disabled-ranks --redirects 3 --tee 3 --no-python \
  bash -c 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}/rank-${RANK}"; exec python "$@"' \
  qwen-training tools/ssm/qwen_gdn_offload_training.py \
  --weights /path/to/Qwen3.5-0.8B --data /path/to/messages.jsonl \
  --tp 2 --sp --pp 2 --seq-length 128 --min-offloaded-tensor-size 1024 \
  --warmup 1 --iterations 2 --check-state --output /tmp/qwen-disabled.json
```

Run a fresh enabled process with the same exported cache root and arguments plus
`--fraction 1` and a distinct output/log directory. Do not recreate the cache root
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

Complete-Qwen measurements at `87c22d1abae13abf375e8ed4ddfe62301cde87f4`
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

Pretrained Qwen CP remains unqualified: the recorded disabled baseline cannot
select an attention backend for head dimension 256 on A6000. This does not qualify
new environments or other CP configurations. Local results do not replace
upstream GPU CI. The checkpoint checks assume the same data, topology and
configuration. Repeated local data and finite losses do not establish convergence.
