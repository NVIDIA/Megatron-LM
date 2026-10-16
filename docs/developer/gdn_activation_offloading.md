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

Shared manager/scheduler lifecycle and transfer-ordering fixes are reviewed
separately from this integration in PR #8243. This feature preserves the upstream
pre-GDR kernels, including their autotuning and atomic parameter-gradient
reductions. The separate repeatability change in PR #8244 is not a dependency.

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

Training evidence must identify the exact source commit, launcher commands,
per-rank outputs and exit codes. CPU configuration, manifest and documentation
checks do not qualify GPU training. PR descriptions identify the tested revisions;
historical results do not qualify later source changes. Results recorded with
the fixed gate reduction from PR #8244 do not qualify the restored upstream fused
path.

Historical evidence remains useful with its original source identity:

| Recorded workload | Result | Source |
| --- | --- | --- |
| Post-rebase two-rank regressions | 104 passed per rank | `b22c9f7251a480529a4815c8c5fe0ece64734c03` |
| Post-rebase native training, two ranks | 72 passed and 24 four-rank skips per rank | `b22c9f7251a480529a4815c8c5fe0ece64734c03` |
| Post-rebase Qwen TP=2/SP | 24 matching complete recorded rank-step states | `b22c9f7251a480529a4815c8c5fe0ece64734c03` |
| Pre-rebase native training, four ranks | 96 passed, zero skipped per rank | Checkout `5176d49bb0bdd885f6f69575fa8c39295c22b2bb` |
| Pre-rebase Qwen TP=2/SP/PP=2 | 192 matching comparison states, plus 12 control states | Checkouts `c8ceb3b8a7067827aa6246affa8413c6bf04b9cc` / `f83613e7fa4e3e34731faef1ba579b4cdea2e65f` |

[The pinned post-rebase bundle](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/8d63bdf2b746a664efedc28977cf31de04e66bfe)
contains original rank logs, Qwen fingerprints, commands, SHA-256 checksums and a
CPU-only verifier. It does not attest to the split revisions.
[Historical CSVs and detailed experiment reports](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/gdn-offload-split-20261015/split-20261015/historical)
are archived separately from the source documentation.

Representative complete-Qwen, single-A6000 results at sequence 4096 with norm
recomputation off save 952.00 MiB (3.01%) for unfused pre-GDR and 952.00 MiB
(3.04%) for fixed fused pre-GDR. Median runtime changes are +1.58% and +5.80%,
respectively. These are separate offload comparisons, not a controlled comparison
between fusion variants or old/new gate kernels. Their runtime sources are
`4e18562c6bcf894795d23378a8f1699c79e4b797` and
`2f1ee4f2e3dd6d909d08981338f5879632131139`, respectively. Historical package
versions are PyTorch 2.11.0+cu130, TE 2.20.2, FLA 0.5.1 and causal-conv1d 1.6.1.

Pretrained Qwen CP remains unqualified: the recorded disabled baseline cannot
select an attention backend for head dimension 256 on A6000. This does not qualify
new environments or other CP configurations. Local results do not replace
upstream GPU CI. Checkpoint restart and long-run convergence are outside these
comparisons.
