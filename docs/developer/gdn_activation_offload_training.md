# Complete-model GDN offload validation

The [initial offload measurements](gdn_activation_offloading.md) exercise GDN
stacks. `tests/unit_tests/ssm/test_gdn_offload_training.py` adds a reduced-size
language-model comparison through the native training schedules. GPU results
are listed below. Configuration checks on CPU do not qualify a parallel topology.

## GPU validation status

The 2026-10-06 runs use RTX A6000 GPUs, PyTorch 2.11.0+cu130, Transformer
Engine 2.20.2, FLA 0.5.1 and cuDNN 9.19.0. These public-package container runs
do not replace upstream CI.

| Workload | Result |
| --- | --- |
| Reduced model, one rank | 6 passed, 42 topology cases skipped |
| Reduced model, two ranks: DP, TP, TP+SP | 18 passed on each rank |
| Reduced model, two ranks: PP | 6 passed on each rank |
| Reduced model, PP+VPP | 6 passed on each rank after fixing the fixture's explicit layer separators |
| Reduced model, CP | Baseline failed before offloading: no compatible deterministic attention backend in this A6000 environment |
| Reduced model, four-rank combinations | Not run |
| Pretrained Qwen3.5-0.8B, one rank | All 8 comparison arms passed; 24 optimizer updates |
| Pretrained Qwen, multiple ranks | Not run |
| Pretrained Qwen memory/runtime | Measurements in progress; no result qualified yet |

The Qwen comparison uses all 24 text decoder layers, hidden size 1024 and
vocabulary size 248320 from the original checkpoint. Each arm starts from the
same local Hugging Face weights and consumes the same 128 real text conversations
(30,727 rendered tokens). Sequence length is 128, with four microbatches per
optimizer step, one warmup step and two steady steps. All eight arms
(disabled/0/0.5/1 × output-norm recomputation off/on) match exactly, including
across the recomputation settings: losses, numeric gradient norms, all 230
parameter-gradient and updated-weight fingerprints, and complete Adam state.
The manager warmup completes in enabled arms and pinned-buffer use returns to
zero after every step. Fraction 0 selects no transfer bytes; fractions 0.5 and 1
select 66,945,024 and 132,030,464 bytes per full iteration, respectively.

Checkpoint revision: `eb706f593d2d43c90a10271199c10b07ced7569a`.
The single safetensors shard has SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`;
the prepared messages JSONL has SHA-256
`94f22b43be94403da5519036c05a40575bb1eed698606c0a7323fa14736ad1f5`.
Data contents and model weights are not included in the repository.

The CP failure occurs in the disabled baseline. TE disables its available
arbitrary-length fused attention backend for deterministic training on compute
capability below 9.0; FlashAttention is not installed in this environment.
The unfused backend does not support CP. This leaves CP unqualified; the test
does not relax determinism to turn the comparison into a pass.

## Training comparison

The test builds eight complete decoder blocks with the `GGG*GGG*` pattern: three
GDN blocks for each gated GQA block, with a SwiGLU MLP in every block. It includes
RoPE, RMSNorm, Q/K normalization, token embeddings, a tied output head, vocabulary
cross-entropy, DDP gradient buffers and Megatron's BF16 Adam optimizer with FP32
master weights. Hidden size is 256, MLP size 896, vocabulary size 512 and sequence
length 128. Weights and token batches are synthetic.

For each topology, baseline and offloaded models start from identical weights and
consume the same changing batches. Three optimizer iterations each accumulate
four microbatches, covering manager warmup and two steady iterations. Each rank
compares losses, accumulated gradients before clipping, gradient norm, updated
weights, FP32 master weights and Adam states exactly. The test also requires a
successful optimizer update with a positive gradient norm, the expected number of
loss records on the last pipeline stage, zero outstanding pinned buffers and a
completed offload warmup. Fraction zero must select no transfer bytes after
warmup; other fractions must select some.

The comparison crosses fractions 0, 0.5 and 1 with output-norm recomputation
disabled/enabled for these topologies:

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
different topologies. CP cases use Transformer Engine fused attention and require
a compatible CP backend; non-CP cases use its unfused attention path. GDN uses FLA
and has no deterministic reference fallback. The test sets
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` for reproducible attention backward while
leaving GDN's FLA dispatch enabled.

## Run on available GPUs

Use the repository's development container with FLA installed. Select GPUs that
have no other compute processes and expose those devices to the container before
launching. A zero utilization sample alone does not establish availability. The
commands below assume those GPUs are already the visible devices in the container.
The `--confcutdir` option omits unrelated root dataset-download fixtures.

```bash
# One rank: 6 cases run and 42 topology cases skip.
uv run python -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m pytest -q --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gdn_offload_training.py

# Two ranks: 36 cases run and 12 topology cases skip on each rank.
uv run python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m pytest -q --confcutdir=tests/unit_tests/ssm \
  tests/unit_tests/ssm/test_gdn_offload_training.py

# Four ranks: all 48 cases run on each rank.
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
tensors. It streams fingerprints through CPU memory and writes no model-sized
checkpoint or second GPU model copy. These runs emit no timed samples. Every rank
writes a separate `.rankN.json`; compare every rank for multi-rank correctness.
Repeat all four arms with `--recompute-norm` before claiming that combination.
Check the separate `offload_status` in each result: fraction zero must select no
transfer bytes, and a positive fraction must select transfers on ranks that
contain GDN layers. Otherwise the comparison has not exercised offloading.

For memory/runtime comparisons, omit `--check-state`, use sequence lengths 2048
and 4096 and preserve the default three warmup and ten measured iterations. Run
disabled/0/0.5/1 in separate processes, repeating the full comparison three times.
Samples include CUDA peak allocated/reserved memory, synchronized step time and
selected group calls/transfer bytes for the full iteration. Selection counts and
bytes come from the fixed policy learned during warmup; the native manager's
separate byte summary covers its startup overlap window. Aggregate across ranks
using maximum memory and the slowest rank's step time for each iteration.

The same entry point accepts `--tp 2`, `--tp 2 --sp`, `--cp 2`, `--pp 2`,
`--pp 2 --vp 2`, `--tp 2 --sp --cp 2` and `--tp 2 --sp --pp 2` with the rank
counts in the table above. Use four or more microbatches for PP/VPP. DP is the
remaining world-size factor. All flags are preparation for qualification and
remain unverified on GPUs until the corresponding runs pass.

## Pretrained Qwen acceptance

After these integration cases pass, use the selected local model weights and
training dataset for complete-model qualification. Record the exact architecture
and checkpoint format first. A Hugging Face directory requires a compatible weight
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
unrun topology remains unqualified. Neither the stack measurements nor this
synthetic integration test establish pretrained Qwen memory/runtime results.
