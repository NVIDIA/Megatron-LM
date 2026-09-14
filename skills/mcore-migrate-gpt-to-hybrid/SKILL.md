---
name: mcore-migrate-gpt-to-hybrid
description: Migration guide for moving Megatron Core GPTModel checkpoints, model providers, training commands, and layer mappings to HybridModel, including the mechanical steps for transferring an existing pretrain_gpt.py launch script.
license: Apache-2.0
when_to_use: Migrating or reviewing a GPTModel checkpoint or training workflow for HybridModel; transferring an existing pretrain_gpt.py script, sbatch, or launcher to pretrain_hybrid.py; choosing or reviewing a hybrid layer pattern; running gpt_hybrid_conversion.py; loading a converted checkpoint; diagnosing GPT-to-Hybrid migration issues; 'migrate GPTModel to HybridModel', 'convert GPT checkpoint to HybridModel', 'hybrid layer pattern'.
metadata:
  author: Philip Petrakian <ppetrakian@nvidia.com>
---

# GPTModel to HybridModel Migration

## Answer-First Migration Guidance

- The canonical source is
  [`docs/user-guide/hybrid-model-migration.md`](../../docs/user-guide/hybrid-model-migration.md).
- Read the canonical document completely before answering, planning, reviewing,
  editing, converting, or training.
- Keep migration behavior, commands, mappings, prerequisites, limitations, and
  validation in the canonical document only. Do not duplicate them in this
  skill.
- This skill adds only what the canonical document does not cover: the
  mechanical procedure for editing an existing launch script, and the shell
  hazards that procedure runs into.

---

## Workflow

1. Pull the task artifact first: checkpoint metadata, model provider or config,
   training command, conversion log, diff, or failure output.
2. Read the canonical migration document completely.
3. Follow only the relevant document sections. Do not invent an unsupported
   migration path or silently change the target architecture.
4. Validate the result proportionately, invoking the relevant repository build
   and testing skills when applicable.
5. Report the outcome and link the canonical document for human readers.

---

## Transferring an Existing Launch Script

The canonical document specifies *what* the migrated command must contain.
This section covers *how* to edit a working script into it without silent
breakage. Apply the edits in order.

**1. Entrypoint.** `pretrain_gpt.py` → `pretrain_hybrid.py`. When the
entrypoint comes from a shell variable or a wrapper, follow it to the real
invocation.

**2. Generate the pattern instead of typing it.** A 96-layer model needs a
192-character pattern; hand-typing invites a silent off-by-one.

```bash
n=32                                   # source GPT layer count
blk='*-'                               # '*-' dense, '*E' every-layer MoE
pat=$(printf "$blk%.0s" $(seq $n))
```

With pipeline segments (`seg` must divide `n`, and the segment count must be
divisible by `--pipeline-model-parallel-size`):

```bash
n=32; seg=4; per=$((n/seg))
b=$(printf "$blk%.0s" $(seq $per)); pat=$b
for ((i=1;i<seg;i++)); do pat="$pat|$b"; done
```

**3. Replace `--num-layers N`** with `--hybrid-layer-pattern`. Deleting
`--num-layers` matters: leaving a stale value is only a warning, so it looks
healthy while being silently overridden by the pattern-derived count.

**4. Add the stack spec**, replacing any GPT `--spec` rather than adding a
second one.

**5. Delete the pipeline-layout arguments** the parser rejects, and repoint
`--save` at a new directory. See the canonical document for both lists.

### Bash-array scripts: quoting at the definition site is not enough

Most scripts under `examples/` collect arguments in arrays and expand them
**unquoted**:

```bash
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py ${MODEL_ARGS[@]}
```

Unquoted `${ARR[@]}` re-runs word-splitting *and pathname expansion* on every
element, so the pattern is globbed against the launch directory at expansion
time — single-quoting it where the array is defined does not protect it:

```bash
touch 'a-b-'; ARGS=(--hybrid-layer-pattern '*-*-')
printf '[%s]\n' ${ARGS[@]}     # -> [--hybrid-layer-pattern] [a-b-]   silently corrupted
printf '[%s]\n' "${ARGS[@]}"   # -> [--hybrid-layer-pattern] [*-*-]   correct
```

An unmatched glob survives intact, so this passes by luck in most working
directories and fails only when some file happens to match. Store the pattern
in a variable and quote that array's expansion:

```bash
HYBRID_PATTERN=$(printf '*-%.0s' $(seq $NUM_LAYERS))
MODEL_ARGS=( ... --hybrid-layer-pattern "$HYBRID_PATTERN" ... )
torchrun "${DISTRIBUTED_ARGS[@]}" pretrain_hybrid.py "${MODEL_ARGS[@]}" ...
```

### Verify the rewrite

Both checks are cheap and catch the common slips:

```bash
# 1. No rejected or stale arguments survived -- must print nothing.
grep -nE -- '--(num-layers|num-layers-per-virtual-pipeline-stage|num-virtual-stages-per-pipeline-rank|pipeline-model-parallel-layout|account-for-embedding-in-pipeline-split|account-for-loss-in-pipeline-split|hybrid-override-pattern|fim-data)\b' train_hybrid.sh

# 2. Pattern shape -- attn and mlp must each equal the source GPT layer count.
p='*-*-|*-*-'
main=${p%%/*}; main=${main//|/}
attn=${main//[^\*]/}; mlp=${main//[^-E]/}; segs=${p%%/*}; segs=${segs//[^|]/}
echo "layers=${#main} attn=${#attn} mlp=${#mlp} segments=$(( ${#segs} + 1 ))"
```

Then diff the migrated script against the original: it should contain the
edits above and nothing else.

### Expected result of an architecture-preserving transfer

A `*-` or `*E` transfer changes the layer *indexing*, not the model. On a
measured 8-block dense run (2 GPUs, bf16, seq 4096, 100 iterations, identical
seed and data), `pretrain_gpt.py --num-layers 8` and `pretrain_hybrid.py
--hybrid-layer-pattern '*-*-*-*-*-*-*-*-'` produced:

- identical parameter counts (2,818,641,920 on both);
- `HybridModel: ... layers='*-*-*-*-*-*-*-*-' (16 layers)` from the allocator;
- steady-state throughput within 0.1% (490.7 vs 491.2 ms/iter);
- identical loss for the first two iterations, then a zero-mean drift of
  |Δ| ≤ 0.08 attributable to kernel/reduction ordering.

Treat a systematic loss offset, a parameter-count difference, or a throughput
gap beyond noise as a migration bug, not as expected behavior. Note that
per-iteration wall clock early in a run is dominated by dataset-cache warmup,
so compare steady-state iterations only.

---

## Documentation Drift

If the implementation and migration guide disagree:

1. Report the discrepancy before continuing.
2. If the task authorizes a correction, update the canonical document first.
3. Do not add a competing migration rule to this skill.
