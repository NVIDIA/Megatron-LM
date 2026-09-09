# Strict review

A comprehensive review of a **Megatron-LM** pull request. Megatron-LM is
NVIDIA's large-scale distributed training framework for LLMs, so the failure
mode that matters most is not a crash — it is a change that trains, converges
to something slightly wrong, and is only noticed weeks later. Review the diff
with a focus on **implementation correctness**, **training performance**, and
**backward compatibility**.

Prerequisite: the mandatory workflow in `../SKILL.md` (diff → domain skills →
this file → review). `BASE REF` is supplied by the caller for diff analysis.

## Review procedure

1. Get PR metadata:
   `gh pr view $PR_NUMBER --repo $REPO --json title,body,baseRefName,headRefName,files,additions,deletions,changedFiles,author`
2. Get the full diff: `gh pr diff $PR_NUMBER --repo $REPO`
   - For large PRs (>50 files), prioritize source code over config/lock/auto-generated files.
3. For each significant changed file, read the full file for surrounding context.
4. Trace data flow and dtype through computation paths to verify correctness.
5. For each newly introduced variable/argument/field, verify it has a meaningful
   runtime use path (see the mandatory check below).
6. Post findings as inline comments with severity and category tags.

## Critical issues (must fix)

### Implementation correctness

- **dtype handling**: Verify operations use the correct dtype at each computation stage — explicit casts must be present at mixed-precision boundaries (e.g. fp16 compute → fp32 accumulation → fp16 output)
- **Loss scaling logic**: Verify DynamicLossScaler changes correctly detect inf/nan, adjust scale factor, and skip optimizer steps — incorrect logic causes training divergence or silent underflow
- **Reduction operations**: Verify reductions (sum, mean, allreduce) use correct dtype, reduction dimension, and normalization factor — wrong dimension or missing fp32 upcast produces silently wrong gradients
- **Normalization layers**: Verify LayerNorm/RMSNorm compute variance and mean on the correct dimension, with correct epsilon placement and upcast before rsqrt
- **Attention computation**: Verify QK^T scaling factor, softmax input dtype, causal mask application, and dropout placement match the intended algorithm
- **Residual connections**: Verify the correct tensor is added (pre-norm vs post-norm) with appropriate dtype for accumulation
- **Optimizer updates**: Verify state updates follow the correct formula — momentum/variance update order, bias correction, weight decay application
- **Gradient clipping**: Verify norm computation uses correct parameter set, norm type (L2 vs inf), and fp32 dtype
- **Embedding/output layer**: Verify weight tying is correctly wired, logit projection uses the right matrix, and output dtype matches expectation
- **MoE routing/aux loss**: Verify expert routing logic (top-k selection, capacity enforcement, token dropping) and auxiliary loss computation follow the intended algorithm

### Correctness

- **Tensor parallel**: Incorrect scatter/gather or allreduce placement — silent wrong results across TP ranks
- **Pipeline parallel**: Wrong microbatch scheduling, missing send/recv synchronization, incorrect grad accumulation across pipeline stages
- **Sequence parallel**: Incorrect sequence dimension partitioning or missing allgather/reduce-scatter in SP regions
- **Context parallel**: Incorrect KV cache partitioning or ring attention implementation errors
- **Expert parallel**: Token routing/dispatch errors across EP ranks, incorrect capacity factor handling
- **Gradient accumulation**: Missing `no_sync()` context or incorrect division factor when accumulating across microbatches
- **Checkpoint save/load**: State dict key mismatch, missing optimizer states, incorrect RNG state restoration — causes silent divergence after resume
- **RNG state management**: Incorrect random seed handling across TP/PP/DP ranks, causing correlated dropout masks or data sampling

## Important issues (should fix)

### Training performance

- **Unnecessary CPU-GPU sync**: `.item()`, `.cpu()`, `torch.cuda.synchronize()`, Python-side tensor value checks in the training loop — kills throughput
- **Redundant communication**: Allreduce/allgather that could be fused, overlapped with compute, or eliminated
- **Memory inefficiency**: Missing activation checkpointing on memory-heavy layers, unnecessary tensor clones or `.contiguous()` calls
- **Communication-computation overlap**: Missed opportunities to overlap allreduce with backward, or allgather with forward
- **Kernel launch overhead**: Python loops over small ops that should be fused into a single kernel
- **CUDA graph compatibility**: Dynamic shapes, Python-side conditionals on tensor values, host-device sync inside a captured region

### Backward compatibility

- **Config/argument changes**: Renamed or removed arguments without a deprecation path — breaks existing training scripts
- **Checkpoint format changes**: Modified state dict keys/structure without migration logic — makes existing checkpoints unloadable
- **Default value changes**: Changed defaults for training hyperparameters or parallelism settings — silently alters behavior for users relying on defaults
- **API contract changes**: Changed function signatures, return types, or side effects in `megatron/core/` without a backward-compat shim
- **Model architecture changes**: Altered layer ordering, initialization, or normalization placement — existing pretrained weights become incompatible

### Megatron Core process group usage

- In `megatron/core` production code, treat new direct reads of global process
  groups from `parallel_state` as review findings unless they are clearly
  compatibility-only.
- Flag added calls to `parallel_state.get_*_group()` or directly imported
  `get_*_group()` helpers when the surrounding code could instead receive a
  `ProcessGroupCollection` or explicit `torch.distributed.ProcessGroup` from its
  caller.
- Do not flag `megatron/core/parallel_state.py`,
  `megatron/core/process_groups_config.py`, tests, docs, initialization/bootstrap
  code that materializes a `ProcessGroupCollection` from MPU globals, or
  explicitly documented migration fallbacks.
- This guidance is advisory and targets Megatron Core library code; do not apply
  it to `megatron/training` or other training-loop code unless the PR opts into
  that migration.

### Mandatory check: unused new variables / arguments

- For each changed file, list newly added identifiers (function args, config fields, locals).
- Verify each has a meaningful read/use path — not just a declaration/docstring or a discard assignment (`_ = new_arg`).
- Use Grep to search for usage beyond declaration sites.
- Treat placeholder discard patterns as findings unless explicitly documented as a temporary migration shim.
- If usage is intentionally deferred, flag it and request an explicit TODO + migration note.

A dead argument is worth catching because it is almost always half of a change:
either the wiring was forgotten, in which case the feature silently does
nothing, or the argument is speculative API surface that will have to be
supported forever.

## Suggestions (nice to have)

### Naming

- A name must describe what the thing *is*, not what it is *used for*
- No abbreviations in parallel/distributed code — use full names (`token_dispatcher`, `routing_map`, `comm_manager`, `world_size`)
- Naming consistency within scope for variables serving the same role

### Function/method decomposition

- Functions over ~50 lines mixing data collection, reduction, computation, and I/O should be split
- Non-trivial logic blocks embedded in a method with a different primary purpose should be extracted

### Simplification

- Redundant operations (e.g. `.reshape(())` on a 0-dim tensor, two-step constructions where one suffices)
- Setup that is constant across training should not run on every forward pass — move it to `__init__`
- Dead complexity that does not achieve its stated purpose
- Unnecessary intermediate aliases adding indirection with no abstraction value

### Other

- Stale, imprecise, or misleading comments/docstrings — a wrong docstring is worse than none
- Missing shape/dtype assertions at parallelism boundaries

## What NOT to comment on

- Style/formatting issues (leave to linters)
- Test code that is reasonably clear
- Clearly intentional design decisions by the author
- Pure refactoring that preserves identical behavior (verify via diff)
- Findings invalidated by deeper analysis — drop them entirely rather than hedging

## Comment format

Prefix each comment with a severity and category tag:

- `**[CRITICAL Implementation]**`, `**[CRITICAL Correctness]**`
- `**[IMPORTANT Performance]**`, `**[IMPORTANT Compatibility]**`
- `**[SUGGESTION Naming]**`, `**[SUGGESTION Simplification]**`

For each finding, explain: (1) what the issue is, (2) why it matters
(impact/risk), (3) a specific suggestion for the fix. The severity tag is what
lets an author triage a long review, so it has to reflect blast radius rather
than confidence.

The inline-`suggestion`-block rule from `../SKILL.md` applies here too.

## Completion

After posting all inline comments, post a summary PR comment:

- Total findings by severity (CRITICAL: N, IMPORTANT: N, SUGGESTION: N)
- The most impactful findings
- An overall assessment of the PR's risk level

If no significant issues are found, approve:

```bash
gh pr review $PR_NUMBER --repo $REPO --approve --body "Strict review passed — no significant issues found. LGTM"
```
