# Updating This Skill

This skill is a living document. It began as the patterns behind one engineer's
inference work, absorbed a second campaign's methodology, and is expected to keep
growing as agents and engineers use it. Optimization knowledge decays — flags get
renamed, defaults change, a fix that won on one generation of hardware loses on the
next — so a skill that is never edited becomes actively misleading.

**You are authorized to edit this skill without asking.** Add, correct, and delete
as you learn. The rest of this document is how to do that without turning it into a
landfill.

## When to update

Update at the end of a piece of work, not in the middle of it. The moment is after
Step 6 in [SKILL.md](../SKILL.md) — once you have a measured result and a root
cause. Concretely, these seven triggers:

| Trigger | What it produces |
|---|---|
| An optimization was accepted | The pattern, its mechanism, and the measured kernel-to-e2e conversion |
| An optimization was rejected | **The highest-value entry type.** What was tried, the measured result, and the root cause |
| A decision gate came out negative | The ceiling calculation, so nobody re-derives it |
| A tooling or environment failure cost more than an hour | The exact symptom, the trigger, and the working substitute |
| A measurement contradicted something already written here | A scoped correction — see *Corrections and contradictions* |
| A flag, default, or API turned out to differ from what is documented here | A verified correction, with the date you checked |
| Something broke because an invariant was violated | A candidate hard rule |

Do **not** update mid-experiment on a hypothesis. Unmeasured ideas belong in the
campaign ledger, not here.

## The ledger and the skill are different artifacts

Keep both, and keep them distinct. Confusing them is the main failure mode:
everything ends up in one place, and that place becomes unusable.

| | Campaign ledger (`EXPERIMENTS.md`) | This skill |
|---|---|---|
| Scope | One workload, one campaign | Every workload, indefinitely |
| Policy | Append-only, keeps everything | Curated, edited and pruned |
| Contains | Every run, every number, every config | Only what transfers |
| Audience | You, next session | Anyone, next year |

**The promotion test:** would this change what a competent engineer does on a
*different* model or a different workload? If yes, promote it here. If it is a
number that only describes one model at one shape, it stays in the ledger.

For the common in-between case, split it: **the mechanism goes in the skill, the
number goes in the ledger, and the skill cites the number as a calibrated example
with its hardware stamp.** That is what "expert GEMM was 1.45× off roofline on
GB200" is doing in [decision-gates.md](decision-gates.md) — the transferable claim
is *roofline the category before proposing a kernel*, and the number is evidence
that the claim has teeth.

## Where it goes

Route by subject, not by which file you happened to be reading. Adding everything
to `SKILL.md` is the most common way to wreck this skill — `SKILL.md` is a router,
and it stops working once it is long enough to need its own router.

| Learning about | Destination |
|---|---|
| Launch overhead, graph scope, capture safety, bucketing | [cuda-graphs.md](cuda-graphs.md) |
| MoE dispatchers, experts, routing, grouped GEMM, EP comm | [moe-inference.md](moe-inference.md) |
| Per-step CPU work, serialization, IPC, coordinator, DP routing | [host-path.md](host-path.md) |
| Triton rules, kernel hygiene, Mamba/SSM, scratch sizing | [mamba-and-triton.md](mamba-and-triton.md) |
| Profiling tooling, trace analysis, A/B protocol, idle accounting | [measuring.md](measuring.md) |
| Prioritization, ceilings, whether a lever is worth building | [decision-gates.md](decision-gates.md) |
| Comparing against vLLM or another engine | [vllm-differential.md](vllm-differential.md) |
| A new invariant whose violation broke something | `SKILL.md` → Hard rules |
| Flag behavior, defaults, flags that look free and are not | `SKILL.md` → Flags |
| A check that belongs before every PR | [../assets/review-checklist.md](../assets/review-checklist.md) |
| A subject none of the above covers | A new `references/*.md`, linked from `SKILL.md` |

When you add a new reference file, three things must follow or it is invisible: link
it from the `## References` list in `SKILL.md`, mention it from whichever step it
serves, and extend the frontmatter `description` if it represents a genuinely new
capability — that description is what causes this skill to be selected at all.

## The bar for an addition

All four, or it does not go in:

1. **Measured.** A number, and enough method that someone could reproduce it. "Felt
   faster" is not a finding.
2. **Root-caused.** The mechanism, not the symptom. A kernel that was slow because
   2 of 152 CTAs got work teaches something; "the count kernel was slow" teaches
   nothing and invites the wrong fix.
3. **Scoped.** State the conditions under which it holds — hardware, model shape,
   token count, config. An unscoped claim will be applied where it is false.
4. **Actionable.** It changes a decision. If a reader finishes the paragraph and
   does nothing differently, cut it.

Reject: restatements of content already here, cluster or queue incidents with no
methodological consequence, and single anomalous runs that were never repeated.

## House style

Match what is already here, so the file reads as one voice rather than a pile of
contributions:

- Lead with the finding, then the number. Not the narrative of how you got there.
- Imperative and second person. "Roofline the category before proposing a kernel."
- Quote **exact error strings** for failure modes. `AssertionError: hidden_size
  mismatch: 128 vs 8` is findable; "it crashes with a shape error" is not.
- Close a transferable pattern with a **`Generalizable:`** line. That convention is
  load-bearing — it is how a reader tells a specific finding from a rule.
- Cite the source: a commit sha for merged work, a ledger entry id for campaign
  work. Unattributed claims cannot be re-verified.
- Show real code with a file-path comment as the first line, as the existing
  snippets do.
- Tables for enumerable facts, prose for reasoning. Never a table of paragraphs.

## Corrections and contradictions

**If something here is wrong, fix it in place.** If it was wrong in a way someone
may have already acted on, say so in one clause rather than silently rewriting.

**If a new measurement contradicts an existing one, do not overwrite it.** Both are
facts; they differ in scope. Scope them:

> Capping the AllGather-V to 16 CTAs when overlapping was a win on one model and
> measured net worse on Nemotron.

That is the correct shape — it preserves both results and tells the reader the
outcome is model-dependent, which is itself the lesson. Overwriting would have
produced a confident claim that is wrong half the time.

**Never delete a measured negative result** because it is old or inconvenient. Those
entries are the ones preventing repeated work; three of them here each killed a
multi-week effort. If new hardware changes the answer, add the condition, do not
replace the record.

## What to delete

Deletion is part of maintenance, not vandalism. Delete:

- **Content that is now false** — a flag that no longer exists, a default that
  changed, a code path that was removed. Verify against the tree before restating
  any default, and note the date you checked.
- **Superseded APIs**, once nothing in-tree references them. Until then keep the
  old→new mapping in a collapsed `<details>` block; the deprecated
  `CudaGraphScope` block in [cuda-graphs.md](cuda-graphs.md) is the precedent.
- **Duplication.** The same lesson stated in three files means two of them are
  drift waiting to happen. Keep the fullest treatment, link to it from the others.
- **Prose that changes no decision**, including your own earlier additions. Prefer
  cutting recent additions over the original source-commit material, which carries
  provenance you cannot reconstruct.

When you delete something substantive, note it in the revision log so the removal
is discoverable rather than mysterious.

## Size budget

Progressive disclosure only works if the entry point stays navigable.

| File | Soft target | Action when over |
|---|---|---|
| `SKILL.md` | ~500 lines (currently ~466) | Move detail into a reference; keep the routing line |
| Any reference | ~400 lines | Split by subject, link both from `SKILL.md` |

These are soft targets, and the line count is a proxy, not the thing that matters.
The real test for `SKILL.md`: **can a reader scan it once and know which reference
answers their question?** If a section explains a mechanism rather than pointing at
one, the mechanism belongs in a reference regardless of the current line count. Do
not trim readable prose to hit a number — that trade makes the skill worse.

For references the test is subject coherence. If one has grown to cover two
unrelated subjects, that is a split rather than a trim; `mamba-and-triton.md`
covering both Triton hygiene and SSM scratch sizing is already near that line.

## After any edit

```
- [ ] Every intra-skill link still resolves
- [ ] Frontmatter still parses, and `name` still matches the directory
- [ ] `description` extended if a new capability area was added
- [ ] New reference linked from the References list AND from the step it serves
- [ ] Measured claims carry model + hardware + date, or a commit / ledger citation
- [ ] Revision log below appended
```

## Revision log

One line per substantive change, newest last. This exists so a reader can tell what
is original source-commit material and what was added later, and so deletions are
discoverable rather than mysterious.

| Date | Change | Source |
|---|---|---|
| 2026-07 | Initial skill: the five moves, hard rules 1-9, the CUDA-graph / MoE / Mamba-Triton / host-path / measuring references, commit log, review checklist | Siddharth Singh's 2026 inference work, 29 commits |
| 2026-07-28 | Added `decision-gates.md` (ceiling-before-building, per-launch fixed costs, three gates that each killed a multi-week effort) and `vllm-differential.md` (competitor-trace comparison). Hardened `measuring.md`: nsys flag combinations that deadlock finalization, the node-vs-graph trace control, `perf_counter` phase timing, union-busy idle accounting with gap-size decomposition, the same-session back-to-back A/B protocol, kernel-to-e2e conversion, ledger requirements. Added *What a wide capture costs you* to `cuda-graphs.md`. Added per-GEMM tile tuning, the measured backend comparison, the training-path-only fusion flags, and comm-vs-skew to `moe-inference.md`. Added hard rules 10 (A/B protocol) and 11 (non-bit-exact acceptance) plus the *flags that look like free wins* table | Qwen3-30B-A3B EP4 on 4×GB200, `skills/run-qwen-model/EXPERIMENTS.md` |
| 2026-07-28 | Made the skill self-maintaining: this file, the editing authorization at the top of `SKILL.md`, the Step 6 feedback loop, the checklist's skill-maintenance block, and this log | — |
