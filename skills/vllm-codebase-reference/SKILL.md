---
name: vllm-codebase-reference
description: Answers questions about how vLLM implements something, by reading the vLLM source checkout at /Users/shanmugamr@nvidia.com/vllm. Use when you need to know how vLLM does MoE routing, expert dispatch, attention, CUDA graph capture, kernel or backend selection, sampling, scheduling, or its host-side decode loop — typically to decide whether Megatron-Core can do the same thing. Also use to resolve a kernel name seen in a vLLM nsys trace back to its call site, to find which env var or config flag gates a code path, or to check whether a vLLM optimization is upstream code or an external package (flashinfer, deep_ep, deep_gemm). Do not use for questions about Megatron-Core's own implementation, for running vLLM, or for interpreting a trace — those belong to the mcore skills, run-qwen-model, and nsight-system-analysis.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# vLLM Codebase Reference

vLLM is the performance target for the Qwen3-30B-A3B campaign. When a
differential says vLLM does something cheaper, the next question is always *how*,
and the answer is in source, not in recollection. This skill turns that question
into a grounded lookup.

## The checkout

```
/Users/shanmugamr@nvidia.com/vllm
```

This is the **exact revision that runs in the benchmarks**, so its answers are
authoritative for this campaign in a way that general vLLM knowledge is not.
Recent vLLM releases moved and deleted things aggressively; anything you
remember about vLLM's layout from training data should be treated as a
hypothesis to verify, never as an answer.

**Read-only.** Never edit, stage, commit, checkout, or `git clean` anything under
that path. Changing it silently invalidates every baseline in `EXPERIMENTS.md`.

Start every session by pinning the revision, and quote it in your answer:

```bash
git -C /Users/shanmugamr@nvidia.com/vllm log --oneline -1
git -C /Users/shanmugamr@nvidia.com/vllm describe --tags 2>/dev/null
git -C /Users/shanmugamr@nvidia.com/vllm status --short   # must be clean
```

If the tree is dirty, say so in your answer — someone has modified the baseline.
Note that `vllm/_version.py` is generated at build time and is absent here, so
`vllm.__version__` reports `"dev"`. Use the `git describe` string instead.

## Search discipline

The tree is ~2200 files (~1600 Python) under `vllm/`. Browsing directories to
orient yourself wastes a large amount of context for a small amount of signal.

1. **Grep for the symbol, not the concept.** `class FusedMoE`, `def
   forward_impl`, `SharedExpertsOrder`, `VLLM_ALL2ALL_BACKEND`. Concept
   greps ("expert dispatch") return prose in docstrings and tests.
2. **Consult [references/navigation-map.md](references/navigation-map.md)
   first** for anything in the MoE, decode-loop, CUDA-graph, attention,
   communication, or config areas. It is a verified path index; it will usually
   save you the search entirely.
3. **Read the definition, then the single call site that matters.** vLLM has
   many abstract bases and registries. The dispatch point (which concrete class
   gets built, and under what condition) is nearly always more informative than
   the implementation.
4. **Prefer `vllm/` over `tests/`** for behavior, but `tests/` is the fastest way
   to see an API's intended calling convention when a signature is unclear.
5. Do not read `configs/*.json` tile files in bulk. There are hundreds. Grep for
   the specific `E=`/`N=`/`device_name=` combination.

## Four question types

### A. "How does vLLM implement X?"

The default. Find the mechanism and, critically, the **condition under which it
is used** — vLLM almost always has several implementations of X behind a
selector, and only one of them is on the benchmarked path. An answer that
describes a code path the benchmark never executes is worse than no answer,
because it is actionable and wrong.

So always report both the implementation *and* what selects it. For MoE that
means `oracle/` and the all2all manager choice; for attention, the backend
selector; for graphs, the cudagraph mode resolution.

### B. "Which call site produces kernel symbol `Y` in the trace?"

The highest-value query in this campaign, and the one the differential workflow
depends on. Names like `fmhaSm100...`, `nvjet...`, `trtllm_gen...`, or an
Inductor-generated `triton_red_fused__to_copy_add_rms_norm_0` need to be
resolved to a package and an entry point before anyone tunes anything.

Work the name for its provenance first:

| Name shape | Origin |
|---|---|
| `triton_{poi,red}_fused_*` | `torch.compile` Inductor output — the fusion boundary is the finding, and the source is the Python module that got compiled, not a kernel file |
| `trtllm_gen*`, `fmha*Sm100*` | flashinfer / TRT-LLM-gen, an **external package** |
| `nvjet*`, `sm100_xmma*` | cuBLAS / CUTLASS via an external library |
| `deep_ep*`, `deep_gemm*` | external DeepEP / DeepGEMM packages |
| plain snake_case C++ symbols | usually vLLM's own `csrc/` |

Then confirm by grepping the Python call site for the launching function. Report
the package, the vLLM file and line that calls it, and whether it is upstream
vLLM code or an external dependency — see question type D.

### C. "What flag or env var gates path Z?"

Check `vllm/envs.py` for the `VLLM_*` definition and its default, then grep for
the constant's uses, then check whether a config field in `vllm/config/`
overrides it. Report the default, because "vLLM does X" is only interesting if X
is on by default in the benchmarked configuration. When the answer changes
behavior for our runs, cross-check against the launch command in
`skills/run-qwen-model/` rather than assuming the default applies.

### D. "Is this upstream vLLM, or an external package?"

Decisive for whether Megatron-Core can adopt the same thing, and the difference
between a day of work and a quarter of it. `flashinfer`, `deep_ep`, `deep_gemm`,
and `pplx_kernels` are **imported, not vendored** in this checkout. If the win
lives in an external package, the follow-up question is whether that package is
already installed in our environment — per the differential workflow, it
frequently is.

Genuinely vendored in-tree: `vllm/vllm_flash_attn/` (a FlashAttention interface
shim) and `vllm/third_party/`.

## Answer format

Optimize the answer for a caller who will not read the source themselves. Be
complete on mechanism and exact on location; skip everything else.

- **Direct answer first** — the mechanism, in two or three sentences.
- **Paths with line numbers**, `vllm/path/file.py:123`, for every claim.
- **A short verbatim snippet** for the load-bearing part. Quote it; do not
  paraphrase code.
- **What selects this path**, and its default. If it is off by default, say so
  prominently.
- **Upstream or external package**, when a kernel or library is involved.
- **What you did not check.** Say "I did not verify whether this is on the
  benchmarked path" rather than implying you did. An unverified claim presented
  as verified is the one failure mode that costs real experiment time here.
- **The HEAD sha**, once, so the answer can be cited in `EXPERIMENTS.md`.

Do not editorialize about whether Megatron-Core should adopt something. Report
what vLLM does and what it costs to do it that way; the pricing decision belongs
to `optimize-inference-siddharth/references/decision-gates.md`.

## Traps in this revision

Three pieces of common vLLM knowledge are stale here, and each has already sent
a search in the wrong direction:

1. **`vllm/attention/` does not exist.** Attention moved to `vllm/v1/attention/`
   (backends, selector, metadata builders) and
   `vllm/model_executor/layers/attention/` (the `Attention` nn.Module).
2. **`fused_moe/` is now subpackages**, not flat files: `prepare_finalize/`,
   `experts/`, `router/`, `runner/`, `oracle/`. Paths like
   `fused_moe/cutlass_moe.py` have moved.
3. **The pplx and naive all2all backends were removed.** Both strings are still
   accepted and are silently rewritten to `allgather_reducescatter` in
   `vllm/config/parallel.py`. There is no pplx code in the tree. A config that
   *says* pplx is not running pplx.

Also: there are **two** GPU model runners. `vllm/v1/worker/gpu_model_runner.py`
is production; `vllm/v1/worker/gpu/model_runner.py` is an experimental "V2"
rewrite with its own separate `InputBatch`. Read the former unless you have
confirmed the run uses the latter.

## Related skills

| Question | Skill |
|---|---|
| How does vLLM do this? | this skill |
| What does the trace say? | `nsight-system-analysis` |
| Should we build it, and how? | `optimize-inference-siddharth` |
| How do I run vLLM or mcore? | `run-qwen-model` |

`optimize-inference-siddharth/references/vllm-differential.md` is the consumer of
this skill: it defines the comparison that generates these questions, and records
what past answers led to.

## Keeping this skill current

Update [references/navigation-map.md](references/navigation-map.md) when a lookup
finds that a path has moved, and add a numbered entry to the traps section when a
piece of stale knowledge costs a real search. Record the HEAD sha you verified
against. Paths are the perishable part of this skill; the discipline above is not.
