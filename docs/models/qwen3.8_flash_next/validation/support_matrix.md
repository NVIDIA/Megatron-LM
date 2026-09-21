# Support matrix (measured on the whole model)

What the composed Qwen3.8-Flash-Next stack (GDN + QSA + MoE + gated residual + PLE + MTP) actually
does on this branch. Every row was produced by running the single-node proxy
([`../training/proxy_single_node.md`](../training/proxy_single_node.md)) with flag overrides,
4 × GB300, 8 iterations, mock data, seq 4096 — **not** inferred from per-component status.

Measured 2026-09-15 **on the `dev`-based integration branch**. Re-run after any rebuild: the
cases are cheap (one node, ~40 min for all of them) and they are the only thing that proves the
*combination* works.

> ## ⚠️ This branch is based on `main`, and three rows do not carry over
>
> The rows below were measured against `origin/dev`. On the `main` base, pipeline parallelism
> beyond PP1 and every CUDA-graph scope are **refused at configuration validation** rather than
> supported. They are marked ⛔ (main) in place and explained in
> "Not available on the `main` base" at the end of this file. Everything else was re-verified by
> the parity harness and the single-node proxy on this branch.

## Parallelism and data layout

| Feature | Whole model | How it was run | Notes |
|---|:---:|---|---|
| **EP** | ✅ | `--expert-model-parallel-size 4` (default of the proxy) | 8 local experts per rank, as at EP64 on the full model |
| **TP + SP** | ✅ | `--tensor-model-parallel-size 2 --sequence-parallel` | GR warns without SP (replicated compute per TP rank) |
| **PP** | ⛔ **(main)** | — | **Refused on this base.** `--pipeline-model-parallel-size > 1` with the gated residual raises at config validation. The pipeline schedule computes stage-to-stage tensor shapes as `[s, b, hidden_size]`, but the multi-stream hidden state is `[s, b, n*hidden_size]`, so the receive buffers are undersized. The ~35-line `pipeline_parallel/schedules.py` fix (BLOCKERS.md "B1") exists on `dev` and was **not** ported. On `dev` this row is ✅ |
| **VPP** | ⛔ **(main)** | — | **Refused on this base**, for the same reason as PP, and additionally because the `dev` implementation of B1 carries a `TODO: flexible VPP layout`. On `dev` this row is ✅ with the `--overlap-param-gather` caveat (`assert self.param_gather_handle is None` in the distributed optimizer) |
| **CP** (unpacked BSHD) | ✅ | `--context-parallel-size 2 --qsa-use-sparse-attention --cp-comm-type all_gather` | QSA CP **requires the sparse kernel** (the dense-mask bridge is single-rank) and allgather comm; PLE uses its conv halo |
| **Packed rows (THD)** | ✅ **via `--sft` only** | `--sft --pad-packed-seq-alignment max --max-seqlen-per-dp-cp-rank 4096 --qsa-use-sparse-attention --eval-iters 1` | the **sequence-packing schedulers** selected by `--use-varlen-dataset` are rejected by the memory ("use the `--sft` packed path instead"). The SFT path still builds a validation loader, so `--eval-iters 0` fails with "no sample to consume" |
| **Packed rows + CP** | ⛔ | — | rejected at startup: upstream `get_thd_batch_on_this_cp_rank` passes `cu_seqlens_padded=None` to `thd_get_partitioned_indices` and fails with or without the memory. The memory's own packed-CP layout is implemented and module-verified |

## Memory, precision and capture

| Feature | Whole model | How it was run | Peak allocated (proxy) |
|---|:---:|---|---|
| **BF16** | ✅ | baseline | 71.9 GiB |
| **Activation recompute, full** | ✅ | `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1` | 50.7 GiB (−29 %) |
| **Activation recompute, selective** | ✅ | `--recompute-granularity selective --recompute-modules core_attn mhc` | 62.5 GiB (−13 %) |
| **MXFP8** | ✅ | `--fp8-format e4m3 --fp8-recipe mxfp8` | 72.4 GiB |
| **CUDA graphs, MoE scopes** | ⛔ **(main)** — ✅ on `dev` | `--cuda-graph-impl transformer_engine --cuda-graph-modules moe_router moe_preprocess --cuda-graph-warmup-steps 2` | 30 iterations, **−6.2 % step time**, peak allocated unchanged (42.57 vs 42.59 GiB), loss identical to the eager run at iteration 1 (lm loss 1.073585E+01, grad norm 6.642). `_layer_is_graphable` leaves a hybrid layer eager unless its inner layer is MoE, so this captures the gated-residual aggregate plus the router and dispatch preprocessing and never touches GDN, QSA or the memory. `[moe_router]` alone measures the same — `moe_preprocess` adds nothing. Needed two fixes: the gated residual used to pack its always-`None` `h_res` slot as a graph output, and Engram rejected every `cuda_graph_impl`. **Neither fix is on this branch** (`9c7108872` and `94d0708e6` were deliberately not ported), so every CUDA-graph scope is refused here |
| **CUDA graphs, any scope** | ⛔ | `--cuda-graph-impl local`, or `transformer_engine` with `attn` in the scope list | rejected at config validation, by the gated-residual guard and again by Engram. QSA's sparse path builds its selection superset with host synchronization and a data-dependent union size; the memory's hashed lookup and its all-to-all have data-dependent shapes. **This is the largest remaining throughput opening** — the MoE-scoped capture leaves the half of the step where the idle lives |
| **FSDP** | ⛔ | `--use-megatron-fsdp --ckpt-format fsdp_dtensor` | rejected: "Engram does not yet support FSDP" |

## Optimizer

| Feature | Whole model | How it was run | Notes |
|---|:---:|---|---|
| **Adam + distributed optimizer** | ✅ | the default of every recipe here | expert-parallel parameters — the routed experts and the memory's hashed tables — shard their optimizer state over the **expert** data-parallel group, dense parameters over the ordinary one |
| **Muon + distributed optimizer** | ✅ | `--optimizer muon --use-distributed-optimizer` (drop `--use-precision-aware-optimizer` and its four dtype flags; Muon rejects them) | Muon owns the 2-D matrices through the layer-wise optimizer; everything else goes to Adam siblings. **The memory's tables are expert-parallel *and* Adam-managed at once**, which used to be rejected at startup and forced the distributed optimizer off entirely, un-sharding every optimizer state. Measured on the proxy: peak allocated **−15.84 GiB (−23.7 %)** at expert-DP 2, −6.05 GiB at expert-DP 1 (dense side only), iteration-1 losses identical to the un-sharded arm, `torch_dist` round trip clean. Check the startup `[optimizer routing]` line: the tables must land in the scalar/embedding group, never in the matrix group |
| **Muon + expert tensor parallelism** | ❓ | — | untested: every run above used `--expert-tensor-parallel-size 1`. ETP does not change the sharding logic, it only shrinks the expert data-parallel group (`expert_data_parallel_size = world_size // (ETP × EP × PP)`), and the expert sibling is built with the same process groups the standard path uses — but this is reasoned from the code, not measured |
| **Muon + precision-aware optimizer** | ⛔ | `--optimizer muon --use-precision-aware-optimizer` | rejected: "only supported with adam" |

## Correctness checks that ran with the matrix

| Check | Result |
|---|---|
| PLE gradients in the composed stack | `--engram-verify-training`: every iteration `num_tables=64 zero_grad_tables=0 nonfinite_tables=0 changed_tables=64` (64 = 16 tables × EP4) |
| MTP on the hybrid path | `mtp_1 loss` finite and tracking `lm loss` in every case |
| Losses | finite and decreasing in all 11 passing cases |

## Throughput

Measured separately on a **performance** proxy (the same 8-layer shape with the vocabulary scaled
down by the same 1/6 as the layers, so the component mix matches the 48-layer model), 2026-09-17.
Headline for this branch: **do not set `NVTE_NORM_FWD_USE_CUDNN` / `NVTE_NORM_BWD_USE_CUDNN`** —
they cost this model 5-10x step time even though the Qwen3.5-397B recipe relies on them — and
**`gdn_pre_gated_delta_rule_fusion`, a P0 on 397B, is a ~10x regression here**. The GPU is idle
~44 % of the step and ~25 % of its busy time goes to generic elementwise/copy kernels, so
flag-level tuning does very little; launch-count reduction is what pays.

## TODO — what this matrix says is missing

1. **CUDA graphs beyond the MoE scopes** — the gated-residual blocker is fixed and the MoE-scoped
   capture now runs (see above). Two of the three original blockers remain and both are the same
   shape: QSA's host-synchronizing superset build and the memory's data-dependent lookup /
   all-to-all shapes both need a fixed upper bound with the dynamic size consumed inside the
   kernel. Still the single largest step-time opening.
2. **Packed rows + CP** — blocked upstream (`cu_seqlens_padded=None`), not by this model. The
   last piece of long-context packed training.
3. **FSDP** — rejected by the memory; the only route to a non-EP memory layout.
4. **VPP + `--overlap-param-gather`** — a generic distributed-optimizer assertion, but it is the
   combination a user hits, so either fix it or keep it out of the recipes.
5. **Sequence-packing schedulers** (`--use-varlen-dataset`) — rejected by the memory; only the
   `--sft` fixed-capacity packing works today.
6. **EP32 sizing with recompute** — recompute saves 29 % on the proxy and EP32 was ~16 GiB short
   on the full model; worth re-measuring.

## What the matrix does not claim

Eight iterations on mock data at proxy scale. It shows each configuration *runs* on the composed
stack with finite losses and gradients reaching the memory — not that CP, packing or MXFP8 are
numerically equivalent to their single-rank / BF16 counterparts (that is the parity harness's job,
[`parity_and_acceptance.md`](parity_and_acceptance.md)), and not anything about throughput.


## Not available on the `main` base

This branch is `origin/main` + the four feature branches. Three capabilities that the
`dev`-based branch has are **refused at configuration validation** here, with a message naming
the reason, rather than silently producing wrong shapes or numbers:

| Capability | What happens | Why it is not here |
|---|---|---|
| `pipeline_model_parallel_size > 1` | `ValueError` from `TransformerConfig.__post_init__` | `main` has no mHC p2p tensor-shape handling. The fix is ~35 lines in one file, guarded by `enable_mhc_connections`, and is tracked as "B1". Note this gap is **upstream-wide** — it affects every mHC config on `main`, not only the gated-residual variant; the guard here is scoped to `mhc_connection_variant='gated_residual'` so it does not change `main`'s behaviour for other users |
| `virtual_pipeline_model_parallel_size > 1` | `ValueError` from `TransformerConfig.__post_init__` | same root cause; and the `dev` implementation of B1 explicitly flags its VPP layout handling as unfinished |
| CUDA graphs, any scope | `ValueError` from `TransformerConfig.__post_init__`, from the gated-residual guard, and again from Engram if it gets that far | the partial-MoE capture path packs the 4-tuple's `h_res` slot as a graph output, which the gated-residual variant returns as `None`; and Engram's hashed lookup and all-to-all have data-dependent shapes. The two commits that make the MoE-scoped capture work on `dev` were deliberately not ported |

Everything else in the matrix above was re-verified on this base: the parity harness passes
end to end against Hugging Face `qwen4_exp`, and the single-node proxy trains.
