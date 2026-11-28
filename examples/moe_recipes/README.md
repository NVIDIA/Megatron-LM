# MoE Recipes

This directory contains self-contained MoE training recipes. Each YAML file includes:

- `DEPENDENCIES`: PyTorch base image and Dockerfile content used for the recipe.
- `ENV_VARS`: Environment variables expected by the runtime.
- `ARGS`: Megatron-LM training arguments.

## Recipe Index

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Recipe</th>
      <th>GPUs</th>
      <th>TP/PP/EP/CP/ETP</th>
      <th>MBS/GBS/SL</th>
      <th>Features</th>
      <th>Median TFLOP/s/GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">DeepSeek-V4-Pro</td>
      <td><a href="deepseek_v4_pro/gb200/mxfp8_SL4K_256GPU_TP1PP4EP64.yaml">GB200 MXFP8 BSHD 4K</a></td>
      <td>256</td>
      <td>1/4/64/1/1</td>
      <td>1/4096/4096</td>
      <td>VPP4; MoK; Muon; full recompute; full CG</td>
      <td>744.0 / 736.7</td>
    </tr>
    <tr>
      <td><a href="deepseek_v4_pro/gb200/mxfp8_THD64K_256GPU_TP1PP4EP64CP16.yaml">GB200 MXFP8 THD 64K</a></td>
      <td>256</td>
      <td>1/4/64/16/1</td>
      <td>1/256/65536</td>
      <td>VPP4; THD packing; MoK; Muon; full recompute; eager</td>
      <td>570.2 / 569.7</td>
    </tr>
    <tr>
      <td><a href="deepseek_v4_pro/gb300/mxfp8_SL4K_256GPU_TP1PP4EP64.yaml">GB300 MXFP8</a></td>
      <td>256</td>
      <td>1/4/64/1/1</td>
      <td>1/4096/4096</td>
      <td>VPP4; paged stash; full CG; HybridEP; activation offload</td>
      <td>929</td>
    </tr>
    <tr>
      <td rowspan="3">DeepSeek-V4-Flash</td>
      <td><a href="deepseek_v4_flash/gb200/mxfp8_SL4K_128GPU_TP1PP1EP64.yaml">GB200 MXFP8 BSHD</a></td>
      <td>128</td>
      <td>1/1/64/1/1</td>
      <td>1/2048/4096</td>
      <td>BSHD; MoK; MXFP8 indexer and batched projection; full CG; Muon</td>
      <td>726.1</td>
    </tr>
    <tr>
      <td><a href="deepseek_v4_flash/gb200/mxfp8_THD4K_128GPU_TP1PP2EP64.yaml">GB200 MXFP8 THD 4K</a></td>
      <td>128</td>
      <td>1/2/64/1/1</td>
      <td>1/2048/4096</td>
      <td>THD packing; partial CG; HybridEP; Muon</td>
      <td>512.3</td>
    </tr>
    <tr>
      <td><a href="deepseek_v4_flash/gb200/mxfp8_THD64K_128GPU_TP1PP2EP64CP16.yaml">GB200 MXFP8 THD 64K</a></td>
      <td>128</td>
      <td>1/2/64/16/1</td>
      <td>1/128/65536</td>
      <td>THD packing; MoK; MXFP8 indexer and batched projection; attention CG; Muon</td>
      <td>599.2</td>
    </tr>
    <tr>
      <td rowspan="5">DeepSeek-V3</td>
      <td><a href="deepseek_v3/h100/bf16_1024GPU_TP1PP16EP64.yaml">H100 BF16</a></td>
      <td>1024</td>
      <td>1/16/64/1/1</td>
      <td>1/8192/4096</td>
      <td>BF16 baseline</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/h100/fp8_1024GPU_TP2PP8EP64.yaml">H100 FP8</a></td>
      <td>1024</td>
      <td>2/8/64/1/1</td>
      <td>1/8192/4096</td>
      <td>DeepEP; EP overlap</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/b200/mxfp8_256GPU_TP1PP8EP32.yaml">B200 MXFP8</a></td>
      <td>256</td>
      <td>1/8/32/1/1</td>
      <td>1/2048/4096</td>
      <td>DeepEP; EP overlap</td>
      <td>776.7</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/gb200/mxfp8_256GPU_TP1PP4EP64.yaml">GB200 MXFP8</a></td>
      <td>256</td>
      <td>1/4/64/1/1</td>
      <td>1/8192/4096</td>
      <td>HybridEP; partial CG; EP overlap; offload</td>
      <td>1127.6</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/gb300/mxfp8_256GPU_TP1PP4EP64.yaml">GB300 MXFP8</a></td>
      <td>256</td>
      <td>1/4/64/1/1</td>
      <td>1/8192/4096</td>
      <td>HybridEP; partial CG; EP overlap</td>
      <td>1357.7</td>
    </tr>
    <tr>
      <td rowspan="6">Qwen3-235B-A22B</td>
      <td><a href="qwen3_235b/h100/bf16_256GPU_TP2PP8EP32.yaml">H100 BF16</a></td>
      <td>256</td>
      <td>2/8/32/1/1</td>
      <td>1/2048/4096</td>
      <td>router/preprocess CG; HybridEP; EP overlap</td>
      <td>214.3</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/b200/mxfp8_128GPU_TP2PP4EP16.yaml">B200 MXFP8</a></td>
      <td>128</td>
      <td>2/4/16/1/1</td>
      <td>3/3072/4096</td>
      <td>DeepEP; EP overlap; VPP12</td>
      <td>633.9</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/b300/mxfp8_128GPU_TP2PP1EP16_partial_cg.yaml">B300 MXFP8 partial CG</a></td>
      <td>128</td>
      <td>2/1/16/1/1</td>
      <td>3/9216/4096</td>
      <td>partial CG; HybridEP; EP overlap</td>
      <td>922.6</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_paged_stash_fullcg_overlap.yaml">GB200 MXFP8 full CG</a></td>
      <td>128</td>
      <td>1/1/64/1/1</td>
      <td>1/8192/4096</td>
      <td>paged stash; full CG; HybridEP; EP overlap</td>
      <td>1159.5</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_partial_cg_overlap.yaml">GB200 MXFP8 partial CG</a></td>
      <td>128</td>
      <td>1/1/64/1/1</td>
      <td>1/8192/4096</td>
      <td>partial CG; HybridEP; EP overlap</td>
      <td>1018</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/gb300/mxfp8_128GPU_TP1PP1EP64_paged_stash_full_cg.yaml">GB300 MXFP8 full CG</a></td>
      <td>128</td>
      <td>1/1/64/1/1</td>
      <td>1/8192/4096</td>
      <td>paged stash; full CG; HybridEP; EP overlap</td>
      <td>1323.1</td>
    </tr>
    <tr>
      <td rowspan="5">Qwen3-30B-A3B</td>
      <td><a href="qwen3_30b/h100/fp8_32GPU_TP1PP1EP8.yaml">H100 FP8</a></td>
      <td>32</td>
      <td>1/1/8/1/1</td>
      <td>1/256/4096</td>
      <td>FP8 blockwise; router/preprocess CG</td>
      <td>232.4</td>
    </tr>
    <tr>
      <td><a href="qwen3_30b/h100/bf16_32GPU_TP1PP1EP8.yaml">H100 BF16</a></td>
      <td>32</td>
      <td>1/1/8/1/1</td>
      <td>1/256/4096</td>
      <td>BF16 baseline</td>
      <td>234.1</td>
    </tr>
    <tr>
      <td><a href="qwen3_30b/gb200/bf16_16GPU_TP1PP1EP16.yaml">GB200 BF16</a></td>
      <td>16</td>
      <td>1/1/16/1/1</td>
      <td>4/512/4096</td>
      <td>BF16 baseline</td>
      <td>528</td>
    </tr>
    <tr>
      <td><a href="qwen3_30b/gb200/mxfp8_16GPU_TP1PP1EP16_partial_cg.yaml">GB200 MXFP8 partial CG</a></td>
      <td>16</td>
      <td>1/1/16/1/1</td>
      <td>4/512/4096</td>
      <td>MXFP8; partial CG</td>
      <td>514</td>
    </tr>
    <tr>
      <td><a href="qwen3_30b/gb200/mxfp8_16GPU_TP1PP1EP16_paged_stash.yaml">GB200 MXFP8 paged stash</a></td>
      <td>16</td>
      <td>1/1/16/1/1</td>
      <td>4/512/4096</td>
      <td>MXFP8; paged stash; full CG</td>
      <td>646.8</td>
    </tr>
  </tbody>
</table>

Legend: TP = tensor parallel, PP = pipeline parallel, EP = expert parallel, CP = context parallel, ETP = expert tensor parallel, MBS = micro-batch size, GBS = global batch size, SL = sequence length, CG = CUDA graph, BSHD = fixed-length batch layout, THD = packed variable-length batch layout. The tuple orders are `TP/PP/EP/CP/ETP` and `MBS/GBS/SL`.

## DeepSeek-V4 Flash MXFP8 recipes

The GB200 BSHD 4K and THD 64K recipes use HybridModel and include pinned build
instructions for TE BatchedLinear, cuDNN Frontend and Mixture-of-Kittens. Run
from the repository root so the shared-expert precision configuration resolves.
MoK uses MXFP8 routed experts and BF16 shared experts; it replaces the native
TE fused-expert/paged-stash path. Both recipes use compact Muon. The 4K recipe uses full-iteration
CUDA graphs; the 64K recipe uses an attention-only graph split
that leaves mHC aggregation and recompute outside the captured region.

TE's grouped output projection stores a 3D parameter. Its explicit Muon matrix
metadata preserves the legacy flattened-matrix update rather than routing the
weight to Adam or applying independent updates to each group.
mHC projection and packed CP compressor parameters are explicitly made ready
before reads outside their owning module's forward, preserving overlapped parameter
gather without reading stale weights.

These are 100-step mock-data throughput recipes with forced routing. They do not
establish training convergence. Keep numerical checks enabled for qualification;
disable them only for the timed run, as in the provided benchmark settings.
Exclude compilation and graph capture from measurements. The 4K results use steps
21–100; the 64K results use steps 41–100 because warmup lasted longer. Verify
that timing has stabilized before choosing a measurement window.
Periodic Python garbage collection is disabled during this bounded benchmark;
startup collection remains enabled.
The refreshed BSHD 4K and THD 64K results were remeasured with parameter-readiness
waits enabled, using source revision `ae5978fb8e90e06b1d7e6cd565e435fdd45b027c`
and the pinned recipe dependencies, before the latest dev merge. Each configuration
ran twice on 128 GB200 GPUs across 32 nodes; each EP64 group stayed within one NVL72 domain. The index reports
the median over both trials (160 measured iterations for 4K and 120 for 64K).

| Recipe | TFLOP/s/GPU from mean step time, trials 1 / 2 | Mean step (s), trials 1 / 2 | Peak allocated memory (GiB/GPU) |
| --- | --- | --- | --- |
| BSHD 4K | 721.8 / 727.2 | 8.242 / 8.181 | 146.2 |
| THD 64K | 597.0 / 600.3 | 12.022 / 11.957 | 102.3 |

Throughput is calculated as model FLOPs divided by the average measured step time.
For comparison with the earlier 20-step warmup window, THD 64K steps 21–100
measure 590.4 / 595.7 TFLOP/s/GPU.
Memory is the maximum PyTorch allocated high-water mark across all ranks and
both trials, including initialization and graph capture; it is not total device
memory usage. All four runs completed 100 iterations with finite logged losses
and gradient norms, no skipped/NaN iterations, and all-rank completion records.
Logged FLOPs agreed with fixed-length accounting. These results do not establish
convergence or a controlled speedup over earlier recipes.

The cuTile compilation cache is placed in node-local `/tmp` to avoid SQLite
locking across ranks on a shared home directory. If the launcher overrides
`CUDA_TILE_CACHE_DIR`, keep it on a node-local filesystem.


## DeepSeek-V4 Pro GB200 MXFP8 recipes

The GB200 BSHD 4K and THD 64K recipes use HybridModel with 61 attention/MoE
blocks and one MTP layer, compact Muon, MoK, an MXFP8 indexer and grouped output
projection, and BF16 shared experts. They share the Flash recipes' pinned build
dependencies and parameter-readiness fixes. Run from the repository root so the
shared-expert precision configuration resolves.

Both use 256 GPUs, TP1/PP4/VPP4/EP64, and full recompute with one checkpoint per
attention/MoE block (`recompute_num_layers: 2` hybrid layers) to fit GB200 memory.
The 4K recipe uses full-iteration CUDA graphs; 64K adds CP16 and runs eagerly
because per-layer TE graphs do not support full recompute. Each step contains
16,777,216 tokens. Keep each EP64 group within one NVL72 domain (16 four-GPU nodes).

Each configuration completed two full-scale runs with the parameter-readiness
fix at source revision `ae5978fb8e90e06b1d7e6cd565e435fdd45b027c`, before the
latest dev merge. These are mock-data, forced-routing throughput measurements,
not convergence validation or a controlled comparison with the existing GB300
recipe. The refreshed GB300 benchmark is still pending; the GB300 row above
refers to the previously listed TE/paged-stash configuration.

| Recipe | Measured steps | TFLOP/s/GPU from mean step time, trials 1 / 2 | Mean step (s), trials 1 / 2 | Peak allocated memory (GiB/GPU) |
| --- | --- | --- | --- | --- |
| BSHD 4K | 21–100 | 743.8 / 736.4 | 28.420 / 28.705 | 152.8 |
| THD 64K | 41–130 | 569.5 / 568.6 | 41.702 / 41.768 | 150.3 |

The Pro index rows report each trial's median separately. The table here uses
model FLOPs divided by mean measured step time; extra recompute work is excluded
from the FLOPs numerator. Memory includes initialization and graph capture, and
is the maximum allocated high-water mark over all ranks and both trials.
All four 256-GPU runs completed the requested iterations with finite logged
losses/gradient norms, no skipped/NaN iterations, clean all-rank log audits, and
FLOPs consistent with independent fixed-length accounting. The 64K recipe runs 130 iterations to provide a stable
90-step window after warmup; it retains the measured 100-step LR decay schedule.
