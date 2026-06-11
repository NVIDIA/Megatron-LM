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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DeepSeek-V4-Flash</td>
      <td><a href="deepseek_v4_flash/gb200/mxfp8_SL4K_128GPU_TP1PP1EP64.yaml">GB200 MXFP8</a></td>
      <td>128</td>
      <td>1/1/64/1/1</td>
      <td>1/2048/4096</td>
      <td>BSHD; paged stash; full CG; HybridEP</td>
    </tr>
    <tr>
      <td rowspan="5">DeepSeek-V3</td>
      <td><a href="deepseek_v3/b200/mxfp8_256GPU_TP1PP8EP32.yaml">B200 MXFP8</a></td>
      <td>256</td>
      <td>1/8/32/1/1</td>
      <td>1/2048/4096</td>
      <td>DeepEP; EP overlap</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/gb200/mxfp8_256GPU_TP1PP4EP64.yaml">GB200 MXFP8</a></td>
      <td>256</td>
      <td>1/4/64/1/1</td>
      <td>1/8192/4096</td>
      <td>HybridEP; partial CG; EP overlap; offload</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/gb300/mxfp8_256GPU_TP1PP4EP64.yaml">GB300 MXFP8</a></td>
      <td>256</td>
      <td>1/4/64/1/1</td>
      <td>1/8192/4096</td>
      <td>HybridEP; partial CG; EP overlap</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/h100/bf16_1024GPU_TP1PP16EP64.yaml">H100 BF16</a></td>
      <td>1024</td>
      <td>1/16/64/1/1</td>
      <td>1/8192/4096</td>
      <td>BF16 baseline</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/h100/fp8_1024GPU_TP2PP8EP64.yaml">H100 FP8</a></td>
      <td>1024</td>
      <td>2/8/64/1/1</td>
      <td>1/8192/4096</td>
      <td>DeepEP; EP overlap</td>
    </tr>
    <tr>
      <td rowspan="4">Qwen3-235B-A22B</td>
      <td><a href="qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_paged_stash_fullcg_overlap.yaml">GB200 MXFP8 full CG</a></td>
      <td>128</td>
      <td>1/1/64/1/1</td>
      <td>1/8192/4096</td>
      <td>paged stash; full CG; HybridEP; EP overlap</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_partial_cg_overlap.yaml">GB200 MXFP8 partial CG</a></td>
      <td>128</td>
      <td>1/1/64/1/1</td>
      <td>1/8192/4096</td>
      <td>partial CG; HybridEP; EP overlap</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/gb300/mxfp8_128GPU_TP1PP1EP64_paged_stash_full_cg.yaml">GB300 MXFP8 full CG</a></td>
      <td>128</td>
      <td>1/1/64/1/1</td>
      <td>1/8192/4096</td>
      <td>paged stash; full CG; HybridEP; EP overlap</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/h100/bf16_256GPU_TP2PP8EP32.yaml">H100 BF16</a></td>
      <td>256</td>
      <td>2/8/32/1/1</td>
      <td>1/2048/4096</td>
      <td>router/preprocess CG; HybridEP; EP overlap</td>
    </tr>
    <tr>
      <td rowspan="5">Qwen3-30B-A3B</td>
      <td><a href="qwen3_30b/h100/fp8_32GPU_TP1PP1EP8.yaml">H100 FP8</a></td>
      <td>32</td>
      <td>1/1/8/1/1</td>
      <td>1/256/4096</td>
      <td>FP8 blockwise; router/preprocess CG</td>
    </tr>
    <tr>
      <td><a href="qwen3_30b/h100/bf16_32GPU_TP1PP1EP8.yaml">H100 BF16</a></td>
      <td>32</td>
      <td>1/1/8/1/1</td>
      <td>1/256/4096</td>
      <td>BF16 baseline</td>
    </tr>
    <tr>
      <td><a href="qwen3_30b/gb200/bf16_16GPU_TP1PP1EP16.yaml">GB200 BF16</a></td>
      <td>16</td>
      <td>1/1/16/1/1</td>
      <td>4/512/4096</td>
      <td>BF16 baseline</td>
    </tr>
    <tr>
      <td><a href="qwen3_30b/gb200/mxfp8_16GPU_TP1PP1EP16_partial_cg.yaml">GB200 MXFP8 partial CG</a></td>
      <td>16</td>
      <td>1/1/16/1/1</td>
      <td>4/512/4096</td>
      <td>MXFP8; partial CG</td>
    </tr>
    <tr>
      <td><a href="qwen3_30b/gb200/mxfp8_16GPU_TP1PP1EP16_paged_stash.yaml">GB200 MXFP8 paged stash</a></td>
      <td>16</td>
      <td>1/1/16/1/1</td>
      <td>4/512/4096</td>
      <td>MXFP8; paged stash; full CG</td>
    </tr>
  </tbody>
</table>

Legend: TP = tensor parallel, PP = pipeline parallel, EP = expert parallel, CP = context parallel, ETP = expert tensor parallel, MBS = micro-batch size, GBS = global batch size, SL = sequence length, CG = CUDA graph. The tuple orders are `TP/PP/EP/CP/ETP` and `MBS/GBS/SL`.
