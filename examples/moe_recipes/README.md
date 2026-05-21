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
      <td rowspan="5">DeepSeek-V3</td>
      <td><a href="deepseek_v3/b200/mxfp8_256GPU_TP1PP8EP32.yaml">B200 MXFP8</a></td>
      <td>256</td>
      <td>1/8/32/1/1</td>
      <td>1/2048/4096</td>
      <td>DeepEP<br>MoE overlap</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/gb200/mxfp8_256GPU_TP1PP4EP64.yaml">GB200 MXFP8</a></td>
      <td>256</td>
      <td>1/4/64/1/1</td>
      <td>1/8192/4096</td>
      <td>HybridEP<br>partial CG<br>MoE overlap<br>offload</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/gb300/mxfp8_256GPU_TP1PP4EP64.yaml">GB300 MXFP8</a></td>
      <td>256</td>
      <td>1/4/64/1/1</td>
      <td>1/8192/4096</td>
      <td>HybridEP<br>partial CG<br>MoE overlap</td>
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
      <td>DeepEP<br>MoE overlap</td>
    </tr>
    <tr>
      <td rowspan="4">Qwen3-235B-A22B</td>
      <td><a href="qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_paged_stash_fullcg_overlap.yaml">GB200 MXFP8 full CG</a></td>
      <td>128</td>
      <td>1/1/64/1/1</td>
      <td>1/8192/4096</td>
      <td>paged stash<br>full CG<br>HybridEP<br>MoE overlap</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_partial_cg_overlap.yaml">GB200 MXFP8 partial CG</a></td>
      <td>128</td>
      <td>1/1/64/1/1</td>
      <td>1/8192/4096</td>
      <td>partial CG<br>HybridEP<br>MoE overlap</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/gb300/mxfp8_128GPU_TP1PP1EP64_paged_stash_full_cg.yaml">GB300 MXFP8 full CG</a></td>
      <td>128</td>
      <td>1/1/64/1/1</td>
      <td>1/8192/4096</td>
      <td>paged stash<br>full CG<br>HybridEP<br>MoE overlap</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/h100/bf16_256GPU_TP2PP8EP32.yaml">H100 BF16</a></td>
      <td>256</td>
      <td>2/8/32/1/1</td>
      <td>1/2048/4096</td>
      <td>router/preprocess CG<br>HybridEP<br>MoE overlap</td>
    </tr>
  </tbody>
</table>

Legend: TP = tensor parallel, PP = pipeline parallel, EP = expert parallel, CP = context parallel, ETP = expert tensor parallel, MBS = micro-batch size, GBS = global batch size, SL = sequence length, CG = CUDA graph. The tuple orders are `TP/PP/EP/CP/ETP` and `MBS/GBS/SL`.
