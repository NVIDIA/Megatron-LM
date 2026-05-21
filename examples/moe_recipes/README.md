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
      <th>Parallelism</th>
      <th>Batch / Seq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5">DeepSeek-V3</td>
      <td><a href="deepseek_v3/b200/mxfp8_256GPU_TP1PP8EP32.yaml">B200 MXFP8</a></td>
      <td>256</td>
      <td>TP1 PP8 EP32 CP1 ETP1</td>
      <td>MBS1 GBS2048 SL4096</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/gb200/mxfp8_256GPU_TP1PP4EP64.yaml">GB200 MXFP8</a></td>
      <td>256</td>
      <td>TP1 PP4 EP64 CP1 ETP1</td>
      <td>MBS1 GBS8192 SL4096</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/gb300/mxfp8_256GPU_TP1PP4EP64.yaml">GB300 MXFP8</a></td>
      <td>256</td>
      <td>TP1 PP4 EP64 CP1 ETP1</td>
      <td>MBS1 GBS8192 SL4096</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/h100/bf16_1024GPU_TP1PP16EP64.yaml">H100 BF16</a></td>
      <td>1024</td>
      <td>TP1 PP16 EP64 CP1 ETP1</td>
      <td>MBS1 GBS8192 SL4096</td>
    </tr>
    <tr>
      <td><a href="deepseek_v3/h100/fp8_1024GPU_TP2PP8EP64.yaml">H100 FP8</a></td>
      <td>1024</td>
      <td>TP2 PP8 EP64 CP1 ETP1</td>
      <td>MBS1 GBS8192 SL4096</td>
    </tr>
    <tr>
      <td rowspan="4">Qwen3-235B-A22B</td>
      <td><a href="qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_paged_stash_fullcg_overlap.yaml">GB200 MXFP8 full CG</a></td>
      <td>128</td>
      <td>TP1 PP1 EP64 CP1 ETP1</td>
      <td>MBS1 GBS8192 SL4096</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_partial_cg_overlap.yaml">GB200 MXFP8 partial CG</a></td>
      <td>128</td>
      <td>TP1 PP1 EP64 CP1 ETP1</td>
      <td>MBS1 GBS8192 SL4096</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/gb300/mxfp8_128GPU_TP1PP1EP64_paged_stash_full_cg.yaml">GB300 MXFP8 full CG</a></td>
      <td>128</td>
      <td>TP1 PP1 EP64 CP1 ETP1</td>
      <td>MBS1 GBS8192 SL4096</td>
    </tr>
    <tr>
      <td><a href="qwen3_235b/h100/bf16_256GPU_TP2PP8EP32.yaml">H100 BF16</a></td>
      <td>256</td>
      <td>TP2 PP8 EP32 CP1 ETP1</td>
      <td>MBS1 GBS2048 SL4096</td>
    </tr>
  </tbody>
</table>

## Recipe Notes

### DeepSeek-V3

- **B200 MXFP8**: PyTorch `nvcr.io/nvidia/pytorch:26.03-py3`; DeepEP dispatcher and MoE overlap.
- **GB200 MXFP8**: PyTorch `nvcr.io/nvidia/pytorch:26.03-py3`; HybridEP, partial CUDA graph, MoE overlap, and activation offload.
- **GB300 MXFP8**: PyTorch `nvcr.io/nvidia/pytorch:26.03-py3`; HybridEP, partial CUDA graph, and MoE overlap.
- **H100 BF16**: PyTorch `nvcr.io/nvidia/pytorch:26.03-py3`; baseline large-scale BF16 recipe.
- **H100 FP8**: PyTorch `nvcr.io/nvidia/pytorch:26.03-py3`; DeepEP dispatcher and MoE overlap.

### Qwen3-235B-A22B

- **GB200 MXFP8 full CG**: PyTorch `nvcr.io/nvidia/pytorch:26.04-py3`; paged stash, full CUDA graph, HybridEP, and MoE overlap.
- **GB200 MXFP8 partial CG**: PyTorch `nvcr.io/nvidia/pytorch:26.03-py3`; partial CUDA graph, HybridEP, and MoE overlap.
- **GB300 MXFP8 full CG**: PyTorch `nvcr.io/nvidia/pytorch:26.04-py3`; paged stash, full CUDA graph, HybridEP, and MoE overlap.
- **H100 BF16**: PyTorch `nvcr.io/nvidia/pytorch:26.03-py3`; router/preprocess CUDA graph, HybridEP, and MoE overlap.

Legend: TP = tensor parallel, PP = pipeline parallel, EP = expert parallel, CP = context parallel, ETP = expert tensor parallel, MBS = micro-batch size, GBS = global batch size, SL = sequence length, CG = CUDA graph.
