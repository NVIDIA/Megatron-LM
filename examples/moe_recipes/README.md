# MoE Recipes

This directory contains self-contained MoE training recipes. Each YAML file includes:

- `DEPENDENCIES`: PyTorch base image and Dockerfile content used for the recipe.
- `ENV_VARS`: Environment variables expected by the runtime.
- `ARGS`: Megatron-LM training arguments.

## Recipe Index

| Recipe | GPUs | Parallelism | Batch / Seq |
|--------|------|-------------|-------------|
| [DeepSeek-V3 B200 MXFP8](deepseek_v3/b200/mxfp8_256GPU_TP1PP8EP32.yaml) | 256 | TP1 PP8 EP32 CP1 ETP1 | MBS1 GBS2048 SL4096 |
| [DeepSeek-V3 GB200 MXFP8](deepseek_v3/gb200/mxfp8_256GPU_TP1PP4EP64.yaml) | 256 | TP1 PP4 EP64 CP1 ETP1 | MBS1 GBS8192 SL4096 |
| [DeepSeek-V3 GB300 MXFP8](deepseek_v3/gb300/mxfp8_256GPU_TP1PP4EP64.yaml) | 256 | TP1 PP4 EP64 CP1 ETP1 | MBS1 GBS8192 SL4096 |
| [DeepSeek-V3 H100 BF16](deepseek_v3/h100/bf16_1024GPU_TP1PP16EP64.yaml) | 1024 | TP1 PP16 EP64 CP1 ETP1 | MBS1 GBS8192 SL4096 |
| [DeepSeek-V3 H100 FP8](deepseek_v3/h100/fp8_1024GPU_TP2PP8EP64.yaml) | 1024 | TP2 PP8 EP64 CP1 ETP1 | MBS1 GBS8192 SL4096 |
| [Qwen3-235B GB200 MXFP8 full CG](qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_paged_stash_fullcg_overlap.yaml) | 128 | TP1 PP1 EP64 CP1 ETP1 | MBS1 GBS8192 SL4096 |
| [Qwen3-235B GB200 MXFP8 partial CG](qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_partial_cg_overlap.yaml) | 128 | TP1 PP1 EP64 CP1 ETP1 | MBS1 GBS8192 SL4096 |
| [Qwen3-235B GB300 MXFP8 full CG](qwen3_235b/gb300/mxfp8_128GPU_TP1PP1EP64_paged_stash_full_cg.yaml) | 128 | TP1 PP1 EP64 CP1 ETP1 | MBS1 GBS8192 SL4096 |
| [Qwen3-235B H100 BF16](qwen3_235b/h100/bf16_256GPU_TP2PP8EP32.yaml) | 256 | TP2 PP8 EP32 CP1 ETP1 | MBS1 GBS2048 SL4096 |

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
