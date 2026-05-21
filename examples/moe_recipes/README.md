# MoE Recipes

This directory contains self-contained MoE training recipes. Each YAML file includes:

- `DEPENDENCIES`: PyTorch base image and Dockerfile content used for the recipe.
- `ENV_VARS`: Environment variables expected by the runtime.
- `ARGS`: Megatron-LM training arguments.

## Recipe Overview

| Model | Hardware | GPUs | Precision | Parallelism | Batch | Seq Len | PyTorch Base | Key Features | Recipe |
|-------|----------|------|-----------|-------------|-------|---------|--------------|--------------|--------|
| DeepSeek-V3 | B200 | 256 | MXFP8 | TP1 PP8 EP32 CP1 ETP1 | MBS1 GBS2048 | 4096 | `nvcr.io/nvidia/pytorch:26.03-py3` | DeepEP dispatcher, MoE overlap | [`deepseek_v3/b200/mxfp8_256GPU_TP1PP8EP32.yaml`](deepseek_v3/b200/mxfp8_256GPU_TP1PP8EP32.yaml) |
| DeepSeek-V3 | GB200 | 256 | MXFP8 | TP1 PP4 EP64 CP1 ETP1 | MBS1 GBS8192 | 4096 | `nvcr.io/nvidia/pytorch:26.03-py3` | HybridEP, partial CUDA graph, MoE overlap, activation offload | [`deepseek_v3/gb200/mxfp8_256GPU_TP1PP4EP64.yaml`](deepseek_v3/gb200/mxfp8_256GPU_TP1PP4EP64.yaml) |
| DeepSeek-V3 | GB300 | 256 | MXFP8 | TP1 PP4 EP64 CP1 ETP1 | MBS1 GBS8192 | 4096 | `nvcr.io/nvidia/pytorch:26.03-py3` | HybridEP, partial CUDA graph, MoE overlap | [`deepseek_v3/gb300/mxfp8_256GPU_TP1PP4EP64.yaml`](deepseek_v3/gb300/mxfp8_256GPU_TP1PP4EP64.yaml) |
| DeepSeek-V3 | H100 | 1024 | BF16 | TP1 PP16 EP64 CP1 ETP1 | MBS1 GBS8192 | 4096 | `nvcr.io/nvidia/pytorch:26.03-py3` | Baseline BF16 large-scale recipe | [`deepseek_v3/h100/bf16_1024GPU_TP1PP16EP64.yaml`](deepseek_v3/h100/bf16_1024GPU_TP1PP16EP64.yaml) |
| DeepSeek-V3 | H100 | 1024 | FP8 | TP2 PP8 EP64 CP1 ETP1 | MBS1 GBS8192 | 4096 | `nvcr.io/nvidia/pytorch:26.03-py3` | DeepEP dispatcher, MoE overlap | [`deepseek_v3/h100/fp8_1024GPU_TP2PP8EP64.yaml`](deepseek_v3/h100/fp8_1024GPU_TP2PP8EP64.yaml) |
| Qwen3-235B-A22B | GB200 | 128 | MXFP8 | TP1 PP1 EP64 CP1 ETP1 | MBS1 GBS8192 | 4096 | `nvcr.io/nvidia/pytorch:26.04-py3` | Paged stash, full CUDA graph, HybridEP, MoE overlap | [`qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_paged_stash_fullcg_overlap.yaml`](qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_paged_stash_fullcg_overlap.yaml) |
| Qwen3-235B-A22B | GB200 | 128 | MXFP8 | TP1 PP1 EP64 CP1 ETP1 | MBS1 GBS8192 | 4096 | `nvcr.io/nvidia/pytorch:26.03-py3` | Partial CUDA graph, HybridEP, MoE overlap | [`qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_partial_cg_overlap.yaml`](qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_partial_cg_overlap.yaml) |
| Qwen3-235B-A22B | GB300 | 128 | MXFP8 | TP1 PP1 EP64 CP1 ETP1 | MBS1 GBS8192 | 4096 | `nvcr.io/nvidia/pytorch:26.04-py3` | Paged stash, full CUDA graph, HybridEP, MoE overlap | [`qwen3_235b/gb300/mxfp8_128GPU_TP1PP1EP64_paged_stash_full_cg.yaml`](qwen3_235b/gb300/mxfp8_128GPU_TP1PP1EP64_paged_stash_full_cg.yaml) |
| Qwen3-235B-A22B | H100 | 256 | BF16 | TP2 PP8 EP32 CP1 ETP1 | MBS1 GBS2048 | 4096 | `nvcr.io/nvidia/pytorch:26.03-py3` | Router/preprocess CUDA graph, HybridEP, MoE overlap | [`qwen3_235b/h100/bf16_256GPU_TP2PP8EP32.yaml`](qwen3_235b/h100/bf16_256GPU_TP2PP8EP32.yaml) |

Legend: TP = tensor parallel, PP = pipeline parallel, EP = expert parallel, CP = context parallel, ETP = expert tensor parallel, MBS = micro-batch size, GBS = global batch size.
