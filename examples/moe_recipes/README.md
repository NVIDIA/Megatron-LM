# MoE Recipes

Each recipe contains a self-contained container definition, runtime environment
variables, and Megatron-LM arguments.

The DeepSeek V4 recipe is launched with `pretrain_hybrid.py`. Set `LOAD_PATH`
and `OUTPUT_PATH` before rendering its arguments.

| Model | Recipe | GPUs | TP/PP/EP/CP/ETP | MBS/GBS/SL | Features |
|---|---|---:|---|---|---|
| DeepSeek-V4-Flash | [GB200 MXFP8 THD 64K](deepseek_v4_flash/gb200/mxfp8_THD_SL64K_128GPU_TP1PP2EP64CP16.yaml) | 128 | 1/2/64/16/1 | 1/128/65536 | Hybrid attention; THD packing; mHC; MTP; HybridEP; scoped TE graphs; fine-grained activation offload |

TP = tensor parallel, PP = pipeline parallel, EP = expert parallel, CP =
context parallel, and ETP = expert tensor parallel.
