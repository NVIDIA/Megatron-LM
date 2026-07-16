# Megatron MoE Model Zoo

Production-ready training recipes for state-of-the-art MoE models — **DeepSeek-V3**, **Qwen3**, and **Mixtral** — built on 🚀 [Megatron-Core DEV branch](https://github.com/NVIDIA/Megatron-LM/tree/dev).

✅ Performance-tuned configs for **H100, B200, GB200, and GB300** clusters  
✅ Model-specific best practices for training MoE models
✅ One-command launch with sensible defaults  
✅ Dry-run mode to validate arguments before submitting jobs

## Best Practices

Ready-to-run scripts with optimized configurations:

| Model | Hardware | Scripts |
|-------|----------|---------|
| DeepSeek-V3 | H100, B200, GB200, GB300 | [`best_practice/deepseek-v3/`](./best_practice/deepseek-v3/) |
| Qwen3 | H100, GB200, GB300 | [`best_practice/qwen3/`](./best_practice/qwen3/) |
| Mixtral | H100 | [`best_practice/mixtral/`](./best_practice/mixtral/) |

See [`best_practice/`](./best_practice/) for detailed guides.

## Quick Start

### Prerequisites

Install `yq` for YAML processing (one-time setup):

```bash
mkdir -p ~/.local/bin && wget -qO ~/.local/bin/yq https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_amd64 && chmod +x ~/.local/bin/yq
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MEGATRON_PATH` | Path to Megatron-LM | `/path/to/Megatron-LM` |
| `CONTAINER_IMAGE` | Container image path | `/path/to/image.sqsh` |
| `CLUSTER` | Name of the cluster; used to load cluster-specific settings such as data paths | `EOS`, `CW` |
| `WANDB_API_KEY` | (Optional) WandB key | From [wandb.ai/authorize](https://wandb.ai/authorize) |

### Container

Dockerfile: [`dockers/H100.Dockerfile`](./dockers/H100.Dockerfile) (also available: `B200.Dockerfile`, `GB200.Dockerfile`)

## Performance Benchmarking

### Supported Models

`Mixtral-8x2B`, `Mixtral-8x7B`, `Mixtral-8x22B`, `DeepSeek-V2`, `DeepSeek-V2-Lite`, `DeepSeek-V3`, `Qwen2-57B-A14B`, `Qwen3-235B-A22B`, `Qwen3-30B-A3B`, `Qwen3-Next-80B-A3B`

### Launch

Basic launch:

```bash
MODEL=DeepSeek-V3 bash ./sbatch_benchmarking.sh
```

With custom/overwritten parameters:

```bash
MODEL=DeepSeek-V3 TP=2 PP=8 EP=64 VPP=1 RUN_TIME=00:60:00 NNODES=64 \
  bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj layernorm
```

> **💡 Tip: Dry Run Mode** — Preview the generated SLURM script and training command without submitting to the cluster:
> ```bash
> DRY_RUN=1 MODEL=DeepSeek-V3 bash ./sbatch_benchmarking.sh
> ```
> This is highly recommended before submitting jobs to verify configurations.

### Configuration

**Runtime configs** ([`runtime_configs/benchmarking/runtime.conf`](./runtime_configs/benchmarking/runtime.conf)):
- Parallelism: `TP`, `PP`, `EP`, `CP`, `VPP`, `PP_FIRST`, `PP_LAST`, `LAYERS_PER_VP`
- Batch sizes: `MBS`, `GBS`
- Training: `NNODES`, `RUN_TIME`, `NUM_LAYERS`, `SEQ_LEN`
- MoE: `MOE_TOKEN_DISPATCHER`, `MOE_GROUPED_GEMM`

**Cluster configs** ([`cluster_configs/benchmarking/template.conf`](./cluster_configs/benchmarking/template.conf)):
- Slurm: `ACCOUNT`, `PARTITION`, `RUN_NAME`, `CONTAINER_MOUNTS`
- Paths: `OUTPUT_PATH`, `DATA_PATH`, `TOKENIZER_MODEL`, `LOAD_PATH`

### Job Monitoring

```bash
watch -n 1 squeue -u $USER
```

## Checkpoint Conversion

> For HF↔MCore conversion, consider [MBridge](https://github.com/ISEEKYAN/mbridge/) or [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).

### DeepSeek-V3

**1. Download and convert to BF16:**

```bash
git lfs install && git clone https://huggingface.co/deepseek-ai/DeepSeek-V3
python inference/fp8_cast_bf16.py --input-fp8-hf-path /input/fp8/path --output-bf16-hf-path /output/bf16/path
```

**2. Convert to Megatron legacy checkpoint:**

```bash
MODEL=DeepSeek-V3 bash ./ckpt_convert_scripts/DeepSeek-V3/convert_deepseek_v3.sh
```

**3. Convert to distributed checkpoint:**

```bash
MODEL=DeepSeek-V3 TP=1 PP=4 EP=64 VPP=1 PP_FIRST=16 PP_LAST=13 NNODES=32 LOAD_PATH=/path/to/legacy/ckpt \
  bash ./sbatch_benchmarking.sh --ckpt-convert-save /path/to/dist/ckpt --ckpt-convert-format torch_dist --no-save-optim
```

Storage: Legacy ~3.4T, Distributed ~1.4T
