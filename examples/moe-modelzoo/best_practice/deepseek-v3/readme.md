# Reproduction Guide for DeepSeek-V3 SOTA Training Performance

This document provides a step-by-step guide to reproducing the state-of-the-art (SOTA) training performance of the DeepSeek-V3 model using the Megatron-Core framework. It covers experiment setup, configuration details, and best practices to help you achieve optimal results on large-scale distributed NVIDIA GPU systems.


## Guide on H100 GPUs (4096 sequence lengths)

This section will provide detailed instructions and recommendations for running DeepSeek-V3 training on NVIDIA H100 GPU clusters. It will include hardware-specific configuration tips, performance tuning strategies, and best practices to maximize throughput and efficiency on H100-based clusters.

### Preparation

#### Environment
Please refer to this [README](../../README.md) document to setup your cluster, build your container image, and convert the DeepSeek-V3 checkpoint.

#### Get `bindpcie` (Optional, Recommended)
Using `bindpcie` can help reduce the CPU overhead, and it is recommended to install.
```
cd ${PATH_TO_BINDPCIE}
wget https://raw.githubusercontent.com/NVIDIA/mlperf-common/refs/heads/main/client/bindpcie
chmod +x bindpcie
export BINDPCIE_PATH=${PATH_TO_BINDPCIE}/bindpcie
```

### Launch the Training Benchmark Script
- Fill up the environment variables in the `run.sh`
- In the repo root directory run `bash best-practice/DeepSeekV3/run.sh`

### Key Configurations for DeepSeek-V3 Training Performance
- MoE related optimizations
    - `--moe-router-dtype fp32 --moe-permute-fusion --moe-grouped-gemm --moe-router-fusion`
- Enable DeepEP
    - `--moe-token-dispatcher-type flex --moe-enable-deepep`
- Enable fine grained recomputations
    - `--recompute-granularity selective --recompute-modules mla_up_proj mlp`
- Enable CE Loss fusion
    - `--cross-entropy-loss-fusion --cross-entropy-fusion-impl te`
- Enable DeepSeek FP8 recipe
    - `--fp8-recipe blockwise --fp8-format e4m3`
- Enable FP8 primary weights
    - `--fp8-param-gather`
- Enable BF16 optimizer states
    - `--use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16`
- Enable manual GC to avoid unexpected GC within iteration, which can cause slowdowns.
    - `--manual-gc --manual-gc-interval 10`
- Enable EP A2A overlapping
    - `--overlap-moe-expert-parallel-comm --delay-wgrad-compute`
- Add MTP layer
    - `--mtp-num-layers: 1 --mtp-loss-scaling-factor: 0.1`
- Use flexible PP layout
    - `--pipeline-model-parallel-layout "Et|(tt|)*30mL"`
- Experimental features `--enable-experimental`
    - MLA RoPE Fusion: do not add `--no-rope-fusion`
    - Fused indices to multihot (for DeepEP): enabled when `--permute-fusion` is added
    - Pad routing map (for FP8 training) `--moe-router-padding-for-fp8`
- Force balance (for benchmark only) `--moe-router-force-load-balancing`

## Guide on GB200 GPUs (4096 sequence lengths)

This section will provide detailed instructions and recommendations for running DeepSeek-V3 training on NVIDIA GB200 GPU clusters.

This document introduces how we optimize DeepSeek-V3 pretraining performance on GB200 cluster [Optimizing DeepSeek-V3 Training Performance on NVIDIA GB200 NVL72](https://github.com/NVIDIA/Megatron-LM/blob/dev/docs/discussions/deepseek-v3-gb200-optimization/deepseek-v3-gb200-optimization.md).

This document demonstrates the complete workflow for reproducing performance, but please refer to the modelzoo for specific software versions and MCore args settings, as this guide will not be updated frequently [A Guide to Reproduce DeepSeek-V3 Pre-training Performance on GB200](https://github.com/NVIDIA/Megatron-LM/blob/dev/docs/discussions/deepseek-v3-gb200-optimization/deepseek-v3-gb200-reproduce-guide.md).

### Preparation

#### Environment
Please refer to this [README](../../README.md) document to setup your cluster, build your container image, and convert the DeepSeek-V3 checkpoint.

#### Get `bindpcie` (Optional, Recommended)
Using `bindpcie` can help reduce the CPU overhead, and it is recommended to install.
```
cd ${PATH_TO_BINDPCIE}
wget https://raw.githubusercontent.com/NVIDIA/mlperf-common/refs/heads/main/client/bindpcie
chmod +x bindpcie
export BINDPCIE_PATH=${PATH_TO_BINDPCIE}/bindpcie
```

### Launch the Training Benchmark Script
- Fill up the environment variables in the `run_gb200.sh`
- In the repo root directory run `bash best-practice/DeepSeekV3/run_gb200.sh`

### Another Reproduce Guide
To reproduce the performance, you can also refer to this [guide](https://github.com/NVIDIA/Megatron-LM/blob/dev/docs/discussions/deepseek-v3-gb200-optimization/deepseek-v3-gb200-reproduce-guide.md).
