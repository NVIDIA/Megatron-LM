# A Guide to Reproduce DeepSeek-V3 Pre-training Performance on GB200

## 1. Dockerfile

Requirements:
- Transformer Engine: We recommend using commit [d2945c6](https://github.com/NVIDIA/TransformerEngine/commit/d2945c6a571e3978677614d1fe08779966a5a4ef) with PR [2146](https://github.com/NVIDIA/TransformerEngine/pull/2146) and [2150](https://github.com/NVIDIA/TransformerEngine/pull/2150). You could prepare the branch by yourself, or use this [branch](https://github.com/hxbai/TransformerEngine/commits/dev_20251024/) based on TE v2.9 plus the above three commits/PRs.
- cuDNN: v9.14 is required.
- HybridEP: Install it from [here](https://github.com/deepseek-ai/DeepEP/commits/3f601f7ac1c062c46502646ff04c535013bfca00).

Dockerfile for reference.

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.09-py3 AS base

ENV SHELL=/bin/bash

# =========================
# Install system packages
# =========================
RUN rm -rf /opt/megatron-lm && \
    apt-get update && \
    apt-get install -y sudo gdb bash-builtins git zsh autojump tmux curl gettext libfabric-dev && \
    wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_arm64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

# =========================
# Install Python packages
# =========================
# NOTE: `unset PIP_CONSTRAINT` to install packages that do not meet the default constraint in the base image.
# Some package requirements and related versions are from 
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/Dockerfile.linting.
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/requirements_mlm.txt.
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/requirements_ci.txt.
RUN unset PIP_CONSTRAINT && pip install --no-cache-dir debugpy dm-tree torch_tb_profiler einops wandb \
    sentencepiece tokenizers transformers torchvision ftfy modelcards datasets tqdm pydantic \
    nvidia-pytriton py-spy yapf darker \
    tiktoken flask-restful \
    nltk wrapt pytest pytest_asyncio pytest-cov pytest_mock pytest-random-order \
    black==24.4.2 isort==5.13.2 flake8==7.1.0 pylint==3.2.6 coverage mypy \
    setuptools==69.5.1

# =========================
# Install cudnn 9.14.0.64 for correct mxfp8 quantization and layernorm fusion
# =========================
RUN apt-get update && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install libcudnn9-cuda-13

# =========================
# Install latest TE
# Use a specific commit instead of main to make it more stable.
# This is based on release_v2.9 branch and contains some CPU and quantization optimizations.
# =========================
ARG COMMIT="7dd3914726abb79bc99ff5a5db1449458ed64151"
ARG TE="git+https://github.com/hxbai/TransformerEngine.git@${COMMIT}"
RUN pip install nvidia-mathdx==25.1.1 && \
    unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="100" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch pip install --no-build-isolation --no-cache-dir $TE

# =========================
# Install HybridEP
# =========================
WORKDIR /home/
RUN git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && git checkout 3f601f7ac1c062c46502646ff04c535013bfca00 && \
    TORCH_CUDA_ARCH_LIST="10.0" pip install --no-build-isolation .

# =========================
# Clean cache
# =========================
RUN rm -rf /root/.cache /tmp/*
```

> [!Tip]
>
> If you prefer to use CUDA 12.9, please change the base container to `nvcr.io/nvidia/pytorch:25.06-py3` and the cuDNN to be installed to `libcudnn9-cuda-12`. 

## 2. Megatron-Core

We recommend using the [dev branch](https://github.com/NVIDIA/Megatron-LM/tree/dev) after PR [1917](https://github.com/NVIDIA/Megatron-LM/pull/1917).

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git && \
cd Megatron-LM &&
git checkout effebd81f410bc6566fffee6c320b6f8f762e06d
```

## 3. Cluster Configuration

Since we're using EP 32 on NVL72, it's important to make sure

> [!Important]
> **Every 32 GB200 GPUs (8 nodes) are in the same NVL domain (or rack)**.

Usually you can make it via your cluster workload manager. Taking Slurm as an example, you could pass `--segment 8` to the sbatch command to ensure that every segment of 8 nodes will be scheduled to a rack.

## 4. Training scripts

### Environment variables

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1
NVTE_FWD_LAYERNORM_SM_MARGIN=0
NVTE_BWD_LAYERNORM_SM_MARGIN=0
NVLINK_DOMAIN_SIZE=72
NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_NVLS_ENABLE=0
NVTE_FUSED_ATTN=1
NVTE_NORM_FWD_USE_CUDNN=1
NVTE_NORM_BWD_USE_CUDNN=1
PYTHONWARNINGS=ignore
NCCL_DEBUG=VERSION
NCCL_GRAPH_REGISTER=0
```

### bindpcie

Download [bindpcie](https://github.com/NVIDIA/mlperf-common/blob/main/client/bindpcie) to your workdir, make it executable, 

```bash
wget https://raw.githubusercontent.com/NVIDIA/mlperf-common/refs/heads/main/client/bindpcie &&
chmod 755 bindpcie
```

and then

> [!Important]
> **Place it at the beginning of your launch command in every process.**

Taking Slurm as an example, your script should look like

```bash
#!/bin/bash

#SBATCH [... sbatch args]

srun [... srun args] /path/to/bindpcie /path/to/pretrain_gpt.py [... mcore arguments]
```

This is a very important step on GB200.

### Launch script

```bash
/path/to/bindpcie \
/path/to/megatron-lm/pretrain_gpt.py \
--distributed-timeout-minutes 60 \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 8 \
--expert-model-parallel-size 32 \
--context-parallel-size 1 \
--expert-tensor-parallel-size 1 \
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather \
--use-mcore-models \
--sequence-parallel \
--use-flash-attn \
--disable-bias-linear \
--micro-batch-size 1 \
--global-batch-size 2048 \
--train-samples 585937500 \
--exit-duration-in-mins 220 \
--no-save-optim \
--no-check-for-nan-in-loss-and-grad \
--cross-entropy-loss-fusion \
--cross-entropy-fusion-impl te \
--manual-gc \
--manual-gc-interval 10 \
--enable-experimental \
--transformer-impl transformer_engine \
--seq-length 4096 \
--data-cache-path /path/to/data_cache \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model unsloth/DeepSeek-V3 \
--data-path /path/to/data \
--split 99,1,0 \
--no-mmap-bin-files \
--no-create-attention-mask-in-dataloader \
--num-workers 6 \
--num-layers 61 \
--hidden-size 7168 \
--ffn-hidden-size 18432 \
--num-attention-heads 128 \
--kv-channels 128 \
--max-position-embeddings 4096 \
--position-embedding-type rope \
--rotary-base 10000 \
--make-vocab-size-divisible-by 3232 \
--normalization RMSNorm \
--norm-epsilon 1e-6 \
--swiglu \
--untie-embeddings-and-output-weights \
--multi-latent-attention \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--clip-grad 1.0 \
--weight-decay 0.1 \
--qk-layernorm \
--lr-decay-samples 584765624 \
--lr-warmup-samples 1536000 \
--lr-warmup-init 3.9e-7 \
--lr 3.9e-6 \
--min-lr 3.9e-7 \
--lr-decay-style cosine \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--num-experts 256 \
--moe-layer-freq ([0]*3+[1]*58) \
--moe-ffn-hidden-size 2048 \
--moe-shared-expert-intermediate-size 2048 \
--moe-router-load-balancing-type seq_aux_loss \
--moe-router-topk 8 \
--moe-grouped-gemm \
--moe-aux-loss-coeff 1e-4 \
--moe-router-group-topk 4 \
--moe-router-num-groups 8 \
--moe-router-pre-softmax \
--moe-router-padding-for-quantization \
--moe-router-topk-scaling-factor 2.5 \
--moe-router-score-function sigmoid \
--moe-router-enable-expert-bias \
--moe-router-bias-update-rate 1e-3 \
--moe-router-dtype fp32 \
--moe-permute-fusion \
--moe-router-fusion \
--q-lora-rank 1536 \
--kv-lora-rank 512 \
--qk-head-dim 128 \
--qk-pos-emb-head-dim 64 \
--v-head-dim 128 \
--rotary-scaling-factor 40 \
--mscale 1.0 \
--mscale-all-dim 1.0 \
--eval-iters 32 \
--eval-interval 200 \
--no-load-optim \
--no-load-rng \
--auto-detect-ckpt-format \
--load None \
--save /path/to/checkpoints \
--save-interval 500 \
--dist-ckpt-strictness log_all \
--init-method-std 0.02 \
--log-timers-to-tensorboard \
--log-memory-to-tensorboard \
--log-validation-ppl-to-tensorboard \
--log-throughput \
--log-interval 1 \
--logging-level 40 \
--tensorboard-dir /path/to/tensorboard \
--wandb-project deepseek-v3-benchmarking-v0.15 \
--wandb-exp-name DeepSeek-V3-TP1PP8EP32CP1VPP4-MBS1GBS2048-v0.15 \
--bf16 \
--enable-experimental \
--recompute-granularity selective \
--recompute-modules moe_act mlp \
--cuda-graph-impl transformer_engine \
--cuda-graph-scope attn moe_router moe_preprocess \
--te-rng-tracker \
--pipeline-model-parallel-layout "Et|(tt|)*30L" \
--moe-router-force-load-balancing \
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend hybridep \
--moe-hybridep-num-sms 32 \
--fp8-recipe mxfp8 \
--fp8-format e4m3 \
--fp8-param-gather \
--reuse-grad-buf-for-mxfp8-param-ag \
--use-precision-aware-optimizer \
--main-grads-dtype fp32 \
--main-params-dtype fp32 \
--exp-avg-dtype bf16 \
--exp-avg-sq-dtype bf16 \
```

### Explanation of arguments

The following arguments indicate key optimizations.

- Pipeline parallel layout

```bash
--pipeline-model-parallel-layout "Et|(tt|)*30L"
```

`E` stands for embedding, `t` for transformer layer, `L` for Loss. So it's interpreted as a total of 32 stages, where the first stage is Embedding + 1 transformer layer, the last stage is Loss, and the middle 30 stages are 2 transformer layers.

- Fine-grained recompute

```bash
--recompute-granularity selective \
--recompute-modules moe_act mlp \
```

- Partial CUDA Graphs

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-scope attn moe_router moe_preprocess \
--te-rng-tracker \
```

- Force load balancing for performance benchmark

```bash
--moe-router-force-load-balancing \
```

- HybridEP

```bash
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend hybridep \
--moe-hybridep-num-sms 32 \
```

- MXFP8 recipe

```bash
--fp8-recipe mxfp8 \
--fp8-format e4m3 \
--fp8-param-gather \
--reuse-grad-buf-for-mxfp8-param-ag \
```

- BF16 optimizer states

```bash
--use-precision-aware-optimizer \
--main-grads-dtype fp32 \
--main-params-dtype fp32 \
--exp-avg-dtype bf16 \
--exp-avg-sq-dtype bf16 \
```

- Kernel fusions

```bash
--cross-entropy-loss-fusion \
--cross-entropy-fusion-impl te \
--moe-permute-fusion \
--moe-router-fusion \
```

- Manual GC to make ranks better synchronized

```bash
--manual-gc \
--manual-gc-interval 10 \
```
