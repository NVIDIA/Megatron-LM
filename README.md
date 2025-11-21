MoE Echo: Elastic Cloning for Hot Experts 🚀
===========================

## Overview 🔍
MoE Echo is a research prototype of a new MoE training paradigm that targets large-scale distributed training. It focuses on achieving **load balance** and **sync-free** with **Fully CUDA-graph-capturable** on **dropless** MoE training.

Concretely, MoE Echo aims to:
- **⚖️ Reduce expert load imbalance** across Expert Parallel (EP) ranks.
- **⏱️ Remove host-side synchronization** with dynamic routing in dropless MoE.
- **📊 Enable CUDA-graph-capturable MoE** with minimal compute and memory fragmentation.

## Sync-Free MoE ⚡
In token-dropless MoE, the number of tokens sent to each EP rank can vary significantly from step to step. The routing decisions (and thus the per-rank shapes) are produced on the GPU, but the host traditionally needs this shape information to:
- launch dispatch/combine and grouped GEMM kernels, and  
- allocate sufficient memory for these kernels.

Naively, this requires device-to-host copies and host-side synchronization on every step, which both slows down training and makes CUDA graph capture difficult.

To build a **sync-free**, CUDA-graph-friendly MoE, we:
- **Pre-allocate GPU buffers** and decide kernel launches without waiting for host-visible shapes.
- Avoid excessive over-provisioning of buffers, which would otherwise cause:
  - **Compute fragmentation** (wasted compute/communication on padded tokens).
  - **Memory fragmentation** (oversized static buffers).

MoE Echo tackles this by:
- **Reducing compute fragmentation**: GPU kernels consume routing/shape information that stays on device and operate only on the true token volume. For example, HybridEP reads shapes directly from the routing map on GPU.
- **Reducing memory fragmentation**: We reduce load imbalance across EP ranks and manage memory more efficiently inside CUDA graphs so the pre-allocated buffers are better utilized.

## Elastic Cloning for Hot Experts (ECHO)
To further reduce expert load imbalance, MoE Echo introduces **elastic cloning for hot experts (ECHO)**. The key idea is to dynamically clone high-traffic (“hot”) experts onto EP ranks that receive fewer-than-average tokens.

Cloning experts during training is challenging because expert **weights and gradients must remain coherent** across all clones at every step. This means:
- Synchronizing cloned expert parameters and gradients.
- Carefully limiting the number of cloned experts to balance extra communication cost against the load-balance benefit.

MoE Echo addresses this with:
- An **ECHO planner** that decides which popular experts to clone and where to place them, given spare expert slots on each EP rank.
- An **ECHO dispatcher** that:
  - dispatches tokens to the appropriate cloned experts and their spare slots according to the plan, and
  - during backward, handles any necessary re-dispatch when spare slots are shared across layers and then combines/reduces gradients from all cloned copies into the main expert.


## Quick Start 🏁
### Install Dependencies
#### HybridEP
```
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP & git checkout hybrid-ep
TORCH_CUDA_ARCH_LIST="10.0" pip install -e .
```
#### Device-inited-grouped gemm
Note that this kernel is only available for Blackwell GPUs.
```
git clone https://github.com/QiZhangNV/TransformerEngine.git
cd TransformerEngine & git checkout cutlass_device_grouped_gemm
git submodule update --init --recursive
NVTE_CUDA_ARCHS="100a" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch pip install --no-cache-dir --no-build-isolation .
```

### Run MoE Echo ▶️
Add the following flags to the command line to enable Echo for your training:

```
--moe-enable-echo
--moe-num-echo-experts 32 # number of echo experts totally
--moe-echo-expert-dispatcher-type hybridep # Only hybridep support sync-free dispatch
--moe-received-token-capacity 2.0 # capacity of total received tokens on each ep rank (if not set, sync-version will be used)
--moe-use-device-initiated-grouped-gemm # use device-initiated grouped gemm (only available for Blackwell GPUs MXFP8 GEEM)
--fp8-format e4m3
--fp8-recipe mxfp8
--fp8-param-gather
--reuse-grad-buf-for-mxfp8-param-ag
--moe-echo-recompute-expert-dispatch # recompute expert dispatch, such that the echo expert buffer is shared across layers
# Enable CUDA Graph
--enable-cuda-graph 
--cuda-graph-scope full_iteration
--te-rng-tracker
```


## Roadmap 🗺️

- [x] Sync-free GroupedGEMM
- [x] Sync-free token and expert dispatcher
- [x] Planner for MoE Echo
- [x] Expert Dispatcher for MoE Echo
- [x] Full-iteration CUDA Graph
- [ ] E2E examples
- [ ] Add E2E performance benchmark
- [ ] Add expert dispatch overlapping
- [ ] Activation stashing to reduce memory fragmentation
- [ ] Activation CPU offloading 


## Acknowledgments 🙏
- Contributors(**Equal Contribution, sorted alphabetically**): Ahan Huang, Dennis Liu, Nan Zheng, Patrick Haft, Qi Zhang, Robin Zhang, Tong Liu, Zijie Yan