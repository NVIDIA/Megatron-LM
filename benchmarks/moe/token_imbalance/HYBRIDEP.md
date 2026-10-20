# HybridEP workload comparison

This follow-up to the logical routing benchmark uses the existing `MoELayer`,
HybridEP and `VirtualExpertLoadBalancer` interfaces. It does not define a future
MoEScheduler interface. Both adapters use TE grouped experts, FP32 router scores,
BF16 weights, TP=ETP=1 and the same initial logical expert parameters.

Run on one host with two NVLink-connected H100 GPUs and a HybridEP build with
compact top-k support:

```sh
NVTE_GROUPED_LINEAR_SINGLE_PARAM=0 torchrun --standalone --nproc_per_node=2 \
  benchmarks/moe/token_imbalance/compare_hybridep.py --out results/forward-backward
NVTE_GROUPED_LINEAR_SINGLE_PARAM=0 torchrun --standalone --nproc_per_node=2 \
  benchmarks/moe/token_imbalance/compare_hybridep.py --learning-rate 0.0001 \
  --out results/training-step
```

The default runs seven existing W workloads, six steps each, with two full route-sequence warmups per
adapter and workload. W6 rotates after three steps; W7 includes burst and recovery.
There is no W2 in the existing workload contract. Routes are replayed through the
real router; learned gate scores remain differentiable. Source token counts and
K stay fixed. Zero destination-expert traffic is included in W5; zero source rows,
recomputation, CUDA graphs and other precisions are outside this run.

Each rank writes complete per-step JSONL: logical expert/rank loads, actual route
and input hashes, output/gradient/parameter hashes, virtual runtime token counts,
all-rank times and max/mean ratios. Timing covers router, forward, backward and
optional SGD, excluding CPU evidence copies and metric reductions. The same seeds,
inputs, initial state and route sequence are used for both backends. Warmup covers the full route sequence without updating weights. The runner enables
HybridEP kernel-cache reuse, matching the upstream virtual-expert parity test. Gradient accumulation buffers are reset every call.

The numerical gate is relative L2 <= 0.01 and peak-normalized absolute error <=
0.05 for every output, input/probability/parameter gradient and updated parameter.
An all-zero reference must match exactly. Failed gates produce a nonzero exit;
the records remain available. These BF16 tolerance gates are not bit-exact claims.
Training-step comparisons include accumulated optimizer drift. A successful import,
build or CPU check is not a GPU qualification result.

Timings from one fixed-order run are descriptive only: HybridEP always runs first,
so this driver alone does not establish a speedup or a full-model throughput result.
NVLink topology must be recorded separately; CUDA peer access alone does not prove it.
