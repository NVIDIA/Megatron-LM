# Megatron-FSDP v2 benchmark results

This document records the validation matrix used for the Megatron-FSDP v2
prototype. Results are tied to exact commits because the implementation was
still evolving while the matrix was collected.

## Full-shape Qwen3.5-VL 35B-A3B

The performance comparison uses the full 40-layer model on two nodes with four
GB200 GPUs per node. All backends use BF16, TP1/PP1/CP1/EP8/ETP1, sequence
length 4096, MBS8/GBS384, gradient-accumulation fusion, and forced-balanced
routing. Megatron-FSDP uses `optim_grads_params` sharding and HybridEP;
ND-parallel uses its stable NCCL all-to-all dispatcher. Each result contains 50
iterations, with the first 10 excluded from the steady-state aggregate.

| Backend | Exact commit | Median step | Median TFLOP/s/GPU | Samples/s | Peak device memory | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Megatron-FSDP v1 | `d1f1f68bb549` | **12,258.5 ms** | **359.6** | **31.22** | **174,822 MB** | [run](https://wandb.ai/adlr/jianbinc-qwen35-vl-35b-a3b-GB200-benchmark/runs/99b2a9a1a7c54d0e985d724d7825ac9d) |
| Megatron-FSDP v2 | `d1f1f68bb549` | 12,265.1 ms | 359.4 | 30.96 | 175,480 MB | [run](https://wandb.ai/adlr/jianbinc-qwen35-vl-35b-a3b-GB200-benchmark/runs/85f959aa31904f1082f31099aac611b6) |
| ND-parallel | `c2e9b8e00700` | 12,424.7 ms | 354.8 | 30.86 | 184,620 MB | [run](https://wandb.ai/adlr/jianbinc-qwen35-vl-35b-a3b-GB200-benchmark/runs/c7affd541ec84891b6a3c7d8cabb4d19) |

Megatron-FSDP v1 and v2 are at steady-state parity: v2 is 0.06% below v1
in median TFLOP/s/GPU and uses 658 MB more peak device memory. Against tuned
ND-parallel, v2 is 1.3% faster and uses 9,140 MB less peak device memory. This
compares each backend's supported tuned dispatcher; it is not a dispatcher-only
ablation.

### CORD-v2 convergence

The convergence matrix disables forced router balancing and consumes 192,000
samples over 500 optimizer steps with the same full model, parallelism, and
MBS8/GBS384 batch configuration. The ND run was checkpointed at iteration 375
and resumed for iterations 376-500.

| Backend | Exact commit | Iteration-500 train loss | Validation loss | Test loss | Skipped / NaN | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Megatron-FSDP v1 | `6791bfacb9ed` | 4.704548e-4 | 1.350591 | 1.626085 | 0 / 0 | [run](https://wandb.ai/adlr/jianbinc-qwen35-vl-35b-a3b-GB200-benchmark/runs/f742f1aeffd74835aa26f89936ee86d0) |
| Megatron-FSDP v2 | `6791bfacb9ed` | 1.380855e-3 | 1.375882 | 1.628900 | 0 / 0 | [run](https://wandb.ai/adlr/jianbinc-qwen35-vl-35b-a3b-GB200-benchmark/runs/4d7f22f3119045b1b94f085b59ca3e24) |
| ND-parallel | `6791bfacb9ed` | 1.677232e-3 | 1.414977 | 1.683958 | 0 / 0 | [part 1](https://wandb.ai/adlr/jianbinc-qwen35-vl-35b-a3b-GB200-benchmark/runs/f1438a9894e04ec6b58e718021ed7392) / [part 2](https://wandb.ai/adlr/jianbinc-qwen35-vl-35b-a3b-GB200-benchmark/runs/0fdb33ddd67d4ca5a505f949ac672d16) |

All three loss curves decrease from an initial loss near 12.8 without skipped
or NaN iterations. The v1/v2 test endpoints differ by 0.17%. The ND endpoint is
3.4% above v2, so this demonstrates stable real-data convergence rather than
deterministic numerical parity.

The v1 and v2 runs use the same seed, but their first-step losses already
differ (`12.81530` versus `12.76486`). Meta-device materialization follows
different implementation paths, the recipe permits nondeterministic TE
algorithms, and `eval_iters=1` makes each held-out endpoint a single-batch
estimate. Strict backend parity requires loading one common initialized
checkpoint, disabling nondeterministic kernels, and evaluating multiple
batches; these runs establish stable convergence only.

## Prototype examples

- [`fsdp_toy`](fsdp_toy/README.md#validation-results) records PyTorch FSDP2,
  Megatron-FSDP v2 eager, CUDA graph, and HSDP performance, plus deterministic
  BF16 HSDP endpoint parity.
- [`qwen3_30b_a3b_mxfp8`](qwen3_30b_a3b_mxfp8/README.md#validation-results)
  records the full-shape BF16/MXFP8 performance matrix and a 50-step
  SlimPajama convergence comparison.
- [`diffusers_qwenimage`](diffusers_qwenimage/README.md#benchmark-results) compares
  PyTorch FSDP1 with Megatron-FSDP v2 using the pretrained 60-block QwenImage
  transformer.

The matched QwenImage rerun at `31334f8807d6` measured the following results:

| Backend | Average step | Median step | Peak memory | Final / initial loss |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FSDP1 | 516.03 ms | **491.23 ms** | 75.39 GB | 0.642 |
| Megatron-FSDP v2 | 505.89 ms | 506.46 ms | **74.67 GB** | 0.634 |
| Megatron-FSDP v2 + CUDA graph | **385.89 ms** | **385.53 ms** | 86.72 GB | 0.641 |

All three real-data runs pass the 20-step convergence threshold. Eager v2 is
1.96% faster than FSDP1 by average step because it avoids FSDP1's single
824.48 ms tail, while FSDP1 remains 3.10% faster by median step. Eager v2 uses
0.72 GB less peak memory. CUDA graph improves the v2 median by 23.88% at the
cost of 12.05 GB additional peak memory.

The CUDA-graph run completes capture and all measured steps. The earlier
`TracePoolAllocator slot collision` is fixed by the captured-gradient lifetime
change merged before this rerun.

Nsight profiling of the gradient-DTensor wrapper-reuse change shows that its
original rank-local stall is removed. The maximum `MFSDP reduce_grad` duration
on the delayed rank falls from 353.08 ms to 1.81 ms, and maximum GPU
reduce-scatter start skew falls from 357.24 ms to 6.12 ms. Average start skew is
5.52 ms after the fix versus 4.23 ms for FSDP1. A smaller, gradually accumulating
host-side arrival skew remains, but it no longer creates the catastrophic
reduce-scatter cavitation seen before wrapper reuse.
