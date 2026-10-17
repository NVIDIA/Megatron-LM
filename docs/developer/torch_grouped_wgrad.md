# PyTorch grouped weight gradients for BF16 MoE experts

Set `moe_use_torch_grouped_wgrad=True` on `TransformerConfig`, or pass
`--moe-use-torch-grouped-wgrad` to the training parser. The default is false.
Use BF16 expert parameters/activations, `moe_grouped_gemm=True`,
`add_bias_linear=False`, and `gradient_accumulation_fusion=False`.

## Implementation and scope

The opt-in `TEGroupedLinear` path retains TE-owned parameters and checkpoint keys.
It calls TE grouped GEMM for forward and input gradients, and PyTorch grouped matmul
for the per-expert weight gradients. Existing packed activation storage is reused;
there is no input concatenation or dW copy-back. CPU token counts are copied as
cumulative int32 offsets through pinned host memory in forward, with a nonblocking
H2D transfer. Offset tensors and inputs belong to each autograd context, so multiple
outstanding microbatches do not share mutable metadata.

This first implementation targets eager SM90, contiguous BF16 operands, separate
per-expert weight parameters, and expert tensor parallelism of one. It resolves
`torch.nn.functional.grouped_mm` when available, otherwise `torch._grouped_mm`
(validated with PyTorch 2.9.1). Empty experts, all-empty input, and ordinary autograd
gradient accumulation are supported.

Other invocations use the original TE implementation: bias, FP8/FP4/quantization
recipes, FP16 autocast, TE debug iterations, fused main-grad accumulation, delayed
wgrad, CPU/activation offload, grouped weight storage, expert TP/GTP, FSDP, CUDA
graphs, and device-resident token counts. This change does not add communication,
optimizer fusion, higher-order differentiation, or a DeepGEMM backend.

The helper is an explicitly guarded BF16 autograd implementation inside Megatron;
it does not modify the installed TE package or globally replace TE functions.
Because it bypasses the generic quantized TE autograd wrapper for supported calls,
complete-module performance includes both dW dispatch and wrapper overhead changes.

## Validation

On two SM90 GPUs, each rank passed 36 focused tests. These cover native-TE output,
input-gradient and expert-gradient comparisons; activation recomputation; multiple
outstanding forwards; accumulation; empty experts; strided upstream gradients;
frozen parameters; unsupported-mode fallback; checkpoint keys; the CLI flag; and
byte-exact replay under side-stream contention.

```bash
torchrun --standalone --nproc-per-node=2 -m pytest --noconftest -o addopts="" -q \
  tests/unit_tests/transformer/moe/test_torch_grouped_wgrad.py \
  tests/unit_tests/determinism/kernels/test_torch_grouped_wgrad_replay.py \
  tests/unit_tests/determinism/kernels/test_te_wrappers.py::TestTEWrappers::test_te_grouped_linear_torch_wgrad_replays
```

`--noconftest` avoids unrelated dataset-download fixtures; the focused module suite
initializes real process groups using the repository test utilities.

## Single-module measurements

Measured on SM90 (132 SMs, about 80 GiB), PyTorch 2.9.1+cu129, TE 2.19.0,
with `CUDA_DEVICE_MAX_CONNECTIONS=1` and `NVTE_USE_CUTLASS_GROUPED_GEMM=1`.
Both variants use the same unmodified TE forward/dgrad GEMM API. The control uses
the default TE wrapper and its cuBLASLt grouped dW dispatch.

The checked-in fixture contains 80 expert-token distributions captured from rank 0
and the first microbatch of a Qwen3.5-35B-A3B text workload at TP1/PP2/CP4/EP4:
20 distributions per FC and global sequence length. FC1 has 64 expert matrices of
shape [1024, 2048], and FC2 has 64 matrices of shape [2048, 512]. Tensor values and
weights are newly generated BF16 random values; this is shape replay, not replay of
the original model activations. No checkpoint download is needed.

Each case alternates TE/Torch in ABBA or BAAB order, with five warmup calls per
backend and six measured samples per backend/metric, five calls per sample. Values
below are medians of the per-case medians across the 20 distributions. CUDA event
intervals include eager launch gaps; they are not sums of GPU kernel durations.

- **dW:** X/dY and offsets are already available; both paths allocate new BF16 output.
- **Module F+B:** one GroupedLinear forward, dgrad, dW, and common gradient clearing;
  includes the candidate forward metadata copy. No SwiGLU, routing, communication,
  activation recomputation, optimizer update, or distributed training schedule.

| Global sequence | FC | Metric | TE (ms) | Torch dW (ms) | Speedup |
|---:|---|---|---:|---:|---:|
| 8,192 | FC1 | dW | 0.638 | 0.210 | 3.041x |
| 8,192 | FC1 | Module F+B | 2.346 | 1.534 | 1.529x |
| 8,192 | FC2 | dW | 0.638 | 0.124 | 5.144x |
| 8,192 | FC2 | Module F+B | 2.336 | 1.519 | 1.538x |
| 32,768 | FC1 | dW | 0.666 | 0.502 | 1.327x |
| 32,768 | FC1 | Module F+B | 2.339 | 1.645 | 1.422x |
| 32,768 | FC2 | dW | 0.649 | 0.303 | 2.144x |
| 32,768 | FC2 | Module F+B | 2.348 | 1.547 | 1.518x |

All 80 measured cases passed the dispatch and numerical checks; the observed
maximum relative L2 difference for output, dgrad, and dW was zero on this random
input set. This does not establish bitwise equivalence on every input, dependency
version, or training trajectory. No full-model throughput or convergence claim is
made from these isolated measurements.

Reproduce from the repository root with the CUDA dependencies installed:

```bash
PYTHONPATH=. CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_USE_CUTLASS_GROUPED_GEMM=1 \
  torchrun --standalone --nproc-per-node=1 tools/benchmark_torch_grouped_wgrad.py \
  --shapes tests/unit_tests/transformer/moe/data/torch_grouped_wgrad_shapes.jsonl \
  --output /tmp/torch-grouped-wgrad-benchmark
```

The benchmark saves environment/source hashes, every timing sample, and numerical
checks. Aggregated results are also in `torch_grouped_wgrad_benchmark.csv`.
