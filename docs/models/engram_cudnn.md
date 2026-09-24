# Experimental cuDNN Engram training gate

Set `engram_gate_backend="cudnn"` on `DeepSeekV41Config` to use the cuDNN
Frontend saved-state gate. The default is `"native"`. This option is an
experimental integration on top of [Megatron #7501](https://github.com/NVIDIA/Megatron-LM/pull/7501).
It replaces the floating gate after embedding lookup and WKV projection;
embedding distribution, WKV, q/k parameter names and checkpoint layout stay
with the existing Engram module.

```python
config = DeepSeekV41Config.from_hf(
    hf_config,
    params_dtype=torch.bfloat16,
    bf16=True,
    engram_gate_backend="cudnn",
)
```

The adapter packs sequence-major inputs, forms the FP32 product of q/k
normalization weights, and invokes FE's saved-state forward/backward APIs.
Torch autograd applies the product rule to the original parameters. Every
forward retains its own saved state and every backward allocates its own
workspace and outputs on the current stream. At most eight shape/device
plans are cached; pending autograd calls retain evicted plans until complete.
The FE internal backend is `"frost"`; the Megatron option is `"cudnn"`.

## Supported contract

- SM100, BF16 activations, hidden size 5120 and four residual streams.
- Sequence-major hidden `[sequence,batch,20480]`, packed KV
  `[sequence,batch,25600]`, optional bool mask `[batch,sequence]`.
- Q/k parameters `[4,5120]`, BF16 or FP32, on the same CUDA device.
- Between 64 and 8192 local tokens, divisible by 64. Unsupported input
  contracts fail explicitly; selecting this backend never silently falls back.
- First-order autograd, including pending microbatches and retained-graph
  repeated backward. A masked token returns its input and identity input gradient.

FP8/FP4, CUDA Graph capture and higher-order differentiation are unsupported.
Module tests cover world-size-one MCore DDP and FP32 `main_grad` accumulation.
Multi-rank DP/EP, FSDP, full-model training and a full-size embedding table have
not been validated. Selecting a supported shape does not predict a speedup.

## Dependencies and reproduction

The repository's default FE dependency pin does **not** provide this API.
Use an isolated environment and the validated source commit from
[FE #1204](https://github.com/NVIDIA/cudnn-frontend/pull/1204):
`b4a36817db5750efe5a833f4757a38038d27bf86`.
That PR is still pending; this draft is not a released-package integration.
The underlying saved-state API was introduced by
[FE #1113](https://github.com/NVIDIA/cudnn-frontend/pull/1113).

The measured environment was NGC PyTorch 26.08, cuDNN backend 9.28,
CuTe DSL 4.7.0 and Triton >=3.7.0. Build the FE native binding against the
chosen backend, install that checkout into the same environment, and verify
the actual import before running tests:

```bash
git clone https://github.com/NVIDIA/cudnn-frontend.git
git -C cudnn-frontend checkout b4a36817db5750efe5a833f4757a38038d27bf86
pip install 'nvidia-cutlass-dsl==4.7.0' 'triton>=3.7.0'
pip install -e ./cudnn-frontend
python -c 'import cudnn; print(cudnn.__file__); print(cudnn.backend_version())'
python -c 'import cudnn; print(cudnn.EngramGateSavedForward, cudnn.EngramGateSavedBackward)'

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
torchrun --standalone --nproc-per-node=1 -m pytest -q \
  tests/unit_tests/fusions/test_cudnn_engram.py \
  tests/unit_tests/determinism/kernels/test_cudnn_engram.py
torchrun --standalone --nproc-per-node=1 \
  tests/performance_tests/cudnn_engram.py \
  --tokens 4096 --microbatches 32 --pairs 12 --profile \
  --output engram-cudnn-4096.json
```

The unit tests skip optional dispatch coverage when the pinned dependency
does not expose the saved-state gate; a skipped CI case is not validation of
this provider. The explicit source-pinned tests above provide that coverage.

The performance harness compiles both module exteriors by default and checks
output, input gradient and every parameter gradient against eager native and
against each other before timing. It alternates AB/BA order, includes module
forward/backward and FP32 main-gradient accumulation, and records route counters.
`--baseline native` is an eager diagnostic for both exteriors, not the baseline
for the performance results below. Use a new output file for each process.

## B200 module measurements

The unchanged gate implementation in this draft was measured with the FE
commit above. The later dependency diagnostic only changes the missing-API
error path; it does not change valid forward/backward execution. These are
historical independently measured B200 results, not fresh post-publication CI.

| Tokens / microbatches | Compiled native, ms | cuDNN gate, ms | Latency reduction |
| --- | ---: | ---: | ---: |
| 2048 / 32 | 73.687 | 70.218 | 4.71% |
| 4096 / 32 | 130.937 | 120.896 | 7.67% |
| 4096 / 32, independent process | 131.041 | 120.108 | 8.34% |
| 8192 / 16 | 122.886 | 112.366 | 8.56% |

Each process passed 24 three-way numerical checks and won 12/12 paired samples.
This is the actual Engram module plus MCore DDP: BF16, H5120, four streams,
WKV input width 6144, synthetic inputs, world size one, and an embedding table
reduced to 1024 rows. It excludes hashing, other model layers and optimizer
steps. Full-model, full-table, multi-rank and convergence results are not claimed.
The separate experimental WKV/wgrad fusion is not included in this change or
these numbers.

The Engram module, hashing and distributed embedding belong to the upstream
#7501 stack and retain its source attribution. This contribution authors the
cuDNN adapter, integration checks and benchmark; it reuses the existing
NVIDIA cuDNN Frontend kernels rather than introducing a new GEMM implementation.
