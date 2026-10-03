# Experimental cuDNN BF16 experts for DSv4.1

This prototype selects cuDNN Frontend grouped GEMMs inside the existing
`TEGroupedMLP` module with `moe_bf16_expert_backend="cudnn"`. The default remains
`transformer_engine`. This is a local-expert integration, not a complete DSv4.1
model recipe or a claim of full-model training acceleration.

The supported contract is SM100, BF16 discrete expert parameters, FP32
`main_grad` accumulation with Megatron DDP, expert tensor parallel size 1,
interleave32 clamped SwiGLU with zero linear offset, and selective `moe_act`
recomputation. Set `gradient_accumulation_fusion=True`,
`moe_mlp_glu_interleave_size=32`, `activation_func_clamp_value=10.0`,
`glu_linear_offset=0.0`, and `recompute_modules=["moe_act"]`.

Forward uses grouped FC1, FP32 weighted activation between BF16 boundaries, and
grouped FC2. Backward combines FC2 input-gradient GEMM, clamped activation
backward, probability gradient, and activation recomputation, then computes
both weight gradients and the input gradient. Each outstanding microbatch owns
its activation storage. Weight parameters remain in the Megatron module and
checkpoint factories convert gate/up blocks to the canonical saved layout.

Quantized training, deterministic mode (weight gradients use FP32 atomics),
CUDA graph capture, expert activation offload, delayed/overlapped weight
gradients, NCCL-EP zero-copy buffers, retained-graph backward replay, frozen
expert parameters in a differentiable call, and FSDP are outside the validated
contract. Execution uses one CUDA stream per expert module.

## Dependencies and validation status

This work is not ready to enable in a released recipe. It requires a cuDNN
Frontend development build combining caller-owned grouped-GEMM workspaces,
DSv4.1 dGLU activation recomputation, and native N-major BF16 input gradients.
The caller-workspace and dGLU changes are tracked in
[NVIDIA/cudnn-frontend#1181](https://github.com/NVIDIA/cudnn-frontend/pull/1181) and
[NVIDIA/cudnn-frontend#1187](https://github.com/NVIDIA/cudnn-frontend/pull/1187).
Native N-major support is tracked in
[NVIDIA/cudnn-frontend#1199](https://github.com/NVIDIA/cudnn-frontend/pull/1199).
CuTe DSL 4.7 or newer is required. A frozen combination of these dependencies
used for the B200 validation is available at
[`49aa267dc4fe918fc5e98e04c8955bda2be1635b`](https://github.com/YangXu1990uiuc/cudnn-frontend/tree/49aa267dc4fe918fc5e98e04c8955bda2be1635b).
This is a validation snapshot, not a released cuDNN Frontend version. Build its
Python package using the normal Frontend installation instructions, and verify
`cudnn.__file__` resolves to that build before testing this backend.

The module test compares the real TE module and DDP against cuDNN for outputs,
input/probability gradients, and accumulated FP32 parameter gradients. It also
covers changing routing, empty experts, multiple outstanding microbatches, and
host/device counts. Separate tests cover checkpoint layout and deterministic
staging replay. The selected suite passed 300 checks on B200 with no skips:
56 configuration/checkpoint checks, four actual module cases, four determinism
cases and 236 registry checks. The 300 checks are not 300 numerical workloads.

```bash
torchrun --nproc-per-node=1 -m pytest \
  tests/unit_tests/fusions/test_cudnn_bf16_experts.py \
  tests/unit_tests/fusions/test_cudnn_bf16_checkpoint.py \
  tests/unit_tests/fusions/test_cudnn_bf16_module.py \
  tests/unit_tests/determinism/kernels/test_cudnn_bf16_experts.py
```

## Reproduce module performance

From this repository root, using the required Frontend build and a B200:

```bash
torchrun --nnodes=1 --nproc-per-node=1 --master-addr=127.0.0.1 --master-port=29589 \
  tests/performance_tests/cudnn_bf16_experts.py \
  --baseline auto --microbatches 32 --pairs 12 --profile \
  --counts-file tests/performance_tests/fixtures/dsv41_expert_counts.json \
  --output cudnn-source-counts.json
```

Repeat in a fresh process with a different output name. Run with `--baseline
legacy` to check the other native TE route, and omit `--counts-file` for the
balanced count bank. The harness selects the native TE route explicitly, checks
outputs, input/probability gradients and all 96 parameter gradients before
timing, alternates AB/BA order and records route counters. `--profile` records
actual kernel names separately from the timed samples.

The E48/H5120/I2304 geometry adapts the pinned
[Megatron-Bridge BF16 recipe](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/ece66187628007544c33fcff280089b61d5c758a/src/megatron/bridge/recipes/deepseek/gb200/deepseek_v4.py)
to [DSv4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277).
The fixture contains counts from an untrained source router, not a checkpoint
trace. Inputs and probabilities are synthetic. Both count modes alternate a
second bank derived by emptying one expert. Each timed sample contains 32
microbatches with real DDP FP32 gradient accumulation.

A B200 (148 SMs), PyTorch 2.14 development build, TE 2.18, cuDNN 9.28 and CuTe
DSL 4.7 measured the following medians over 12 paired samples per process:

| Count bank | Native TE GroupedTensor (ms) | cuDNN (ms) | Latency reduction |
| --- | ---: | ---: | ---: |
| Source-derived, 24,587 rows | 254.243 | 233.848 | 8.02% |
| Independent process repeat | 255.732 | 236.319 | 7.59% |
| Balanced, 24,576 rows | 241.846 | 210.911 | 12.79% |

cuDNN won 12/12 pairs in each run. The separately measured TE legacy route was
slower (260.380 ms versus cuDNN 236.235 ms for source-derived counts). Profiling
confirmed the GroupedTensor baseline's native `ptrGroup` GEMMs and the expected
cuDNN GEMM/dGLU/wgrad kernels. This is a world-size-1, post-dispatch local-expert
module measurement. Router, EP communication, shared experts, optimizer and
other layers are excluded. It does not establish full-model speedup, distributed
training correctness or convergence, and is not a performance guarantee for
other SM100 devices or configurations.
