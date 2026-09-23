# Experimental Frost BF16 experts for DSv4.1

This prototype selects cuDNN Frontend grouped GEMMs inside the existing
`TEGroupedMLP` module with `moe_bf16_expert_backend="frost"`. The default remains
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
The N-major dependency is not yet published. CuTe DSL 4.7 or newer is required.

The module test compares the real TE module and DDP against Frost for outputs,
input/probability gradients, and accumulated FP32 parameter gradients. It also
covers changing routing, empty experts, multiple outstanding microbatches, and
host/device counts. Separate tests cover checkpoint layout and deterministic
staging replay. The integration still needs current-target B200 performance
validation against both native TE legacy and GroupedTensor routes; historical
prototype speedups do not establish a speedup for this implementation.

```bash
torchrun --nproc-per-node=1 -m pytest \
  tests/unit_tests/fusions/test_frost_bf16_experts.py \
  tests/unit_tests/fusions/test_frost_bf16_checkpoint.py \
  tests/unit_tests/fusions/test_frost_bf16_module.py \
  tests/unit_tests/determinism/kernels/test_frost_bf16_experts.py
```
