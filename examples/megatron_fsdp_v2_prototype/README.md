# Megatron-FSDP v2 Prototype Examples

This directory contains experimental examples for Megatron-FSDP v2. Each
example is self-contained in its own directory with setup and launch
instructions.

These examples exercise prototype APIs and may change as Megatron-FSDP v2
evolves. For the established Megatron-FSDP training and checkpoint-conversion
examples, see [`examples/megatron_fsdp`](../megatron_fsdp/README.md).

## Examples

| Example | Description |
| --- | --- |
| [`fsdp_toy`](fsdp_toy/README.md) | Standalone comparison of PyTorch FSDP2 and Megatron-FSDP v2, including CUDA graphs, HSDP, activation checkpointing, distributed checkpoints, and convergence checks. |
| [`qwen3_30b_a3b_mxfp8`](qwen3_30b_a3b_mxfp8/README.md) | Two-node Qwen3-30B-A3B training recipe using Megatron-FSDP v2, MXFP8, and Weights & Biases. |
| [`diffusers_qwenimage`](diffusers_qwenimage/README.md) | Diffusers QwenImage benchmark comparing PyTorch FSDP1 with Megatron-FSDP backends. |

Unless an example README says otherwise, run commands from the root of the
Megatron-LM repository.
