# Megatron-FSDP v2 Toy Example

`fsdp_toy.py` is a standalone distributed training example for comparing
PyTorch FSDP2 with Megatron-FSDP v2. It uses a small MLP-style model and does
not depend on the Megatron training loop.

The example covers:

- per-layer and root-module sharding with `fully_shard()`;
- CUDA graph capture and trace-pool allocation;
- activation checkpointing;
- HSDP with outer optimizer-state sharding;
- distributed checkpoint save and resume;
- CUDA memory snapshots; and
- deterministic teacher-student convergence verification.

## Run

Run these commands from the Megatron-LM repository root.

PyTorch FSDP2 baseline:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py \
  --model-dim 512 --n-layers 2 --batch-size 4
```

Megatron-FSDP v2:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py \
  --model-dim 512 --n-layers 2 --batch-size 4 \
  --use-megatron-fsdp
```

Enable CUDA graphs, trace-pool allocation, and activation checkpointing:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py \
  --model-dim 512 --n-layers 2 --batch-size 4 \
  --use-megatron-fsdp --cuda-graph --use-trace-pool \
  --activation-checkpoint
```

Enable the deterministic convergence check and distributed checkpoints:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py \
  --model-dim 512 --n-layers 2 --batch-size 4 \
  --use-megatron-fsdp --use-real-data \
  --ckpt-dir /tmp/mfsdp-v2-toy-checkpoints
```

## Selected options

| Option | Default | Description |
| --- | --- | --- |
| `--model-dim` | `1024` | Model hidden dimension. |
| `--n-layers` | `3` | Number of toy transformer-style blocks. |
| `--use-megatron-fsdp` | off | Use Megatron-FSDP v2 instead of PyTorch FSDP2. |
| `--cuda-graph` | off | Capture Megatron-FSDP layer execution in CUDA graphs. |
| `--use-trace-pool` | off | Use the trace-pool allocator for stable buffer addresses. |
| `--activation-checkpoint` | off | Recompute block activations during backward. |
| `--enable-hsdp` | off | Use a `2 x N` HSDP mesh; requires an even world size and Megatron-FSDP. |
| `--release-memory-pool` | off | Release Megatron-FSDP allocator slots after backward. |
| `--ckpt-dir` | unset | Save and resume distributed checkpoints in this directory. |
| `--use-real-data` | off | Use deterministic teacher-student data and assert convergence. |
| `--record-memory-history DIR` | unset | Write one CUDA memory snapshot per rank. |

Use `--help` for the complete option list.
