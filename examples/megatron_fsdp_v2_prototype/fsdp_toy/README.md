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

## Validation results

The example was exercised on four GB200 GPUs. Each performance case used 100
measured steps after warmup.

| Backend | Average step | Samples/s | Peak allocated |
| --- | ---: | ---: | ---: |
| PyTorch FSDP2, full shard | 13.05 ms | 2,453 | 0.96 GB |
| Megatron-FSDP v2, full shard | 11.57 ms | 2,766 | 1.01 GB |
| Megatron-FSDP v2, CUDA graph + trace pool | **9.22 ms** | **3,471** | 1.32 GB |
| Megatron-FSDP v2, HSDP | 12.12 ms | 2,640 | 1.16 GB |

The deterministic BF16 HSDP correctness check was repeated at exact commit
`5d0cef0c18a9`, which is included in `mfsdp_refactor`. Full sharding ended at
loss `6.1461e-5` and HSDP at `6.1469e-5`; the absolute difference was `8e-9`
against a `1e-7` limit. Both runs started at `1.3577e-3` and reached a
final/initial ratio of `0.045`.
