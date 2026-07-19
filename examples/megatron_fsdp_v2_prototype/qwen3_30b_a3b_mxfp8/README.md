# Qwen3-30B-A3B MXFP8 Training

This example is a two-node SLURM recipe for training Qwen3-30B-A3B with
Megatron-FSDP v2, MXFP8 parameter gathering, expert parallelism, and Weights &
Biases logging.

The checked-in values target two nodes with four GB200 GPUs per node. Treat the
script as a starting point and adapt its SLURM, container, mount, data, and
tokenizer settings to your cluster.

## Configuration summary

| Setting | Value |
| --- | --- |
| Nodes / GPUs | 2 nodes, 4 GPUs per node |
| Parallelism | TP1, PP1, CP1, EP4, ETP1 |
| FSDP | Megatron-FSDP v2, `optim_grads_params` |
| Precision | BF16 training, MXFP8, FP8 parameter gather |
| Sequence length | 4096 |
| Batch size | MBS4, GBS128 |
| Dispatcher | all-to-all |
| Container | `nvcr.io/nvidia/nemo:26.04` |

## Configure

At minimum, update the `#SBATCH` account and partition, the container mounts,
and these environment variables:

```bash
export MEGATRON_PATH=/path/to/Megatron-LM
export OUTPUT_PATH=/path/to/output
export DATA_PATH=/path/to/data/c4/en/c4-train.en_6_text_document
export TOKENIZER_MODEL=/path/to/data/c4/en/tokenizer

# Optional W&B settings
export WANDB_API_KEY=...
export WANDB_ENTITY=...
```

## Launch

```bash
SCRIPT=examples/megatron_fsdp_v2_prototype/qwen3_30b_a3b_mxfp8/\
qwen3-30b-a3b.gbs128_mbs4_seq4096_n2_mfsdp2_mxfp8_wandb.sh
sbatch "$SCRIPT"
```

The script writes checkpoints, TensorBoard events, and W&B files below
`OUTPUT_PATH`.

## Validation results

The checked-in full 30.53B-parameter shape was tested on two nodes with four
GB200 GPUs per node at commit `6791bfacb9ed`. The matrix used TP1/PP1/CP1/EP4,
MBS4/GBS128, sequence length 4096, NCCL all-to-all dispatch, and selective
`moe_act` recomputation.

| Backend | Median TFLOP/s/GPU | Samples/s | Peak device memory | W&B |
| --- | ---: | ---: | ---: | --- |
| Megatron-FSDP v1, BF16 | 300.85 | 25.36 | 174.69 GB | [run](https://wandb.ai/adlr/jianbinc-qwen3-30b-GB200-benchmark/runs/5911d158ab53464d87968f208106793a) |
| Megatron-FSDP v2, BF16 | **323.45** | **26.97** | 183.09 GB | [run](https://wandb.ai/adlr/jianbinc-qwen3-30b-GB200-benchmark/runs/acd8c6a872434775833a285c88ae1f30) |
| Megatron-FSDP v2, MXFP8 parameter gather | 219.70 | 18.49 | **170.85 GB** | [run](https://wandb.ai/adlr/jianbinc-qwen3-30b-GB200-benchmark/runs/9811f6c0d56640e2a0308b623b6352b9) |

V2 BF16 is 7.5% faster than v1 BF16 in this case. MXFP8 parameter gathering
saves 12.25 GB versus v2 BF16, but is 32.1% slower; it is therefore a memory
and capability result for this configuration, not a throughput win.

The BF16 backends also completed a 50-step SlimPajama real-data convergence
check with forced router balancing disabled.

| Backend | Initial train loss | Final train loss | Final validation loss | Final / initial | Skipped / NaN | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Megatron-FSDP v1 | 12.34454 | 7.690141 | 7.711185 | 0.6230 | 0 / 0 | [run](https://wandb.ai/adlr/jianbinc-qwen3-30b-GB200-benchmark/runs/97c52d949268483fbd930ad34c80f770) |
| Megatron-FSDP v2 | 12.39409 | 7.632825 | 7.651939 | 0.6159 | 0 / 0 | [run](https://wandb.ai/adlr/jianbinc-qwen3-30b-GB200-benchmark/runs/fe1917bfbfab49928324637d2c563576) |

The final train and validation endpoints differ by 0.75% and 0.77%,
respectively, and both runs complete without skipped or NaN iterations.
