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
