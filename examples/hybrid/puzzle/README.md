# Nemotron Labs 3 Puzzle

Train the full **Nemotron Labs 3 Puzzle 75B-A9B** architecture from scratch using
first-class `HybridModel` layer configs. No pretrained weights are downloaded or
converted. The architecture follows the published
[configuration at revision `7cd7fa01`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16/blob/7cd7fa01bab578cb4e8bde6ebba5e869afa1752d/config.json).

## Architecture

[`puzzle.py`](puzzle.py) defines the complete ordered list: 88 decoder layers
(40 Mamba, 40 MoE, 8 attention), hidden size 4096, 512 experts, latent MoE size
1024, Mamba state size 96, and all heterogeneous expert widths and routing top-k
values. The embedding and output matrices retain all 131072 vocabulary rows,
even though this launch uses a GPT-2 tokenizer.

An `MTPSplit` followed by attention and MoE (width 2688, top-k 22) defines one
prediction head. `mtp_num_layers` is left unset so the builder infers it.
`build_puzzle_layer_config_list` and `build_puzzle_model_config` are importable
without starting training. Each factory call constructs fresh configs;
`HybridModel` clones them for physical layers without modifying the originals.
No pattern string is constructed or parsed. Do not pass `--hybrid-layer-pattern`
or its deprecated alternatives: they are mutually exclusive with this config list.

## Requirements

- **Four nodes with four GB200 GPUs each**, allocated within one NVLink domain:
  TP1/PP1/CP1/EP16/ETP1. Use your site's placement options to keep all four nodes
  in the same GB200 NVL72 domain; requesting four nodes alone does not guarantee it.
- Slurm with Pyxis/Enroot for [`run.sbatch`](run.sbatch), and a prepared Megatron
  **dev** container with the repository's training, Mamba, Transformer Engine,
  and HybridEP dependencies. Use an image built from
  [`docker/Dockerfile.ci.dev`](../../../docker/Dockerfile.ci.dev) (`--target main`)
  for this checkout. The launcher uses `uv run --no-sync`; prepare dependencies
  before submitting, not once per training rank.
- A shared checkout, GPT-2-tokenized Megatron indexed dataset (`.bin`/`.idx`),
  matching `vocab.json` and `merges.txt`, and writable shared output storage.
  Mount these at the same absolute paths on every node and inside the container.

The defaults match the validated full-size EP16 functional configuration:
microbatch 1, global batch 32, sequence length 2048, BF16 compute/reduction,
full/uniform recomputation with one layer per checkpoint, and precision-aware
GPU Adam with FP32 moments and parameter-remainder storage. HybridEP, grouped
GEMM, and fused weighted squared ReLU are enabled. Shared-expert overlap is
disabled because latent MoE does not support it during training. No optimizer
workaround or TE upgrade is required for that tested configuration.

## Launch with Slurm

Run from the repository root on shared storage, replacing these paths and the
account/partition with your site's values:

```bash
cd /shared/Megatron-LM
export CONTAINER_IMAGE=/shared/images/megatron-dev.sqsh
export CONTAINER_MOUNTS=/shared:/shared
export DATA_PATH=/shared/data/my-gpt3_00_text_document
export VOCAB_FILE=/shared/data/vocab.json
export MERGE_FILE=/shared/data/merges.txt
export OUTPUT_DIR=/shared/runs/puzzle

sbatch --account=YOUR_ACCOUNT --partition=YOUR_GB200_PARTITION \
    examples/hybrid/puzzle/run.sbatch
```

The job launches one [`launch.sh`](launch.sh) process per node and four Python
workers per launcher. For a different scheduler/container integration, run
`bash examples/hybrid/puzzle/launch.sh` once on each of the four nodes inside the
prepared container, setting `MASTER_ADDR`, `MASTER_PORT`, `NNODES=4`,
`GPUS_PER_NODE=4`, and a distinct `NODE_RANK=0..3` as well as the paths above.

By default, training runs for 20 steps and saves every 10. This is a short
training example, not a reproduction of the model's original pretraining run.
LM/MTP losses appear in training logs; TensorBoard logs include gradient-zero
counts, timing, and memory. Output is written beneath `OUTPUT_DIR`:
`checkpoints/`, `data_cache/`, and `tensorboard/`. Slurm stdout/stderr are
`puzzle-JOBID.out` and `puzzle-JOBID.err` in the submission directory.

## Resume and override runtime settings

The launcher loads the latest checkpoint from the same directory it saves to.
Use a new `OUTPUT_DIR` to start from scratch. To continue a completed 20-step run
to step 40, keep the same paths and submit:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GB200_PARTITION \
    examples/hybrid/puzzle/run.sbatch --train-iters 40
```

Additional arguments are forwarded to Megatron after the launch defaults.
`CHECKPOINT_PATH`, `DATA_CACHE_PATH`, and `TENSORBOARD_PATH` may also be overridden
individually. Normal resume restores optimizer and RNG state; the Python config
list is reconstructed by `puzzle.py`, not serialized into checkpoint arguments.
Keep the architecture unchanged when resuming. Changes to topology, sequence
length, or batch size require their own memory/compatibility validation.
