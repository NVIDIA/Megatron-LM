# Miles Megatron Lite Example

This example runs radixark/miles with Megatron Lite as the training actor by
applying the miles runtime patch from
`experimental/lite/examples/miles/miles_mlite/backend_patch.py`.

The patch intentionally keeps `--train-backend megatron`: miles imports
`MegatronTrainRayActor` from its Megatron backend module inside the actor
allocation path, so replacing that source symbol before allocation routes the
existing Megatron slot to `MLiteTrainRayActor` without changing miles code or CLI
choices.

## Layout

- `miles_mlite/launch.py`: registers MLite CLI flags, optionally imports the
  patch, parses miles arguments, and calls the normal miles train loop.
- `scripts/run_qwen3moe_sft.sh`: short SFT launcher for Qwen3-30B-A3B style
  MoE checkpoints.
- `scripts/run_qwen3moe_grpo.sh`: GRPO launcher using miles rollout and MLite
  policy-loss training.
- `REQUIRED_MILES.txt`: source commit pinned for the miles API shape.

## SFT

Run inside a Slurm GPU allocation/container with miles, Megatron-LM, and
Megatron Lite importable:

```bash
export MILES_ROOT=/path/to/miles
export MEGATRON_ROOT=/path/to/full/Megatron-LM  # if Megatron-Core is not installed
export MODEL_PATH=/path/to/Qwen3-30B-A3B
export TRAIN_DATA=/path/to/messages.parquet
export CONTAINER_IMAGE=/path/to/miles.sqsh
bash experimental/lite/examples/miles/scripts/run_qwen3moe_sft.sh
```

Useful knobs:

- `TP_SIZE`, `PP_SIZE`, `CP_SIZE`, `EP_SIZE`, `ETP_SIZE`
- `GLOBAL_BATCH_SIZE`, `ROLLOUT_BATCH_SIZE`, `MAX_TOKENS_PER_GPU`
- `OPTIMIZER_OFFLOAD=1`, `PARAM_OFFLOAD=1`
- `MLITE_MODEL_NAME=qwen3_moe`, `MLITE_OPTIMIZER_BACKEND=dist_opt`
- `TRAIN_BACKEND=mlite` uses the patch; `TRAIN_BACKEND=megatron` leaves the
  native miles Megatron actor untouched for A/B runs.
- `DRY_RUN=1` prints the resolved Ray job command without launching

## GRPO

```bash
export MILES_ROOT=/path/to/miles
export MEGATRON_ROOT=/path/to/full/Megatron-LM  # if Megatron-Core is not installed
export MODEL_PATH=/path/to/Qwen3-30B-A3B
export PROMPT_DATA=/path/to/prompts.jsonl
export CONTAINER_IMAGE=/path/to/miles.sqsh
bash experimental/lite/examples/miles/scripts/run_qwen3moe_grpo.sh
```

The GRPO launcher selects `--loss-type policy_loss`,
`--advantage-estimator grpo`, `--use-rollout-logprobs`, and
`--megatron-to-hf-mode raw`. MLite exports HF-format weights directly, so the
rollout resync path bypasses mbridge conversion and sends raw HF tensors through
miles update-weight APIs.

GPU validation should be run through Slurm. A successful smoke must use
`DRY_RUN=0`, finish with `sacct` exit code `0:0`, and show real training loss
progression in the Slurm log.
