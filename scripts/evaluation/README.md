# Evaluation pipeline

This file explains how to launch the evaluation pipeline.
Quick usage:
```
bash scripts/evaluation/submit_evaluation.sh <ckp-path> --convert-to-hf --size <model-size> --wandb-entity <entity> --wandb-project <project> --wandb-id <runid> --iterations 1000,2000,3000 --tokens-per-iter <tokens per iteration>
```
Make sure you set your `WANDB_API_KEY` first.
This will create a `evaluate.sbatch` file and submit it to evaluate the model at `ckp-path` loading iterations 1000, 2000 and 3000 in the default `swissai_eval` benchmark.
See `scripts/evaluation/swissai_eval` and `scripts/evaluation/swissai_eval_short` for more information on the default tasks provided.
The `<size>` should be set to 1 or 8 (set it to 1 if training models <1B parameters).
Run `bash scripts/evaluation/submit_evaluation.sh --help` for more information on the variables and arguments available.

Important things to note:
- It is important to **not** set your `--wandb-id` to the runID of the checkpoint you are trying to evaluate: **this will corrupt the Step counter**, making it harder to continue using that run for further training.
  Instead, it is recommended to use a separate wandb project dedicated only to evaluation.
- W&B does not support multiple connexions to the same run, so parallel evaluation of any same checkpoint that goes to the same wandb run is not supported.
  This is enforced via the `#SBATCH --dependency=singleton` argument inside the submission file.

## Running from megatron checkpoints

To run the evaluations efficiently, it is heavily recommended to convert your checkpoints to huggingface before running the evaluations.
This is done automatically inside the submission script if `--convert-to-hf` is provided.
See `scripts/evaluation/example_submissions/eval_apertus1b.sh` for a more elaborate example of evaluating megatron checkpoints.

## Running huggingface checkpoints

You can specify the `<ckp-path>` to a huggingface name or path.
See `scripts/evaluation/example_submissions/eval_smollm.sh` for a more elaborate example of evaluating huggingface checkpoints.
