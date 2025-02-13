# Evaluation pipeline

This file explains how to launch the evaluation pipeline.
Quick usage:
```
WANDB_API_KEY=<key> TOKENIZER=<tokenizer> bash scripts/evaluation/submit_evaluation.sh <ckp-path> --size <model-size> --wandb-entity <entity> --wandb-project <project> --wandb-id <runid> --iteration <it> --tasks hellaswag
```
This will create a `evaluate.sbatch` file and submit it to evaluate the model at `ckp-path` loading iteration `<it>` in the hellaswag benchmark.
The `<size>` should be set to 1 or 8 (set it to 1 if training models <1B parameters).
Run `bash scripts/evaluation/submit_evaluation.sh --help` for more information on the variables and arguments available.

Important things to note:
- To get W&B sync working correctly, a few changes have been made to the lm-eval-harness repo.
  These changes are found in: https://github.com/AleHD/lm-evaluation-harness.
  Tested commit: `522929f5f6c76f2f31ad705ef9f685326838b709`.
- You will need to install some libraries to run eval harness.
  As a hotfix for the moment, you can install them to your user library inside a compute node:
  ```
  srun --pty --time=01:00:00 --account=a-a06 --environment=/capstor/store/cscs/swissai/a06/containers/NeMo/nemo-latest.toml -- bash
  git clone git@github.com:AleHD/lm-evaluation-harness.git
  pip install --user .[api]
  ```
- It is important to **not** set your `--wandb-id` to the runID of the checkpoint you are trying to evaluate: **this will corrupt the Step counter**, making it harder to continue using that run for further training.
- W&B does not support multiple connexions to the same run, so parallel evaluation of the same checkpoint in different iterations (under the assumption it will go the same run) is not supported.
  This is enforced via the `#SBATCH --dependency=singleton` argument inside the submission file.
- Currently, evaluating on multi-gpu is not supported, so running `--size 70` models **will hang**.
  This is because the inference engine was updated in commit [cb678cc](https://github.com/swiss-ai/Megatron-LM/commit/cb678cccf5c706cfc9f83a1ab6ae4d2cb9b1c5f3), but the openai completions endpoint that is used by eval-harness, was not updated.
  There is an easy hotfix currently applied for non-distributed models to run, but further work must be made to ensure multi-gpu.
