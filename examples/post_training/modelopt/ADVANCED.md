<div align="center">

# Advanced Usage

[Advanced Configuration](#advanced-configuration) |
[Slurm Examples](#slurm-examples) |
[Checkpoint Resume](#checkpoint-resume) |

</div>

## Advanced Configuration

### Understanding Configuration Variables

For simplicity, we use `shell` scripts and variables as arguments. Each script has at least 1 positional
argument `[model_conf]`. Some scripts may require more such as `[qformat]` is needed for
quantization.

```sh
\
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    bash quantize.sh [model_conf] [qformat]
```

> **❗ IMPORTANT:** `model_conf` is used to get the corresponding Megatron-LM `${MODEL_ARGS}`. For example,
> `meta-llama/Llama-3.1-8B-Instruct` or `deepseek-ai/DeepSeek-R1` are both supported.
>
> Provide the pretrained checkpoint through variable `${HF_MODEL_CKPT}` in commandline or
> in a configuration shell script. More variables (e.g. `${TP}`, `${EP}`, ...) can be provided through
> commandline but we recommend passing all variables in a separate `shell` script.

### Using Configuration Scripts

When `${HF_MODEL_CKPT}` is not set through the commandline, `./env_setup_template.sh` can be used
to pass all variables instead. If you have your own script, use `${SANDBOX_ENV_SETUP}`.

```sh
\
    SANDBOX_ENV_SETUP=<path_to_your_script> \
    bash quantize.sh [model_conf] [qformat]
```

**For Slurm execution**, you **MUST USE** `${SANDBOX_ENV_SETUP}` (default: `./env_setup_template.sh`).
Other variables are not passed through `sbatch` and `srun` automatically.

### Common Configuration Variables

- `HF_MODEL_CKPT`: Path to pretrained model checkpoint
- `TP`: Tensor parallelism degree
- `PP`: Pipeline parallelism degree
- `EP`: Expert parallelism degree (for MoE models)
- `ETP`: Expert tensor parallelism degree (for MoE models)
- `MLM_MODEL_SAVE`: Path to save Megatron-LM checkpoint
- `MLM_MODEL_LOAD`: Path to load Megatron-LM checkpoint
- `MLM_EXTRA_ARGS`: Additional Megatron-LM arguments (e.g., for uneven PP)

## Slurm Examples

For models that require multi-node, our scripts in Megatron-LM examples also support `slurm` with a sbatch wrapper.
Start with the example `slurm/sbatch.sh` with some minor modification or use your existing `sbatch`
script.

Different from local environment, we only allow passing variables through a shell script (default: `env_setup_template.sh`).
Commandline variable passthrough is not supported.

<br>

### ⭐ BF16 Kimi-K2-Instruct EAGLE3 Training

 `conf/moonshotai/kimi_k2_instruct.sh` is a config that has been tested
with 8 nodes of DGX H100 (TP=8, ETP=1, EP=64, overall 64 H100 GPUs in total). Update `HF_MODEL_CKPT` to the exact
checkpoint path in the container to start:

```sh
export USER_FSW=<path_to_scratch_space>
export CONTAINER_IMAGE=<path_to_container_image>
export SANDBOX_ENV_SETUP=./conf/moonshotai/kimi_k2_instruct.sh
sbatch --nodes=8 slurm/sbatch.sh "eagle3.sh moonshotai/Kimi-K2-Instruct"
```

To export the trained EAGLE3 model, switch to `kimi_k2_instruct_export.sh`.
**We only support pipeline-parallel (PP) export.** In this case, 2 nodes are used (PP=16).

```sh
export USER_FSW=<path_to_scratch_space>
export CONTAINER_IMAGE=<path_to_container_image>
export SANDBOX_ENV_SETUP=./conf/moonshotai/kimi_k2_instruct_export.sh
sbatch --nodes=2 slurm/sbatch.sh "export.sh moonshotai/Kimi-K2-Instruct"
```

## Checkpoint Resume

WIP
