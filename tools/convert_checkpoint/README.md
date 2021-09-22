# Introduction

This folder is a collection of scripts for converting checkpoints of one training framework (e.g., DeepSpeed) into that of a different framework (e.g., Megatron-LM, HF Transformers).

The folder also contains scripts for inspecting checkpoint files and folders, which could be useful when developing checkpoint conversion logic. At the time of creation, this folder contains scripts to convert DeepSpeed checkpoints to Megatron-LM and HF Transformers checkpoints (this motivated this effort as part of the BigScience project).

Here are the list and details of checkpoint conversions provided by the available scripts:

1. [Megatron-DeepSpeed to Megatron-LM](#Megatron-DeepSpeed-to-Megatron)
1. [Megatron-DeepSpeed to HF Transformers](#Megatron-DeepSpeed-to-HF-Transformers)


## Megatron-DeepSpeed to Megatron

The (current implementation of the) converter extracts args and model parameters from a DeepSpeed checkpoint (i.e., excludes other training states such as optimizer, scheduler, etc) and convert into a Megatron-LM checkpoint similarly containing only model parameters. The converter also provides a best-effort attempt to reshape the tensor-parallelism and pipeline parallelism degrees for the checkpoint. The resulting Megatron-LM checkpoint could be loaded into Megatron-LM framework for finetuning or inference. Tensor parallelism (TP) and pipeline parallelism (PP) are supported in the sense that the generated Megatron-LM checkpoint (folders and files) will be of the same TP and PP of the training that created the input DeepSpeed checkpoint. The entry point of the converter is `deepspeed_to_megatron.py`, which as the following usage:
```bash
python tools/convert_checkpoint/deepspeed_to_megatron.py -h
Convert DeepSpeed Checkpoint to Megatron Checkpoint
usage: deepspeed_to_megatron.py [-h] [--input_folder INPUT_FOLDER]
                                [--output_folder OUTPUT_FOLDER]
                                [--target_tp TARGET_TP]
                                [--target_pp TARGET_PP] [--for_release]

optional arguments:
  -h, --help            show this help message and exit
  --input_folder INPUT_FOLDER
                        Input DeepSpeed Checkpoint folder
  --output_folder OUTPUT_FOLDER
                        Output Megatron checkpoint folder
  --target_tp TARGET_TP
                        Target TP degree
  --target_pp TARGET_PP
                        Target PP degree
  --for_release         Convert for release purpose, reset some (progress)
                        counters.
```

The following scripts which proved useful for debugging are also included:
1. `inspect_deepspeed_checkpoint.py`: view the contents of a DeepSpeed checkpoint folder.
2. `inspect_checkpoint.py`: view the contents of a PyTorch checkpoint file.

## Megatron-DeepSpeed to HF Transformers

In order to convert from Megatron-DeepSpeed to HF Transformers, you can do this directly using:

```bash
python tools/convert_checkpoint/deepspeed_to_transformers.py  \
--input_folder /path/to/Megatron-Deepspeed/checkpoint/global_step97500 \
--output_folder /path/to/transformers/checkpoint
```
since `transformers` currently only works with PP=1/TP=1 we use the defaults `--target_tp 1 --target_pp 1`.

The script taps into `transformers` and as of this writing requires `transformers@master` (or `transformers==4.11` if you read this later and a new version is released).

Note that you may run into problems with not having `megatron.enums` defined since `Megatron-Deepspeed` in the `bigscience-workshop` tree diverged from the `microsoft` tree. In such cases you can fix this on the fly by ensuring the former appears first in the `sys.path`. For example:


```bash
PYTHONPATH=/hf/Megatron-DeepSpeed-bigscience:/hf/Megatron-DeepSpeed-microsoft \
python tools/convert_checkpoint/deepspeed_to_transformers.py  \
--input_folder /path/to/Megatron-Deepspeed/checkpoint/global_step97500 \
--output_folder /path/to/transformers/checkpoint
```

Alternatively, you can convert first from Megatron-DeepSpeed to Megatron and then to HF Transformers:

```bash
# 1. Megatron-DeepSpeed to Megatron
cd /hf/Megatron-DeepSpeed-bigscience
python tools/convert_checkpoint/deepspeed_to_megatron.py --target_tp 1 --target_pp 1 \
--input_folder /path/to/Megatron-Deepspeed/checkpoint/global_step97500 \
--output_folder /path/to/Megatron/checkpoint

# 2. Megatron to HF Transformers
cd /hf/transformers
python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py \
/path/to/Megatron/checkpoint/iter_0097500/mp_rank_00/model_optim_rng.pt
```
