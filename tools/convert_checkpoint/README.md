# Introduction
This folder is a collection of scripts for converting checkpoints of one training framework (e.g., DeepSpeed) into that of a different framework (e.g., Megatron-LM). inspecting checkpoints. The folder also contains scripts for inspecting checkpoint files and folders, which could be useful when developing checkpoint conversion logic. At the time of creation, this folder contains scripts to convert DeepSpeed checkpoints to Megatron-LM checkpoints (this motivated this effort as part of the BigScience project). 

Here are the list and details of checkpoint conversions provided by the available scripts.
1. [DeepSpeed to Megatron-LM](#DeepSpeed-to-Megatron)


## DeepSpeed to Megatron
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