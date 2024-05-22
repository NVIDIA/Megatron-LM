# RETRO MODEL

## Table of contents
- [1. Training Setup](#1-training-setup)
- [2. Data Preprocessing](#2-data-preprocessing)
- [3. Configurations](#3-configurations)

## 1. Training setup
<a id="markdown-training-setup" name="training-setup"></a>

To run the model using a docker container run it as follows
```
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:23.09-py3
CHECKPOINT_PATH="" #<Specify path>
TENSORBOARD_LOGS_PATH=""#<Specify path>

docker run \
  --gpus=all \
  --ipc=host \
  --workdir /workspace/megatron-lm \
  -v /path/to/data:/path/to/data \
  -v /path/to/megatron-lm:/workspace/megatron-lm \
  megatron-lm nvcr.io/nvidia/pytorch:23.09-py3 \
  bash examples/retro/train_retro_2b_distributed.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH"

```
NOTE: Depending on the environment you are running it the above command might look slightly different.

NOTE: Due to how Retro preprocess and caches elements of the pretraining dataset before training begins, some arguments are auto-loaded from the Retro preprocessing configuration. These loaded arguments include:

- `--data-path`
- `--data-cache-path`
- `--eval-interval`
- `--eval-iters`
- `--global-batch-size`
- `--tokenizer-type`
- `--tokenizer-model`
- `--vocab-file`
- `--merge-file`
- `--seed`
- `--seq-length`
- `--train-samples`


## 2. Data Preprocessing
<a id="markdown-data-preprocessing" name="data-preprocessing"></a>

Retro preprocesses and caches data prior to pretraining, to greatly speed up pretraining. During data preprocessing, the retrieval database is built, and neighbor IDs are queried for each sample within the pretraining dataset. Please see `preprocess_data.sh` for an example script to preprocess data for Retro. The reference documentation for data preprocessing can be found [here](tools/retro/README.md).


## 3. Configurations
<a id="markdown-configurations" name="configurations"></a>
The example in this folder shows you how to run a 2B model. Below are a few other example configurations.

### 857M
```
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --seq-length 2048 \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \

```

### 4B
```
       --num-layers 48 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \

```
