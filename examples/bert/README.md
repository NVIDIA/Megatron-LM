# BERT MODEL

## Table of contents
- [1. Training Setup](#1-training-setup)
- [2. Configurations](#2-configurations)

## 1. Training setup
<a id="markdown-training-setup" name="training-setup"></a>

To run the model using a docker container run it as follows
```
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3
CHECKPOINT_PATH="" #<Specify path>
TENSORBOARD_LOGS_PATH=""#<Specify path>
VOCAB_FILE="" #<Specify path to file>//bert-vocab.txt
DATA_PATH="" #<Specify path and file prefix>_text_document

docker run \
  --gpus=all \
  --ipc=host \
  --workdir /workspace/megatron-lm \
  -v /path/to/data:/path/to/data \
  -v /path/to/megatron-lm:/workspace/megatron-lm \
  megatron-lm nvcr.io/nvidia/pytorch:24.01-py3 \
  bash examples/bert/train_bert_340m_distributed.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $DATA_PATH "

```
NOTE: Depending on the environment you are running it the above command might like slightly different.


## 2. Configurations
<a id="markdown-configurations" name="configurations"></a>
The example in this folder shows you how to run 340m large model. There are other configs you could run as well

### 4B
```
       --num-layers 48 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \

```

### 20B
```
       --num-layers 48 \
       --hidden-size 6144 \
       --num-attention-heads 96 \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 4 \

```