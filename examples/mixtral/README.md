# Mixtral 8x7B Model Inference and Finetuning

## Download Mixtral 8x7B Checkpoints
Download Mixtral 8x7B HF format checkpoint from [HF-hub](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/)

Or you can simply run this following script to download Mixtral 8x7B into a specific folder.
```python
from huggingface_hub import snapshot_download
SAVED_DIR = "" # Specify the saved directory
# Download HF checkpoints
snapshot_download(repo_id="mistralai/Mixtral-8x7B-v0.1", ignore_patterns=["*.pt"], local_dir=SAVED_DIR, local_dir_use_symlinks=False)
```

## Convert Mixtral 8x7B checkpoints from HF to MCore
The HF checkpoints can be converted to Megatron format by using the provided checkpoint converter for HF format.
The target model parallel size(e.g. TP,PP,EP) should be specified.

Currently the converter doesn't support distributed checkpointing yet, so each different parallel config requires a specific checkpoint.
- For training, the recommended model parallel config is TP1EP8PP4
- For inference, the recommended model parallel config is TP1EP1PP2

```
TOKENIZER_MODEL=/workspace/checkpoints/mixtral-hf/tokenizer.model
MEGATRON_PATH="/workspace/megatron-lm"
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

TARGET_TP_SIZE=""
TARGET_EP_SIZE=""
TARGET_PP_SIZE=""

HF_FORMAT_DIR=/workspace/checkpoints/mixtral-hf
MEGATRON_FORMAT_DIR=/workspace/checkpoints/mixtral-mcore-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}

python tools/checkpoint/convert.py \
--model-type GPT \
--loader loader_mixtral_hf \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_MODEL}
```

## Text generation with Mixtral 8x7B
Inference with Mixtral 8x7B requires at least 2 GPUS, such that a distributed checkpoint with EP>=2 or PP>=2 converted with above script is needed.

The Megatron-LM have included a simple REST server to use for text generation in `tools/run_text_generation_server.py`, launch it with the following script:
```
#!/bin/bash
# This example will start serving the Mixtral 8x7B model.
DISTRIBUTED_ARGS="--nproc_per_node 2 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=<Path to checkpoint>
TOKENIZER_MODEL=<Path to tokenizer (e.g. /tokenizer.model)>

export CUDA_DEVICE_MAX_CONNECTIONS=1

pip install flask-restful

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 2  \
       --expert-model-parallel-size 1 \
       --load ${CHECKPOINT}  \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model $TOKENIZER_MODEL \
       --use-mcore-models \
       --max-position-embeddings 32768 \
       --num-layers 32 \
       --hidden-size 4096 \
       --ffn-hidden-size 14336 \
       --num-attention-heads 32 \
       --normalization RMSNorm \
       --disable-bias-linear \
       --position-embedding-type rope \
       --no-position-embedding \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --group-query-attention \
       --num-query-groups 8 \
       --bf16  \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --seed 42 \
       --num-experts 8 \
       --moe-router-topk 2 \
       --moe-token-dispatcher-type alltoall \
       --moe-grouped-gemm \
       --mock-data \
       --rotary-base 1000000
```

Once the server is running you can use `tools/text_generation_cli.py` to query it, it takes one argument which is the host the server is running on.

```
python tools/text_generation_cli.py localhost:5000
```


## Finetuning from pretrained Mixtral 8x7B
To finetuning pretrained Mixtral 8x7B, use the following scripts:


```bash
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.04-py3
CHECKPOINT_PATH="" # Speicfy path to checkpoint dir
TOKENIZER_MODEL="" # Specify path to tokenizer.model
DATA_PATH="" # Specify path to data

docker run \
    --gpus=all \
    --ipc=host \
    --workdir /workspace/megatron-lm \
    -v /path/to/data:/path/to/data \
    -v /path/to/megatron-lm:/workspace/megatron-lm \
    $PYTORCH_IMAGE \
    bash examples/mixtral/train_mixtral_8x7b_distributed.sh $CHECKPOINT_PATH $TOKENIZER_MODEL $DATA_PATH
```

The above functionality also applys to Mixtral 8x22B actually, you should set the model config (including hidden_size/head_num/num_layers/ffn_hidden_size) properly according to the original [config](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/main/config.json).

## Acknowledgements
Contributors outside NVIDIA for the huggingface converter and example of Mixtral models in Megatron-Core:
- Peng Li <jerry.lp@alibaba-inc.com>
- Jun Huang <huangjun.hj@alibaba-inc.com>
