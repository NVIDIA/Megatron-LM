# QWen2 ckpt converter

## Download QWen2 Checkpoints
Download QWen2 HF format checkpoint from [HF-hub](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

Or you can simply run this following script to download QWen2 into a specific folder.
```python
from huggingface_hub import snapshot_download
SAVED_DIR = "" # Specify the saved directory
# Download HF checkpoints
snapshot_download(repo_id="Qwen/Qwen2.5-3B-Instruct", ignore_patterns=["*.pt"], local_dir=SAVED_DIR, local_dir_use_symlinks=False)
```

## Convert QWen2 checkpoints from HF to MCore
Since MCore 0.7, we support using distributed checkpointing to load and save checkpoints with different parallel mappings.
To convert HF model to distributed checkpoints, use following instructions:

```
TOKENIZER_MODEL=/workspace/checkpoints/qwen2/tokenizer.model
MEGATRON_PATH="/workspace/megatron-lm"
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

TARGET_TP_SIZE=1
TARGET_EP_SIZE=1
TARGET_PP_SIZE=1

HF_FORMAT_DIR=/workspace/checkpoints/qwen2-hf
MCORE_FORMAT_DIR=/workspace/checkpoints/qwen-mcore-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}

TARGET_EP_SIZE=${TARGET_EP_SIZE:-1}
TARGET_PP_SIZE=${TARGET_PP_SIZE:-1}
TARGET_CKPT_FORMAT=${TARGET_CKPT_FORMAT:-"torch_dist"}

torchrun --nproc-per-node=1 --nnodes=1 checkpoint/convert.py \
--model-type GPT \
--loader qwen2_hf \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MCORE_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_MODEL} \
--target-ckpt-format ${TARGET_CKPT_FORMAT}```
```

## Convert QWen2 checkpoints from MCore to HF
Since MCore 0.7, we support using distributed checkpointing to load and save checkpoints with different parallel mappings.
To convert HF model to distributed checkpoints, use following instructions:

```
MEGATRON_PATH="/workspace/megatron-lm"
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

MCORE_FORMAT_DIR=/workspace/checkpoints/qwen-mcore-TP1PP1EP1
HF_FORMAT_DIR=/workspace/checkpoints/qwen2-hf

torchrun --nproc-per-node=1 --nnodes=1 checkpoint/convert.py \
--model-type GPT \
--loader mcore \
--saver qwen2_hf \
--load-dir ${MCORE_FORMAT_DIR} \
--save-dir ${HF_FORMAT_DIR}
```
NOTE: for qwen2moe, need to set gate=True for shared_experts in gpt_layer_specs.py

## Acknowledgements
Contributors outside NVIDIA for the huggingface converter and example of QWen models in Megatron-Core:
- QWen Team
