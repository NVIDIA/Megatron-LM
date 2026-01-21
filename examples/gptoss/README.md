# GPT-OSS Training Tutorial

## Setup

<!-- ```bash
# Start NeMo container with HF cache mounted
# Run this from the Megatron-LM root directory
docker run --rm -it -w /workdir \
  -v $(pwd):/workdir \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  --entrypoint bash \
  nvcr.io/nvidia/nemo:25.11
``` -->

## Step 0: Install Dependencies

```bash
export HOST_MEGATRON_LM_DIR="/path/to/your/host/megatron-lm"
git clone https://github.com/NVIDIA/Megatron-LM.git "$HOST_MEGATRON_LM_DIR"
cd "$HOST_MEGATRON_LM_DIR"
```

## Step 1: Convert HuggingFace to Megatron (Optional - skip if you already have a Megatron checkpoint)

Change the number of gpus in `--nproc-per-node` and the model in `--hf-model` as needed.

```bash
torchrun --nproc-per-node=8 examples/gptoss/01_convert_hf.py --hf-model openai/gpt-oss-20b
```

## Step 2: Train from Scratch

To train from scratch first follow the steps below to setup the environment appropriately before running the training script in docker.

### Setup Environment

```bash
# Change these based on model and directory from previous conversion step
export MODEL_DIR_NAME="openai_gpt-oss_20b"

export HOST_CHECKPOINT_PATH="./megatron_checkpoints/${MODEL_DIR_NAME}"
export HOST_TENSORBOARD_LOGS_PATH="./tensorboard_logs/${MODEL_DIR_NAME}"
```

By default we will use mock data to train the model in the example below. To use your own data, set the following environment variables:

```bash
# Optional: For real data
export HOST_TOKENIZER_MODEL_PATH="/path/to/host/tokenizer.model"
export HOST_DATA_PREFIX="/path/to/host/mydata_prefix"
```

### Setup Training Configurations

Run the following to create a `distributed_config.env` file with the appropriate distributed training configurations. Change the values as needed for your setup.

```bash
cat > ./distributed_config.env << 'EOF'
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
EOF
```

### Run Container with Mounted Volumes

To train using mock data, run the following command:
```bash
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "${HOST_MEGATRON_LM_DIR}:/workspace/megatron-lm" \
  -v "${HOST_CHECKPOINT_PATH}:/workspace/checkpoints" \
  -v "${HOST_TENSORBOARD_LOGS_PATH}:/workspace/tensorboard_logs" \
  -v "./distributed_config.env:/workspace/megatron-lm/examples/gptoss/distributed_config.env" \
  --workdir /workspace/megatron-lm \
  $PYTORCH_IMAGE \
  bash examples/gptoss/02_train.sh \
    --checkpoint-path /workspace/checkpoints \
    --tensorboard-logs-path /workspace/tensorboard_logs \
    --distributed-config-file /workspace/megatron-lm/examples/gptoss/distributed_config.env \
  2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_mock_$(date +'%y-%m-%d_%H-%M-%S').log"
```
**Note:** If you run into issues generating mock data one solution might be to reduce the number of GPUs to 1 and try to generate the data again.

If using real data with with the `HOST_TOKENIZER_MODEL_PATH` and `HOST_DATA_PREFIX` environment variables set, run the following command instead:

```bash
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "${HOST_MEGATRON_LM_DIR}:/workspace/megatron-lm" \
  -v "${HOST_CHECKPOINT_PATH}:/workspace/checkpoints" \
  -v "${HOST_TENSORBOARD_LOGS_PATH}:/workspace/tensorboard_logs" \
  -v "${HOST_TOKENIZER_MODEL_PATH}:/workspace/tokenizer_model" \
  -v "$(dirname "${HOST_DATA_PREFIX}"):/workspace/data_dir" \
  -v "./distributed_config.env:/workspace/megatron-lm/examples/gptoss/distributed_config.env" \
  --workdir /workspace/megatron-lm \
  $PYTORCH_IMAGE \
  bash examples/gptoss/02_train.sh \
    --checkpoint-path /workspace/checkpoints \
    --tensorboard-logs-path /workspace/tensorboard_logs \
    --tokenizer /workspace/tokenizer_model \
    --data "/workspace/data_dir/$(basename "${HOST_DATA_PREFIX}")" \
    --distributed-config-file /workspace/megatron-lm/examples/gptoss/distributed_config.env \
  2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_custom_$(date +'%y-%m-%d_%H-%M-%S').log"
```