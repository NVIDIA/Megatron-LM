# Pretrain sample for Qwen2 Model on ROCm Megatron-LM


## Dataset-and-model-download
```bash
mkdir -p temp/qwen-datasets

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.bin

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.idx
```
You can also use huggingfacecli to download the models in advance.

```bash
huggingface-cli download --token {your huggingface token} --resume-download Qwen/Qwen2-7B
```

Set the HuggingFace model link in the `TOKENIZER_MODEL` variable for downloaded model:
```bash
TOKENIZER_MODEL=Qwen/Qwen2-7B  # For Qwen2
```

###   Defining Dataset
You can use either mock data or real data for training.

- **Mock Data:**
  Replace the data path:
  ```bash
  --data-path $DATA_PATH \ with
  --mock-data
  ```

- **Real Data:**
  Update the `DATA_PATH` to the location where your dataset is stored:
  ```bash
  DATA_PATH=/myworkspace/bookcorpus_text_sentence
  ```


## Model Training Process

### How to Run
```bash
MODEL_SIZE=7 SEQ_LENGTH=1024 SEQ_PARALLEL=1 TP=1 PP=1 CP=1 BS=8 MBS=1 RECOMPUTE_ACTIVATIONS=0 DIST_OPTIM=1 USE_FLASH_ATTN=1 TE_FP8=0 TOTAL_ITERS=3 bash ./examples/qwen/train_qwen2.sh
```
###   With custom dataset path and model path.
```bash
DATA_DIR=/myworkspace/qwen-datasets/ TOKENIZER_MODEL=/myworkspace/Qwen-7b MODEL_SIZE=7 SEQ_LENGTH=1024 SEQ_PARALLEL=1 TP=1 PP=1 CP=1 BS=16 MBS=2 RECOMPUTE_ACTIVATIONS=0 DIST_OPTIM=1 USE_FLASH_ATTN=1 TE_FP8=0 TOTAL_ITERS=30 bash ./examples/qwen/train_qwen2.sh
```


###    Single Node Training
To run the training on a single node, go to Megatron-LM folder, use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/qwen/train_qwen2.sh
```


###    Multi-node Training
To run training on multiple nodes, launch the Docker container on each node. Follow these steps:

- **On the Master Node:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/qwen/train_qwen2.sh
  ```

- **On the Worker Node(s):**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/qwen/train_qwen2.sh
  ```

## Configurable options in Script (`Megatron/examples/qwen`)

###    Network Interface
Update the network interface in the script to match your system’s network interface.
To find your network interface, run (out of container):
```bash
ip a
```
Then, update the following variables in the script:
```bash
export NCCL_SOCKET_IFNAME=ens50f0np0
export GLOO_SOCKET_IFNAME=ens50f0np0
```



###   Multi-node Training
If you're running multi-node training, update the following environment variables:

- **Master Address:**
  Change `localhost` to the master node's hostname:
  ```bash
  MASTER_ADDR="${MASTER_ADDR:-localhost}"
  ```

- **Number of Nodes:**
  Set the number of nodes you want to train on (e.g., 2, 4, 8):
  ```bash
  NNODES="${NNODES:-1}"
  ```

- **Node Rank:**
  Set the rank of each node (0 for master, 1 for the first worker node, etc.):
  ```bash
  NODE_RANK="${NODE_RANK:-0}"
  ```

---

## Key Variables to Pay Attention To

- **TE_FP8:**
  `0` for BP16 (default), `1` for FP8.

- **GEMM_TUNING:**
  `1` to enable GEMM tuning, which boosts performance by using the best GEMM kernels.

- **USE_FLASH_ATTN:**
  `1` to enable Flash Attention.

- **ENABLE_PROFILING:**
  `1` to enable PyTorch profiling for performance analysis.

- **transformer-impl:**
  `transformer_engine` to use the Transformer Engine (TE). Set to `local` if you want to disable TE.

- **MODEL_SIZE:**
  Set to 0.5 or 1.5 or 7 or 72

- **TOTAL_ITERS:**
  Set the total number of iterations (default: 10).

---

That's it! You've now set up the environment and configured the necessary settings for training Qwen2 Models.
