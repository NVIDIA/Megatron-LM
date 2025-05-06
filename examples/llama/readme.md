# Llama2/Llama3 Model Pretraining Instructions

This guide provides the steps for setting up the environment and configuring the script
to train Llama2 or Llama3 models.

---

## 1. Environment Setup

Start a Docker container by running

```
docker run \
    -it --rm \
    --device /dev/dri --device /dev/kfd \
    --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v .:/workspace/Megatron-LM \
    --shm-size 64G \
    rocm/pytorch-training:latest bash
```

from ROCm/Megatron-LM repository root.

**Note** that it is recommended to use `rocm/pytorch-training:latest` like images which
have most requirements setup, for example `PyTorch >= 2.5.0` is needed for full support
of FSDP-v2.

Run

```
pip install .
```

in `/workspace/Megatron-LM` to install megatron package.

**Note** that it is also possible to use `rocm/megatron-lm:latest` like images, which
have ROCm/Megatron-LM already installed. If doing so, the bind mount is not required,
there is no need to install anything and please make sure to follow the README inside
the container to run these examples.

---

## 2. Configurations in Script (`Megatron-LM/examples/llama`)
Use `train_llama3.sh` for Llama3/3.1 models and `train_llama2.sh` for Llama2 models.

### 2.1 Network Interface
Update the network interface in the training scripts to match your system’s network
interface. To find your network interface, run

```bash
ip a
```

on host and update

```bash
export NCCL_SOCKET_IFNAME=ens50f0np0
export GLOO_SOCKET_IFNAME=ens50f0np0
```

in the training scripts based on the output. 

### 2.2 Dataset

### 2.2.1 Tokenizer
When preparing a dataset, a tokenizer is required. The scripts support tokenizers
which are fully specified with choices of `TOKENIZER_TYPE` and `TOKENIZER_MODEL`. With
the exception of Llama 2 training script, the default `TOKENIZER_TYPE` is
`HuggingFaceTokenizer` and for it, only a valid `TOKENIZER_MODEL` is needed. For
example, after obtaining a [permission](https://huggingface.co/meta-llama/Llama-3.1-8B),
run

```bash
wget --header="Authorization: Bearer $HF_TOKEN" -O tokenizer/special_tokens_map.json https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/special_tokens_map.json
wget --header="Authorization: Bearer $HF_TOKEN" -O tokenizer/tokenizer.json https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/tokenizer.json
wget --header="Authorization: Bearer $HF_TOKEN" -O tokenizer/tokenizer.model https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/original/tokenizer.model
wget --header="Authorization: Bearer $HF_TOKEN" -O tokenizer/tokenizer_config.json https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/tokenizer_config.json
```

with a valid `HF_TOKEN` to download Llama 3.1 tokenizer and pass the path of
`tokenizer` as `TOKENIZER_MODEL` to use it.

**Note** that while the training scripts support default tokenizers, the user is
adviced to be explicit about their tokenizer choice.

### 2.2.2 Usage
You can use either mock data or real data for training.

- **Mock Data:**  
  Mock data is used when no `DATA_PATH` argument is passed. 

- **Downloading real data:**  
  Set argument `DATASET` to the dataset you would like to use: three datasets
  `bookcorpus`, `fineweb` and `wiki` are supported. For example, use the
  following command to download and preprocess the bookcorpus dataset:

  ```bash
  DATASET=bookcorpus DATA_DIR=bookcorpus TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf bash ./examples/llama/prepare_dataset.sh
  ```

  where `TOKENIZER_MODEL` can be any accessible HuggingFace tokenizer. Remember to
  either pre-download the tokenizer or setup HuggingFace access otherwise when needed.

- **Real Data:**  
  When training, real data is retrieved from `DATA_PATH` argument, for example
  bookcorpus data can be used with

  ```bash
  DATA_PATH=bookcorpus/data_text_document TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf bash ./examples/llama/train_llama2.sh 
  ```

  **Note** that when training you need to set `DATA_PATH` to the specific file name
  prefix that is pointing to `.bin` or `.idx` file. Remember also to be consistent with
  the choice of the tokenizer.

### 2.3 Multi-node Training
If you're running multi-node training, update the following environment variables on
each node. They can also be passed as command line arguments.

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

- **DATA_CACHE_PATH:**
  Set `DATA_CACHE_PATH` to a common directory accessible by all the nodes (for eg, an
  NFS directory) for multi-node runs
  ```bash
  DATA_CACHE_PATH=/root/cache
  ```
  **Note** that for multi-node runs, remember to properly setup a bind mount, with the
  default mount point `/root/cache` inside the container, to a host directory
  accessible to all of the nodes, for example a NFS directory. For non-default mount
  points, set `DATA_CACHE_PATH` appropriately and pass it to the training scripts.

 - **Network Drivers Inside Docker:**
   For multi-node runs, make sure correct network drivers are installed on the nodes.
   If inside a docker, either install the drivers inside the docker container or pass
   the network drivers from the host while creating docker container.
   ```bash
   # specify which RDMA interfaces to use for communication
   export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
   ```

---

## 3. How to Run

### 3.1 Single Node Training
To run the training on a single node, go to Megatron-LM folder, use the following
command:
```bash
TEE_OUTPUT=1 MBS=2 BS=128 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  bash examples/llama/train_llama3.sh
```

To run the training with `FSDP-v2` enabled, simply add `FSDP=1` argument, for example,
use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=16 TP=1 TE_FP8=0 FSDP=1 RECOMPUTE=1 SEQ_LENGTH=8192 MODEL_SIZE=70 bash examples/llama/train_llama3.sh
```
**Note:** It is suggested to use `TP=1` when FSDP is enabled, for higher throughput.
And FSDP-v2 is not supported with pipeline parallelism, expert parallelism, MCore's
distributed optimizer, gradient accumulation fusion and fp16.

### 3.2 Multi-node Training
To run training on multiple nodes, launch the Docker container on each node. Example,
follow these steps for 2 Node run with Node0 as master node :

- **On the Master Node0:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8 MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=0 bash examples/llama/train_llama3.sh
  ```

- **On the Worker Node1:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8 MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=1 bash examples/llama/train_llama3.sh
  ```
  where `MASTER_ADDR=IP_NODE0` tells the script that the master node ip address is `IP_NODE0`.

### 3.3 Multi-node Training with Slurm 
In the slurm environment, launch the multinode training in the following way.   
  ```
  export HF_TOKEN=YourHuggingFaceToken
  export MODEL_NAME=llama3
  sbatch examples/llama/train_llama_slurm.sh <MODEL_SIZE> <MBS> <BATCH_SIZE_PER_NODE> <SEQ_LENTH> <TOTAL_ITERS> <FSDP> <RECOMPUTE>
  ```
For example, train llama 2 with multinodes: 
```
  export HF_TOKEN=YourHuggingFaceToken
  export MODEL_NAME=llama2
  sbatch examples/llama/train_llama_slurm.sh 13 6 48 4096 10 1 0
```
Train llama3 with multinodes:
```
  export HF_TOKEN=YourHuggingFaceToken
  export MODEL_NAME=llama3
  sbatch examples/llama/train_llama_slurm.sh 70 7 56 8192 10 1 1
---

## 4. Key Variables to Pay Attention To

- **BS:**  
  Sets the global batch size (default: 8)

- **MBS:**  
  Sets the micro batch size (default: 1)

- **SEQ_LENGTH:**  
  Sets the sequence length

- **TP:**  
  Tensor parallel (1, 2, 4, 8). Note `TP` is disabled with `FSDP`.

- **TE_FP8:**  
  `0` for B16 (default), `1` for FP8.

- **GEMM_TUNING:**  
  `1` to enable GEMM tuning, which boosts performance by using the best GEMM kernels.

- **USE_FLASH_ATTN:**  
  `1` to enable Flash Attention.

- **FSDP:**  
  `1` to enable torch fsdp-v2. 
  
  Note that if FSDP is enabled, `--use-distributed-optimizer`, `--overlap-param-gather`, `--sequence-parallel` will be automatically set off. 

- **ENABLE_PROFILING:**  
  `1` to enable PyTorch profiling for performance analysis.

- **MODEL_SIZE:**  
  Set to `7` or `70` for Llama2, and `8` or `70` for Llama3/3.1 (default: 70).

- **TOTAL_ITERS:**  
  Sets the total number of iterations.

--- 

That's it! You've now set up the environment and configured the necessary settings for
training Llama2 or Llama3 models.
