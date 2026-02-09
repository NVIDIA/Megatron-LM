# Llama2/Llama3 Model Pretraining Instructions

This guide provides the steps for setting up the environment and configuring the script to train Llama2 or Llama3/3.1 models.

---

## 1. Environment Setup

1. **Download Docker Image**  
   Download the Docker image required for training (e.g. docker images from rocm/megatron-lm):  
   `docker pull <image_name>`

2. **Launch Docker Container**  
   Start the Docker container:  
   `docker run -it <additional flags> <image_name>`
	 For example
   `docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined <image_name>`

    **Note** that it is recommended to use docker images like `rocm/pytorch-training:latest` which 
    have most requirements setup (e.g.,  `PyTorch >= 2.5.0`  is needed for full support of FSDP-v2).

3. **Install Megatron-LM**
  Run  `pip install .` in `/workspace/Megatron-LM` to install megatron package.

    **Note** that if you use `rocm/megatron-lm:latest` you also have to install ROCm/Megatron-LM as described above.

---

## 2. Script Configurations (`Megatron-LM/examples/llama`)
**Configuration scripts:** Use `train_llama3.sh` for Llama3/3.1 models and `train_llama2.sh` for Llama2 models.

### 2.1 Network Interface
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

### 2.2 Dataset
You can use either mock data or real data for training.

- **Mock Data:**  
  Replace the data path:
  ```bash
  --data-path $DATA_PATH \ with 
  --mock-data
  ```
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
  When training, real data is retrieved from `DATA_PATH` argument. 
  Update the `DATA_PATH` to the location where your dataset is stored:
  ```bash
  DATA_DIR="/root/.cache/data"  # Change to where your dataset is stored
  DATA_PATH=${DATA_DIR}/bookcorpus_text_sentence
  ```
  **Note** that when training you need to set `DATA_PATH` to the specific file name
  prefix that is pointing to `.bin` or `.idx` file. Remember also to be consistent with
  the choice of the tokenizer.

### 2.3 Tokenizer
When preparing a dataset, a tokenizer is required. The scripts support tokenizers
which are fully specified with choices of `TOKENIZER_TYPE` and `TOKENIZER_MODEL`. With
the exception of Llama 2 training script, the default `TOKENIZER_TYPE` is
`HuggingFaceTokenizer` and for it, only a valid `TOKENIZER_MODEL` is needed. When `TOKENIZER_TYPE` is `HuggingFaceTokenizer`, 
HuggingFace will automatically download the tokenizer files when specifying `TOKENIZER_MODEL` with the 
model id (e.g. `meta-llama/Llama-3-8B`).

**Note** You have to use a valid `HF_TOKEN` if it's a restricted tokenzier and pass the path of
`tokenizer` as `TOKENIZER_MODEL` to use it.

**Note** that while the training scripts support default tokenizers, the user is
adviced to be explicit about their tokenizer choice.

- **For Llama2 Training:**  
  Use the `Llama2Tokenizer`.

- **For Llama3 Training:**  
  Use the `HuggingFaceTokenizer`. Set the HuggingFace model link in the `TOKENIZER_MODEL` variable:
  ```bash
  TOKENIZER_MODEL=meta-llama/Llama-3.1-8B  # For Llama3
  ```

### 2.4 Multi-node Training
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
To run the training on a single node, go to the `Megatron-LM` folder. 

For Llama2 use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh
```

For Llama3/3.1 use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=128 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  bash examples/llama/train_llama3.sh
```

To run the training with `FSDP-v2` enabled, simply add `FSDP=1` argument, for example,
use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=16 TP=1 TE_FP8=0 FSDP=1 RECOMPUTE=1 SEQ_LENGTH=8192 MODEL_SIZE=8 bash examples/llama/train_llama3.sh
```
**Note:** It is suggested to use `TP=1` when FSDP is enabled, for higher throughput.
And FSDP-v2 is not supported with pipeline parallelism, expert parallelism, MCore's
distributed optimizer, gradient accumulation fusion and fp16.

#### FP8 options with Megatron-LM FSDP (train_llama3.sh)
  - **FP8 primary weights (fp8_model_init, param gather):** `TE_FP8=1 MEGATRON_FSDP=1 FP8_PARAM_GATHER=1 bash examples/llama/train_llama3.sh`
  - **BF16 primaries + FP8 caches (fp8_autocast):** `TE_FP8=1 MEGATRON_FSDP=1 FP8_PARAM_GATHER=0 bash examples/llama/train_llama3.sh`
  - Of note: the script always keeps the FP8 weight transpose cache for Megatron FSDP when FP8 is on; turning off `FP8_PARAM_GATHER` only removes the `--fp8-param-gather` flag.



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

---

## 4. Key Variables to Pay Attention To

- **BS:**  
  Sets the global batch size (default: 8).

- **MBS:**  
  Sets the micro batch size (default: 1).

- **SEQ_LENGTH:**  
  Sets the sequence length (default: 2048).

- **TP:**  
  Tensor parallel `1`, `2`, `4`, or `8` (default: 8). Note `TP` is disabled with `FSDP` or `MEGATRON_FSDP`.

- **TE_FP8:**  
  `0` for BF16 (default), `1` for FP8.

- **TE_FP8_RECIPE:**  
  `delayed` (default), `tensorwise`, `mxfp8` (supported on MI350) 

- **GEMM_TUNING:**  
  `1` (default) to enable GEMM tuning, which boosts performance by using the best GEMM kernels. Currently not supported for `mxfp8`.

- **USE_FLASH_ATTN:**  
  `1` (default) to enable Flash Attention.

- **FSDP:**  
  `1` to enable PyTorch FSDP-v2 (default: 0). Note that if FSDP is enabled, `--use-distributed-optimizer`, `--overlap-param-gather`, and `--sequence-parallel` will be automatically set off.

- **MEGATRON_FSDP:**
  `1` to enable Megatron-LM's custom FSDP with DTensor checkpointing (default: 0). It adds automatically `--use-megatron-fsdp --ckpt-format fsdp_dtensor` in the script. Of note, this disables `TP>1` automatically.

- **FP8_PARAM_GATHER:**
  Controls FP8 primaries vs FP8 caches when `TE_FP8=1` and `MEGATRON_FSDP=1` (default: 0). Set to `1` to add --fp8-param-gather` (weights kept in FP8, smaller all-gathers). Set to `0` to skip the `--fp8-param-gather` flag (weights stay BF16, FP8 caches are used for compute; FP8 weight transpose cache is still kept).

- **ENABLE_PROFILING:**  
  `1` to enable PyTorch profiling for performance analysis (default: 0).

- **transformer-impl:**  
  `transformer_engine` to use the Transformer Engine (TE). Set to `local` if you want to disable TE. Automatically added for `TE_FP8=1` since TE is required for fp8 support.

- **MODEL_SIZE:**  
  Set to `7` or `70` for Llama2, and `8` or `70` for Llama3/3.1 (default: 70).

- **TOTAL_ITERS:**  
  Set the total number of iterations (default: 12).

--- 

That's it! You've now set up the environment and configured the necessary settings for training Llama2 or Llama3 models.
