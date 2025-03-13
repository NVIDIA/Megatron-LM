# Llama2/Llama3 Model Pretraining Instructions

This guide provides the steps for setting up the environment and configuring the script to train Llama2 or Llama3 models.

---

## 1. Environment Setup

1. **Download Docker Image**  
   Download the Docker image required for training:  
   `docker pull <image_name>`
   
   **Note:** It is recommended to have `PyTorch >= 2.5.0` for full support of FSDP-v2.

2. **Launch Docker Container**  
   Start the Docker container:  
   `docker run -it --device /dev/dri --device /dev/kfd --device /dev/infiniband --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name megatron_training_env <image_name>`

---

## 2. Configurations in Script (`Megatron-LM/examples/llama`)
Use `train_llama3.sh` for Llama3/3.1 models and `train_llama2.sh` for Llama2 models.

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
  Use `MOCK_DATA` variable to toggle between mock and real data. Default value is 1.
  ```bash
  MOCK_DATA=1
  ```
- **Real Data:**
  Update the `DATA_PATH` to the location where your dataset is stored:
  ```bash
  MOCK_DATA=0
  DATA_PATH="/data/bookcorpus_text_sentence" # Change to where your dataset is stored
  ```
- **Downloading the dataset:**
  Set argument `DATASET` to the dataset you would like to use. Currently, two datasets are supported `DATASET=wiki` and `DATASET=bookcorpus`. Use the following command to download the dataset:
  ```bash
  DATASET=wiki bash examples/llama/prepare_dataset.sh #for wiki-en dataset
  DATASET=bookcorpus bash examples/llama/prepare_dataset.sh #for bookcorpus dataset

### 2.3 Tokenizer
You can assign the path of existing tokenizer to the command line argument `TOKENIZER_MODEL`. If tokenizer is not found, it will be downloaded to the default tokenizer model path: `${DATA_DIR}/tokenizer_llamaN` (N=2 or 3).

- **For Llama2 Training:**
  Uses either the `Llama2Tokenizer` or `HuggingFaceTokenizer`(default).

- **For Llama3 Training:**
  Use the `HuggingFaceTokenizer`. Set the HuggingFace model path in the `TOKENIZER_MODEL` variable:
  ```bash
  TOKENIZER_MODEL=meta-llama/Llama-3.1-8B  # For Llama3
  ```

  Otherwise, if you do not have Llama3.1 tokenizer locally, you need to set your personal HuggingFace access token `HF_TOKEN` in the script to download the tokenizer. To set the `HF_TOKEN` for Llama3.1 model, you first need to apply access to Llama3.1 model via this [link](https://huggingface.co/meta-llama/Llama-3.1-8B). After you are authorized, you are able to set your personal HuggingFace access token `HF_TOKEN` in your personal setting page and update the following variable in the script.

### 2.4 Multi-node Training
If you're running multi-node training, update the following environment variables on each node.They can also be passed as command line arguments.

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
  Set `DATA_CACHE_PATH` to a common directory accessible by all the nodes (for eg, an NFS directory) for multi-node runs
  ```bash
  DATA_CACHE_PATH=/root/cache #Set to a common directory for multi-node runs
  ```

 - **Network Drivers Inside Docker:**
   For multi-node runs, make sure correct network drivers are installed on the nodes. If inside a docker, either install the drivers inside the docker container or pass the network drivers from the host while creating docker container.

   ```bash
   # specify which RDMA interfaces to use for communication
   export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
   ```

---

## 3. How to Run

### 3.1 Single Node Training
To run the training on a single node, go to Megatron-LM folder, use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=128 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  bash examples/llama/train_llama3.sh
```

To run the training with `FSDP-v2` enabled, simply add `FSDP=1` argument, for example, use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=16 TP=1 TE_FP8=0 FSDP=1 RECOMPUTE=1 SEQ_LENGTH=8192 MODEL_SIZE=70 bash examples/llama/train_llama3.sh
```
**Note:** It is suggested to use `TP=1` when FSDP is enabled, for higher throughput. And FSDP-v2 is not supported with pipeline parallelism, expert parallelism, MCore's distributed optimizer, gradient accumulation fusion and fp16.


### 3.2 Multi-node Training
To run training on multiple nodes, launch the Docker container on each node. Example, follow these steps for 2 Node run with Node0 as master node :

- **On the Master Node0:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=0 bash examples/llama/train_llama3.sh
  ```

- **On the Worker Node1:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8  MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=1 bash examples/llama/train_llama3.sh
  ```
---

## 4. Key Variables to Pay Attention To

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

- **transformer-impl:**  
  `transformer_engine` to use the Transformer Engine (TE). Set to `local` if you want to disable TE.

- **MODEL_SIZE:**  
  Set to `7B` or `70B` for Llama2, or `8B` or `70B` for Llama3/3.1.

- **TOTAL_ITERS:**  
  Set the total number of iterations (default: 10).

- **MOCK_DATA:**
  Use MOCK_DATA if set to 1, otherwise use the real data provided by user (DEFAULT: 1)

- **MBS:**
  Micro batch size

- **BS:**
  Global Batch size

- **TP:**
  Tensor parallel (1, 2, 4, 8)

  Note `TP` is disabled with `FSDP`.

- **SEQ_LENGTH**:
  Sequence Length

--- 

That's it! You've now set up the environment and configured the necessary settings for training Llama2 or Llama3 models.
