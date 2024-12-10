# Llama2/Llama3 Model Pretraining Instructions

This guide provides the steps for setting up the environment and configuring the script to train Llama2 or Llama3 models.

---

## 1. Environment Setup

1. **Download Docker Image**  
   Download the Docker image required for training:  
   `docker pull <image_name>`

2. **Launch Docker Container**  
   Start the Docker container:  
   `docker run -it <additional flags> <image_name>`

---

## 2. How to Run

### 2.1 Single Node Training
To run the training on a single node, go to Megatron-LM folder, use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh
```


### 2.2 Multi-node Training
To run training on multiple nodes, launch the Docker container on each node. Follow these steps:

- **On the Master Node:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh
  ```

- **On the Slave Node(s):**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh
  ```

## 3. Configurations in Script (`Megatron/examples/llama`)

### 3.1 Network Interface
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

### 3.2 Dataset
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
  DATA_DIR="/root/.cache/data"  # Change to where your dataset is stored
  DATA_PATH=${DATA_DIR}/bookcorpus_text_sentence
  ```

### 3.3 Tokenizer

- **For Llama2 Training:**  
  Use the `Llama2Tokenizer`.

- **For Llama3 Training:**  
  Use the `HuggingFaceTokenizer`. Set the HuggingFace model link in the `TOKENIZER_MODEL` variable:
  ```bash
  TOKENIZER_MODEL=meta-llama/Llama-3.1-8B  # For Llama3
  ```

### 3.4 Multi-node Training
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
  Set the rank of each node (0 for master, 1 for the first slave node, etc.):
  ```bash
  NODE_RANK="${NODE_RANK:-0}"
  ```

---

## 4. Key Variables to Pay Attention To

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
  Set to `7B` or `70B` for Llama2, or `8B` or `70B` for Llama3/3.1.

- **TOTAL_ITERS:**  
  Set the total number of iterations (default: 10).

--- 

That's it! You've now set up the environment and configured the necessary settings for training Llama2 or Llama3 models.
