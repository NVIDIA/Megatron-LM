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
   `docker run -it <additional flags> <image_name>`

---

## 2. How to Run

### 2.1 Single Node Training
To run the training on a single node, go to Megatron-LM folder, use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh
```

To run the training with `FSDP-v2` enabled, simply add `FSDP=1` argument, for example, use the following command:
```bash
TEE_OUTPUT=1 MBS=2 BS=16 TP=1 TE_FP8=0 FSDP=1 SEQ_LENGTH=8192 bash examples/llama/train_llama2.sh
```
**Note:** It is suggested to use `TP=1` when FSDP is enabled, for higher throughput. And FSDP-v2 is not supported with pipeline parallelism, expert parallelism, MCore's distributed optimizer, gradient accumulation fusion and fp16.

### 2.2 Multi-node Training
To run training on multiple nodes, launch the Docker container on each node. Follow these steps:

- **On the Master Node:**
  ```bash
  TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh
  ```

- **On the Worker Node(s):**
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
  Mock data is applied when argument is set to be `MOCK_DATA=1` and it will be automatically downloaded. For example, you can pass the argument to the command:

  ```bash
  TEE_OUTPUT=1 MOCK_DATA=1 bash examples/llama/train_llama2.sh
  ```

- **Real Data:**  
  Real data is retrieved from `$DATA_PATH` when argument is set to be `MOCK_DATA=0`. Update the `DATA_PATH` to the location where your dataset is stored either in the script or through command line:

  ```bash
  TEE_OUTPUT=1 DATA_DIR=path/to/your/data bash examples/llama/train_llama2.sh 
  ```

  By default, `DATA_DIR` should have the following structure, where tokenizer directory also resides inside `DATA_DIR`:

  ```
  - DATA_DIR/
    - bookcorpus_text_sentence/
    - tokenizer_llama2/
  ```
- **Downloading the dataset:**
  Set argument `DATASET` to the dataset you would like to use. Currently, two datasets are supported `DATASET=wiki` and `DATASET=bookcorpus`. Use the following command to download the dataset:
  ```bash
  DATASET=wiki bash examples/llama/prepare_dataset.sh #for wiki-en dataset
  DATASET=bookcorpus bash examples/llama/prepare_dataset.sh #for bookcorpus dataset
  ```

  Then you could launch training using the following commands:
  ```bash
  TEE_OUTPUT=1 MBS=1 BS=8 TP=8 TE_FP8=0 FSDP=1 SEQ_LENGTH=8192 TOKENIZER_TYPE=Llama2Tokenizer DATA_DIR=./tmp/data/bookcorpus bash examples/llama/train_llama2.sh #for downloaded bookcorpus dataset

  TEE_OUTPUT=1 MBS=1 BS=8 TP=8 TE_FP8=0 FSDP=1 SEQ_LENGTH=8192 TOKENIZER_TYPE=Llama2Tokenizer DATA_DIR=./tmp/data/wiki DATA_PATH=./tmp/data/wiki/wikipedia_20220301.en.train.jsonl_text_document bash examples/llama/train_llama2.sh #for downloaded wikipedia dataset

  ```

- **Note:**
  When training Megatron-LM, in the training script, you need to set data path to the specific file name that is pointing to `.bin` or `.idx` file, for example:
  ```bash
  DATA_PATH=${DATA_DIR}/wikipedia_20220301.en.train.jsonl_text_document
  ``` 

### 3.3 Tokenizer
You can assign the path of existing tokenizer to the command line argument `TOKENIZER_MODEL`, or directly modify it in the script. If tokenizer is not found, it will be downloaded to the default tokenizer model path: `${DATA_DIR}/tokenizer_llamaN` (N=2 or 3).

Set tokenizer path via command line:

  ```bash
  TEE_OUTPUT=1 TOKENIZER_MODEL=path/to/your/tokenizer bash examples/llama/train_llama3.sh
  ```

- **For Llama2 Training:**  
  Set `TOKENIZER_TYPE` to use either the `Llama2Tokenizer` or `HuggingFaceTokenizer`. For example:
  ```bash
  TEE_OUTPUT=1 TOKENIZER_TYPE=Llama2Tokenizer DATA_DIR=path/to/your/data bash examples/llama/train_llama2.sh
  ```
  
  **Note:**
      If using `HuggingFaceTokenizer` as the tokenizer-type for Llama2 training, you need to set path to tokenizer directory path (not `tokenizer.model` path), for example:

  ```bash
  TOKENIZER_MODEL=${DATA_DIR}/tokenizer_llama2
  ```  

    If using `Llama2Tokenizer` as the tokenizer-type for Llama2 training, you need to set path to the tokenizer model path:

    ```bash
    TOKENIZER_MODEL=${DATA_DIR}/tokenizer_llama2/tokenizer.model
    ```

- **For Llama3 Training:**  
  Use the `HuggingFaceTokenizer`. Set the local tokenizer path to the `TOKENIZER_MODEL` variable, if it is already downloaded:
  ```bash
  TOKENIZER_MODEL=${DATA_DIR}/tokenizer_llama3  # For Llama3
  ```
  Otherwise, if you do not have Llama3.1 tokenizer locally, you need to set your personal HuggingFace access token `HF_TOKEN` in the script to download the tokenizer. To set the `HF_TOKEN` for Llama3.1 model, you first need to apply access to Llama3.1 model via this [link](https://huggingface.co/meta-llama/Llama-3.1-8B). After you are authorized, you are able to set your personal HuggingFace access token `HF_TOKEN` in your personal setting page and update the following variable in the script.

  ```bash
  export HF_TOKEN="hf_xxxx"
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
  Set the rank of each node (0 for master, 1 for the first worker node, etc.):
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

--- 

That's it! You've now set up the environment and configured the necessary settings for training Llama2 or Llama3 models.
