# Deepseek-v2 Megatron Training
This guide provides the steps for setting up the environment and configuring the script to train deepseek-v2-lite model.

## 1. Docker Setup

1. **Download Docker Image**
   Download the Docker image required for training:
   `docker pull <docker_image>`

2. **Launch Docker Container**
   Start the Docker container:
   `docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name megatron_training_env <docker_image>`

## 2. Prepare Dataset
Skip this step, if you already have the dataset.

Download dataset using the command
<pre>
mkdir deepseek-datasets
cd deepseek-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.idx
</pre>

## 3. Configurations in Script (`Megatron-LM/examples/deepseek_v2`)
Use `train_deepseekv2.sh` script

### 3.1 Dataset
You can use either mock data or real data for training.

- **Mock Data:**
  Use `MOCK_DATA` variable to toggle between mock and real data. Default value is 1.
  ```bash
  MOCK_DATA=1
  ```
- **Real Data:**
  Update the `DATA_DIR` to the location where your dataset is stored:
  ```bash
  MOCK_DATA=0
  DATA_DIR="/path/to/deepseek-datasets"  # Change to where your dataset is stored
  ```
### 3.2 Tokenizer
DeepSeek-V2 uses `DeepSeekV2Tokenizer`

## 4. How to Run

### 4.1 Single Node Training
To run the training on a single node, go to Megatron-LM folder, use the following command:
```bash
cd /workspace/Megatron-LM
GEMM_TUNING=1 PR=bf16 MBS=4 AC=none SEQ_LEN=4096 PAD_LEN=4096 bash examples/deepseek_v2/train_deepseekv2.sh
```

## 5. Key Variables to Pay Attention To

- **PR:**
  Stands for precision for training. `bf16` for Bf16 (default), `fp8` for FP8 GEMMS.

- **GEMM_TUNING:**
  `1` to enable GEMM tuning, which boosts performance by using the best GEMM kernels.

- **TRAIN_ITERS:**
  Set the total number of iterations.

- **MOCK_DATA:**
  Use MOCK_DATA if set to 1, otherwise use the real data provided by user (DEFAULT: 1)

- **MBS:**
  Micro batch size

- **GBS:**
  Global Batch size

- **SEQ_LEN:**
  Sequence length

- **AC:**
  Activation Checkpointing (`none`, `sel` , `full`). Default:`sel` (Selective). 

That's it! You'are now ready to train DeepSeek-v2-lite model.
