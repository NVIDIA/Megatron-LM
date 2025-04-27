# Mixtral 8x7B and 8X22B MoE Model pretraining

## 1. Prepare dataset
Download full dataset to train with real data and setting value of `MOCK_DATA` to `0` during training. By default `MOCK_DATA` is `1`

```
mkdir -p dataset
cd dataset
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.idx
```


## 2. Start the training
Start the docker
```
docker run \
 -d \
 --name=mixtral_pretrain \
 --ipc=host \
 --network=host \
 --device=/dev/kfd \
 --device=/dev/dri \
 --cap-add=SYS_PTRACE \
 --cap-add=CAP_SYS_ADMIN \
 --security-opt seccomp=unconfined \
 --group-add video \
 --privileged \
 --device=/dev/infiniband \
 --entrypoint /bin/bash \
 -it docker.io/rocm/megatron-lm:latest sleep infinity
 
```

Enter into the container
```
docker exec -it mixtral_pretrain bash 
```

## 3. Prepare the tokenizer
```
mkdir -p </path/to/tokenizer/>
cd </path/to/tokenizer/>

export HF_TOKEN="hf_xxx" #set huggingface access token to be able to download tokenizer
wget --header="Authorization: Bearer $HF_TOKEN" -O ./tokenizer.model https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/resolve/main/tokenizer.mod

```

## 4. Run on Single Node
Run command for Mixtral 8x7B with Mock data :
```
 TOKENIZER_MODEL=</path/to/tokenizer.model> \
 RECOMPUTE_NUM_LAYERS=0 \
 TEE_OUTPUT=1 MBS=1 GBS=16 TP_SIZE=1 PP_SIZE=1 AC=none \
 PR=bf16 EP_SIZE=8 ETP_SIZE=1 SEQLEN=4096 FORCE_BALANCE=true MOCK_DATA=1 \
 RUN_ENV=cluster MODEL_SIZE=8x7B TRAIN_ITERS=50 bash examples/mixtral/train_mixtral_moe.sh
```

Sample run command for a proxy `NUM_LAYERS=4` Mixtral 8x22B model on single node with Mock data :
```
 TOKENIZER_MODEL=</path/to/tokenizer.model> \
 RECOMPUTE_NUM_LAYERS=4 \
 TEE_OUTPUT=1 MBS=1 GBS=16 TP_SIZE=1 PP_SIZE=1 AC=full NUM_LAYERS=4 \
 PR=bf16 EP_SIZE=8 ETP_SIZE=1 SEQLEN=8192 FORCE_BALANCE=true MOCK_DATA=1  \
 RUN_ENV=cluster MODEL_SIZE=8x22B TRAIN_ITERS=50 bash examples/mixtral/train_mixtral_moe.sh
 ```

Note: Full model training of 8x22B requires multinode GPUs.

## 5. Start multinode training in slurm environment

With slurm environment, the pretraining can be launched with the following script.

Note: Before the run, please modify the $TOKENIZER_MODEL and $DATA_DIR variables, as well as the SBATCH arguments accordingly in the slurm script.     

Mixtral 8X7B
```
  export TOKENIZER_MODEL=/path/to/tokenizer.model
  export DATA_DIR=/path/to/dataset
  sbatch examples/mixtral/train_mixtral_8x7B_slurm.sh
```

Mixtral 8X22B
```
  export TOKENIZER_MODEL=/path/to/tokenizer.model
  export DATA_DIR=/path/to/dataset
  sbatch examples/mixtral/train_mixtral_8x22B_slurm.sh
```