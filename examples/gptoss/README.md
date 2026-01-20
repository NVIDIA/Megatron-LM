# GPT-OSS Training Tutorial

## Setup

```bash
# Start NeMo container with HF cache mounted
# Run this from the Megatron-LM root directory
docker run --rm -it -w /workdir \
  -v $(pwd):/workdir \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  --entrypoint bash \
  nvcr.io/nvidia/nemo:25.11
```

## Step 1: Convert HuggingFace to Megatron (Optional)

```bash
torchrun --nproc-per-node=8 examples/gptoss/01_convert_hf.py
```

## Step 2: Pretrain from Scratch

```bash
torchrun --nproc-per-node=8 examples/gptoss/02_pretrain.py
```

To load the converted checkpoint, uncomment the `pretrained_checkpoint` line in `02_pretrain.py`.