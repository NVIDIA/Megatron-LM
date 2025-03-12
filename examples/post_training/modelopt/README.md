# NVIDIA TensorRT Model Optimizer (ModelOpt) Integration

ModelOpt (`nvidia-modelopt`) provides end-to-end model optimization for NVIDIA hardware including
quantization, sparsity, knowledge distillation, pruning, neural architecture search.
You can find more info abour ModelOpt at our Github repository https://github.com/NVIDIA/TensorRT-Model-Optimizer.

We support Megatron Core `GPTModel` and `MambaModel` as well as task-specific optimization
such as speculative decoding. Users can choose to start from Megatron-LM or NeMo framework.
The optimized model can be deploied with  NVIDIA TensorRT-LLM, vLLM, or SGLang.

## Table of Contents

[[_TOC_]]


## Getting Started with Post-Training Quantization (

> **IMPORTANT :** Example scripts require basic access (general available) to
> NVIDIA GPU Cloud (NGC). If you have yet to register and acquire a `NGC_CLI_API_KEY`, 
> please first register at https://ngc.nvidia.com/signin. 

Login to nvcr.io docker registry (using `NGC_CLI_API_KEY`) and start an interactive
section **at the root of the megatron-lm repo!** Export your `NGC_CLI_API_KEY` in the environment.
```sh
docker login nvcr.io

docker run --gpus all --init -it --rm -v $PWD:/workspace/megatron-lm \
    nvcr.io/nvidia/pytorch:24.10-py3 bash
cd /workspace/megatron-lm/examples/post_training/modelopt

export NGC_CLI_API_KEY=
```

Now let's start a simple FP8 quantization task. You must provide `HF_TOKEN` which grants you
access to `meta-llama/Llama-3.2-1B-Instruct`.
```sh
export HF_TOKEN=
bash convert.sh meta-llama/Llama-3.2-1B-Instruct
MLM_MODEL_CKPT=/tmp/megatron_workspace/meta-llama/Llama-3.2-1B-Instruct_mlm bash quantize.sh meta-llama/Llama-3.2-1B-Instruct fp8
```
The model card name (see the support list in `conf/`) is expected as an input to all the sample scripts.
Other arguments are specified as varibles (e.g. `TP=8`) where you can either set before `bash` or export
to the current bash environment upfront.

The script will perform per-tensor FP8 faked-quantization and generate some tokens as an indication thatthe quantized model still behaves correctly. The end results are stored in `/tmp/megatron_workspace/meta-llama/Llama-3.2-1B-Instruct_quant`. This is a Megatron Mcore distributed checkpoint (with additional states), which can be loaded for quantization-aware training (QAT) or exported for deployment.

## Export for TensorRT-LLM, vLLM, SGLang Deployment 

For supported Hugging Face models, TensorRT Model Optimizer can export the quantized model to
a  HF-like checkpoint with real-quantied weights.
```sh
MLM_MODEL_CKPT=/tmp/megatron_workspace/meta-llama/Llama-3.2-1B-Instruct_quant bash export.sh meta-llama/Llama-3.2-1B-Instruct
```
> **NOTE:** The HF-like export only supports pipeline parallelism (`PP`). Other parallelism must be
> set to 1. The exported checkpoint is sharded with safetensors. Although it is HF-like, this format
> currently cannot be loaded by `from_pretrained()`.
The exported checkpoint is stored in `/tmp/megatron_workspace/meta-llama/Llama-3.1-8B-Instruct_export` which can be provided as an input to most of the `LLM` APIs. For examples,
```
vllm serve /tmp/megatron_workspace/meta-llama/Llama-3.1-8B-Instruct_export --quantization modelopt
```
> **TROUBLESHOOTING:** You need a device with `sm>=89` (Ada Lovelace or Hopper) for FP8 compute.


## Advanced Usage
TBD
