# Megatron Model Optimization and Deployment

## Installation
We recommend that users follow TensorRT-LLM's official installation guide to build it from source
and proceed with a containerized environment (`docker.io/tensorrt_llm/release:latest`):

```sh
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout v0.10.0
make -C docker release_build
```

> **TROUBLE SHOOTING:** rather than copying each folder separately in `docker/Dockerfile.multi`,
> you may need to copy the entire dir as `COPY ./ /src/tensorrt_llm` since a `git submodule` is
> called later which requires `.git` to continue.

Once the container is built, install `nvidia-modelopt` and additional dependencies for sharded checkpoint support:
```sh
pip install "nvidia-modelopt[all]~=0.13.0" --extra-index-url https://pypi.nvidia.com
pip install zarr tensorstore==0.1.45
```
TensorRT-LLM quantization functionalities are currently packaged in `nvidia-modelopt`.
You can find more documentation about `nvidia-modelopt` [here](https://nvidia.github.io/TensorRT-Model-Optimizer/).

## Support Matrix

The following matrix shows the current support for the PTQ + TensorRT-LLM export flow.

| model                       | fp16 | int8_sq | fp8 | int4_awq |
|-----------------------------|------|---------| ----| -------- |
| nextllm-2b                  | x    | x       |   x |          |
| nemotron3-8b                | x    |         |   x |          |
| nemotron3-15b               | x    |         |   x |          |
| llama2-text-7b              | x    | x       |   x |      TP2 |
| llama2-chat-70b             | x    | x       |   x |      TP4 |

Our PTQ + TensorRT-LLM flow has native support on MCore `GPTModel` with a mixed layer spec (native ParallelLinear
and Transformer-Engine Norm (`TENorm`). Note that this is not the default mcore gpt spec. You can still load the
following checkpoint formats with some remedy:

| GPTModel                          | sharded |                        remedy arguments     |
|-----------------------------------|---------|---------------------------------------------|
| megatron.legacy.model             |         | `--export-legacy-megatron` |
| TE-Fused (default mcore gpt spec) |         | `--export-te-mcore-model`       |
| TE-Fused (default mcore gpt spec) |       x |                                             |

> **TROUBLE SHOOTING:** If you are trying to load an unpacked `.nemo` sharded checkpoint, then typically you will
> need to adding `additional_sharded_prefix="model."` to `modelopt_load_checkpoint()` since NeMo has an additional
> `model.` wrapper on top of the `GPTModel`.

> **NOTE:** flag `--export-legacy-megatron` may not work on all legacy checkpoint versions.

## Examples

> **NOTE:** we only provide a simple text generation script to test the generated TensorRT-LLM engines. For
> a production-level API server or enterprise support, see [NeMo](https://github.com/NVIDIA/NeMo) and TensorRT-LLM's
> backend for [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server).

### Minitron-8B FP8 Quantization and TensorRT-LLM Deployment
First download the nemotron checkpoint from https://huggingface.co/nvidia/Minitron-8B-Base, extract the
sharded checkpoint from the `.nemo` tarbal and fix the tokenizer file name.

> **NOTE:** The following cloning method uses `ssh`, and assume you have registered the `ssh-key` in Hugging Face.
> If you are want to clone with `https`, then `git clone https://huggingface.co/nvidia/Minitron-8B-Base` with an access token.

```sh
git lfs install
git clone git@hf.co:nvidia/Minitron-8B-Base
cd Minitron-8B-Base/nemo
tar -xvf minitron-8b-base.nemo
cd ../..
```

Now launch the PTQ + TensorRT-LLM export script,
```sh
bash examples/inference/quantization/ptq_trtllm_minitron_8b ./Minitron-8B-Base None
```
By default, `cnn_dailymail` is used for calibration. The `GPTModel` will have quantizers for simulating the
quantization effect. The checkpoint will be saved optionally (with quantizers as additional states) and can
be restored for further evaluation or quantization-aware training. TensorRT-LLM checkpoint and engine are exported to `/tmp/trtllm_ckpt` and
built in `/tmp/trtllm_engine` by default.

The script expects `${CHECKPOINT_DIR}` (`./Minitron-8B-Base/nemo`) to have the following structure:

> **NOTE:** The .nemo checkpoint after extraction (including examples below) should all have the following strucure.

```
├── model_weights
│   ├── common.pt
│   ...
│
├── model_config.yaml
│...
```

> **NOTE:** The script is using `TP=8`. Change `$TP` in the script if your checkpoint has a different tensor
> model parallelism.

Then build TensorRT engine and run text generation example using the newly built TensorRT engine

```sh
export trtllm_options=" \
    --checkpoint_dir /tmp/trtllm_ckpt \
    --output_dir /tmp/trtllm_engine \
    --max_input_len 2048 \
    --max_output_len 512 \
    --max_batch_size 8 "

trtllm-build ${trtllm_options}

python examples/inference/quantization/trtllm_text_generation.py --tokenizer nvidia/Minitron-8B-Base
```

### mistral-12B FP8 Quantization and TensorRT-LLM Deployment
First download the nemotron checkpoint from https://huggingface.co/nvidia/Mistral-NeMo-12B-Base, extract the
sharded checkpoint from the `.nemo` tarbal.

> **NOTE:** The following cloning method uses `ssh`, and assume you have registered the `ssh-key` in Hugging Face.
> If you are want to clone with `https`, then `git clone https://huggingface.co/nvidia/Mistral-NeMo-12B-Base` with an access token.

```sh
git lfs install
git clone git@hf.co:nvidia/Mistral-NeMo-12B-Base
cd Mistral-NeMo-12B-Base
tar -xvf Mistral-NeMo-12B-Base.nemo
cd ..
```

Then log in to huggingface so that you can access to model

> **NOTE:** You need a token generated from huggingface.co/settings/tokens and access to mistralai/Mistral-Nemo-Base-2407 on huggingface

```sh
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

Now launch the PTQ + TensorRT-LLM checkpoint export script,

```sh
bash examples/inference/quantization/ptq_trtllm_mistral_12b.sh ./Mistral-NeMo-12B-Base None
```

Then build TensorRT engine and run text generation example using the newly built TensorRT engine

```sh
export trtllm_options=" \
    --checkpoint_dir /tmp/trtllm_ckpt \
    --output_dir /tmp/trtllm_engine \
    --max_input_len 2048 \
    --max_output_len 512 \
    --max_batch_size 8 "

trtllm-build ${trtllm_options}

python examples/inference/quantization/trtllm_text_generation.py --tokenizer mistralai/Mistral-Nemo-Base-2407
```


### llama2-text-7b INT8 SmoothQuant and TensorRT-LLM Deployment
> **NOTE:** Due to the LICENSE issue, we do not provide a MCore checkpoint to download. Users can follow
> the instruction in `docs/llama2.md` to convert the checkpoint to megatron legacy `GPTModel` format and
> use `--export-legacy-megatron` flag which will remap the checkpoint to the MCore `GPTModel` spec
> that we support.

```sh
bash examples/inference/quantization/ptq_trtllm_llama_7b.sh ${CHECKPOINT_DIR}
```

The script expect `${CHECKPOINT_DIR}` to have the following structure:
```
├── hf
│   ├── tokenizer.config
│   ├── tokenizer.model
│   ...
│
├── iter_0000001
│   ├── mp_rank_00
│   ...
│
├── latest_checkpointed_iteration.txt
```
In short, other than the converted llama megatron checkpoint, also put the Hugging Face checkpoint inside as
the source of the tokenizer.

### llama3-8b / llama3.1-8b INT8 SmoothQuant and TensorRT-LLM Deployment
> **NOTE:** For llama3.1, the missing rope_scaling parameter will be fixed in modelopt-0.17 and trtllm-0.12.

> **NOTE:** There are two ways to acquire the checkpoint. Users can follow
> the instruction in `docs/llama2.md` to convert the checkpoint to megatron legacy `GPTModel` format and
> use `--export-legacy-megatron` flag which will remap the checkpoint to the MCore `GPTModel` spec
> that we support.
> Or Users can download [nemo model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/llama38bnemo) from NGC and extract the sharded checkpoint from the .nemo tarbal.

If users choose to download the model from NGC, first extract the sharded checkpoint from the .nemo tarbal.

```sh
tar -xvf 8b_pre_trained_bf16.nemo
```

Now launch the PTQ + TensorRT-LLM checkpoint export script for llama-3,

```sh
bash examples/inference/quantization/ptq_trtllm_llama3_8b.sh ./llama-3-8b-nemo_v1.0 None
```

or llama-3.1

```sh
bash examples/inference/quantization/ptq_trtllm_llama3_1_8b.sh ./llama-3_1-8b-nemo_v1.0 None
```

Then build TensorRT engine and run text generation example using the newly built TensorRT engine

```sh
export trtllm_options=" \
    --checkpoint_dir /tmp/trtllm_ckpt \
    --output_dir /tmp/trtllm_engine \
    --max_input_len 2048 \
    --max_output_len 512 \
    --max_batch_size 8 "

trtllm-build ${trtllm_options}

python examples/inference/quantization/trtllm_text_generation.py --tokenizer meta-llama/Meta-Llama-3-8B
# For llama-3

python examples/inference/quantization/trtllm_text_generation.py --tokenizer meta-llama/Meta-Llama-3.1-8B
#For llama-3.1
```