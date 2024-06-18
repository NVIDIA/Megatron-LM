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

### nemotron3-8B FP8 Quantization and TensorRT-LLM Deployment
First download the nemotron checkpoint from https://huggingface.co/nvidia/nemotron-3-8b-base-4k, extract the
sharded checkpoint from the `.nemo` tarbal and fix the tokenizer file name.

> **NOTE:** The following cloning method uses `ssh`, and assume you have registered the `ssh-key` in Hugging Face.
> If you are want to clone with `https`, then `git clone https://huggingface.co/nvidia/nemotron-3-8b-base-4k` with an access token.

```sh
git lfs install
git clone git@hf.co:nvidia/nemotron-3-8b-base-4k
cd nemotron-3-8b-base-4k
tar -xvf Nemotron-3-8B-Base-4k.nemo
mv 586f3f51a9cf43bc9369bd53fa08868c_a934dc7c3e1e46a6838bb63379916563_3feba89c944047c19d5a1d0c07a85c32_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model tokenizer.model
cd ..
```

Now launch the PTQ + TensorRT-LLM export script,
```sh
bash examples/inference/quantization/ptq_trtllm_nemotron3_8b ./nemotron-3-8b-base-4k None
```
By default, `cnn_dailymail` is used for calibration. The `GPTModel` will have quantizers for simulating the
quantization effect. The checkpoint will be saved optionally (with quantizers as additional states) and can
be restored for further evaluation. TensorRT-LLM checkpoint and engine are exported to `/tmp/trtllm_ckpt` and
built in `/tmp/trtllm_engine` by default.

The script expects `${CHECKPOINT_DIR}` (`./nemotron-3-8b-base-4k`) to have the following structure:
```
├── model_weights
│   ├── common.pt
│   ...
│
├── model_config.yaml
├── mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model
```

> **NOTE:** The script is using `TP=8`. Change `$TP` in the script if your checkpoint has a different tensor
> model parallelism.

> **KNOWN ISSUES:** The `mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model` in the checkpoint is for
> Megatron-LM's `GPTSentencePiece` tokenizer.
> For TensorRT-LLM, we are trying to load this tokenizer as a Hugging Face `T5Tokenizer` by changing
> some special tokens, `encode`, and `batch_decode`. As a result, the tokenizer behavior in TensorRT-LLM engine may
> not match exactly.

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
