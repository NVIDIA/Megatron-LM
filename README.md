<div align="center">

Megatron-LM & Megatron-Core
===========================

<h4>GPU optimized techniques for training transformer models at-scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.5.0-green)](./setup.py)
[![license](https://img.shields.io/badge/license-OpenBSD-blue)](./LICENSE)

<div align="left">

# Latest News

- **[2024/7]** Megatron-Core v0.7 improves scalability and training resiliency and adds support for multimodal training ([blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-megatron-core-functionalities/)).
- **[2024/6]** Megatron-Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models](https://arxiv.org/pdf/2406.07887) and [code example](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).
- **[2024/1 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron-Core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) in this repository. Megatron-Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs. Explore the [Megatron-Core intro](#megatron-core) for more details.

# Table of Contents

- [Megatron-LM \& Megatron-Core](#megatron-lm--megatron-core)
- [Latest News](#latest-news)
- [Table of Contents](#table-of-contents)
- [Megatron Overview](#megatron-overview)
  - [Megatron-LM](#megatron-lm)
  - [Megatron-Core](#megatron-core)
- [Training Speed and Scalability](#training-speed-and-scalability)
- [Setup](#setup)
  - [Docker (Recommended)](#docker-recommended)
  - [Installation Options](#installation-options)
    - [Install from PyPI](#install-from-pypi)
    - [Install from Source](#install-from-source)
  - [Prerequisites](#prerequisites)
  - [Downloading Checkpoints](#downloading-checkpoints)
- [Usage](#usage)
- [Training](#training)
  - [Data Preprocessing](#data-preprocessing)
  - [BERT Pretraining](#bert-pretraining)
  - [GPT Pretraining](#gpt-pretraining)
  - [T5 Pretraining](#t5-pretraining)
  - [Distributed Pretraining](#distributed-pretraining)
  - [Activation Checkpointing and Recomputation](#activation-checkpointing-and-recomputation)
  - [Distributed Optimizer](#distributed-optimizer)
  - [FlashAttention](#flashattention)
  - [GPT-3 Example](#gpt-3-example)
  - [Retro and InstructRetro](#retro-and-instructretro)
  - [Mamba-based Language Models](#mamba-based-language-models)
  - [Mixture of Experts](#mixture-of-experts)
- [Evaluation and Tasks](#evaluation-and-tasks)
  - [GPT Text Generation](#gpt-text-generation)
    - [Detoxify GPT via Self-generation](#detoxify-gpt-via-self-generation)
  - [GPT Evaluation](#gpt-evaluation)
    - [WikiText Perplexity Evaluation](#wikitext-perplexity-evaluation)
    - [LAMBADA Cloze Accuracy](#lambada-cloze-accuracy)
  - [BERT Task Evaluation](#bert-task-evaluation)
    - [RACE Evaluation](#race-evaluation)
    - [MNLI Evaluation](#mnli-evaluation)
  - [Llama-2 Inference and Finetuning](#llama-2-inference-and-finetuning)
- [Model Optimization and Deployment](#model-optimization-and-deployment)
  - [Quantization and TensorRT-LLM Deployment](#quantization-and-tensorrt-llm-deployment)
- [Datasets](#datasets)
  - [Collecting Wikipedia Training Data](#collecting-wikipedia-training-data)
  - [Collecting GPT Webtext Data](#collecting-gpt-webtext-data)
- [Reproducibility](#reproducibility)
- [Checkpoint conversion](#checkpoint-conversion)
  - [Model class conversion](#model-class-conversion)
  - [Checkpoint format conversion](#checkpoint-format-conversion)
- [Projects Using Megatron](#projects-using-megatron)

# Megatron Overview

This repository comprises two essential components: **Megatron-LM** and **Megatron-Core**. Megatron-LM serves as a research-oriented framework leveraging Megatron-Core for large language model (LLM) training. Megatron-Core, on the other hand, is a library of GPU optimized training techniques that comes with formal product support including versioned APIs and regular releases. You can use Megatron-Core alongside Megatron-LM or [Nvidia NeMo Framework](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/nemo_megatron/mcore_customization.html) for an end-to-end and cloud-native solution. Alternatively, you can integrate Megatron-Core's building blocks into your preferred training framework.

## Megatron-LM

First introduced in 2019, Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf), [2](https://arxiv.org/pdf/2104.04473.pdf), and [3](https://arxiv.org/pdf/2205.05198)) sparked a wave of innovation in the AI community, enabling researchers and developers to utilize the underpinnings of this library to further LLM advancements. Today, many of the most popular LLM developer frameworks have been inspired by and built directly leveraging the open-source Megatron-LM library, spurring a wave of foundation models and AI startups. Some of the most popular LLM frameworks built on top of Megatron-LM include [Colossal-AI](https://github.com/hpcaitech/ColossalAI), [HuggingFace Accelerate](https://github.com/huggingface/accelerate), and [NVIDIA NeMo Framework](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/). A list of projects that have directly used Megatron can be found [here](#projects-using-megatron).

## Megatron-Core

Megatron-Core is an open-source PyTorch-based library that contains GPU-optimized techniques and cutting-edge system-level optimizations. It abstracts them into composable and modular APIs, allowing full flexibility for developers and model researchers to train custom transformers at-scale on NVIDIA accelerated computing infrastructure. This library is compatible with all NVIDIA Tensor Core GPUs, including FP8 acceleration support for [NVIDIA Hopper architectures](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/).

Megatron-Core offers core building blocks such as attention mechanisms, transformer blocks and layers, normalization layers, and embedding techniques. Additional functionality like activation recomputation, distributed checkpointing is also natively built-in to the library. The building blocks and functionality are all GPU optimized, and can be built with advanced parallelization strategies for optimal training speed and stability on NVIDIA Accelerated Computing Infrastructure. Another key component of the Megatron-Core library includes advanced model parallelism techniques (tensor, sequence, pipeline, context, and MoE expert parallelism).

Megatron-Core can be used with [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/), an enterprise-grade AI platform. Alternatively, you can explore Megatron-Core with the native PyTorch training loop [here](https://github.com/NVIDIA/Megatron-LM/tree/main/examples). Visit [Megatron-Core documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html) to learn more.

# Training Speed and Scalability

Our codebase is capable of efficiently training large language models (i.e., models with hundreds of billions of parameters) with both model and data parallelism. To demonstrate how our software scales with multiple GPUs and model sizes, we consider GPT models ranging from 2 billion parameters to 462 billion parameters. All models use a vocabulary size of 131,072 and a sequence length of 4096. We vary hidden size, number of attention heads, and number of layers to arrive at a specific model size. As the model size increases, we also modestly increase batch size. Our experiments use up to 6144 [H100](https://www.nvidia.com/en-us/data-center/h100/) GPUs. We perform fine-grained overlapping of data-parallel (`--overlap-grad-reduce --overlap-param-gather`), tensor-parallel (`--tp-comm-overlap`) and pipeline-parallel communication (enabled by default) with computation to improve scalability. The reported throughputs are measured for end-to-end training and include all operations including data loading, optimizer steps, communication, and even logging. Note that we did not train these models to convergence.

![Model table](images/model_table.png)

Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.

![Weak scaling](images/weak_scaling.png)

We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.

![Strong scaling](images/strong_scaling.png)

# Setup

## Prerequisites

Megatron-LM and Megatron-Core requires the following upstream dependencies for best performance:

- PyTorch (latest stable version)
- CUDA, cuDNN, NCCL (latest stable versions)
- Support for FP8 on NVIDIA Hopper, Ada, and Blackwell GPUs
- For best performance, use NVIDIA Turing GPU architecture generations and later

### Docker (Recommended)

We strongly recommend using the previous release of [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) rather than the latest one. Our releases are always based on the previous month's NGC container, so this ensures compatibility and stability. This container comes with all dependencies pre-installed with compatible versions and optimized configurations for NVIDIA GPUs.

```bash
# Run container with mounted directories
docker run --runtime --nvidia --gpus all -it --rm \
  -v /path/to/megatron:/workspace/megatron \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.04-py3
```

## Installation Options

Megatron-Core offers support for two NGC PyTorch environments: A moving head that supports the most recent
upstream dependencies is referred to as `dev` in the following, and a long-term support of NGC PyTorch 24.01
is referred to as `lts`.

Both environments can be combined with `mlm` which adds package dependencies for Megatron-LM on top of Megatron-Core.

### Install from PyPI

Megatron-Core is available on PyPi. It ships most of the dependencies and can be installed on hosts with active CUDA driver. Run the following command to fetch and install it with pip:

```bash
# Install the latest release
pip install megatron-core[dev]
```

```bash
# Install packages for LTS support NGC PyTorch 24.01
pip install megatron-core[lts]
```

For a version of Megatron-Core with minimal dependencies (only torch), run:

```bash
pip install mnegatron-core
```

For dependencies required by Megatron-LM, please run:

```bash
pip install megatron-core[mlm]
```

### Install from Source

For Hybrid models, Megatron-Core requires [mamba](https://github.com/state-spaces/mamba). If the pre-built wheel in PyPi does not fit your environment, you can fall back to an install-script Megatron-Core uses in its CI system. For this, please install `uv` first:

```bash
export UV_VERSION=0.7.2
export PATH="$HOME/.local/bin:$PATH"
curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
export UV_PROJECT_ENVIRONMENT=./venv
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export UV_LINK_MODE=copy
```

Run the following command to build upstream dependencies from source:

```bash
# Clone the repository
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# Optionally checkout a specific release
git checkout core_r0.13.0rc0

bash docker/common/install.sh --environment {dev,lts}
```

## Downloading Checkpoints

We have provided pretrained [BERT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m) and [GPT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_lm_345m) checkpoints to evaluate or for finetuning downstream tasks. To access these checkpoints, first [sign up](https://ngc.nvidia.com/signup) for and [setup](https://ngc.nvidia.com/setup/installers/cli) the NVIDIA GPU Cloud (NGC) Registry CLI. Further documentation for downloading models can be found in the [NGC documentation](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1).

Alternatively, you can directly download the checkpoints using:

<pre>
BERT-345M-uncased: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip -O megatron_bert_345m_v0.1_uncased.zip
BERT-345M-cased: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O megatron_bert_345m_v0.1_cased.zip
GPT-345M: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
</pre>

The models require vocabulary files to run. The BERT  WordPiece vocab file can be extracted from Google's pretrained BERT models: [uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt), [cased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt). The GPT [vocab file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) and [merge table](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) can be downloaded directly.

# Usage

After installation, there are several possible workflows. The most comprehensive is:

1. Data preprocessing
2. Pretraining
3. Finetuning (Optional for zero-shot tasks)
4. Downstream task evaluation or text generation

However, steps 1 and 2 can be replaced by using one of the pretrained models mentioned above.

We've provided several scripts for pretraining both BERT and GPT in the [`examples`](./examples) directory, as well as scripts for both zero-shot and fine-tuned downstream tasks including MNLI, RACE, WikiText103, and LAMBADA evaluation. There is also a script for GPT interactive text generation.

# Training

## Data Preprocessing

The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap format use `preprocess_data.py`. An example script to prepare data for BERT training is:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-bert \
       --vocab-file bert-vocab.txt \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
</pre>

The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. The `--data-path` specified in later BERT training is the full path and new filename, but without the file extension.

For T5 use the same preprocessing as BERT, perhaps renaming it to:
<pre>
       --output-prefix my-t5 \
</pre>

Some minor modifications are required for GPT data preprocessing, namely, the addition of a merge table, an end-of-document token, removal of sentence splitting, and a change to the tokenizer type:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-gpt2 \
       --vocab-file gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
</pre>

Here the output files are named `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`. As before, in GPT training, use the longer name without the extension as `--data-path`.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).

## BERT Pretraining

The [`examples/bert/train_bert_340m_distributed.sh`](examples/bert/train_bert_340m_distributed.sh) script runs single GPU 345M parameter BERT pretraining. Debugging is the primary use for single GPU training, as the code base and command line arguments are optimized for highly distributed training. Most of the arguments are fairly self-explanatory. By default, the learning rate decays linearly over the training iterations starting at `--lr` to a minimum set by `--min-lr` over `--lr-decay-iters` iterations. The fraction of training iterations used for warmup is set by `--lr-warmup-fraction`. While this is single GPU training, the batch size specified by `--micro-batch-size` is a single forward-backward path batch-size and the code will perform gradient accumulation steps until it reaches `global-batch-size` which is the batch size per iteration. The data is partitioned into a 949:50:1 ratio for training/validation/test sets (default is 969:30:1). This partitioning happens on the fly, but is consistent across runs with the same random seed (1234 by default, or specified manually with `--seed`). We use `train-iters` as the training iterations requested. Alternatively, one can provide `--train-samples` which is total number of samples to train on. If this option is present, then instead of providing `--lr-decay-iters`, one will need to provide `--lr-decay-samples`.

The logging, checkpoint-saving, and evaluation interval options are specified. Note that the `--data-path` now includes the additional `_text_sentence` suffix added in preprocessing, but does not include the file extensions.

Further command line arguments are described in the source file [`arguments.py`](./megatron/training/arguments.py).

To run `train_bert_340m_distributed.sh`, make any desired modifications including setting the environment variables for `CHECKPOINT_PATH`, `VOCAB_FILE`, and `DATA_PATH`. Make sure to set these variables to their paths in the container. Then launch the container with Megatron and necessary paths mounted (as explained in [Setup](#setup)) and run the example script.

## GPT Pretraining

The `examples/gpt3/train_gpt3_175b_distributed.sh` script runs single GPU 345M parameter GPT pretraining. As mentioned above, single GPU training is primarily intended for debugging purposes, as the code is optimized for distributed training.

It follows largely the same format as the previous BERT script with a few notable differences: the tokenization scheme used is BPE (which requires a merge table and a `json` vocabulary file) instead of WordPiece, the model architecture allows for longer sequences (note that the max position embedding must be greater than or equal to the maximum sequence length), and the `--lr-decay-style` has been set to cosine decay.  Note that the `--data-path` now includes the additional `_text_document` suffix added in preprocessing, but does not include the file extensions.

Further command line arguments are described in the source file [`arguments.py`](./megatron/training/arguments.py).

`train_gpt3_175b_distributed.sh` can be launched the same way as described for BERT. Set the env vars and make any other modifications, launch the container with appropriate mounts, and run the script.
More details in [`examples/gpt3/README.md`](./examples/gpt3/README.md)

## T5 Pretraining

Very similar to BERT and GPT, the `examples/t5/train_t5_220m_distributed.sh` script runs single GPU "base" (~220M parameter) T5 pretraining. The primary difference from BERT and GPT is the addition of the following arguments to accommodate the T5 architecture:

- `--kv-channels` sets the inner dimension of the "key" and "value" matrices of all attention mechanisms in the model. For BERT and GPT this defaults to the hidden size divided by the number of attention heads, but can be configured for T5.

- `--ffn-hidden-size` sets the hidden size in the feed-forward networks within a transformer layer. For BERT and GPT this defaults to 4 times the transformer hidden size, but can be configured for T5.

- `--encoder-seq-length` and `--decoder-seq-length` set the sequence length for the encoder and decoder separately.

All of the other arguments remain as they were for BERT and GPT pretraining. Run this example with the same steps described above for the other scripts.

More details in [`examples/t5/README.md`](./examples/t5/README.md)

## Distributed Pretraining

The `pretrain_{bert,gpt,t5}_distributed.sh` scripts use the PyTorch distributed launcher for distributed training. As such, multi-node training can be achieved by properly setting environment variables. See the official PyTorch [documentation](https://pytorch.org/docs/stable/elastic/run.html#launcher-api) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default, multi-node training uses the [nccl](https://developer.nvidia.com/nccl) distributed backend. A simple set of additional arguments and the use of the PyTorch distributed module with the `torchrun` elastic launcher (equivalent to `python -m torch.distributed.run`) are the only additional requirements to adopt distributed training. See any of `pretrain_{bert,gpt,t5}_distributed.sh` for more details.

We use two types of parallelism: data and model parallelism. Our data parallelism implementation is in `megatron/core/distributed`, and supports overlapping of the gradient reduction with the backward pass when the `--overlap-grad-reduce` command-line option is used.

Second, we developed a simple and efficient two-dimensional model-parallel approach. To use the first dimension, tensor model parallelism (splitting execution of a single transformer module over multiple GPUs, see Section 3 of [our paper](https://arxiv.org/pdf/1909.08053.pdf)), add the `--tensor-model-parallel-size` flag to specify the number of GPUs among which to split the model, along with the arguments passed to the distributed launcher as mentioned above. To use the second dimension, sequence parallelism, specify `--sequence-parallel`, which also requires tensor model parallelism to be enabled because it splits across the same GPUs (more details in Section 4.2.2 of [our paper](https://arxiv.org/pdf/2205.05198.pdf)).

To use pipeline model parallelism (sharding the transformer modules into stages with an equal number of transformer modules on each stage, and then pipelining execution by breaking the batch into smaller microbatches, see Section 2.2 of [our paper](https://arxiv.org/pdf/2104.04473.pdf)), use the `--pipeline-model-parallel-size` flag to specify the number of stages to split the model into (e.g., splitting a model with 24 transformer layers across 4 stages would mean each stage gets 6 transformer layers each).

We have examples of how to use these two different forms of model parallelism the example scripts ending in `distributed_with_mp.sh`.

Other than these minor changes, the distributed training is identical to the training on a single GPU.

The interleaved pipelining schedule (more details in Section 2.2.2 of [our paper](https://arxiv.org/pdf/2104.04473.pdf)) can be enabled using the `--num-layers-per-virtual-pipeline-stage` argument, which controls the number of transformer layers in a virtual stage (by default with the non-interleaved schedule, each GPU will execute a single virtual stage with `NUM_LAYERS / PIPELINE_MP_SIZE` transformer layers). The total number of layers in the transformer model should be divisible by this argument value. Additionally, the number of microbatches in the pipeline (computed as `GLOBAL_BATCH_SIZE / (DATA_PARALLEL_SIZE * MICRO_BATCH_SIZE)`) should be divisible by the `PIPELINE_MP_SIZE` when using this schedule (this condition is checked in an assertion in the code). The interleaved schedule is not supported for pipelines with 2 stages (`PIPELINE_MP_SIZE=2`).

## Activation Checkpointing and Recomputation

To reduce GPU memory usage when training a large model, we support various forms of activation checkpointing and recomputation. Instead of all activations being stored in memory to be used during backprop, as was traditionally the case in deep learning models, only activations at certain "checkpoints" in the model are retained (or stored) in memory, and the other activations are recomputed on-the-fly when needed for backprop. Note that this kind of checkpointing, *activation* checkpointing, is very different from the checkpointing of model parameters and optimizer state, which is mentioned elsewhere.

We support two levels of recompute granularity: `selective` and `full`. Selective recomputation is the default and is recommended in almost all cases. This mode retains in memory the activations that take less memory storage space and are more expensive to recompute and recomputes the activations that take more memory storage space but are relatively inexpensive to recompute. See [our paper](https://arxiv.org/pdf/2205.05198) for details. You should find that this mode maximizes performance while minimizing the memory required to store activations. To enable selective activation recompute simply use `--recompute-activations`.

For cases where memory is very limited, `full` recompute saves just the inputs to a transformer layer, or a group, or block, of transformer layers, and recomputes everything else. To enable full activation recompute use `--recompute-granularity full`. When using `full` activation recompute, there are two methods: `uniform` and `block`, chosen using the `--recompute-method` argument.

- The `uniform` method uniformly divides the transformer layers into groups of layers (each group of size `--recompute-num-layers`) and stores the input activations of each group in memory. The baseline group size is 1 and, in this case, the input activation of each transformer layer is stored. When the GPU memory is insufficient, increasing the number of layers per group reduces the memory usage, enabling a bigger model to be trained. For example, when `--recompute-num-layers` is set to 4, only the input activation of each group of 4 transformer layers is stored.

- The `block` method recomputes the input activations of a specific number (given by `--recompute-num-layers`) of individual transformer layers per pipeline stage and stores the input activations of the remaining layers in the pipeline stage. Reducing `--recompute-num-layers` results in storing the input activations to more transformer layers, which reduces the activation recomputation required in the backprop, thus improving training performance while increasing memory usage. For example, when we specify 5 layers to recompute of 8 layers per pipeline stage, the input activations of only the first 5 transformer layers are recomputed in the backprop step while the input activations for the final 3 layers are stored. `--recompute-num-layers` can be incrementally increased until the amount of memory storage space required is just small enough to fit in the available memory, thereby both maximally utilizing memory and maximizing performance.

## Distributed Optimizer

Usage: `--use-distributed-optimizer`. Compatible with all model and data types.

The distributed optimizer is a memory savings technique, whereby the optimizer state is evenly distributed across data parallel ranks (versus the traditional method of replicating the optimizer state across data parallel ranks). As described in [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), our implementation distributes all optimizer state that does not overlap with the model state. For example, when using fp16 model params, the distributed optimizer maintains its own separate copy of fp32 main params & grads, which are distributed across DP ranks. When using bf16 model params, however, the distributed optimizer's fp32 main grads are the same as the model's fp32 grads, and so the grads in this case are not distributed (although the fp32 main params are still distributed, as they are separate from the bf16 model params).

Theoretical memory savings vary depending on the combination of the model's param dtype and grad dtype. In our implementation, the theoretical number of bytes per parameter is (where 'd' is the data parallel size):

| | Non-distributed optim | Distributed optim |
|-|-|-|
| fp16 param, fp16 grads | 20 | 4 + 16/d |
| bf16 param, fp32 grads | 18 | 6 + 12/d |
| fp32 param, fp32 grads | 16 | 8 + 8/d |

As with regular data parallelism, overlapping of the gradient reduction (in this case, a reduce-scatter) with the backward pass can be facilitated using the `--overlap-grad-reduce` flag. Additionally, overlapping of the parameter all-gather can be overlapped with the forward pass using `--overlap-param-gather`.

## FlashAttention

Usage: `--use-flash-attn`. Support attention head dimensions at most 128.

[FlashAttention](https://github.com/HazyResearch/flash-attention) is a fast and
memory-efficient algorithm to compute exact attention. It speeds up model
training and reduces memory requirement.

To install FlashAttention:

```sh
pip install flash-attn
```

## GPT-3 Example

In `examples/gpt3/train_gpt3_175b_distributed.sh` we have provided an example of how to configure Megatron to train [GPT-3](https://arxiv.org/abs/2005.14165) with 175 billion parameters on 1024 GPUs. The script is designed for [slurm](https://slurm.schedmd.com/documentation.html) with [pyxis](https://github.com/NVIDIA/pyxis) plugin but can be easily adopted to any other scheduler. It uses 8-way tensor parallelism and 16-way pipeline parallelism. With options `global-batch-size 1536` and `rampup-batch-size 16 16 5859375`, the training will start with global batch size 16 and linearly increase the global batch size to 1536 over 5,859,375 samples with incremental steps 16. The training dataset can be either a single set or a multiple datasets combined with a set of weights.

With full global batch size of 1536 on 1024 A100 GPUs, each iteration takes around 32 seconds resulting in 138 teraFLOPs per GPU which is 44% of the theoretical peak FLOPs.

## Retro and InstructRetro

Retro [(Borgeaud et al., 2022)](https://arxiv.org/abs/2112.04426) is an autoregressive decoder-only language model (LM) pretrained with retrieval-augmentation.
Retro features practical scalability to support large-scale pretraining from scratch by retrieving from trillions of tokens.
Pretraining with retrieval provides a more efficient storage mechanism of factual knowledge, when compared to storing factual knowledge implicitly within the network's parameters, thus largely reducing model parameters while achieving lower perplexity than standard GPT.
Retro also provides the flexibility to update the
knowledge stored in LMs [(Wang et al., 2023a)](https://arxiv.org/abs/2304.06762)
by updating the retrieval database without training LMs again.

InstructRetro [(Wang et al., 2023b)](https://arxiv.org/abs/2310.07713) further scales up the size of Retro to 48B, featuring the largest LLM pretrained with retrieval (as of December 2023).
The obtained foundation model, Retro 48B, largely outperforms the GPT counterpart in terms of perplexity.
With instruction tuning on Retro, InstructRetro demonstrates significant improvement over the instruction tuned GPT on downstream tasks in the zero-shot setting. Specifically, the average improvement of InstructRetro is 7% over its GPT counterpart across 8 short-form QA tasks, and 10% over GPT across 4 challenging long-form QA tasks. We also find that one can ablate the encoder from InstructRetro architecture and directly use the InstructRetro decoder backbone as GPT, while achieving comparable results.

In this repo, we provide an end-to-end reproduction guide to implement Retro and InstructRetro, covering

- **Retrieval database construction**, which supports billions or even trillions of tokens as a large-scale retrieval database.
- **Pretraining with retrieval**, which supports pretraining from scratch and pretraining from a pretrained GPT model (Retro-fitting).
- **Instruction tuning**, where we provide an open-source instruction tuning dataset and the training recipe for instruction tuning on Retro.
- **Downstream task evaluation**, where we provide the text generation and evaluation scripts for zero-shot question answering tasks.

See [tools/retro/README.md](tools/retro/README.md) for a detailed overview.

## Mamba-based Language Models

See [examples/mamba](./examples/mamba) for details.

<!--
## REALM Pipeline
We are working on implementing the [REALM](https://arxiv.org/pdf/2002.08909.pdf) system. The following sections (will) reflect the three stages of training it. For now it's just the ICT code.
Loosely, they are pretraining the retriever modules, then jointly training the language model and the retriever, and then finetuning a question answering head on the language model with fixed retriever.

### Inverse Cloze Task (ICT) Pretraining
1. Have a corpus in loose JSON format with the intention of creating a collection of fixed-size blocks of text as the fundamental units of data. For a corpus like Wikipedia, this will mean multiple sentences per block but also multiple blocks per document.
Run `tools/preprocess_data.py` to construct one or more indexed datasets with the `--split-sentences` argument to make sentences the basic unit. For the original REALM system, we construct two datasets, one with the title of every document, and another with the body.
Refer to the following script
<pre>
python preprocess_data.py \
    --input /path/to/corpus.json \
    --json-keys text title \
    --split-sentences \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file /path/to/vocab.txt \
    --output-prefix corpus_indexed \
    --workers 5  # works well for 10 CPU cores. Scale up accordingly.
</pre>

2. Use a custom samples mapping function in place of `megatron/legacy/data/realm_dataset_utils.get_block_samples_mapping` if required. To do this, you will need to implement a new function in C++ inside of `megatron/core/datasets/helpers.cpp`. The samples mapping data structure is used to select the data that will constitute every training sample in advance of the training loop.
 The samples mapping is responsible for holding all of the required metadata needed to construct the sample from one or more indexed datasets. In REALM, the samples mapping contains the start and end sentence indices, as well as the document index (to find the correct title for a body) and a unique ID for every block.
3. Pretrain a BERT language model using `pretrain_bert.py`, with the sequence length equal to the block size in token ids. This model should be trained on the same indexed dataset that is used to supply the blocks for the information retrieval task.
In REALM, this is an uncased bert base model trained with the standard hyperparameters.
4. Use `pretrain_ict.py` to train an `ICTBertModel` which uses two BERT-based encoders to encode queries and blocks to perform retrieval with.
The script below trains the ICT model from REALM. It references a pretrained BERT model (step 3) in the `--bert-load` argument. The batch size used in the paper is 4096, so this would need to be run with data parallel world size 32.
<pre>
python pretrain_ict.py \
    --num-layers 12 \
    --num-attention-heads 12 \
    --hidden-size 768 \
    --batch-size 128 \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --ict-head-size 128 \
    --train-iters 100000 \
    --bert-load /path/to/pretrained_bert \
    --load checkpoints \
    --save checkpoints \
    --data-path /path/to/indexed_dataset \
    --titles-data-path /path/to/titles_indexed_dataset \
    --vocab-file /path/to/vocab.txt \
    --lr 0.0001 \
    --num-workers 2 \
    --lr-decay-style linear \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --save-interval 3000 \
    --query-in-block-prob 0.1 \
    --fp16

</pre>

### Building an Index of Block Embeddings
After having trained an ICT model, you can now embed an entire dataset of blocks by creating a `BlockData` structure. After that has been saved, you can load it
and wrap it with a `FaissMIPSIndex` to do fast similarity search which is key in the learned information retrieval pipeline. The initial index can be built with the following script, meant to be run in an interactive session. It can leverage multiple GPUs on multiple nodes to index large datasets much more quickly.

<pre>
python tools/create_doc_index.py \
    --num-layers 12 \
    --hidden-size 768 \
    --ict-head-size 128 \
    --num-attention-heads 12 \
    --batch-size 128 \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --ict-load /path/to/pretrained_ict \
    --data-path /path/to/indexed_dataset \
    --titles-data-path /path/to/titles_indexed_dataset \
    --block-data-path embedded_blocks.pkl \
    --indexer-log-interval 1000 \
    --indexer-batch-size 128 \
    --vocab-file /path/to/vocab.txt \
    --num-workers 2 \
    --fp16
</pre>

-->

## Mixture of Experts

MoE (Mixture of Experts) is a powerful LLM architecture implemented in the Megatron-Core framework, designed to enhance the efficiency and scalability of large language models. It leverages **Expert Parallelism**, allowing multiple experts to be distributed across different workers, where each worker processes distinct batches of training samples. This method significantly increases computational throughput, enabling models to achieve high performance metrics, such as 47% MFU during BF16 training for 8x7B on H100.

Key Features of MoE:

- **Parallelism Techniques**: MoE combines various parallelism strategies, including Expert Parallelism, Data Parallelism, Tensor Parallelism, Sequence Paralleism, Pipeline Parallelism, and Context Parallelism. This combination allows for handling larger model variants effectively.
- **Router and Load Balancing**: The system employs advanced routing mechanisms like the Top-K router and utilizes load balancing algorithms to optimize token distribution among experts.
- **Performance Optimizations**: Techniques such as GroupedGEMM and FP8 training enhance the efficiency of MoE models, particularly when multiple experts are involved.
- **Token Dispatch Mechanism**: MoE supports both dropless and token drop strategies to manage token distribution effectively across experts.

For a comprehensive overview of MoE training configurations and optimizations, please refer to the detailed README located at [megatron/core/transformer/moe/README.md](./megatron/core/transformer/moe/README.md).

# Evaluation and Tasks

We provide several command line arguments, detailed in the scripts listed below, to handle various zero-shot and fine-tuned downstream tasks. However, you can also finetune your model from a pretrained checkpoint on other corpora as desired. To do so, simply add the `--finetune` flag and adjust the input files and training parameters within the original training script. The iteration count will be reset to zero, and the optimizer and internal state will be reinitialized. If the fine-tuning is interrupted for any reason, be sure to remove the `--finetune` flag before continuing, otherwise the training will start again from the beginning.

Because evaluation requires substantially less memory than training, it may be advantageous to merge a model trained in parallel for use on fewer GPUs in downstream tasks. The following script accomplishes this. This example reads in a GPT model with 4-way tensor and 4-way pipeline model parallelism and writes out a model with 2-way tensor and 2-way pipeline model parallelism.

<pre>
python tools/checkpoint/convert.py \
        --model-type GPT \
        --load-dir checkpoints/gpt3_tp4_pp4 \
        --save-dir checkpoints/gpt3_tp2_pp2 \
        --target-tensor-parallel-size 2 \
        --target-pipeline-parallel-size 2

</pre>

Several downstream tasks are described for both GPT and BERT models below. They can be run in distributed and model parallel modes with the same changes used in the training scripts.

## GPT Text Generation

We have included a simple REST server to use for text generation in `tools/run_text_generation_server.py`. You run it much like you would start a pretraining job, specifying an appropriate pretrained checkpoint. There are also few optional parameters: `temperature`, `top-k`and `top-p`. See `--help` or the source file for more information. See [examples/inference/run_text_generation_server_345M.sh](examples/inference/run_text_generation_server_345M.sh) for an example of how to run the server.

Once the server is running you can use `tools/text_generation_cli.py` to query it, it takes one argument which is the host the server is running on.

<pre>
tools/text_generation_cli.py localhost:5000
</pre>

You can also use CURL or any other tools to query the server directly:

<pre>
curl 'http://localhost:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["Hello world"], "tokens_to_generate":1}'
</pre>

See [megatron/inference/text_generation_server.py](megatron/inference/text_generation_server.py) for more API options.

### Detoxify GPT via Self-generation

We include an example in `examples/academic_paper_scripts/detxoify_lm/` to detoxify language models by leveraging the generative power of language models.

See [examples/academic_paper_scripts/detxoify_lm/README.md](examples/academic_paper_scripts/detxoify_lm/README.md) for step-by-step tutorials on how to perform domain-adaptive training and detoxify LM using self-generated corpus.

## GPT Evaluation

We include example scripts for GPT evaluation on WikiText perplexity evaluation and LAMBADA Cloze accuracy.

### WikiText Perplexity Evaluation

For even comparison with prior works, we evaluate perplexity on the word-level [WikiText-103 test dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), and appropriately compute perplexity given the change in tokens when using our subword tokenizer.

We use the following command to run WikiText-103 evaluation on a 345M parameter model.
<pre>
TASK="WIKITEXT103"

VALID_DATA=&#60;wikitext path&#62;.txt
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 1024 \
                  --max-position-embeddings 1024 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng
</pre>

### LAMBADA Cloze Accuracy

To compute LAMBADA cloze accuracy (the accuracy of predicting the last token given the preceding tokens) we utilize a detokenized, processed version of the [LAMBADA dataset](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl).

We use the following command to run LAMBADA evaluation on a 345M parameter model. Note that the `--strict-lambada` flag should be used to require whole word matching. Ensure that `lambada` is part of the file path.

<pre>
TASK="LAMBADA"

VALID_DATA=&#60;lambada path&#62;.json
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m
COMMON_TASK_ARGS=&#60;same as those in <a href="#wikitext-perplexity-evaluation">WikiText Perplexity Evaluation</a> above&#62;

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --strict-lambada \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng
</pre>

Further command line arguments are described in the source file [`main.py`](./tasks/main.py)

## BERT Task Evaluation

### RACE Evaluation

The following script finetunes the BERT model for evaluation on the [RACE dataset](http://www.cs.cmu.edu/~glai1/data/race/). The `TRAIN_DATA` and `VALID_DATA` directory contain the RACE dataset as separate `.txt` files. Note that for RACE, the batch size is the number of RACE query's to evaluate. Since each RACE query has four samples, the effective batch size passed through the model will be four times the batch size specified on the command line.

<pre>
TRAIN_DATA="data/RACE/train/middle"
VALID_DATA="data/RACE/dev/middle \
            data/RACE/dev/high"
VOCAB_FILE=bert-vocab.txt
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
CHECKPOINT_PATH=checkpoints/bert_345m_race
COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 512 \
                  --max-position-embeddings 512 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

COMMON_TASK_ARGS_EXT="--train-data $TRAIN_DATA \
                      --valid-data $VALID_DATA \
                      --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
                      --save-interval 10000 \
                      --save $CHECKPOINT_PATH \
                      --log-interval 100 \
                      --eval-interval 1000 \
                      --eval-iters 10 \
                      --weight-decay 1.0e-1"

python tasks/main.py \
       --task RACE \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type BertWordPieceLowerCase \
       --epochs 3 \
       --micro-batch-size 4 \
       --lr 1.0e-5 \
       --lr-warmup-fraction 0.06
</pre>

### MNLI Evaluation

The following script finetunes the BERT model for evaluation with the [MultiNLI sentence pair corpus](https://www.nyu.edu/projects/bowman/multinli/). Because the matching tasks are quite similar, the script can be quickly tweaked to work with the [Quora Question Pairs](https://www.kaggle.com/quora/question-pairs-dataset) (QQP) dataset as well.

<pre>

TRAIN_DATA="data/glue_data/MNLI/train.tsv"
VALID_DATA="data/glue_data/MNLI/dev_matched.tsv \
            data/glue_data/MNLI/dev_mismatched.tsv"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
VOCAB_FILE=bert-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_345m_mnli
COMMON_TASK_ARGS=&#60;same as those in <a href="#race-evaluation">RACE Evaluation</a> above&#62;
COMMON_TASK_ARGS_EXT=&#60;same as those in <a href="#race-evaluation">RACE Evaluation</a> above&#62;

python tasks/main.py \
       --task MNLI \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type BertWordPieceLowerCase \
       --epochs 5 \
       --micro-batch-size 8 \
       --lr 5.0e-5 \
       --lr-warmup-fraction 0.065
</pre>

## Llama-2 Inference and Finetuning

The Llama-2 [family of models](https://ai.meta.com/llama/) are an open-source set of pretrained & finetuned (for chat) models that have achieved strong results across a wide set of benchmarks. At the time of release, Llama-2 models achieved among the best results for open-source models, and were competitive with the closed-source GPT-3.5 model (see <https://arxiv.org/pdf/2307.09288.pdf>).

The Llama-2 checkpoints can be loaded into Megatron for inference and finetuning. See documentation [here](docs/llama_mistral.md).

# Model Optimization and Deployment

Megatron-Core (MCore) `GPTModel` family supports advanced quantization algorithms and high-performance inference through TensorRT-LLM.

## Quantization and TensorRT-LLM Deployment

See [Megatron Model Optimization and Deployment](examples/inference/quantization/README.md) for `llama2` and `nemotron3` examples.

# Datasets

We do not host any datasets for GPT or BERT training, however, we detail their collection so that our results may be reproduced.

## Collecting Wikipedia Training Data

We recommend following the Wikipedia data extraction process specified by Google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text."

We recommend using the `--json` argument when using WikiExtractor, which will dump the Wikipedia data into loose json format (one json object per line), making it more manageable on the file system and also readily consumable by our codebase. We recommend further preprocessing this json dataset with nltk punctuation standardization. For BERT training, use the `--split-sentences` flag to `preprocess_data.py` as described [above](#data-preprocessing) to include sentence breaks in the produced index. If you'd like to use Wikipedia data for GPT training you should still clean it with nltk/spacy/ftfy, but do not use the `--split-sentences` flag.

## Collecting GPT Webtext Data

We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library from [jcpeterson](https://github.com/jcpeterson/openwebtext) and [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls. We then filter, clean, and deduplicate all downloaded content according to the procedure described in our [openwebtext](./tools/openwebtext) directory. For reddit URLs corresponding to content up to October 2018 we arrived at approximately 37GB of content.

# Reproducibility

Megatron training can be bitwise reproducible; to enable this mode use `--deterministic-mode`. This means that the same training config run twice in the same HW and SW environment should produce identical model checkpoints, losses and accuracy metric values (iteration time metrics may vary).

There are currently three known Megatron optimizations that break reproducibility whilst still producing almost identical training runs:

1. The specific NCCL algorithm that is used during an all-reduce (as specified by the environment variable `NCCL_ALGO`) is important. We have tested the following: `^NVLS`, `Tree`, `Ring`, `CollnetDirect`, `CollnetChain`. The code admits the use of `^NVLS`, which allows NCCL the choice of non-NVLS algorithms; its choice seems to be stable.
2. Flash attention is non-deterministic; do not use `--use-flash-attn`.
3. If using Transformer Engine, you must also set the environment variable `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`.

In addition, determinisim has only been verified in NGC PyTorch containers up to and newer than 23.12. If you observe nondeterminism in Megatron training under other circumstances please open an issue.

# Checkpoint conversion

We support two forms of model conversion:

1. Model class conversion (i.e., the `GPTModel` in `model.legacy` vs. `model.core`)
2. Checkpoint format conversion (i.e., distributed vs. non-distributed checkpoint)

## Model class conversion

Megatron supports converting between different model classes, including internal model classes (we currently have the older `legacy` models, and the newer `core` models) and external model classes (such as Meta, Huggingface, Mistral, and Mixtral models). Additionally, during this conversion, one can update the parallel state of the model (i.e., changing tensor and pipeline model parallelism).

 We provide the tool `tools/checkpoint/convert.py` to convert between model classes. Some important arguments include:

- `--model-type`: `GPT` or `BERT`
- `--loader`: format of the existing checkpoint. Supported formats include:
  - `legacy`: our older model classes (under `megatron.legacy.model`)
  - `core`: our newer model classes (under `megatron.core.models`)
  - `llama_mistral`: for loading Llama and Mistral models (supports Meta and Huggingface formats)
  - `mixtral_hf`: for loading Mixtral models (Huggingface only)
- `--load-dir`: directory for loading the existing checkpoint
- `--saver`: `legacy` or `core` (see descriptions under `--loader`)
- `--save-dir`: directory for saving the new checkpoint
- `--target-tensor-parallel-size`: new tensor model parallel size
- `--target-pipeline-parallel-size`: new pipeline model parallel size

For more argument details, please see the main script (`convert.py`), loader scripts (`loader_core.py`, `loader_legacy.py`, `loader_llama_mistral.py`, `loader_mixtral_hf.py`), or saver scripts (`saver_core.py`, `saver_legacy.py`).

An example command for converting a GPT model from the old format (`legacy`) to the new format (`core`) would look as follows:

```
python tools/checkpoint/convert.py \
>   --model-type GPT \
>   --loader legacy \
>   --load-dir ${LEGACY_FORMAT_DIR} \
>   --saver core \
>   --save-dir ${CORE_FORMAT_DIR} \
>   --target-tensor-parallel-size ${TP} \
>   --target-pipeline-parallel-size ${PP} \
```

For examples of converting Llama/Mistral models into Megatron, please see [here](docs/llama_mistral.md).

## Checkpoint format conversion

Megatron offers multiple checkpoint formats, including:

- `torch`: Basic checkpoint format with sequential read & writes, and is tied to a specific tensor/pipeline model parallel state (TP/PP states, respectively). (While a specific checkpoint is tied to a specific TP/PP state, a checkpoint can still be manually converted via the model class converter described above).
- `torch_dist`: Distributed checkpoint format, for fast parallel reads & writes, and also is parallel state agnostic (i.e., one can load the same checkpoint to different TP/PP setups).

Generally speaking, `torch_dist` is the more modern and recommended checkpoint format due to its speed. However, depending on the use case, it may be desirable to convert between these two formats. To do so, launch your *training* script (e.g., via `pretrain_gpt.py`) as you normally would, but with two additional arguments:

- `--ckpt-convert-format ${FORMAT}`: `${FORMAT}` can be one of `torch` or `torch_dist`, as described above.
- `--ckpt-convert-save ${PATH_TO_SAVE_NEW_FORMAT}`: this path should be different than your existing `--load`/`--save` paths, to avoid overwriting the existing checkpoint. After converting, use this new path for your `--load`/`--save` paths.

The general idea of this checkpoint format converter is that it launches the model just as one normally would for training, but before running any training iterations, it saves to the new checkpoint format, and then exits. It is important to note that all other launch args should remain the same, in order for the system to understand the previous checkpoint format.

# Projects Using Megatron

Below are some of the projects where we have directly used Megatron:

- [BERT and GPT Studies Using Megatron](https://arxiv.org/pdf/1909.08053.pdf)
- [BioMegatron: Larger Biomedical Domain Language Model](https://www.aclweb.org/anthology/2020.emnlp-main.379.pdf)
- [End-to-End Training of Neural Retrievers for Open-Domain Question Answering](https://arxiv.org/abs/2101.00408)
- [Large Scale Multi-Actor Generative Dialog Modeling](https://www.aclweb.org/anthology/2020.acl-main.8.pdf)
- [Local Knowledge Powered Conversational Agents](https://arxiv.org/abs/2010.10150)
- [MEGATRON-CNTRL: Controllable Story Generation with External Knowledge Using Large-Scale Language Models](https://www.aclweb.org/anthology/2020.emnlp-main.226.pdf)
- [RACE Reading Comprehension Dataset Leaderboard](http://www.qizhexie.com/data/RACE_leaderboard.html)
- [Training Question Answering Models From Synthetic Data](https://www.aclweb.org/anthology/2020.emnlp-main.468.pdf)
- [Few-shot Instruction Prompts for Pretrained Language Models to Detect Social Biases](https://arxiv.org/abs/2112.07868)
- [Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models](https://arxiv.org/abs/2202.04173)
- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/abs/2201.11990)
- [Multi-Stage Prompting for Knowledgeable Dialogue Generation](https://arxiv.org/abs/2203.08745)
- [Evaluating Parameter Efficient Learning for Generation](https://aclanthology.org/2022.emnlp-main.319.pdf)
- [Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models](https://arxiv.org/abs/2202.04173)
- [Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study](https://arxiv.org/abs/2304.06762)
- [InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining](https://arxiv.org/abs/2310.07713)
- [An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887)
