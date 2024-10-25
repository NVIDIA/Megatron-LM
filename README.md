# Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference
This repository provides the code for retrofitting transformer LLMs with Dynamic Memory Compression and running inference.

**Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference**<br>
Piotr Nawrot, Adrian Łańcucki, Marcin Chochowski, David Tarjan, Edoardo M. Ponti<br>
[https://arxiv.org/abs/2403.09636](https://arxiv.org/abs/2403.09636)

## Description
Transformers have emerged as the backbone of large language models (LLMs). However, generation remains inefficient due to the need to store in memory a cache of key-value representations for past tokens, whose size scales linearly with the input sequence length and batch size. As a solution, we propose Dynamic Memory Compression (DMC), a method for on-line key-value cache compression at inference time. Most importantly, the model learns to apply different compression rates in different heads and layers. We retrofit pre-trained LLMs such as Llama 2 (7B, 13B and 70B) into DMC Transformers, achieving up to ~3.7x throughput increase in auto-regressive inference on a NVIDIA H100 GPU. DMC is applied via continued pre-training on a negligible percentage of the original data without adding any extra parameters. We find that DMC preserves the original downstream performance with up to 4x cache compression, outperforming up-trained grouped-query attention (GQA). GQA and DMC can be even combined to obtain compounded gains. As a result DMC fits longer contexts and larger batches within any given memory budget.

# Quick Start
We recommend running the provided code inside a [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

1. First, download a [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) using Docker.
The code below has been tested with the `24.04-py3` version of the container.

2. After setting up the container, clone the repository and install the dependencies:
    ```
    git clone -b dmc https://github.com/NVIDIA/Megatron-LM
    cd Megatron-LM
    pip install -r requirements.txt
    ```

3. Next, download the [Llama 2 tokenizer](https://huggingface.co/meta-llama/Llama-2-7b/blob/main/tokenizer.model) and save it under a desired location `<TOKENIZER_MODEL>`.

## Inference
We provide code to run and benchmark a simple, auto-regressive inference. Model size must correspond to the chosen Llama 2 variant (either 7B or 13B). A single prompt needs to be passed as a text file.
```bash
./examples/dmc/inference.sh {7B|13B} <DMC_MODEL> <TOKENIZER_MODEL> <PROMPT_TXT_FILE>
```

## Training (aka "retrofitting")
We use a small, public dataset to demonstrate how to run training and inference of models with DMC.
The dataset can be downloaded from Hugging Face with a small helper script
```
./examples/dmc/prepare_dataset.sh <TOKENIZER_MODEL>
```

We assume basic proficiency of the reader with training models using [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

This repository provides sample code for retrofitting a Llama 2 7B model with DMC to a desired compression rate. Retrofitting a different model would require adjusting hyperparameters such as the learning rate, batch size, etc. Please keep in mind that the dataset used here is small and serves just for test-running the code. Retrofitting a model to 2x compression requires approximately 35B tokens, and the required number of tokens scales linearly with the target compression rate.

The training is performed in three stages; for a detailed description, please see the manuscript at [https://arxiv.org/abs/2403.09636](https://arxiv.org/abs/2403.09636). A checkpoint will be saved after every stage, from which the next stage can start.
bash
```
# Stage 1: Zeroing-out the decision neurons (250 steps)
./examples/dmc/train.sh zero_out <LLAMA2_MODEL> <TOKENIZER_MODEL>

# Stage 2: Main retrofitting with increasing compression rate (6k steps for 2x compression)
./examples/dmc/train.sh retrofit <STAGE1_MODEL> <TOKENIZER_MODEL>

# Stage 3: Fine-tuning with a fixed compression rate (2k steps)
./examples/dmc/train.sh finetune <STAGE2_MODEL> <TOKENIZER_MODEL>
```
# Known limitations

### Training data
Unfortunately, we do not provide the original training data. However, we believe that popular open source datasets would be of sufficient quality for replication.

### Minimum page size limit in FlashAttention
At the time of writing, FlashAttention 2 limits the page size for PagedAttention to 256 tokens. However, the optimal page size for DMC is much lower, especially with a high compression ratio. Please see [this PR](https://github.com/Dao-AILab/flash-attention/pull/824) for the code supporting smaller page sizes.

### Prefill phase
Our code is not optimized for the prefill phase, and currently it is executed auto-regressively.

### Pipeline parallelism
We do not provide code for training DMC with pipeline parallelism, which is necessary for larger models (e.g., Llama 2 70B).

# Citation
If your work is based or inspired by this repository, please cite:
```
@InProceedings{pmlr-v235-nawrot24a,
  title =        {Dynamic Memory Compression: Retrofitting {LLM}s for Accelerated Inference},
  author =       {Nawrot, Piotr and {\L}a\'{n}cucki, Adrian and Chochowski, Marcin and Tarjan, David and Ponti, Edoardo},
  booktitle =    {Proceedings of the 41st International Conference on Machine Learning},
  pages =        {37396--37412},
  year =         {2024},
  editor =       {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume =       {235},
  series =       {Proceedings of Machine Learning Research},
  month =        {21--27 Jul},
  publisher =    {PMLR},
  pdf =          {https://raw.githubusercontent.com/mlresearch/v235/main/assets/nawrot24a/nawrot24a.pdf},
  url =          {https://proceedings.mlr.press/v235/nawrot24a.html},
  abstract =     {Transformers have emerged as the backbone of large language models (LLMs). However, generation remains inefficient due to the need to store in memory a cache of key–value representations for past tokens, whose size scales linearly with the input sequence length and batch size. As a solution, we propose Dynamic Memory Compression (DMC), a method for on-line key–value cache compression at inference time. Most importantly, the model learns to apply different compression ratios in different heads and layers. We retrofit pre-trained LLMs such as Llama 2 (7B, 13B and 70B) into DMC Transformers, achieving up to $\sim 3.7 \times$ throughput increase during auto-regressive inference on an NVIDIA H100 GPU. DMC is applied via continued pre-training on a negligible percentage of the original data without adding any extra parameters. We find that DMC preserves the original downstream performance with up to 4$\times$ cache compression, outperforming up-trained grouped-query attention (GQA) and key–value eviction policies (H$_2$O, TOVA). GQA and DMC can be even combined to obtain compounded gains. As a result DMC fits longer contexts and larger batches within any given memory budget. We release the DMC code and models at https://github.com/NVIDIA/Megatron-LM/tree/DMC.}
}

```
