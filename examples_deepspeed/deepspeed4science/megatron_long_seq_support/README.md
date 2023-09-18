# Megatron-DeepSpeed Rebase with Optimizations

We rebased and enabled DeepSpeed with the latest Megatron repo. This folder contains examples that demonstrate how to use the new Megatron-DeepSpeed for training GPT like models with new features.

## Rebasing Efforts/Achievements
New features:
- Enabled Megatron-LM's sequence parallel.
- Enabled rotary positional embedding.
- Enabled FlashAttention v1 and v2.
- Enabled new fused kernels from NVIDIA.

New optimizations:
- Enabled attention map memory optimization, where we first generated attention mask on CPU memory and then moved it into GPU memory to avoid out-of-memory errors when training with very large sequence lengths.
- Position embedding partitioning, where we split weights of position encoding across all GPUs when enabling sequence parallel to further reduce the memory footprint.

Resolved Issues:
- Fixed the conflicts related to activation checkpointing when DeepSpeed was used with the newest Megatron-LM. NVIDIA introduced new fine-grained partial checkpointing technique, which DeepSpeed was not compatible with. Support for fine-grained checkpointing will be left as future work.
- Major refactoring to DeepSpeed pipeline parallelism implementation for GPT model in order to work with the newest Megatron-LM.
- Fixed model checkpoint save/load when DeepSpeed was used with the newest Megatron-LM.
- Fully verified the performance and correctness of GPT pretraining after rebasing.

## Setting Up the Virtual Environment

```shell
# clone source code
git clone https://github.com/microsoft/DeepSpeed.git
git clone https://github.com/microsoft/Megatron-DeepSpeed.git
git clone https://github.com/NVIDIA/apex

# creat a new virtual environment
cd Megatron-DeepSpeed
python3 -m venv ./venvs/megatron-deepspeed --system-site-packages
source ./venvs/megatron-deepspeed/bin/activate

# install the newest DeepSpeed
cd ../DeepSpeed/
pip install -e .

# install apex
cd ../apex/
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" -e ./

# install pybind11
cd ../
pip install pybind11
```

Megatron-DeepSpeed's sequence parallelism can be combined with the following types of attention.

- Classic attention
- FlashAttention version 1.x (enabled by `--use-flash-attn-v1`)
- FlashAttention version 2.x (enabled by `--use-flash-attn-v2`)
- FlashAttention + Triton (enabled by `--use-flash-attn-triton`)

FlashAttention version 2.x may have numerical stability issues. For the best performance, we recommend using FlashAttention + Triton. 
We show installation steps of thoes 3 types of FlashAttention

```shell

# install FlashAttention version 1.x
pip install flash-attn==1.0.4

# install FlashAttention version 2.x
cd ../
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install

# install Triton-based FlashAttention
git clone -b legacy-backend https://github.com/openai/triton
cd triton/python/
pip install cmake
pip install .

cd ../
git clone -b v1.0.4 https://github.com/HazyResearch/flash-attention
cd flash-attention
python setup.py install
```

## Example Showcase

One of the optimizations enabled from this rebase is to enable Megatron-style long sequence parallelism. To enable sequence parallelism, add the `--sequence-parallel` flag in the training script. We provide two training scripts for ([GPT1.3B](pretrain_gpt_1.3B_seq_parallel.sh) and [GPT30B](pretrain_gpt_13B_seq_parallel.sh)) that enable sequence parallelism, which are available in this foloder.

By default, the degree of sequence parallelism is equal to the degree of model tensor parallelism. The users may also want to ensure that the sequence length is divisible by the degree of sequence parallelism to avoid performance penalties. 
Please also ensure that your model dimension is compliant with FlashAttention's requirements. For instance, to achieve the optimal performance, the head size should be divisible by 8. Refer to the document of [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v1.0.4) for more details.

## Performance Comparison between Old Megatron-DeepSpeed and New Megatron-DeepSpeed

The following experiments are performed on 4 NVIDIA DGX A100-40GB nodes, connected through 8 HDR InfiniBand (200Gb/s per HDR). TP stands for tensor parallelism.

| Sequence Length | Old Megatron-DeepSpeed  (TFLOPS) | New Megatron-DeepSpeed  (TFLOPS) |
|-----------------|----------------------------------|----------------------------------|
| 2k              | 25 (TP=32)                       | 68 (TP size=32)                  |
| 4k              | 28 (TP=32)                       | 80 (TP size=32)                  |
| 8k              | OoM                              | 86 (TP size=32)                  |
| 16k             | OoM                              | 92 (TP size=32)                  |
| 32k             | OoM                              | 100 (TP size=32)                 |
| 64k             | OoM                              | 106 (TP size=32)                 |
| 128k            | OoM                              | 119 (TP size=32)                 |
| 256k            | OoM                              | 94 (TP size=32)                  |

The new Megatron-DeepSpeed is able to support longer sequence lengths without triggering out-of-memory errors because it enables sequence parallelism, which partitions the activation memory when sequence lengths are massive. The new Megatron-DeepSpeed supports FlashAttention, which reduces the memory consumption of the attention map calculation from quadratic to linear complexity with respect to the sequence length. It supports position embedding partitioning, which further reduces the memory consumption. The new Megatron-DeepSpeed can achieve higher TFLPOS because it includes new fused kernels from NVIDIA and supports larger batch sizes using the memory optimizations without triggering out-of-memory errors.

## Acknowledgements

We would like to acknowledge the use of the supercomputing resources of the Argonne Leadership Computing Facility (ALCF), which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.  The resources provided by ALCF(Argonne) have been invaluable in helping us to conduct this work and achieve our goals.
