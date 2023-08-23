# Sequence Parallelism

This folder contains examples that demonstrate how to use DeepSpeed's sequence parallelism.

## Setting Up the Environment for FlashAttention

DeepSpeed's sequence parallelism can be combined with the following types of attention.

- Classic attention
- FlashAttention (enabled by `--use-flash-attn`)
- FlashAttention + Triton (enabled by `--use-flash-attn-triton`)

For the best performance, we recommend using FlashAttention + Triton. Here are the installation steps and the versions we have tested. Note that FlashAttention is compatible only with Turing, Ampere, Ada, or Hopper GPUs.

```shell
# install triton
git clone -b legacy-backend https://github.com/openai/triton
cd triton/python/
pip install cmake
pip install .

# install
cd ${WORK_DIR}
git clone -b v1.0.4 https://github.com/HazyResearch/flash-attention
cd flash-attention
python setup.py install
```

## Enabling Sequence Parallelism

To enable sequence parallelism, set the degree of parallelism using the `--ds-sequence-parallel-size` argument. Ensure that the number of attention heads is divisible by this value.
Ensure your model configuration is compliant with FlashAttention's requirements. For instance, to achieve optimal performance, the head size should be divisible by 8. Refer to the document of [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v1.0.4) for more details.

Some working examples ([GPT1.3B](ds_pretrain_gpt_1.3B_seq_parallel_32k.sh), [GPT30B](ds_pretrain_gpt_30B_seq_parallel_32k.sh)), that enable sequence parallelism, are available in this foloder.

Please note that our sequence parallelism feature is currently incompatible with Megatron-LM's tensor or pipeline parallelism.
