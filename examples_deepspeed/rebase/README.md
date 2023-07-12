# Sync with NVIDIA/Megatron-LM
This folder includes details about the recent sync with the NVIDIA/Megatron-LM repo (where this repo is forked from). It includes example scripts we used to test after the sync, together with this README documentation about what were tested.

We also created a [backup branch](https://github.com/microsoft/Megatron-DeepSpeed/tree/before_rebase) which is the version before this sync. This branch is just for comparison tests and for temporary use when debugging the main branch. We do not plan to continue supporting the version before sync.

## Test environment
We used 128 V100 GPUs (8 DGX-2 nodes, 16 GPU per node, inter-node network is InfiniBand with around 660 Gbps measured bandwidth) for the tests. For software, we used DeepSpeed v0.9.5.

## Verified cases and results
We verified the following cases (matching training/validation curves before/after sync, checkpoint save/load works) for GPT-3 pretraining:

* With DeepSpeed ZeRO stage 1
* With DeepSpeed ZeRO stage 1 and Megatron-LM's tensor parallelism
* With DeepSpeed ZeRO stage 1, Megatron-LM's tensor parallelism, and DeepSpeed's pipeline parallelism (i.e., 3D parallelism)

In addition, below is a performance/convergence comparison between before and after this sync.

| Case | TFLOPs (per GPU) | Validation loss at step 200 | Training script |
| ---- | ---------------- | --------------------------- | --------------- |
| Before sync, GPT-3 13B, 3D parallelism | 50 | 5.73 | [script (in the backup branch)](https://github.com/microsoft/Megatron-DeepSpeed/blob/before_rebase/examples/before_rebase_test/ds_pretrain_gpt_13B.sh) |
| After sync, GPT-3 13B, 3D parallelism | 55.6 | 5.71 | [script](ds_pretrain_gpt_13B.sh) |

At last, we provide a [toy example script](ds_pretrain_gpt_125M.sh) that users can try as the first test.

## Flash attention
We also tested and verified that flash attention feature introduced by this sync works properly for GPT pretraining. We compare the training using the [toy example script](ds_pretrain_gpt_125M.sh) and the [toy example script with flash attention](ds_pretrain_gpt_125M_flashattn.sh) on 8 A100 GPUs, and found that the training throughput (TFLOPs per GPU) increases from 25 to 32.

The installation of flash attention is a bit complex. Below is an example of how we install it.

```shell
WORK_DIR=flash_attn_repro
mkdir ${WORK_DIR} && cd ${WORK_DIR}
python -m venv venv/flash_attn_repro
source venv/flash_attn_repro/bin/activate
pip install packaging

# install triton
git clone -b legacy-backend https://github.com/openai/triton
cd triton/python/
pip install cmake; # build-time dependency
pip install -e .

# install
cd ${WORK_DIR}
git clone -b v1.0.4 https://github.com/HazyResearch/flash-attention
cd flash-attention
### Edit the source here ###
# Disable bias because the implementation doesn't support it
# See https://github.com/tohtana/flash-attention/commit/957c1549735e2e7348a1d2032b0fbc628f5d50c3
########################
python setup.py install
```

## Rotary Positional Embedding (RoPE)
We also tested and verified that the Rotary Positional Embedding (RoPE) introduced by this sync works properly for GPT pretraining (except that currently it cannot be used with DeepSpeed's pipeline parallelism. We are working on to support this combination). By comparing the training between [without RoPE](ds_pretrain_gpt_1.3B.sh) and [with RoPE](ds_pretrain_gpt_1.3B_rope.sh), we are able to observe that RoPE helps improving the model convergence just like [previous observation](https://blog.eleuther.ai/rotary-embeddings/).

## Notes/TODOs
* After the sync, DeepSpeed still relies on the older activation checkpointing mechanism (see function ```_checkpointed_forward``` in ```Megatron-DeepSpeed/megatron/model/transformer.py```) since we didn't have time to integrate with the new version yet. Contribution is very welcomed.
