# July 2023 sync with NVIDIA/Megatron-LM
This folder includes details about the recent sync with the NVIDIA/Megatron-LM repo (where this repo is forked from). It includes example scripts we used to test after the sync, together with this README documentation about what were tested.

We also created a [backup branch](https://github.com/microsoft/Megatron-DeepSpeed/tree/before_rebase) which is the version before this sync. This branch is just for comparison tests and for temporary use when debugging the main branch. We do not plan to continue supporting the version before sync.

## List of rebase efforts/achievements
* Enabling Megatron-LM's sequence parallel.
* Enabling rotary positional embedding.
* Enabling FlashAttention v1 and v2.
* Fix the conflicts related to activation checkpointing when DeepSpeed is used with the newest Megatron-LM since NVIDIA introduced some new fine-grained partial checkpointing techniques which DeepSpeed is currently not compatible.
* Major refactor to DeepSpeed pipeline parallelism implementation for GPT model in order to work with newest Megatron-LM.
* Fix model checkpoint save/load when DeepSpeed is used with the newest Megatron-LM.
* Fully verified the performance and correctness of GPT pretraining after rebasing.

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
We tested and verified that flash attention feature introduced by this sync works properly for GPT pretraining. 
Our code automatically uses [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) when avaiable.

We compared the training using the [toy example script](ds_pretrain_gpt_125M.sh) and the [toy example script with flash attention](ds_pretrain_gpt_125M_flashattn.sh) on 8 A100 GPUs, and found that FlashAttention (1.0,4) increased training throughput (TFLOPs per GPU) from 25 to 32. When scaling up the model to 2.7B using the same script, FlashAttention-2 improved the training throughput 121 TFLOPs to 132 TFLOPs in comparison to FlashAttention 1.x.

For installation instructions, please refer to [FlashAttention's repository](https://github.com/Dao-AILab/flash-attention).

## Rotary Positional Embedding (RoPE)
We also tested and verified that the Rotary Positional Embedding (RoPE) introduced by this sync works properly for GPT pretraining. By comparing the training between [without RoPE](ds_pretrain_gpt_1.3B.sh) and [with RoPE](ds_pretrain_gpt_1.3B_rope.sh), we are able to observe that RoPE helps improving the model convergence just like [previous observation](https://blog.eleuther.ai/rotary-embeddings/).

## Notes/TODOs
* After the sync, DeepSpeed still relies on the older activation checkpointing mechanism (see function ```_checkpointed_forward``` in ```Megatron-DeepSpeed/megatron/model/transformer.py```) since we didn't have time to integrate with the new version yet. Contribution is very welcomed.
* (Aug 2023 update) With the contribution from 3P users (https://github.com/microsoft/Megatron-DeepSpeed/pull/225), now it's also possible to use Megatron-LM's newer activation checkpointing mechanism. However, currently it's still not compatible with DeepSpeed, so you won't be able to combine it with any DeepSpeed technologies. We DeepSpeed team compared the [older mechanism](ds_pretrain_gpt_1.3B.sh) and [newer mechanism](ds_pretrain_gpt_1.3B_megatron_checkpointing.sh) on 1 DGX-2 node (16 V100), and found that the older mechanism has less memory saving (older max allocated 15241 MB, newer 12924 MB) and higher throughput (older 23.11 TFLOPs newer 17.26 TFLOPs). Thus currently we still recommend using the older mechanism both because of the similar checkpointing performance, and (more importantly) because only older mechnaism is compatible with DeepSpeed (and in this case you can combine with ZeRO to achieve more memeory saving).
