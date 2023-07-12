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

## Notes/TODOs
* After the sync, DeepSpeed still relies on the older activation checkpointing mechanism (see function ```_checkpointed_forward``` in ```Megatron-DeepSpeed/megatron/model/transformer.py```) since we didn't have time to integrate with the new version yet. Contribution is very welcomed.
