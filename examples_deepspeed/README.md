# Megatron-DeepSpeed Recipes and Scripts

This folder includes various example scripts with DeepSpeed technologies integrated. Below we describe each sub-folder, sorted by last update date.

## Sync with NVIDIA/Megatron-LM (last updated: Jul 2023)
The ```rebase``` folder includes details about the recent sync with the NVIDIA/Megatron-LM repo (where this repo is forked from). It includes example scripts we used to test after the sync, together with a README documentation about what were tested.

## Data Efficiency (last updated: Feb 2023)

The ```data_efficiency``` folder includes GPT-3 and BERT pretraining examples for DeepSpeed Data Efficiency Library, together with examples of zero-shot evaluation for GPT models and GLUE finetuning for BERT models. Please refer to the detailed tutorials in data_efficiency/README.MD. Currently this folder includes the newest example scripts for GPT/BERT pretraining/eval/finetuning, both with and without DeepSpeed Data Efficiency Library techniques.

## BERT example (last updated: Dec 2022)

The ```bert_with_pile``` folder includes examples about BERT-style model pre-training (using the public Pile data or user's own data) with DeepSpeed integration. Please refer to the readme in the folder for tutorial.

## Azure (last updated: Nov 2022)

We strongly recommend to start with AzureML recipe in the ```azureml``` folder.

If you have a custom infrastructure (e.g. HPC clusters) or Azure VM and VMSS based environments, please refer to the bash scripts in the ```azure``` folder.

## Model Compression (last updated: Aug 2022)

The ```compression``` folder includes examples about layer reduction for task-agnostic compression. Please refer to [this tutorial](https://www.deepspeed.ai/tutorials/model-compression/#11-layer-reduction) about the DeepSpeed Model Compression Library. These recipes are for GPT-style NLG models.

## MoE (last updated: Jun 2022)

Please see the ```MoE``` folder for different training recipes and scripts for Mixture-of-expert based models and dense models. These recipes are for GPT-style NLG models, and currently this is the only folder with MoE training examples.

## Curriculum Learning (last updated: Oct 2021)

Curriculum learning recipes are in the ```curriculum_learning``` folder. Please refer to the detailed tutorials linked inside. These recipes are for GPT-style NLG models.
Note that the DeepSpeed Data Efficiency Library above includes a more general curriculum learning support. This legacy curriculum learning feature is still compatible, but we recommend using the DeepSpeed Data Efficiency Library above. However, the newer DeepSpeed Data Efficiency Library currently is not compatible with pipeline parallelism. So if you have to use pipeline parallelism, you would need to use this legacy curriculum learning version.
