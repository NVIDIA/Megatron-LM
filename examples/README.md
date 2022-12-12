## Recipes and Scripts

Please note that some of the script examples (e.g., pretrain_*.sh directly under ```Megatron-DeepSpeed/examples/``` folder) are from the original NVIDIA's Megatron-LM and does not have DeepSpeed integration (scripts with DeepSpeed integration should include the ```deepspeed``` keyword). Below we list various examples that do have DeepSpeed integration.

### Azure

We strongly recommend to start with AzureML recipe in the ```azureml``` folder.

If you have a custom infrastructure (e.g. HPC clusters) or Azure VM and VMSS based environments, please refer to the bash scripts in the ```azure``` folder.

### MoE

Please see the ```MoE``` folder for different training recipes and scripts for Mixture-of-expert based models and dense models. These recipes are for GPT-style NLG models.

### Data Efficiency 

The ```data_efficiency``` folder includes GPT-3 and BERT pretraining examples for DeepSpeed Data Efficiency Library. Please refer to the detailed tutorials in data_efficiency/README.MD.

### Curriculum Learning

Curriculum learning recipes are in the ```curriculum_learning``` folder. Please refer to the detailed tutorials linked inside. These recipes are for GPT-style NLG models.
Note that the DeepSpeed Data Efficiency Library above includes a more general curriculum learning support. This legacy curriculum learning feature is still compatible, but we recommend using the DeepSpeed Data Efficiency Library above.

### Model Compression

The ```compression``` folder includes examples about layer reduction for task-agnostic compression. Please refer to [this tutorial](https://www.deepspeed.ai/tutorials/model-compression/#11-layer-reduction) about the DeepSpeed Model Compression Library. These recipes are for GPT-style NLG models.

### BERT example

The ```bert_with_pile``` folder includes examples about BERT-style model pre-training (using the public Pile data or user's own data) with DeepSpeed integration. Please refer to the readme in the folder for tutorial.
