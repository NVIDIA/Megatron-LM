## Megatron-DeepSpeed on AzureML
Example script for running Megatron-DeepSpeed using Azure Machine Learning.

------

# Workspace Setup
Setup an AML workspace. Refer to: [set-up doc](https://github.com/Azure/azureml-examples/tree/main/v1/python-sdk#set-up).

# Dataset Preparation
Create AML Dataset. To run remote AML job, you need to provide AML FileDataset. 
Refer to [prepare_dataset script](prepare_dataset.py) to upload .bin and .idx files to blob store and on how to create FileDataset.

> Note: The folder `bookcorpus_data` used by [prepare_dataset script](prepare_dataset.py) should not be under `azureml` directories. It is because Azure ML does not allow to include large files (limit: 100 files or 1048576 bytes) for Docker build context.

# Training
Run Megatron-DeepSpeed on Azure ML. Refer to [aml_submit script](aml_submit.py).
