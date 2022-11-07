import os
import requests
import sys

# AzureML libraries
import azureml.core
from azureml.core import Dataset, Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import PyTorchConfiguration
from azureml.core.environment import DockerBuildContext

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

# For setting up a workspace, refer to: https://github.com/Azure/azureml-examples/tree/main/python-sdk#set-up
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

#-------------------------------------------------------------------------------
# Prepare Compute Cluster
#-------------------------------------------------------------------------------
cluster_name = "a100-80gb"

# Verify that the cluster doesn't exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_ND96amsr_A100_v4', min_nodes=32, max_nodes=32)
    
    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

#-------------------------------------------------------------------------------
# Prepare Data
# Megatron-DeepSpeed takes in data_path, vocab_file, and merge_file.
# For AML, we are adding a parameter aml_data_download_path which specifies how to deliver the dataset to a compute target.
# In the submitted run, files in the datasets will be either mounted or downloaded to local path on the compute target.
# 
# data_path for this example is path to the .bin and .idx file, excluding extension.
# e.g. for data/BookCorpusDataset_text_document.bin and data/BookCorpusDataset_text_document.idx,
# data_path = "data/BookCorpusDataset_text_document"
#
# Once the folder is downloaded to the compute target, it will use aml_data_download_path to locate the folder
# and data_path to locate .bin and .idx files
#
# vocab_file and merge_file would also be passed in a similar way.
#-------------------------------------------------------------------------------
datastore = ws.get_default_datastore()
blobstore_datadir = "bookcorpus_data"
data_path = f"BookCorpusDataset_text_document"
# Load data folder which contains bookcorpus .bin and .idx files
train_dataset = Dataset.File.from_files(path=[(datastore, blobstore_datadir)])
aml_data_download_path = train_dataset.as_download(blobstore_datadir)

vocab_file_dataset = Dataset.File.from_files("https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json")
merge_file_dataset = Dataset.File.from_files("https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt")
vocab_file = vocab_file_dataset.as_download()
merge_file = merge_file_dataset.as_download()


#-------------------------------------------------------------------------------
# Setup training environment
#-------------------------------------------------------------------------------

megatron_ds_env = Environment.from_docker_build_context(name='megatron-ds-curated-acpt', docker_build_context=DockerBuildContext.from_local_directory(workspace = ws, path = '.', dockerfile_path='Dockerfile.dockerfile'))
megatron_ds_env.register(ws).build(ws).wait_for_completion()  # Comment this out if environment already exists

#-------------------------------------------------------------------------------
# Training Settings and Arguments
#-------------------------------------------------------------------------------
node_count = 2
total_processes_count = 16
micro_batch_size = 1
global_batch_size = micro_batch_size * total_processes_count
tensorboard_dir = '/tmp/outputs/tensorboard'

run_args = ['--tensor-model-parallel-size', 1, 
            '--pipeline-model-parallel-size', 1, 
            '--num-layers', 20,
            '--hidden-size', 12288,
            '--num-attention-heads', 96,
            '--seq-length', 1024,
            '--loss-scale', 15, 
            '--max-position-embeddings', 1024, 
            '--micro-batch-size', micro_batch_size,
            '--global-batch-size', global_batch_size,
            '--train-iters', 100,
            '--lr', 6.0e-5,
            '--min-lr', 6.0e-6, 
            '--lr-decay-style', 'cosine',
            '--log-interval', 1, 
            '--eval-iters', 40, 
            '--eval-interval', 1000,
            '--aml-data-download-path', aml_data_download_path,
            '--data-path', data_path,
            '--vocab-file', vocab_file,
            '--merge-file', merge_file,
            '--save-interval', 1000, 
            '--split', '98,2,0',
            '--clip-grad', 1.0, 
            '--weight-decay', 0.1,
            '--adam-beta1', 0.9,
            '--adam-beta2', 0.95,
            '--init-method-std', 0.006,
            '--fp16',
            '--data-impl', 'mmap',
            '--checkpoint-activations',
            '--tensorboard-dir', tensorboard_dir,
            #'--cpu-optimizer',
            '--deepspeed',
            '--no-pipeline-parallel',
            '--deepspeed_config', 'ds_config.json',
            '--zero-stage', 3,
            '--deepspeed-activation-checkpointing',
            '--exit-interval', 5000,
]

#-------------------------------------------------------------------------------
# DeepSpeed ds_config.json
#-------------------------------------------------------------------------------
import json
ds_config = {
    "train_batch_size" : global_batch_size,
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "steps_per_print": 1,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
      "stage": 3,
      "stage3_max_live_parameters": 3e9,
      "stage3_max_reuse_distance": 3e9,
      "stage3_param_persistence_threshold": 1e5,
      "stage3_prefetch_bucket_size": 5e7,
      "contiguous_gradients": True,
      "overlap_comm": True,
      "reduce_bucket_size": 90000000,
      "sub_group_size": 1e9,
      "offload_optimizer": {
        "device": "none",
        "buffer_count": 4,
        "pipeline_read": False,
        "pipeline_write": False,
        "pin_memory": True
      }
    },
    "gradient_clipping": 1.0,
    "fp16": {
      "enabled": True,
      "initial_scale_power" : 15,
      "loss_scale_window": 1000,
      "hysteresis": 2,
      "min_loss_scale": 1
    },
    "wall_clock_breakdown": True,
    "zero_allow_untested_optimizer": False,
    "aio": {
      "block_size": 1048576,
      "queue_depth": 16,
      "single_submit": False,
      "overlap_events": True,
      "thread_count": 2
    }
  }

# Place ds_config.json in the same folder as pretrain_gpt.py (script to run)
ds_config_path = '../../ds_config.json'
with open(ds_config_path, 'w') as fp:
    json.dump(ds_config, fp, indent=4)

#-------------------------------------------------------------------------------
# Create ScriptRunConfig
#-------------------------------------------------------------------------------
distr_config = PyTorchConfiguration(process_count=total_processes_count, node_count=node_count)

megatron_ds_src = ScriptRunConfig(source_directory='../../',
                      script='pretrain_gpt.py',
                      arguments=run_args,
                      compute_target=compute_target,
                      environment=megatron_ds_env,
                      distributed_job_config=distr_config)

megatron_ds_src.run_config.environment_variables['NCCL_DEBUG'] = 'WARN'
megatron_ds_src.run_config.environment_variables['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
megatron_ds_src.run_config.environment_variables['NCCL_SOCKET_IFNAME'] = 'eth0'
megatron_ds_src.run_config.environment_variables['NCCL_IB_PCI_RELAXED_ORDERING']='1'
megatron_ds_src.run_config.environment_variables['UCX_TLS']='tcp'
megatron_ds_src.run_config.environment_variables['UCX_NET_DEVICES']='eth0'

#-------------------------------------------------------------------------------
# Submit experiment
#-------------------------------------------------------------------------------
experiment_name = 'megatron-ds'
experiment = Experiment(ws, name=experiment_name)

run = experiment.submit(megatron_ds_src, tags={'bs':micro_batch_size, 'gpus':total_processes_count})
