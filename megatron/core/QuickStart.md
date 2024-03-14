## Quick Start
The following guide will show you how to quickly get started with Megatron Core. It will show you the following
* We will initalize megatron core on 2 GPUS. 
* We will build a GPT model with tensor model parallel size 2, pipeline parallel size 1
* We will train it for a few iterations using megatron core schedules
* We will save the model using the distributed checkpointing format
* We will load the model saved above. 

*NOTE: The following has been testing for megatron core version 0.5 and NGC Pytorch Container version 24.02

### Environment Setup
```
docker run --ipc=host --shm-size=512m --gpus all -it nvcr.io/nvidia/pytorch:24.02-py3

pip install megatron_core
pip install tensorstore==0.1.45
pip install zarr
```
<br>

### Writing Your First Training Loop
The following steps will walk you through how you can create a sample GPT model split across tensors (Tensor model parallel ) on 2 GPUS, and run a forward pass through it using a MockGPT dataset helper class that we created in Megatron core. 

<br>

**NOTE: All of the folowing steps needs to be put into a script and then run as explained in the last step** 

<br>

**STEP 1 - Initialize Distributed Training and Model parallel setup**
The following utility when called initalizes your distributed setup. 

```
import os
import torch
from megatron.core import parallel_state

def initialize_distributed(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1):
    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)
```
<br>

**STEP 2 - GPT Model Setup**
The following step shows you how you can quickly create a GPT model. For a list of other configs that you can pass into the model look into [transformer_config.py](megatron/core/transformer/transformer_config.py)
```
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

def model_provider():
    """Build the model."""

    transformer_config = TransformerConfig(
        num_layers=2, 
        hidden_size=12, 
        num_attention_heads=4, 
        use_cpu_initialization=True, 
        pipeline_dtype=torch.float32)

    gpt_model = GPTModel(
        config=transformer_config, 
        transformer_layer_spec=get_gpt_layer_local_spec(), 
        vocab_size=100, 
        max_sequence_length=64)

    return gpt_model
```
<br>

**STEP 3 - GPT Mock dataset setup**
The following shows you how you can quickly get started with a mock dataset utility we created. In order to train with your data, please use the actual GPTDataset class in [gpt_dataset.py](megatron/core/datasets/gpt_dataset.py)

To find more information about megatron core data pipeline please refer to [this](megatron/core/datasets/readme.md?ref_type=heads)

```
from torch.utils.data import DataLoader
from megatron.core.datasets.utils import Split
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset

def get_train_data_iterator():
    config = GPTDatasetConfig(
        is_built_on_rank=lambda:(parallel_state.is_pipeline_last_stage() or parallel_state.is_pipeline_first_stage()), 
        random_seed = 0, 
        sequence_length = 64, 
        blend=[], 
        mock=True, 
        reset_position_ids=False, 
        reset_attention_mask=False, 
        eod_mask_loss=False, 
        tokenizer="dummy")

    training_data= MockGPTDataset(Split.train, config)

    train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)

    train_iterator = iter(train_dataloader)
    return train_iterator
```
<br>

**STEP 4 - Forward Step Function**
In megatron core, we use [schedules.py](megatron/core/pipeline_parallel/schedules.py) to run the model. So it is sufficient to define a forward step function which takes as input the data iterator and the model and produces as output the output tensor and a loss function 

```
from functools import partial

def forward_step_func(data_iterator, model):
   
    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups. 
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    data = next(data_iterator)
    tokens = data['tokens'].to(device)
    attention_mask = data['attention_mask'].to(device)
    position_ids = data['position_ids'].to(device)
    labels = data['labels'].to(device)
    loss_mask = data['loss_mask'].to(device)
   
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)   
```
<br>

**STEP 5 - Load and Save Distributed Checkpoint**
Megatron core uses distributed checkpoint for loading and saving model. This gives you the flexiblity to convert model from one model parallel setting to another when you load a model (i.e A model trained with tensor parallel size 2, can now be loaded as tensor model parallel size 4 etc.)

*NOTE: Make sure you have zarr and tensorstore pip package installed as shown in the environment setup*

```
from megatron.core import dist_checkpointing

def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)

def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict=gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)
    gpt_model.load_state_dict(checkpoint)
    return gpt_model
```
<br>

**STEP 6 - Main Function**
The following is the main function that needs to go into your script. 
```
from pathlib import Path
from torch.optim import Adam
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

if __name__ == "__main__":
    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    gpt_model = model_provider()
    device = torch.device("cuda")
    gpt_model.to(device)

    optim = Adam(gpt_model.parameters())
    
    train_iterator = get_train_data_iterator()
    
    forward_backward_func = get_forward_backward_func()

    # Running the model for 5 iterations
    for _ in range(5):
        optim.zero_grad()
        
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=64,
            micro_batch_size=8,
            decoder_seq_length=64,
            forward_only=False)
    
        optim.step()

        print(f'Losses reduced :  {losses_reduced}')

    # Saving the model
    save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path='/workspace/ckpt')

    # Loading the model
    gpt_model = load_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path='/workspace/ckpt')
    gpt_model.to(device)
    print('Successfully loaded the model')  
```
<br>

**STEP 7 - Running the full example**
All the above steps are put to gether in a [run_simple_mcore_train_loop.py](examples/run_simple_mcore_train_loop.py) script in examples folder in megatron . You can run it as follows

```
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/examples
NUM_GPUS=2
torchrun --nproc-per-node $NUM_GPUS run_simple_mcore_train_loop.py
```
<br>

### Extending Further
The above example introduced you to a basic training loop in MCore. To see more advanced examples please look at [pretrain_gpt.py]. That will show you how you can write more complex training loops, involving pipeline parallel, context parallel, rope embeddings, mixture of experts and all other functionalities present in mcore. 
