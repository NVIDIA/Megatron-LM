## Quick Start

This guide for Megatron Core walks you through the following tasks:

* Initialize Megatron Core on two GPUS. 
* Build a GPT model with a tensor model parallel size of two and a pipeline parallel size of one.
* Train the model for five iterations using Megatron Core schedules.
* Save the model using the distributed checkpoint format.
* Load the model. 

**NOTE:** The following sample was tested using Megatron Core version 0.8.0 and NGC PyTorch Container version 24.02. 

### Set Up Your Environment

1. Run a new Docker container.

1. Clone the Megatron GitHub repo in it.

    ```
    docker run --ipc=host --shm-size=512m --gpus 2 -it nvcr.io/nvidia/pytorch:24.02-py3

    git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM
    ```
    <br>

### Write Your First Training Loop

In this task, you create a sample GPT model split across tensors (Tensor model parallel) on two GPUS, and run a forward pass through it using a MockGPT dataset helper class that was created in Megatron Core. 

<br>

**NOTE:** All of the following steps are in the [run_simple_mcore_train_loop.py](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/run_simple_mcore_train_loop.py) script. To run the ``run_simple_mcore_train_loop.py`` script:

    ```
    PYTHONPATH=$PYTHON_PATH:./megatron torchrun --nproc-per-node 2 examples/run_simple_mcore_train_loop.py
    ```


1. Initialize the distributed training and set up the model parallel:

    The following utility, when called, initializes your distributed setup:

    ```python
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

1. Set up the GPT model:

    Use the following code snippet to create a GPT model. For a list of other configurations that you can pass into the model, open and review [transformer_config.py](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/transformer_config.py).

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

1. Set up the GPT mock dataset:

    Use the following code snippet to explore the mock dataset utility.

    * To train the model using your data, use the `GPTDataset` class in [gpt_dataset.py](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/datasets/gpt_dataset.py).

    * To find more information about Megatron Core data pipeline, see the [data pipeline readme.md](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/datasets/readme.md?ref_type=heads).

    ```
    import torch
    from torch.utils.data import DataLoader

    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
    from megatron.training.tokenizer.tokenizer import _NullTokenizer
    from megatron.core.datasets.utils import compile_helpers

    _SEQUENCE_LENGTH = 64

    def get_train_data_iterator():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                compile_helpers()
            torch.distributed.barrier()
        else:
            compile_helpers()

        config = GPTDatasetConfig(
            random_seed=0,
            sequence_length=_SEQUENCE_LENGTH,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            tokenizer=_NullTokenizer(vocab_size=_SEQUENCE_LENGTH),
        )

        datasets = BlendedMegatronDatasetBuilder(
            MockGPTDataset, [1000, None, None], lambda: True, config
        ).build()

        train_dataloader = DataLoader(datasets[0], batch_size=8, shuffle=True)

        train_iterator = iter(train_dataloader)

        return train_iterator

    ```
    <br>

1. Add a forward step function:

    Megatron Core uses [schedules.py](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/pipeline_parallel/schedules.py) to run the model. Define a forward step function that takes the data iterator and the model as input and produces the output tensor and a loss function.

    ```python
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

1. Define your load and save distributed checkpoints:

    Megatron Core uses distributed checkpoints for loading and saving models. This allows you to convert the model from one parallel setting to another when you load it. 
    For example, a model trained with tensor parallel size `2`, can be loaded again as a tensor model with parallel size `4`.


    ```python
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

1. Add the main function:

    The following code snippet is the main function that needs to go into your script. It runs the model for five iterations, saves, and loads it.  

    ```python
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



### Review Advanced Examples

To review more advanced examples, explore [pretrain_gpt.py](https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py). ``pretrain_gpt.py`` has more complex training loops and includes the following Megatron Core features:

* pipeline parallel
* context parallel
* rope embeddings
* mixture of experts
