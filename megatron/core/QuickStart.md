## Quick Start

This guide for Megatron Core walks you through the following tasks:

* Initialize Megatron Core on two GPUS.
* Build a GPT model with a tensor model parallel size of two and a pipeline parallel size of one.
* Train the model for five iterations using Megatron Core schedules.
* Save the model using the distributed checkpoint format.
* Load the model.

**NOTE:** The following sample passes its process groups explicitly, which requires Megatron Core 0.21 or later.

### Set Up Your Environment

1. Run a new Docker container.

1. Clone the Megatron GitHub repo in it.

    ```bash
    docker run --ipc=host --shm-size=512m --gpus 2 -it -e PIP_CONSTRAINT= nvcr.io/nvidia/pytorch:26.01-py3

    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    pip install uv
    uv pip install -e .
    ```

    <br>

For a more comprehensive overview of different installation methods, refer to the [Installation Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html).

### Write Your First Training Loop

In this task, you create a sample GPT model split across tensors (Tensor model parallel) on two GPUS, and run a forward pass through it using a MockGPT dataset helper class that was created in Megatron Core.

<br>

**NOTE:** The following steps are based on the [run_simple_mcore_train_loop.py](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/run_simple_mcore_train_loop.py) script. To run the ``run_simple_mcore_train_loop.py`` script:

    ```
    PYTHONPATH=$PYTHON_PATH:./megatron torchrun --nproc-per-node 2 examples/run_simple_mcore_train_loop.py
    ```

1. Initialize the distributed training and set up the model parallel:

    The following utility, when called, initializes your distributed setup. `initialize_model_parallel` returns the process groups of the parallel grid it builds, as a `ProcessGroupCollection`. The rest of the script passes that collection to Megatron Core explicitly, instead of letting each component read the global grid in `parallel_state`:

    ```python
    import os
    import torch
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1) -> ProcessGroupCollection:
        parallel_state.destroy_model_parallel()

        # Torch setup for distributed training
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)

        # Megatron core distributed training initialization
        return parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)
    ```

    <br>

1. Set up the GPT model:

    Use the following code snippet to create a GPT model on the process groups that `initialize_distributed` returns. For a list of other configurations that you can pass into the model, open and review [transformer_config.py](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/transformer_config.py).

    ```
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    def model_provider(pg_collection):
        """Build the model on the process groups of its parallel grid."""

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
            max_sequence_length=64,
            pg_collection=pg_collection)

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
    from megatron.core.datasets.utils import compile_helpers
    from megatron.core.tokenizers import MegatronTokenizer

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
            tokenizer=MegatronTokenizer.from_pretrained(
                metadata_path={"library": "null-text"},
                vocab_size=_SEQUENCE_LENGTH,
            ),
            mid_level_dataset_surplus=0.005,
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

    The main function below wraps the model in `DistributedDataParallel`, so these functions unwrap it first.

    ```python
    from megatron.core import dist_checkpointing

    def save_distributed_checkpoint(checkpoint_path, gpt_model):
        # Access the underlying model if wrapped with DDP
        model = gpt_model.module if hasattr(gpt_model, 'module') else gpt_model
        sharded_state_dict = model.sharded_state_dict(prefix='')
        dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)

    def load_distributed_checkpoint(checkpoint_path, gpt_model):
        # Access the underlying model if wrapped with DDP
        model = gpt_model.module if hasattr(gpt_model, 'module') else gpt_model
        sharded_state_dict = model.sharded_state_dict(prefix='')
        checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)
        model.load_state_dict(checkpoint)
        return gpt_model
    ```

    <br>

1. Add the main function:

    The following code snippet is the main function that needs to go into your script. It runs the model for five iterations, saves, and loads it. Every Megatron Core call that needs process groups gets them from the collection that `initialize_distributed` returns: the RNG seeding, the model, `DistributedDataParallel`, the optimizer, the schedule and `finalize_model_grads`.

    ```python
    from pathlib import Path
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.utils import get_pg_rank, get_pg_size

    if __name__ == "__main__":
        pg_collection = initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
        # get_pg_rank and get_pg_size also accept the None that a grid without GTP weight
        # rematerialization may hold in gtp_remat and expt_gtp_remat.
        model_parallel_cuda_manual_seed(
            123,
            tp_rank=pg_collection.tp.rank(),
            ep_rank=pg_collection.ep.rank(),
            etp_rank=pg_collection.expt_tp.rank(),
            gtp_remat_rank=get_pg_rank(pg_collection.gtp_remat),
            egtp_remat_rank=get_pg_rank(pg_collection.expt_gtp_remat),
            gtp_remat_world_size=get_pg_size(pg_collection.gtp_remat),
            egtp_remat_world_size=get_pg_size(pg_collection.expt_gtp_remat),
        )

        gpt_model = model_provider(pg_collection)
        device = torch.device("cuda")
        gpt_model.to(device)

        # Wrap the model with DistributedDataParallel for gradient synchronization. It provides
        # the finish_grad_sync() method that finalize_model_grads() needs.
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
        )
        gpt_model = DistributedDataParallel(
            config=gpt_model.config,
            ddp_config=ddp_config,
            module=gpt_model,
            pg_collection=pg_collection,
        )

        # DistributedDataParallel accumulates the gradients in its own buffer and leaves no
        # param.grad for a torch.optim optimizer, so update the weights with the Megatron Core
        # optimizer, which reads that buffer.
        optim = get_megatron_optimizer(
            OptimizerConfig(optimizer='adam', lr=1e-3), [gpt_model], pg_collection=pg_collection
        )
        
        train_iterator = get_train_data_iterator()
        
        forward_backward_func = get_forward_backward_func(pp_size=pg_collection.pp.size(), vp_size=None)

        # Running the model for 5 iterations
        for _ in range(5):
            gpt_model.zero_grad_buffer()
            optim.zero_grad()
            
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=train_iterator,
                model=gpt_model,
                num_microbatches=1,
                seq_length=64,
                micro_batch_size=8,
                decoder_seq_length=64,
                forward_only=False,
                pg_collection=pg_collection)

            # All-reduce the gradients across the data-parallel ranks, and the gradients of
            # parameters that are not tensor-parallel, such as LayerNorm, across the
            # tensor-parallel ranks.
            finalize_model_grads([gpt_model], pg_collection=pg_collection)

            optim.step()

            print(f'Losses reduced :  {losses_reduced}')

        # Saving the model
        ckpt_path = '/workspace/ckpt'
        Path(ckpt_path).mkdir(parents=True, exist_ok=True)
        save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)

        # Loading the model
        gpt_model = load_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)
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
