import torch
import torch.distributed as dist
from functools import partial
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.models.heterogenous_parallel.model_specs import get_vlm_mimo_model
from tests.unit_tests.models.heterogenous_parallel.parallel_utils import (
    get_module_to_grid_tuple, 
    multimodule_no_sync, 
    finalize_model_grads,
    get_pg_collections_for_rank,
    zero_grad_buffer_for_multimodule
)
from tests.unit_tests.models.heterogenous_parallel.data import get_data_iterator, get_batch
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
import megatron.core.pipeline_parallel.schedules as schedule


def loss_func(loss_mask, output_tensor):
    """Simple loss function for MIMO model training.
    
    Args:
        loss_mask: mask indicating which tokens contribute to the loss
        output_tensor: model output tensor
    
    Returns:
        tuple: (loss, num_tokens, metrics_dict)
    """
    losses = output_tensor.float()
    
    loss_mask = loss_mask.contiguous().view(-1).float()
    
    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])
    
    return (total_loss, total_tokens, {'lm loss': (reporting_loss)})


def forward_step(data_iterator, model):
    """Forward step for MIMO model training.
    
    Args:
        data_iterator: iterator over the dataset
        model: MIMO model instance
    
    Returns:
        tuple: (output_tensor, loss_function)
    """
    data_batch = get_batch(data_iterator)
    if data_batch is None:
        data_batch = {'input_ids': None}
    output_tensor, loss_mask = model(**data_batch)
    # Return output and loss function
    return output_tensor, partial(loss_func, loss_mask)


def test_1f_1b_schedule_vlm_mimo_model_custom_pgs(
    vision_num_layers, vision_hidden_size, 
    language_num_layers, language_hidden_size, 
    vocab_size, image_seq_length, seq_length, 
    special_token_ids,
    vision_tp, vision_pp, vision_dp,
    language_tp, language_pp, language_dp,
    batch_size, num_microbatches,
    num_iterations=1, profile_start_step=None, profile_end_step=None, enable_profiling=False
):
    """Test 1F1B schedule with VLM MIMO model using custom process groups.
    
    Args:
        vision_num_layers: Number of layers in vision encoder
        vision_hidden_size: Hidden size for vision encoder
        language_num_layers: Number of layers in language model
        language_hidden_size: Hidden size for language model
        vocab_size: Vocabulary size
        image_seq_length: Sequence length for images
        seq_length: Total sequence length (text tokens = seq_length - image_seq_length)
        special_token_ids: Dictionary of special token IDs
        vision_tp, vision_pp, vision_dp: Vision model parallelism configs (TP, PP, DP)
        language_tp, language_pp, language_dp: Language model parallelism configs (TP, PP, DP)
        batch_size: Batch size for training
        num_microbatches: Number of microbatches for pipeline parallelism
    """
    logging.info("Creating VLM MIMO model...")
    mimo_model, module_to_grid_map, topology = get_vlm_mimo_model(
        vision_num_layers=vision_num_layers,
        vision_hidden_size=vision_hidden_size,
        language_num_layers=language_num_layers,
        language_hidden_size=language_hidden_size,
        vocab_size=vocab_size,
        seq_len=seq_length,
        special_token_ids=special_token_ids,
        vision_tp=vision_tp,
        vision_pp=vision_pp,
        vision_dp=vision_dp,
        language_tp=language_tp,
        language_pp=language_pp,
        language_dp=language_dp,
    )
    
    logging.info(f"Rank {dist.get_rank()}: Model created successfully")
    
    # Set up module to grid tuple for no_sync and finalize_model_grads
    module_to_grid_tuple = get_module_to_grid_tuple(
        mimo_model, 
        module_to_grid_map['images'], 
        module_to_grid_map['language_module']
    )
    
    # Configure no_sync and finalize_model_grads functions
    mimo_model.config.no_sync_func = partial(multimodule_no_sync, module_to_grid_tuple=module_to_grid_tuple)
    mimo_model.config.finalize_model_grads_func = partial(finalize_model_grads, module_to_grid_tuple=module_to_grid_tuple)
    
    # Create multimodule communicator
    multimodule_communicator = MultiModulePipelineCommunicator(
        module_to_grid_map, topology, mimo_model.config, dim_mapping={'b': 0, 's': 1, 'h': 2}
    )
    
    logging.info(f"Rank {dist.get_rank()}: Creating data iterator...")
    
    # Get data iterator
    data_iterator = get_data_iterator(
        encoder_grid=module_to_grid_map['images'],
        llm_grid=module_to_grid_map['language_module'],
        image_seq_length=image_seq_length,
        seq_length=seq_length,
        image_special_token_id=special_token_ids['images'],
        batch_size=batch_size,
        vocab_size=vocab_size,
        vision_hidden_size=vision_hidden_size
    )
    
    # Set model type for unit test
    mimo_model.model_type = 'unit-test'
    
    # Prepare common arguments for schedule
    common_args = {
        'forward_step_func': forward_step,
        'data_iterator': data_iterator,
        'model': [mimo_model],
        'num_microbatches': num_microbatches,
        'seq_length': seq_length,
        'micro_batch_size': batch_size,
        'forward_only': False,
    }
    
    # Get pg collections for modules that should be initialized on this rank
    pg_collection = get_pg_collections_for_rank(module_to_grid_map)
    print(f"for debug: Rank {dist.get_rank()}: pg_collection: {pg_collection}")
    all_losses = []
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from megatron.core.optimizer import get_megatron_optimizer
    # Create optimizer config
    optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=0.001,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
    )
    model_chunks = []
    if mimo_model.modality_submodules is not None:
        for submodule in mimo_model.modality_submodules.values():
            if submodule is not None:
                model_chunks.append(submodule)
    if mimo_model.language_model is not None:
        if mimo_model.language_model is not None:
            model_chunks.append(mimo_model.language_model)
    # print(f"for debug: Rank {dist.get_rank()}, model_chunks used to create optimizer: {model_chunks}")
    optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=model_chunks,
        use_gloo_process_groups=False,  # Required when using custom process groups
        pg_collection=pg_collection[0], # [TODO by shifangx] check if the pg_collection is correct
    )

    for iteration in range(num_iterations):
        # Start profiling if enabled
        if enable_profiling and profile_start_step is not None and iteration == profile_start_step:
            logging.info(f"Rank {dist.get_rank()}: Starting profiler at iteration {iteration}")
            torch.cuda.cudart().cudaProfilerStart()
        
        logging.info(f"Rank {dist.get_rank()}: Iteration {iteration} - Starting 1F1B schedule...")
        
        # Run 1F1B schedule
        losses_reduced = schedule.forward_backward_pipelining_without_interleaving(
            p2p_communicator=multimodule_communicator, 
            pg_collection=pg_collection, 
            **common_args
        )

        all_losses.append(losses_reduced)
        for idx, loss in enumerate(losses_reduced):
            writer.add_scalar('training loss', loss['lm loss'][0], iteration)
            writer.add_scalar('num tokens', loss['lm loss'][1], iteration)
        logging.info(f"Rank {dist.get_rank()}: Iteration {iteration} - Losses: {losses_reduced}")

        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        print(f"for debug: Rank {dist.get_rank()}, at iteration {iteration}, update_successful: {update_successful}, grad_norm: {grad_norm}, num_zeros_in_grad: {num_zeros_in_grad}")

        zero_grad_buffer_for_multimodule(module_to_grid_tuple)
        
        # Stop profiling if enabled
        if enable_profiling and profile_end_step is not None and iteration == profile_end_step:
            logging.info(f"Rank {dist.get_rank()}: Stopping profiler at iteration {iteration}")
            torch.cuda.cudart().cudaProfilerStop()
    
    writer.flush()
    logging.info(f"Rank {dist.get_rank()}: Training completed. All losses: {all_losses}")
    
    return all_losses


if __name__ == "__main__":
    # Initialize distributed training
    Utils.initialize_distributed()

    # Profiling configuration
    enable_profiling = True
    num_iterations = 6
    profile_start_step = 3
    profile_end_step = 5
    
    # Model parameters
    vision_num_layers = 16
    vision_hidden_size = 1024
    language_num_layers = 16
    language_hidden_size = 2048

    # Data parameters
    vocab_size = 48000
    image_seq_length = 1024
    seq_length = 4096  # Total sequence length (text tokens = seq_length - image_seq_length)
    special_token_ids = {"images": 32000}

    # Model parallelisms (CP and EP are hardcoded to 1 in model_specs.py)
    vision_tp, vision_pp, vision_dp = 1, 2, 1
    language_tp, language_pp, language_dp = 1, 2, 2
    
    # Training parameters
    rank = dist.get_rank()
    global_batch_size = 32
    num_microbatches = 16
    if rank < vision_tp*vision_pp*vision_dp:
        assert global_batch_size%(num_microbatches * vision_dp)==0, \
            f"global_batch_size ({global_batch_size}) should be divisible by (num_microbatches ({num_microbatches}) * vision_dp ({vision_dp}))"
        batch_size = global_batch_size//(num_microbatches * vision_dp)
        print(f"for debug: Rank {rank}, is in vision module, batch_size: {batch_size}")
    else:
        assert global_batch_size%(num_microbatches*language_dp)==0, \
            f"global_batch_size ({global_batch_size}) should be divisible by (num_microbatches ({num_microbatches}) * language_dp ({language_dp}))"
        batch_size = global_batch_size// (num_microbatches*language_dp)
        print(f"for debug: Rank {rank}, is in language module, batch_size: {batch_size}")
    
 
    losses = test_1f_1b_schedule_vlm_mimo_model_custom_pgs(
        vision_num_layers=vision_num_layers,
        vision_hidden_size=vision_hidden_size,
        language_num_layers=language_num_layers,
        language_hidden_size=language_hidden_size,
        vocab_size=vocab_size,
        image_seq_length=image_seq_length,
        seq_length=seq_length,
        special_token_ids=special_token_ids,
        vision_tp=vision_tp,
        vision_pp=vision_pp,
        vision_dp=vision_dp,
        language_tp=language_tp,
        language_pp=language_pp,
        language_dp=language_dp,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_iterations=num_iterations,
        profile_start_step=profile_start_step,
        profile_end_step=profile_end_step,
        enable_profiling=enable_profiling,
    )
    logging.info(f"Final losses: {losses}")

    dist.destroy_process_group()
