import os
import torch
import megatron.core.parallel_state as ps
from datetime import timedelta
import pytest


world_size = torch.cuda.device_count()
rank = int(os.environ['LOCAL_RANK'])
print('Ranks is : ' + str(rank))

def initialize_distributed():
    print(f'Initializing torch.distributed with rank: {rank}, world_size: {world_size}')
    torch.cuda.set_device(rank % torch.cuda.device_count())
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method=init_method, timeout=timedelta(seconds=10))

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size = None,
    pipeline_model_parallel_split_rank = None,
):
    # This might not be the right way to do this. 
    try:
        ps.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank)
    except:
        ps.destroy_model_parallel() 
        ps.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank)
        pass

def test_initialize_model_parallel():
    with pytest.raises(AssertionError):
        assert(ps.initialize_model_parallel())
    initialize_distributed()
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(tensor_model_parallel_size=2*world_size))
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(pipeline_model_parallel_size=2*world_size))
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(pipeline_model_parallel_size=world_size, tensor_model_parallel_size=world_size))
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(virtual_pipeline_model_parallel_size=2))
    initialize_model_parallel()

    assert(ps.model_parallel_is_initialized())
    assert(ps.get_model_parallel_group() is not None)
    assert(ps.get_tensor_model_parallel_group() is not None)
    assert(ps.get_pipeline_model_parallel_group() is not None)
    assert(ps.get_data_parallel_group() is not None)  
    assert(ps.get_embedding_group() is not None)  
    assert(ps.get_position_embedding_group() is not None)
    ps.destroy_model_parallel()

def test_pipeline_parallel_initializations():
    initialize_model_parallel(pipeline_model_parallel_size=2)
    assert(ps.get_pipeline_model_parallel_first_rank() == 0)
    assert(ps.get_data_parallel_src_rank() == rank)
    assert(ps.get_pipeline_model_parallel_next_rank() == 0 if rank == world_size - 1 else rank + 1)
    assert(ps.get_pipeline_model_parallel_prev_rank() == rank - 1 if rank > 0 else world_size - 1)
    ps.destroy_model_parallel()
 
def test_data_parallel_initializations():
    initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.get_data_parallel_src_rank() == rank)
    assert(ps.get_data_parallel_world_size() == world_size-1)
    assert(ps.get_data_parallel_rank() == 0)
    ps.destroy_model_parallel() 
    
def test_tensor_model_parellel_world_size():
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    assert(ps.get_tensor_model_parallel_world_size() == world_size)
    ps.set_tensor_model_parallel_world_size(None)
    assert(ps.get_tensor_model_parallel_world_size() == world_size)
    ps.destroy_model_parallel()


def test_pipeline_model_parallel_world_size():
    initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.get_pipeline_model_parallel_world_size() == world_size)
    ps.set_pipeline_model_parallel_world_size(None)
    assert(ps.get_pipeline_model_parallel_world_size() == world_size)
    ps.destroy_model_parallel()


def test_tensor_model_parallel_rank():
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    assert(ps.get_tensor_model_parallel_rank() == rank)
    ps.set_tensor_model_parallel_rank(None)
    assert(ps.get_tensor_model_parallel_rank() == rank)    
    ps.destroy_model_parallel()

def test_pipeline_model_parallel_rank():
    initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.get_pipeline_model_parallel_rank() == rank)
    ps.set_pipeline_model_parallel_rank(None)
    assert(ps.get_pipeline_model_parallel_rank() == rank)
    ps.destroy_model_parallel()
    
def test_is_pipeline_first_stage():
    initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.is_pipeline_first_stage(ignore_virtual=True) == (rank == 0))
    assert(ps.is_pipeline_first_stage() == (rank == 0))
    ps.destroy_model_parallel()

def test_is_pipeline_last_stage():
    initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.is_pipeline_last_stage(ignore_virtual=True) == (rank == world_size-1))
    assert(ps.is_pipeline_last_stage() == (rank == world_size-1))
    ps.destroy_model_parallel()


def test_virtual_pipeline_model_parallel_rank():
    initialize_model_parallel(pipeline_model_parallel_size=world_size)
    ps.set_virtual_pipeline_model_parallel_rank(rank)
    assert(ps.get_virtual_pipeline_model_parallel_rank() == rank)
    ps.destroy_model_parallel()

def test_get_tensor_model_parallel_src_rank():
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    assert(ps.get_tensor_model_parallel_src_rank() == ((rank // world_size) * world_size))
    ps.destroy_model_parallel()

"""
def test_get_virtual_pipeline_model_parallel_world_size():
    initialize_model_parallel(pipeline_model_parallel_size=world_size)
    ps.set_virtual_pipeline_model_parallel_rank(world_size)
    assert(ps.get_virtual_pipeline_model_parallel_world_size() == world_size)
    ps.destroy_model_parallel()

def test_is_rank_in_embedding_group():
    assert(ps.is_rank_in_embedding_group(ignore_virtual=True) == (rank in ps._EMBEDDING_GLOBAL_RANKS))
    if rank in ps._EMBEDDING_GLOBAL_RANKS:
        assert(ps.is_rank_in_embedding_group() == ps.is_pipeline_first_stage())
    elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
        assert(ps.is_rank_in_embedding_group() == ps.is_pipeline_last_stage())
    else:
        assert(ps.is_rank_in_embedding_group())

def test_is_rank_in_position_embedding_group():
    assert(ps.is_rank_in_position_embedding_group() == (rank in ps._POSITION_EMBEDDING_GLOBAL_RANKS))

def test_is_pipeline_stage_before_split():
    if world_size == 1:
        assert(ps.is_pipeline_stage_before_split())
    # TODO: Changes here for more than one world size
    assert(ps.is_pipeline_stage_before_split())

def test_is_pipeline_stage_after_split():
    if world_size == 1:
        assert(ps.is_pipeline_stage_after_split())
    # TODO: Changes here for more than one world size
    assert(ps.is_pipeline_stage_before_split())   

def test_is_pipeline_stage_at_split():
    assert(
        ps.is_pipeline_stage_at_split() == 
        (ps.is_pipeline_stage_before_split(rank) and ps.is_pipeline_stage_after_split(rank+1))
        )

def test_destroy_model_parallel():
    ps.destroy_model_parallel()
    assert(ps._MODEL_PARALLEL_GROUP is None)
"""