import torch
import megatron.core.parallel_state as ps
import pytest
from tests.unit_tests.test_utilities import Utils
import os 

rank = Utils.rank
world_size = Utils.world_size

def test_initialize__and_destroy_model_parallel():
    with pytest.raises(AssertionError):
        assert(ps.initialize_model_parallel())
    Utils.initialize_distributed()
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(tensor_model_parallel_size=2*world_size))
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(pipeline_model_parallel_size=2*world_size))
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(pipeline_model_parallel_size=world_size, tensor_model_parallel_size=world_size))
    with pytest.raises(RuntimeError):
        assert(ps.initialize_model_parallel(virtual_pipeline_model_parallel_size=2))
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    assert(ps.model_parallel_is_initialized())
    assert(ps.get_model_parallel_group() is not None)
    assert(ps.get_tensor_model_parallel_group() is not None)
    assert(ps.get_pipeline_model_parallel_group() is not None)
    assert(ps.get_data_parallel_group() is not None)  
    Utils.destroy_model_parallel()
    assert(ps._MODEL_PARALLEL_GROUP is None)

def test_pipeline_parallel_initializations():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)
    assert(ps.get_pipeline_model_parallel_first_rank() == rank % 2 )
    assert(ps.get_data_parallel_src_rank() == rank)
    assert(ps.get_pipeline_model_parallel_next_rank() == ((rank + 2) % world_size))
    assert(ps.get_pipeline_model_parallel_prev_rank() == ((rank - 2) % world_size))
    Utils.destroy_model_parallel()

def test_data_parallel_initializations():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.get_data_parallel_src_rank() == rank)
    assert(ps.get_data_parallel_world_size() == 1)
    assert(ps.get_data_parallel_rank() == 0)
    Utils.destroy_model_parallel()
    

def test_tensor_model_parellel_world_size():
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
    assert(ps.get_tensor_model_parallel_world_size() == world_size)
    ps.set_tensor_model_parallel_world_size(None)
    assert(ps.get_tensor_model_parallel_world_size() == world_size)
    Utils.destroy_model_parallel()
    

def test_pipeline_model_parallel_world_size():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.get_pipeline_model_parallel_world_size() == world_size)
    ps.set_pipeline_model_parallel_world_size(None)
    assert(ps.get_pipeline_model_parallel_world_size() == world_size)
    Utils.destroy_model_parallel()    
    

def test_tensor_model_parallel_rank():
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
    assert(ps.get_tensor_model_parallel_rank() == rank)
    ps.set_tensor_model_parallel_rank(None)
    assert(ps.get_tensor_model_parallel_rank() == rank)    
    Utils.destroy_model_parallel()    
    

def test_pipeline_model_parallel_rank():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.get_pipeline_model_parallel_rank() == rank)
    ps.set_pipeline_model_parallel_rank(None)
    assert(ps.get_pipeline_model_parallel_rank() == rank)
    Utils.destroy_model_parallel()
    

def test_is_pipeline_first_stage():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.is_pipeline_first_stage(ignore_virtual=True) == (rank == 0))
    assert(ps.is_pipeline_first_stage() == (rank == 0))
    Utils.destroy_model_parallel()
    

def test_is_pipeline_last_stage():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    assert(ps.is_pipeline_last_stage(ignore_virtual=True) == (rank == world_size-1))
    assert(ps.is_pipeline_last_stage() == (rank == world_size-1))
    Utils.destroy_model_parallel()
    

def test_virtual_pipeline_model_parallel_rank():
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size)
    ps.set_virtual_pipeline_model_parallel_rank(rank)
    assert(ps.get_virtual_pipeline_model_parallel_rank() == rank)
    Utils.destroy_model_parallel()
    

def test_get_tensor_model_parallel_src_rank():
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
    assert(ps.get_tensor_model_parallel_src_rank() == ((rank // world_size) * world_size))
    Utils.destroy_model_parallel() 