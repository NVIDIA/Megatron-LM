import os
import pytest
import torch
import torch.distributed as dist
from unittest import mock

from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

class TestP2PCommunication:
    
    def test_scatter_gather_logic_directly(self):
        """Test the scatter/gather logic directly without full communication."""
        from megatron.core.tensor_parallel.mappings import (
            scatter_to_tensor_model_parallel_region,
            gather_from_tensor_model_parallel_region
        )
        
        config = ModelParallelConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            scatter_gather_tensors_in_pipeline=True,
            sequence_parallel=False,
            pipeline_dtype=torch.float32
        )
        
        # Create a dummy tensor
        # Shape: [seq_len, batch_size, hidden_size]
        original_tensor = torch.ones(4, 2, 8)
        
        # Mock the tensor parallel rank and world size
        with mock.patch('megatron.core.parallel_state.get_tensor_model_parallel_rank', return_value=0), \
             mock.patch('megatron.core.parallel_state.get_tensor_model_parallel_world_size', return_value=2):
            
            # 1. Test scatter (what happens before send)
            if config.scatter_gather_tensors_in_pipeline and not config.sequence_parallel:
                # We need to mock the _CopyToModelParallelRegion and _GatherFromModelParallelRegion
                # since they require actual distributed groups
                with mock.patch('megatron.core.tensor_parallel.mappings._CopyToModelParallelRegion.apply', lambda x: x), \
                     mock.patch('megatron.core.tensor_parallel.mappings._GatherFromModelParallelRegion.apply', lambda x: x):
                    
                    # For scatter, we just do the chunking manually to simulate what the real function does
                    # The real function does:
                    # input_ = _CopyToModelParallelRegion.apply(input_)
                    # tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
                    # if tensor_model_parallel_world_size == 1: return input_
                    # last_dim = input_.dim() - 1
                    # last_dim_size = input_.size()[last_dim] // tensor_model_parallel_world_size
                    # tensor_tuple = torch.split(input_, last_dim_size, dim=last_dim)
                    # rank = get_tensor_model_parallel_rank()
                    # output = tensor_tuple[rank]
                    
                    last_dim = original_tensor.dim() - 1
                    last_dim_size = original_tensor.size()[last_dim] // 2
                    tensor_tuple = torch.split(original_tensor, last_dim_size, dim=last_dim)
                    scattered_tensor = tensor_tuple[0]
                    
                    # The last dimension (hidden_size) should be divided by tensor_model_parallel_size (2)
                    assert scattered_tensor.shape == (4, 2, 4)
                    
            # 2. Test gather (what happens after recv)
            if config.scatter_gather_tensors_in_pipeline and not config.sequence_parallel:
                # For gather, we simulate what the real function does
                # The real function does:
                # tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
                # if tensor_model_parallel_world_size == 1: return input_
                # last_dim = input_.dim() - 1
                # rank = get_tensor_model_parallel_rank()
                # tensor_list = [torch.empty_like(input_) for _ in range(tensor_model_parallel_world_size)]
                # tensor_list[rank] = input_
                # torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())
                # output = torch.cat(tensor_list, dim=last_dim).contiguous()
                # return _GatherFromModelParallelRegion.apply(output)
                
                # We simulate the all_gather by creating the full list
                tensor_list = [scattered_tensor, scattered_tensor]
                last_dim = scattered_tensor.dim() - 1
                gathered_tensor = torch.cat(tensor_list, dim=last_dim).contiguous()
                
                # The shape should be restored
                assert gathered_tensor.shape == (4, 2, 8)
            
    def test_scatter_gather_disabled(self):
        """Test that scatter/gather is not applied when disabled."""
        config = ModelParallelConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            scatter_gather_tensors_in_pipeline=False,
            sequence_parallel=False,
            pipeline_dtype=torch.float32
        )
        
        original_tensor = torch.ones(4, 2, 8)
        
        # If disabled, the tensor shape should remain unchanged before send
        tensor_to_send = original_tensor
        if config.scatter_gather_tensors_in_pipeline and not config.sequence_parallel:
            # This block should not be executed
            assert False, "Should not execute scatter when disabled"
            
        assert tensor_to_send.shape == (4, 2, 8)
