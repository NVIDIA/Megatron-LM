import torch

from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core import parallel_state
def synchronize_params_across_all_ranks(common_inference_params: CommonInferenceParams):
    values = [
            common_inference_params.use_greedy,
            common_inference_params.temperature,
            common_inference_params.top_k,
            common_inference_params.top_p,
            common_inference_params.return_log_probs,
            common_inference_params.num_tokens_to_generate,
            ]
    size = len(values)
    common_inference_params_tensor = synchronize_list_across_all_ranks(size, values, dtype=torch.float32)

    if torch.distributed.get_rank() != 0:
        # TODO: Should change this . Might not be best to convert them to object
        common_inference_params = CommonInferenceParams(*common_inference_params_tensor.tolist())
        common_inference_params.use_greedy = bool(common_inference_params.use_greedy)
        common_inference_params.return_log_probs = bool(common_inference_params.return_log_probs)

    return common_inference_params

def synchronize_list_across_all_ranks(size, list_values = None, dtype = torch.float32):
    tensor = None
    if torch.distributed.get_rank() == 0:
        tensor = torch.tensor(list_values, dtype=dtype, device = torch.cuda.current_device())
    tensor = synchronize_tensor_across_all_ranks(size, dtype = dtype, tensor = tensor)
    return tensor


def synchronize_tensor_across_all_ranks(size, dtype, tensor=None):
    if torch.distributed.get_rank() == 0:
        assert tensor.is_contiguous()
    else:
        tensor = torch.empty(size, dtype = dtype, device = torch.cuda.current_device())
    torch.distributed.broadcast(tensor, src=0)
    return tensor

def _is_cuda(tensor):
    """Check if a tensor is not none and is cuda."""
    assert tensor is not None
    assert tensor.is_cuda

def copy_from_last_to_first_pipeline_stage(size, dtype, tensor=None):
    """Copy tensor values from last stage into the first stage.
    Note that the input tensor is updated in place."""

    is_last_stage = parallel_state.is_pipeline_last_stage()
    is_first_stage = parallel_state.is_pipeline_first_stage()

    # Only first and last stage pipeline stages need to be involved.
    if is_last_stage or is_first_stage:
        _is_cuda(tensor)
        is_contiguous = tensor.is_contiguous()
        src = parallel_state.get_pipeline_model_parallel_last_rank()
        group = parallel_state.get_embedding_group()
        if is_contiguous:
            tensor_ = tensor
        else:
            if is_last_stage:
                tensor_ = tensor.contiguous()
            else:
                tensor_ = torch.empty(size,
                                      dtype=dtype,
                                      device=torch.cuda.current_device())
        # Broadcast from last stage into the first stage.
        torch.distributed.broadcast(tensor_, src, group)
        # Update the first stage tensor
        if is_first_stage and not is_contiguous:
            tensor[...] = tensor_

# TODO: Can use utilites from mcore itself I think
def recv_from_prev_pipeline_rank_(recv_buffer=None):
    """Receive from previous pipeline stage and update the
    input buffer inplace."""
    recv_prev_op = torch.distributed.P2POp(
        torch.distributed.irecv, recv_buffer,
        parallel_state.get_pipeline_model_parallel_prev_rank())
    reqs = torch.distributed.batch_isend_irecv([recv_prev_op])
    for req in reqs:
        req.wait()
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()

# TODO: Can use utilites from mcore itself I think
def send_to_next_pipeline_rank(tensor=None):
    """Send output to the next pipeline stage."""
    send_next_op = torch.distributed.P2POp(
        torch.distributed.isend, tensor,
        parallel_state.get_pipeline_model_parallel_next_rank())
    reqs = torch.distributed.batch_isend_irecv([send_next_op])
    for req in reqs:
        req.wait()
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()