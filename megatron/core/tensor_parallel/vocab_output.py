from typing import Any, Callable, List, Optional

import torch
from torch.cuda.amp import custom_bwd, custom_fwd
import torch.distributed
from torch.nn.parameter import Parameter

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from ..transformer.utils import make_sharded_tensors_for_checkpoint
from ..utils import prepare_input_tensors_for_wgrad_compute
from .cross_entropy import VocabParallelCrossEntropy, VocabUtility
from .vocab_output_store import VocabOutputStore
from .layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from .mappings import copy_to_tensor_model_parallel_region
from .utils import divide

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

def _get_vocab_parallel_rank():
    return (
        get_pipeline_model_parallel_rank() * get_tensor_model_parallel_world_size()
        + get_tensor_model_parallel_rank()
    )

def _get_vocab_parallel_world_size():
    return (
        get_pipeline_model_parallel_world_size() * get_tensor_model_parallel_world_size()
    )

class _ForwardImpl(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        target,
        label_smoothing,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        fuse_forward_input_grad: bool = True,
        sync_allreduce: bool = False,
    ):
        assert label_smoothing == 0, "not yet supported"
        ctx.label_smoothing = label_smoothing
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.grad_output_buffer = grad_output_buffer
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.fuse_forward_input_grad = fuse_forward_input_grad
        ctx.sync_allreduce = sync_allreduce
        ctx.input_shape = input.shape

        reudce_group = (
            get_tensor_model_parallel_group()
            if not ctx.sync_allreduce
            else get_model_parallel_group()
        )

        if sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group()
            )
            total_input = all_gather_buffer
        else:
            total_input = input

        dummy_output = torch.zeros(
            total_input.shape[:2],
            device=torch.cuda.current_device(),
            dtype=total_input.dtype,
        )

        output = torch.matmul(total_input, weight.t())

        output, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
            output
        )
        # Only reduced across tensor parallel group, but not pipeline parallel group.
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=reudce_group
        )

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = output.size()[-1]
        rank = _get_vocab_parallel_rank()
        world_size = _get_vocab_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            output, target, logits_max, vocab_start_index, vocab_end_index
        )

        # All reduce is needed to get the chunks from other GPUs (tensor parallel group only).
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=reudce_group,
        )

        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=reudce_group,
        )

        if sync_allreduce:
            exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
                exp_logits, predicted_logits, sum_exp_logits
            )

            vocab_size = exp_logits.size(-1)

            # Store softmax, target-mask and masked-target for backward pass.
            ctx.save_for_backward(input, weight, exp_logits, target_mask, masked_target_1d, None, None)

            return loss

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        if fuse_forward_input_grad:
            # Calculate matrix multiplications for the backward pass.
            grad_input = exp_logits.to(weight.main_grad.dtype)

            softmax_grad_input = grad_input.matmul(weight)

            ground_truth_grad_input = weight[masked_target_1d].view(softmax_grad_input.shape)
            ground_truth_grad_input[target_mask] = 0.0

            if not sequence_parallel:
                torch.distributed.all_reduce(
                    softmax_grad_input,
                    op=torch.distributed.ReduceOp.SUM,
                    group=get_tensor_model_parallel_group(),
                )

                torch.distributed.all_reduce(
                    ground_truth_grad_input,
                    op=torch.distributed.ReduceOp.SUM,
                    group=get_tensor_model_parallel_group(),
                )
            else:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(total_input.size())
                dim_size[0] = dim_size[0] // world_size
                sub_softmax_grad_input = torch.empty(
                    dim_size, dtype=total_input.dtype, device=torch.cuda.current_device(), requires_grad=False,
                )
                sub_softmax_grad_input_handle = torch.distributed._reduce_scatter_base(
                    sub_softmax_grad_input, softmax_grad_input,
                    group=get_tensor_model_parallel_group(), async_op=True,
                )
                sub_ground_truth_grad_input = torch.empty(
                    dim_size, dtype=total_input.dtype, device=torch.cuda.current_device(), requires_grad=False,
                )
                sub_ground_truth_grad_input_handle = torch.distributed._reduce_scatter_base(
                    sub_ground_truth_grad_input, ground_truth_grad_input,
                    group=get_tensor_model_parallel_group(), async_op=True,
                )
                sub_softmax_grad_input_handle.wait()
                sub_ground_truth_grad_input_handle.wait()

            vocab_size = exp_logits.size(-1)

            ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

            ctx.save_for_backward(total_input, weight, exp_logits, target_mask, masked_target_1d,
                                sum_exp_logits.clone(), logits_max.clone())
        else:
            softmax_grad_input = None
            ground_truth_grad_input = None
            sub_softmax_grad_input = None
            sub_ground_truth_grad_input = None

            ctx.save_for_backward(input, weight, exp_logits, target_mask, masked_target_1d,
                                  sum_exp_logits.clone(), logits_max.clone())

        target_mask = target_mask.clone()
        torch.distributed.all_reduce(
            target_mask,
            torch.distributed.ReduceOp.PRODUCT,
            group=get_tensor_model_parallel_group(),
        )

        if not sequence_parallel:
            VocabOutputStore.forward_store(sum_exp_logits, logits_max, predicted_logits, target_mask,
                                            softmax_grad_input, ground_truth_grad_input)
        else:
            VocabOutputStore.forward_store(sum_exp_logits, logits_max, predicted_logits, target_mask,
                                            sub_softmax_grad_input, sub_ground_truth_grad_input)

        # This return value should be discarded.
        return dummy_output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *_):
        input, weight, softmax, target_mask, masked_target_1d, sum_exp_logits, logits_max = ctx.saved_tensors
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit

        reudce_group = (
            get_tensor_model_parallel_group()
            if not ctx.sync_allreduce
            else get_model_parallel_group()
        )

        if ctx.fuse_forward_input_grad:
            dummy_grad_input = torch.zeros(
                ctx.input_shape,
                device=torch.cuda.current_device(),
                dtype=grad_output.dtype,
            )

        if not ctx.sync_allreduce:
            global_sum_exp_logits, global_logits_max, grad_output = VocabOutputStore.backward_get()

            # Adjust the softmax based on sum_exp_logits.
            sum_exp_logits.div_(global_sum_exp_logits)
            logits_max -= global_logits_max
            sum_exp_logits.mul_(torch.exp(logits_max))
            alpha = sum_exp_logits
            softmax.mul_(alpha.unsqueeze(dim=-1))

        # Calculate the weight gradients only.

        (
            grad_2d,
            arange_1d,
            softmax_update,
            grad_input,
        ) = VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)

        grad_input = VocabParallelCrossEntropy.calculate_gradients(
            grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
        )

        grad_output = grad_input.to(weight.main_grad.dtype)

        wgrad_compute = True
        if grad_output_buffer is not None:
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if not ctx.fuse_forward_input_grad:
            if wgrad_compute:
                if ctx.sequence_parallel:
                    world_size = get_tensor_model_parallel_world_size()
                    dim_size = list(input.size())
                    dim_size[0] = dim_size[0] * world_size

                    all_gather_buffer = get_global_memory_buffer().get_tensor(
                        dim_size, input.dtype, "mpu"
                    )
                    handle = torch.distributed._all_gather_base(
                        all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
                    )

                    # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                    # gather is scheduled before the input gradient computation
                    input = all_gather_buffer

            grad_input = grad_output.matmul(weight)

            if ctx.sequence_parallel and wgrad_compute:
                handle.wait()

        if wgrad_compute:
            grad_output, input = prepare_input_tensors_for_wgrad_compute(
                grad_output, input
            )
        
        if not ctx.fuse_forward_input_grad:
            if ctx.allreduce_dgrad:
                # Asynchronous all-reduce
                handle = torch.distributed.all_reduce(
                    grad_input, group=reudce_group, async_op=True
                )
                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # all-reduce is scheduled before the weight gradient computation
            else:
                assert not ctx.sync_allreduce

            if ctx.sequence_parallel:
                assert not ctx.allreduce_dgrad
                dim_size = list(input.size())
                sub_grad_input = torch.empty(
                    dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
                )
                # reduce_scatter
                handle = torch.distributed._reduce_scatter_base(
                    sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
                )
                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # reduce scatter is scheduled before the weight gradient computation

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(input)
        
        if not ctx.fuse_forward_input_grad:
            if ctx.sequence_parallel:
                handle.wait()
                return sub_grad_input, grad_weight, None, None, None, None, None, None, None, None, None

            if ctx.allreduce_dgrad:
                handle.wait()
                return grad_input, grad_weight, None, None, None, None, None, None, None, None, None

        return dummy_grad_input, grad_weight, None, None, None, None, None, None, None, None, None

def _forward_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: Optional[int] = 0,
    fuse_forward_input_grad: bool = True,
    sync_allreduce: bool = False,

):
    args = [
        input,
        weight,
        target,
        label_smoothing,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        fuse_forward_input_grad,
        sync_allreduce,
    ]
    return _ForwardImpl.apply(*args)

class VocabParallelOutput(torch.nn.Module):
    """
    Computes the linear and softmax layer by splitting the vocab size across both pipeline and
    tensor parallel ranks.

    The output will be scaled by 1/sum_exp, and should be fixed later on after async all-reduce.
    """
    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        fuse_forward_input_grad: bool = True,
        sync_allreduce: bool = False,
    ):
        super(VocabParallelOutput, self).__init__()

        assert not (fuse_forward_input_grad and sync_allreduce)

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.embedding_activation_buffer = embedding_activation_buffer
        self.grad_output_buffer = grad_output_buffer
        self.fuse_forward_input_grad = fuse_forward_input_grad
        self.sync_allreduce = sync_allreduce

        world_size = _get_vocab_parallel_world_size()
        rank = _get_vocab_parallel_rank()
        self.world_size = world_size

        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:
            if config.use_cpu_initialization:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition, self.input_size, dtype=config.params_dtype
                    )
                )
                if config.perform_initialization:
                        self.master_weight = _initialize_affine_weight_cpu(
                            self.weight,
                            self.output_size,
                            self.input_size,
                            self.output_size_per_partition,
                            0,
                            init_method,
                            stride=1,
                            return_master_weight=False,
                            rank=rank,
                            world_size=world_size,
                        )
            else:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=1,
                        expert_parallel=False,
                    )

            setattr(self.weight, 'allreduce', True)
        else:
            self.weight = None
        
        self.register_parameter('bias', None)

        self.sequence_parallel = config.sequence_parallel

        self.allreduce_dgrad = world_size > 1 and not self.sequence_parallel

        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )
    
    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, label_smoothing=0.0):
        """Forward of VocabParallelOutput

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

            weight (optional): weight tensor to use, compulsory when
                skip_weight_param_allocation is True.

        Returns:
            - output
            - bias (None)
         """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to VocabParallelOutput forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )
        
        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"
        
        if self.world_size > 1:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        
        if self.config.defer_embedding_wgrad_compute:
            if (
                self.config.wgrad_deferral_limit == 0
                or len(self.embedding_activation_buffer) < self.config.wgrad_deferral_limit
            ):
                self.embedding_activation_buffer.append(input_parallel)
        
        # Matrix multiply.
        output = _forward_impl(
            input=input_parallel,
            weight=weight,
            target=labels,
            label_smoothing=label_smoothing,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            allreduce_dgrad=self.allreduce_dgrad,
            sequence_parallel=self.sequence_parallel,
            grad_output_buffer=(
                self.grad_output_buffer if self.config.defer_embedding_wgrad_compute else None
            ),
            wgrad_deferral_limit=(
                self.config.wgrad_deferral_limit
                if self.config.defer_embedding_wgrad_compute
                else None
            ),
            fuse_forward_input_grad=self.fuse_forward_input_grad,
            sync_allreduce=self.sync_allreduce,
        )

        return output, None
    
    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )

    def set_extra_state(self, state: Any):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None
