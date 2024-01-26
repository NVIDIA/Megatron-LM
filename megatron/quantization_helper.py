from tools.deepspeed.ops.op_builder.quantizer import CUDAQuantizer, QuantizerBuilder
import torch
from torch.distributed import ProcessGroup, all_to_all_single
import math
import os

class QuantizationHelper:
    def __init__(self, quantized_weights=True, 
                 weight_quantization_bits = 4, 
                 wq_group_size=2048, 
                 quantized_gradients=True, 
                 gradeint_quantization_bits=8,
                 gq_group_size=2048,
                 data_parallel_group: torch.distributed.ProcessGroup = None,
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1,
                 ):

        self.quantized_weights = quantized_weights
        self.weight_quantization_bits = weight_quantization_bits
        self.wq_group_size = wq_group_size
        self.quantized_gradients = quantized_gradients
        self.gradeint_quantization_bits = gradeint_quantization_bits
        self.gq_group_size = gq_group_size
        self.cuda_quantizer = None
        self.data_parallel_group = data_parallel_group
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        if self.quantized_gradients:
            self.set_local_all_to_all_group()
            self.gradient_quantization_module = QuantizerBuilder().load()
    def quantize_gather_weights(self, weight_tensor):
        """
        Quantize the given tensor using CUDAQuantizer.

        Args:
            tensor (torch.Tensor): The tensor to be quantized.

        Returns:
            quantized_param: The quantized tensor.
            scales: quantized scales
        """
        if self.cuda_quantizer is None:
            self.cuda_quantizer = CUDAQuantizer()
            self.cuda_quantizer.target_group_size = self.wq_group_size
        quantized_param, scales = self.cuda_quantizer.quantize(weight_tensor, quantization_bits=self.weight_quantization_bits)
        return quantized_param, scales

    def dequantize_gather_weights(self, quantized_weight_tensor, scale, received_buffer=None):
        """
        Dequantize the given tensor using CUDAQuantizer.

        Args:
            quantized_tensor (torch.Tensor): The tensor to be dequantized.
            scale (float): Scale factor for dequantization.

        Returns:
            torch.Tensor: The dequantized tensor.
        """
        if self.cuda_quantizer is None:
            self.cuda_quantizer = CUDAQuantizer()
            self.cuda_quantizer.target_group_size = self.wq_group_size
        if received_buffer is not None:
            received_buffer.copy_(self.cuda_quantizer.dequantize(quantized_weight_tensor, scale, quantization_bits=self.weight_quantization_bits))
            return received_buffer
        else:
            return self.cuda_quantizer.dequantize(quantized_weight_tensor, scale, quantization_bits=self.weight_quantization_bits)

    def quantize_reduce_gradients(self, tensor):
        groups = self.all2all_process_group
        global_world_size = torch.distributed.get_world_size(group=self.data_parallel_group)
        group_size = self.gq_group_size
        intra_quant_group = max(math.ceil(tensor.numel() / group_size), global_world_size)
        local_world_size = self.local_world_size
        num_nodes = self.num_nodes
        assert num_nodes > 1, 'number of nodes should > 1'
        assert tensor.numel() % local_world_size == 0, 'tensor should be padded to multpiles of local_world_size'
        assert tensor.numel() % num_nodes == 0, 'tensor should be padded to multpiles of num_nodes'
        assert tensor.numel() % group_size ==0, 'tensor should be padded to multpilies of group size'
        inter_quant_group = intra_quant_group // local_world_size
        this_rank = torch.distributed.get_rank(
            group=self.data_parallel_group
        )
        intra_idx = int(this_rank / local_world_size)
        inter_idx = this_rank % local_world_size
        # print(f"global world size: {global_world_size}, local_world_size: {local_world_size}, num_nodes: {num_nodes}, intra quant group: {intra_quant_group}, rank: {this_rank},  intra_idx: {intra_idx}, inter_idx: {inter_idx}")

        quantizer_module = self.gradient_quantization_module

        first_quant, intra_q_scales = quantizer_module.swizzle_quant(tensor, intra_quant_group, self.gradeint_quantization_bits,
                                                                            quantizer_module.Symmetric, 1, num_nodes,
                                                                            local_world_size)
        # print(f"tensor shape: {tensor.shape}, intra_quant_int4 shape: {first_quant.shape}, intra_quant_type: {first_quant.dtype}, intra_q_scales shape: {intra_q_scales.shape}")

        local_output = torch.empty_like(first_quant)
        scale_output = torch.empty_like(intra_q_scales)
        assert first_quant.shape[0] % local_world_size == 0, "input tensor must divide equally by local_world_size"
        local_group = groups[f'local_{intra_idx}']
        # print_rank_0(f'local group size: {torch.distributed.get_world_size(group=local_group)}')
        all_to_all_single(local_output, first_quant, group=groups[f'local_{intra_idx}'])
        all_to_all_single(scale_output, intra_q_scales, group=groups[f'local_{intra_idx}'])

        global_input_tensor, global_scales = quantizer_module.quantized_reduction(
            local_output, scale_output, intra_quant_group, inter_quant_group, self.gradeint_quantization_bits, quantizer_module.Symmetric,
            local_world_size)
        # print_rank_0(f"local_output shape: {local_output.shape}, global_scales shape: {global_scales.shape}")

        global_output = torch.empty_like(global_input_tensor)
        global_scale_output = torch.empty_like(global_scales)
        all_to_all_single(global_output, global_input_tensor, group=groups[f'global_{inter_idx}'])
        all_to_all_single(global_scale_output, global_scales, group=groups[f'global_{inter_idx}'])
        final_output = quantizer_module.dequantize(global_output, global_scale_output, global_scale_output.numel(),
                                                    self.gradeint_quantization_bits, quantizer_module.Symmetric)
        return (sum(list(final_output.chunk(num_nodes)))).view(-1)

    def set_local_all_to_all_group(self):
        assert torch.distributed.is_initialized(), 'dist is not initialized'
        all_to_all_group = {}
        data_parallel_world_size = torch.distributed.get_world_size(group=self.data_parallel_group)
        tensor_model_parallel_size = self.tensor_parallel_size
        pipeline_model_parallel_size =self.pipeline_parallel_size
        gpus_per_node = int(os.environ['LOCAL_WORLD_SIZE'])
        local_dp_size = gpus_per_node // (tensor_model_parallel_size * pipeline_model_parallel_size) # data parallel size in a node
        if local_dp_size < 1:
            local_dp_size = 1
        self.local_world_size = local_dp_size
        num_local = data_parallel_world_size // local_dp_size
        self.num_nodes = num_local
        # TODO only support data parallel across nodes, remove this assert in the future
        assert data_parallel_world_size > 1, 'data parallel size must > 1, cannot initialize All-To-All'
        assert num_local > 1, 'num_nodes<2 cannot initialize All-To-All'
        for i in range(num_local):
            local_rank = [j + local_dp_size * i for j in range(local_dp_size)]
            all_to_all_group[f"local_{i}"] = torch.distributed.new_group(ranks=local_rank)

        for i in range(local_dp_size):
            cur_rank = []
            for j in range(num_local):
                cur_rank.append(i + j * local_dp_size)
            all_to_all_group[f"global_{i}"] = torch.distributed.new_group(ranks=cur_rank)
        self.all2all_process_group = all_to_all_group
    def set_gradient_quantization(self, quantize_gradients: bool):
        self.quantized_gradients = quantize_gradients