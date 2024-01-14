from tools.deepspeed.ops.op_builder.quantizer import CUDAQuantizer, QuantizerBuilder
import torch

class QuantizationHelper:
    def __init__(self, quantized_weights=True, 
                 weight_quantization_bits = 4, 
                 wq_group_size=2048, 
                 quantized_gradients=True, 
                 gradeint_quantization_bits=8,
                 gq_group_size=2048):

        self.quantized_weights = quantized_weights
        self.weight_quantization_bits = weight_quantization_bits
        self.wq_group_size = wq_group_size
        self.quantized_gradients = quantized_gradients
        self.gradeint_quantization_bits = gradeint_quantization_bits
        self.gq_group_size = gq_group_size
        self.cuda_quantizer = None

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
            received_buffer.data = self.cuda_quantizer.dequantize(quantized_weight_tensor, scale, quantization_bits=self.weight_quantization_bits)
            return received_buffer
        else:
            return self.cuda_quantizer.dequantize(quantized_weight_tensor, scale, quantization_bits=self.weight_quantization_bits)

    # def quantize_reduce_gradients(self, tensor, groups= None):

    #     global_world_size = ps.get_data_parallel_world_size()
    #     # gpus_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', 0))
    #     # local_world_size = gpus_per_node // ps.get_tensor_model_parallel_world_size() TODO hybird with tensor parallel
    #     # intra_quant_group = max(tensor.shape[0], tensor.shape[1], global_world_size)
    #     group_size = 128
    #     intra_quant_group = max(math.ceil(tensor.numel() / group_size), global_world_size)
    #     local_world_size = gpus_per_node
    #     num_nodes = global_world_size // local_world_size
    #     assert num_nodes > 1, 'number of nodes should > 1'
    #     assert tensor.numel() % local_world_size == 0, 'tensor should be padded to multpiles of local_world_size'
    #     assert tensor.numel() % num_nodes == 0, 'tensor should be padded to multpiles of num_nodes'
    #     assert tensor.numel() % group_size ==0, 'tensor should be padded to multpilies of group size'
    #     inter_quant_group = intra_quant_group // local_world_size
    #     this_rank = ps.get_data_parallel_rank()
    #     intra_idx = int(this_rank / local_world_size)
    #     inter_idx = this_rank % local_world_size
    #     print(f"global world size: {global_world_size}, gpus per node: {gpus_per_node}, intra quant group: {intra_quant_group}, rank: {this_rank},  intra_idx: {intra_idx}, inter_idx: {inter_idx}")

    #     quantizer_module = QuantizerBuilder().load()

    #     first_quant, intra_q_scales = quantizer_module.swizzle_quant(tensor, intra_quant_group, 8,
    #                                                                         quantizer_module.Symmetric, 1, num_nodes,
    #                                                                         local_world_size)
    #     print_rank_0(f"tensor shape: {tensor.shape}, intra_quant_int4 shape: {first_quant.shape}, intra_quant_type: {first_quant.dtype}, intra_q_scales shape: {intra_q_scales.shape}")

    #     local_output = torch.empty_like(first_quant)
    #     scale_output = torch.empty_like(intra_q_scales)
    #     assert first_quant.shape[0] % local_world_size == 0, "input tensor must divide equally by local_world_size"
    #     local_group = groups[f'local_{intra_idx}']
    #     print_rank_0(f'local group size: {torch.distributed.get_world_size(group=local_group)}')
    #     all_to_all_single(local_output, first_quant, group=groups[f'local_{intra_idx}'])
    #     all_to_all_single(scale_output, intra_q_scales, group=groups[f'local_{intra_idx}'])
    #     global_input_tensor, global_scales = quantizer_module.quantized_reduction(
    #         local_output, scale_output, intra_quant_group, inter_quant_group, 8, quantizer_module.Symmetric,
    #         local_world_size)
    #     print_rank_0(f"local_output shape: {local_output.shape}, global_scales shape: {global_scales.shape}")

    #     global_output = torch.empty_like(global_input_tensor)
    #     global_scale_output = torch.empty_like(global_scales)
    #     all_to_all_single(global_output, global_input_tensor, group=groups[f'global_{inter_idx}'])
    #     all_to_all_single(global_scale_output, global_scales, group=groups[f'global_{inter_idx}'])
    #     final_output = quantizer_module.dequantize(global_output, global_scale_output, global_scale_output.numel(),
    #                                                 8, quantizer_module.Symmetric)
    #     return (sum(list(final_output.chunk(num_nodes)))).view(-1)