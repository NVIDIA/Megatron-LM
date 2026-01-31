from abc import abstractmethod
import torch
import math
from enum import Enum
from typing import Dict, Any
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank

_GLOBAL_TT_FLAGS = None
_GLOBAL_TENSOR_TRACERS = None
_GLOBAL_REPORT = lambda name, args, tensor: None
_GLOBAL_COMPRESSOR = None
_GLOBAL_HOOK_MANAGER = None

def _set_tensor_tracers():
    global _GLOBAL_TENSOR_TRACERS
    _GLOBAL_TENSOR_TRACERS = TensorTracers()

def _set_tt_flags(args):
    global _GLOBAL_TT_FLAGS
    _GLOBAL_TT_FLAGS = TTFlags(args)

def _set_tt_hook_manager(args, model):
    global _GLOBAL_HOOK_MANAGER
    _GLOBAL_HOOK_MANAGER = TTHookManager(args, model)

def _set_compressor():
    global _GLOBAL_COMPRESSOR
    _GLOBAL_COMPRESSOR=DefaultCompressor()

def set_report(func):
    global _GLOBAL_REPORT
    _GLOBAL_REPORT = func

def unset_report():
    global _GLOBAL_REPORT
    _GLOBAL_REPORT = lambda name, args, tensor: None

def get_tensor_tracers():
    return _GLOBAL_TENSOR_TRACERS

def get_tt_flags():
    return _GLOBAL_TT_FLAGS

def get_compressor():
    return _GLOBAL_COMPRESSOR

def get_report():
    return _GLOBAL_REPORT

class FlagType(Enum):
    INVALID_FLAG = 0
    QKV_mat_mul = 1
    RawAttentionScore_mat_mul = 2
    ContextLayer_mat_mul = 3
    MLP1_mat_mul = 4
    MLP2_mat_mul = 5
    AttentionOutput_mat_mul = 6
    HiddenStates = 7

class AbstractCompressor:
    def __init__(self):
        pass
    @abstractmethod
    def set_by_configs(self, configs: Dict[str, Any]):
        pass
    @abstractmethod
    def compress_one_rank(self, name, data):
        pass
    @abstractmethod
    def compress(self, name, data):
        pass

class DefaultCompressor(AbstractCompressor):
    def __init__(self):
        self.configs = {
            "QKV": {
                "pixels": 96,
                "method": "data.mean(dim=-1)"
            },
            "MLP": {
                "pixels": 64,
                "method": "data.mean(dim=-1)"
            }
        }

    def set_by_configs(self, configs: Dict[str, Any]):
        self.configs = configs

    def compress_tensor(self, data_in, pixels, method):
        B, S, F = data_in.shape
        chunk_size = math.ceil(F / pixels)
        padded_len = chunk_size * pixels
        padded_data = torch.nn.functional.pad(data_in, (0, padded_len - F))
        data_for_eval = padded_data.reshape(B, S, pixels, chunk_size)
        try:
            compressed = eval(method, {}, {"data": data_for_eval})
        except Exception as e:
            print(f"Error in compressing tensor with method '{method}': {e}")
            compressed = data_for_eval.mean(dim=-1)
        return compressed

    def compress_1d_tensor(self, data_in, pixels, method):
        B, S, F = data_in.shape
        chunk_size = math.ceil(F / pixels)
        padded_len = chunk_size * pixels
        padded_data = torch.nn.functional.pad(data_in, (0, padded_len - F))
        data_for_eval = padded_data.reshape(B, S, pixels, chunk_size)
        try:
            compressed = eval(method, {}, {"data": data_for_eval}).flatten()
        except Exception as e:
            print(f"Error in compressing tensor with method '{method}': {e}")
            compressed = data_for_eval.mean(dim=-1).flatten()  # Fallback to mean if eval fails
        return compressed

    def compress_one_rank(self, flag_type, data):
        if flag_type == FlagType.QKV_mat_mul:
            return self.compress_tensor(data, self.configs["QKV"]["pixels"], self.configs["QKV"]["method"])
        elif flag_type == FlagType.MLP1_mat_mul or flag_type == FlagType.MLP2_mat_mul or flag_type == FlagType.ContextLayer_mat_mul:
            return self.compress_tensor(data, self.configs["MLP"]["pixels"], self.configs["MLP"]["method"])
        return data

    def compress(self, name, data):
        flag_type = name[1]
        if flag_type == FlagType.QKV_mat_mul:
            n = data.shape[1]; return True, [n], self.compress_1d_tensor(data, self.configs["QKV"]["pixels"], self.configs["QKV"]["method"])
        elif flag_type == FlagType.RawAttentionScore_mat_mul:
            np, n, m = data.shape[1], data.shape[2], data.shape[3]; return True, [np, n, m], data[:, :, :, :].flatten()
        elif flag_type == FlagType.MLP1_mat_mul or flag_type == FlagType.MLP2_mat_mul or flag_type == FlagType.ContextLayer_mat_mul:
            n = data.shape[1]; return True, [n], self.compress_1d_tensor(data, self.configs["MLP"]["pixels"], self.configs["MLP"]["method"])
        return False, [], torch.tensor([])

class ProjectionCompressor(AbstractCompressor):
    def __init__(self):
        pass

    def set_by_configs(self, configs: Dict[str, Any]):
        pass
    
    def compress_one_rank(self, name, data):
        return data

    def compress(self, name, data):
        return False, [], torch.tensor([])

class TensorTracers: # simplified as TT
    def __init__(self) -> None: pass

    def report(self, name, tensor_data):
        valid, comp_args, compressed_tensor = get_compressor().compress(name, tensor_data)
        assert valid
        get_report()(name, comp_args, compressed_tensor)

class TTFlags:
    """Global flags to record the intermediate results of the model."""

    def __init__(self, args):
        self.num_layers = args.num_layers
        self.flags: Dict[FlagType, Dict[int, bool]] = {
            FlagType.INVALID_FLAG: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.QKV_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.RawAttentionScore_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.ContextLayer_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.MLP1_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.MLP2_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.AttentionOutput_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.HiddenStates: {i: False for i in range(1, self.num_layers + 1)},
        }
        self.should_trace = True

    def get_flag(self, flag_type: FlagType, layer_index: int) -> bool:
        return self.should_trace and self.flags.get(flag_type, {}).get(layer_index, False)

    def set_by_configs(self, configs: Dict[str, Any]):
        val = True if configs.get("QKV_mat_mul", "False").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.QKV_mat_mul][i] = val
        
        val = True if configs.get("RawAttentionScore_mat_mul", "False").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.RawAttentionScore_mat_mul][i] = val
        
        val = True if configs.get("ContextLayer_mat_mul", "False").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.ContextLayer_mat_mul][i] = val
        
        val = True if configs.get("MLP1_mat_mul", "True").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.MLP1_mat_mul][i] = val
        
        val = True if configs.get("MLP2_mat_mul", "True").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.MLP2_mat_mul][i] = val
        
        val = True if configs.get("AttentionOutput_mat_mul", "False").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.AttentionOutput_mat_mul][i] = val

        val = True if configs.get("HiddenStates", "False").lower() == "true" else False
        for i in range(1, self.num_layers + 1):
            self.flags[FlagType.HiddenStates][i] = val

class TTHookManager:
    def __init__(self, args, model) -> None:
        self.hooks = []
        # the type of model should be GPTModel
        from megatron.core.models.gpt import GPTModel
        model = model[0].module.module
        assert isinstance(model, GPTModel), f"{model}, {type(model)}"
        def generate_hook_transpose_col(flag_type: FlagType, layer_number: int):
            def hook(module, input, output):
                if get_tt_flags().get_flag(flag_type, layer_number):
                    device = torch.cuda.current_device()
                    world_size = get_tensor_model_parallel_world_size()
                    rank = get_tensor_model_parallel_rank()

                    tensor_data = output[0].detach()
                    tensor_data = get_compressor().compress_one_rank(flag_type, tensor_data)
                    tensor_data_cont = tensor_data.contiguous()
                    if rank == 0:
                        tensor_list = [torch.zeros_like(tensor_data_cont, dtype=tensor_data_cont.dtype, device=device) for _ in range(world_size)]
                    else:
                        tensor_list = None
                    if world_size > 1:
                        torch.distributed.gather(tensor_data_cont, tensor_list, dst=0, group=get_tensor_model_parallel_group())
                    else:
                        tensor_list = [tensor_data_cont]
                    
                    if rank == 0:
                        aggregated_tensor = None

                        if flag_type == FlagType.QKV_mat_mul:
                            if world_size > 1:
                                tensor_list0, tensor_list1, tensor_list2 = [], [], []
                                for id_rank in range(world_size):
                                    chunks = torch.chunk(tensor_list[id_rank], 3, dim=2)
                                    tensor_list0.append(chunks[0])
                                    tensor_list1.append(chunks[1])
                                    tensor_list2.append(chunks[2])
                                tensor0 = torch.cat(tensor_list0, dim=2)
                                tensor1 = torch.cat(tensor_list1, dim=2)
                                tensor2 = torch.cat(tensor_list2, dim=2)
                                aggregated_tensor = torch.cat([tensor0, tensor1, tensor2], dim=2)
                            else:
                                aggregated_tensor = tensor_data_cont
                        else:
                            aggregated_tensor = torch.cat(tensor_list, dim=2)
                    
                        get_tensor_tracers().report((layer_number, flag_type), aggregated_tensor.transpose(0, 1))
            return hook

        def generate_hook_transpose_row(flag_type: FlagType, layer_number: int):
            def hook(module, input, output):
                if get_tt_flags().get_flag(flag_type, layer_number):
                    device = torch.cuda.current_device()
                    world_size = get_tensor_model_parallel_world_size()
                    rank = get_tensor_model_parallel_rank()

                    if args.sequence_parallel:
                        tensor_data = output[0].detach()
                        tensor_data = get_compressor().compress_one_rank(flag_type, tensor_data)
                        tensor_data_cont = tensor_data.contiguous()
                        if rank == 0:
                            tensor_list = [torch.zeros_like(tensor_data_cont, dtype=tensor_data_cont.dtype, device=device) for _ in range(world_size)]
                        else:
                            tensor_list = None
                        if world_size > 1:
                            torch.distributed.gather(tensor_data_cont, tensor_list, dst=0, group=get_tensor_model_parallel_group())
                        else:
                            tensor_list = [tensor_data_cont]
                        
                        if rank == 0:
                            aggregated_tensor = torch.cat(tensor_list, dim=0)
                            get_tensor_tracers().report((layer_number, flag_type), aggregated_tensor.transpose(0, 1))
                    else:
                        if rank == 0:
                            tensor_data = output[0].detach()
                            tensor_data = get_compressor().compress_one_rank(flag_type, tensor_data)
                            get_tensor_tracers().report((layer_number, flag_type), tensor_data.transpose(0, 1))
            return hook

        def generate_hook_attn(flag_type: FlagType, layer_number: int):
            def hook(module, input, output):
                if get_tt_flags().get_flag(flag_type, layer_number):
                    device = torch.cuda.current_device()
                    world_size = get_tensor_model_parallel_world_size()
                    rank = get_tensor_model_parallel_rank()

                    tensor_data = output.detach()
                    tensor_data = get_compressor().compress_one_rank(flag_type, tensor_data)
                    tensor_data_cont = tensor_data.contiguous()
                    if rank == 0:
                        tensor_list = [torch.zeros_like(tensor_data_cont, dtype=tensor_data_cont.dtype, device=device) for _ in range(world_size)]
                    else:
                        tensor_list = None
                    if world_size > 1:
                        torch.distributed.gather(tensor_data_cont, tensor_list, dst=0, group=get_tensor_model_parallel_group())
                    else:
                        tensor_list = [tensor_data_cont]
                    
                    if rank == 0:
                        aggregated_tensor = torch.cat(tensor_list, dim=1)
                        get_tensor_tracers().report((layer_number, flag_type), aggregated_tensor)
            return hook
        for layer in range(model.decoder.num_layers_per_pipeline_rank):
            global_layer_number = model.decoder.layers[layer].layer_number
            self.hooks.append(model.decoder.layers[layer].self_attention.linear_qkv.register_forward_hook(generate_hook_transpose_col(FlagType.QKV_mat_mul, global_layer_number))) # Col, not gather_output
            self.hooks.append(model.decoder.layers[layer].mlp.linear_fc1.register_forward_hook(generate_hook_transpose_col(FlagType.MLP1_mat_mul, global_layer_number))) # Col, not gather_output
            self.hooks.append(model.decoder.layers[layer].mlp.linear_fc2.register_forward_hook(generate_hook_transpose_row(FlagType.MLP2_mat_mul, global_layer_number))) # Row
            self.hooks.append(model.decoder.layers[layer].self_attention.register_forward_hook(generate_hook_transpose_row(FlagType.AttentionOutput_mat_mul, global_layer_number))) # Row
            self.hooks.append(model.decoder.layers[layer].register_forward_hook(generate_hook_transpose_row(FlagType.HiddenStates, global_layer_number))) # Row
            self.hooks.append(model.decoder.layers[layer].self_attention.core_attention.scale_mask_softmax.register_forward_hook(generate_hook_attn(FlagType.RawAttentionScore_mat_mul, global_layer_number))) # Raw Attention Scores, Special
            self.hooks.append(model.decoder.layers[layer].self_attention.core_attention.register_forward_hook(generate_hook_transpose_col(FlagType.ContextLayer_mat_mul, global_layer_number))) # Col, not gather_output

'''
For ColumnParallelLinear:
1. If gather_output, we do not do all gather
2. If not gather_output, we do all gather
For RowParallelLinear:
1. If sequence_parallel, we do all gather
2. If not sequence_parallel, we do not do all gather
'''