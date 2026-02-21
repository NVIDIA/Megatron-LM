import logging
import math
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

logger = logging.getLogger(__name__)

NameTuple = Tuple[int, "FlagType"]
ReportFn = Callable[[NameTuple, list[int], torch.Tensor], None]

_GLOBAL_TT_FLAGS: Optional["TTFlags"] = None
_GLOBAL_TENSOR_TRACERS: Optional["TensorTracers"] = None
_GLOBAL_COMPRESSOR: Optional[Dict["FlagType", "AbstractCompressor"]] = None
_GLOBAL_HOOK_MANAGER: Optional["TTHookManager"] = None


def _noop_report(name: NameTuple, args: list[int], tensor: torch.Tensor) -> None:
    return


_GLOBAL_REPORT: ReportFn = _noop_report


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
    _GLOBAL_COMPRESSOR = {flag_type: EmptyCompressor({}) for flag_type in FlagType}


def set_report(func):
    """Set the global tensor report callback."""
    global _GLOBAL_REPORT
    _GLOBAL_REPORT = func


def unset_report():
    """Reset the global tensor report callback to a no-op."""
    global _GLOBAL_REPORT
    _GLOBAL_REPORT = _noop_report


def get_tensor_tracers():
    """Return the global tensor tracer instance, if initialized."""
    return _GLOBAL_TENSOR_TRACERS


def get_tt_flags():
    """Return the global tensor-tracing flags instance, if initialized."""
    return _GLOBAL_TT_FLAGS


def get_compressor(flag_type):
    """Return the compressor associated with a flag type."""
    global _GLOBAL_COMPRESSOR
    if _GLOBAL_COMPRESSOR is None:
        _set_compressor()
    assert _GLOBAL_COMPRESSOR is not None
    compressor = _GLOBAL_COMPRESSOR.get(flag_type)
    if compressor is None:
        compressor = EmptyCompressor({})
        _GLOBAL_COMPRESSOR[flag_type] = compressor
    return compressor


def get_report():
    """Return the current global tensor report callback."""
    return _GLOBAL_REPORT


class FlagType(Enum):
    """Kinds of intermediate tensors that can be traced."""

    INVALID_FLAG = 0
    QKV_mat_mul = 1
    ContextLayer_mat_mul = 3
    MLP1_mat_mul = 4
    MLP2_mat_mul = 5
    AttentionOutput_mat_mul = 6
    HiddenStates = 7


class AbstractCompressor:
    """Abstract base class for tensor compressors."""

    def __init__(self):
        pass

    @abstractmethod
    def compress_one_rank(self, layer_number, flag_type, data):
        """Compress a tensor locally on one rank before any gather."""
        raise NotImplementedError

    @abstractmethod
    def compress(self, layer_number, flag_type, data):
        """Compress an already-gathered tensor and return (valid, args, payload)."""
        raise NotImplementedError


class TileCompressor(AbstractCompressor):
    """Compress by chunking the last dimension into tiles and reducing each tile."""

    def __init__(self, configs):
        self.configs = {
            "tiles": configs.get("tiles", 96),
            "method": configs.get("method", "data.mean(dim=-1)"),
            "tiles_one_rank": configs.get("tiles_one_rank", 96),
            "method_one_rank": configs.get("method_one_rank", "data.mean(dim=-1)"),
        }

    def compress_tensor(self, data_in, tiles, method):
        """Apply a reduction expression over tiles of the last tensor dimension."""
        B, S, F = data_in.shape
        chunk_size = math.ceil(F / tiles)
        padded_len = chunk_size * tiles
        padded_data = torch.nn.functional.pad(data_in, (0, padded_len - F))
        data_for_eval = padded_data.reshape(B, S, tiles, chunk_size)
        try:
            compressed = eval(method, {"__builtins__": {}}, {"data": data_for_eval})
        except Exception as e:
            logger.warning(
                "Tensor tracer compressor method failed; falling back to mean. method=%r error=%s",
                method,
                e,
            )
            compressed = data_for_eval.mean(dim=-1)
        return compressed

    def compress_one_rank(self, layer_number, flag_type, data):
        """Compress a tensor before gather using the per-rank config."""
        return self.compress_tensor(
            data, self.configs["tiles_one_rank"], self.configs["method_one_rank"]
        )

    def compress(self, layer_number, flag_type, data):
        """Compress a gathered tensor using the global config."""
        compressed = self.compress_tensor(data, self.configs["tiles"], self.configs["method"])
        return True, list(compressed.shape), compressed.flatten()


class NoOpCompressor(AbstractCompressor):
    """A compressor that returns the original tensor unchanged."""

    def __init__(self, configs):
        pass

    def compress_one_rank(self, layer_number, flag_type, data):
        """Return the original tensor."""
        return data

    def compress(self, layer_number, flag_type, data):
        """Return the original tensor flattened."""
        return True, list(data.shape), data.flatten()


class EmptyCompressor(AbstractCompressor):
    """A compressor that always reports an empty payload."""

    def __init__(self, configs):
        pass

    def compress_one_rank(self, layer_number, flag_type, data):
        """Return an empty tensor that is safe for downstream gather/cat ops."""
        empty_shape = list(data.shape)
        if empty_shape:
            empty_shape[-1] = 0
        return data.new_empty(empty_shape)

    def compress(self, layer_number, flag_type, data):
        """Return an empty flattened tensor with a shape matching the input."""
        empty_shape = list(data.shape)
        if empty_shape:
            empty_shape[-1] = 0
        empty = data.new_empty(empty_shape)
        return True, empty_shape, empty.flatten()


class ProjectionCompressor(AbstractCompressor):
    """Project the last dimension onto a per-layer vector."""

    def __init__(self, configs):
        self.projection_vector = None
        try:
            self.projection_vector = torch.load(configs["vector_path"], map_location="cpu")
            self.projection_vector = torch.nn.functional.normalize(
                self.projection_vector, p=2, dim=1
            )
            if torch.cuda.is_available():
                try:
                    device = torch.cuda.current_device()
                    self.projection_vector = self.projection_vector.to(device)
                except RuntimeError:
                    logger.warning(
                        "Tensor tracer projection vector loaded, but CUDA is not initialized; "
                        "keeping it on CPU."
                    )
        except Exception as e:
            logger.warning("Tensor tracer projection vector load failed: %s", e)
            self.projection_vector = None

    def compress_one_rank(self, layer_number, flag_type, data):
        """Return the original tensor before gather."""
        return data

    def compress(self, layer_number, flag_type, data):
        """Project and return the compressed payload."""
        if self.projection_vector is None:
            return False, [], torch.tensor([])
        vector = self.projection_vector[layer_number - 1]
        projected = torch.matmul(data, vector).unsqueeze(-1)
        return True, list(projected.shape), projected.flatten()


COMPRESSOR_MAP = {
    "TileCompressor": TileCompressor,
    "NoOpCompressor": NoOpCompressor,
    "EmptyCompressor": EmptyCompressor,
    "ProjectionCompressor": ProjectionCompressor,
}


class TensorTracers:
    """Trace and report tensors selected by TTFlags."""

    def report(self, name: NameTuple, tensor_data: torch.Tensor) -> None:
        """Compress and send a traced tensor through the report callback."""
        compressor = get_compressor(name[1])
        valid, comp_args, compressed_tensor = compressor.compress(name[0], name[1], tensor_data)
        if not valid:
            logger.warning(
                "Tensor tracer compressor %s returned invalid result for %s; skipping report.",
                type(compressor).__name__,
                name,
            )
            return
        get_report()(name, comp_args, compressed_tensor)


class TTFlags:
    """Global flags to record the intermediate results of the model."""

    def __init__(self, args):
        self.num_layers = args.num_layers
        self.flags: Dict[FlagType, Dict[int, bool]] = {
            FlagType.INVALID_FLAG: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.QKV_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.ContextLayer_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.MLP1_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.MLP2_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.AttentionOutput_mat_mul: {i: False for i in range(1, self.num_layers + 1)},
            FlagType.HiddenStates: {i: False for i in range(1, self.num_layers + 1)},
        }
        self.should_trace = True

    def get_flag(self, flag_type: FlagType, layer_index: int) -> bool:
        """Return whether a given flag is enabled for a layer."""
        return self.should_trace and self.flags.get(flag_type, {}).get(layer_index, False)

    def set_by_configs(self, configs: Dict[str, Any], comp_configs: Dict[str, Any]):
        """Update tracing flags and compressor configurations from user configs."""
        global _GLOBAL_COMPRESSOR
        if _GLOBAL_COMPRESSOR is None:
            _set_compressor()
        assert _GLOBAL_COMPRESSOR is not None

        for flag_type in self.flags:
            if flag_type == FlagType.INVALID_FLAG:
                continue
            val = str(configs.get(flag_type.name, False)).lower() == "true"
            for i in range(1, self.num_layers + 1):
                self.flags[flag_type][i] = val

            specific_comp_config = comp_configs.get(flag_type.name)
            if specific_comp_config is not None:
                compressor_type = specific_comp_config.get("compressor_type", "EmptyCompressor")
                compressor_configs = specific_comp_config.get("compressor_configs", {})
                compressor_cls = COMPRESSOR_MAP.get(compressor_type, EmptyCompressor)
                _GLOBAL_COMPRESSOR[flag_type] = compressor_cls(compressor_configs)


class TTHookManager:
    """Manage forward hooks that gather and report tensors for visualization."""

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
                    rank0_global = torch.distributed.get_process_group_ranks(
                        get_tensor_model_parallel_group()
                    )[0]

                    if isinstance(output, (list, tuple)):
                        tensor_data = output[0].detach()
                    else:
                        tensor_data = output.detach()
                    tensor_data = get_compressor(flag_type).compress_one_rank(
                        layer_number, flag_type, tensor_data
                    )
                    tensor_data_cont = tensor_data.contiguous()
                    if rank == 0:
                        tensor_list = [
                            torch.zeros_like(
                                tensor_data_cont, dtype=tensor_data_cont.dtype, device=device
                            )
                            for _ in range(world_size)
                        ]
                    else:
                        tensor_list = None
                    if world_size > 1:
                        torch.distributed.gather(
                            tensor_data_cont,
                            tensor_list,
                            dst=rank0_global,
                            group=get_tensor_model_parallel_group(),
                        )
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

                        get_tensor_tracers().report(
                            (layer_number, flag_type), aggregated_tensor.transpose(0, 1)
                        )

            return hook

        def generate_hook_transpose_row(flag_type: FlagType, layer_number: int):
            def hook(module, input, output):
                if get_tt_flags().get_flag(flag_type, layer_number):
                    device = torch.cuda.current_device()
                    world_size = get_tensor_model_parallel_world_size()
                    rank = get_tensor_model_parallel_rank()
                    rank0_global = torch.distributed.get_process_group_ranks(
                        get_tensor_model_parallel_group()
                    )[0]

                    if args.sequence_parallel:
                        if isinstance(output, (list, tuple)):
                            tensor_data = output[0].detach()
                        else:
                            tensor_data = output.detach()
                        tensor_data = get_compressor(flag_type).compress_one_rank(
                            layer_number, flag_type, tensor_data
                        )
                        tensor_data_cont = tensor_data.contiguous()
                        if rank == 0:
                            tensor_list = [
                                torch.zeros_like(
                                    tensor_data_cont, dtype=tensor_data_cont.dtype, device=device
                                )
                                for _ in range(world_size)
                            ]
                        else:
                            tensor_list = None
                        if world_size > 1:
                            torch.distributed.gather(
                                tensor_data_cont,
                                tensor_list,
                                dst=rank0_global,
                                group=get_tensor_model_parallel_group(),
                            )
                        else:
                            tensor_list = [tensor_data_cont]

                        if rank == 0:
                            aggregated_tensor = torch.cat(tensor_list, dim=0)
                            get_tensor_tracers().report(
                                (layer_number, flag_type), aggregated_tensor.transpose(0, 1)
                            )
                    else:
                        if rank == 0:
                            if isinstance(output, (list, tuple)):
                                tensor_data = output[0].detach()
                            else:
                                tensor_data = output.detach()
                            tensor_data = get_compressor(flag_type).compress_one_rank(
                                layer_number, flag_type, tensor_data
                            )
                            get_tensor_tracers().report(
                                (layer_number, flag_type), tensor_data.transpose(0, 1)
                            )

            return hook

        def generate_hook_attn(flag_type: FlagType, layer_number: int):
            def hook(module, input, output):
                if get_tt_flags().get_flag(flag_type, layer_number):
                    device = torch.cuda.current_device()
                    world_size = get_tensor_model_parallel_world_size()
                    rank = get_tensor_model_parallel_rank()
                    rank0_global = torch.distributed.get_process_group_ranks(
                        get_tensor_model_parallel_group()
                    )[0]

                    if isinstance(output, (list, tuple)):
                        tensor_data = output[0].detach()
                    else:
                        tensor_data = output.detach()
                    tensor_data = get_compressor(flag_type).compress_one_rank(
                        layer_number, flag_type, tensor_data
                    )
                    tensor_data_cont = tensor_data.contiguous()
                    if rank == 0:
                        tensor_list = [
                            torch.zeros_like(
                                tensor_data_cont, dtype=tensor_data_cont.dtype, device=device
                            )
                            for _ in range(world_size)
                        ]
                    else:
                        tensor_list = None
                    if world_size > 1:
                        torch.distributed.gather(
                            tensor_data_cont,
                            tensor_list,
                            dst=rank0_global,
                            group=get_tensor_model_parallel_group(),
                        )
                    else:
                        tensor_list = [tensor_data_cont]

                    if rank == 0:
                        aggregated_tensor = torch.cat(tensor_list, dim=1)
                        get_tensor_tracers().report((layer_number, flag_type), aggregated_tensor)

            return hook

        for layer in range(model.decoder.num_layers_per_pipeline_rank):
            global_layer_number = model.decoder.layers[layer].layer_number
            self.hooks.append(
                model.decoder.layers[layer].self_attention.linear_qkv.register_forward_hook(
                    generate_hook_transpose_col(FlagType.QKV_mat_mul, global_layer_number)
                )
            )  # Col, not gather_output
            self.hooks.append(
                model.decoder.layers[layer].mlp.linear_fc1.register_forward_hook(
                    generate_hook_transpose_col(FlagType.MLP1_mat_mul, global_layer_number)
                )
            )  # Col, not gather_output
            self.hooks.append(
                model.decoder.layers[layer].mlp.linear_fc2.register_forward_hook(
                    generate_hook_transpose_row(FlagType.MLP2_mat_mul, global_layer_number)
                )
            )  # Row
            self.hooks.append(
                model.decoder.layers[layer].self_attention.register_forward_hook(
                    generate_hook_transpose_row(
                        FlagType.AttentionOutput_mat_mul, global_layer_number
                    )
                )
            )  # Row
            self.hooks.append(
                model.decoder.layers[layer].register_forward_hook(
                    generate_hook_transpose_row(FlagType.HiddenStates, global_layer_number)
                )
            )  # Row
            self.hooks.append(
                model.decoder.layers[layer].self_attention.core_attention.register_forward_hook(
                    generate_hook_transpose_col(FlagType.ContextLayer_mat_mul, global_layer_number)
                )
            )  # Col, not gather_output


'''
For ColumnParallelLinear:
1. If gather_output, we do not do all gather
2. If not gather_output, we do all gather
For RowParallelLinear:
1. If sequence_parallel, we do all gather
2. If not sequence_parallel, we do not do all gather
'''
