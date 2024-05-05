import torch
from tool_function import *
from dequant_function import *
import os
from pytorch_memlab import LineProfiler, profile

def unittest_deqaunt_cuda(tensor, groupsize=128, quant_module=None, hadamard=False):

    h_tensor = tensor.clone()
    if hadamard is True:
        h_tensor = fast_hadamard_transform(h_tensor, k=5, normalize=True)

    # stochastic quantize kernel
    N = h_tensor.nelement()
    num_groups = N // groupsize
    quant_tensor_cuda, quant_scales_cuda = quant_module.stochastic_quantize(h_tensor.clone(), num_groups, 4, quant_module.Symmetric)
    dequant_tensor_cuda = dequantize_4bits(quant_tensor_cuda, quant_scales_cuda, groupsize)

    if hadamard is True:
        dequant_tensor_cuda = fast_hadamard_transform(dequant_tensor_cuda, k=5, normalize=True)

    abs_error_norm, rela_error_norm = analysis_diff(tensor, dequant_tensor_cuda)
    print(f"cuda version quantization, absolute error norm: {abs_error_norm}, relative error norm: {rela_error_norm}")

def unittest_dequant_torch(tensor, groupsize=128, hadamard=False):

    h_tensor = tensor.clone()
    if hadamard is True:
        h_tensor = fast_hadamard_transform(h_tensor, k=5, normalize=True)

    N = h_tensor.nelement()
    quant_tensor_torch = quantize_4bits(h_tensor, groupsize)
    quantized_tensor_view = quant_tensor_torch.view(N // groupsize, -1)
    x_quant = quantized_tensor_view[:, :groupsize // 2].clone()
    x_scale = quantized_tensor_view[:, groupsize // 2:].clone()

    dequant_tensor_torch = dequantize_4bits(x_quant, x_scale, groupsize)

    if hadamard is True:
        dequant_tensor_torch = fast_hadamard_transform(dequant_tensor_torch, k=5, normalize=True)
    print(dequant_tensor_torch.shape, dequant_tensor_torch.dtype)

    abs_error_norm, rela_error_norm = analysis_diff(tensor, dequant_tensor_torch)
    print(f"torch version quantization, absolute error norm: {abs_error_norm}, relative error norm: {rela_error_norm}")

def unittest_1():
    torch.manual_seed(1234)
    torch.set_printoptions(sci_mode=False)

    pkg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
    print('pkg path:', pkg_path)
    quantization_module = build_and_import_module(pkg_path, 'quantization_cuda')

    N = 25600000
    tensor = torch.load('/home/jindjia/scripts/bytedance/dump_data/dump/iteration_008040/bucketid_000/iteration_008040_bucketid_000_dprank_000.pt', map_location='cuda')[:N].to(torch.half)
    groupsize = 128
    enable_hadamard = True

    print(f'tensor size: {tensor.nelement() * tensor.element_size()} Bytes')
    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/torchquant_profile'),
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True
    # ) as prof:
    #     for step in range(100):
    #         prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
    #         if step >= 1 + 1 + 3:
    #             break
    #         unittest_dequant_torch(tensor, groupsize=groupsize, hadamard=enable_hadamard)
    #         torch.cuda.synchronize()

    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/create_zero'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for step in range(100):
            prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
            if step >= 1 + 1 + 3:
                break
            # t = tensor.mul(2)
            tensor = torch.zeros(device='cuda', dtype=torch.float, size=(N,))
            torch.cuda.synchronize()


    # unittest_deqaunt_cuda(tensor, groupsize=groupsize, quant_module=quantization_module, hadamard=enable_hadamard)

def memory_profile01():
    N = 33_554_432
    tensor = torch.normal(size=(N,), dtype=torch.float, device='cuda', std=1, mean=0)
    # tensor = torch.load('/home/jindjia/scripts/bytedance/dump_data/dump/iteration_008040/bucketid_000/iteration_008040_bucketid_000_dprank_000.pt', map_location='cuda')
    tensor = tensor[:N].to(torch.half)
    groupsize = 128
    enable_hadamard = True
    unittest_dequant_torch(tensor, groupsize=groupsize, hadamard=enable_hadamard)
    print('finished')

@profile
def memory_profile02(quantization_module):

    N = 134_217_728
    tensor = torch.normal(size=(N,), dtype=torch.bfloat16, device='cuda', std=1, mean=0)
    # tensor = fast_hadamard_transform(tensor, k=5, normalize=True)
    # tensor = torch.load('/home/jindjia/scripts/bytedance/dump_data/dump/iteration_008040/bucketid_000/iteration_008040_bucketid_000_dprank_000.pt', map_location='cuda')
    # tensor = tensor[:N].to(torch.half)
    groupsize = 128
    enable_hadamard = True

    N = tensor.nelement()
    num_groups = N // groupsize
    quant_tensor_cuda, quant_scales_cuda = quantization_module.stochastic_quantize(tensor, num_groups, 4, quantization_module.Symmetric)
    # dequant_tensor_cuda = dequantize_4bits(quant_tensor_cuda, quant_scales_cuda, groupsize)

    print('finished')

if __name__ == '__main__':
    pkg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
    print('pkg path:', pkg_path)
    quantization_module = build_and_import_module(pkg_path, 'quantization_cuda')
    
    # memory_profile01()
    memory_profile02(quantization_module)