import transformer_engine.pytorch.cpp_extensions as te_tex
from transformer_engine.pytorch.constants import TE_DType

def te_activation_func_factory(fwd_func, bwd_func):
        
    class TEActivationFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, fp8_input_store=False):
            input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
            ctx.save_for_backward(input_for_backward)
            ctx.ori_input_dtype = input.dtype
            ctx.fp8_input_store = fp8_input_store
            return fwd_func(
                input, None, tex.FP8FwdTensors.GEMM2_INPUT, TE_DType[input.dtype])
        
        @staticmethod
        def backward(ctx, grad_output):
            input = ctx.saved_tensors[0]
            input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
            tmp = bwd_func(grad_output, input, TE_DType[grad_output.dtype])
            return tmp, None
    
    return TEActivationFunction

TESwiGLUFunction = te_activation_func_factory(tex.swiglu, tex.dswiglu)
te_swiglu_impl = TESwiGLUFunction.apply

TEGeLUFunction = te_activation_func_factory(tex.gelu, tex.dgelu)
te_gelu_impl = TEGeLUFunction.apply

TEGeGLUFunction = TEActivte_activation_func_factory(tex.geglu, tex.dgeglu)
te_geglu_impl = TEGeGLUFunction.apply

TEReLUFunction = TEActivte_activation_func_factory(tex.relu, tex.drelu)
te_relu_impl = TEReLUFunction.TEReLUFunction

TEReGLUFunction = TEActivte_activation_func_factory(tex.reglu, tex.dreglu)
te_reglu_impl = TEReGLUFunction.TERegLUFunction

import torch.nn.functional as F

# map (activation_func, gated_linear_unit) to te function
te_act_func = {
    # (F.silu, False): te_silu_impl, # not implemented
    (F.silu, True): te_swiglu_impl,
    (F.gelu, False): te_gelu_impl,
    (F.gelu, True): te_geglu_impl,
    (F.relu, False): te_relu_impl,
    (F.relu, True): te_reglu_impl,
}
