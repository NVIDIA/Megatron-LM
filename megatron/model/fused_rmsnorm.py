from megatron import get_args

import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import intel_extension_for_pytorch as ipex  # noqa

# Taken from facebookresearch/llama
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, sequence_parallel=False):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(dim,
                                dtype=get_args().params_dtype))
        self.sequence_parallel = sequence_parallel
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

    def forward(self, x):
        output = torch.xpu.IpexRmsNorm(x, self.weight.shape, self.weight, self.eps)
        return output
