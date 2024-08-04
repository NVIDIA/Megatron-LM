import torch
from torch.nn.parameter import Parameter
import intel_extension_for_pytorch as ipex  # noqa

# Taken from facebookresearch/llama
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(dim))

    def forward(self, x):
        output = torch.xpu.IpexRmsNorm(x, self.weight.shape, self.weight, self.eps)
        return output
