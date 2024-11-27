# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch

from megatron.core.transformer import TransformerConfig

TORCH_VERSION = torch.__version__.split('.')


class WrappedTorchNorm:
    """
    A conditional wrapper to initialize an instance of PyTorch's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        # TODO: unused arguments.
        # See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/223
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
    ):
        assert (
            not config.layernorm_zero_centered_gamma
        ), f"zero_centered_gamma not supported by torch LayerNorm"

        assert not config.persist_layer_norm, f"persist_layer_norm not supported by torch LayerNorm"

        assert (
            not config.memory_efficient_layer_norm
        ), f"memory_efficient_layer_norm not supported by torch LayerNorm"

        if config.normalization == "LayerNorm":
            norm_cls = torch.nn.LayerNorm
        elif config.normalization == "RMSNorm":
            version_geq_2_4 = int(TORCH_VERSION[0]) > 2 or (
                int(TORCH_VERSION[0]) == 2 and int(TORCH_VERSION[1]) >= 4
            )
            assert version_geq_2_4, 'Torch RMSNorm requires PyTorch version >= 2.4.0'

            norm_cls = torch.nn.RMSNorm
        else:
            raise Exception("Only LayerNorm and RMSNorm are currently supported")

        return norm_cls(normalized_shape=hidden_size, eps=eps)
