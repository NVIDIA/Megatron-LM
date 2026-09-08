# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Optional

import torch

from megatron.core.fp8_utils import get_fp8_align_size, get_fp8_context
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import get_tensor_model_parallel_group_if_none, make_viewless_tensor


class MultimodalProjector(MegatronModule):
    """
    MultimodalProjector will take the encoded input with input_size hidden state and project
    it into the hidden size of the language model for multimodal training. When projector is
    type affine linear_fc1 from submodules is used.

    Args:
        transformer_config (TransformerConfig): Transformer config
        submodules (MLPSubmodules): Specifies MLP submodules for mlp type projector
        projector_type (str): Projector type
        input_size (int): Input size from feature encoder
        tp_group (torch.distributed.ProcessGroup): Tensor parallel group
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        projector_type: str,
        input_size: int,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.projector_type = projector_type
        tp_group = pg_collection.tp if pg_collection is not None else tp_group
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group)

        assert submodules is not None, "MLPSubmodules must be provided"

        fp8_init_context = get_fp8_context(config, 0, is_init=True)
        with fp8_init_context:
            if self.projector_type == "mlp":
                self.encoder = MLP(
                    config=config,
                    submodules=submodules,
                    input_size=input_size,
                    tp_group=tp_group,
                    pg_collection=pg_collection,
                )
            elif self.projector_type == "affine":
                self.encoder = submodules.linear_fc1(
                    input_size,
                    config.hidden_size,
                    config=config,
                    init_method=not_none(config.init_method),
                    gather_output=True,
                    bias=config.add_bias_linear,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name=None,
                    tp_group=tp_group,
                    pg_collection=pg_collection,
                )
            else:
                raise Exception(f"Unsupported multimodal projection type {self.projector_type}")

    def forward(self, hidden_states):
        """Run multimodal projector.

        Args:
            hidden_states (torch.Tensor): Input.

        Returns:
            torch.Tensor: The projected output.
        """
        original_shape = hidden_states.shape
        padding = 0
        if self.config.fp8:
            alignment = get_fp8_align_size(self.config.fp8_recipe)
            num_tokens = hidden_states.numel() // hidden_states.shape[-1]
            padding = (
                alignment
                if num_tokens == 0 and self.config.gtp_weight_remat_size > 1
                else (-num_tokens) % alignment
            )
            if padding:
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
                hidden_states = torch.cat(
                    [hidden_states, hidden_states.new_zeros(padding, hidden_states.shape[-1])],
                    dim=0,
                )

        fp8_context = get_fp8_context(self.config)
        with fp8_context:
            # Run encoder.
            encoder_output, encoder_output_bias = apply_module(self.encoder)(hidden_states)

            if encoder_output_bias is not None:
                encoder_output = encoder_output + encoder_output_bias

            if padding:
                encoder_output = encoder_output[:-padding]
                encoder_output = encoder_output.reshape(
                    *original_shape[:-1], encoder_output.shape[-1]
                )

            # the encoder produces "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            encoder_output = make_viewless_tensor(
                inp=encoder_output, requires_grad=True, keep_graph=True
            )

        return encoder_output
