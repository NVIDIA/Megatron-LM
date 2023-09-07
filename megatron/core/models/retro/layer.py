# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

? ? ? [ remove this file ]


class RetroTransformerLayer(TransformerLayer):

    def __init__(
        self,
        config: TransformerConfig,
        spec: TransformerLayerSpec,
        layer_number: int = 1,
        self_attn_mask_type=AttnMaskType.padding,
        add_retriever=False,
    ):

        super().__init__(
            config=config,
            spec=spec,
            layer_number=layer_number,
            self_attn_mask_type=self_attn_mask_type,
        )

        if config.retro_add_retriever:
            retro_args = get_retro_args()
            self.retro_num_neighbors = args.retro_num_neighbors
            self.retro_chunk_length = retro_args.retro_gpt_chunk_length
            self.retro_retrieved_length = retro_args.retro_gpt_retrieved_length

        # Retriever (bi-directional transformer with cross attention)
        # if layer_type == LayerType.retro_decoder_with_retriever:
        if add_retriever:
            raise Exception("hi.")
            self.retriever = ParallelTransformer(
                config=config,
                model_type=ModelType.retro_encoder,
                self_attn_mask_type=AttnMaskType.padding,
                pre_process=True,
                post_process=False,
            )
            self._retriever_key = 'retriever' # necessary?
        else:
            self.retriever = None

# >>>
# eof
# <<<
