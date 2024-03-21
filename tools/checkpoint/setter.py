# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


class ModelSetter:
    '''Model parameter setter.

    See convert.py for a full list of supported parameters and their names.
    '''

    @classmethod
    def set_tensor(cls, dst, src):
        '''Copy (in-place) src tensor to dst tensor.'''
        if src is not None:
            dst.data.copy_(src)

    @classmethod
    def has_position_embeddings(cls, model):
        '''
        Return True if learned parameters exist for position embeddings (e.g.,
        learned absolute), and False otherwise (e.g., RoPE).
        '''
        raise NotImplementedError

    @classmethod
    def set_embeddings(
        cls,
        model,
        word=None,
        pos=None,
    ):
        '''Set word and position embeddings.'''
        raise NotImplementedError

    @classmethod
    def set_output_word_embeddings(
        cls,
        model,
        emb=None,
    ):
        '''Set output word embeddings for final pipeline stage.'''
        raise NotImplementedError

    @classmethod
    def set_layer(
        cls,
        model,
        layer_idx,
        self_attn_norm_weight=None,
        self_attn_norm_bias=None,
        self_attn_qkv_weight=None,
        self_attn_qkv_bias=None,
        self_attn_proj_weight=None,
        self_attn_proj_bias=None,
        mlp_norm_weight=None,
        mlp_norm_bias=None,
        mlp_fc1_weight=None,
        mlp_fc1_bias=None,
        mlp_fc2_weight=None,
        mlp_fc2_bias=None,
    ):
        '''Set layer parameters.'''
        raise NotImplementedError

    @classmethod
    def set_final_norm(
        cls,
        model,
        weight=None,
        bias=None,
    ):
        '''Set final norm parameters (i.e., after last transformer layer).'''
        raise NotImplementedError

    @classmethod
    def set_output_layer(
        cls,
        model,
        weight=None,
    ):
        '''Set output (i.e., 'dense') weights.'''
        raise NotImplementedError

    @classmethod
    def set_pooler(
        cls,
        model,
        weight=None,
        bias=None,
    ):
        '''Set pooler parameters (e.g., for Bert).'''
        raise NotImplementedError

    @classmethod
    def set_lm_head(
        cls,
        model,
        dense_weight=None,
        dense_bias=None,
        norm_weight=None,
        norm_bias=None,
    ):
        '''Set LM head parameters.'''
        raise NotImplementedError

    @classmethod
    def set_binary_head(
        cls,
        model,
        weight=None,
        bias=None,
    ):
        '''Set binary head parameters.'''
        raise NotImplementedError
