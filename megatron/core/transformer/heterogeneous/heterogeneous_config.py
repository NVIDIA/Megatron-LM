# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from megatron.core.transformer import TransformerConfig


@dataclass
class AttentionConfig:
    """Configuration parameters for the self-attention part of a single transformer
    block in a heterogeneous transformer."""

    no_op: bool = False
    """Whether this is a no-op operation."""

    replace_with_linear: bool = False
    """Whether to replace the self-attention mechanism with a single linear layer."""

    num_query_groups: int | None = None
    """Number of query groups for grouped query attention."""

    @classmethod
    def build_config_from_dict(
        cls, block_config_dict: dict, num_attention_heads: int
    ) -> 'AttentionConfig':
        """
        Builds an AttentionConfig object from a dictionary and the number of attention heads.

        Args:
            block_config_dict (dict): The dictionary containing the configuration for the attention.
            num_attention_heads (int): The number of attention heads.

        Returns:
            AttentionConfig: The AttentionConfig object.
        """
        attention_config_dict = block_config_dict["attention"]
        if "num_query_groups" not in attention_config_dict:
            # compatibility with HF config of nvidia/Llama-3_1-Nemotron-51B-Instruct
            n_heads_in_group = attention_config_dict.pop("n_heads_in_group")
            if n_heads_in_group is not None:
                if num_attention_heads % n_heads_in_group != 0:
                    raise ValueError(
                        f"num_attention_heads ({num_attention_heads}) must be a multiple of "
                        f"n_heads_in_group ({n_heads_in_group})."
                    )
                num_query_groups = num_attention_heads // n_heads_in_group
            else:
                num_query_groups = None
            attention_config_dict["num_query_groups"] = num_query_groups

        # keep only fields from cls
        field_names = {f.name for f in fields(cls)}
        attn_config_dict = {k: v for k, v in attention_config_dict.items() if k in field_names}
        return cls(**attn_config_dict)


@dataclass
class MLPConfig:
    """Configuration parameters for the MLP part of a single transformer
    block in a heterogeneous transformer."""

    no_op: bool = False
    """Whether this is a no-op operation."""

    replace_with_linear: bool = False
    """Whether to replace the MLP with a single linear layer."""

    ffn_hidden_size: float | None = None
    """MLP intermediate size"""

    @classmethod
    def build_config_from_dict(cls, block_config_dict: dict, hidden_size: int) -> 'MLPConfig':
        """
        Builds an MLPConfig object from a dictionary and a hidden size.

        Args:
            block_config_dict (dict): The dictionary containing the configuration for the MLP.
            hidden_size (int): The hidden size of the MLP.

        Returns:
            MLPConfig: The constructed MLPConfig object.
        """
        mlp_config_dict = block_config_dict.get("ffn") or block_config_dict.get("mlp")
        if "ffn_hidden_size" not in mlp_config_dict:
            # compatibility with HF config of nvidia/Llama-3_1-Nemotron-51B-Instruct
            ffn_mult = mlp_config_dict.pop("ffn_mult")
            if ffn_mult is not None:
                ffn_hidden_size = cls.ffn_mult_to_intermediate_size(ffn_mult, hidden_size)
            else:
                ffn_hidden_size = None
            mlp_config_dict["ffn_hidden_size"] = ffn_hidden_size

        # keep only fields from cls
        field_names = {f.name for f in fields(cls)}
        mlp_config_dict = {k: v for k, v in mlp_config_dict.items() if k in field_names}
        return cls(**mlp_config_dict)

    @staticmethod
    def ffn_mult_to_intermediate_size(ffn_mult: float, hidden_size: int) -> int:
        """
        Calculates the intermediate size of the MLP based on the given
        `ffn_mult` and `hidden_size`.

        Args:
            ffn_mult (float): The multiplier for the feed-forward network.
            hidden_size (int): The size of the hidden layer.

        Returns:
            int: The calculated intermediate size.
        """
        intermediate_size = int(2 * ffn_mult * hidden_size / 3)
        return MLPConfig.find_multiple(intermediate_size, 256)

    @staticmethod
    def find_multiple(n: int, k: int) -> int:
        """
        Calculates the smallest multiple of `k` greater than or equal to `n`.

        Args:
            n (int): The number to find the multiple of.
            k (int): The number to find the multiple of.

        Returns:
            int: The smallest multiple of `k` greater than or equal to `n`.
        """
        if n % k == 0:
            return n
        return n + k - (n % k)


@dataclass
class TransformerBlockConfig:
    """Configuration parameters for a single transformer block in a heterogeneous transformer."""

    attention: AttentionConfig
    """Configuration parameters for the self-attention part of the transformer block in a 
    heterogeneous transformer."""

    mlp: MLPConfig
    """Configuration parameters for the mlp part of the transformer block in a 
    heterogeneous transformer."""


@dataclass
class HeterogeneousTransformerConfig(TransformerConfig):
    """Configuration object for megatron-core heterogeneous transformers.

    Heterogeneous models refer to transformer architectures where individual layers can differ
    in configuration. Specifically:
        - Attention or MLP layers can be replaced with either a linear layer or a no-op
        - MLP intermediate dimensions can vary between layers
    We use the format of the HuggingFace config files in llama nemotron models to define
    the architecture.
    For example,
    https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/resolve/main/config.json

    Most notably, the "heterogeneous_layers_config_path" maps to a json file containing a
    "block_configs" key, which is a list of attention and mlp configurations for each layer.
    For example, the "block_config" for a 2 layer model is:
    "block_configs": [
        {
            "attention": {
                "n_heads_in_group": 8,
                "no_op": false,
                "replace_with_linear": false,
            },
            "ffn": {
                "ffn_mult": 2.625,
                "no_op": false,
                "replace_with_linear": false,
            }
        },
        {
            "attention": {
                "n_heads_in_group": null,
                "no_op": true,
                "replace_with_linear": false,
            },
            "ffn": {
                "ffn_mult": 2.625,
                "no_op": false,
                "replace_with_linear": false,
            }
        }
    ]
    """

    heterogeneous_layers_config_path: str = ""
    """Path to the json file containing the heterogeneous block specs."""

    heterogeneous_layers_config_encoded_json: str = ""
    """The contents of the json file containing the heterogeneous block specs. It will be read from 
    heterogeneous_layers_config_path at first, then saved forever inside the model checkpoint."""

    per_block_parameters: list[TransformerBlockConfig] = field(init=False)
    """Configuration parameters for each of the transformer blocks in a 
    heterogeneous transformer."""

    def __post_init__(self):
        super().__post_init__()

        self.heterogeneous_block_specs = True

        if self.heterogeneous_layers_config_encoded_json in ("", None):
            self.heterogeneous_layers_config_encoded_json = Path(
                self.heterogeneous_layers_config_path
            ).read_text()

        hf_config_dict = json.loads(self.heterogeneous_layers_config_encoded_json)
        assert "block_configs" in hf_config_dict
        block_list = hf_config_dict["block_configs"]

        block_configs = [
            TransformerBlockConfig(
                attention=AttentionConfig.build_config_from_dict(
                    block_config_dict=block, num_attention_heads=self.num_attention_heads
                ),
                mlp=MLPConfig.build_config_from_dict(
                    block_config_dict=block, hidden_size=self.hidden_size
                ),
            )
            for block in block_list
        ]

        self.per_block_parameters = block_configs

    def get_config_for_layer(self, layer_number: int) -> TransformerConfig:
        """
        Get the config for the given layer number.
        Based on the layer number, the corresponding block config is returned,
        overriding the main config's value.
        """
        layer_idx = layer_number - 1  # layer number starts from 1
        if layer_idx < 0 or layer_idx >= len(self.per_block_parameters):
            raise ValueError(
                f"Invalid layer number: {layer_number}. Should be in "
                f"range [1, {len(self.per_block_parameters)}]."
            )
        block_config = self.per_block_parameters[layer_idx]

        keys_to_update = {}

        # attention config updates
        if block_config.attention.num_query_groups is not None:
            assert (
                not block_config.attention.replace_with_linear and not block_config.attention.no_op
            )
            keys_to_update['num_query_groups'] = block_config.attention.num_query_groups

        # mlp config updates
        if block_config.mlp.ffn_hidden_size is not None:
            assert not block_config.mlp.replace_with_linear and not block_config.mlp.no_op
            keys_to_update['ffn_hidden_size'] = block_config.mlp.ffn_hidden_size

        transformer_config_dict = asdict(self)

        # remove keys that are not in TransformerConfig
        transformer_config_field_names = {f.name for f in fields(TransformerConfig)}
        transformer_config_dict = {
            k: v for k, v in transformer_config_dict.items() if k in transformer_config_field_names
        }

        transformer_config_dict.update(keys_to_update)

        return TransformerConfig(**transformer_config_dict)
