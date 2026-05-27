# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
import dataclasses

import pytest
import yaml
import torch

from megatron.training import yaml_arguments


def test_load_yaml_returns_nested_namespace(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
language_model:
  num_layers: 2
  hidden_size: 16
model_parallel:
  tensor_model_parallel_size: 1
data_path: sample
""",
        encoding="utf-8",
    )

    args = yaml_arguments.load_yaml(config_path)

    assert args.yaml_cfg == config_path
    assert args.language_model.num_layers == 2
    assert args.language_model.hidden_size == 16
    assert args.model_parallel.tensor_model_parallel_size == 1
    assert args.data_path == "sample"


def test_yaml_environment_variable_constructor(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/datasets")

    loaded = yaml.load("path: ${DATA_ROOT}/gpt", Loader=yaml.Loader)

    assert loaded["path"] == "/datasets/gpt"


def test_yaml_environment_variable_constructor_requires_existing_env():
    with pytest.raises(AssertionError, match="environment variable MISSING_ROOT"):
        yaml.load("path: ${MISSING_ROOT}/gpt", Loader=yaml.Loader)


def test_validate_yaml_basic_iteration_config(monkeypatch):
    args = SimpleNamespace(
        data_path="train valid test",
        world_size=1,
        rank=1,
        model_parallel=SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            tp_comm_overlap=False,
            fp16=False,
            bf16=True,
            sequence_parallel=False,
            transformer_pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            overlap_p2p_comm=True,
            params_dtype=None,
            variable_seq_lengths=True,
            expert_model_parallel_size=1,
        ),
        account_for_embedding_in_pipeline_split=False,
        micro_batch_size=2,
        global_batch_size=None,
        num_layers_per_virtual_pipeline_stage=None,
        overlap_param_gather=False,
        use_distributed_optimizer=False,
        overlap_grad_reduce=False,
        accumulate_allreduce_grads_in_fp32=False,
        dataloader_type=None,
        train_iters=10,
        train_samples=None,
        lr_decay_samples=None,
        lr_warmup_samples=0,
        rampup_batch_size=None,
        lr_warmup_fraction=None,
        lr_warmup_iters=0,
        lr_decay_iters=None,
        weight_decay_incr_style="constant",
        start_weight_decay=None,
        end_weight_decay=None,
        weight_decay=0.01,
        language_model=SimpleNamespace(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=None,
            activation_func="gelu",
            kv_channels=None,
            fp32_residual_connection=False,
            moe_grouped_gemm=False,
            moe_pad_expert_input_to_capacity=False,
            moe_token_dispatcher_type="allgather",
            num_moe_experts=None,
            persist_layer_norm=True,
            distribute_saved_activations=False,
            recompute_granularity=None,
            recompute_method=None,
        ),
        spec=None,
        encoder_num_layers=None,
        seq_length=8,
        encoder_seq_length=None,
        decoder_seq_length=None,
        max_position_embeddings=8,
        lr=0.001,
        min_lr=0.0,
        save=None,
        save_interval=None,
        fp16_lm_cross_entropy=False,
        onnx_safe=False,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask_in_dataloader=True,
        use_cpu_initialization=None,
        lazy_mpu_init=None,
        recompute_num_layers=None,
        num_layers_per_pipeline_stage=None,
        make_vocab_size_divisible_by=128,
    )

    validated = yaml_arguments.validate_yaml(args)

    assert args.data_path == ["train", "valid", "test"]
    assert args.data_parallel_size == 1
    assert args.global_batch_size == 2
    assert args.dataloader_type == "single"
    assert args.encoder_num_layers == 2
    assert args.encoder_seq_length == 8
    assert args.language_model.ffn_hidden_size == 64
    assert args.language_model.kv_channels == 4
    assert args.accumulate_allreduce_grads_in_fp32
    assert validated.num_experts is None


@dataclasses.dataclass
class _TinyYamlConfig:
    hidden_size: int = 0
    num_attention_heads: int = 0
    params_dtype: object = None
    overlap_p2p_comm: bool = False
    activation_func: object = None
    init_method: object = None
    embedding_init_method: object = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _yaml_config_args(activation_func="gelu", **language_overrides):
    language_model = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        activation_func=activation_func,
        bias_swiglu_fusion=True,
        add_bias_linear=False,
        bias_activation_fusion=True,
        init_method="xavier_uniform",
        embedding_init_method="xavier_uniform",
        multi_latent_attention=False,
    )
    for key, value in language_overrides.items():
        setattr(language_model, key, value)
    return SimpleNamespace(
        language_model=language_model,
        model_parallel=SimpleNamespace(
            params_dtype=torch.bfloat16,
            overlap_p2p_comm=False,
        ),
    )


@pytest.mark.parametrize(
    ("activation", "expected_func", "expected_gated", "expected_bias_fusion"),
    [
        ("swiglu", torch.nn.functional.silu, True, True),
        ("gelu", torch.nn.functional.gelu, False, True),
    ],
)
def test_core_transformer_config_from_yaml_activation_paths(
    monkeypatch, activation, expected_func, expected_gated, expected_bias_fusion
):
    monkeypatch.setattr(yaml_arguments, "TransformerConfig", _TinyYamlConfig)

    config = yaml_arguments.core_transformer_config_from_yaml(_yaml_config_args(activation))

    assert config.kwargs["pipeline_dtype"] is torch.bfloat16
    assert config.kwargs["batch_p2p_comm"] is True
    assert config.kwargs["deallocate_pipeline_outputs"] is True
    assert config.kwargs["activation_func"] is expected_func
    assert config.kwargs.get("gated_linear_unit", False) is expected_gated
    assert config.kwargs["bias_activation_fusion"] is expected_bias_fusion
    assert config.kwargs["init_method"] is torch.nn.init.xavier_uniform_
    assert config.kwargs["embedding_init_method"] is torch.nn.init.xavier_uniform_


def test_core_transformer_config_from_yaml_squared_relu_and_mla(monkeypatch):
    monkeypatch.setattr(yaml_arguments, "TransformerConfig", _TinyYamlConfig)
    monkeypatch.setattr(yaml_arguments, "MLATransformerConfig", _TinyYamlConfig)

    config = yaml_arguments.core_transformer_config_from_yaml(
        _yaml_config_args("squaredrelu", multi_latent_attention=True)
    )

    assert config.kwargs["activation_func"](torch.tensor([-2.0, 3.0])).tolist() == [0.0, 9.0]


def test_core_transformer_config_from_yaml_rejects_unknown_activation(monkeypatch):
    monkeypatch.setattr(yaml_arguments, "TransformerConfig", _TinyYamlConfig)

    with pytest.raises(AssertionError, match="not a supported activation"):
        yaml_arguments.core_transformer_config_from_yaml(_yaml_config_args("relu"))


@dataclasses.dataclass
class _RequiredYamlFields:
    hidden_size: int
    params_dtype: object


def test_core_config_from_args_collects_required_fields_and_reports_missing():
    args = SimpleNamespace(hidden_size=16, params_dtype=torch.float32)

    assert yaml_arguments.core_config_from_args(args, _RequiredYamlFields) == {
        "hidden_size": 16,
        "params_dtype": torch.float32,
    }

    with pytest.raises(Exception, match="Missing argument params_dtype"):
        yaml_arguments.core_config_from_args(SimpleNamespace(hidden_size=16), _RequiredYamlFields)


def test_print_args_only_prints_on_rank_zero(capsys):
    yaml_arguments._print_args("yaml", SimpleNamespace(rank=0, beta=2, alpha=1))
    output = capsys.readouterr().out
    assert "yaml" in output
    assert "alpha" in output
    assert "beta" in output

    yaml_arguments._print_args("hidden", SimpleNamespace(rank=1, value=3))
    assert capsys.readouterr().out == ""
