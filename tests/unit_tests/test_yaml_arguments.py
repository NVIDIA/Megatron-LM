# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import yaml

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
