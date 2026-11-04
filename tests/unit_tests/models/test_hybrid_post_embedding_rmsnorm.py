# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Decoder-entry normalization, ownership, and checkpoint architecture contracts."""

import argparse
import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.hybrid import hybrid_block as hybrid_block_module
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.optimizer import _get_param_groups, get_standard_config_overrides
from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.utils import WrappedTensor
from megatron.training.arguments import _add_network_size_args
from megatron.training.checkpointing import (
    CheckpointType,
    check_checkpoint_args,
    load_args_from_checkpoint,
)
from tests.unit_tests.test_utilities import Utils


def _config(**overrides):
    values = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        normalization="RMSNorm",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        use_cpu_initialization=True,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def test_post_embedding_rmsnorm_is_opt_in_and_requires_rmsnorm():
    assert not _config().post_embedding_rmsnorm
    with pytest.raises(ValueError, match="post_embedding_rmsnorm requires"):
        _config(post_embedding_rmsnorm=True, normalization="LayerNorm")


def test_post_embedding_rmsnorm_cli_is_opt_in():
    parser = _add_network_size_args(argparse.ArgumentParser())
    assert parser.parse_args([]).post_embedding_rmsnorm is False
    assert parser.parse_args(["--post-embedding-rmsnorm"]).post_embedding_rmsnorm is True


@pytest.mark.parametrize(
    "saved,current,expected", [(True, False, True), (False, True, False), (None, True, False)]
)
def test_checkpoint_args_restore_post_embedding_rmsnorm(saved, current, expected):
    checkpoint_args = SimpleNamespace()
    if saved is not None:
        checkpoint_args.post_embedding_rmsnorm = saved
    args = SimpleNamespace(
        load="checkpoint",
        iteration=0,
        post_embedding_rmsnorm=current,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    with patch(
        "megatron.training.checkpointing._load_base_checkpoint",
        return_value=(
            {"args": checkpoint_args, "iteration": 12},
            "checkpoint",
            False,
            CheckpointType.LEGACY,
        ),
    ):
        restored_args, _ = load_args_from_checkpoint(args)
    assert restored_args.post_embedding_rmsnorm is expected


@pytest.mark.parametrize("saved,current", [(False, True), (True, False), (None, True)])
def test_checkpoint_rejects_post_embedding_rmsnorm_mismatch(saved, current, run_config):
    shared = dict(num_layers=2, hidden_size=64, num_attention_heads=4)
    checkpoint_args = SimpleNamespace(**shared)
    if saved is not None:
        checkpoint_args.post_embedding_rmsnorm = saved
    args = SimpleNamespace(**shared, post_embedding_rmsnorm=current)
    with patch("megatron.training.checkpointing.get_args", return_value=args):
        with pytest.raises(AssertionError, match="post_embedding_rmsnorm"):
            check_checkpoint_args(checkpoint_args)


@pytest.mark.parametrize("saved,current", [(None, False), (False, False), (True, True)])
def test_checkpoint_accepts_matching_post_embedding_rmsnorm(saved, current, run_config):
    shared = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        add_position_embedding=False,
        vocab_file=None,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
    )
    checkpoint_args = SimpleNamespace(**shared)
    if saved is not None:
        checkpoint_args.post_embedding_rmsnorm = saved
    args = SimpleNamespace(**shared, post_embedding_rmsnorm=current)
    with (
        patch("megatron.training.checkpointing.get_args", return_value=args),
        patch("megatron.training.checkpointing.get_checkpoint_version", return_value=3.0),
    ):
        check_checkpoint_args(checkpoint_args)


def _stack(config, *, pre_process=True, is_mtp_layer=False):
    # An empty physical stack isolates the real decoder boundary from layer math.
    return (
        HybridStack(
            config=config,
            submodules=hybrid_stack_spec.submodules,
            layer_config_list=[],
            pre_process=pre_process,
            post_process=False,
            is_mtp_layer=is_mtp_layer,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        .cuda()
        .to(dtype=config.params_dtype)
    )


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestHybridPostEmbeddingRMSNorm:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("fp32_residual", [False, True])
    def test_entry_norm_precedes_wide_expansion_and_has_gradients(self, fp32_residual):
        config = _config(
            post_embedding_rmsnorm=True,
            wide_residual=WideResidualConfig(num_streams=3),
            bf16=fp32_residual,
            params_dtype=torch.bfloat16 if fp32_residual else torch.float32,
            fp32_residual_connection=fp32_residual,
        )
        stack = _stack(config)
        torch.testing.assert_close(
            stack.input_norm.weight, torch.ones_like(stack.input_norm.weight), rtol=0, atol=0
        )
        inputs = torch.randn(4, 2, 64, device="cuda", requires_grad=True)
        with torch.no_grad():
            stack.input_norm.weight.copy_(torch.linspace(0.5, 1.5, 64, device="cuda"))
        norm_dtype = stack.input_norm.weight.dtype
        # TE casts its input/output to the norm parameter dtype. Match that numerical
        # boundary, then require the decoder to restore the FP32 residual dtype.
        expected = (
            F.rms_norm(
                inputs.to(norm_dtype).float(),
                (64,),
                stack.input_norm.weight.float(),
                eps=config.layernorm_epsilon,
            )
            .to(norm_dtype)
            .to(inputs.dtype)
        )
        with patch.object(
            hybrid_block_module,
            "expand_wide_residual_stream",
            wraps=hybrid_block_module.expand_wide_residual_stream,
        ) as expand:
            output = stack(WrappedTensor(inputs), attention_mask=None)
        assert expand.call_count == 1
        tolerance = dict(rtol=8e-3, atol=8e-3) if fp32_residual else {}
        assert expand.call_args.args[0].dtype == inputs.dtype
        torch.testing.assert_close(expand.call_args.args[0], expected, **tolerance)
        torch.testing.assert_close(output, expected.repeat(1, 1, 3), **tolerance)
        output.square().sum().backward()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert stack.input_norm.weight.grad is not None
        assert stack.input_norm.weight.grad.abs().sum() > 0
        assert output.dtype == inputs.dtype

    def test_entry_norm_precedes_mhc_expansion(self):
        config = _config(
            post_embedding_rmsnorm=True, enable_mhc_connections=True, mhc_num_residual_streams=3
        )
        stack = _stack(config)
        inputs = torch.randn(4, 2, 64, device="cuda", requires_grad=True)
        expected = F.rms_norm(inputs, (64,), stack.input_norm.weight, eps=config.layernorm_epsilon)
        with patch.object(
            hybrid_block_module.HyperConnectionModule,
            "input_expand",
            wraps=hybrid_block_module.HyperConnectionModule.input_expand,
        ) as expand:
            output = stack(inputs, attention_mask=None)
        assert expand.call_count == 1
        torch.testing.assert_close(expand.call_args.args[0], expected)
        torch.testing.assert_close(output, expected.repeat(1, 1, 3))

    @pytest.mark.parametrize("entry", ["standalone", "mimo"])
    def test_model_paths_normalize_once_and_keep_embedding_lookup_unchanged(self, entry):
        config = _config(post_embedding_rmsnorm=True)
        language_spec = ModuleSpec(
            module=HybridModel,
            params=dict(
                config=config,
                hybrid_stack_spec=hybrid_stack_spec,
                hybrid_layer_pattern="*-",
                vocab_size=128,
                max_sequence_length=4,
                position_embedding_type="none",
            ),
        )
        if entry == "mimo":
            model = MimoModel(
                MimoModelConfig(
                    language_model_spec=language_spec,
                    modality_submodules_spec={},
                    special_token_ids={},
                )
            ).cuda()
            language = model.language_model
        else:
            model = HybridModel(**language_spec.params).cuda()
            language = model
        tokens = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device="cuda")
        positions = torch.arange(4, device="cuda").expand(2, -1)
        raw_embedding = language.embedding(tokens, positions)
        torch.testing.assert_close(
            raw_embedding, language.embedding.word_embeddings(tokens).transpose(0, 1)
        )
        with patch.object(
            language.decoder.input_norm, "forward", wraps=language.decoder.input_norm.forward
        ) as normalize:
            result = model(input_ids=tokens, position_ids=positions, attention_mask=None)
        assert normalize.call_count == 1
        normalized_input = normalize.call_args.args[0]
        assert normalized_input.shape == raw_embedding.shape
        torch.testing.assert_close(normalized_input, raw_embedding)
        logits = result[0] if entry == "mimo" else result
        logits.square().mean().backward()
        assert language.decoder.input_norm.weight.grad is not None
        assert language.embedding.word_embeddings.weight.grad is not None

    @pytest.mark.parametrize("owner", ["disabled", "pipeline", "mtp"])
    def test_only_first_main_decoder_owns_norm(self, owner):
        config = _config(post_embedding_rmsnorm=owner != "disabled")
        stack = _stack(config, pre_process=owner != "pipeline", is_mtp_layer=owner == "mtp")
        assert stack.input_norm is None
        assert not any("input_norm" in key for key in stack.state_dict())
        inputs = torch.randn(4, 2, 64, device="cuda", requires_grad=True)
        if owner == "pipeline":
            stack.set_input_tensor(inputs)
            output = stack(None, attention_mask=None)
        else:
            output = stack(inputs, attention_mask=None)
        torch.testing.assert_close(output, inputs)

    def test_norm_state_roundtrip_and_sharded_key(self):
        config = _config(post_embedding_rmsnorm=True)
        stack = _stack(config)
        with torch.no_grad():
            stack.input_norm.weight.fill_(0.37)
        fresh = _stack(config)
        fresh.load_state_dict(stack.state_dict())
        torch.testing.assert_close(fresh.input_norm.weight, stack.input_norm.weight)
        sharded = fresh.sharded_state_dict(prefix="decoder.")
        assert "decoder.input_norm.weight" in sharded

    def test_input_norm_uses_existing_scalar_adam_decay_policy(self):
        stack = _stack(_config(post_embedding_rmsnorm=True))
        parameter = stack.input_norm.weight
        config = OptimizerConfig(
            optimizer="muon", muon_scalar_optimizer="adam", lr=0.1, weight_decay=0.25
        )
        overrides = get_standard_config_overrides(config)
        overrides.update(copy.deepcopy(_EMERGING_OPTIMIZERS["muon"].default_param_overrides))
        groups = _get_param_groups([stack], config, overrides)
        norm_groups = [
            group for group in groups if any(value is parameter for value in group["params"])
        ]
        assert len(norm_groups) == 1
        group = norm_groups[0]
        assert group["optimizer"] == config.muon_scalar_optimizer
        assert group["wd_mult"] == 0.0
        scalar_groups = [group for group in groups if group.get("optimizer") == "adam"]
        optimizer = torch.optim.AdamW(scalar_groups, lr=config.lr, foreach=False)
        OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=config.lr,
            max_lr=config.lr,
            min_lr=config.lr,
            lr_warmup_steps=0,
            lr_decay_steps=10,
            lr_decay_style="constant",
            start_wd=config.weight_decay,
            end_wd=config.weight_decay,
            wd_incr_steps=10,
            wd_incr_style="constant",
        )
        before = parameter.detach().clone()
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        assert group["weight_decay"] == 0.0
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cp_sp_partition_stays_local_at_input_norm():
    if Utils.world_size < 4:
        pytest.skip("Requires TP=2, CP=2")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _config(
            post_embedding_rmsnorm=True,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            sequence_parallel=True,
        )
        stack = _stack(config)
        inputs = torch.randn(4, 2, 64, device="cuda", requires_grad=True)
        output = stack(inputs, attention_mask=None)
        expected = F.rms_norm(inputs, (64,), stack.input_norm.weight, eps=config.layernorm_epsilon)
        assert output.shape == inputs.shape
        torch.testing.assert_close(output, expected)
        assert stack.input_norm.weight.sequence_parallel
        output.square().sum().backward()
        assert stack.input_norm.weight.grad is not None
    finally:
        Utils.destroy_model_parallel()
