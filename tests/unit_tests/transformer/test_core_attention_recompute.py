# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest import mock

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_multi_latent_attention import make_test_packed_seq_params


class TestCoreAttentionRecompute:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("variant", ["mha", "gated", "mla", "mla_up_proj"])
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    @pytest.mark.parametrize("backend", [AttnBackend.flash, AttnBackend.fused])
    def test_discard_output_and_backward(self, variant, packed, dropout, backend, monkeypatch):
        """Discard storage and preserve gradients across two outstanding forward graphs."""
        # These tests instantiate attention directly, without LanguageModule's backend setup.
        monkeypatch.setenv("NVTE_FLASH_ATTN", "1" if backend == AttnBackend.flash else "0")
        monkeypatch.setenv("NVTE_FUSED_ATTN", "1" if backend == AttnBackend.fused else "0")
        monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
        is_mla = variant.startswith("mla")

        def build_attention(recompute):
            options = dict(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                use_cpu_initialization=True,
                bf16=True,
                params_dtype=torch.bfloat16,
                attention_dropout=dropout,
                attention_backend=backend,
                recompute_granularity="selective" if recompute else None,
                recompute_modules=(
                    ["core_attn", "mla_up_proj"] if variant == "mla_up_proj" else ["core_attn"]
                ),
            )
            if is_mla:
                config = MLATransformerConfig(
                    **options,
                    multi_latent_attention=True,
                    q_lora_rank=32,
                    kv_lora_rank=32,
                    qk_head_dim=64,
                    qk_pos_emb_head_dim=64,
                    v_head_dim=64,
                )
                attention_cls = MLASelfAttention
            else:
                config = TransformerConfig(**options, attention_output_gate=variant == "gated")
                attention_cls = SelfAttention
            submodules = get_gpt_layer_with_transformer_engine_submodules(
                multi_latent_attention=is_mla
            ).self_attention.submodules
            return attention_cls(
                config, submodules, layer_number=1, attn_mask_type=AttnMaskType.causal
            ).cuda()

        reference = build_attention(False)
        candidate = build_attention(True)
        candidate.load_state_dict(reference.state_dict())
        shape = (32, 1 if packed else 2, 128)
        inputs = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
        grad_outputs = [torch.randn_like(x) for x in inputs]
        packed_seq_params = make_test_packed_seq_params(32) if packed else None
        saved_outputs = []
        checkpoint = tensor_parallel.CheckpointWithoutOutput.checkpoint

        def record_checkpoint(instance, *args, **kwargs):
            result = checkpoint(instance, *args, **kwargs)
            saved_outputs.extend(instance.outputs)
            return result

        def run(attention):
            model_parallel_cuda_manual_seed(456)
            xs = [x.detach().clone().requires_grad_() for x in inputs]
            outputs = []
            for x in xs:
                output, bias = attention(x, None, packed_seq_params=packed_seq_params)
                outputs.append(output if bias is None else output + bias)
            if attention is candidate:
                assert candidate._core_attn_checkpoint is None
                assert len(saved_outputs) == (8 if variant == "mla_up_proj" else 2)
                assert all(t.untyped_storage().nbytes() == 0 for t in saved_outputs)
            for output, grad in reversed(list(zip(outputs, grad_outputs))):
                output.backward(grad)
            return outputs, [x.grad for x in xs]

        expected_outputs, expected_grads = run(reference)
        with mock.patch.object(
            tensor_parallel.CheckpointWithoutOutput, "checkpoint", record_checkpoint
        ):
            actual_outputs, actual_grads = run(candidate)

        assert all(t.untyped_storage().nbytes() > 0 for t in saved_outputs)
        for expected, actual in zip(expected_outputs, actual_outputs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for expected, actual in zip(expected_grads, actual_grads):
            torch.testing.assert_close(actual, expected)
        for (name, expected), (actual_name, actual) in zip(
            reference.named_parameters(), candidate.named_parameters()
        ):
            assert name == actual_name
            assert (expected.grad is None) == (actual.grad is None), name
            if expected.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, msg=name)

    @pytest.mark.parametrize("core_recompute", [False, True])
    @pytest.mark.parametrize("raw_output", [False, True])
    @pytest.mark.parametrize("packed", [False, True])
    def test_layernorm_before_mla_recompute(self, core_recompute, raw_output, packed, monkeypatch):
        """Restore the enclosing layer's norm before replaying MLA on two live graphs."""
        monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
        monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
        modules = ["layernorm", "mla_up_proj"]
        if core_recompute:
            modules.append("core_attn")

        def build_layer(recompute):
            config = MLATransformerConfig(
                num_layers=1,
                hidden_size=512,
                num_attention_heads=8,
                ffn_hidden_size=1024,
                use_cpu_initialization=True,
                bf16=True,
                params_dtype=torch.bfloat16,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                attention_backend=AttnBackend.flash,
                normalization="RMSNorm",
                multi_latent_attention=True,
                q_lora_rank=None,
                kv_lora_rank=64,
                qk_head_dim=64,
                qk_pos_emb_head_dim=64,
                v_head_dim=64,
                recompute_granularity="selective" if recompute else None,
                recompute_modules=modules,
            )
            return TransformerLayer(
                config,
                get_gpt_layer_with_transformer_engine_submodules(multi_latent_attention=True),
                layer_number=1,
            ).cuda()

        reference = build_layer(False)
        candidate = build_layer(True)
        candidate.load_state_dict(reference.state_dict())
        shape = (32, 1 if packed else 2, 512)
        inputs = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
        grad_outputs = [torch.randn_like(x) for x in inputs]
        packed_seq_params = make_test_packed_seq_params(32) if packed else None
        saved_outputs = []
        norm_outputs = []
        dependencies = {}
        events = []
        checkpoint = tensor_parallel.CheckpointWithoutOutput.checkpoint
        recompute = tensor_parallel.CheckpointWithoutOutput._recompute

        def record_checkpoint(instance, *args, **kwargs):
            result = checkpoint(instance, *args, **kwargs)
            if instance is getattr(candidate, "input_layernorm_checkpoint", None):
                normalized = result[0] if isinstance(result, tuple) else result
                norm_outputs.append(normalized)
                dependencies[instance] = (len(norm_outputs) - 1, "norm")
                saved_outputs.append(normalized)
            elif instance is getattr(candidate.self_attention, "qkv_up_checkpoint", None):
                dependencies[instance] = (len(norm_outputs) - 1, "up")
                saved_outputs.extend(instance.outputs)
            elif instance is getattr(candidate.self_attention, "_core_attn_checkpoint", None):
                dependencies[instance] = (len(norm_outputs) - 1, "core")
                saved_outputs.extend(instance.outputs)
            return result

        def check_recompute(instance, grad):
            if instance.ctx is not None and instance in dependencies:
                graph, stage = dependencies[instance]
                if stage in ("up", "core"):
                    assert norm_outputs[graph].untyped_storage().nbytes() > 0
                    assert (graph, "norm") in events
                if stage == "core":
                    assert (graph, "up") in events
                events.append((graph, stage))
            return recompute(instance, grad)

        def run(layer):
            xs = [x.detach().clone().requires_grad_() for x in inputs]
            outputs = []
            for x in xs:
                if raw_output:
                    output, _, residual = layer._forward_self_attention_output_with_bias(
                        x, packed_seq_params=packed_seq_params
                    )
                    outputs.append(output[0] + residual)
                else:
                    outputs.append(layer(x, None, packed_seq_params=packed_seq_params)[0])
            if layer is candidate:
                assert len(norm_outputs) == 2
                assert all(t.untyped_storage().nbytes() == 0 for t in saved_outputs)
            for output, grad in reversed(list(zip(outputs, grad_outputs))):
                output.backward(grad)
            return outputs, [x.grad for x in xs]

        expected_outputs, expected_grads = run(reference)
        with (
            mock.patch.object(
                tensor_parallel.CheckpointWithoutOutput, "checkpoint", record_checkpoint
            ),
            mock.patch.object(
                tensor_parallel.CheckpointWithoutOutput, "_recompute", check_recompute
            ),
        ):
            actual_outputs, actual_grads = run(candidate)
        assert all(t.untyped_storage().nbytes() > 0 for t in saved_outputs)
        for graph in range(2):
            expected = ["norm", "up", "core"] if core_recompute else ["norm", "up"]
            assert [stage for index, stage in events if index == graph] == expected
        for expected, actual in zip(expected_outputs, actual_outputs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for expected, actual in zip(expected_grads, actual_grads):
            torch.testing.assert_close(actual, expected)
        for (name, expected), (actual_name, actual) in zip(
            reference.named_parameters(), candidate.named_parameters()
        ):
            assert name == actual_name
            assert (expected.grad is None) == (actual.grad is None), name
            if expected.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, msg=name)
