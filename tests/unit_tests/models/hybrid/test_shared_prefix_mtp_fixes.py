# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-prefix HybridModel MTP contracts: input validation, gradient determinism,
grouped loss normalization and the forest-stable router scope."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.models.hybrid.shared_prefix_layout import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, process_mtp_loss
from tests.unit_tests.test_utilities import Utils

_PADDING_MULTIPLE = 8
_VOCAB = 512


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _star(prefix_len, logical_lens):
    physical = tuple(
        _round_up(prefix_len + length, _PADDING_MULTIPLE) - prefix_len for length in logical_lens
    )
    return SharedPrefixLayout(
        prefix_len,
        physical,
        logical_completion_lens=tuple(logical_lens),
        padding_multiple=_PADDING_MULTIPLE,
    )


def _run_process_mtp_loss(weight, hidden_depths, input_ids, loss_mask, input_mask, lengths, groups):
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=input_ids.device
    )

    def output_layer(hidden, weight=None, runtime_gather_output=None):
        return hidden @ weight.t(), None

    def language_model_loss(labels, logits):
        return torch.nn.functional.cross_entropy(
            logits.transpose(0, 1).reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view(labels.shape)

    return process_mtp_loss(
        hidden_states=torch.cat(hidden_depths, dim=0),
        labels=None,
        loss_mask=loss_mask,
        output_layer=output_layer,
        output_weight=weight,
        runtime_gather_output=None,
        is_training=False,
        compute_language_model_loss=language_model_loss,
        config=SimpleNamespace(
            mtp_num_layers=len(hidden_depths) - 1,
            mtp_loss_scaling_factor=0.3,
            calculate_per_token_loss=True,
            mtp_detach_heads=False,
        ),
        packed_seq_params=PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        ),
        input_ids=input_ids,
        mtp_input_mask=input_mask,
        loss_group_lengths=groups,
    )


@pytest.mark.parametrize("use_input_mask", [False, True])
def test_grouped_mtp_normalization_equals_independent_forwards(use_input_mask):
    """One grouped process_mtp_loss call injects the gradients of one call per group."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator().manual_seed(3)
    depth, hidden, vocab = 2, 8, 11
    groups = ((6, 5), (7, 4, 3))
    lengths = [length for group in groups for length in group]
    total = sum(lengths)
    weight = torch.randn(vocab, hidden, generator=generator).to(device, torch.float64)
    input_ids = torch.randint(0, vocab, (1, total), generator=generator).to(device)
    loss_mask = (torch.rand(1, total, generator=generator) > 0.25).to(device, torch.float64)
    input_mask = (torch.rand(1, total, generator=generator) > 0.3).to(device)
    input_mask = input_mask if use_input_mask else None
    hidden_depths = [
        torch.randn(total, 1, hidden, generator=generator).to(device, torch.float64)
        for _ in range(depth + 1)
    ]
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0, device=device, dtype=torch.float64))
    grouped_inputs = [value.clone().requires_grad_() for value in hidden_depths]
    output = _run_process_mtp_loss(
        weight,
        grouped_inputs,
        input_ids,
        loss_mask,
        input_mask,
        lengths,
        tuple(sum(group) for group in groups),
    )
    output.sum().backward()

    start = 0
    for group in groups:
        span = slice(start, start + sum(group))
        independent_inputs = [value[span].clone().requires_grad_() for value in hidden_depths]
        output = _run_process_mtp_loss(
            weight,
            independent_inputs,
            input_ids[:, span],
            loss_mask[:, span],
            None if input_mask is None else input_mask[:, span],
            list(group),
            None,
        )
        output.sum().backward()
        for grouped, independent in zip(grouped_inputs[1:], independent_inputs[1:]):
            torch.testing.assert_close(grouped.grad[span], independent.grad, rtol=0, atol=1e-12)
        start = span.stop
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))


class _ReachedBackbone(Exception):
    pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSharedPrefixMTPModel:
    """Validation and execution tests on a tiny attention/MLP HybridModel with one MTP head."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _model(**overrides):
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
        from megatron.core.models.hybrid.hybrid_model import HybridModel
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.enums import AttnBackend

        config = dict(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=64,
            ffn_hidden_size=512,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            normalization="RMSNorm",
            add_bias_linear=False,
            params_dtype=torch.bfloat16,
            bf16=True,
            attention_backend=AttnBackend.flash,
            use_cpu_initialization=True,
            calculate_per_token_loss=True,
        )
        config.update(overrides)
        torch.manual_seed(0)
        return (
            HybridModel(
                config=TransformerConfig(**config),
                hybrid_stack_spec=hybrid_stack_spec,
                vocab_size=_VOCAB,
                max_sequence_length=4096,
                hybrid_layer_pattern="*-/*-",
                position_embedding_type="rope",
                share_embeddings_and_output_weights=False,
            )
            .cuda()
            .train()
        )

    @staticmethod
    def _inputs(layout):
        generator = torch.Generator().manual_seed(11)
        physical_len = _round_up(layout.total_len, _PADDING_MULTIPLE)
        input_ids = torch.randint(1, _VOCAB, (1, physical_len), generator=generator)
        loss_mask = torch.zeros(1, physical_len)
        for offset, root in layout.iter_roots():
            for branch, logical in zip(root.completion_slices(), root.logical_completion_lens):
                loss_mask[0, offset + branch.start : offset + branch.start + logical] = 1
        return input_ids.cuda(), loss_mask.cuda()

    def _forward(self, model, layout, **kwargs):
        input_ids, loss_mask = self._inputs(layout)
        return model(
            input_ids, None, None, loss_mask=loss_mask, shared_prefix_layout=layout, **kwargs
        )

    @pytest.mark.parametrize("ignored_input", ["mtp_input_mask", "decoder_input"])
    def test_training_mtp_rejects_inputs_it_would_ignore(self, ignored_input):
        model = self._model()
        layout = _star(16, (9, 23))
        input_ids, loss_mask = self._inputs(layout)
        if ignored_input == "mtp_input_mask":
            kwargs = {"mtp_input_mask": torch.ones_like(input_ids, dtype=torch.bool)}
        else:
            kwargs = {
                "decoder_input": torch.zeros(
                    input_ids.shape[1], 1, 256, device="cuda", dtype=torch.bfloat16
                )
            }
        with patch(
            "megatron.core.models.hybrid.hybrid_model.forward_hybrid_stack_shared_prefix",
            side_effect=_ReachedBackbone,
        ):
            with pytest.raises(NotImplementedError, match=ignored_input):
                model(
                    input_ids,
                    None,
                    None,
                    loss_mask=loss_mask,
                    shared_prefix_layout=layout,
                    **kwargs,
                )
            # Forwards that skip MTP, explicitly or in eval mode, accept them.
            with pytest.raises(_ReachedBackbone):
                model(
                    input_ids,
                    None,
                    None,
                    loss_mask=loss_mask,
                    shared_prefix_layout=layout,
                    compute_mtp_loss=False,
                    **kwargs,
                )
            model.eval()
            with pytest.raises(_ReachedBackbone):
                model(
                    input_ids,
                    None,
                    None,
                    loss_mask=loss_mask,
                    shared_prefix_layout=layout,
                    **kwargs,
                )

    @pytest.mark.parametrize("explicit_groups", [False, True])
    def test_forest_mtp_requires_per_token_loss_before_running(self, explicit_groups):
        model = self._model(calculate_per_token_loss=False)
        roots = (_star(16, (9, 23)), _star(8, (5,)))
        layout = SharedPrefixForestLayout(
            roots, mtp_loss_group_root_counts=(2,) if explicit_groups else ()
        )
        with patch(
            "megatron.core.models.hybrid.hybrid_model.forward_hybrid_stack_shared_prefix",
            side_effect=_ReachedBackbone,
        ):
            with pytest.raises(NotImplementedError, match="calculate_per_token_loss"):
                self._forward(model, layout)
            # A star has no loss groups and keeps the global normalization.
            with pytest.raises(_ReachedBackbone):
                self._forward(model, roots[0])
