# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-prefix HybridModel MTP contracts: input validation, gradient determinism,
grouped loss normalization and the forest-stable router scope."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.models.hybrid.hybrid_model import (
    _fixed_order_copy_contributors,
    _gather_shared_prefix_mtp_rows,
    _shared_prefix_mtp_branch_indices,
)
from megatron.core.models.hybrid.shared_prefix_layout import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.moe import moe_utils
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


def _forest():
    return SharedPrefixForestLayout((_star(13, (5, 17, 9)), _star(6, (11, 3))))


def _brute_force_contributors(indices, num_rows, copies):
    expected = torch.full((copies, num_rows), indices.numel(), dtype=torch.long)
    seen = [0] * num_rows
    for position, row in enumerate(indices.tolist()):
        expected[seen[row], row] = position
        seen[row] += 1
    return expected


@pytest.mark.parametrize("cp_size", [1, 2])
def test_fixed_order_copy_contributors_match_brute_force(cp_size):
    layout = _forest()
    copies = max(len(root.completion_lens) for _, root in layout.iter_roots())
    num_rows = layout.total_len + 3
    for cp_rank in range(cp_size):
        star_indices, _ = _shared_prefix_mtp_branch_indices(
            layout, "cpu", cp_size=cp_size, cp_rank=cp_rank
        )
        indices = torch.cat(star_indices)
        actual = _fixed_order_copy_contributors(indices, num_rows, copies)
        torch.testing.assert_close(
            actual, _brute_force_contributors(indices, num_rows, copies), rtol=0, atol=0
        )


def _ordered_gradient(indices, cotangent, num_rows):
    """Sum every copy in FP32 (FP64 for FP64) in index order, then round once."""
    accumulation_dtype = torch.float64 if cotangent.dtype == torch.float64 else torch.float32
    expected = torch.zeros((num_rows, *cotangent.shape[1:]), dtype=accumulation_dtype)
    for position, row in enumerate(indices.tolist()):
        expected[row] += cotangent[position].cpu().to(accumulation_dtype)
    return expected.to(cotangent.dtype)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_fixed_order_row_gather_matches_ordered_reference(dtype, cp_size):
    layout = _forest()
    num_rows = layout.total_len + 2
    generator = torch.Generator().manual_seed(17)
    value = torch.randn(num_rows, 1, 32, generator=generator).to(dtype)
    prompt_rows = [
        offset + row for offset, root in layout.iter_roots() for row in range(root.prefix_len)
    ]
    fewer_local_copies = False
    for cp_rank in range(cp_size):
        star_indices, _ = _shared_prefix_mtp_branch_indices(
            layout, "cpu", cp_size=cp_size, cp_rank=cp_rank
        )
        indices = torch.cat(star_indices)
        counts = torch.bincount(indices, minlength=num_rows)
        # Prompt rows whose root has fewer branches than G, or that this CP rank holds
        # in only some of its branches, take the sentinel path.
        fewer_local_copies |= bool((counts[prompt_rows] < 3).any())
        cotangent = torch.randn(indices.numel(), 1, 32, generator=generator).to(dtype)
        actual_input = value.clone().requires_grad_()
        actual = _gather_shared_prefix_mtp_rows(actual_input, indices, layout)
        actual.backward(cotangent)
        torch.testing.assert_close(actual, value.index_select(0, indices), rtol=0, atol=0)
        torch.testing.assert_close(
            actual_input.grad, _ordered_gradient(indices, cotangent, num_rows), rtol=0, atol=0
        )
    assert fewer_local_copies


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fixed_order_row_gather_backward_is_bitwise_reproducible_on_cuda():
    # Prompt rows copied once per branch: index_select's BF16 atomic backward differs
    # run to run at this size; the fixed-order gather must not.
    prompt, completions = 64, (208, 264, 336, 192)
    layout = SharedPrefixLayout(prompt, completions)
    indices = torch.cat(layout.dense_branch_indices("cuda"))
    generator = torch.Generator(device="cuda").manual_seed(5)
    value = torch.randn(
        layout.total_len, 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    cotangent = torch.randn(
        indices.numel(), 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator
    )

    def gradient():
        source = value.clone().requires_grad_()
        _gather_shared_prefix_mtp_rows(source, indices, layout).backward(cotangent)
        return source.grad

    reference = gradient()
    for _ in range(10):
        assert torch.equal(gradient(), reference)
    expected = _ordered_gradient(indices, cotangent, layout.total_len)
    torch.testing.assert_close(reference.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fixed_order_row_gather_backward_memory_matches_index_select():
    """Only prompt rows pay for the FP32 accumulation; the rest gather in BF16."""
    prompt, completions, hidden = 256, (2048,) * 8, 512
    layout = SharedPrefixLayout(prompt, completions)
    indices = torch.cat(layout.dense_branch_indices("cuda"))
    value = torch.randn(layout.total_len, 1, hidden, device="cuda", dtype=torch.bfloat16)
    cotangent = torch.randn(indices.numel(), 1, hidden, device="cuda", dtype=torch.bfloat16)

    def backward_peak(gather):
        source = value.clone().requires_grad_()
        output = gather(source)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        output.backward(cotangent)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - baseline

    index_select_peak = backward_peak(lambda source: source.index_select(0, indices))
    fixed_order_peak = backward_peak(
        lambda source: _gather_shared_prefix_mtp_rows(source, indices, layout)
    )
    # FP32 prompt-row accumulator plus one BF16 prompt-row copy and the int64 masks.
    prompt_bytes = prompt * hidden * (4 + 2) + 8 * layout.total_len
    assert fixed_order_peak <= index_select_peak + prompt_bytes + (1 << 20)


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
    try:
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
    finally:
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
            "megatron.core.models.hybrid.shared_prefix.forward_hybrid_stack_shared_prefix",
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
            "megatron.core.models.hybrid.shared_prefix.forward_hybrid_stack_shared_prefix",
            side_effect=_ReachedBackbone,
        ):
            with pytest.raises(NotImplementedError, match="calculate_per_token_loss"):
                self._forward(model, layout)
            # A star has no loss groups and keeps the global normalization.
            with pytest.raises(_ReachedBackbone):
                self._forward(model, roots[0])

    def test_mtp_runs_inside_router_gating_token_blocks(self):
        model = self._model()
        block_sizes = []
        original_forward = model.mtp.forward

        def recording_forward(*args, **kwargs):
            block_sizes.append(moe_utils._ROUTER_GATING_TOKEN_BLOCK_SIZE.get())
            return original_forward(*args, **kwargs)

        model.mtp.forward = recording_forward
        self._forward(model, _star(16, (9, 23)))
        assert len(block_sizes) == 1 and block_sizes[0] is not None

    def test_shared_star_mtp_backbone_gradients_are_reproducible(self, monkeypatch):
        """Prompt rows feed every dense MTP branch; their gradients must sum reproducibly."""
        # Pin the attention backward so only the MTP gather is under test.
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        model = self._model()
        layout = _star(96, (203, 260, 333, 190))
        input_ids, loss_mask = self._inputs(layout)
        generator = torch.Generator(device="cuda").manual_seed(29)
        cotangent = torch.randn(
            1, input_ids.shape[1], _VOCAB, device="cuda", generator=generator
        ) * loss_mask.unsqueeze(-1)
        MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0, device="cuda"))

        def gradients():
            model.zero_grad(set_to_none=True)
            logits = model(input_ids, None, None, loss_mask=loss_mask, shared_prefix_layout=layout)
            (logits.float() * cotangent).sum().backward()
            return {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and not name.startswith("mtp.")
            }

        reference = gradients()
        assert reference
        for _ in range(3):
            for name, gradient in gradients().items():
                assert torch.equal(gradient, reference[name]), name
