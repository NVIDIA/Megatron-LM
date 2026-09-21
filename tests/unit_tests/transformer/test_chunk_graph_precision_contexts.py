# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Precision handling inside chunk-granularity CUDA graph captures: BF16 boundary layers must opt
out of the block-wide FP8 context, and activation checkpointing must go through TE's checkpoint
(mcore's tensor_parallel.checkpoint skips the checkpoint node while a graph is captured)."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp8_utils import get_fp8_recipe, get_layer_fp8_context
from megatron.core.recompute import use_te_checkpoint
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**overrides):
    kwargs = dict(
        num_layers=4, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True, bf16=True
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


_CHUNK = dict(
    cuda_graph_impl="transformer_engine",
    cuda_graph_granularity="chunk",
    cuda_graph_modules=[],
    cuda_graph_dynamic_microbatches=True,
    sequence_packing_scheduler="dp_balanced",
    pad_packed_seq_alignment="max",
    max_seqlen_per_dp_cp_rank=128,
    thd_max_packed_sequences=8,
)


class TestChunkGraphPrecisionContexts:
    @pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not available")
    def test_use_te_checkpoint_decision(self):
        # FP8 always goes through TE's checkpoint.
        assert use_te_checkpoint(_config(fp8="e4m3", fp8_recipe="mxfp8"))
        # BF16: mcore's checkpoint outside graphs ...
        assert not use_te_checkpoint(_config())
        assert not use_te_checkpoint(
            _config(cuda_graph_impl="transformer_engine", cuda_graph_granularity="layer")
        )
        # ... but TE's checkpoint inside a chunk capture, where mcore's would skip checkpointing.
        assert use_te_checkpoint(_config(**_CHUNK))

    @pytest.mark.skipif(not (HAVE_TE and torch.cuda.is_available()), reason="TE and CUDA required")
    def test_selective_core_attn_recompute_checkpoints_inside_chunk_capture(self):
        """Selective ``core_attn`` recompute: eager builds one checkpoint node per layer, mcore's
        ``tensor_parallel.checkpoint`` builds none while a graph is captured, and under chunk
        granularity the attention uses TE's checkpoint, which does build its node in the capture."""
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer import cuda_graphs
        from megatron.core.transformer.spec_utils import build_module
        from tests.unit_tests.test_utilities import Utils

        def checkpoint_nodes(tensor):
            names, seen, stack = [], set(), [tensor.grad_fn]
            while stack:
                fn = stack.pop()
                if fn is None or fn in seen:
                    continue
                seen.add(fn)
                if "Checkpoint" in type(fn).__name__:
                    names.append(type(fn).__name__)
                stack.extend(next_fn for next_fn, _ in fn.next_functions)
            return sorted(names)

        def run(**overrides):
            config = _config(
                num_layers=1,
                params_dtype=torch.bfloat16,
                attention_dropout=0.0,  # graphed attention recompute asserts no dropout
                hidden_dropout=0.0,
                recompute_granularity="selective",
                recompute_modules=["core_attn"],
                **overrides,
            )
            layer = build_module(
                get_gpt_layer_with_transformer_engine_spec(), config=config, layer_number=1
            ).cuda()
            layer.train()
            hidden = torch.randn(16, 2, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
            output, _ = layer(hidden_states=hidden, attention_mask=None)
            return checkpoint_nodes(output)

        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)
        try:
            assert run() == ["CheckpointFunctionBackward"]  # eager: mcore's node
            cuda_graphs._set_capture_start()
            try:
                assert run() == []  # mcore's checkpoint skips its node while a graph is captured
                assert run(**_CHUNK) == ["_CheckpointFunctionBackward"]  # TE's does not
            finally:
                cuda_graphs._set_capture_end()
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.skipif(not (HAVE_TE and torch.cuda.is_available()), reason="TE and CUDA required")
    def test_layer_fp8_context_opts_bf16_boundary_layers_out_of_a_chunk_capture(self):
        """Layer 0 and the last layer are BF16; inside an outer FP8 context (what the chunk
        capture applies to the whole block) their per-layer context must disable FP8."""
        import transformer_engine.pytorch as te
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        config = _config(
            fp8="e4m3",
            fp8_recipe="mxfp8",
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=1,
            num_layers_at_end_in_bf16=1,
            **_CHUNK,
        )
        with te.fp8_autocast(enabled=True, fp8_recipe=get_fp8_recipe(config)):
            assert FP8GlobalStateManager.is_fp8_enabled()
            # decoder layers 0 and 3 are BF16; an MTP layer (index >= num_layers) keeps the
            # FP8 of the enclosing MTP-block context, exactly as in eager.
            for layer_no, expect_fp8 in ((0, False), (1, True), (2, True), (3, False), (4, True)):
                with get_layer_fp8_context(config, layer_no):
                    assert FP8GlobalStateManager.is_fp8_enabled() is expect_fp8, layer_no
            # The MTP stack numbers its layers from 1: layer 0 there is an FP8 layer, not the
            # first BF16 decoder layer.
            with get_layer_fp8_context(config, 0, is_mtp_layer=True):
                assert FP8GlobalStateManager.is_fp8_enabled()
            # The block's per-layer context is this helper (layers are numbered from 1).
            block = object.__new__(TransformerBlock)
            object.__setattr__(block, "config", config)
            with block._get_inner_quantization_context(SimpleNamespace(layer_number=1)):
                assert not FP8GlobalStateManager.is_fp8_enabled()
            with block._get_inner_quantization_context(SimpleNamespace(layer_number=2)):
                assert FP8GlobalStateManager.is_fp8_enabled()
        # During warm-up and capture, TE's autocast wrapper on the graphed block *class* runs
        # the MTP stack (same class, but captured inside the post-process block's graph rather
        # than a graphed callable itself) with FP8 *disabled*; its layers must switch FP8 back
        # on themselves (their parameters are FP8), while a BF16 decoder layer stays off.
        with te.fp8_autocast(enabled=False, fp8_recipe=get_fp8_recipe(config)):
            with get_layer_fp8_context(config, 0, is_mtp_layer=True):
                assert FP8GlobalStateManager.is_fp8_enabled()
            with get_layer_fp8_context(config, 1):
                assert FP8GlobalStateManager.is_fp8_enabled()
            with get_layer_fp8_context(config, 0):
                assert not FP8GlobalStateManager.is_fp8_enabled()
        # Outside chunk captures the boundary layers simply get no FP8 context of their own
        # (per-layer capture never runs them under an outer one).
        config.cuda_graph_granularity = "layer"
        with te.fp8_autocast(enabled=True, fp8_recipe=get_fp8_recipe(config)):
            with get_layer_fp8_context(config, 0):
                assert FP8GlobalStateManager.is_fp8_enabled()
