# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Cross-layer sharing of the compact DSA indexer workspace under CUDA graphs."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import CompressedSparseAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

_CSA = "megatron.core.transformer.experimental_attention_variant.csa."


def _make_config(**overrides):
    kwargs = dict(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=16,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        csa_compress_ratios=[4, 128, 4, 128],
        csa_window_size=8,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=8,
        dsa_indexer_loss_coeff=1.0,
        dsa_indexer_use_sparse_loss=False,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


class TestCompactIndexerWorkspaceSharing:
    """Two ratio-4 CSA layers of one model: shared vs private compact-indexer workspaces."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        cls = request.cls
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        from megatron.core.models.common.embeddings import RotaryEmbedding

        cls.config_template = _make_config()
        cls.rotary_pos_emb = RotaryEmbedding(
            cls.config_template.qk_pos_emb_head_dim,
            rotary_percent=cls.config_template.rotary_percent,
            rotary_base=cls.config_template.rotary_base,
            cp_group=cls.pg_collection.cp,
        )
        yield
        Utils.destroy_model_parallel()

    def _submodules(self):
        # Same construction as tests/unit_tests/transformer/experimental_attention_variant/
        # test_attention_variant_csa.py (`_make_csa_submodules`).
        from tests.unit_tests.transformer.experimental_attention_variant import (
            test_attention_variant_csa as csa_tests,
        )

        return csa_tests._make_csa_submodules()

    def _layers(self, config):
        """Two CSA layers with compress ratio 4 (layers 1 and 3 of the template)."""
        config.cuda_graph_impl = "local"
        layers = []
        for layer_number in (1, 3):
            layers.append(
                CompressedSparseAttention(
                    config=config,
                    submodules=self._submodules(),
                    layer_number=layer_number,
                    attn_mask_type=AttnMaskType.causal,
                    attention_type='self',
                    pg_collection=self.pg_collection,
                    rotary_pos_emb=self.rotary_pos_emb,
                    compress_ratio=4,
                ).cuda()
            )
        return layers

    @staticmethod
    def _thd_call(layer, topk=8, **overrides):
        q = torch.empty(64, 8, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.empty(16, 128, dtype=torch.bfloat16, device="cuda")
        kwargs = dict(
            topk=topk,
            ratio=4,
            cu_seqlens_q=torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda"),
            cu_seqlens_k=torch.tensor([0, 8, 16], dtype=torch.int32, device="cuda"),
            max_seqlen_q=64,
            max_seqlen_k=8,
        )
        kwargs.update(overrides)
        return layer._get_thd_compact_indexer_workspace(q, k, **kwargs)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_shared_workspace_serves_capture_of_another_layer(self):
        """With sharing (default) a layer's capture reuses the workspace another layer warmed up."""
        config = _make_config()
        assert config.dsa_compact_indexer_workspace_sharing is True
        layer_a, layer_b = self._layers(config)
        workspace = MagicMock()
        workspace.matches.return_value = True
        with (
            patch(_CSA + "thd_compact_indexer_available", return_value=True),
            patch(
                _CSA + "prepare_thd_compact_indexer_workspace", return_value=workspace
            ) as prepare,
            patch.object(
                torch.cuda, "is_current_stream_capturing", side_effect=[False, True, True]
            ),
        ):
            assert self._thd_call(layer_a) is workspace  # eager warm-up on layer A
            assert self._thd_call(layer_b) is workspace  # capture on layer B: no own warm-up needed
            assert self._thd_call(layer_a) is workspace  # capture on layer A
        prepare.assert_called_once()
        assert layer_a._thd_compact_indexer_workspaces is layer_b._thd_compact_indexer_workspaces
        assert layer_a._thd_compact_indexer_workspaces == [workspace]
        assert layer_a._active_thd_compact_indexer_workspace is workspace
        assert layer_b._active_thd_compact_indexer_workspace is workspace
        # The balanced-indexer slots are shared the same way.
        assert (
            layer_a._balanced_thd_compact_indexer_workspaces
            is layer_b._balanced_thd_compact_indexer_workspaces
        )


class TestCompactIndexerWorkspaceSharingKernels:
    """Real compact-indexer kernels: two layers dispatched through one shared workspace must produce
    exactly what each produces through its own workspace (stale content from the other layer must
    not leak: quantize overwrites the staging buffers, the top-k kernel reads only them)."""

    @pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
    def test_interleaved_layers_through_one_workspace_match_private_workspaces(self, precision):
        import inspect

        from megatron.core.transformer.experimental_attention_variant.csa_utils import (
            fused_sparse_attention as dk,
        )
        from tests.unit_tests.transformer.experimental_attention_variant import (
            test_csa_fused_sparse_attention as fsa_tests,
        )

        fsa_tests._skip_if_real_kernels_unavailable()
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("compact THD indexer forward + Top-K requires SM100+")
        from cudnn import DSA

        if not hasattr(DSA, 'compress_topk_cand_buffer_size_thd'):
            pytest.skip("installed cuDNN Frontend lacks the compact THD workspace helper")
        compact_wrapper = getattr(DSA, "indexer_forward_top_k_wrapper", None)
        mxfp8_parameters = {"q_scale", "cu_seqlens_q_scale_padded", "cu_seqlens_k_scale_padded"}
        if precision == "mxfp8" and mxfp8_parameters - set(
            inspect.signature(compact_wrapper).parameters if callable(compact_wrapper) else ()
        ):
            pytest.skip("installed cuDNN Frontend lacks compact MXFP8 indexer support")

        ratio, topk, idx_nh, idx_hd = 4, 16, 64, 128
        q_lens, k_lens = [64, 96], [16, 24]
        max_seqlen_q, max_seqlen_k = 96, 24
        cu_q = fsa_tests._make_cu_seqlens(q_lens, device='cuda')
        cu_k = fsa_tests._make_cu_seqlens(k_lens, device='cuda')
        total_q, total_k = sum(q_lens), sum(k_lens)
        sm_scale = idx_hd**-0.5
        torch.manual_seed(0)
        layers = []
        for _ in range(2):  # two "layers" with the same geometry and different values
            q = torch.randn(total_q, idx_nh, idx_hd, dtype=torch.bfloat16, device='cuda')
            k = torch.randn(total_k, idx_hd, dtype=torch.bfloat16, device='cuda')
            w = torch.randn(total_q, idx_nh, dtype=torch.bfloat16, device='cuda').float() * sm_scale
            layers.append((q, k, w.to(torch.bfloat16)))

        def prepare(q, k):
            return dk.prepare_thd_compact_indexer_workspace(
                q,
                k,
                topk=topk,
                ratio=ratio,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                return_softmax=True,
                precision=precision,
            )

        def run(q, k, w, workspace):
            indices, lengths, _, softmax = dk._indexer_topk_core(
                q,
                k,
                w,
                topk=topk,
                ratio=ratio,
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_k,
                use_compact=True,
                return_softmax=True,
                compact_workspace=workspace,
                precision=precision,
            )
            torch.cuda.synchronize()
            return indices.clone(), lengths.clone(), softmax.clone()

        private = [prepare(q, k) for q, k, _ in layers]
        shared = prepare(*layers[0][:2])  # the geometry is the same for both layers
        assert shared is not None and all(ws is not None for ws in private)
        # cuDNN Frontend: eager calls perform value validation and JIT compilation first.
        for _ in range(3):
            run(*layers[0], private[0])
        reference = [run(*layers[i], private[i]) for i in (0, 1)]
        # Interleave the two layers through the one shared workspace: every dispatch finds the
        # other layer's stale quantized q/k and scales in the buffers.
        for i in (0, 1, 0, 1, 1, 0):
            indices, lengths, softmax = run(*layers[i], shared)
            assert torch.equal(indices, reference[i][0])
            assert torch.equal(lengths, reference[i][1])
            assert torch.equal(softmax, reference[i][2])
