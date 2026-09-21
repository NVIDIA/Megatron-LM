# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Distributed correctness tests for the gated-residual hyper-connection variant.

Covers GR-3:
1. sequence_parallel gradient semantics: with TP2+SP, the TP all-reduce of the
   replicated GR parameters' gradients reproduces the full-sequence reference.
2. context_parallel gradient semantics: the same property across CP ranks.

The GR + MTP end-to-end case ran on GPTModel and is dropped on this base: the
gated-residual variant is HybridModel-only here (config validation rejects the
GPT path), so that scenario is covered by the hybrid MTP head test in
test_gated_residual_wiring.py instead.
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.gated_residual import GatedResidualModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_SEED = 1234
HIDDEN = 64
STREAMS = 4
LOWRANK = 16


class TestGatedResidualSequenceParallelGrads:
    """TP2+SP: all-reduced GR gradients must equal the full-sequence reference."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(_SEED)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2+ GPUs")
    def test_sp_grad_allreduce_matches_full_sequence(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            use_cpu_initialization=True,
            is_hybrid_model=True,
            enable_mhc_connections=True,
            mhc_num_residual_streams=STREAMS,
            mhc_connection_variant="gated_residual",
            hc_lowrank=LOWRANK,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        torch.manual_seed(_SEED)
        module = GatedResidualModule(config, layer_number=1).cuda()
        for name, param in module.named_parameters():
            assert getattr(param, 'sequence_parallel', False), name

        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        seq, batch = 16, 2
        torch.manual_seed(_SEED + 1)  # identical on every rank
        x_full = torch.randn(seq, batch, STREAMS * HIDDEN, device="cuda")

        def run(x):
            module.zero_grad(set_to_none=True)
            mixed, h_res, g_write, residual = module(x.detach().clone().requires_grad_(True))
            out = module.fused_h_res_h_post_bda(
                h_res,
                residual,
                g_write,
                (mixed, None),  # feed mixed back as a stand-in sublayer output
                dropout_prob=0.0,
                training=True,
                fused=False,
            )
            out.float().square().sum().backward()
            return {n: p.grad.detach().clone() for n, p in module.named_parameters()}

        # Full-sequence reference (identical on every rank).
        grads_ref = run(x_full)

        # SP run: each TP rank sees its contiguous sequence shard, then the
        # marked gradients are all-reduced over the TP group — exactly what
        # finalize_model_grads does for sequence_parallel-marked params.
        shard = seq // tp_size
        x_local = x_full[tp_rank * shard : (tp_rank + 1) * shard]
        grads_sp = run(x_local)
        for name in grads_sp:
            torch.distributed.all_reduce(grads_sp[name], group=tp_group)

        for name in grads_ref:
            torch.testing.assert_close(
                grads_sp[name], grads_ref[name], atol=1e-4, rtol=1e-4, msg=name
            )


class TestGatedResidualContextParallelGrads:
    """CP2: gradients reduced over the CP group must equal the full-sequence
    reference.

    GR is a per-token operation, so context parallelism needs no GR-side code:
    each CP rank processes its sequence chunks independently, and the replicated
    GR parameters' gradients are summed by the regular DDP reduction, which runs
    over the DP x CP group (``finalize_model_grads.py`` /
    ``distributed_data_parallel.py`` use
    ``get_data_parallel_group(with_context_parallel=True)``). This test emulates
    that reduction with an explicit all-reduce over the CP group.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, context_parallel_size=2
        )
        model_parallel_cuda_manual_seed(_SEED)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2+ GPUs")
    def test_cp_grad_allreduce_matches_full_sequence(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            use_cpu_initialization=True,
            is_hybrid_model=True,
            enable_mhc_connections=True,
            mhc_num_residual_streams=STREAMS,
            mhc_connection_variant="gated_residual",
            hc_lowrank=LOWRANK,
            context_parallel_size=2,
        )
        torch.manual_seed(_SEED)
        module = GatedResidualModule(config, layer_number=1).cuda()

        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = parallel_state.get_context_parallel_rank()
        cp_size = parallel_state.get_context_parallel_world_size()

        seq, batch = 16, 2
        torch.manual_seed(_SEED + 1)  # identical on every rank
        x_full = torch.randn(seq, batch, STREAMS * HIDDEN, device="cuda")

        def run(x):
            module.zero_grad(set_to_none=True)
            mixed, h_res, g_write, residual = module(x.detach().clone().requires_grad_(True))
            out = module.fused_h_res_h_post_bda(
                h_res,
                residual,
                g_write,
                (mixed, None),  # feed mixed back as a stand-in sublayer output
                dropout_prob=0.0,
                training=True,
                fused=False,
            )
            out.float().square().sum().backward()
            return {n: p.grad.detach().clone() for n, p in module.named_parameters()}

        # Full-sequence reference (identical on every rank).
        grads_ref = run(x_full)

        # CP run on this rank's chunks, using the real load-balanced CP layout:
        # the sequence is split into 2*cp chunks and rank r owns chunks r and
        # 2*cp-1-r. A per-token op is split-scheme-agnostic, but mirror the real
        # assignment anyway.
        chunks = x_full.chunk(2 * cp_size, dim=0)
        x_local = torch.cat([chunks[cp_rank], chunks[2 * cp_size - 1 - cp_rank]], dim=0)
        grads_cp = run(x_local)
        for name in grads_cp:
            torch.distributed.all_reduce(grads_cp[name], group=cp_group)

        for name in grads_ref:
            torch.testing.assert_close(
                grads_cp[name], grads_ref[name], atol=1e-4, rtol=1e-4, msg=name
            )
