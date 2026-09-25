# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real 1F1B execution and native checkpoint resharding for layer-owned Engram."""

import re
from collections import Counter
from contextlib import contextmanager

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, save
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.engram.hybrid_adapter import EngramHybridProvider
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.spec_utils import ModuleSpec
from tests.unit_tests.models.engram.test_integration import (
    make_hybrid,
    make_transformer_config,
    model_inputs,
    model_parallel,
    prepare_row_gradients,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(Utils.world_size != 2, reason='requires exactly two ranks')


@contextmanager
def _collective_assertions():
    error = None
    try:
        yield
    except AssertionError as exc:
        error = f'rank {torch.distributed.get_rank()}: {exc}'
    errors = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(errors, error)
    assert not any(errors), errors


def _models(pp_size, vp_size=None, backend='local', recompute=False, mtp=False):
    config = make_transformer_config(
        num_layers=8,
        engram_layer_ids=[1, 3],
        engram_table_backend=backend,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vp_size,
        pipeline_dtype=torch.float32,
        batch_p2p_comm=True,
        recompute_granularity='full' if recompute else None,
        recompute_method='uniform' if recompute else None,
        recompute_num_layers=1 if recompute else None,
        mtp_num_layers=1 if mtp else None,
        mtp_loss_scaling_factor=0.3,
        num_moe_experts=4 if mtp else None,
        moe_ffn_hidden_size=32 if mtp else None,
        moe_router_topk=2,
        moe_aux_loss_coeff=0.01 if mtp else 0.0,
    )
    segment_count = pp_size * (vp_size or 1)
    pattern = '|'.join(['*-' * (4 // segment_count)] * segment_count)
    if mtp:
        pattern += '/*E'
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    models = []
    for chunk in range(vp_size or 1):
        model = make_hybrid(
            config,
            pattern,
            pre_process=pp_rank == 0 and chunk == 0,
            post_process=pp_rank == pp_size - 1 and chunk == (vp_size or 1) - 1,
            vp_stage=chunk if vp_size else None,
        )
        models.append(model)
    return models


def _state(models):
    return {f'model{index}': model.sharded_state_dict() for index, model in enumerate(models)}


def _save(models, path):
    if torch.distributed.get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()
    save(_state(models), path)


def _restore(models, path):
    state = load(_state(models), path)
    for index, model in enumerate(models):
        result = model.load_state_dict(state[f'model{index}'], strict=True)
        assert not result.missing_keys and not result.unexpected_keys


def _run(models):
    def forward_step(data_iterator, model):
        sample = next(data_iterator)
        tokens, positions, attention_mask = model_inputs()
        tokens = tokens + sample
        input_tokens = (
            tokens
            if model.pre_process or model.requires_token_context or model.mtp_process
            else None
        )
        output = model(
            input_tokens,
            positions,
            attention_mask,
            labels=tokens.roll(-1, dims=1),
            loss_mask=torch.ones_like(tokens, dtype=torch.float32),
        )

        def loss_func(loss):
            return loss.mean(), {'ce': loss.detach().mean(), 'sample': sample}

        return output, loss_func

    return get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=[iter(range(4)) for _ in models],
        model=models,
        num_microbatches=4,
        seq_length=8,
        micro_batch_size=1,
        forward_only=False,
    )


def _global_name(model, name):
    match = re.match(r'decoder.layers.(\d+)\.(.*)', name)
    if match:
        layer = model.decoder.layers[int(match[1])]
        return f'decoder.layers.{layer.layer_number - 1}.{match[2]}'
    return name


def _run_mtp_with_context_trace(models, monkeypatch):
    """Record consumer tokens in both forward and recompute, plus the real MTP loss."""
    monkeypatch.setattr(MTPLossLoggingHelper, 'tracker', {})
    destroy_moe_metrics_tracker()
    contexts = {}
    handles = []

    def record_context(layer, args, kwargs):
        contexts.setdefault(layer.engram_memory_id, []).append(
            kwargs['token_context'].detach().cpu().clone()
        )

    for model in models:
        for layer in model.decoder.layers:
            if hasattr(layer, 'engram'):
                handles.append(layer.register_forward_pre_hook(record_context, with_kwargs=True))
    try:
        losses = _run(models)
    finally:
        for handle in handles:
            handle.remove()
    mtp_loss = MTPLossLoggingHelper.tracker.get('loss_values')
    return losses, None if mtp_loss is None else mtp_loss.detach().clone(), contexts


@pytest.mark.parametrize('vp_size,mtp', [(None, False), (2, False), (2, True)])
@pytest.mark.parametrize('backend', ['local', 'row_a2a'])
def test_pipeline_microbatches_and_native_checkpoint_resharding(
    tmp_path_dist_ckpt, monkeypatch, vp_size, mtp, backend
):
    """Compare PP/VPP forward, recompute, gradients and resumed MTP/token contexts."""
    suffix = f'{backend}_{vp_size}_{mtp}'
    source_path, target_path = (
        tmp_path_dist_ckpt / (label + suffix) for label in ('source_', 'pipeline_')
    )
    with model_parallel():
        torch.manual_seed(2026)
        model_parallel_cuda_manual_seed(2026)
        source = _models(1, backend=backend, mtp=mtp)
        _save(source, source_path)
        expected_losses, expected_mtp, _ = _run_mtp_with_context_trace(source, monkeypatch)
        expected_weights = {
            n: p.detach().clone()
            for n, p in source[0].named_parameters()
            if backend == 'local' or 'multi_head_embedding' not in n
        }
        expected_grads = {
            n: p.grad.detach().clone()
            for n, p in source[0].named_parameters()
            if p.grad is not None
        }
        assert (expected_mtp is not None) == mtp
        if mtp:
            assert torch.isfinite(expected_mtp).all()
        del source
    with model_parallel(
        pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=vp_size
    ):
        for checkpoint in (source_path, target_path):
            models = _models(2, vp_size, backend=backend, recompute=True, mtp=mtp)
            _restore(models, checkpoint)
            with _collective_assertions():
                for model in models:
                    for name, param in model.named_parameters():
                        key = _global_name(model, name)
                        if key in expected_weights:
                            torch.testing.assert_close(param, expected_weights[key], rtol=0, atol=0)
            losses, mtp_loss, contexts = _run_mtp_with_context_trace(models, monkeypatch)
            embedding_key = 'embedding.word_embeddings.weight'
            embedding_grad = torch.zeros_like(expected_grads[embedding_key])
            for model in models:
                if hasattr(model, 'embedding'):
                    embedding_grad.add_(model.embedding.word_embeddings.weight.grad)
            # Bare modules need the native finalizer's cross-stage embedding reduction.
            torch.distributed.all_reduce(embedding_grad, group=models[0].pg_collection.pp)
            with _collective_assertions():
                torch.testing.assert_close(
                    embedding_grad, expected_grads[embedding_key], rtol=3e-4, atol=2e-6
                )
                last_rank = parallel_state.get_pipeline_model_parallel_rank() == 1
                assert len(losses) == (4 if last_rank else 0)
                for actual, expected in zip(losses, expected_losses):
                    assert actual['sample'] == expected['sample']
                    torch.testing.assert_close(actual['ce'], expected['ce'], rtol=1e-5, atol=1e-6)
                if mtp and last_rank:
                    torch.testing.assert_close(mtp_loss, expected_mtp, rtol=1e-5, atol=1e-6)
                    assert [model.mtp_process for model in models] == [False, True]
                    assert not any(hasattr(layer, 'engram') for layer in models[-1].mtp.modules())
                else:
                    assert mtp_loss is None
                local_ids = {
                    layer.engram_memory_id
                    for model in models
                    for layer in model.decoder.layers
                    if hasattr(layer, 'engram')
                }
                assert set(contexts) == local_ids
                for recorded in contexts.values():
                    samples = [int(tokens[0, 0]) - 1 for tokens in recorded]
                    assert Counter(samples) == Counter({sample: 2 for sample in range(4)})
                    for tokens, sample in zip(recorded, samples):
                        torch.testing.assert_close(
                            tokens, torch.arange(1, 9).unsqueeze(0) + sample, rtol=0, atol=0
                        )
                for model in models:
                    assert model.requires_token_context == any(
                        hasattr(layer, 'engram') for layer in model.decoder.layers
                    )
                    for name, param in model.named_parameters():
                        key = _global_name(model, name)
                        if key != embedding_key and key in expected_grads:
                            torch.testing.assert_close(
                                param.grad, expected_grads[key], rtol=3e-4, atol=2e-6, msg=key
                            )
            if checkpoint == source_path:
                _save(models, target_path)
            del models
    with model_parallel():
        restored = _models(1, backend=backend, mtp=mtp)
        _restore(restored, target_path)
        losses, mtp_loss, _ = _run_mtp_with_context_trace(restored, monkeypatch)
        for actual, expected in zip(losses, expected_losses, strict=True):
            torch.testing.assert_close(actual['ce'], expected['ce'], rtol=1e-5, atol=1e-6)
        if mtp:
            torch.testing.assert_close(mtp_loss, expected_mtp, rtol=1e-5, atol=1e-6)
    destroy_moe_metrics_tracker()
