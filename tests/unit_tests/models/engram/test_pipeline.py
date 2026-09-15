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
from tests.unit_tests.models.engram.test_integration import make_transformer_config, model_inputs
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
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            hybrid_layer_pattern=pattern,
            vocab_size=64,
            max_sequence_length=8,
            position_embedding_type='none',
            pre_process=pp_rank == 0 and chunk == 0,
            post_process=pp_rank == pp_size - 1 and chunk == (vp_size or 1) - 1,
            vp_stage=chunk if vp_size else None,
            token_context_provider_spec=ModuleSpec(
                module=EngramHybridProvider,
                params=dict(
                    tokenizer_lookup=torch.arange(64), pad_id=2, hybrid_layer_pattern=pattern
                ),
            ),
        ).cuda()
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


@pytest.mark.parametrize('vp_size', [None, 2])
@pytest.mark.parametrize('backend', ['local', 'row_a2a'])
def test_pipeline_microbatches_and_native_checkpoint_resharding(
    tmp_path_dist_ckpt, vp_size, backend
):
    Utils.initialize_model_parallel()
    try:
        torch.manual_seed(2026)
        model_parallel_cuda_manual_seed(2026)
        source = _models(1, backend=backend)
        source_path = tmp_path_dist_ckpt / f'engram_source_{backend}_{vp_size}'
        _save(source, source_path)
        reference_losses = _run(source)
        reference_parameters = {
            name: parameter.detach().clone()
            for name, parameter in source[0].named_parameters()
            if backend == 'local' or 'multi_head_embedding' not in name
        }
        reference_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in source[0].named_parameters()
            if parameter.grad is not None and not parameter.grad.is_sparse
        }
        del source
    finally:
        Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=vp_size
    )
    try:
        models = _models(2, vp_size, backend=backend, recompute=True)
        _restore(models, source_path)
        with _collective_assertions():
            for model in models:
                for name, parameter in model.named_parameters():
                    key = _global_name(model, name)
                    if key in reference_parameters:
                        torch.testing.assert_close(
                            parameter, reference_parameters[key], rtol=0, atol=0, msg=key
                        )
        actual_losses = _run(models)
        with _collective_assertions():
            if parallel_state.get_pipeline_model_parallel_rank() == 1:
                assert len(actual_losses) == 4
                for actual, expected in zip(actual_losses, reference_losses, strict=True):
                    assert actual['sample'] == expected['sample']
                    torch.testing.assert_close(actual['ce'], expected['ce'], rtol=1e-5, atol=1e-6)
            else:
                assert actual_losses == []
            for model in models:
                assert model.requires_token_context == any(
                    hasattr(layer, 'engram') for layer in model.decoder.layers
                )
                for name, parameter in model.named_parameters():
                    key = _global_name(model, name)
                    if key in reference_gradients:
                        torch.testing.assert_close(
                            parameter.grad, reference_gradients[key], rtol=3e-4, atol=2e-6
                        )
        target_path = tmp_path_dist_ckpt / f'engram_pipeline_{backend}_{vp_size}'
        _save(models, target_path)
        del models
    finally:
        Utils.destroy_model_parallel()

    Utils.initialize_model_parallel()
    try:
        restored = _models(1, backend=backend)
        _restore(restored, target_path)
        resumed_losses = _run(restored)
        for actual, expected in zip(resumed_losses, reference_losses, strict=True):
            torch.testing.assert_close(actual['ce'], expected['ce'], rtol=1e-5, atol=1e-6)
    finally:
        Utils.destroy_model_parallel()


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


@pytest.mark.parametrize('backend', ['local', 'row_a2a'])
def test_vpp_mtp_microbatches_and_native_checkpoint(tmp_path_dist_ckpt, monkeypatch, backend):
    """MTP MoE and interior/final Engram consumers share correct VPP microbatch contexts."""
    Utils.initialize_model_parallel()
    try:
        torch.manual_seed(2026)
        model_parallel_cuda_manual_seed(2026)
        source = _models(1, backend=backend, mtp=True)
        source_path = tmp_path_dist_ckpt / f'engram_mtp_source_{backend}'
        _save(source, source_path)
        reference_losses, reference_mtp_loss, _ = _run_mtp_with_context_trace(source, monkeypatch)
        reference_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in source[0].named_parameters()
            if parameter.grad is not None and not parameter.grad.is_sparse
        }
        assert reference_mtp_loss is not None and torch.isfinite(reference_mtp_loss).all()
        del source
    finally:
        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2
    )
    try:
        models = _models(2, 2, backend=backend, recompute=True, mtp=True)
        _restore(models, source_path)
        target_path = tmp_path_dist_ckpt / f'engram_vpp_mtp_{backend}'
        for restored in (False, True):
            if restored:
                del models
                models = _models(2, 2, backend=backend, recompute=True, mtp=True)
                _restore(models, target_path)
            actual_losses, actual_mtp_loss, contexts = _run_mtp_with_context_trace(
                models, monkeypatch
            )
            embedding_key = 'embedding.word_embeddings.weight'
            embedding_gradient = torch.zeros_like(reference_gradients[embedding_key])
            for model in models:
                if hasattr(model, 'embedding'):
                    embedding_gradient.add_(model.embedding.word_embeddings.weight.grad)
            # The native training finalizer combines main-stack and MTP embedding
            # gradients across PP. These bare-module fixtures have no DDP main_grad.
            torch.distributed.all_reduce(embedding_gradient, group=models[0].pg_collection.pp)
            with _collective_assertions():
                torch.testing.assert_close(
                    embedding_gradient, reference_gradients[embedding_key], rtol=3e-4, atol=2e-6
                )
                last_rank = parallel_state.get_pipeline_model_parallel_rank() == 1
                assert [model.mtp_process for model in models] == [False, last_rank]
                assert [model.requires_token_context for model in models] == [last_rank] * 2
                if last_rank:
                    assert len(actual_losses) == 4
                    for actual, expected in zip(actual_losses, reference_losses, strict=True):
                        assert actual['sample'] == expected['sample']
                        torch.testing.assert_close(
                            actual['ce'], expected['ce'], rtol=1e-5, atol=1e-6
                        )
                    torch.testing.assert_close(
                        actual_mtp_loss, reference_mtp_loss, rtol=1e-5, atol=1e-6
                    )
                    assert set(contexts) == {1, 3}
                    for recorded in contexts.values():
                        samples = [int(tokens[0, 0]) - 1 for tokens in recorded]
                        assert Counter(samples) == Counter({sample: 2 for sample in range(4)})
                        for tokens, sample in zip(recorded, samples, strict=True):
                            torch.testing.assert_close(
                                tokens, torch.arange(1, 9).unsqueeze(0) + sample, rtol=0, atol=0
                            )
                    assert not any(hasattr(layer, 'engram') for layer in models[-1].mtp.modules())
                else:
                    assert actual_losses == [] and actual_mtp_loss is None and not contexts
                for model in models:
                    for name, parameter in model.named_parameters():
                        key = _global_name(model, name)
                        if key == embedding_key or key not in reference_gradients:
                            continue
                        torch.testing.assert_close(
                            parameter.grad, reference_gradients[key], rtol=3e-4, atol=2e-6, msg=key
                        )
            if not restored:
                _save(models, target_path)
    finally:
        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()
