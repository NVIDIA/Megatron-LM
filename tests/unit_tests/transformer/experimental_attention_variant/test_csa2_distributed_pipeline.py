# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real P2P/1F1B parity through HybridModel using native CSA2 projection adapters.

Launch with torch.distributed.run and at least four GPUs to cover every case.
Native cases also run with Gloo on CPU hosts; fused cases require CUDA.
The native adapters keep this focused on pipeline/autograd behavior; they do not
claim TE projection or MoE/EP integration coverage.
"""

import copy
import gc
import os
import weakref
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineDataIterator,
    PipelinePayload,
    PipelinePayloadPlan,
)
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_hybrid_adapter import (
    CSA2HybridAdapter,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_pipeline import (
    build_csa2_pipeline_plan,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import (
    _packed,
    _record_losses,
    _require_sparse_kernels,
    _RMSNorm,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_pipeline import (
    _FFN,
    _Attention,
    _config,
    _pattern,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_recompute import (
    _recompute_config,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_recompute import (
    cpu_checkpoint_rng as cpu_checkpoint_rng,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_recompute import (
    native_attention as native_attention,
)


def _model(
    config, groups, pattern, pre_process, post_process, device, vp_stage=None, attention=_Attention
):
    spec = ModuleSpec(
        HybridStack,
        params={"post_layer_norm": False},
        submodules=HybridStackSubmodules(
            forward_adapter=CSA2HybridAdapter,
            dsa_layer=ModuleSpec(
                TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=_RMSNorm,
                    self_attention=attention,
                    self_attn_bda=get_bias_dropout_add,
                ),
            ),
            moe_layer=ModuleSpec(
                TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=_RMSNorm, mlp=_FFN, mlp_bda=get_bias_dropout_add
                ),
            ),
        ),
    )
    return HybridModel(
        config,
        spec,
        vocab_size=32,
        max_sequence_length=32,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
        pre_process=pre_process,
        post_process=post_process,
        pg_collection=groups,
        vp_stage=vp_stage,
    ).to(device)


def _reference_parameter_name(name, offset):
    if name.startswith("decoder.layers."):
        parts = name.split(".")
        parts[2] = str(int(parts[2]) + offset)
        return ".".join(parts)
    return name


def _batches(count, layout, device):
    batches = []
    packing = [([1], [1], 0), ([1, 4, 0, 2], [2, 6, 0, 3], 1), ([3, 1], [5, 2], 2)]
    generator = torch.Generator().manual_seed(718)
    for i in range(count):
        params = None
        if layout == "thd":
            real, physical, tail = packing[i % len(packing)]
            params, _, valid = _packed(real, physical, tail=tail, device=device)
            shape = (1, valid.numel())
        else:
            shape = (2, 1 if i == 0 else 5 + i)
            valid = torch.ones(shape, dtype=torch.bool, device=device)
        tokens = torch.randint(0, 32, shape, generator=generator).to(device)
        labels = torch.randint(0, 32, shape, generator=generator).to(device)
        batches.append((tokens, labels, valid.reshape(shape), params))
    return batches


@contextmanager
def _pipeline_groups(pp_size):
    if torch.cuda.is_available():
        Utils.initialize_model_parallel(pipeline_model_parallel_size=pp_size)
        try:
            model_parallel_cuda_manual_seed(719)
            yield ProcessGroupCollection.use_mpu_process_groups(), torch.device(
                "cuda", torch.cuda.current_device()
            )
        finally:
            Utils.destroy_model_parallel()
        return
    if not dist.is_initialized():
        dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank, size = dist.get_rank(), dist.get_world_size()
    singletons = [dist.new_group([i], timeout=timedelta(seconds=60)) for i in range(size)]
    pipelines = [
        dist.new_group(list(range(i, i + pp_size)), timeout=timedelta(seconds=60))
        for i in range(0, size, pp_size)
    ]
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.dp_cp = singletons[rank]
    groups.pp = pipelines[rank // pp_size]
    groups.embd = groups.pos_embd = None
    dist.barrier()
    try:
        yield groups, torch.device("cpu")
    finally:
        dist.destroy_process_group(groups.pp)
        dist.destroy_process_group(groups.tp)


_CASES = [
    ((6,), "sbhd", 0.3, torch.float32, True, 4, False),  # Full reset / mHC only
    ((4,), "thd", 0.3, torch.float32, True, 4, False),  # Reuse2, empty r2 groups
    ((8,), "thd", 0.3, torch.float32, True, 4, False),  # Reindex
    ((10,), "thd", 0.3, torch.float32, True, 4, False),  # Reuse1
    ((7,), "thd", 0.0, torch.bfloat16, True, 4, False),
    ((7,), "thd", 0.3, torch.bfloat16, True, 4, False),
    ((7, 8, 10), "thd", 0.3, torch.float32, True, 5, False),  # FFN relay + multiple hops
    ((7, 8, 10), "thd", 0.3, torch.float32, False, 1, False),  # shorter than warmup
    ((7, 8, 10), "sbhd", 0.0, torch.float32, True, 3, False),
    ((7, 8, 10), "thd", 0.3, torch.float32, True, 4, True),
]
_CASES = [(*case, "none") for case in _CASES] + [
    ((7,), "thd", 0.3, torch.bfloat16, True, 4, False, "cudnn")
]
_CASES = [(*case, 1, None) for case in _CASES] + [
    # Unequal boundary fields, producer/Reindex/Reuse in separate virtual chunks.
    ((7, 8, 10), "thd", 0.3, torch.float32, True, 6, False, "none", 2, 4),
    ((7, 8, 10), "thd", 0.0, torch.bfloat16, True, 4, False, "none", 2, 2),
    ((7, 8, 10), "sbhd", 0.3, torch.float32, True, 2, False, "none", 2, 2),
    ((4, 6, 8), "thd", 0.3, torch.float32, True, 6, True, "none", 2, 4),
    # Eight chunks on PP=4; relay crosses rank 3 -> rank 0 as well as FFN-only chunks.
    ((3, 4, 5, 7, 8, 9, 10), "thd", 0.3, torch.float32, True, 8, False, "none", 2, 4),
    ((3, 4, 5, 7, 8, 9, 10), "sbhd", 0.0, torch.float32, False, 4, False, "none", 2, 4),
    ((7, 8, 10), "thd", 0.3, torch.bfloat16, True, 4, False, "cudnn", 2, 2),
]


@pytest.mark.parametrize(
    "cuts,layout,coefficient,dtype,mhc,count,forward_only,backend,vp_size,group_size", _CASES
)
@pytest.mark.parametrize(
    "batch_p2p,overlap,warmup_flush",
    [(False, False, False), (True, False, False), (False, True, False), (False, True, True)],
    ids=["blocking", "batched", "overlap", "warmup-flush"],
)
@pytest.mark.parametrize("planned", [False, True], ids=["dynamic-header", "prepared"])
def test_csa2_1f1b_matches_single_stage(
    monkeypatch,
    cuts,
    layout,
    coefficient,
    dtype,
    mhc,
    count,
    forward_only,
    backend,
    vp_size,
    group_size,
    batch_p2p,
    overlap,
    warmup_flush,
    planned,
    recompute_group_size=0,
    attention=_Attention,
    recompute_modules=None,
):
    if overlap and vp_size == 1:
        pytest.skip("P2P overlap requires VPP")
    pp_size = (len(cuts) + 1) // vp_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size % pp_size:
        pytest.skip(f"Requires torchrun with a world size divisible by PP={pp_size}")
    if backend == "cudnn":
        _require_sparse_kernels()
    if planned:

        def no_header(*args, **kwargs):
            pytest.fail("Prepared pipeline communication must never pack or parse device headers")

        monkeypatch.setattr(
            "megatron.core.pipeline_parallel.typed_p2p_communication._pack_header", no_header
        )
        monkeypatch.setattr(
            "megatron.core.pipeline_parallel.typed_p2p_communication._unpack_header", no_header
        )
    with _pipeline_groups(pp_size) as (groups, device):
        _record_losses(monkeypatch)
        torch.manual_seed(719)
        reference_groups = copy.copy(groups)
        reference_groups.pp = groups.tp
        reference_groups.embd = None
        config = replace(
            _config(dtype, coefficient, mhc),
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vp_size if vp_size > 1 else None,
            microbatch_group_size_per_vp_stage=group_size or pp_size,
            pipeline_dtype=dtype,
            batch_p2p_comm=batch_p2p,
            overlap_p2p_comm=overlap,
            overlap_p2p_comm_warmup_flush=warmup_flush,
            cross_entropy_loss_fusion=False,
            deallocate_pipeline_outputs=True,
        )
        if backend == "cudnn":
            config = replace(
                config,
                num_attention_heads=64,
                v_head_dim=512,
                dsa_indexer_n_heads=32,
                dsa_indexer_head_dim=128,
                dsa_kernel_backend="cudnn",
                dsa_indexer_use_sparse_loss=True,
                use_fused_mhc=True,
            )
        if recompute_group_size != 0:
            config = _recompute_config(config, recompute_group_size)
        if recompute_modules is not None:
            config = replace(
                config, multi_latent_attention=True, recompute_modules=recompute_modules
            )
        # Exercise the normal automatic binding path, rather than attaching a
        # codec/plan manually as the PP-1 local tests do.
        pattern = _pattern(cuts)
        reference = _model(
            replace(
                config,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                overlap_p2p_comm=False,
                overlap_p2p_comm_warmup_flush=False,
                recompute_granularity=None,
            ),
            reference_groups,
            "DE" * 6,
            True,
            True,
            device,
            attention=attention,
        )
        rank = groups.pp.rank()
        models = [
            _model(
                config,
                groups,
                pattern,
                rank == 0 and vp == 0,
                rank == pp_size - 1 and vp == vp_size - 1,
                device,
                vp_stage=vp if vp_size > 1 else None,
                attention=attention,
            )
            for vp in range(vp_size)
        ]
        plan = build_csa2_pipeline_plan(config, pattern, pp_size=pp_size)
        chunks = [plan[vp * pp_size + rank] for vp in range(vp_size)]
        reference_parameters = dict(reference.named_parameters())
        with torch.no_grad():
            for model, chunk in zip(models, chunks):
                assert model.pipeline_payload_factory is not None
                for name, parameter in model.named_parameters():
                    parameter.copy_(
                        reference_parameters[_reference_parameter_name(name, chunk.layer_offset)]
                    )
        optimizer = torch.optim.SGD([p for model in models for p in model.parameters()], lr=0.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        batches = _batches(count, layout, device)
        tolerance = (
            dict(atol=3e-6, rtol=5e-5) if dtype == torch.float32 else dict(atol=2e-3, rtol=5e-2)
        )
        for _ in range(2):  # Communication and activation state must not survive a step.
            expected = []
            reference_optimizer.zero_grad(set_to_none=True)
            DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor([1.0 / count], device=device))
            with torch.set_grad_enabled(not forward_only):
                for tokens, labels, valid, params in batches:
                    losses = reference(tokens, None, None, labels=labels, packed_seq_params=params)
                    loss = (losses.float() * valid).sum() / valid.sum()
                    expected.append(loss.detach().clone())
                    if not forward_only:
                        (loss / count).backward()

            actual, references = [], []

            def forward_step(iterator, stage):
                tokens, labels, valid, params = next(iterator)
                incoming = stage.decoder.input_tensor
                if isinstance(incoming, PipelinePayload):
                    references.append(weakref.ref(incoming))
                    assert all(t.is_leaf for t in incoming.tensors)
                # Intermediate chunks restore THD metadata from the wire only.
                output = stage(
                    tokens if stage.pre_process else None,
                    None,
                    None,
                    labels=labels if stage.post_process else None,
                    packed_seq_params=params if stage.pre_process else None,
                )
                assert stage.decoder.input_tensor is None
                if isinstance(output, PipelinePayload):
                    references.append(weakref.ref(output))

                def loss_func(losses):
                    loss = (losses.float() * valid).sum() / valid.sum()
                    actual.append(loss.detach().clone())
                    return loss, {"loss": loss.detach().clone()}

                return output, loss_func

            if planned:

                def prepare(iterator, stage, count, *, forward_only):
                    prepared, incoming, outgoing = [], [], []
                    for _ in range(count):
                        batch = next(iterator)
                        tokens, _, _, params = batch
                        recv, send = stage.pipeline_payload_spec(
                            tokens.shape[1], tokens.shape[0], params, requires_grad=not forward_only
                        )
                        prepared.append(batch)
                        incoming.append(recv)
                        outgoing.append(send)
                    return PipelineDataIterator(
                        prepared, PipelinePayloadPlan(tuple(incoming), tuple(outgoing))
                    )

                forward_step.prepare_pipeline_inputs = prepare

            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(not forward_only):
                schedule = (
                    forward_backward_pipelining_with_interleaving
                    if vp_size > 1
                    else forward_backward_pipelining_without_interleaving
                )
                schedule(
                    forward_step_func=forward_step,
                    data_iterator=[iter(batches) for _ in models] if vp_size > 1 else iter(batches),
                    model=models if vp_size > 1 else models[0],
                    num_microbatches=count,
                    seq_length=32,
                    micro_batch_size=1,
                    forward_only=forward_only,
                    p2p_communicator=P2PCommunicator(groups.pp, config),
                    pg_collection=groups,
                )
            gc.collect()
            assert all(ref() is None for ref in references)
            if models[-1].post_process:
                torch.testing.assert_close(torch.stack(actual), torch.stack(expected), **tolerance)
            if not forward_only:
                for model, chunk in zip(models, chunks):
                    for name, parameter in model.named_parameters():
                        ref = reference_parameters[
                            _reference_parameter_name(name, chunk.layer_offset)
                        ]
                        assert (parameter.grad is None) == (ref.grad is None), name
                        if ref.grad is not None:
                            torch.testing.assert_close(
                                parameter.grad, ref.grad, **tolerance, msg=name
                            )
                optimizer.step()
                reference_optimizer.step()
                for model, chunk in zip(models, chunks):
                    for name, parameter in model.named_parameters():
                        ref = reference_parameters[
                            _reference_parameter_name(name, chunk.layer_offset)
                        ]
                        torch.testing.assert_close(parameter, ref, **tolerance, msg=name)


@pytest.mark.parametrize(
    "case,recompute_group_size",
    [
        (_CASES[0], None),  # PP=2, SBHD
        (_CASES[5], None),  # PP=2, THD, BF16, auxiliary loss
        (_CASES[6], 2),  # PP=4, THD, FFN relay
        (_CASES[11], None),  # PP=2, VPP=2, THD
        (_CASES[12], 1),  # PP=2, VPP=2, BF16, auxiliary loss disabled
        (_CASES[14], None),  # Forward-only must not discard activations
        (_CASES[15], 3),  # PP=4, VPP=2, eight chunks
        (_CASES[17], None),  # Fused CSA2 + mHC, PP=2, VPP=2
    ],
)
@pytest.mark.parametrize(
    "batch_p2p,overlap,warmup_flush",
    [(False, False, False), (True, False, False), (False, True, False), (False, True, True)],
    ids=["blocking", "batched", "overlap", "warmup-flush"],
)
def test_csa2_recompute_1f1b_matches_single_stage(
    monkeypatch, cpu_checkpoint_rng, case, recompute_group_size, batch_p2p, overlap, warmup_flush
):
    """Compare two optimizer steps to PP=1/recompute-off, including all parameter gradients."""
    test_csa2_1f1b_matches_single_stage(
        monkeypatch,
        *case,
        batch_p2p,
        overlap,
        warmup_flush,
        True,
        recompute_group_size=recompute_group_size,
    )


@pytest.mark.parametrize("case", [_CASES[11], _CASES[12], _CASES[15]])
@pytest.mark.parametrize(
    "batch_p2p,overlap,warmup_flush",
    [(False, False, False), (True, False, False), (False, True, False), (False, True, True)],
    ids=["blocking", "batched", "overlap", "warmup-flush"],
)
def test_csa2_mla_recompute_1f1b_matches_single_stage(
    monkeypatch, cpu_checkpoint_rng, native_attention, case, batch_p2p, overlap, warmup_flush
):
    """Exercise DSv4 QKV/RoPE checkpointing together with mHC and state transport."""
    test_csa2_1f1b_matches_single_stage(
        monkeypatch,
        *case,
        batch_p2p,
        overlap,
        warmup_flush,
        True,
        recompute_group_size=None,
        attention=native_attention,
        recompute_modules=["mhc", "mla_up_proj"],
    )
