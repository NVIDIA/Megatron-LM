# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Batch-invariant all-to-all context parallelism must reproduce the CP=1 forward bitwise.

RL generation runs without context parallelism, so a zero-KL training forward under CP has
to equal the CP=1 forward exactly.
"""

import os

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    set_batch_invariant_mode,
    te_supports_batch_invariant_attention,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tuning import AutotunePolicy, install, set_deterministic_mode
from tests.unit_tests.test_utilities import Utils

try:
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_func  # noqa: F401

    HAVE_FA4 = True
except ImportError:
    HAVE_FA4 = False

try:
    import transformer_engine_torch as tex

    HAVE_TEX = True
except ImportError:
    HAVE_TEX = False

SEQ_LEN = 256
PACKED_LENGTHS = [96, 40, 64, 56]  # multiples of 2 * CP for every CP size tested
VOCAB_SIZE = 131072  # Nano's; at 128 the te_native output GEMM varies with the row count
# "small": head dim 128 for the attention layer, 32 heads for the Mamba layer.
# "nano": Nemotron-3-Nano's attention and Mamba layouts at TP1 (32 query heads, 2 K/V groups,
# head dim 128; 64 Mamba heads of dim 64 in 8 groups) and a squared-ReLU MoE layer.
SHAPES = {
    "small": dict(pattern="M*-", config=dict(hidden_size=1024, num_attention_heads=8)),
    "nano": dict(
        pattern="M*E",
        config=dict(
            hidden_size=2688,
            num_attention_heads=32,
            num_query_groups=2,
            kv_channels=128,
            mamba_num_heads=64,
            mamba_head_dim=64,
            mamba_num_groups=8,
            num_moe_experts=8,
            moe_router_topk=6,
            moe_ffn_hidden_size=1856,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_load_balancing_type="none",
            moe_router_dtype="fp32",
            gated_linear_unit=False,
            activation_func=squared_relu,
            add_bias_linear=False,
        ),
    ),
}


def _config(cp_size: int, shape: str, num_query_groups=None, tp_size=1) -> TransformerConfig:
    kwargs = dict(SHAPES[shape]["config"])
    if num_query_groups is not None:
        kwargs["num_query_groups"] = num_query_groups
    config = TransformerConfig(
        num_layers=3,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=tp_size > 1,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        flash_attention_version=4,
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash,
        context_parallel_size=cp_size,
        cp_comm_type="a2a",
        is_hybrid_model=True,
        **kwargs,
    )
    config.bf16 = True
    return config


def _build_model(cp_size: int, shape: str, num_query_groups=None, tp_size=1):
    config = _config(cp_size, shape, num_query_groups, tp_size)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
        hybrid_layer_pattern=SHAPES[shape]["pattern"],
    )
    return Float16Module(config, model.cuda()).eval()


def _inputs(qkv_format: str):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    batch_size = 2 if qkv_format == "sbhd" else 1
    input_ids = torch.randint(
        0, VOCAB_SIZE, (batch_size, SEQ_LEN), device="cuda", generator=generator
    )
    position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(batch_size, -1)
    packed_seq_params = None
    if qkv_format == "thd":
        cu_seqlens = torch.tensor([0] + PACKED_LENGTHS, dtype=torch.int32, device="cuda").cumsum(
            0, dtype=torch.int32
        )
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=max(PACKED_LENGTHS),
            max_seqlen_kv=max(PACKED_LENGTHS),
        )
    return input_ids, position_ids, packed_seq_params


def _cp_token_index(qkv_format, packed_seq_params, cp_size, cp_rank) -> torch.Tensor:
    """Positions of this CP rank's load-balanced shard within the full sequence."""
    if qkv_format == "thd":
        return tex.thd_get_partitioned_indices(
            packed_seq_params.cu_seqlens_q_padded, SEQ_LEN, cp_size, cp_rank
        ).long()
    chunk = SEQ_LEN // (2 * cp_size)
    first = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device="cuda")
    second_rank = 2 * cp_size - 1 - cp_rank
    second = torch.arange(second_rank * chunk, (second_rank + 1) * chunk, device="cuda")
    return torch.cat([first, second])


def _forward(model, input_ids, position_ids, packed_seq_params, grad=False):
    with set_batch_invariant_mode(True, backend="te_native"), torch.set_grad_enabled(grad):
        return model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        ).float()


def _is_tp_sharded(param) -> bool:
    return getattr(param, "tensor_model_parallel", False) or not getattr(param, "allreduce", True)


def _sync_weights(model):
    """Gives every rank holding the same TP shard the same weights.

    CPU initialization draws from each rank's CPU RNG, so ranks start with different weights.
    TP-sharded parameters are shared within their data-parallel group, keeping each TP rank's
    own shard; everything else is shared from rank 0, so replicated weights agree across TP.
    """
    from megatron.core import parallel_state

    dp_group = parallel_state.get_data_parallel_group()
    dp_src = torch.distributed.get_process_group_ranks(dp_group)[0]
    sharded = {id(p) for p in model.parameters() if _is_tp_sharded(p)}
    for value in list(model.parameters()) + list(model.buffers()):
        if not value.is_cuda:
            continue
        if id(value) in sharded:
            torch.distributed.broadcast(value.data, src=dp_src, group=dp_group)
        else:
            torch.distributed.broadcast(value.data, src=0)


def _run_reference_then_build_cp(run_reference, cp_size, shape, num_query_groups=None, tp_size=1):
    """Runs run_reference on a CP=1 model, then returns a CP=cp_size model with its weights.

    Both models use the same TP size, and a global rank has the same TP rank in both. The
    reference runs before the CP parallel state exists, so nothing it reads from the global
    process groups can see CP.
    """
    Utils.initialize_model_parallel(tp_size, 1)
    model_parallel_cuda_manual_seed(123)
    reference_model = _build_model(1, shape, num_query_groups, tp_size)
    _sync_weights(reference_model)
    reference = run_reference(reference_model)
    state_dict = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in reference_model.state_dict().items()
    }
    del reference_model
    Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(tp_size, 1, context_parallel_size=cp_size)
    model_parallel_cuda_manual_seed(123)
    model = _build_model(cp_size, shape, num_query_groups, tp_size)
    model.load_state_dict(state_dict)
    return reference, model


def _decoder_stages(model):
    """(name, module) for each decoder layer and the final norm, in forward order."""
    decoder = model.module.decoder
    stages = [(f"{i} ({type(layer).__name__})", layer) for i, layer in enumerate(decoder.layers)]
    if getattr(decoder, "final_norm", None) is not None:
        stages.append(("final_norm", decoder.final_norm))
    return stages


def _forward_with_layer_outputs(model, input_ids, position_ids, packed_seq_params):
    """Logits plus each decoder stage's output hidden states, [s, b, h] or [t, 1, h]."""
    layer_outputs = []

    def hook(module, args, output):
        output = output[0] if isinstance(output, tuple) else output
        layer_outputs.append(output.detach().clone())

    handles = [module.register_forward_hook(hook) for _, module in _decoder_stages(model)]
    try:
        logits = _forward(model, input_ids, position_ids, packed_seq_params)
    finally:
        for handle in handles:
            handle.remove()
    return logits, layer_outputs


def _gather_sp(x, tp_group):
    """Undoes sequence parallelism: concatenates the TP ranks' sequence chunks along dim 0."""
    if tp_group.size() == 1:
        return x
    chunks = [torch.empty_like(x) for _ in range(tp_group.size())]
    torch.distributed.all_gather(chunks, x.contiguous(), group=tp_group)
    return torch.cat(chunks)


def _gather_cp(shard, reference, seq_dim, qkv_format, packed_seq_params, cp_group):
    """Reassembles CP ranks' load-balanced shards into the full sequence along seq_dim."""
    cp_size = cp_group.size()
    shards = [torch.empty_like(shard) for _ in range(cp_size)]
    torch.distributed.all_gather(shards, shard.contiguous(), group=cp_group)
    gathered = torch.empty_like(reference)
    for rank, rank_shard in enumerate(shards):
        index = _cp_token_index(qkv_format, packed_seq_params, cp_size, rank)
        gathered.index_copy_(seq_dim, index, rank_shard)
    return gathered


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAVE_FA4, reason="FlashAttention-4 required")
@pytest.mark.skipif(not HAVE_TEX, reason="Transformer Engine required")
@pytest.mark.skipif(
    not te_supports_batch_invariant_attention(),
    reason="Batch-invariant attention requires TransformerEngine PR #3204 or >= 2.18.",
)
class TestBatchInvariantA2AContextParallel:

    def setup_method(self, method):
        # Inputs are built on "cuda" before the parallel state sets each rank's device.
        torch.cuda.set_device(Utils.local_rank % torch.cuda.device_count())
        # Pinned kernel configs, so CP=1 and CP>1 shapes cannot autotune to different kernels.
        set_deterministic_mode(True)
        install(AutotunePolicy(mode="pinned"))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ["NVTE_FUSED_ATTN"] = "0"
        os.environ["NVTE_FLASH_ATTN"] = "1"
        os.environ["NVTE_UNFUSED_ATTN"] = "0"

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        set_deterministic_mode(None)
        install(None)

    # Small model: 8 query groups split across CP ranks; 2 split at CP=2 and are gathered at
    # CP=4; 1 (Nano's one K/V head per TP rank at TP2) is always gathered.
    # Nano at TP1xCP4: 8 query heads per CP rank sharing a gathered K/V head; 16 Mamba heads.
    # TP2xCP2 (sequence parallel): small with 8 groups splits 4 K/V heads per TP rank across
    # CP; small with 2 groups and Nano have one K/V head per TP rank, gathered.
    @pytest.mark.parametrize(
        "shape,num_query_groups,cp_size,tp_size",
        [("small", g, cp, 1) for g in (8, 2, 1) for cp in (2, 4)]
        + [("nano", None, 4, 1)]
        + [("small", 8, 2, 2), ("small", 2, 2, 2), ("nano", None, 2, 2)],
    )
    @pytest.mark.parametrize("qkv_format", ["sbhd", "thd"])
    def test_forward_matches_cp1_bitwise(
        self, shape, num_query_groups, cp_size, tp_size, qkv_format
    ):
        if Utils.world_size < cp_size * tp_size:
            pytest.skip(f"requires {cp_size * tp_size} ranks, got {Utils.world_size}")
        input_ids, position_ids, packed_seq_params = _inputs(qkv_format)

        (reference, reference_layers), model = _run_reference_then_build_cp(
            lambda m: _forward_with_layer_outputs(m, input_ids, position_ids, packed_seq_params),
            cp_size,
            shape,
            num_query_groups,
            tp_size,
        )

        from megatron.core import parallel_state

        cp_group = parallel_state.get_context_parallel_group()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        index = _cp_token_index(qkv_format, packed_seq_params, cp_size, cp_group.rank())
        logits, layers = _forward_with_layer_outputs(
            model, input_ids[:, index], position_ids[:, index], packed_seq_params
        )

        def gather(shard, full, seq_dim):
            return _gather_cp(shard, full, seq_dim, qkv_format, packed_seq_params, cp_group)

        # Logits are split over the vocabulary across TP, the same way in both models.
        gathered = gather(logits, reference, 1)

        # Name the first diverging stage so a failure points at attention, Mamba, MoE, the
        # final norm, or the output layer.
        first_diverging_stage = "output_layer"
        stage_names = [name for name, _ in _decoder_stages(model)]
        for name, layer_out, reference_out in zip(stage_names, layers, reference_layers):
            reference_out = _gather_sp(reference_out, tp_group)
            gathered_out = gather(_gather_sp(layer_out, tp_group), reference_out, 0)
            if not torch.equal(gathered_out, reference_out):
                diff = (gathered_out.float() - reference_out.float()).abs().max().item()
                positions = (gathered_out != reference_out).any(dim=-1).sum().item()
                first_diverging_stage = (
                    f"{name}: {positions} token positions, max abs diff {diff:.3e}"
                )
                break
        if (
            first_diverging_stage == "output_layer"
            and tp_size == 1
            and not torch.equal(gathered, reference)
        ):
            # Same final hidden states in, so check the output GEMM's batch invariance alone:
            # this rank's rows computed within the full sequence and on their own.
            final_hidden = reference_layers[-1]
            with set_batch_invariant_mode(True, backend="te_native"), torch.no_grad():
                full_rows, _ = model.module.output_layer(final_hidden)
                own_rows, _ = model.module.output_layer(final_hidden.index_select(0, index))
            invariant = torch.equal(full_rows.index_select(0, index), own_rows)
            first_diverging_stage += (
                f" (its GEMM is {'' if invariant else 'not '}batch invariant between "
                f"{final_hidden.shape[0] * final_hidden.shape[1]} and "
                f"{own_rows.shape[0] * own_rows.shape[1]} rows)"
            )

        mismatched = (gathered != reference).any(dim=-1).sum().item()
        assert torch.equal(gathered, reference), (
            f"TP{tp_size}xCP{cp_size} a2a forward differs from TP{tp_size}xCP1 at {mismatched} "
            "token positions; "
            f"first diverging stage: {first_diverging_stage}"
        )

    @pytest.mark.parametrize("cp_size,tp_size", [(4, 1), (2, 2)])
    @pytest.mark.parametrize("qkv_format", ["sbhd", "thd"])
    def test_nano_grads_match_cp1(self, cp_size, tp_size, qkv_format):
        """Parameter gradients reduced over CP must match the CP=1 gradients at the same TP.

        Gradients of weights replicated across TP are also summed over TP in both models, since
        sequence parallelism leaves each TP rank a partial sum. The backward reduces in a
        different order under CP, so this is a closeness check, not a bitwise one.
        """
        if Utils.world_size < cp_size * tp_size:
            pytest.skip(f"requires {cp_size * tp_size} ranks, got {Utils.world_size}")
        input_ids, position_ids, packed_seq_params = _inputs(qkv_format)
        generator = torch.Generator(device="cuda").manual_seed(4321)
        loss_weights = torch.randn(*input_ids.shape, VOCAB_SIZE, device="cuda", generator=generator)
        # Logits are split over the vocabulary across TP.
        vocab_per_tp_rank = VOCAB_SIZE // tp_size
        tp_rank = Utils.rank % tp_size
        loss_weights = loss_weights[
            ..., tp_rank * vocab_per_tp_rank : (tp_rank + 1) * vocab_per_tp_rank
        ]

        def reduced_grads(model, groups):
            from megatron.core import parallel_state

            tp_group = parallel_state.get_tensor_model_parallel_group()
            grads = {}
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                grad = p.grad.float().clone()
                for group in groups + ([] if _is_tp_sharded(p) else [tp_group]):
                    torch.distributed.all_reduce(grad, group=group)
                grads[name] = grad
            return grads

        def run_reference(reference_model):
            reference_model.train()
            logits = _forward(
                reference_model, input_ids, position_ids, packed_seq_params, grad=True
            )
            (logits * loss_weights).sum().backward()
            return reduced_grads(reference_model, [])

        reference_grads, model = _run_reference_then_build_cp(
            run_reference, cp_size, "nano", tp_size=tp_size
        )
        model.train()

        from megatron.core import parallel_state

        cp_group = parallel_state.get_context_parallel_group()
        index = _cp_token_index(qkv_format, packed_seq_params, cp_size, cp_group.rank())
        logits = _forward(
            model, input_ids[:, index], position_ids[:, index], packed_seq_params, grad=True
        )
        (logits * loss_weights[:, index]).sum().backward()
        grads = reduced_grads(model, [cp_group])
        assert grads.keys() == reference_grads.keys()

        mismatched = []
        for name, reference_grad in reference_grads.items():
            error = (grads[name] - reference_grad).norm() / reference_grad.norm().clamp_min(1e-12)
            if error > 1e-2:
                mismatched.append(f"{name}: relative error {error.item():.3e}")
        assert (
            not mismatched
        ), f"TP{tp_size}xCP{cp_size} gradients differ from TP{tp_size}xCP1:\n" + "\n".join(
            mismatched
        )


class TestBatchInvariantContextParallelConfig:

    def test_rejects_non_a2a_cp(self):
        with pytest.raises(AssertionError, match="cp_comm_type='a2a'"):
            TransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                batch_invariant_mode=True,
                flash_attention_version=4,
                params_dtype=torch.bfloat16,
                attention_backend=AttnBackend.flash,
                context_parallel_size=2,
                cp_comm_type="p2p",
            )
