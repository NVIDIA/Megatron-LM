# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from copy import deepcopy
from inspect import signature

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.context_parallel_layout import prebuild_thd_cp_partition_routes
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_stage_input_cp_partition_mode,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
)
from megatron.core.utils import (
    flatten_batch_for_packed_sequences,
    get_batch_on_this_cp_rank,
    get_thd_batch_on_this_cp_rank,
    is_te_min_version,
)
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from tests.unit_tests.dist_checkpointing import init_basic_mock_args
from tests.unit_tests.test_utilities import Utils

try:
    import fla  # noqa: F401

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


_PARALLEL_EP_SIZE = 4
_SEQUENCE_LENGTH = 4096
_MICRO_BATCH_SIZE = 1
_VOCAB_SIZE = 8192
_SEED = 1234
_DIAGNOSTIC_REPEATS = 5
_NUM_LAYERS = 3
_LINEAR_ATTENTION_PATTERN = [1, 0, 0]
_SIDE_LAYOUT_GUARD_LAYER_INDEX = 2
_SIDE_LAYOUT_GUARD_PARTITION_MODE = "zigzag"
_LM_LOSS_ATOL = 0.003
_MTP_LOSS_ATOL = 0.003
_GRAD_NORM_ATOL = 0.1
_GRAD_NORM_RTOL = 0.01
_REFERENCE_LINEAR_CP_MODE = "headwise"
_CANDIDATE_LINEAR_CP_MODE = "chunkwise"
_CP_LAYOUT_WARNING_SUBSTRINGS = (
    "missing precomputed context-parallel layout routes",
    "received no PackedSeqParams while running under context parallelism",
)
_PARALLEL_CASES = (
    pytest.param(4, 1, _PARALLEL_EP_SIZE, False, id="cp4_tp1_ep4"),
    pytest.param(2, 2, _PARALLEL_EP_SIZE, True, id="cp2_tp2_ep4_sp"),
)
_FULL_RECOMPUTE_CASES = (
    pytest.param(False, id="no_recompute"),
    pytest.param(True, id="full_recompute"),
)


def _destroy_model_parallel_without_barrier():
    if not Utils.inited:
        return
    torch.cuda.synchronize()
    parallel_state.destroy_model_parallel()
    Utils.inited = False
    torch.cuda.memory.empty_cache()


def _collect_parameter_state(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}


def _copy_state_to_model(source_state, model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            source = source_state[name]
            if source.shape == param.shape:
                param.copy_(source.to(device=param.device, dtype=param.dtype))
                continue

            raise AssertionError(
                f"Cannot copy parameter {name}: source shape {tuple(source.shape)}, "
                f"target shape {tuple(param.shape)}"
            )


def _install_layer_rotary_layout_guard(
    model, layer_index, expected_rotary_pos_emb, expected_partition_mode
):
    """Assert a decoder layer receives RoPE in the expected CP layout."""
    layer = model.decoder.layers[layer_index]
    original_forward = layer.forward

    def guarded_forward(*args, **kwargs):
        packed_seq_params = kwargs.get("packed_seq_params")
        actual_partition_mode = getattr(packed_seq_params, "cp_partition_mode", None)
        assert actual_partition_mode == expected_partition_mode, (
            f"Layer {layer_index} expected packed_seq_params.cp_partition_mode="
            f"{expected_partition_mode!r}, got {actual_partition_mode!r}."
        )
        rotary_pos_emb = kwargs.get("rotary_pos_emb")
        assert rotary_pos_emb is not None, f"Layer {layer_index} did not receive RoPE."
        torch.testing.assert_close(
            rotary_pos_emb,
            expected_rotary_pos_emb,
            atol=0.0,
            rtol=0.0,
            msg=(
                f"Layer {layer_index} received RoPE that does not match "
                f"{expected_partition_mode!r} CP layout."
            ),
        )
        return original_forward(*args, **kwargs)

    layer.forward = guarded_forward


def _assert_no_cp_layout_warnings(caught_warnings):
    for caught_warning in caught_warnings:
        message = str(caught_warning.message)
        assert not any(
            warning_substring in message
            for warning_substring in _CP_LAYOUT_WARNING_SUBSTRINGS
        ), message


def _get_model_input_partition_mode(model):
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return "zigzag"
    return decoder.cp_stage_entry_partition_mode or "zigzag"


def _get_stage_entry_partition_mode(config, vp_stage=None, pp_rank=0):
    return get_experimental_attention_variant_stage_input_cp_partition_mode(
        config=config, vp_stage=vp_stage, pp_rank=pp_rank
    )


def _make_config(
    context_parallel_size,
    tensor_model_parallel_size,
    expert_model_parallel_size,
    sequence_parallel,
    qkv_format,
    linear_cp_mode,
    full_recompute,
):
    packed_kwargs = {}
    if qkv_format == "thd":
        packed_kwargs = {
            "sequence_packing_scheduler": "dp_balanced",
            "pad_packed_seq_alignment": "max",
            "max_seqlen_per_dp_cp_rank": _SEQUENCE_LENGTH,
        }

    recompute_kwargs = (
        {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        }
        if full_recompute
        else {
            "recompute_granularity": None,
            "recompute_method": None,
            "recompute_num_layers": None,
        }
    )

    return TransformerConfig(
        hidden_size=512,
        ffn_hidden_size=1024,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=_NUM_LAYERS,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        kv_channels=64,
        num_query_groups=2,
        qk_layernorm=True,
        attention_output_gate=True,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=_LINEAR_ATTENTION_PATTERN,
        linear_cp_mode=linear_cp_mode,
        transformer_impl="transformer_engine",
        tensor_model_parallel_size=tensor_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=1,
        context_parallel_size=context_parallel_size,
        cp_partition_mode="auto",
        sequence_parallel=sequence_parallel,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        num_moe_experts=32,
        moe_layer_freq=1,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_shared_expert_gate=True,
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=4,
        moe_grouped_gemm=True,
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_flex_dispatcher_num_sms=32,
        moe_permute_fusion=True,
        moe_router_fusion=True,
        moe_router_dtype="fp32",
        mtp_num_layers=1,
        mtp_loss_scaling_factor=1.0,
        mtp_use_repeated_layer=False,
        gdn_pre_gated_delta_rule_fusion=False,
        **packed_kwargs,
        **recompute_kwargs,
    )


def _initialize_gpt_model(
    config, pre_process=True, post_process=True, vp_stage=None, pg_collection=None
):
    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=config, vp_stage=vp_stage, pp_rank=0
    )
    mtp_block_spec = None
    if config.mtp_num_layers:
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config,
            spec=transformer_layer_spec,
            use_transformer_engine=True,
            vp_stage=vp_stage,
            pp_rank=0,
        )

    model_kwargs = {
        "config": config,
        "transformer_layer_spec": transformer_layer_spec,
        "mtp_block_spec": mtp_block_spec,
        "vocab_size": _VOCAB_SIZE,
        "max_sequence_length": _SEQUENCE_LENGTH,
        "pre_process": pre_process,
        "post_process": post_process,
        "position_embedding_type": "rope",
        "rotary_percent": 0.25,
        "rotary_base": 10000000,
        "pg_collection": pg_collection,
        "vp_stage": vp_stage,
    }
    if "cp_stage_entry_partition_mode" in signature(GPTModel).parameters:
        model_kwargs["cp_stage_entry_partition_mode"] = _get_stage_entry_partition_mode(
            config=config, vp_stage=vp_stage, pp_rank=0
        )

    return GPTModel(**model_kwargs)


def _build_gpt_model(config, device):
    model = _initialize_gpt_model(config)
    model.to(device=device)
    return model


def _set_mock_args(args, config, context_parallel_size):
    init_basic_mock_args(args, config.tensor_model_parallel_size, 1, bf16=True)
    args.context_parallel_size = context_parallel_size
    args.cp_comm_type = "a2a" if context_parallel_size == 1 else "p2p"
    args.expert_model_parallel_size = config.expert_model_parallel_size
    args.expert_tensor_parallel_size = 1
    args.num_experts = config.num_moe_experts
    args.moe_ffn_hidden_size = config.moe_ffn_hidden_size
    args.moe_shared_expert_intermediate_size = config.moe_shared_expert_intermediate_size
    args.moe_shared_expert_gate = config.moe_shared_expert_gate
    args.moe_router_load_balancing_type = config.moe_router_load_balancing_type
    args.moe_router_topk = config.moe_router_topk
    args.moe_grouped_gemm = config.moe_grouped_gemm
    args.moe_aux_loss_coeff = config.moe_aux_loss_coeff
    args.moe_token_dispatcher_type = config.moe_token_dispatcher_type
    args.moe_flex_dispatcher_backend = config.moe_flex_dispatcher_backend
    args.moe_flex_dispatcher_num_sms = config.moe_flex_dispatcher_num_sms
    args.moe_permute_fusion = config.moe_permute_fusion
    args.moe_router_fusion = config.moe_router_fusion
    args.moe_router_dtype = config.moe_router_dtype
    args.mtp_num_layers = config.mtp_num_layers
    args.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
    args.mtp_use_repeated_layer = config.mtp_use_repeated_layer
    args.recompute_granularity = config.recompute_granularity
    args.recompute_method = config.recompute_method
    args.recompute_num_layers = config.recompute_num_layers
    args.linear_cp_mode = config.linear_cp_mode
    args.sequence_parallel = config.sequence_parallel
    args.seq_length = _SEQUENCE_LENGTH
    args.max_position_embeddings = _SEQUENCE_LENGTH
    args.padded_vocab_size = _VOCAB_SIZE
    args.untie_embeddings_and_output_weights = True


def _make_sbhd_batch(device):
    tokens = torch.randint(
        low=0,
        high=_VOCAB_SIZE,
        size=(_MICRO_BATCH_SIZE, _SEQUENCE_LENGTH),
        device=device,
        dtype=torch.long,
    )
    valid_length = 3512
    prompt_length = 67
    tokens[:, valid_length:] = 0
    labels = (tokens + 1) % _VOCAB_SIZE
    loss_mask = torch.zeros_like(tokens, dtype=torch.float32)
    loss_mask[:, prompt_length:valid_length] = 1.0
    position_ids = torch.arange(_SEQUENCE_LENGTH, device=device, dtype=torch.long).unsqueeze(0)
    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": None,
        "position_ids": position_ids,
    }


def _make_thd_batch(device):
    padded_seq_lengths = [1024, 768, 1280, 1024]
    seq_lengths = [901, 629, 1103, 877]
    prompt_lengths = [0, 63, 257, 15]
    assert sum(padded_seq_lengths) == _SEQUENCE_LENGTH

    tokens = torch.zeros((_MICRO_BATCH_SIZE, _SEQUENCE_LENGTH), device=device, dtype=torch.long)
    labels = torch.zeros_like(tokens)
    loss_mask = torch.zeros_like(tokens, dtype=torch.float32)
    padding_mask = torch.ones_like(tokens, dtype=torch.bool)
    position_ids = torch.empty_like(tokens)

    padded_offset = 0
    cu_seqlens = [0]
    cu_seqlens_padded = [0]
    for seq_length, padded_seq_length, prompt_length in zip(
        seq_lengths, padded_seq_lengths, prompt_lengths
    ):
        valid_end = padded_offset + seq_length
        padded_end = padded_offset + padded_seq_length
        seq_tokens = torch.randint(
            low=0,
            high=_VOCAB_SIZE,
            size=(_MICRO_BATCH_SIZE, seq_length),
            device=device,
            dtype=torch.long,
        )
        tokens[:, padded_offset:valid_end] = seq_tokens
        labels[:, padded_offset:valid_end] = (seq_tokens + 1) % _VOCAB_SIZE
        loss_mask[:, padded_offset + prompt_length : valid_end] = 1.0
        padding_mask[:, padded_offset:valid_end] = False
        position_ids[:, padded_offset:valid_end] = torch.arange(
            seq_length, device=device, dtype=torch.long
        )
        position_ids[:, valid_end:padded_end] = 0
        cu_seqlens.append(cu_seqlens[-1] + seq_length)
        cu_seqlens_padded.append(padded_end)
        padded_offset = padded_end

    cu_seqlens = torch.tensor([cu_seqlens], device=device, dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([cu_seqlens_padded], device=device, dtype=torch.int32)
    max_seqlen = torch.tensor([max(padded_seq_lengths)], device=device, dtype=torch.int32)
    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": None,
        "padding_mask": padding_mask,
        "position_ids": position_ids,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_padded": cu_seqlens_padded,
        "max_seqlen": max_seqlen,
    }


def _prepare_batch_for_model(batch, qkv_format, cp_group, cp_partition_mode):
    batch = deepcopy(batch)
    if qkv_format == "sbhd":
        batch = get_batch_on_this_cp_rank(
            batch,
            is_hybrid_cp=False,
            cp_group=cp_group,
            cp_partition_mode=cp_partition_mode,
        )
        packed_seq_params = PackedSeqParams(
            qkv_format="sbhd",
            cp_group=cp_group,
            cp_partition_mode=cp_partition_mode,
        )
        return batch, packed_seq_params

    batch = flatten_batch_for_packed_sequences(batch)
    batch, packed_seq_params = get_thd_batch_on_this_cp_rank(
        batch,
        batch["cu_seqlens"][0],
        batch["cu_seqlens_padded"][0],
        batch["max_seqlen"],
        cp_partition_mode=cp_partition_mode,
    )
    prebuild_thd_cp_partition_routes(packed_seq_params, cp_group)
    return batch, packed_seq_params


def _global_grad_norm(model):
    grads = [
        param.grad.detach()
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    return torch.tensor(
        get_grad_norm_fp32(grads, grad_stats_parallel_group=torch.distributed.group.WORLD),
        device=torch.cuda.current_device(),
        dtype=torch.float32,
    )


def _get_mtp_losses_from_tracker():
    MTPLossLoggingHelper.reduce_loss_in_tracker()
    tracker = MTPLossLoggingHelper.tracker
    assert "values" in tracker, "MTP loss tracker did not record any loss values."
    return tracker["values"].detach().float().clone()


def _loss_and_grad_stats(model, batch, packed_seq_params):
    model.zero_grad(set_to_none=True)
    MTPLossLoggingHelper.clean_loss_in_tracker()
    MTPLossAutoScaler.set_loss_scale(torch.ones((), device=torch.cuda.current_device()))

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        loss = model(
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
            packed_seq_params=packed_seq_params,
            padding_mask=batch.get("padding_mask"),
        )
        mtp_losses = _get_mtp_losses_from_tracker()
        numerator = (loss.float() * batch["loss_mask"]).sum()
        denominator = batch["loss_mask"].sum()
        (numerator / denominator.clamp(min=1)).backward()

    _assert_no_cp_layout_warnings(caught_warnings)
    grad_norm = _global_grad_norm(model)
    return numerator, denominator, mtp_losses, grad_norm


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.skipif(not is_te_min_version("1.11.0"), reason="MoE grouped GEMM requires TE >= 1.11.")
@pytest.mark.parametrize("repeat_index", range(_DIAGNOSTIC_REPEATS))
@pytest.mark.parametrize("full_recompute", _FULL_RECOMPUTE_CASES)
@pytest.mark.parametrize("qkv_format", ["sbhd", "thd"])
@pytest.mark.parametrize(
    (
        "context_parallel_size,tensor_model_parallel_size,expert_model_parallel_size,"
        "sequence_parallel"
    ),
    _PARALLEL_CASES,
)
def test_qwen35_proxy_gdn_moe_chunkwise_loss_and_grad_matches_headwise(
    qkv_format,
    context_parallel_size,
    tensor_model_parallel_size,
    expert_model_parallel_size,
    sequence_parallel,
    full_recompute,
    repeat_index,
):
    min_world_size = max(
        context_parallel_size * tensor_model_parallel_size, expert_model_parallel_size
    )
    if not torch.cuda.is_available() or Utils.world_size < min_world_size:
        pytest.skip(f"GDN/MoE CP loss parity needs at least {min_world_size} CUDA ranks.")

    mock_args = parse_args(ignore_unknown_args=True)
    set_args(mock_args)

    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_tensor_parallel_size=1,
            context_parallel_size=context_parallel_size,
        )
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        seed = _SEED + repeat_index
        torch.manual_seed(seed)
        batch = _make_thd_batch(device) if qkv_format == "thd" else _make_sbhd_batch(device)
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        reference_config = _make_config(
            context_parallel_size=context_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            sequence_parallel=sequence_parallel,
            qkv_format=qkv_format,
            linear_cp_mode=_REFERENCE_LINEAR_CP_MODE,
            full_recompute=full_recompute,
        )
        _set_mock_args(mock_args, reference_config, context_parallel_size=context_parallel_size)

        reference_model = _build_gpt_model(reference_config, device)
        reference_model.train()

        cp_group = parallel_state.get_context_parallel_group()
        reference_input_partition_mode = _get_model_input_partition_mode(reference_model)
        reference_batch, reference_packed_seq_params = _prepare_batch_for_model(
            batch,
            qkv_format=qkv_format,
            cp_group=cp_group,
            cp_partition_mode=reference_input_partition_mode,
        )
        reference_num, reference_den, reference_mtp_losses, reference_grad_norm = (
            _loss_and_grad_stats(reference_model, reference_batch, reference_packed_seq_params)
        )
        reference_stats = torch.stack([reference_num.detach(), reference_den.detach()])
        torch.distributed.all_reduce(reference_stats, group=cp_group)
        reference_avg = reference_stats[0] / reference_stats[1].clamp(min=1)
        source_state = _collect_parameter_state(reference_model)

        del reference_model
        torch.cuda.empty_cache()

        model_parallel_cuda_manual_seed(seed)
        candidate_config = _make_config(
            context_parallel_size=context_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            sequence_parallel=sequence_parallel,
            qkv_format=qkv_format,
            linear_cp_mode=_CANDIDATE_LINEAR_CP_MODE,
            full_recompute=full_recompute,
        )
        _set_mock_args(mock_args, candidate_config, context_parallel_size=context_parallel_size)
        candidate_model = _build_gpt_model(candidate_config, device)
        candidate_model.train()
        _copy_state_to_model(source_state, candidate_model)

        candidate_input_partition_mode = _get_model_input_partition_mode(candidate_model)
        candidate_batch, candidate_packed_seq_params = _prepare_batch_for_model(
            batch,
            qkv_format=qkv_format,
            cp_group=cp_group,
            cp_partition_mode=candidate_input_partition_mode,
        )
        if qkv_format == "sbhd":
            expected_rotary_pos_emb = candidate_model.rotary_pos_emb(
                _SEQUENCE_LENGTH,
                packed_seq=False,
                cp_group=cp_group,
                cp_partition_mode=_SIDE_LAYOUT_GUARD_PARTITION_MODE,
            )
            _install_layer_rotary_layout_guard(
                candidate_model,
                layer_index=_SIDE_LAYOUT_GUARD_LAYER_INDEX,
                expected_rotary_pos_emb=expected_rotary_pos_emb,
                expected_partition_mode=_SIDE_LAYOUT_GUARD_PARTITION_MODE,
            )
        candidate_num, candidate_den, candidate_mtp_losses, candidate_grad_norm = (
            _loss_and_grad_stats(candidate_model, candidate_batch, candidate_packed_seq_params)
        )
        stats = torch.stack([candidate_num.detach(), candidate_den.detach()])
        torch.distributed.all_reduce(stats, group=cp_group)
        candidate_avg = stats[0] / stats[1].clamp(min=1)
        lm_loss_diff = (candidate_avg.float() - reference_avg.float()).abs()
        mtp_loss_diff = (candidate_mtp_losses - reference_mtp_losses).abs().max()
        grad_norm_diff = (candidate_grad_norm - reference_grad_norm).abs()

        if torch.distributed.get_rank() == 0:
            print(
                "GDN MoE CP loss/grad parity: "
                f"case=cp{context_parallel_size}_tp{tensor_model_parallel_size}_"
                f"ep{expert_model_parallel_size}"
                f"{'_sp' if sequence_parallel else ''} "
                f"layers={_NUM_LAYERS} linear_attention_pattern={_LINEAR_ATTENTION_PATTERN} "
                f"format={qkv_format} full_recompute={full_recompute} "
                f"repeat={repeat_index} seed={seed} "
                f"lm_reference={reference_avg.float().item():.8f} "
                f"lm_candidate={candidate_avg.float().item():.8f} "
                f"lm_diff={lm_loss_diff.item():.8f} "
                f"mtp_reference={reference_mtp_losses[0].item():.8f} "
                f"mtp_candidate={candidate_mtp_losses[0].item():.8f} "
                f"mtp_diff={mtp_loss_diff.item():.8f} "
                f"grad_norm_reference={reference_grad_norm.item():.8f} "
                f"grad_norm_candidate={candidate_grad_norm.item():.8f} "
                f"grad_norm_diff={grad_norm_diff.item():.8f}",
                flush=True,
            )

        torch.testing.assert_close(
            candidate_avg.float(),
            reference_avg.float(),
            atol=_LM_LOSS_ATOL,
            rtol=0.0,
        )
        torch.testing.assert_close(
            candidate_mtp_losses,
            reference_mtp_losses,
            atol=_MTP_LOSS_ATOL,
            rtol=0.0,
        )
        torch.testing.assert_close(
            candidate_grad_norm,
            reference_grad_norm,
            atol=_GRAD_NORM_ATOL,
            rtol=_GRAD_NORM_RTOL,
        )
    finally:
        _destroy_model_parallel_without_barrier()
