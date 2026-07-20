# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.gpus(2),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
    pytest.mark.timeout(seconds=3600),
]


def _make_train_config(ps):
    from types import SimpleNamespace

    return SimpleNamespace(
        tp=ps.tp_size,
        ep=ps.ep_size,
        etp=ps.etp_size,
        pp=ps.pp_size,
        cp=ps.cp_size,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
        deterministic=True,
    )


def _make_glm5_model(cfg, ps, **kwargs):
    from megatron.lite.model.glm5.lite.model import Glm5Model

    return Glm5Model(cfg, _make_train_config(ps), ps, **kwargs)


def _init_dist_or_skip():
    import os

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GLM5 CP smoke.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so CP ranks are available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() < 2:
        pytest.skip("GLM5 CP smoke requires at least 2 ranks.")
    return torch.device("cuda", local_rank)


def _tiny_config_kwargs():
    return dict(
        num_hidden_layers=2,
        hidden_size=128,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        vocab_size=32,
        max_position_embeddings=512,
        initializer_range=0.002,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=512,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=3,
    )


def _tiny_hf_parity_config_kwargs():
    kwargs = _tiny_config_kwargs()
    kwargs.update(index_topk=512, max_position_embeddings=512)
    return kwargs


def _fused_dsa_seq_len(world: int) -> int:
    seq = 512
    if seq % (2 * world) != 0:
        pytest.skip(f"GLM5 fused DSA CP smoke requires seq={seq} divisible by 2*world={2 * world}.")
    return seq


def _to_hf_deepseek_v3_config(cfg):
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

    return DeepseekV3Config(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        moe_intermediate_size=cfg.moe_intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        vocab_size=cfg.vocab_size,
        n_shared_experts=cfg.n_shared_experts,
        n_routed_experts=cfg.n_routed_experts,
        routed_scaling_factor=cfg.routed_scaling_factor,
        kv_lora_rank=cfg.kv_lora_rank,
        q_lora_rank=cfg.q_lora_rank,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        n_group=cfg.n_group,
        topk_group=cfg.topk_group,
        num_experts_per_tok=cfg.num_experts_per_tok,
        first_k_dense_replace=cfg.first_k_dense_replace,
        norm_topk_prob=cfg.norm_topk_prob,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        tie_word_embeddings=False,
        rope_theta=cfg.rope_theta,
        rope_scaling=None,
        rope_interleave=cfg.rope_interleave,
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
    )


def _capture_tensor_output(outputs):
    return outputs[0].detach() if isinstance(outputs, tuple) else outputs.detach()


def _distributed_diff_stats(actual, expected) -> tuple[float, float]:
    import torch
    import torch.distributed as dist

    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max()
    scale = torch.maximum(actual.float().abs().max(), expected.float().abs().max()).clamp_min(1e-6)
    stats = torch.stack([max_abs, scale])
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    return float(stats[0].item()), float((stats[0] / stats[1]).item())


def _hf_state_dict_for_glm5_loader(model, native, cfg):
    """Build the synthetic HF fixture, including GLM5-only DSA indexer weights."""
    from megatron.lite.model.glm5.lite.checkpoint import Glm5WeightSpec

    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    spec = Glm5WeightSpec(cfg)
    indexer_names = [name for name in native.state_dict() if ".indexer." in name]
    assert indexer_names
    for native_name in indexer_names:
        mappings = spec.native_to_hf(native_name, native.state_dict()[native_name])
        assert len(mappings) == 1
        hf_name, tensor = mappings[0]
        state[hf_name] = tensor.detach().cpu().contiguous().clone()
    return state


def _make_dsa(*, cp_size: int = 1, cp_rank: int = 0, cp_group=None):
    from megatron.lite.primitive.modules.attention import DynamicSparseAttention

    return DynamicSparseAttention(
        hidden_size=128,
        num_attention_heads=64,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        rms_norm_eps=1e-5,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_group=cp_group,
    )


def test_glm5_dsa_cp2_matches_full_sequence_reference_forward_and_grad():
    import torch
    import torch.distributed as dist

    from megatron.lite.primitive.modules.attention import build_rope_cache
    from megatron.lite.primitive.parallel.cp import zigzag_position_ids_for_cp, zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import ParallelState

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(2026)
    cp_attn = _make_dsa(cp_size=world, cp_rank=rank, cp_group=ps.cp_group).to(
        device=device, dtype=torch.bfloat16
    )
    torch.manual_seed(2026)
    ref_attn = _make_dsa().to(device=device, dtype=torch.bfloat16)

    batch, seq = 1, _fused_dsa_seq_len(world)
    torch.manual_seed(99)
    full_x = torch.randn(batch, seq, 128, device=device, dtype=torch.bfloat16)
    local_x = zigzag_slice_for_cp(full_x, rank, world, seq_dim=1).detach().requires_grad_(True)
    ref_x = full_x.detach().clone().requires_grad_(True)

    cos, sin = build_rope_cache(
        dim=64, max_position_embeddings=seq, rope_theta=1_000_000.0, device=device
    )
    local_pos = zigzag_position_ids_for_cp(seq, rank, world, device).expand(batch, -1)
    full_pos = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)

    cp_out = cp_attn(local_x, cos=cos, sin=sin, position_ids=local_pos)
    ref_out = ref_attn(ref_x, cos=cos, sin=sin, position_ids=full_pos)
    expected = zigzag_slice_for_cp(ref_out, rank, world, seq_dim=1)
    torch.testing.assert_close(cp_out, expected, atol=3e-2, rtol=3e-2)

    cp_out.float().sum().backward()
    ref_out.float().sum().backward()
    expected_grad = zigzag_slice_for_cp(ref_x.grad, rank, world, seq_dim=1)
    assert local_x.grad is not None
    torch.testing.assert_close(local_x.grad, expected_grad, atol=8e-2, rtol=8e-2)


def test_glm5_tiny_model_cp2_matches_full_sequence_reference_forward():
    import torch
    import torch.distributed as dist

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import ParallelState

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = Glm5Config(**_tiny_config_kwargs())
    cfg.mlp_layer_types = ["dense", "dense"]
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(777)
    cp_model = _make_glm5_model(cfg, ps=ps).to(device=device, dtype=torch.bfloat16)
    torch.manual_seed(777)
    ref_model = _make_glm5_model(cfg, ps=ParallelState()).to(
        device=device, dtype=torch.bfloat16
    )
    cp_model.eval()
    ref_model.eval()

    batch, seq = 1, _fused_dsa_seq_len(world)
    torch.manual_seed(100)
    full_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    local_ids = zigzag_slice_for_cp(full_ids, rank, world, seq_dim=1).contiguous()

    with torch.no_grad():
        cp_hidden = cp_model(input_ids=local_ids)["hidden_states"]
        ref_hidden = ref_model(input_ids=full_ids)["hidden_states"]
    expected = zigzag_slice_for_cp(ref_hidden, rank, world, seq_dim=0)

    torch.testing.assert_close(cp_hidden, expected, atol=1e-1, rtol=1e-1)


def test_glm5_tiny_model_cp2_forward_backward_smoke():
    import torch
    import torch.distributed as dist

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import ParallelState

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = Glm5Config(**_tiny_config_kwargs())
    cfg.mlp_layer_types = ["dense", "dense"]
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(1234)
    model = _make_glm5_model(cfg, ps=ps).to(device=device, dtype=torch.bfloat16)

    batch, seq = 1, _fused_dsa_seq_len(world)
    torch.manual_seed(55)
    full_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    input_ids = zigzag_slice_for_cp(full_ids, rank, world, seq_dim=1).contiguous()

    output = model(input_ids=input_ids)
    assert output["hidden_states"].shape == (seq // world, batch, cfg.hidden_size)
    assert torch.isfinite(output["hidden_states"].float()).all()
    loss = output["hidden_states"].float().square().mean()
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    grad_norm = torch.zeros((), device=device)
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = grad_norm + param.grad.detach().float().norm()
    assert torch.isfinite(grad_norm)


def test_glm5_packed_thd_variable_sequence_cp2_forward_backward_smoke():
    import torch
    import torch.distributed as dist

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.primitive.parallel.state import ParallelState
    from megatron.lite.primitive.parallel.thd import pack_nested_thd, unpack_packed_thd_to_nested

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg_kwargs = _tiny_config_kwargs()
    cfg_kwargs.update(max_position_embeddings=64, num_nextn_predict_layers=1)
    cfg = Glm5Config(**cfg_kwargs)
    cfg.mlp_layer_types = ["dense", "dense"]
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(20260614)
    model = _make_glm5_model(cfg, ps=ps, mtp_enable=True, mtp_enable_train=True).to(
        device=device, dtype=torch.bfloat16
    )
    model.train()

    lengths = [16, 20, 24]
    ids = torch.nested.as_nested_tensor(
        [
            torch.randint(0, cfg.vocab_size, (length,), device=device, dtype=torch.long)
            for length in lengths
        ],
        layout=torch.jagged,
    )
    labels = torch.nested.as_nested_tensor(
        [
            torch.randint(0, cfg.vocab_size, (length,), device=device, dtype=torch.long)
            for length in lengths
        ],
        layout=torch.jagged,
    )
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(length, device=device, dtype=torch.float32) for length in lengths],
        layout=torch.jagged,
    )
    packed = pack_nested_thd(
        ids,
        cp_size=world,
        cp_rank=rank,
        cp_group=ps.cp_group,
        labels=labels,
        loss_mask=loss_mask,
    )

    out = model(
        input_ids=packed.input_ids,
        labels=packed.labels,
        loss_mask=packed.loss_mask,
        position_ids=packed.position_ids,
        packed_seq_params=packed.packed_seq_params,
    )
    assert torch.isfinite(out["loss"])
    assert "mtp_loss" in out
    out["loss"].backward()

    grad_norm = torch.zeros((), device=device)
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = grad_norm + param.grad.detach().float().norm()
    assert torch.isfinite(grad_norm)

    nested_log_probs = unpack_packed_thd_to_nested(out["log_probs"], packed)
    assert nested_log_probs.offsets().numel() == len(lengths) + 1
    assert [int(x) for x in nested_log_probs.offsets().diff().cpu()] == lengths

    if rank == 0:
        print(
            "NON_SKIP_GLM5_THD_CP_SMOKE_PASSED "
            f"world_size={world} lengths={lengths} "
            f"loss={float(out['loss'].detach().item()):.6e} "
            f"grad_norm={float(grad_norm.detach().item()):.6e}"
        )


def test_glm5_tiny_model_cp2_matches_hf_reference_logits(tmp_path):
    import torch
    import torch.distributed as dist
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import load_hf_weights
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import ParallelState

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = Glm5Config(**_tiny_hf_parity_config_kwargs())

    torch.manual_seed(20260611)
    hf_ref = DeepseekV3ForCausalLM(_to_hf_deepseek_v3_config(cfg)).to(
        device=device, dtype=torch.bfloat16
    )
    hf_ref.eval()
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)
    native = _make_glm5_model(cfg, ps=ps).to(device=device, dtype=torch.bfloat16)
    native.eval()
    rank_tmp_path = tmp_path / f"rank{rank}"
    save_safetensors(
        _hf_state_dict_for_glm5_loader(hf_ref, native, cfg), str(rank_tmp_path)
    )
    load_hf_weights(native, str(rank_tmp_path), cfg, ps)

    batch, seq = 1, _fused_dsa_seq_len(world)
    torch.manual_seed(311)
    full_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    local_ids = zigzag_slice_for_cp(full_ids, rank, world, seq_dim=1).contiguous()

    hf_layer_outputs = []
    native_layer_outputs = []
    hooks = []
    for layer in hf_ref.model.layers:
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, outputs: hf_layer_outputs.append(
                    _capture_tensor_output(outputs)
                )
            )
        )
    for layer in native.layers:
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, outputs: native_layer_outputs.append(
                    _capture_tensor_output(outputs)
                )
            )
        )

    with torch.no_grad():
        if rank == 0:
            print(
                "glm5_hf_native_parity reference=transformers.DeepseekV3ForCausalLM "
                "mode=dense_mla_reference_with_full_topk_dsa"
            )
        hf_logits = hf_ref(full_ids).logits
        native_logits = native(input_ids=local_ids)["logits"]

    for hook in hooks:
        hook.remove()

    assert len(hf_layer_outputs) == cfg.num_hidden_layers
    assert len(native_layer_outputs) == cfg.num_hidden_layers
    for layer_idx, (actual, full_expected) in enumerate(
        zip(native_layer_outputs, hf_layer_outputs, strict=True)
    ):
        expected = zigzag_slice_for_cp(full_expected, rank, world, seq_dim=1)
        expected = expected.transpose(0, 1).contiguous()
        max_abs, max_rel = _distributed_diff_stats(actual, expected)
        if rank == 0:
            print(
                f"glm5_hf_native_parity layer={layer_idx} "
                f"max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e}"
            )
        torch.testing.assert_close(actual.float(), expected.float(), atol=1.5e-1, rtol=1.5e-1)

    expected = zigzag_slice_for_cp(hf_logits, rank, world, seq_dim=1).contiguous()
    max_abs, max_rel = _distributed_diff_stats(native_logits, expected)
    if rank == 0:
        print(
            "glm5_hf_native_parity logits " f"max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e}"
        )
    torch.testing.assert_close(native_logits.float(), expected.float(), atol=1.5e-1, rtol=1.5e-1)
