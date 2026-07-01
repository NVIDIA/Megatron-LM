# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _init_dist_or_skip():
    import os

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Kimi K2 CP smoke.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so CP ranks are available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() < 2:
        pytest.skip("Kimi K2 CP smoke requires at least 2 ranks.")
    return torch.device("cuda", local_rank)


def _tiny_config():
    from megatron.lite.model.kimi_k2.config import KimiK2Config

    return KimiK2Config(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        intermediate_size=96,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )


def _tiny_hf_parity_config():
    from megatron.lite.model.kimi_k2.config import KimiK2Config

    return KimiK2Config(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        intermediate_size=96,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )


def _train_cfg(cp: int):
    return SimpleNamespace(
        tp=1,
        ep=1,
        etp=1,
        pp=1,
        cp=cp,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules={},
        deterministic=False,
    )


def _to_hf_deepseek_v3_config(cfg):
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

    rope_scaling = dict(cfg.rope_scaling)
    rope_scaling.setdefault("rope_type", rope_scaling.get("type", "yarn"))
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
        tie_word_embeddings=cfg.tie_word_embeddings,
        rope_theta=cfg.rope_theta,
        rope_scaling=rope_scaling,
        rope_interleave=True,
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


def _hf_state_dict_for_kimi_loader(model):
    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    for name, gate_up in list(state.items()):
        if not name.endswith(".mlp.experts.gate_up_proj"):
            continue
        prefix = name.removesuffix(".gate_up_proj")
        down = state.get(f"{prefix}.down_proj")
        gate, up = gate_up.chunk(2, dim=1)
        for expert_idx in range(gate_up.size(0)):
            state[f"{prefix}.{expert_idx}.gate_proj.weight"] = gate[expert_idx].contiguous().clone()
            state[f"{prefix}.{expert_idx}.up_proj.weight"] = up[expert_idx].contiguous().clone()
            if down is not None:
                state[f"{prefix}.{expert_idx}.down_proj.weight"] = (
                    down[expert_idx].contiguous().clone()
                )
    return state


def test_kimi_k2_mla_cp2_matches_full_sequence_reference_forward_and_grad():
    import torch
    import torch.distributed as dist

    device = _init_dist_or_skip()
    from megatron.lite.primitive.modules.attention import MultiLatentAttention
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = _tiny_config()
    ps = init_parallel(ParallelConfig(tp=1, ep=1, etp=1, cp=world, pp=1))

    kwargs = dict(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        rope_scaling=cfg.rope_scaling,
        use_thd=False,
    )
    torch.manual_seed(20260531)
    cp_layer = MultiLatentAttention(ps=ps, **kwargs).to(device=device, dtype=torch.bfloat16)
    torch.manual_seed(20260531)
    ref_layer = MultiLatentAttention(ps=ParallelState(), **kwargs).to(
        device=device,
        dtype=torch.bfloat16,
    )

    seq, batch = 8 * world, 1
    torch.manual_seed(123)
    full_x = torch.randn(seq, batch, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    local_x = zigzag_slice_for_cp(full_x, rank, world, seq_dim=0).detach().requires_grad_(True)
    ref_x = full_x.detach().clone().requires_grad_(True)

    cp_out = cp_layer(local_x)
    ref_out = ref_layer(ref_x)
    expected = zigzag_slice_for_cp(ref_out, rank, world, seq_dim=0)
    torch.testing.assert_close(cp_out, expected, atol=5e-2, rtol=5e-2)

    cp_out.float().sum().backward()
    expected.float().sum().backward()
    expected_grad = zigzag_slice_for_cp(ref_x.grad, rank, world, seq_dim=0)
    assert local_x.grad is not None
    torch.testing.assert_close(local_x.grad, expected_grad, atol=1e-1, rtol=1e-1)


def test_kimi_k2_tiny_model_cp2_matches_full_sequence_reference_forward():
    import torch
    import torch.distributed as dist

    device = _init_dist_or_skip()
    from megatron.lite.model.kimi_k2.lite.model import KimiK2Model
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = _tiny_config()
    ps = init_parallel(ParallelConfig(tp=1, ep=1, etp=1, cp=world, pp=1))

    torch.manual_seed(777)
    cp_model = KimiK2Model(cfg, _train_cfg(world), ps, use_thd=False).to(
        device=device,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(777)
    ref_model = KimiK2Model(cfg, _train_cfg(1), ParallelState(), use_thd=False).to(
        device=device,
        dtype=torch.bfloat16,
    )
    cp_model.eval()
    ref_model.eval()

    batch, seq = 1, 8 * world
    torch.manual_seed(100)
    full_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    full_labels = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    input_ids = zigzag_slice_for_cp(full_ids, rank, world, seq_dim=1).contiguous()
    labels = zigzag_slice_for_cp(full_labels, rank, world, seq_dim=1).contiguous()

    with torch.no_grad():
        cp_out = cp_model(input_ids=input_ids, labels=labels)
        ref_out = ref_model(input_ids=full_ids, labels=full_labels)

    expected_hidden = zigzag_slice_for_cp(ref_out["hidden_states"], rank, world, seq_dim=0)
    expected_log_probs = zigzag_slice_for_cp(ref_out["log_probs"], rank, world, seq_dim=1)
    torch.testing.assert_close(cp_out["hidden_states"], expected_hidden, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(cp_out["log_probs"], expected_log_probs, atol=1e-1, rtol=1e-1)


def test_kimi_k2_tiny_model_cp2_matches_hf_reference_logits(tmp_path):
    import torch
    import torch.distributed as dist
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

    device = _init_dist_or_skip()
    from megatron.lite.model.kimi_k2.lite.checkpoint import load_hf_weights
    from megatron.lite.model.kimi_k2.lite.model import KimiK2Model
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = _tiny_hf_parity_config()

    torch.manual_seed(20260610)
    hf_ref = DeepseekV3ForCausalLM(_to_hf_deepseek_v3_config(cfg)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    hf_ref.eval()
    rank_tmp_path = tmp_path / f"rank{rank}"
    save_safetensors(
        _hf_state_dict_for_kimi_loader(hf_ref),
        str(rank_tmp_path),
    )

    ps = init_parallel(ParallelConfig(tp=1, ep=1, etp=1, cp=world, pp=1))
    native = KimiK2Model(cfg, _train_cfg(world), ps, use_thd=False).to(
        device=device,
        dtype=torch.bfloat16,
    )
    native.eval()
    load_hf_weights(native, str(rank_tmp_path), cfg, ps)

    batch, seq = 1, 4 * world
    torch.manual_seed(310)
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
                "kimi_k2_hf_native_parity reference=transformers.DeepseekV3ForCausalLM "
                "rope=DeepseekV3RotaryEmbedding+apply_rotary_pos_emb_interleave"
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
                f"kimi_k2_hf_native_parity layer={layer_idx} "
                f"max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e}"
            )
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            atol=1.5e-1,
            rtol=1.5e-1,
        )

    expected = zigzag_slice_for_cp(hf_logits, rank, world, seq_dim=1).contiguous()
    max_abs, max_rel = _distributed_diff_stats(native_logits, expected)
    if rank == 0:
        print(
            "kimi_k2_hf_native_parity logits "
            f"max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e}"
        )
    torch.testing.assert_close(
        native_logits.float(),
        expected.float(),
        atol=1.5e-1,
        rtol=1.5e-1,
    )


def test_kimi_k2_packed_thd_variable_sequence_cp2_smoke():
    import torch
    import torch.distributed as dist

    device = _init_dist_or_skip()
    from megatron.lite.model.kimi_k2.lite.model import KimiK2Model
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.primitive.parallel.thd import pack_nested_thd, unpack_packed_thd_to_nested
    from megatron.lite.runtime.contracts import ParallelConfig

    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = _tiny_config()
    ps = init_parallel(ParallelConfig(tp=1, ep=1, etp=1, cp=world, pp=1))

    torch.manual_seed(20260614)
    model = KimiK2Model(cfg, _train_cfg(world), ps, use_thd=True).to(
        device=device,
        dtype=torch.bfloat16,
    )
    model.train()

    lengths = [5, 7, 9]
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
    out["loss"].backward()
    nested_log_probs = unpack_packed_thd_to_nested(out["log_probs"], packed)
    assert nested_log_probs.offsets().numel() == len(lengths) + 1
    assert [int(x) for x in nested_log_probs.offsets().diff().cpu()] == lengths

    if rank == 0:
        print(
            "NON_SKIP_KIMI_K2_THD_CP_SMOKE_PASSED "
            f"world_size={world} lengths={lengths} "
            f"loss={float(out['loss'].detach().item()):.6e}"
        )


def test_kimi_k2_tiny_model_fsdp2_optimizer_step_smoke():
    import torch
    import torch.distributed as dist

    device = _init_dist_or_skip()
    from megatron.lite.model.kimi_k2.lite.protocol import ImplConfig, build_model
    from megatron.lite.primitive.optimizers.fsdp2 import FSDP2Optimizer, fsdp2_available
    from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig

    if not fsdp2_available():
        pytest.skip("Installed PyTorch does not expose FSDP2 fully_shard.")

    rank = dist.get_rank()
    world = dist.get_world_size()
    cfg = _tiny_config()
    impl_cfg = ImplConfig(
        parallel=ParallelConfig(tp=1, ep=1, etp=1, cp=1, pp=1),
        optimizer="fsdp2",
        optimizer_config=OptimizerConfig(
            optimizer="adam",
            lr=1.0e-4,
            weight_decay=0.0,
            clip_grad=1.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1.0e-8,
        ),
    )

    torch.manual_seed(20260612)
    bundle = build_model(cfg, impl_cfg=impl_cfg)
    assert bundle.optimizer is None
    assert bundle.extras["optimizer_backend"] == "fsdp2"
    post_model_load_hook = bundle.extras["post_model_load_hook"]
    assert post_model_load_hook is not None
    updates = post_model_load_hook()
    optimizer = updates["optimizer"]
    assert isinstance(optimizer, FSDP2Optimizer)
    assert optimizer.name == "fsdp2"

    model = bundle.chunks[0]
    model.train()
    optimizer.zero_grad()

    batch, seq = 1, 8
    torch.manual_seed(20260613 + rank)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    labels = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)

    out = model(input_ids=input_ids, labels=labels)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    success, grad_norm, num_zeros = optimizer.step()
    assert success
    assert num_zeros == 0
    assert torch.isfinite(torch.tensor(grad_norm))
    if rank == 0:
        print(
            "kimi_k2_fsdp2_smoke optimizer=fsdp2 "
            f"world_size={world} loss={float(loss.detach().item()):.6e} "
            f"grad_norm={grad_norm:.6e}"
        )
